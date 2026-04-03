"""Gemma4 MoE custom implementation for PrimeRL training.

Extends the dense Gemma4 model with:
- 128 sparse experts (top-8) running in parallel with a shared dense MLP
- Custom router: RMSNorm (no scale) + learnable scale + softmax + renormalize + per_expert_scale
- Router input is the pre-MLP residual (not the MLP output)
- Expert activation: gelu_pytorch_tanh (not silu)
- MoE weight conversion: HF fused gate_up_proj → PrimeRL w1/w2/w3
"""

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.gemma4.modeling_gemma4 import (
    Gemma4Attention,
    Gemma4DualRotaryEmbedding,
    Gemma4MLP,
    Gemma4RMSNormNoScale,
    Gemma4ScaledWordEmbedding,
    _compute_default_rope_parameters,
    _get_flash_attn_version,
)
from prime_rl.trainer.models.gemma4_moe.converting_gemma4_moe import (
    convert_hf_layer_to_prime,
    convert_hf_to_prime,
    convert_prime_layer_to_hf,
    convert_prime_to_hf,
)
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig

# ---------------------------------------------------------------------------
# Gemma4 MoE Router
# ---------------------------------------------------------------------------


class Gemma4Router(nn.Module):
    """Gemma4 router: RMSNorm(no_scale) → scale → linear → softmax → topk → renorm → per_expert_scale."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.scalar_root_size = self.hidden_size**-0.5

        self.norm = Gemma4RMSNormNoScale(self.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)
        router_probs = F.softmax(expert_scores.float(), dim=-1)

        top_k_weights, top_k_index = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        num_tokens_per_expert = torch.histc(
            top_k_index.reshape(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_k_weights.to(hidden_states.dtype), top_k_index, num_tokens_per_expert


# ---------------------------------------------------------------------------
# Gemma4 MoE Experts (w1/w2/w3 format with gelu_pytorch_tanh)
# ---------------------------------------------------------------------------


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    act_fn,
) -> torch.Tensor:
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()
    num_padding = x.shape[0] - sum(num_tokens_per_expert_list)
    x_splits = torch.split(x[: sum(num_tokens_per_expert_list)], num_tokens_per_expert_list, dim=0)
    out_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        h = act_fn(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        out_splits.append(h)
    out = torch.cat(out_splits, dim=0)
    if num_padding > 0:
        out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
    return out


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    act_fn,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    h = act_fn(torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets))
    h = h * torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)
    return out


class Gemma4Experts(nn.Module):
    """Expert weights in PrimeRL w1/w2/w3 format with configurable activation."""

    def __init__(self, config: Gemma4TextConfig, use_grouped_mm: bool = True):
        super().__init__()
        self.num_experts = config.num_experts
        dim = config.hidden_size
        hidden_dim = config.moe_intermediate_size
        self.w1 = nn.Parameter(torch.empty(self.num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, hidden_dim, dim))
        self.act_fn = ACT2FN[config.hidden_activation]
        self.use_grouped_mm = use_grouped_mm

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        if self.use_grouped_mm:
            return _run_experts_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert, self.act_fn)
        return _run_experts_for_loop(self.w1, self.w2, self.w3, x, num_tokens_per_expert, self.act_fn)


# ---------------------------------------------------------------------------
# MoE Decoder Layer
# ---------------------------------------------------------------------------


class Gemma4MoeDecoderLayer(GradientCheckpointingLayer):
    """Gemma4 decoder layer with shared MLP + sparse MoE in parallel."""

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4Attention(
            config, layer_idx, flash_attn_version=_get_flash_attn_version(config._attn_implementation)
        )
        self.mlp = Gemma4MLP(config)

        # Standard 4 layernorms
        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.pre_feedforward_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_feedforward_layernorm = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )

        # MoE components
        self.router = Gemma4Router(config)
        self.experts = Gemma4Experts(config, use_grouped_mm=getattr(config, "use_grouped_mm", True))

        # Extra norms for MoE parallel path
        self.post_feedforward_layernorm_1 = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )
        self.post_feedforward_layernorm_2 = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )
        self.pre_feedforward_layernorm_2 = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )

        # Per-layer scalar
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # FFN + MoE parallel block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # Shared MLP output normalized
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        # Sparse experts: router takes pre-MLP residual
        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        top_k_weights, top_k_index, num_tokens_per_expert = self.router(hidden_states_flat)

        # Reorder tokens by expert assignment
        selected_flat = top_k_index.reshape(-1)
        token_indices_sorted = torch.argsort(selected_flat, stable=True)
        scores_sorted = top_k_weights.view(-1)[token_indices_sorted]
        token_indices_sorted = token_indices_sorted // self.router.top_k

        dim = hidden_states_flat.shape[-1]
        routed_indices = token_indices_sorted.reshape(-1, 1).expand(-1, dim)

        # Pre-norm for expert input
        expert_input = self.pre_feedforward_layernorm_2(hidden_states_flat)
        routed_input = torch.gather(expert_input, dim=0, index=routed_indices)
        routed_input = (routed_input.float() * scores_sorted.reshape(-1, 1)).to(routed_input.dtype)

        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # Scatter back
        hidden_states_2 = torch.zeros_like(hidden_states_flat)
        hidden_states_2.scatter_add_(dim=0, index=routed_indices, src=routed_output)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        # Combine shared MLP + sparse experts
        hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


# ---------------------------------------------------------------------------
# PreTrained base
# ---------------------------------------------------------------------------


@auto_docstring
class Gemma4MoePreTrainedModel(PreTrainedModelPrimeRL):
    config_class = Gemma4TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma4MoeDecoderLayer"]
    _supports_flash_attn = True

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("experts.gate_up_proj" in k or "experts.0.gate_proj" in k for k in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("experts.w1" in k for k in state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_prime_to_hf(state_dict)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_hf_to_prime(state_dict)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_prime_layer_to_hf(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_hf_layer_to_prime(state_dict, layer_idx)
        return state_dict


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@auto_docstring
class Gemma4MoeModel(Gemma4MoePreTrainedModel):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma4ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma4MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.rotary_emb = Gemma4DualRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config._attn_implementation in ("flash_attention_2", "flash_attention_3", "fa4"):
            flat_position_ids = position_ids.view(-1)
            seqlens = torch.cat(
                [
                    flat_position_ids[0:1],
                    flat_position_ids[:-1][(flat_position_ids == 0)[1:]] + 1,
                    flat_position_ids[-1:] + 1,
                ]
            )
            max_seqlen = seqlens.max().item()
            cu_seqlens = seqlens.cumsum(dim=0, dtype=torch.int32)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        else:
            max_seqlen = None
            cu_seqlens = None

        hidden_states = inputs_embeds

        unique_layer_types = set(self.config.layer_types)
        position_embeddings = {}
        for layer_type in unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[layer_idx]
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


# ---------------------------------------------------------------------------
# Causal LM
# ---------------------------------------------------------------------------


@auto_docstring
class Gemma4MoeForCausalLM(Gemma4MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Gemma4TextConfig):
        super().__init__(config)
        self.model = Gemma4MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_logit_softcapping = config.final_logit_softcapping
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        self.model.embed_tokens.embed_scale.fill_(self.config.hidden_size**0.5)
        rotary_emb = self.model.rotary_emb
        for layer_type in rotary_emb.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            rope_type = rope_params["rope_type"]
            if rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            else:
                rope_init_fn = _compute_default_rope_parameters

            kwargs = {"device": getattr(rotary_emb, f"{layer_type}_inv_freq").device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"

            inv_freq, attn_scaling = rope_init_fn(self.config, **kwargs)
            getattr(rotary_emb, f"{layer_type}_inv_freq").copy_(inv_freq)
            rotary_emb.attention_scaling[layer_type] = attn_scaling
