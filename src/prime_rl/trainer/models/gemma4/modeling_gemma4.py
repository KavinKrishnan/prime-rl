"""Gemma4 custom implementation for PrimeRL training.

Supports both dense (31B) and MoE (26B-A4B) variants, in text-only and VLM modes:
- Hybrid sliding window + global attention (5:1 pattern)
- Dual RoPE: theta=10K for sliding, theta=1M + partial_rotary_factor=0.25 for global
- K=V sharing on global attention layers (no v_proj)
- QKV norms (q/k with scale, v without scale)
- Attention scaling = 1.0 (QK norms handle magnitude)
- Logit softcapping (tanh at 30.0)
- Per-layer learnable scalar
- Scaled embeddings (× sqrt(hidden_size))

VLM mode is auto-detected from the config (presence of vision_config). In VLM mode,
the HF vision tower is used as-is and only the language model uses our custom impl.
"""

from typing import Optional, Union

import torch
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput

# flash-attention-2
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None  # type: ignore

# flash-attention-3
try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
except ImportError:
    flash_attn_3_varlen_func = None  # type: ignore

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
except ImportError:
    flash_attn_4_varlen_func = None  # type: ignore


# ---------------------------------------------------------------------------
# Gemma4 RMSNorm — torch-native to match vLLM's kernel numerics.
# The shared RMSNorm uses quack kernels on Hopper+ which have slightly
# different bf16 accumulation, causing mismatch KL with vLLM inference.
# ---------------------------------------------------------------------------


class Gemma4RMSNorm(nn.Module):
    """RMSNorm with learnable scale, torch-native (no quack)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states.to(input_dtype)).to(input_dtype)


class Gemma4RMSNormNoScale(nn.Module):
    """RMSNorm without a learnable scale parameter, for value normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Scaled word embedding
# ---------------------------------------------------------------------------


class Gemma4ScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


# ---------------------------------------------------------------------------
# Dual RoPE (per layer type)
# ---------------------------------------------------------------------------


def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None, head_dim_key=None):
    config.standardize_rope_params()
    rope_params = config.rope_parameters[layer_type] if layer_type else config.rope_parameters
    base = rope_params["rope_theta"]
    partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, head_dim_key, None) if head_dim_key else None
    if head_dim is None:
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


class Gemma4DualRotaryEmbedding(nn.Module):
    """Stores separate inv_freq buffers per layer type (sliding vs full)."""

    def __init__(self, config: Gemma4TextConfig, device=None):
        super().__init__()
        self.config = config
        self.layer_types = set(config.layer_types)
        self.attention_scaling = {}

        for layer_type in self.layer_types:
            rope_params = config.rope_parameters[layer_type]
            rope_type = rope_params["rope_type"]

            if rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            else:
                rope_init_fn = _compute_default_rope_parameters

            kwargs = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"

            inv_freq, attn_scaling = rope_init_fn(config, **kwargs)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.attention_scaling[layer_type] = attn_scaling

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attn_scaling = self.attention_scaling[layer_type]

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attn_scaling
            sin = emb.sin() * attn_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# RoPE application
# ---------------------------------------------------------------------------


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_single(t, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    t_rot, t_pass = t[..., :rotary_dim], t[..., rotary_dim:]
    t_embed = (t_rot * cos) + (_rotate_half(t_rot) * sin)
    return torch.cat([t_embed, t_pass], dim=-1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

_FLASH_ATTN_FUNCS = {
    2: flash_attn_varlen_func,
    3: flash_attn_3_varlen_func,
    4: flash_attn_4_varlen_func,
}


class Gemma4Attention(nn.Module):
    """Gemma4 attention with hybrid sliding/global, K=V sharing, QKV norms."""

    def __init__(self, config: Gemma4TextConfig, layer_idx: int, flash_attn_version: int = 2):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # Global attention uses larger head_dim
        self.head_dim = config.global_head_dim if (not self.is_sliding and config.global_head_dim) else config.head_dim
        self.num_attention_heads = config.num_attention_heads

        # K=V sharing: only on global layers when enabled
        self.use_kv_sharing = config.attention_k_eq_v and not self.is_sliding
        num_kv_heads = (
            config.num_global_key_value_heads
            if self.use_kv_sharing and config.num_global_key_value_heads
            else config.num_key_value_heads
        )
        self.num_key_value_heads = num_kv_heads
        self.num_key_value_groups = config.num_attention_heads // num_kv_heads

        # Projections
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = (
            nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=config.attention_bias)
            if not self.use_kv_sharing
            else None
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        # QKV norms
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)

        # Flash attention
        self._flash_attn_version = flash_attn_version
        self._flash_attn_func = _FLASH_ATTN_FUNCS[flash_attn_version]
        if self._flash_attn_version == 4:
            self._flash_attn_call = torch._dynamo.disable(self._flash_attn_func)
        else:
            self._flash_attn_call = self._flash_attn_func

    def _compute_flash_attention(self, q, k, v, cu_seqlens, max_seqlen):
        args = [q, k, v, cu_seqlens, cu_seqlens]
        if self._flash_attn_version != 4:
            args.extend([max_seqlen, max_seqlen])
        kwargs = {"causal": True, "softmax_scale": 1.0}
        if self.sliding_window is not None:
            kwargs["window_size"] = (self.sliding_window - 1, 0)
        out = self._flash_attn_call(*args, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _compute_sdpa_attention(self, q, k, v, cu_seqlens):
        """SDPA fallback for head_dim > 256 (global attention layers).

        Handles packed sequences by building a block-diagonal causal mask from cu_seqlens.
        """
        # q/k/v: [total_tokens, heads, dim] -> [1, heads, total_tokens, dim]
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        # GQA: repeat k/v heads to match q heads
        if k.shape[1] != q.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Build block-diagonal causal mask for packed sequences
        total_len = q.shape[2]
        if cu_seqlens is not None and len(cu_seqlens) > 2:
            # Multiple packed sequences — need block-diagonal mask
            mask = torch.full((total_len, total_len), float("-inf"), device=q.device, dtype=q.dtype)
            for i in range(len(cu_seqlens) - 1):
                start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                seq_len = end - start
                causal = torch.tril(torch.zeros(seq_len, seq_len, device=q.device, dtype=q.dtype))
                causal.masked_fill_(
                    torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1),
                    float("-inf"),
                )
                mask[start:end, start:end] = causal
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)

        # [1, heads, total_tokens, dim] -> [total_tokens, heads, dim]
        return out.transpose(1, 2).squeeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        # Q projection + norm + RoPE
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = _apply_rotary_pos_emb_single(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # K projection + norm + RoPE
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        # V: either from v_proj or reuse k_proj output (K=V sharing)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = _apply_rotary_pos_emb_single(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        # Flash attention expects [total_tokens, heads, dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # FlashAttention only supports head_dim <= 256; fall back to SDPA for global layers (head_dim=512)
        if self.head_dim > 256:
            attn_output = self._compute_sdpa_attention(query_states[0], key_states[0], value_states[0], cu_seqlens)
        else:
            attn_output = self._compute_flash_attention(
                query_states[0], key_states[0], value_states[0], cu_seqlens, max_seqlen
            )
        attn_output = attn_output.contiguous().view(1, attn_output.shape[0], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


def _get_flash_attn_version(attn_impl: str) -> int:
    mapping = {
        "flash_attention_2": 2,
        "flash_attention_3": 3,
        "fa4": 4,
    }
    return mapping.get(attn_impl, 2)


class Gemma4DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4Attention(
            config, layer_idx, flash_attn_version=_get_flash_attn_version(config._attn_implementation)
        )
        self.mlp = Gemma4MLP(config)

        # 4 layernorms
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        # FFN block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


# ---------------------------------------------------------------------------
# VLM helpers
# ---------------------------------------------------------------------------


def _has_vlm_keys(state_dict: dict[str, Tensor]) -> bool:
    return any(k.startswith("model.language_model.") for k in state_dict)


def _remap_lm_keys(state_dict: dict[str, Tensor], to_flat: bool = True) -> None:
    """Remap language model keys between VLM and flat format for weight conversion.

    to_flat=True:  model.language_model.* -> model.*
    to_flat=False: model.*               -> model.language_model.*

    Vision keys (model.vision_tower.*, model.embed_vision.*) are never touched.
    """
    VISION_PREFIXES = ("model.vision_tower.", "model.embed_vision.")
    src = "model.language_model." if to_flat else "model."
    dst = "model." if to_flat else "model.language_model."
    for k in [
        k for k in list(state_dict.keys()) if k.startswith(src) and not any(k.startswith(p) for p in VISION_PREFIXES)
    ]:
        state_dict[dst + k[len(src) :]] = state_dict.pop(k)


# ---------------------------------------------------------------------------
# PreTrained base
# ---------------------------------------------------------------------------


@auto_docstring
class Gemma4PreTrainedModel(PreTrainedModelPrimeRL):
    config_class = Gemma4TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma4DecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True

    @classmethod
    def _has_moe_keys(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("experts." in k for k in state_dict)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        if not cls._has_moe_keys(state_dict):
            return True  # Dense: HF format = PrimeRL format
        return any("experts.gate_up_proj" in k or "experts.0.gate_proj" in k for k in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        if not cls._has_moe_keys(state_dict):
            return True  # Dense: HF format = PrimeRL format
        return any("experts.w1" in k for k in state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        from prime_rl.trainer.models.gemma4_moe.converting_gemma4_moe import convert_prime_to_hf

        vlm = _has_vlm_keys(state_dict)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=True)
        convert_prime_to_hf(state_dict)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=False)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        from prime_rl.trainer.models.gemma4_moe.converting_gemma4_moe import convert_hf_to_prime

        vlm = _has_vlm_keys(state_dict)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=True)
        convert_hf_to_prime(state_dict)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=False)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        from prime_rl.trainer.models.gemma4_moe.converting_gemma4_moe import convert_prime_layer_to_hf

        vlm = _has_vlm_keys(state_dict)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=True)
        convert_prime_layer_to_hf(state_dict, layer_idx)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=False)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        from prime_rl.trainer.models.gemma4_moe.converting_gemma4_moe import convert_hf_layer_to_prime

        vlm = _has_vlm_keys(state_dict)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=True)
        convert_hf_layer_to_prime(state_dict, layer_idx)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=False)
        return state_dict


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@auto_docstring
class Gemma4Model(Gemma4PreTrainedModel):
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
        if config.enable_moe_block:
            from prime_rl.trainer.models.gemma4_moe.modeling_gemma4_moe import Gemma4MoeDecoderLayer

            layer_cls = Gemma4MoeDecoderLayer
        else:
            layer_cls = Gemma4DecoderLayer
        self.layers = nn.ModuleList([layer_cls(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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

        # Compute RoPE embeddings per layer type
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
# VLM composite model
# ---------------------------------------------------------------------------


class Gemma4VLMModel(nn.Module):
    """Composite VLM body: HF vision tower + custom PrimeRL text model."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel

        self.vision_tower = Gemma4VisionModel._from_config(config.vision_config)
        self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
        self.language_model = Gemma4Model(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.vision_tower.dtype)
            vision_output = self.vision_tower(pixel_values, return_dict=True)
            image_features = self.embed_vision(vision_output.last_hidden_state)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask = input_ids == self.config.image_token_id
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )


# ---------------------------------------------------------------------------
# Causal LM (unified text-only + VLM)
# ---------------------------------------------------------------------------


@auto_docstring
class Gemma4ForCausalLM(Gemma4PreTrainedModel, GenerationMixin):
    """Unified Gemma4 model for both text-only and VLM configs.

    When config has a vision_config, creates a composite model with HF's frozen
    vision tower + custom text model. Otherwise creates a text-only model.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._is_vlm = hasattr(config, "vision_config")

        if self._is_vlm:
            self.model = Gemma4VLMModel(config)
            text_config = config.text_config
            self._tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
        else:
            self.model = Gemma4Model(config)
            text_config = config

        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.final_logit_softcapping = text_config.final_logit_softcapping
        self.post_init()

    def get_input_embeddings(self):
        if self._is_vlm:
            return self.model.get_input_embeddings()
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        if self._is_vlm:
            self.model.set_input_embeddings(value)
        else:
            self.model.embed_tokens = value

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
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        if self._is_vlm:
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
            )
        else:
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

    def _get_text_config(self):
        return self.config.text_config if self._is_vlm else self.config

    def init_buffers_post_meta(self):
        text_config = self._get_text_config()

        # Reinitialize embed_scale (non-persistent buffer)
        if self._is_vlm:
            embed_tokens = self.model.language_model.embed_tokens
            rotary_emb = self.model.language_model.rotary_emb
        else:
            embed_tokens = self.model.embed_tokens
            rotary_emb = self.model.rotary_emb

        embed_tokens.embed_scale.fill_(text_config.hidden_size**0.5)

        # Initialize dual RoPE inv_freq buffers
        for layer_type in rotary_emb.layer_types:
            rope_params = text_config.rope_parameters[layer_type]
            rope_type = rope_params["rope_type"]
            if rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            else:
                rope_init_fn = _compute_default_rope_parameters

            kwargs = {"device": getattr(rotary_emb, f"{layer_type}_inv_freq").device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"

            inv_freq, attn_scaling = rope_init_fn(text_config, **kwargs)
            getattr(rotary_emb, f"{layer_type}_inv_freq").copy_(inv_freq)
            rotary_emb.attention_scaling[layer_type] = attn_scaling
