from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ring_flash_attn import update_ring_flash_attn_params
from torch import Tensor, nn
from transformers.masking_utils import create_causal_mask

from prime_rl.configs.trainer import MTPConfig
from prime_rl.utils.cp import _get_cu_seqlens_for_cp, shard_for_cp


@dataclass
class MTPLayerBatch:
    input_ids: Tensor
    position_ids: Tensor
    labels: Tensor
    loss_mask: Tensor


@dataclass
class MTPTrainingBatch:
    steps: list[MTPLayerBatch]
    cp_group: dist.ProcessGroup | None = None
    cp_cu_seqlens: Tensor | None = None
    attention_cu_seqlens: Tensor | None = None
    attention_max_seqlen: int | None = None


@dataclass
class MTPRuntime:
    backbone: nn.Module
    embed_tokens: nn.Module
    rotary_emb: nn.Module | None
    layers: list[nn.Module]
    run_layer: Callable[
        [nn.Module, "MTPRuntime", Tensor, Tensor, Tensor, dist.ProcessGroup | None, Tensor | None], Tensor
    ]
    container: nn.Module | None = None
    attention_cu_seqlens: Tensor | None = None
    attention_max_seqlen: int | None = None


def _get_packed_lengths(position_ids: Tensor) -> list[list[int]]:
    lengths_by_batch: list[list[int]] = []
    for row in position_ids.tolist():
        boundaries = [0, *[idx for idx in range(1, len(row)) if row[idx] == 0], len(row)]
        lengths: list[int] = []
        for idx, start in enumerate(boundaries[:-1]):
            end = boundaries[idx + 1]
            lengths.append(end - start)
        lengths_by_batch.append(lengths)
    return lengths_by_batch


def _packed_shift_left(tensor: Tensor, lengths_by_batch: list[list[int]], pad_value: int | float | bool) -> Tensor:
    shifted = torch.full_like(tensor, pad_value)
    for batch_idx, lengths in enumerate(lengths_by_batch):
        start = 0
        for length in lengths:
            end = start + length
            if length > 1:
                shifted[batch_idx, start : end - 1] = tensor[batch_idx, start + 1 : end]
            start = end
    return shifted


def _get_attention_metadata(
    position_ids: Tensor, lengths_by_batch: list[list[int]]
) -> tuple[Tensor | None, int | None]:
    max_seqlen = max((length for lengths in lengths_by_batch for length in lengths), default=0)
    if max_seqlen == 0:
        return None, None
    return _get_cu_seqlens_for_cp(position_ids), max_seqlen


def prepare_mtp_training_batch(
    input_ids: Tensor,
    position_ids: Tensor,
    labels: Tensor,
    loss_mask: Tensor,
    num_layers: int,
) -> MTPTrainingBatch:
    lengths_by_batch = _get_packed_lengths(position_ids)
    return _prepare_mtp_training_batch(input_ids, position_ids, labels, loss_mask, num_layers, lengths_by_batch)


def prepare_mtp_training_batch_from_current_tokens(
    input_ids: Tensor,
    position_ids: Tensor,
    loss_mask: Tensor,
    num_layers: int,
) -> MTPTrainingBatch:
    lengths_by_batch = _get_packed_lengths(position_ids)
    labels = _packed_shift_left(input_ids, lengths_by_batch, pad_value=0)
    aligned_loss_mask = _packed_shift_left(loss_mask.bool(), lengths_by_batch, pad_value=False)
    return _prepare_mtp_training_batch(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
        loss_mask=aligned_loss_mask,
        num_layers=num_layers,
        lengths_by_batch=lengths_by_batch,
    )


def _prepare_mtp_training_batch(
    input_ids: Tensor,
    position_ids: Tensor,
    labels: Tensor,
    loss_mask: Tensor,
    num_layers: int,
    lengths_by_batch: list[list[int]],
) -> MTPTrainingBatch:
    attention_cu_seqlens, attention_max_seqlen = _get_attention_metadata(position_ids, lengths_by_batch)
    current_input_ids = input_ids
    current_position_ids = position_ids
    current_labels = labels
    current_loss_mask = loss_mask.bool()

    steps: list[MTPLayerBatch] = []

    for _ in range(num_layers):
        current_input_ids = _packed_shift_left(current_input_ids, lengths_by_batch, pad_value=0)
        current_position_ids = _packed_shift_left(current_position_ids, lengths_by_batch, pad_value=0)
        current_labels = _packed_shift_left(current_labels, lengths_by_batch, pad_value=0)
        shifted_loss_mask = _packed_shift_left(current_loss_mask, lengths_by_batch, pad_value=False)
        current_loss_mask = current_loss_mask & shifted_loss_mask

        steps.append(
            MTPLayerBatch(
                input_ids=current_input_ids,
                position_ids=current_position_ids,
                labels=current_labels,
                loss_mask=current_loss_mask,
            )
        )

    return MTPTrainingBatch(
        steps=steps,
        attention_cu_seqlens=attention_cu_seqlens,
        attention_max_seqlen=attention_max_seqlen,
    )


def shard_mtp_training_batch_for_cp(
    batch: MTPTrainingBatch,
    original_position_ids: Tensor,
    cp_rank: int,
    cp_world_size: int,
    cp_group: dist.ProcessGroup,
) -> MTPTrainingBatch:
    return MTPTrainingBatch(
        steps=[
            MTPLayerBatch(
                input_ids=shard_for_cp(step.input_ids, cp_rank=cp_rank, cp_world_size=cp_world_size),
                position_ids=shard_for_cp(step.position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size),
                labels=shard_for_cp(step.labels, cp_rank=cp_rank, cp_world_size=cp_world_size),
                loss_mask=shard_for_cp(step.loss_mask, cp_rank=cp_rank, cp_world_size=cp_world_size),
            )
            for step in batch.steps
        ],
        cp_group=cp_group,
        cp_cu_seqlens=_get_cu_seqlens_for_cp(original_position_ids),
        attention_cu_seqlens=batch.attention_cu_seqlens,
        attention_max_seqlen=batch.attention_max_seqlen,
    )


def _looks_like_mimo_layer(layer: nn.Module) -> bool:
    required = (
        "token_layernorm",
        "hidden_layernorm",
        "input_proj",
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "mlp",
        "final_layernorm",
    )
    return all(hasattr(layer, name) for name in required)


def _looks_like_qwen_next_container(container: nn.Module) -> bool:
    required = ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "fc", "norm", "layers")
    return all(hasattr(container, name) for name in required)


def _looks_like_nemotron_container(container: nn.Module) -> bool:
    return getattr(container, "prime_mtp_kind", None) == "nemotron_h" and hasattr(container, "layers")


def _looks_like_megatron_mtp_layer(layer: nn.Module) -> bool:
    required = ("enorm", "hnorm", "eh_proj", "transformer_layer", "final_layernorm")
    return all(hasattr(layer, name) for name in required)


def _resolve_mtp_runtime(model: nn.Module, num_layers: int) -> MTPRuntime:
    backbone = getattr(model, "model", None)
    if not isinstance(backbone, nn.Module):
        raise ValueError("MTP training requires a model backbone at `model.model`.")

    embed_tokens = getattr(backbone, "embed_tokens", None)
    if not isinstance(embed_tokens, nn.Module):
        raise ValueError("MTP training requires a token embedding module at `model.model.embed_tokens`.")

    rotary_emb = getattr(backbone, "rotary_emb", None)

    mtp_layers = getattr(backbone, "mtp_layers", None)
    if isinstance(mtp_layers, nn.ModuleList) and len(mtp_layers) >= num_layers:
        if not _looks_like_mimo_layer(mtp_layers[0]) and not _looks_like_megatron_mtp_layer(mtp_layers[0]):
            raise ValueError("Found `model.model.mtp_layers`, but the layer structure is not supported yet.")
        run_layer = _run_mimo_layer if _looks_like_mimo_layer(mtp_layers[0]) else _run_megatron_style_layer
        return MTPRuntime(
            backbone=backbone,
            embed_tokens=embed_tokens,
            rotary_emb=rotary_emb,
            layers=list(mtp_layers[:num_layers]),
            run_layer=run_layer,
        )

    mtp_container = getattr(model, "mtp", None)
    if mtp_container is None:
        mtp_container = getattr(backbone, "mtp", None)
    if isinstance(mtp_container, nn.Module) and _looks_like_nemotron_container(mtp_container):
        return MTPRuntime(
            backbone=backbone,
            embed_tokens=embed_tokens,
            rotary_emb=rotary_emb,
            layers=[mtp_container for _ in range(num_layers)],
            run_layer=_run_nemotron_container,
            container=mtp_container,
        )
    if isinstance(mtp_container, nn.Module) and _looks_like_qwen_next_container(mtp_container):
        layers = getattr(mtp_container, "layers")
        if isinstance(layers, nn.ModuleList) and len(layers) >= num_layers:
            return MTPRuntime(
                backbone=backbone,
                embed_tokens=embed_tokens,
                rotary_emb=rotary_emb,
                layers=list(layers[:num_layers]),
                run_layer=_run_qwen_next_container_layer,
                container=mtp_container,
            )

    raise ValueError(
        "MTP training is enabled, but the loaded model does not expose a supported MTP module layout. "
        "Currently supported layouts are MiMo-style `model.model.mtp_layers`, Qwen-style `mtp.layers`, "
        "and Nemotron-style shared `mtp` containers."
    )


def configure_mtp_training(model: nn.Module, mtp_config: MTPConfig | None) -> None:
    setattr(model, "_prime_mtp_config", mtp_config)
    if mtp_config is None:
        setattr(model, "_prime_mtp_runtime", None)
        return
    setattr(model, "_prime_mtp_runtime", _resolve_mtp_runtime(model, mtp_config.num_layers))


def _get_configured_mtp_runtime(model: nn.Module) -> tuple[MTPConfig | None, MTPRuntime | None]:
    return getattr(model, "_prime_mtp_config", None), getattr(model, "_prime_mtp_runtime", None)


def _update_mtp_cp_state(cp_group: dist.ProcessGroup | None, cp_cu_seqlens: Tensor | None) -> None:
    if cp_group is None or cp_cu_seqlens is None:
        return
    update_ring_flash_attn_params(cp_cu_seqlens, cp_group)


def _apply_output_head(hidden_states: Tensor, model: nn.Module) -> Tensor:
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, nn.Module) or not hasattr(lm_head, "weight"):
        raise ValueError("MTP training requires an lm_head with a shared `weight` parameter.")

    logits = F.linear(hidden_states, lm_head.weight.detach())
    softcap = getattr(getattr(model, "config", None), "final_logit_softcapping", None)
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return logits


def _build_causal_mask(runtime: MTPRuntime, inputs_embeds: Tensor, position_ids: Tensor) -> Tensor:
    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    return create_causal_mask(
        config=runtime.backbone.config,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )


def _build_linear_attn_mask(runtime: MTPRuntime) -> Tensor | None:
    update_mask = getattr(runtime.backbone, "_update_linear_attn_mask", None)
    if update_mask is None:
        return None
    return update_mask(attention_mask=None, past_key_values=None)


def _get_position_embeddings(
    runtime: MTPRuntime, hidden_states: Tensor, position_ids: Tensor
) -> Tensor | tuple[Tensor, Tensor] | None:
    if runtime.rotary_emb is None:
        return None
    return runtime.rotary_emb(hidden_states, position_ids)


def _run_mimo_layer(
    layer: nn.Module,
    runtime: MTPRuntime,
    input_embeds: Tensor,
    hidden_states: Tensor,
    position_ids: Tensor,
    cp_group: dist.ProcessGroup | None,
    cp_cu_seqlens: Tensor | None,
) -> Tensor:
    _update_mtp_cp_state(cp_group, cp_cu_seqlens)
    attention_mask = _build_causal_mask(runtime, input_embeds, position_ids)
    position_embeddings = _get_position_embeddings(runtime, hidden_states, position_ids)

    input_embeds = layer.token_layernorm(input_embeds)
    hidden_states = layer.hidden_layernorm(hidden_states)
    hidden_states = layer.input_proj(torch.cat([hidden_states, input_embeds], dim=-1))

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states, _ = layer.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
    hidden_states = residual + hidden_states
    hidden_states = layer.final_layernorm(hidden_states)
    return hidden_states


def _run_qwen_next_container_layer(
    layer: nn.Module,
    runtime: MTPRuntime,
    input_embeds: Tensor,
    hidden_states: Tensor,
    position_ids: Tensor,
    cp_group: dist.ProcessGroup | None,
    cp_cu_seqlens: Tensor | None,
) -> Tensor:
    assert runtime.container is not None

    _update_mtp_cp_state(cp_group, cp_cu_seqlens)
    causal_mask = _build_causal_mask(runtime, input_embeds, position_ids)
    linear_attn_mask = _build_linear_attn_mask(runtime)
    position_embeddings = _get_position_embeddings(runtime, hidden_states, position_ids)
    cache_position = torch.arange(input_embeds.shape[1], device=input_embeds.device)

    input_embeds = runtime.container.pre_fc_norm_embedding(input_embeds)
    hidden_states = runtime.container.pre_fc_norm_hidden(hidden_states)
    hidden_states = runtime.container.fc(torch.cat([input_embeds, hidden_states], dim=-1))
    hidden_states = layer(
        hidden_states=hidden_states,
        attention_mask=linear_attn_mask if getattr(layer, "layer_type", None) == "linear_attention" else causal_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
    return runtime.container.norm(hidden_states)


def _run_nemotron_container(
    layer: nn.Module,
    runtime: MTPRuntime,
    input_embeds: Tensor,
    hidden_states: Tensor,
    position_ids: Tensor,
    cp_group: dist.ProcessGroup | None,
    cp_cu_seqlens: Tensor | None,
) -> Tensor:
    _update_mtp_cp_state(cp_group, cp_cu_seqlens)
    output = layer(
        input_embeds=input_embeds,
        hidden_states=hidden_states,
        position_ids=position_ids,
        cu_seqlens=runtime.attention_cu_seqlens,
        max_seqlen=runtime.attention_max_seqlen,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output


def _run_megatron_style_layer(
    layer: nn.Module,
    runtime: MTPRuntime,
    input_embeds: Tensor,
    hidden_states: Tensor,
    position_ids: Tensor,
    cp_group: dist.ProcessGroup | None,
    cp_cu_seqlens: Tensor | None,
) -> Tensor:
    _update_mtp_cp_state(cp_group, cp_cu_seqlens)
    attention_mask = _build_causal_mask(runtime, input_embeds, position_ids)
    position_embeddings = _get_position_embeddings(runtime, hidden_states, position_ids)

    input_embeds = layer.enorm(input_embeds)
    hidden_states = layer.hnorm(hidden_states)
    hidden_states = layer.eh_proj(torch.cat([input_embeds, hidden_states], dim=-1))
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
    hidden_states = layer.transformer_layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
    )
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
    return layer.final_layernorm(hidden_states)


def compute_mtp_loss(
    model: nn.Module,
    hidden_states: Tensor,
    mtp_batch: MTPTrainingBatch | None,
) -> tuple[Tensor | None, Tensor | None]:
    mtp_config, runtime = _get_configured_mtp_runtime(model)
    if mtp_config is None or runtime is None or mtp_batch is None:
        return None, None

    mtp_hidden_states = hidden_states.detach()
    runtime.attention_cu_seqlens = mtp_batch.attention_cu_seqlens
    runtime.attention_max_seqlen = mtp_batch.attention_max_seqlen
    raw_loss_sum = hidden_states.new_zeros(())
    raw_token_count = hidden_states.new_zeros((), dtype=torch.long)
    scaled_loss_sum = hidden_states.new_zeros(())
    layer_scale = mtp_config.loss_scaling_factor / mtp_config.num_layers

    for layer, batch_step in zip(runtime.layers, mtp_batch.steps, strict=True):
        input_ids = batch_step.input_ids
        position_ids = batch_step.position_ids
        labels = batch_step.labels
        loss_mask = batch_step.loss_mask

        input_embeds = runtime.embed_tokens(input_ids).detach()
        mtp_hidden_states = runtime.run_layer(
            layer,
            runtime,
            input_embeds,
            mtp_hidden_states,
            position_ids,
            mtp_batch.cp_group,
            mtp_batch.cp_cu_seqlens,
        )

        logits = _apply_output_head(mtp_hidden_states, model)
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)
        masked_loss_sum = token_loss[loss_mask].sum()
        raw_loss_sum = raw_loss_sum + masked_loss_sum
        raw_token_count = raw_token_count + loss_mask.sum(dtype=torch.long)
        scaled_loss_sum = scaled_loss_sum + masked_loss_sum * layer_scale
    if raw_token_count.item() == 0:
        zero = hidden_states.new_zeros(())
        return zero, zero

    return scaled_loss_sum, raw_loss_sum / raw_token_count


__all__ = [
    "MTPTrainingBatch",
    "compute_mtp_loss",
    "configure_mtp_training",
    "prepare_mtp_training_batch",
    "prepare_mtp_training_batch_from_current_tokens",
    "shard_mtp_training_batch_for_cp",
]
