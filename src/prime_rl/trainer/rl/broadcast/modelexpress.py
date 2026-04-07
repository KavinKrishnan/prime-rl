import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.nccl import (
    filter_state_dict_by_layers,
    preprocess_layer_checkpoint,
)
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.weights import gather_weights_on_master, get_max_layer_num
from prime_rl.trainer.world import get_world
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix

try:
    from modelexpress import MxTrainingPublisher
    MX_AVAILABLE = True
except ImportError:
    MX_AVAILABLE = False


class ModelExpressWeightBroadcast(WeightBroadcast):
    """Broadcast weights to inference via ModelExpress NIXL/RDMA.

    Gathers distributed (DTensor) weights on the master rank, converts to
    HuggingFace format, registers each layer with NIXL via MxTrainingPublisher,
    and publishes metadata to the MX Server. Inference workers discover the
    source and pull weights over RDMA.

    Falls back to filesystem broadcast if ModelExpress is unavailable.
    """

    def __init__(
        self,
        output_dir: Path,
        config,
        lora_config: LoRAConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(output_dir, lora_config)
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self.dtype = dtype
        self._mx_server_url = getattr(config, "mx_server_url", "localhost:8001")
        self._publisher: "MxTrainingPublisher | None" = None
        self._model_name: str = ""

        if not MX_AVAILABLE:
            self.logger.warning(
                "modelexpress package not installed; MX broadcast will fall back to filesystem"
            )

    def _ensure_publisher(self, model: nn.Module) -> bool:
        """Lazily initialize the MxTrainingPublisher on first use."""
        if self._publisher is not None:
            return True
        if not MX_AVAILABLE:
            return False

        model_name = getattr(model, "name_or_path", "") or getattr(
            getattr(model, "config", None), "name_or_path", "unknown"
        )
        self._model_name = model_name

        self._publisher = MxTrainingPublisher(
            agent_name=f"trainer-rank-{self.world.rank}",
            device_id=self.world.local_rank,
            mx_server_url=self._mx_server_url,
        )
        self._publisher.initialize(model_name=model_name)
        self.logger.info(
            f"ModelExpress publisher initialized for model={model_name}"
        )
        return True

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast the model's weights to inference via MX RDMA.

        Follows the same gather-convert-stream pattern as the NCCL backend:
        1. Gather DTensor parameters on master rank (all ranks participate).
        2. Convert to HuggingFace format (layer by layer).
        3. Register each layer with NIXL and publish metadata.
        4. Touch STABLE file for orchestrator notification.
        """
        self.logger.debug("Starting MX weight broadcast to inference")
        start_time = time.perf_counter()

        state_dict = gather_weights_on_master(model, is_master=self.world.is_master, dtype=self.dtype)

        if not self.world.is_master:
            return

        mx_ok = self._ensure_publisher(model)

        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)

        if mx_ok:
            for layer_id, layer_state_dict in filter_state_dict_by_layers(
                state_dict, num_layers, layer_prefix
            ):
                layer_state_dict = self._resolve_dtensors(layer_state_dict)
                layer_state_dict = preprocess_layer_checkpoint(model, layer_state_dict, layer_id)

                gpu_tensors = {
                    k: v.to(f"cuda:{self.world.local_rank}", non_blocking=False)
                    for k, v in layer_state_dict.items()
                    if isinstance(v, Tensor) and v.numel() > 0
                }

                if gpu_tensors:
                    self._publisher.publish_layer(gpu_tensors, layer_id, step)

                del gpu_tensors
                torch.cuda.empty_cache()

            self._publisher.mark_ready()
            self.logger.info(
                f"MX broadcast complete: {num_layers + 1} layers published for step {step}"
            )
        else:
            self.logger.warning("MX unavailable, falling back to filesystem STABLE notification only")

        for idx in self.multi_run_manager.ready_to_update_idxs:
            try:
                save_dir = get_step_path(
                    get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                    self.multi_run_manager.progress[idx].step,
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                self._notify_orchestrator(save_dir)
            except FileNotFoundError:
                self.logger.warning(f"Run {idx} deleted, skipping notification")
            except Exception as e:
                self.logger.error(f"Error notifying orchestrator for run {idx}: {e}")
            finally:
                self.multi_run_manager.ready_to_update[idx] = False

        elapsed = time.perf_counter() - start_time
        self.logger.debug(f"MX weights broadcasted in {elapsed:.2f}s")

    def _resolve_dtensors(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        for key, value in list(state_dict.items()):
            if isinstance(value, DTensor):
                state_dict[key] = cast(DTensor, value.to(self.dtype)).full_tensor()
        return state_dict

    def _notify_orchestrator(self, save_dir: Path):
        """Write STABLE marker for orchestrator to detect (same protocol as filesystem/NCCL)."""
        stable_file = save_dir / "STABLE"
        stable_file.touch()
