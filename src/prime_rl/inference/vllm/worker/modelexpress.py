from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import (
    load_weights_checkpoint,
    postprocess_weights_checkpoint,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("prime_rl.inference.vllm.worker.modelexpress")

MX_AVAILABLE = False
MxRefitReceiver = None
try:
    from modelexpress import MxRefitReceiver as _MxRefitReceiver
    MX_AVAILABLE = True
    MxRefitReceiver = _MxRefitReceiver
except ImportError:
    pass


class MxWeightUpdateWorker(Worker):
    """vLLM worker extension for receiving weights via ModelExpress NIXL/RDMA.

    Uses a scratch-buffer approach: RDMA-receives HF-format weights into
    temporary GPU tensors, then feeds them through model.load_weights()
    which handles name mapping and tensor fusion (e.g. q/k/v -> qkv_proj).
    """

    def init_broadcaster(self, mx_server_url: str = "localhost:8001", **kwargs) -> None:
        """Initialize the MxRefitReceiver NIXL agent (no tensor registration)."""
        import sys

        if not MX_AVAILABLE:
            logger.warning(
                "modelexpress package not installed; MX weight updates will "
                "fall back to filesystem loading"
            )
            self._mx_receiver = None
            return

        from vllm.distributed.parallel_state import get_tp_group, get_dp_group

        tp_rank = get_tp_group().rank_in_group
        dp_rank = get_dp_group().rank_in_group
        global_rank = dp_rank * get_tp_group().world_size + tp_rank

        try:
            self._mx_receiver = MxRefitReceiver(
                agent_name=f"inference-rank-{global_rank}",
                device_id=self.device.index,
                mx_server_url=mx_server_url,
            )
            self._mx_receiver.initialize(model_tensors=None)
            print(
                f"[MX] MxRefitReceiver initialized: rank={global_rank}, "
                f"device={self.device.index}, mx_server={mx_server_url} "
                f"(scratch-buffer mode, tensors registered on-demand)",
                file=sys.stderr, flush=True,
            )
        except Exception as e:
            import traceback
            print(f"[MX] init_broadcaster FAILED: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            self._mx_receiver = None

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Receive updated weights via MX RDMA, falling back to filesystem."""
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        if self._try_mx_update(model):
            return

        logger.info("Falling back to filesystem weight loading")
        self._filesystem_fallback(model, weight_dir)

    def _try_mx_update(self, model: Module) -> bool:
        """Attempt to receive weights from MX via scratch buffers."""
        import sys

        if not hasattr(self, "_mx_receiver") or self._mx_receiver is None:
            return False

        model_name = getattr(
            self.model_runner.model_config, "model", "unknown"
        )

        try:
            source = self._mx_receiver.poll_for_source(
                model_name=model_name,
                timeout_seconds=30.0,
            )
        except Exception as e:
            logger.warning(f"MX poll_for_source failed: {e}")
            return False

        if source is None:
            logger.info("No MX source found for weight update")
            return False

        print(
            f"[MX] Source found: step={source.training_step}, "
            f"id={source.mx_source_id}",
            file=sys.stderr, flush=True,
        )

        try:
            weights_iter = self._mx_receiver.receive_weights_scratch(source)
            model.load_weights(weights_iter)
            process_weights_after_loading(
                model, self.model_runner.model_config, self.device
            )
            print(
                f"[MX] Weight update complete via RDMA "
                f"(step={source.training_step})",
                file=sys.stderr, flush=True,
            )
            return True
        except Exception as e:
            import traceback
            print(f"[MX] Weight transfer failed: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return False

    def _filesystem_fallback(self, model: Module, weight_dir: str) -> None:
        """Load weights from shared filesystem."""
        from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        local_source = DefaultModelLoader.Source(
            weight_dir,
            revision=None,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        weights_iterator = model_loader._get_weights_iterator(local_source)
        model.load_weights(weights_iterator)
        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)
