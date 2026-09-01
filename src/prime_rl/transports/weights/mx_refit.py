"""Publish FSDP weights through ModelExpress's slice-reshard transport (Model B).

Every trainer rank hands its FSDP state_dict to MX's trainer client, which stages
the rank-local shards and advertises them under a per-step WeightVersion. This
rank owns the version lifecycle: rank 0 creates the version and every rank
publishes its shard. Both the trainer and the orchestrator name the version
``{run_uid}:{step}``, so its identity never has to travel between them.
``broadcast_weights`` blocks until the generator has pulled (the orchestrator
moves the version to RELEASING) before letting training reuse its staging
buffers.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import grpc
import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress_rl import (
    FSDPTrainerContext,
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    WeightVersionState,
)

from prime_rl.configs.trainer import MXRefitWeightBroadcastConfig
from prime_rl.orchestrator.clients import init_mx_refit_broadcast, update_weights
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.transports.weights.base import WeightReceiver, WeightSender

RELEASE_POLL_INTERVAL = 0.05


def weight_version_uid(run_uid: str, step: int) -> str:
    """Return the WeightVersion uid the trainer publishes for ``step``.

    MX lets the caller choose a version's identity, and both processes derive it
    from configuration, so the trainer->orchestrator handoff needs no shared
    filesystem. Naming versions after the step also makes a stalled refit legible
    on the server: the live uid says which step is in flight.
    """
    return f"{run_uid}:{step}"


def version_missing(error: grpc.RpcError) -> bool:
    """Whether ``error`` means the trainer has not created the version yet."""
    return error.code() is grpc.StatusCode.NOT_FOUND


async def resolve_ready_version(
    control: ModelExpressControlClient,
    uid: str,
    poll_interval: float = 0.5,
    stopped: asyncio.Event | None = None,
) -> str:
    """Wait until ``uid`` exists and every trainer rank has published into it.

    Shared by the orchestrator's startup sync and the watcher's steady-state gate:
    the mx_refit version lifecycle is uniform, so v0 and every later version use
    the same resolve path. ``stopped``, when given, lets the caller cancel the poll.

    NOT_FOUND is a wait rather than an error. The orchestrator can name a step's
    version before the trainer has reached that step, which is exactly the
    condition the old marker file signalled by its absence.
    """
    while stopped is None or not stopped.is_set():
        try:
            version = await asyncio.to_thread(control.get_weight_version, uid)
        except grpc.RpcError as error:
            if not version_missing(error):
                raise
        else:
            if version.state is WeightVersionState.READY:
                return uid
            if version.state is WeightVersionState.RELEASING:
                raise RuntimeError(f"MX version {uid} was retired before it became READY")
        await asyncio.sleep(poll_interval)
    raise asyncio.CancelledError


class MXRefitWeightSender(WeightSender):
    def __init__(
        self,
        output_dir: Path,
        config: MXRefitWeightBroadcastConfig,
        parallel_dims: ParallelDims,
        model_name: str,
    ) -> None:
        super().__init__(output_dir, config.timeout)
        self.config = config
        del parallel_dims
        self.model_name = model_name
        self._initialized = False
        self._client: ModelExpressTrainerClient | None = None
        self._control: ModelExpressControlClient | None = None
        self._expected_slots: list[str] = []

    @property
    def server_url(self) -> str:
        return f"{self.config.host}:{self.config.port}"

    def _initialize(self, model: nn.Module) -> None:
        self._client = ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                engine_context=FSDPTrainerContext(),
                model_name=self.model_name,
                device_id=self.world.local_rank,
                server_url=self.server_url,
                staging_mode=TrainerStagingMode.COPY_TO_DEVICE,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
            )
        )
        slot = self._client.bind_tensors(model.state_dict())
        self._control = ModelExpressControlClient.connect(server_url=self.server_url)

        # Collect every rank's actual source slot so rank 0 can pass the complete
        # expected_source_slots to create_weight_version. Reading the real slots
        # avoids hardcoding the slot-naming convention; set() stays safe if DP
        # replicas ever share a logical slot.
        gathered: list[str] = [""] * self.world.world_size
        dist.all_gather_object(gathered, slot)
        self._expected_slots = sorted(set(gathered))
        self._initialized = True

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        del step_dir
        if not self._initialized:
            self._initialize(model)
        assert self._client is not None
        assert self._control is not None

        # Every rank derives the uid, so there is nothing to broadcast. The
        # collective still has to stay, as a barrier: it is what guarantees rank 0
        # has created the version before any rank publishes a shard into it.
        uid = weight_version_uid(self.config.run_uid, step)
        if self.world.is_master:
            self._control.create_weight_version(
                model_name=self.model_name,
                idempotency_key=uid,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                expected_source_slots=self._expected_slots,
                uid=uid,
            )
        dist.barrier()

        self._client.publish_version(version=WeightVersionRef(uid))

        # Every rank waits for the receiver to retire the version before releasing
        # its own staging buffer. Independent bounded waits avoid stranding peers
        # at a collective if one publisher or control RPC fails.
        self._wait_released(uid)
        self._client.release_version(version=WeightVersionRef(uid))

    def _wait_released(self, uid: str) -> None:
        assert self._control is not None
        # The receiver independently bounds discovery and transfer by one timeout
        # each, so retain staging for both phases before failing locally.
        release_timeout = 2 * self.timeout
        deadline = time.monotonic() + release_timeout
        while True:
            try:
                state = self._control.get_weight_version(uid).state
            except grpc.RpcError as error:
                if version_missing(error):
                    return  # already retired == generator done
                raise
            if state is WeightVersionState.RELEASING:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"MX version {uid} was not released within {release_timeout}s")
            time.sleep(RELEASE_POLL_INTERVAL)


class MXRefitWeightReceiver(WeightReceiver):
    """Pulls a trainer-published MX version into every inference worker."""

    poll_interval = 0.1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.control: ModelExpressControlClient | None = None

    @property
    def server_url(self) -> str:
        return f"{self.config.host}:{self.config.port}"

    async def initialize(self) -> None:
        await init_mx_refit_broadcast(
            self.admin_clients,
            self.config.host,
            self.config.port,
            self.config.timeout,
        )
        self.control = ModelExpressControlClient.connect(server_url=self.server_url)

    async def receive(self, step: int) -> None:
        assert self.control is not None
        uid = weight_version_uid(self.config.run_uid, step)
        self._ack(step)
        await asyncio.wait_for(
            resolve_ready_version(self.control, uid),
            timeout=self.config.timeout,
        )
        try:
            await update_weights(
                self.admin_clients,
                None,
                step=step,
                version_uid=uid,
                timeout_s=self.config.timeout,
            )
        finally:
            # Retiring on both success and failure unblocks trainer ranks that
            # still own staging buffers for this version.
            await asyncio.shield(asyncio.to_thread(self.control.delete_weight_version, uid))
