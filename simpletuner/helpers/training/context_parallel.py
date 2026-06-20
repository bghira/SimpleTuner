"""SimpleTuner-owned context parallel topology helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

logger = logging.getLogger("SimpleTuner")


@dataclass(frozen=True)
class ContextParallelTopology:
    cp_size: int
    dp_replicate_size: int
    dp_shard_size: int
    strategy: str


def normalize_context_parallel_size(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        cp_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Context parallel size must be an integer, got {value!r}.") from exc
    if cp_size <= 0:
        raise ValueError("Context parallel size must be greater than 0 when specified.")
    return cp_size


def normalize_context_parallel_strategy(value: Any) -> str:
    strategy = value or "allgather"
    strategy = strategy.strip().lower() if isinstance(strategy, str) else "allgather"
    if strategy not in {"allgather", "alltoall"}:
        raise ValueError(f"Unsupported context parallel rotation '{value}'. Valid options are 'allgather' and 'alltoall'.")
    return strategy


def resolve_context_parallel_world_size(config: Any) -> int:
    for world_size_candidate in (
        os.environ.get("WORLD_SIZE"),
        os.environ.get("TRAINING_NUM_PROCESSES"),
        getattr(config, "num_processes", None),
    ):
        if world_size_candidate in (None, "", "None"):
            continue
        try:
            return int(world_size_candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Context parallelism requires an integer process count, got {world_size_candidate!r}."
            ) from exc
    raise ValueError("Context parallelism requires a known process count before Accelerator setup.")


def build_context_parallel_topology(config: Any, world_size: int, cp_size: int, strategy: str) -> ContextParallelTopology:
    if world_size < cp_size:
        raise ValueError(f"Context parallel size ({cp_size}) cannot exceed process count ({world_size}).")
    if world_size % cp_size != 0:
        raise ValueError(f"Context parallel size ({cp_size}) must evenly divide process count ({world_size}).")

    fsdp_enabled = bool(getattr(config, "fsdp_enable", False))
    if fsdp_enabled:
        fsdp_version_value = getattr(config, "fsdp_version", 2)
        try:
            fsdp_version = int(fsdp_version_value)
        except (TypeError, ValueError):
            fsdp_version = 2
        if fsdp_version != 2:
            raise ValueError("Context parallelism currently only supports FSDP version 2 when FSDP is enabled.")
        return ContextParallelTopology(
            cp_size=cp_size,
            dp_replicate_size=1,
            dp_shard_size=world_size // cp_size,
            strategy=strategy,
        )

    return ContextParallelTopology(
        cp_size=cp_size,
        dp_replicate_size=world_size // cp_size,
        dp_shard_size=1,
        strategy=strategy,
    )


def _accelerator_state(accelerator: Any) -> Any:
    state = getattr(accelerator, "state", None)
    if state is None:
        state = SimpleNamespace()
        setattr(accelerator, "state", state)
    return state


def _attach_accelerator_cp_state(accelerator: Any, parallelism_config: Any, device_mesh: Any) -> None:
    state = _accelerator_state(accelerator)
    state.parallelism_config = parallelism_config
    state.device_mesh = device_mesh
    try:
        setattr(accelerator, "parallelism_config", parallelism_config)
    except Exception:
        pass
    try:
        setattr(accelerator, "torch_device_mesh", device_mesh)
    except Exception:
        pass


def configure_cp_only_accelerator(accelerator: Any, topology: ContextParallelTopology) -> None:
    if topology.cp_size <= 1 or topology.dp_shard_size != 1:
        return
    use_ulysses = topology.strategy == "alltoall"
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        raise RuntimeError("Standalone context parallelism requires torch.distributed to be initialized.")

    from torch.distributed.device_mesh import init_device_mesh

    device = getattr(accelerator, "device", None)
    device_type = device.type if isinstance(device, torch.device) else torch._C._get_accelerator().type
    mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(
            topology.dp_replicate_size,
            1 if use_ulysses else topology.cp_size,
            topology.cp_size if use_ulysses else 1,
        ),
        mesh_dim_names=("dp_replicate", "ring", "ulysses"),
    )
    cp_handler = SimpleNamespace(cp_comm_strategy=topology.strategy)
    parallelism_config = SimpleNamespace(
        dp_replicate_size=topology.dp_replicate_size,
        dp_shard_size=topology.dp_shard_size,
        tp_size=1,
        sp_size=1,
        cp_size=topology.cp_size,
        cp_backend="simpletuner",
        cp_handler=cp_handler,
        cp_enabled=topology.cp_size > 1,
        dp_replicate_enabled=topology.dp_replicate_size > 1,
        dp_shard_enabled=False,
        tp_enabled=False,
        sp_enabled=False,
    )
    _attach_accelerator_cp_state(accelerator, parallelism_config, mesh)
    logger.info(
        "Standalone context parallelism enabled (size=%s, rotation=%s, dp_replicate_size=%s).",
        topology.cp_size,
        topology.strategy,
        topology.dp_replicate_size,
    )


def apply_standalone_context_parallel(
    accelerator: Any, module: torch.nn.Module | None, topology: ContextParallelTopology | None
) -> bool:
    if module is None or topology is None or topology.cp_size <= 1 or topology.dp_shard_size != 1:
        return False
    if not hasattr(module, "enable_parallelism"):
        raise ValueError(
            f"{module.__class__.__name__} does not expose enable_parallelism(), so standalone context parallelism "
            "cannot apply its _cp_plan hooks."
        )
    if getattr(module, "_simpletuner_context_parallel_enabled", False):
        return False
    if getattr(module, "_cp_plan", None) is None:
        raise ValueError(
            f"{module.__class__.__name__} does not define _cp_plan, so it cannot be used with context_parallel_size."
        )

    from diffusers.models._modeling_parallel import ContextParallelConfig

    mesh = getattr(accelerator, "torch_device_mesh", None)
    if mesh is None:
        mesh = getattr(getattr(accelerator, "state", None), "device_mesh", None)
    if mesh is None:
        raise RuntimeError("Standalone context parallelism has no device mesh attached to the accelerator.")

    cp_config = ContextParallelConfig(
        ring_degree=1 if topology.strategy == "alltoall" else topology.cp_size,
        ulysses_degree=topology.cp_size if topology.strategy == "alltoall" else 1,
        rotate_method="allgather",
        mesh=mesh,
        ring_anything=topology.strategy == "allgather",
        ulysses_anything=False,
    )
    module.enable_parallelism(config=cp_config)
    setattr(module, "_simpletuner_context_parallel_enabled", True)
    logger.info("Applied standalone context parallel hooks to %s.", module.__class__.__name__)
    return True
