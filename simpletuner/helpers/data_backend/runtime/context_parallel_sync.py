"""Context-parallel batch synchronization for distributed training.

When using context parallelism, all ranks within a model replica must receive the
same batch data. This module provides utilities to synchronize batch sampling so
that:
- Rank 0 of each replicated data-parallel group samples the batch
- Other ranks in the same model replica receive the sampled batch via broadcast

This ensures that when the batch is later split along the sequence dimension by
the model's _cp_plan, all FSDP shard and CP ranks have consistent data.

The synchronization addresses two key issues:
1. Data sharding: The dataset is split by replicated DP rank only, not by FSDP
   shard rank or CP rank.
2. Batch sampling: Only the model-replica leader samples batches; other ranks
   receive via broadcast. This keeps seen_images tracking consistent across the
   model-parallel ranks.
"""

import logging
import numbers
import os
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist

from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger("ContextParallelSync")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)


# Sentinel value for non-leader ranks that skip sampling
CP_SKIP_SAMPLING_SENTINEL = "__CP_SKIP_SAMPLING__"


def _normalize_parallel_size(value: Any, name: str) -> int:
    if value is None:
        return 1
    if isinstance(value, torch.SymInt):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    raise TypeError(f"{name} must be an integer type, got {type(value).__name__}.")


def get_cp_info(accelerator) -> Tuple[bool, Optional[Any], int, int]:
    """
    Extract context-parallel information from the accelerator.

    Returns:
        Tuple of:
        - cp_enabled: Whether context parallelism is active
        - cp_group: The ProcessGroup for this rank's CP group (or None)
        - cp_rank: This rank's position within its CP group (0-indexed)
        - cp_size: Number of ranks in the CP group
    """
    parallelism_config = getattr(accelerator, "parallelism_config", None)
    if parallelism_config is None:
        return False, None, 0, 1

    cp_size = getattr(parallelism_config, "cp_size", None)
    if cp_size is None:
        return False, None, 0, 1
    if isinstance(cp_size, torch.SymInt):
        cp_size = int(cp_size)
    elif isinstance(cp_size, numbers.Integral):
        cp_size = int(cp_size)
    else:
        return False, None, 0, 1
    if cp_size <= 1:
        return False, None, 0, 1

    cp_enabled = getattr(parallelism_config, "cp_enabled", False)
    if not cp_enabled:
        return False, None, 0, 1

    # Get the device mesh from accelerator
    device_mesh = getattr(accelerator, "torch_device_mesh", None)
    if device_mesh is None:
        logger.warning(
            "Context parallelism enabled but no device mesh found on accelerator. " "Batch synchronization will be skipped."
        )
        return False, None, 0, 1

    # Try to find the CP dimension in the mesh
    # The mesh dimension name should be "cp" when using accelerate's ParallelismConfig
    try:
        cp_group = device_mesh.get_group("cp")
        cp_rank = device_mesh.get_local_rank("cp")
        return True, cp_group, cp_rank, cp_size
    except Exception as e:
        # Mesh may use different dimension names depending on config
        logger.debug(f"Could not get 'cp' dimension from mesh: {e}")

        # Try alternative mesh dimension names
        mesh_dim_names = getattr(device_mesh, "mesh_dim_names", None)
        if mesh_dim_names:
            for dim_name in mesh_dim_names:
                if "cp" in dim_name.lower() or "context" in dim_name.lower():
                    try:
                        cp_group = device_mesh.get_group(dim_name)
                        cp_rank = device_mesh.get_local_rank(dim_name)
                        return True, cp_group, cp_rank, cp_size
                    except Exception:
                        pass

        logger.warning(
            f"Could not find CP dimension in mesh (dims: {mesh_dim_names}). Batch synchronization will be skipped."
        )
        return False, None, 0, 1


def sync_batch_for_context_parallel(
    batch: Any,
    accelerator,
    cp_info: Optional[Tuple[bool, Optional[Any], int, int]] = None,
) -> Any:
    """
    Synchronize batch data across ranks in a context-parallel group.

    In FSDP2 + context parallelism, all ranks in the same model replica must
    receive the same input data. FSDP shard ranks and CP ranks are
    model-parallel, while only DP-replicate ranks represent unique data shards.

    Args:
        batch: The batch data (can be a tuple, dict, or any picklable object)
        accelerator: The Accelerator instance with parallelism_config
        cp_info: Pre-computed CP info tuple from get_cp_info (optional, for efficiency)

    Returns:
        The synchronized batch data
    """
    if cp_info is None:
        cp_enabled, cp_group, cp_rank, cp_size = get_cp_info(accelerator)
    else:
        cp_enabled, cp_group, cp_rank, cp_size = cp_info

    if not cp_enabled:
        return batch

    if cp_group is None:
        logger.debug("CP enabled but no process group available, skipping sync")
        return batch

    try:
        data_enabled, data_rank, data_local_rank, data_group_size, data_parallel_size = get_model_replica_data_info(
            accelerator, cp_info
        )
        if not data_enabled:
            return batch

        replica_batch = None
        for replica_rank in range(data_parallel_size):
            leader_global_rank = replica_rank * data_group_size
            batch_list = [batch if data_rank == replica_rank and data_local_rank == 0 else None]
            dist.broadcast_object_list(batch_list, src=leader_global_rank)
            if data_rank == replica_rank:
                replica_batch = batch_list[0]
        return replica_batch
    except Exception as e:
        logger.error(f"Failed to broadcast batch in CP/FSDP model replica: {e}")
        raise RuntimeError("Context-parallel batch broadcast failed.") from e


def get_model_replica_data_info(
    accelerator,
    cp_info: Optional[Tuple[bool, Optional[Any], int, int]] = None,
) -> Tuple[bool, int, int, int, int]:
    if cp_info is None:
        cp_enabled, _cp_group, _cp_rank, cp_size = get_cp_info(accelerator)
    else:
        cp_enabled, _cp_group, _cp_rank, cp_size = cp_info
    if not cp_enabled:
        return False, 0, 0, 1, 1

    parallelism_config = getattr(accelerator, "parallelism_config", None)
    if parallelism_config is None:
        return False, 0, 0, 1, 1

    dp_replicate_size = _normalize_parallel_size(getattr(parallelism_config, "dp_replicate_size", 1), "dp_replicate_size")
    dp_shard_size = _normalize_parallel_size(getattr(parallelism_config, "dp_shard_size", 1), "dp_shard_size")
    world_size = _normalize_parallel_size(getattr(accelerator, "num_processes", 1), "num_processes")
    process_index = _normalize_parallel_size(getattr(accelerator, "process_index", 0), "process_index")

    data_group_size = dp_shard_size * cp_size
    expected_world_size = dp_replicate_size * data_group_size
    if world_size != expected_world_size:
        raise ValueError(
            "Context parallel batch synchronization expected "
            f"num_processes={expected_world_size} from dp_replicate_size={dp_replicate_size}, "
            f"dp_shard_size={dp_shard_size}, cp_size={cp_size}; got {world_size}."
        )

    data_rank = process_index // data_group_size
    data_local_rank = process_index % data_group_size
    return True, data_rank, data_local_rank, data_group_size, dp_replicate_size


class ContextParallelBatchSynchronizer:
    """
    Caches context-parallel information for efficient batch synchronization.

    Usage:
        synchronizer = ContextParallelBatchSynchronizer(accelerator)

        for step in training_loop:
            batch = dataloader_iterator(step)
            batch = synchronizer.sync(batch)
            # All ranks in CP group now have the same batch
    """

    def __init__(self, accelerator):
        self.accelerator = accelerator
        self._cp_info = None
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            self._cp_info = get_cp_info(self.accelerator)
            self._initialized = True

            cp_enabled, cp_group, cp_rank, cp_size = self._cp_info
            if cp_enabled:
                _, data_rank, data_local_rank, data_group_size, data_parallel_size = get_model_replica_data_info(
                    self.accelerator, self._cp_info
                )
                logger.info(
                    "Context parallel batch sync initialized: "
                    f"cp_rank={cp_rank}, cp_size={cp_size}, data_rank={data_rank}, "
                    f"data_local_rank={data_local_rank}, data_group_size={data_group_size}, "
                    f"data_parallel_size={data_parallel_size}"
                )

    @property
    def is_cp_enabled(self) -> bool:
        """Check if context parallelism is enabled."""
        self._ensure_initialized()
        return self._cp_info[0]

    @property
    def is_cp_leader(self) -> bool:
        """Check if this rank samples for its model replica."""
        self._ensure_initialized()
        if not self._cp_info[0]:
            return True
        return get_model_replica_data_info(self.accelerator, self._cp_info)[2] == 0

    @property
    def cp_rank(self) -> int:
        """Get this rank's position within its CP group."""
        self._ensure_initialized()
        return self._cp_info[2]

    @property
    def cp_size(self) -> int:
        """Get the number of ranks in the CP group."""
        self._ensure_initialized()
        return self._cp_info[3]

    def sync(self, batch: Any) -> Any:
        """
        Synchronize batch across the CP group.

        Only one rank per model replica samples; other ranks receive the broadcast.

        Args:
            batch: The batch data from the dataloader

        Returns:
            The synchronized batch (same on all ranks in CP group)
        """
        self._ensure_initialized()
        return sync_batch_for_context_parallel(batch, self.accelerator, self._cp_info)

    def fetch_batch(self, iterator_fn, step: int, *iterator_args) -> Any:
        """
        Fetch a batch with CP-aware sampling.

        When CP is enabled:
        - The model-replica leader calls the iterator to sample a batch
        - Other ranks skip sampling and receive the batch via broadcast
        - This ensures seen_images tracking is consistent across model-parallel ranks

        When CP is disabled:
        - All ranks call the iterator normally

        Args:
            iterator_fn: The iterator function to call (e.g., random_dataloader_iterator)
            step: The current training step
            *iterator_args: Additional arguments to pass to the iterator

        Returns:
            The batch (synchronized across CP group if CP is enabled)
        """
        self._ensure_initialized()
        cp_enabled, cp_group, cp_rank, cp_size = self._cp_info

        if not cp_enabled:
            # No CP - just call the iterator normally
            return iterator_fn(step, *iterator_args)

        data_local_rank = get_model_replica_data_info(self.accelerator, self._cp_info)[2]

        if data_local_rank == 0:
            # Model-replica leader: sample the batch normally
            batch = iterator_fn(step, *iterator_args)
        else:
            # Non-leader: use sentinel (will be replaced by broadcast)
            batch = CP_SKIP_SAMPLING_SENTINEL

        # Broadcast from leader to all model-parallel ranks in the model replica
        return self.sync(batch)
