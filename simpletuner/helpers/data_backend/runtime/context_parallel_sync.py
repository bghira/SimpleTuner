"""Context-parallel batch synchronization for distributed training.

When using context parallelism, all ranks within a CP group must receive the same
batch data. This module provides utilities to synchronize batch sampling so that:
- Rank 0 of each CP group samples the batch
- Other ranks in the same CP group receive the sampled batch via broadcast

This ensures that when the batch is later split along the sequence dimension by
the model's _cp_plan, all ranks in the group have consistent data.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger("ContextParallelSync")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)


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
    if cp_size is None or cp_size <= 1:
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
            f"Could not find CP dimension in mesh (dims: {mesh_dim_names}). " "Batch synchronization will be skipped."
        )
        return False, None, 0, 1


def sync_batch_for_context_parallel(
    batch: Any,
    accelerator,
    cp_info: Optional[Tuple[bool, Optional[Any], int, int]] = None,
) -> Any:
    """
    Synchronize batch data across ranks in a context-parallel group.

    In context parallelism, all ranks within a CP group must receive the same
    input data, which is then split along the sequence dimension. This function
    ensures that rank 0 of each CP group broadcasts its sampled batch to all
    other ranks in the group.

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

    # Broadcast the batch from rank 0 of the CP group
    # Use broadcast_object_list for arbitrary Python objects
    batch_list = [batch if cp_rank == 0 else None]

    try:
        # Get the global rank of the source (rank 0 within the CP group)
        # The src for broadcast must be a global rank within the group
        group_ranks = list(range(dist.get_world_size()))
        if hasattr(cp_group, "ranks"):
            group_ranks = cp_group.ranks()
        elif hasattr(cp_group, "rank"):
            # ProcessGroup may expose ranks differently
            pass

        # For ProcessGroup from device_mesh.get_group(), src=0 means rank 0 within that group
        dist.broadcast_object_list(batch_list, src=0, group=cp_group)
        return batch_list[0]
    except Exception as e:
        logger.error(f"Failed to broadcast batch in CP group: {e}")
        # Return the original batch as fallback (may cause inconsistency)
        return batch


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
                logger.info(f"Context parallel batch sync initialized: " f"cp_rank={cp_rank}, cp_size={cp_size}")

    @property
    def is_cp_enabled(self) -> bool:
        """Check if context parallelism is enabled."""
        self._ensure_initialized()
        return self._cp_info[0]

    @property
    def is_cp_leader(self) -> bool:
        """Check if this rank is the leader (rank 0) of its CP group."""
        self._ensure_initialized()
        return self._cp_info[2] == 0

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

        Only rank 0 of each CP group samples; other ranks receive the broadcast.

        Args:
            batch: The batch data from the dataloader

        Returns:
            The synchronized batch (same on all ranks in CP group)
        """
        self._ensure_initialized()
        return sync_batch_for_context_parallel(batch, self.accelerator, self._cp_info)
