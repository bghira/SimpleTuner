"""
Backfill RamTorch features that may not exist in the current PyPI release.

This module monkeypatches RamTorch to:
- Add attach_shared_ramtorch_parameters for torchrun/Accelerate launches
- Extend broadcast_zero_params with include_ramtorch support
"""

from __future__ import annotations

import inspect
from typing import Iterable, Optional, Tuple

import torch
import torch.distributed as dist


def _ramtorch_params(module: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, p)
        for name, p in module.named_parameters()
        if isinstance(p, torch.nn.Parameter) and getattr(p, "is_ramtorch", False)
    ]


def _shared_metadata(
    param: torch.nn.Parameter,
) -> tuple[tuple[bytes, bytes, int], Tuple[int, ...], Tuple[int, ...], torch.dtype]:
    storage = param.detach().untyped_storage()
    handle = storage._share_filename_cpu_()  # type: ignore[attr-defined]
    return handle, tuple(param.size()), tuple(param.stride()), param.dtype


def _attach_shared_tensor(
    param: torch.nn.Parameter,
    handle: tuple[bytes, bytes, int],
    shape: Tuple[int, ...],
    stride: Tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    manager, name, size = handle
    shared_storage = torch.UntypedStorage._new_shared_filename_cpu(manager, name, size)  # type: ignore[attr-defined]
    shared_tensor = torch.empty(0, dtype=dtype)
    shared_tensor.set_(shared_storage, 0, shape, stride)
    param.data = shared_tensor
    param.is_ramtorch = True


def _monkeypatch_attach_shared_params() -> bool:
    try:
        import ramtorch.helpers as rh  # type: ignore
    except Exception:
        return False

    if hasattr(rh, "attach_shared_ramtorch_parameters"):
        return False

    def attach_shared_ramtorch_parameters(module: torch.nn.Module, process_group: Optional[dist.ProcessGroup] = None) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 0

        params = _ramtorch_params(module)
        if not params:
            return 0

        group = process_group if process_group is not None else dist.group.WORLD
        rank = dist.get_rank(group)

        metadata: list[Optional[tuple[str, tuple[bytes, bytes, int], Tuple[int, ...], Tuple[int, ...], torch.dtype]]] = [
            None for _ in params
        ]

        if rank == 0:
            for idx, (name, param) in enumerate(params):
                metadata[idx] = (name, *_shared_metadata(param))

        dist.broadcast_object_list(metadata, src=0, group=group)

        attached = 0
        for (_, param), entry in zip(params, metadata):
            if entry is None:
                continue

            _, handle, shape, stride, dtype = entry
            if rank != 0:
                _attach_shared_tensor(param, handle, shape, stride, dtype)
            else:
                storage = param.detach().untyped_storage()
                if not storage.is_shared():
                    param.data.share_memory_()

            param.is_ramtorch = True
            attached += 1

        dist.barrier(group)
        return attached

    rh.attach_shared_ramtorch_parameters = attach_shared_ramtorch_parameters  # type: ignore[attr-defined]
    return True


def _monkeypatch_broadcast_zero_params() -> bool:
    try:
        import ramtorch.zero1 as zero1  # type: ignore
    except Exception:
        return False

    if "include_ramtorch" in inspect.signature(zero1.broadcast_zero_params).parameters:  # type: ignore[attr-defined]
        return False

    orig = zero1.broadcast_zero_params  # type: ignore[attr-defined]

    def broadcast_zero_params(rank_param_groups: dict, async_op: bool = True, include_ramtorch: bool = False):
        work_handles: list[dist.Work] = []
        with torch.no_grad():
            for owner_rank, param_groups in rank_param_groups.items():
                for group in param_groups:
                    for param in group["params"]:
                        if getattr(param, "is_ramtorch", False) and not include_ramtorch:
                            continue
                        work_handle = dist.broadcast(param.data, src=owner_rank, async_op=async_op)
                        if async_op:
                            work_handles.append(work_handle)

            if work_handles and async_op:
                for handle in work_handles:
                    handle.wait()

    broadcast_zero_params.__doc__ = (orig.__doc__ or "") + "\n\nMonkeys patched by SimpleTuner to support include_ramtorch."
    zero1.broadcast_zero_params = broadcast_zero_params  # type: ignore[attr-defined]
    return True


def apply_ramtorch_workarounds() -> bool:
    """
    Apply RamTorch monkeypatches if the installed version is missing them.

    Returns:
        True if any patch was applied, False otherwise.
    """
    patched_attach = _monkeypatch_attach_shared_params()
    patched_broadcast = _monkeypatch_broadcast_zero_params()
    return patched_attach or patched_broadcast
