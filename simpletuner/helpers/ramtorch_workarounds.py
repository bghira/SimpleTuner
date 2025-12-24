"""
Backfill RamTorch features that may not exist in the current PyPI release.

This module monkeypatches RamTorch to:
- Add attach_shared_ramtorch_parameters for torchrun/Accelerate launches
- Extend broadcast_zero_params with include_ramtorch support
"""

from __future__ import annotations

import inspect
from typing import Optional, Tuple

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

    broadcast_zero_params.__doc__ = (orig.__doc__ or "") + "\n\nMonkey patched by SimpleTuner to support include_ramtorch."
    zero1.broadcast_zero_params = broadcast_zero_params  # type: ignore[attr-defined]
    return True


def _monkeypatch_skip_frozen_gradients() -> bool:
    """
    Patch ramtorch's BouncingLinearFn.backward to skip gradient computation for frozen weights.

    When using ramtorch with LoRA (frozen base model), the original backward function
    still computes and stores gradients for frozen weights, wasting ~2x model size in RAM.

    This patch makes the backward function check requires_grad before computing gradients.
    """
    try:
        from ramtorch.modules.linear import (
            BouncingLinearFn,
            _get_device_state,
            _invoke_post_accum_tensor_hooks,
            _invoke_tensor_hooks,
            _invoke_zero_2_tensor_hooks,
        )
    except ImportError:
        return False

    if getattr(BouncingLinearFn.backward, "_simpletuner_frozen_grad_patch", False):
        return False

    from torch.profiler import record_function

    @staticmethod
    def _patched_backward(ctx, grad_out):
        """
        Backward pass that skips weight/bias gradient computation for frozen params.

        This is a patched version of ramtorch's BouncingLinearFn.backward that checks
        requires_grad before computing and storing gradients, significantly reducing
        RAM usage when using ramtorch with LoRA/frozen base models.
        """
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        state = _get_device_state(device)
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]
        w_bwd_buffers = state["w_bwd_buffers"]
        w_grad_buffers = state["w_grad_buffers"]
        b_grad_buffers = state["b_grad_buffers"]
        w_grad_accum_buffers = state["w_grad_accum_buffers"]
        b_grad_accum_buffers = state["b_grad_accum_buffers"]
        transfer_backward_finished_event = state["transfer_backward_finished_event"]
        transfer_weight_backward_finished_event = state["transfer_weight_backward_finished_event"]
        compute_backward_start_event = state["compute_backward_start_event"]
        compute_backward_finished_event = state["compute_backward_finished_event"]

        # Check if we need to compute weight/bias gradients
        compute_weight_grad = weight_cpu.requires_grad
        compute_bias_grad = bias_cpu is not None and bias_cpu.requires_grad

        selected_buffer = state["backward_clk"]
        state["backward_clk"] ^= 1

        # Transfer weights for input gradient computation
        with torch.cuda.stream(transfer_stream):
            with record_function("backward_weight_transfer"):
                transfer_stream.wait_event(compute_backward_start_event)
                w_bwd_buffers[selected_buffer] = weight_cpu.to(device, non_blocking=True)

            # Only transfer existing gradients if we need to accumulate
            if compute_weight_grad:
                with record_function("backward_grad_accumulator_transfer"):
                    w_grad_accum_buffers[selected_buffer] = (
                        weight_cpu.grad.to(device, non_blocking=True) if weight_cpu.grad is not None else None
                    )
                    b_grad_accum_buffers[selected_buffer] = (
                        bias_cpu.grad.to(device, non_blocking=True)
                        if bias_cpu is not None and bias_cpu.grad is not None
                        else None
                    )

            transfer_backward_finished_event.record()

        torch.cuda.current_stream().wait_event(transfer_backward_finished_event)
        compute_backward_start_event.record()

        with record_function("backward_linear_compute"):
            if ctx.autocast_enabled:
                grad_out_compute = grad_out.to(ctx.autocast_dtype)
                x_compute = x.to(ctx.autocast_dtype)
                w_compute = w_bwd_buffers[selected_buffer].to(ctx.autocast_dtype)
                grad_input = grad_out_compute @ w_compute

                if compute_weight_grad:
                    torch.cuda.current_stream().wait_event(transfer_weight_backward_finished_event)
                    w_grad_buffers[selected_buffer] = (grad_out_compute.flatten(0, -2).T @ x_compute.flatten(0, -2)).to(
                        weight_cpu.dtype
                    )
            else:
                grad_input = grad_out @ w_bwd_buffers[selected_buffer]

                if compute_weight_grad:
                    torch.cuda.current_stream().wait_event(transfer_weight_backward_finished_event)
                    w_grad_buffers[selected_buffer] = grad_out.flatten(0, -2).T @ x.flatten(0, -2)

            # Weight gradient accumulation (only if needed)
            if compute_weight_grad:
                with record_function("backward_weight_grad_accumulate"):
                    w_grad_buffers[selected_buffer] = _invoke_tensor_hooks(weight_cpu, w_grad_buffers[selected_buffer])
                    if w_grad_accum_buffers[selected_buffer] is not None:
                        w_grad_buffers[selected_buffer] += w_grad_accum_buffers[selected_buffer]
                    weight_cpu.ramtorch_grad = w_grad_buffers[selected_buffer]
                    _invoke_post_accum_tensor_hooks(weight_cpu)
                    del weight_cpu.ramtorch_grad

            # Bias gradient (only if needed)
            if compute_bias_grad:
                reduce_dims = tuple(range(grad_out.ndim - 1))
                if ctx.autocast_enabled:
                    b_grad_buffers[selected_buffer] = grad_out.float().sum(dim=reduce_dims).to(bias_cpu.dtype)
                else:
                    b_grad_buffers[selected_buffer] = grad_out.sum(dim=reduce_dims)

                with record_function("backward_bias_grad_accumulate"):
                    b_grad_buffers[selected_buffer] = _invoke_tensor_hooks(bias_cpu, b_grad_buffers[selected_buffer])
                    if b_grad_accum_buffers[selected_buffer] is not None:
                        b_grad_buffers[selected_buffer] += b_grad_accum_buffers[selected_buffer]
                    bias_cpu.ramtorch_grad = b_grad_buffers[selected_buffer]
                    _invoke_post_accum_tensor_hooks(bias_cpu)
                    del bias_cpu.ramtorch_grad

            compute_backward_finished_event.record()

        # Transfer gradients back to CPU (or just record event for synchronization)
        with torch.cuda.stream(transfer_grad_stream):
            transfer_grad_stream.wait_event(compute_backward_finished_event)

            if compute_weight_grad and w_grad_buffers[selected_buffer] is not None:
                with record_function("backward_grad_transfer"):
                    w_grad_buffers[selected_buffer] = _invoke_zero_2_tensor_hooks(
                        weight_cpu, w_grad_buffers[selected_buffer]
                    )
                    weight_cpu.grad = w_grad_buffers[selected_buffer].to("cpu", non_blocking=True)

            if compute_bias_grad and b_grad_buffers[selected_buffer] is not None:
                with record_function("backward_grad_transfer"):
                    b_grad_buffers[selected_buffer] = _invoke_zero_2_tensor_hooks(bias_cpu, b_grad_buffers[selected_buffer])
                    bias_cpu.grad = b_grad_buffers[selected_buffer].to("cpu", non_blocking=True)

            # Always record the event for proper synchronization with subsequent layers
            transfer_weight_backward_finished_event.record()

        return grad_input, None, None, None

    _patched_backward._simpletuner_frozen_grad_patch = True
    BouncingLinearFn.backward = _patched_backward
    return True


def apply_ramtorch_workarounds() -> bool:
    """
    Apply RamTorch monkeypatches if the installed version is missing them.

    Returns:
        True if any patch was applied, False otherwise.
    """
    patched_attach = _monkeypatch_attach_shared_params()
    patched_broadcast = _monkeypatch_broadcast_zero_params()
    patched_frozen_grad = _monkeypatch_skip_frozen_gradients()
    return patched_attach or patched_broadcast or patched_frozen_grad
