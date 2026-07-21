"""
CPU Linear Module

A memory-efficient linear layer implementation that keeps parameters on CPU
and transfers them to GPU on-demand using asynchronous CUDA streams.

This approach interleave compute and data transfer, making it useful for:
- Very large models that don't fit in GPU memory
- Scenarios where GPU memory is limited but CPU memory is abundant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function  # just for profiling

from simpletuner.helpers.ramtorch import profiling as ramtorch_profile

# --- Per-device global state registry ---
_DEVICE_STATE = {}


def _device_obj(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int) and torch.cuda.is_available():
        return torch.device("cuda", device)
    return torch.device(device)


def _get_device_state(device=None):
    """Get or initialize per-device state (CUDA only)."""
    if device is None:
        device = torch.cuda.current_device()
    device = _device_obj(device)

    if device.type != "cuda":
        raise RuntimeError("_get_device_state is CUDA-only")

    if device not in _DEVICE_STATE:
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                # streams & events
                "transfer_stream": torch.cuda.Stream(device=device),
                "transfer_grad_stream": torch.cuda.Stream(device=device),
                "transfer_backward_finished_event": torch.cuda.Event(),
                "transfer_weight_backward_start_event": torch.cuda.Event(),
                "transfer_weight_backward_finished_event": torch.cuda.Event(),
                "compute_backward_start_event": torch.cuda.Event(),
                "compute_backward_finished_event": torch.cuda.Event(),
                # buffers
                "w_buffers": [None, None],
                "b_buffers": [None, None],
                "forward_buffer_release_events": [None, None],
                "forward_prefetches": {},
                "w_bwd_buffers": [None, None],
                "w_grad_buffers": [None, None],
                "b_grad_buffers": [None, None],
                "w_grad_accum_buffers": [None, None],
                "b_grad_accum_buffers": [None, None],
                # clocks
                "forward_clk": 0,
                "backward_clk": 0,
            }
    return _DEVICE_STATE[device]


def _prefetch_key(weight_cpu, bias_cpu):
    return (id(weight_cpu), id(bias_cpu) if bias_cpu is not None else None)


def _tensor_version(tensor):
    return tensor._version if tensor is not None else None


def _prefetch_versions(weight_cpu, bias_cpu):
    return (_tensor_version(weight_cpu), _tensor_version(bias_cpu))


def prefetch_linear_forward(weight_cpu, bias_cpu, device="cuda"):
    """
    Enqueue the next forward weight transfer without waiting for the layer call.

    The prefetched buffers are consumed by ``BouncingLinearFn.forward`` when it
    sees the same CPU weight/bias tensors and unchanged tensor versions.
    """
    device_obj = _device_obj(device)
    if device_obj.type != "cuda":
        return False

    state = _get_device_state(device_obj)
    key = _prefetch_key(weight_cpu, bias_cpu)
    versions = _prefetch_versions(weight_cpu, bias_cpu)
    tensors = (weight_cpu, bias_cpu)
    existing = state["forward_prefetches"].get(key)
    if existing is not None and existing["versions"] == versions:
        ramtorch_profile.record_prefetch_existing(
            "linear",
            key,
            device_obj,
            ramtorch_profile.tensor_bytes(tensors),
        )
        return False

    allowed, bytes_to_prefetch, free_before, total_before = ramtorch_profile.should_prefetch(
        "linear",
        key,
        device_obj,
        tensors,
    )
    if not allowed:
        return False

    selected_buffer = state["forward_clk"]
    state["forward_clk"] ^= 1

    transfer_stream = state["transfer_stream"]
    release_event = state["forward_buffer_release_events"][selected_buffer]

    with torch.cuda.stream(transfer_stream):
        if release_event is not None:
            transfer_stream.wait_event(release_event)

        with record_function("forward_weight_bias_prefetch"):
            state["w_buffers"][selected_buffer] = weight_cpu.to(device_obj, non_blocking=True)
            state["b_buffers"][selected_buffer] = (
                bias_cpu.to(device_obj, non_blocking=True) if bias_cpu is not None else None
            )

        event = torch.cuda.Event()
        event.record()

    state["forward_prefetches"][key] = {
        "buffer": selected_buffer,
        "event": event,
        "versions": versions,
    }
    ramtorch_profile.record_prefetch_enqueued(
        "linear",
        key,
        device_obj,
        bytes_to_prefetch,
        free_before=free_before,
        total_before=total_before,
    )
    return True


def _consume_linear_forward_prefetch(weight_cpu, bias_cpu, device):
    state = _get_device_state(device)
    key = _prefetch_key(weight_cpu, bias_cpu)
    entry = state["forward_prefetches"].pop(key, None)
    if entry is None:
        return None
    if entry["versions"] != _prefetch_versions(weight_cpu, bias_cpu):
        ramtorch_profile.record_prefetch_stale("linear", key, device)
        return None
    ramtorch_profile.record_prefetch_consumed("linear", key, device)
    return entry


def _transfer_linear_forward(weight_cpu, bias_cpu, device):
    ramtorch_profile.record_fallback_forward_transfer(device, (weight_cpu, bias_cpu))
    state = _get_device_state(device)
    transfer_stream = state["transfer_stream"]
    w_buffers = state["w_buffers"]
    b_buffers = state["b_buffers"]

    selected_buffer = state["forward_clk"]
    state["forward_clk"] ^= 1
    release_event = state["forward_buffer_release_events"][selected_buffer]

    with torch.cuda.stream(transfer_stream):
        if release_event is not None:
            transfer_stream.wait_event(release_event)

        with record_function("forward_weight_bias_transfer"):
            w_buffers[selected_buffer] = weight_cpu.to(device, non_blocking=True)
            b_buffers[selected_buffer] = bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None

        event = torch.cuda.Event()
        event.record()

    return {"buffer": selected_buffer, "event": event}


def _invoke_tensor_hooks(tensor, grad):
    """
    Invoke backward hooks registered on a tensor.

    Args:
        tensor: The parameter tensor that may have hooks
        grad: The gradient tensor to pass through hooks

    Returns:
        The potentially modified gradient after all hooks
    """
    # Standard backward hooks
    if hasattr(tensor, "_ramtorch_backward_hooks") and tensor._ramtorch_backward_hooks:
        for hook_id, hook_fn in tensor._ramtorch_backward_hooks.items():
            result = hook_fn(grad)
            if result is not None:
                grad = result

    return grad


def _invoke_zero_2_tensor_hooks(tensor, grad):
    """
    Invoke backward hooks registered on a tensor.

    Args:
        tensor: The parameter tensor that may have hooks
        grad: The gradient tensor to pass through hooks

    Returns:
        The potentially modified gradient after all hooks
    """
    # Standard backward hooks
    if hasattr(tensor, "_ramtorch_zero_2_hooks") and tensor._ramtorch_zero_2_hooks:
        for hook_id, hook_fn in tensor._ramtorch_zero_2_hooks.items():
            result = hook_fn(grad)
            if result is not None:
                grad = result

    return grad


def _invoke_post_accum_tensor_hooks(tensor):
    """
    Invoke post accumulate grad hooks registered on a tensor.

    Args:
        tensor: The parameter tensor that may have hooks

    """
    # Post-accumulate grad hooks (PyTorch 2.0+)
    if hasattr(tensor, "_ramtorch_post_accumulate_grad_hooks") and tensor._ramtorch_post_accumulate_grad_hooks:
        for hook_id, hook_fn in tensor._ramtorch_post_accumulate_grad_hooks.items():
            hook_fn(tensor)


class BouncingLinearFn(torch.autograd.Function):
    """
    Custom autograd function implementing the bouncing linear operation.

    This function handles:
    1. Asynchronous transfer of weights from CPU to GPU
    2. Proper synchronization between transfer and compute streams
    3. Manual dtype handling when inside autocast context
    """

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device="cuda"):
        """
        Forward pass of bouncing linear layer with autocast support.

        Args:
            ctx: PyTorch autograd context for saving backward pass info
            x (torch.Tensor): Input tensor on GPU
            weight_cpu (torch.Tensor): Weight matrix stored on CPU
            bias_cpu (torch.Tensor, optional): Bias vector stored on CPU
            device (str): Target GPU device for computation

        Returns:
            torch.Tensor: Linear transformation output (x @ weight.T + bias)

        Flow:
            1. Select buffer using ping-pong clock (alternates between 0 and 1)
            2. On transfer stream: wait for previous compute to finish, then
               initiate async transfer of weights/bias to selected GPU buffer
            3. Record transfer completion event
            4. On compute stream: wait for transfer completion, then perform
               linear operation using the transferred weights
            5. Record compute start event for next iteration's synchronization

        The ping-pong buffering prevents race conditions by ensuring the
        transfer stream never overwrites buffers that the compute stream
        is still using.
        """
        device_obj = _device_obj(device)
        device_type = device_obj.type

        # Detect autocast state
        autocast_enabled = torch.is_autocast_enabled()
        autocast_dtype = torch.get_autocast_gpu_dtype() if autocast_enabled and device_type == "cuda" else None

        if device_type != "cuda":
            w = weight_cpu.to(device_obj)
            b = bias_cpu.to(device_obj) if bias_cpu is not None else None
            x_compute = x
            if autocast_enabled:
                x_compute = x_compute.to(autocast_dtype or x_compute.dtype)
                w = w.to(autocast_dtype or w.dtype)
                if b is not None:
                    b = b.to(autocast_dtype or b.dtype)
            out = F.linear(x_compute, w, b)
            ctx.save_for_backward(x_compute, weight_cpu, bias_cpu)
            ctx.device = device_obj
            ctx.device_type = device_type
            ctx.autocast_enabled = autocast_enabled
            ctx.autocast_dtype = autocast_dtype
            return out

        state = _get_device_state(device_obj)
        w_buffers = state["w_buffers"]
        b_buffers = state["b_buffers"]
        transfer = _consume_linear_forward_prefetch(weight_cpu, bias_cpu, device_obj)
        if transfer is None:
            transfer = _transfer_linear_forward(weight_cpu, bias_cpu, device_obj)
        selected_buffer = transfer["buffer"]

        with record_function("forward_linear_compute"):  # for profiling and easy debugging
            # make compute stream wait for this transfer
            torch.cuda.current_stream().wait_event(transfer["event"])

            # Manual casting when autocast is enabled
            if autocast_enabled:
                x_compute = x.to(autocast_dtype)
                w_compute = w_buffers[selected_buffer].to(autocast_dtype)
                b_compute = b_buffers[selected_buffer].to(autocast_dtype) if b_buffers[selected_buffer] is not None else None
                out = F.linear(x_compute, w_compute, b_compute)
            else:
                out = F.linear(x, w_buffers[selected_buffer], b_buffers[selected_buffer])
            release_event = torch.cuda.Event()
            release_event.record()
            state["forward_buffer_release_events"][selected_buffer] = release_event

        # save for backward
        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device_obj
        ctx.device_type = device_type
        ctx.autocast_enabled = autocast_enabled
        ctx.autocast_dtype = autocast_dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass for gradient computation with autocast support.

        Args:
            ctx: Autograd context containing saved forward pass data
            grad_out (torch.Tensor): Gradient w.r.t. layer output

        Returns:
            tuple: Gradients w.r.t. (input, weight, bias, device).
                   The gradients for weight and bias are returned as None
                   because they are handled manually.

        Gradient Handling Note:
            This function manually computes and accumulates gradients for the
            weight and bias parameters, which reside on the CPU. The process is:
            1. Asynchronously transfer the current gradients (if any) from
               CPU to GPU for accumulation.
            2. Compute new gradients on the GPU.
            3. Add the existing gradients to the new ones on the GPU.
            4. Invoke any registered backward hooks on the GPU gradients.
            5. Asynchronously transfer the final accumulated gradients back to
               the .grad attribute of the CPU parameters.

        Because this process bypasses the standard autograd mechanism for
        parameters, this function returns None for the weight and bias
        gradients. Hooks are manually invoked to maintain compatibility.
        """
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        device_type = ctx.device_type
        compute_weight_grad = weight_cpu.requires_grad
        compute_bias_grad = bias_cpu is not None and bias_cpu.requires_grad

        if device_type != "cuda":
            # Compute synchronously on the target device (e.g., MPS or CPU) without CUDA streams.
            w_dev = weight_cpu.to(device)
            if ctx.autocast_enabled and ctx.autocast_dtype is not None:
                grad_out_compute = grad_out.to(ctx.autocast_dtype)
                x_compute = x.to(ctx.autocast_dtype)
                w_compute = w_dev.to(ctx.autocast_dtype)
            else:
                grad_out_compute = grad_out
                x_compute = x
                w_compute = w_dev

            grad_input = grad_out_compute @ w_compute
            if compute_weight_grad:
                w_grad = grad_out_compute.flatten(0, -2).T @ x_compute.flatten(0, -2)
                w_grad = _invoke_tensor_hooks(weight_cpu, w_grad)
                if weight_cpu.grad is not None:
                    w_grad += weight_cpu.grad.to(w_grad)
                weight_cpu.ramtorch_grad = w_grad
                _invoke_post_accum_tensor_hooks(weight_cpu)
                del weight_cpu.ramtorch_grad
                w_grad = _invoke_zero_2_tensor_hooks(weight_cpu, w_grad)
                weight_cpu.grad = w_grad.to("cpu")

            if compute_bias_grad:
                reduce_dims = tuple(range(grad_out.ndim - 1))
                b_grad = grad_out.sum(dim=reduce_dims)
                b_grad = _invoke_tensor_hooks(bias_cpu, b_grad)
                if bias_cpu.grad is not None:
                    b_grad += bias_cpu.grad.to(b_grad)
                bias_cpu.ramtorch_grad = b_grad
                _invoke_post_accum_tensor_hooks(bias_cpu)
                del bias_cpu.ramtorch_grad
                b_grad = _invoke_zero_2_tensor_hooks(bias_cpu, b_grad)
                bias_cpu.grad = b_grad.to("cpu")

            return grad_input, None, None, None

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

        # get index from clock
        selected_buffer = state["backward_clk"]
        # flip the clock!
        state["backward_clk"] ^= 1

        # transfer weights on transfer stream
        with torch.cuda.stream(transfer_stream):
            with record_function("backward_weight_transfer"):  # for profiling and easy debugging
                # if it's a first time, it's a no-op
                # wait for compute event to finish first
                transfer_stream.wait_event(compute_backward_start_event)

                # alternate between buffers to prevent race condition where the transfer stream
                # overwriting the weight buffers before the main stream finish calculating the value
                w_bwd_buffers[selected_buffer] = weight_cpu.to(device, non_blocking=True)

            # load gradient for manual accumulation
            with record_function("backward_grad_accumulator_transfer"):  # for profiling and easy debugging
                w_grad_accum_buffers[selected_buffer] = (
                    weight_cpu.grad.to(device, non_blocking=True)
                    if compute_weight_grad and weight_cpu.grad is not None
                    else None
                )
                b_grad_accum_buffers[selected_buffer] = (
                    bias_cpu.grad.to(device, non_blocking=True) if compute_bias_grad and bias_cpu.grad is not None else None
                )
                # record when transfer is done
                transfer_backward_finished_event.record()

        # Make the compute stream wait for the weight transfer to complete
        torch.cuda.current_stream().wait_event(transfer_backward_finished_event)

        # mark the start of compute event
        compute_backward_start_event.record()

        with record_function("backward_linear_compute"):  # for profiling and easy debugging
            # Manual casting for backward when autocast is enabled
            if ctx.autocast_enabled:
                grad_out_compute = grad_out.to(ctx.autocast_dtype)
                x_compute = x.to(ctx.autocast_dtype)
                w_compute = w_bwd_buffers[selected_buffer].to(ctx.autocast_dtype)

                # compute input grad
                grad_input = grad_out_compute @ w_compute

                if compute_weight_grad:
                    torch.cuda.current_stream().wait_event(transfer_weight_backward_finished_event)
                    w_grad_buffers[selected_buffer] = (grad_out_compute.flatten(0, -2).T @ x_compute.flatten(0, -2)).to(
                        weight_cpu.dtype
                    )
            else:
                # compute input grad
                grad_input = grad_out @ w_bwd_buffers[selected_buffer]

                if compute_weight_grad:
                    torch.cuda.current_stream().wait_event(transfer_weight_backward_finished_event)
                    w_grad_buffers[selected_buffer] = grad_out.flatten(0, -2).T @ x.flatten(0, -2)

            # gradient accumulation is being performed directly
            if compute_weight_grad:
                with record_function("backward_weight_grad_accumulate"):  # for profiling and easy debugging

                    # invoke grad hooks on weight gradient
                    w_grad_buffers[selected_buffer] = _invoke_tensor_hooks(weight_cpu, w_grad_buffers[selected_buffer])

                    if w_grad_accum_buffers[selected_buffer] is not None:

                        # accumulate weight
                        w_grad_buffers[selected_buffer] += w_grad_accum_buffers[selected_buffer]

                    # invoke post accum hooks on weight gradient
                    # use temporary reference to simplify post accum ipl
                    weight_cpu.ramtorch_grad = w_grad_buffers[selected_buffer]
                    _invoke_post_accum_tensor_hooks(weight_cpu)
                    del weight_cpu.ramtorch_grad

            if compute_bias_grad:
                # sum over all batch-like dims, keep only last dim (Out)
                reduce_dims = tuple(range(grad_out.ndim - 1))

                # Keep bias gradient in fp32 for numerical stability when autocast is enabled
                if ctx.autocast_enabled:
                    b_grad_buffers[selected_buffer] = grad_out.float().sum(dim=reduce_dims).to(bias_cpu.dtype)
                else:
                    b_grad_buffers[selected_buffer] = grad_out.sum(dim=reduce_dims)

                with record_function("backward_bias_grad_accumulate"):  # for profiling and easy debugging

                    # invoke grad hooks on bias gradient
                    b_grad_buffers[selected_buffer] = _invoke_tensor_hooks(bias_cpu, b_grad_buffers[selected_buffer])

                    if b_grad_accum_buffers[selected_buffer] is not None:

                        # accumulate bias
                        b_grad_buffers[selected_buffer] += b_grad_accum_buffers[selected_buffer]

                    # invoke post accum hooks on bias gradient
                    # use temporary reference to simplify post accum ipl
                    bias_cpu.ramtorch_grad = b_grad_buffers[selected_buffer]
                    _invoke_post_accum_tensor_hooks(bias_cpu)
                    del bias_cpu.ramtorch_grad
            compute_backward_finished_event.record()

        with record_function("backward_grad_transfer"):  # for profiling and easy debugging
            # directly populate the grad and do grad accumulation here instead of relying on autograd
            with torch.cuda.stream(transfer_grad_stream):
                transfer_grad_stream.wait_event(compute_backward_finished_event)
                # TODO: put zero 2 hooks here and only store grad w.r.t to the assigned gpu

                if compute_weight_grad:
                    w_grad_buffers[selected_buffer] = _invoke_zero_2_tensor_hooks(
                        weight_cpu, w_grad_buffers[selected_buffer]
                    )
                if compute_bias_grad:
                    b_grad_buffers[selected_buffer] = _invoke_zero_2_tensor_hooks(bias_cpu, b_grad_buffers[selected_buffer])
                # transfer_weight_backward_start_event.record()
                if compute_weight_grad and w_grad_buffers[selected_buffer] is not None:
                    weight_cpu.grad = w_grad_buffers[selected_buffer].to("cpu", non_blocking=True)

                if compute_bias_grad:
                    if b_grad_buffers[selected_buffer] is not None:
                        bias_cpu.grad = (
                            b_grad_buffers[selected_buffer].to("cpu", non_blocking=True) if bias_cpu is not None else None
                        )

                # record when transfer is done
                transfer_weight_backward_finished_event.record()

        return grad_input, None, None, None


class CPUBouncingLinear(nn.Module):
    """
    Linear layer with CPU-stored parameters that bounce to GPU on demand.

    This module provides a drop-in replacement for nn.Linear but with different
    memory characteristics:
    - Parameters stored on CPU (using shared memory for multiprocessing)
    - Transferred to GPU only during forward/backward passes
    - Automatic cleanup after each operation

    Trade-offs:
    + Drastically reduced GPU memory usage
    + Enables training much larger models
    - Requires batching to mask the latency

    Best suited for:
    - Models too large for GPU memory
    - Inference scenarios with memory constraints

    Gradient Handling and Autograd Hooks:
        This module uses a custom autograd function that manually computes and
        accumulates gradients for the weight and bias parameters. Backward hooks
        registered on weight/bias tensors ARE supported and will be invoked on
        the GPU before transferring gradients back to CPU.

        Hooks are called in this order:
        1. Gradient computation on GPU
        2. Gradient accumulation on GPU
        3. Backward hooks invoked on GPU (tensor._ramtorch_backward_hooks)
        4. Post-accumulate hooks invoked on GPU (tensor._ramtorch_post_accumulate_grad_hooks)
        5. Transfer modified gradient back to CPU
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=None,
        device=None,
        skip_init=False,
    ):
        """
        Initialize CPU linear layer.

        Args:
            in_features (int): Input feature dimension
            out_features (int): Output feature dimension
            bias (bool): Whether to include learnable bias term
            device (str): Target GPU device for computation

        Note:
            Parameters are initialized on CPU with proper weight initialization.
            share_memory_() enables efficient sharing in multiprocessing contexts.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if device is None:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        if dtype is None:
            dtype = torch.float32

        self.is_ramtorch = True
        pin_memory = torch.cuda.is_available()
        # parameters live on CPU
        weight = torch.empty(out_features, in_features, dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.weight = nn.Parameter(weight)
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=dtype, device="cpu", pin_memory=pin_memory)) if bias else None
        )

        # create a flag to indicate if the tensors are ramtorch tensors
        self.weight.is_ramtorch = True
        if bias:
            self.bias.is_ramtorch = True

        # init
        if not skip_init:
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
            if self.bias is not None:
                fan_in = in_features
                bound = 1 / fan_in**0.5
                nn.init.uniform_(self.bias, -bound, bound)

    def _apply(self, fn):
        """
        Override _apply to allow dtype changes but prevent device moves.
        """
        # Test what fn does with a dummy tensor on CPU
        dummy = torch.tensor(0.0, device="cpu", dtype=self.weight.dtype)
        result = fn(dummy)

        # If dtype changed, apply it (but keep on CPU)
        if result.dtype != dummy.dtype:
            new_dtype = result.dtype
            self.weight.data = self.weight.data.to(dtype=new_dtype)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(dtype=new_dtype)

        # Ignore any device changes by always staying on CPU
        del dummy
        return self

    def cuda(self, device=None):
        """Override .cuda() to no-op."""
        return self

    def cpu(self):
        """Override .cpu() to no-op (already on CPU)."""
        return self

    def forward(self, x):
        """
        Forward pass through CPU linear layer.

        Args:
            x (torch.Tensor): Input tensor (should be on GPU)

        Returns:
            torch.Tensor: Linear transformation output

        Note:
            Input tensor should already be on the target GPU device.
            The autograd function handles all weight transfer logic.
        """
        return BouncingLinearFn.apply(x, self.weight, self.bias, self.device)

    def prefetch_forward(self):
        """Schedule this layer's next forward CPU-to-GPU transfer."""
        return prefetch_linear_forward(self.weight, self.bias, self.device)


Linear = CPUBouncingLinear
