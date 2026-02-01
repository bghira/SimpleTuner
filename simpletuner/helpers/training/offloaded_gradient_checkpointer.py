"""
Unsloth-style gradient checkpointing using CPU offload.

Uses PyTorch's saved_tensors_hooks to intercept tensor saves during checkpoint,
offloading them to CPU asynchronously and restoring during backward pass.
This trades PCIe bandwidth for GPU memory savings.
"""

import torch
from diffusers.utils.torch_utils import is_torch_version
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class CPUOffloadHooks:
    """Context manager hooks that offload saved tensors to CPU during checkpointing."""

    def __init__(self):
        # No global device state; device is tracked per saved tensor via pack/unpack payloads.
        pass

    def pack(self, tensor: torch.Tensor):
        """Called when a tensor is saved for backward - offload to CPU.

        Returns a payload (cpu_tensor, original_device) so that unpack() can
        restore only tensors that were actually offloaded, and to their
        correct original devices.
        """
        if tensor.device.type == "cuda":
            cpu_tensor = tensor.to("cpu", non_blocking=True)
            return cpu_tensor, tensor.device
        # Tensor is already on CPU (or a non-CUDA device); mark as not offloaded.
        return tensor, None

    def unpack(self, payload) -> torch.Tensor:
        """Called when a tensor is needed for backward - restore to original device if needed."""
        # Expect payload of the form (cpu_tensor, original_device).
        try:
            tensor, original_device = payload
        except (TypeError, ValueError):
            # Fallback: if payload is not in the expected form, return it as-is.
            return payload

        if original_device is not None and tensor.device.type == "cpu":
            return tensor.to(original_device, non_blocking=True)
        return tensor


def offloaded_checkpoint(function, *args, use_reentrant: bool = False, **kwargs):
    """
    Drop-in replacement for torch.utils.checkpoint.checkpoint using CPU offload.

    Instead of recomputing activations during backward pass (standard checkpointing),
    this offloads saved tensors to CPU asynchronously and restores them when needed.
    This approach trades PCIe bandwidth for GPU memory savings.

    Args:
        function: The forward function to checkpoint
        *args: Positional arguments to pass to function
        use_reentrant: Whether to use reentrant checkpointing (passed to torch.checkpoint)
        **kwargs: Keyword arguments to pass to torch.checkpoint

    Returns:
        Output of the function

    Note:
        This backend is most effective when PCIe bandwidth is high and can hide
        the CPU<->GPU transfer latency during forward/backward computation.
    """
    hooks = CPUOffloadHooks()
    with torch.autograd.graph.saved_tensors_hooks(hooks.pack, hooks.unpack):
        # Only pass use_reentrant on PyTorch >= 1.11.0
        if is_torch_version(">=", "1.11.0"):
            return torch_checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)
        else:
            return torch_checkpoint(function, *args, **kwargs)
