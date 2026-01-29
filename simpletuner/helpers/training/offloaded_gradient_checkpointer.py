"""
Unsloth-style gradient checkpointing using CPU offload.

Uses PyTorch's saved_tensors_hooks to intercept tensor saves during checkpoint,
offloading them to CPU asynchronously and restoring during backward pass.
This trades PCIe bandwidth for GPU memory savings.
"""

import torch

# Import the ORIGINAL checkpoint function before monkeypatching
# This is stored in gradient_checkpointing_interval.py
from simpletuner.helpers.training.gradient_checkpointing_interval import original_checkpoint as torch_checkpoint


class CPUOffloadHooks:
    """Context manager hooks that offload saved tensors to CPU during checkpointing."""

    def __init__(self):
        self.device = None

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        """Called when a tensor is saved for backward - offload to CPU."""
        if tensor.device.type == "cuda":
            self.device = tensor.device
            return tensor.to("cpu", non_blocking=True)
        return tensor

    def unpack(self, tensor: torch.Tensor) -> torch.Tensor:
        """Called when a tensor is needed for backward - restore to GPU."""
        if self.device is not None and tensor.device.type == "cpu":
            return tensor.to(self.device, non_blocking=True)
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
        return torch_checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)
