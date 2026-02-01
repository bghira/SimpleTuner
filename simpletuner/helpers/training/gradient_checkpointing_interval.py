"""
Gradient checkpointing backend selection.

This module provides the ability to select between different gradient checkpointing
backends (torch native vs unsloth CPU offload).

Note: Per-layer interval checkpointing is implemented directly in transformer models
that support it (Flux, Chroma, SD3, Sana, AuraFlow, etc.) via their
`set_gradient_checkpointing_interval` method.
"""

_checkpoint_backend = "torch"  # "torch" or "unsloth"
_offloaded_checkpoint = None  # Lazy import

_VALID_BACKENDS = ("torch", "unsloth")


def set_checkpoint_backend(backend: str):
    """Set the gradient checkpointing backend globally."""
    global _checkpoint_backend, _offloaded_checkpoint
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Invalid checkpoint backend '{backend}'. Must be one of: {_VALID_BACKENDS}")
    _checkpoint_backend = backend
    if backend == "unsloth" and _offloaded_checkpoint is None:
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

        _offloaded_checkpoint = offloaded_checkpoint


def get_checkpoint_backend() -> str:
    """Get the current gradient checkpointing backend."""
    return _checkpoint_backend


def get_checkpoint_function():
    """Get the appropriate checkpoint function for the current backend."""
    import torch

    if _checkpoint_backend == "unsloth" and _offloaded_checkpoint is not None:
        return _offloaded_checkpoint
    return torch.utils.checkpoint.checkpoint
