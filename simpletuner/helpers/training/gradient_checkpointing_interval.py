import torch
from torch.utils.checkpoint import checkpoint as original_checkpoint

# Global variables to keep track of the checkpointing state
_checkpoint_call_count = 0
_checkpoint_interval = 4
_checkpoint_backend = "torch"  # "torch" or "unsloth"
_offloaded_checkpoint = None  # Lazy import


def reset_checkpoint_counter():
    """Resets the checkpoint call counter. Call this at the beginning of the forward pass."""
    global _checkpoint_call_count
    _checkpoint_call_count = 0


def set_checkpoint_interval(n):
    """Sets the interval at which checkpointing is skipped."""
    global _checkpoint_interval
    _checkpoint_interval = n


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


def checkpoint_wrapper(function, *args, **kwargs):
    """Wrapper function for torch.utils.checkpoint.checkpoint."""
    global _checkpoint_call_count
    _checkpoint_call_count += 1
    use_reentrant = kwargs.pop("use_reentrant", False)

    if _checkpoint_interval > 0 and (_checkpoint_call_count % _checkpoint_interval) == 0:
        # Use the configured checkpoint backend
        if _checkpoint_backend == "unsloth" and _offloaded_checkpoint is not None:
            return _offloaded_checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)
        return original_checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)
    else:
        # Skip checkpointing: execute the function directly
        # Do not pass 'use_reentrant' to the function
        return function(*args, **kwargs)


# Monkeypatch torch.utils.checkpoint.checkpoint
torch.utils.checkpoint.checkpoint = checkpoint_wrapper
