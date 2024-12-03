import torch
from torch.utils.checkpoint import checkpoint as original_checkpoint


# Global variables to keep track of the checkpointing state
_checkpoint_call_count = 0
_checkpoint_interval = 4  # You can set this to any interval you prefer


def reset_checkpoint_counter():
    """Resets the checkpoint call counter. Call this at the beginning of the forward pass."""
    global _checkpoint_call_count
    _checkpoint_call_count = 0


def set_checkpoint_interval(n):
    """Sets the interval at which checkpointing is skipped."""
    global _checkpoint_interval
    _checkpoint_interval = n


def checkpoint_wrapper(function, *args, use_reentrant=True, **kwargs):
    """Wrapper function for torch.utils.checkpoint.checkpoint."""
    global _checkpoint_call_count, _checkpoint_interval
    _checkpoint_call_count += 1

    if (
        _checkpoint_interval > 0
        and (_checkpoint_call_count % _checkpoint_interval) == 0
    ):
        # Use the original checkpoint function
        return original_checkpoint(
            function, *args, use_reentrant=use_reentrant, **kwargs
        )
    else:
        # Skip checkpointing: execute the function directly
        # Do not pass 'use_reentrant' to the function
        return function(*args, **kwargs)


# Monkeypatch torch.utils.checkpoint.checkpoint
torch.utils.checkpoint.checkpoint = checkpoint_wrapper
