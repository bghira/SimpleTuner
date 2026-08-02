"""Gradient checkpointing backend and segmentation helpers."""

from collections.abc import Callable, Sequence
from typing import Any

_checkpoint_backend = "torch"  # "torch", "unsloth", "torch-ffn", or "unsloth-ffn"
_offloaded_checkpoint = None  # Lazy import

_VALID_BACKENDS = ("torch", "unsloth", "torch-ffn", "unsloth-ffn")


def get_checkpoint_backend_base(backend: str | None = None) -> str:
    """Return the tensor-saving backend, without checkpoint scope suffixes."""
    selected = _checkpoint_backend if backend is None else backend
    return selected.removesuffix("-ffn")


def get_checkpoint_backend_scope(backend: str | None = None) -> str:
    """Return the checkpoint scope requested by a backend value."""
    selected = _checkpoint_backend if backend is None else backend
    if selected.endswith("-ffn"):
        return "ffn"
    return "layer"


def set_checkpoint_backend(backend: str):
    """Set the gradient checkpointing backend globally."""
    global _checkpoint_backend, _offloaded_checkpoint
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Invalid checkpoint backend '{backend}'. Must be one of: {_VALID_BACKENDS}")
    _checkpoint_backend = backend
    if get_checkpoint_backend_base(backend) == "unsloth" and _offloaded_checkpoint is None:
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

        _offloaded_checkpoint = offloaded_checkpoint


def get_checkpoint_backend() -> str:
    """Get the current gradient checkpointing backend."""
    return _checkpoint_backend


def get_checkpoint_function():
    """Get the appropriate checkpoint function for the current backend."""
    import torch

    if get_checkpoint_backend_base() == "unsloth" and _offloaded_checkpoint is not None:
        return _offloaded_checkpoint
    return torch.utils.checkpoint.checkpoint


def checkpoint_sequential_state(
    blocks: Sequence[Any],
    segment_size: int,
    state: tuple[Any, ...] | Any,
    run_block: Callable[..., tuple[Any, ...] | Any],
    checkpoint_fn: Callable[..., Any],
    checkpoint_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Checkpoint contiguous chunks of a stateful block sequence.

    Unlike ``torch.utils.checkpoint.checkpoint_sequential``, this supports block
    functions that carry multiple tensors through the sequence.
    """
    if segment_size < 1:
        raise ValueError("segment_size must be greater than 0")

    current_state = state if isinstance(state, tuple) else (state,)
    checkpoint_kwargs = dict(checkpoint_kwargs or {})

    for segment_start in range(0, len(blocks), segment_size):
        segment_blocks = tuple(blocks[segment_start : segment_start + segment_size])

        def run_segment(*segment_state, _segment_start=segment_start, _segment_blocks=segment_blocks):
            next_state = segment_state
            for offset, block in enumerate(_segment_blocks):
                result = run_block(_segment_start + offset, block, *next_state)
                next_state = result if isinstance(result, tuple) else (result,)
            if len(next_state) == 1:
                return next_state[0]
            return next_state

        result = checkpoint_fn(run_segment, *current_state, **checkpoint_kwargs)
        current_state = result if isinstance(result, tuple) else (result,)

    return current_state
