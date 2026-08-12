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


def should_checkpoint_block(
    block_index: int,
    gradient_checkpointing: bool,
    interval: int | None = None,
    segment_stride: int | None = None,
) -> bool:
    """Return whether an individual block should be checkpointed.

    ``interval=None`` means every block, matching the legacy layer checkpointing
    behavior. With a stride, checkpoint the first ``interval`` blocks in each
    stride window, e.g. interval=2 stride=4 checkpoints 0,1,4,5,...
    """
    if not gradient_checkpointing:
        return False
    if interval is None or interval <= 1:
        return True
    if segment_stride is None:
        return block_index % interval == 0
    if segment_stride < interval:
        raise ValueError("segment_stride must be at least interval")
    return block_index % segment_stride < interval


def checkpoint_sequential_state(
    blocks: Sequence[Any],
    segment_size: int,
    state: tuple[Any, ...] | Any,
    run_block: Callable[..., tuple[Any, ...] | Any],
    checkpoint_fn: Callable[..., Any],
    checkpoint_kwargs: dict[str, Any] | None = None,
    segment_stride: int | None = None,
) -> tuple[Any, ...]:
    """Checkpoint contiguous chunks of a stateful block sequence.

    Unlike ``torch.utils.checkpoint.checkpoint_sequential``, this supports block
    functions that carry multiple tensors through the sequence.
    """
    if segment_size < 1:
        raise ValueError("segment_size must be greater than 0")
    if segment_stride is None:
        segment_stride = segment_size
    if segment_stride < segment_size:
        raise ValueError("segment_stride must be at least segment_size")

    current_state = state if isinstance(state, tuple) else (state,)
    checkpoint_kwargs = dict(checkpoint_kwargs or {})

    for segment_start in range(0, len(blocks), segment_stride):
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

        gap_start = segment_start + len(segment_blocks)
        gap_end = min(segment_start + segment_stride, len(blocks))
        for block_index in range(gap_start, gap_end):
            result = run_block(block_index, blocks[block_index], *current_state)
            current_state = result if isinstance(result, tuple) else (result,)

    return current_state
