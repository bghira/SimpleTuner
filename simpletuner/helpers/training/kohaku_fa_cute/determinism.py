"""Honor the live PyTorch policy before experimental atomic backward calls."""

import warnings

import torch


def check_atomic_backward():
    if not torch.are_deterministic_algorithms_enabled():
        return
    message = (
        "simpletuner_cute_attention::atomic_backward uses unordered FP32 dQ atomics and has "
        "no deterministic implementation. Enable deterministic algorithms "
        "before forward through the public API to select its supported fallback."
    )
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(message, UserWarning, stacklevel=3)
    else:
        raise RuntimeError(message)
