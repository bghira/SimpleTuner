"""SimpleTuner - Diffusion training made easy."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly",
    category=UserWarning,
)

_original_warn = warnings.warn


def _suppress_swigvarlink(message, *args, **kwargs):
    text = str(message)
    category = kwargs.get("category", DeprecationWarning)
    if "swigvarlink" in text and category is DeprecationWarning:
        return None
    if "MPS autocast" in text:
        return None
    return _original_warn(message, *args, **kwargs)


warnings.warn = _suppress_swigvarlink

__version__ = "2.2.1"
