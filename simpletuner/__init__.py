"""SimpleTuner - Diffusion training made easy."""

from __future__ import annotations

import os
import warnings
from pathlib import Path as _Path


def _get_package_dir() -> _Path:
    """Return the path to the simpletuner package directory.

    Handles the case where ``__file__`` is ``None`` (e.g. namespace packages
    or certain editable installs) by falling back to ``importlib``.
    """
    if __file__ is not None:
        return _Path(__file__).parent
    import importlib.util

    spec = importlib.util.find_spec("simpletuner")
    if spec is not None and spec.origin is not None:
        return _Path(spec.origin).parent
    raise RuntimeError("Cannot determine simpletuner package directory")

# Suppress SWIG-related deprecation warnings from third-party libraries (faiss, etc.)
# These warnings are issued during import before we can install custom handlers.
warnings.filterwarnings(
    "ignore",
    message=r"builtin type Swig.*has no __module__ attribute",
    category=DeprecationWarning,
)

# Skip heavy imports in CLI mode for fast startup
if os.environ.get("SIMPLETUNER_SKIP_TORCH", "").lower() not in ("1", "true", "yes"):
    from simpletuner.helpers import logging as webhook_logging  # noqa: F401

warnings.filterwarnings(
    "ignore",
    message=r"torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly",
    category=UserWarning,
)

# Hide noisy Pydantic field attribute warnings emitted by optional integrations
try:  # pragma: no cover - depends on pydantic availability
    from pydantic.fields import UnsupportedFieldAttributeWarning
except Exception:  # pragma: no cover - pydantic may be absent
    UnsupportedFieldAttributeWarning = None

if UnsupportedFieldAttributeWarning is not None:
    warnings.filterwarnings(
        "ignore",
        message=r".*'repr' attribute.*",
        category=UnsupportedFieldAttributeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*'frozen' attribute.*",
        category=UnsupportedFieldAttributeWarning,
    )
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

# Filter out websockets deprecation warning about ws_handler second argument
warnings.filterwarnings(
    "ignore",
    message=r"remove second argument of ws_handler",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.jit\.script_method` is deprecated\.",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"TypedStorage is deprecated\.",
    category=UserWarning,
)

_original_warn = warnings.warn


def _suppress_swigvarlink(message, *args, **kwargs):
    text = str(message)
    category = kwargs.get("category", DeprecationWarning)
    if not kwargs and args:
        category = args[0]

    # Suppress all UnsupportedFieldAttributeWarning instances
    if UnsupportedFieldAttributeWarning is not None:
        try:
            if issubclass(category, UnsupportedFieldAttributeWarning):
                return None
        except TypeError:
            if category is UnsupportedFieldAttributeWarning:
                return None
        # Also check if the message is an instance of UnsupportedFieldAttributeWarning
        if isinstance(message, UnsupportedFieldAttributeWarning):
            return None
        # Check the message type directly
        if type(message).__name__ == "UnsupportedFieldAttributeWarning":
            return None

    # Check message text for pydantic field warnings
    if "attribute" in text and "Field()" in text:
        return None

    # Suppress all SWIG-related deprecation warnings from third-party libraries
    if category is DeprecationWarning and any(
        swig_type in text for swig_type in ("swigvarlink", "SwigPyPacked", "SwigPyObject")
    ):
        return None
    if "MPS autocast" in text:
        return None
    if "remove second argument of ws_handler" in text:
        return None

    return _original_warn(message, *args, **kwargs)


warnings.warn = _suppress_swigvarlink


__version__ = "4.0.5"
