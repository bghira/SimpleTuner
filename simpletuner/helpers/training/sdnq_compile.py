import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

_IMPORT_WARNING_EMITTED = False
_CONFIGURED_MODE: str | None = None


def _get_sdnq_compile_mode_from_args(args: Any) -> Any:
    if args is None:
        return "auto"
    try:
        args_dict = vars(args)
    except TypeError:
        return getattr(args, "sdnq_compile_mode", "auto")
    if isinstance(args_dict, dict) and "sdnq_compile_mode" not in args_dict:
        return "auto"
    return getattr(args, "sdnq_compile_mode", "auto")


def configure_sdnq_compile_mode(sdnq_compile_mode: Any = None) -> None:
    global _CONFIGURED_MODE, _IMPORT_WARNING_EMITTED

    if sdnq_compile_mode is None:
        from simpletuner.helpers.training.state_tracker import StateTracker

        args = StateTracker.get_args()
        sdnq_compile_mode = _get_sdnq_compile_mode_from_args(args)

    if sdnq_compile_mode is None:
        return

    mode = str(sdnq_compile_mode).strip().lower()
    if mode in ("", "auto", "none"):
        return
    if mode not in {"compile", "eager"}:
        raise ValueError("--sdnq_compile_mode must be one of: auto, compile, eager.")

    expected_value = "1" if mode == "compile" else "0"
    if "sdnq.common" in sys.modules:
        if _CONFIGURED_MODE == mode or os.environ.get("SDNQ_USE_TORCH_COMPILE") == expected_value:
            return
        if not _IMPORT_WARNING_EMITTED:
            logger.warning(
                "SDNQ was already imported before --sdnq_compile_mode=%s could be applied. "
                "Set SDNQ_USE_TORCH_COMPILE before process startup to force this mode.",
                mode,
            )
            _IMPORT_WARNING_EMITTED = True
        return

    os.environ["SDNQ_USE_TORCH_COMPILE"] = expected_value
    _CONFIGURED_MODE = mode
