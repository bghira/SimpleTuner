import logging
import os
import re
import sys
import warnings

import torch

# Force colors if requested (e.g., when running in subprocess with piped stdout)
FORCE_COLORS = os.environ.get("FORCE_COLOR", "").lower() in ("1", "true", "yes") or os.environ.get(
    "CLICOLOR_FORCE", ""
).lower() in ("1", "true", "yes")

# Check if we're in web server mode - disable colors for web responses
# FORCE_COLORS overrides DISABLE_COLORS
DISABLE_COLORS = not FORCE_COLORS and (
    os.environ.get("SIMPLETUNER_WEB_MODE", "").lower() in ("1", "true", "yes")
    or os.environ.get("SIMPLETUNER_DISABLE_COLORS", "").lower() in ("1", "true", "yes")
)

if not DISABLE_COLORS:
    try:
        from colorama import Back, Fore, Style, init

        COLORAMA_AVAILABLE = True
    except ImportError:
        COLORAMA_AVAILABLE = False
        DISABLE_COLORS = True
else:
    COLORAMA_AVAILABLE = False


# Pattern for stripping ANSI escape codes from text
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return _ANSI_ESCAPE_PATTERN.sub("", text)


class ColorizedFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        if not DISABLE_COLORS and COLORAMA_AVAILABLE:
            self.level_colors = {
                logging.DEBUG: Fore.CYAN,
                logging.INFO: Fore.GREEN,
                logging.WARNING: Fore.YELLOW,
                logging.ERROR: Fore.RED,
                logging.CRITICAL: Fore.RED + Back.WHITE + Style.BRIGHT,
            }
        else:
            self.level_colors = {}
        super().__init__(*args, **kwargs)

    def format(self, record):
        # Try to get torch rank if torch is available
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        except Exception:
            rank = 0

        if not DISABLE_COLORS and COLORAMA_AVAILABLE:
            level_color = self.level_colors.get(record.levelno, "")
            reset_color = Style.RESET_ALL
            message = super().format(record)
            return f"[RANK {rank}] {level_color}{message}{reset_color}"
        else:
            message = super().format(record)
            return f"[RANK {rank}] {message}"


# Initialize colorama only if not disabled
if not DISABLE_COLORS and COLORAMA_AVAILABLE:
    # strip=False ensures ANSI codes are preserved even when stdout is piped
    # convert=True on Windows to convert ANSI to Windows API calls
    # We always init colorama when colors are enabled, regardless of FORCE_COLORS
    init(autoreset=True, strip=False, convert=(sys.platform == "win32"))

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set lowest level to capture everything

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
default_console_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "ERROR").upper()
console_numeric_level = getattr(logging, default_console_level, logging.INFO)
console_handler.setLevel(console_numeric_level)
console_handler.setFormatter(ColorizedFormatter("%(asctime)s [%(levelname)s] %(message)s"))

# blank out the existing debug.log, if exists
if os.path.exists("debug.log"):
    with open("debug.log", "w"):
        pass


# Create a file handler with rank info in the log format
class RankFileFormatter(logging.Formatter):
    def format(self, record):
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        except Exception:
            rank = 0
        message = super().format(record)
        return f"[RANK {rank}] {message}"


file_handler = logging.FileHandler("debug.log")
file_handler.setLevel(logging.DEBUG)  # Capture debug and above
file_handler.setFormatter(RankFileFormatter("%(asctime)s [%(levelname)s] (%(name)s) %(message)s"))

_CUSTOM_HANDLERS = {id(console_handler): console_handler, id(file_handler): file_handler}


def configure_third_party_loggers(include_library_utils: bool = True) -> None:
    """
    Configure third-party library loggers to suppress verbose output.
    This needs to be called after any library initialization that might reset logger levels.

    This function is public and can be called from anywhere to ensure third-party
    loggers are properly configured after importing libraries.

    Args:
        include_library_utils: If True, also configure transformers.utils.logging and diffusers.utils.logging.
                               Set to False at module import to avoid circular imports.
    """
    # Configure transformers and diffusers logging utilities (only if requested to avoid circular imports)
    if include_library_utils:
        try:
            import transformers.utils.logging as transformers_logging

            if hasattr(transformers_logging, "set_verbosity_warning"):
                transformers_logging.set_verbosity_warning()
        except (ImportError, AttributeError):
            # transformers is optional; skip configuration if not installed or if logging API is missing
            pass

        try:
            import diffusers.utils.logging as diffusers_logging

            if hasattr(diffusers_logging, "set_verbosity_warning"):
                diffusers_logging.set_verbosity_warning()
        except (ImportError, AttributeError):
            # diffusers is optional; skip configuration if not installed or if logging API is missing
            pass

    # Configure individual loggers (safe to do at any time, no imports needed)
    forward_logger = logging.getLogger("diffusers.models.unet_2d_condition")
    forward_logger.setLevel(logging.WARNING)

    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)
    pil_logger = logging.getLogger("PIL.Image")
    pil_logger.setLevel(logging.ERROR)
    pil_logger = logging.getLogger("PIL.PngImagePlugin")
    pil_logger.setLevel(logging.ERROR)

    transformers_logger = logging.getLogger("transformers.configuration_utils")
    transformers_logger.setLevel(logging.ERROR)
    transformers_logger = logging.getLogger("transformers.processing_utils")
    transformers_logger.setLevel(logging.ERROR)
    transformers_image_processing_logger = logging.getLogger("transformers.image_processing_base")
    transformers_image_processing_logger.setLevel(logging.ERROR)
    transformers_video_processing_logger = logging.getLogger("transformers.video_processing_utils")
    transformers_video_processing_logger.setLevel(logging.ERROR)
    transformers_import_logger = logging.getLogger("transformers.utils.import_utils")
    transformers_import_logger.setLevel(logging.WARNING)
    transformers_tokenization_logger = logging.getLogger("transformers.tokenization_utils")
    transformers_tokenization_logger.setLevel(logging.ERROR)
    transformers_tokenization_base_logger = logging.getLogger("transformers.tokenization_utils_base")
    transformers_tokenization_base_logger.setLevel(logging.ERROR)
    transformers_generation_logger = logging.getLogger("transformers.generation_utils")
    transformers_generation_logger.setLevel(logging.ERROR)
    transformers_generation_config_logger = logging.getLogger("transformers.generation.configuration_utils")
    transformers_generation_config_logger.setLevel(logging.ERROR)
    transformers_modeling_logger = logging.getLogger("transformers.modeling_utils")
    transformers_modeling_logger.setLevel(logging.ERROR)

    diffusers_logger = logging.getLogger("diffusers.configuration_utils")
    diffusers_logger.setLevel(logging.ERROR)
    diffusers_utils_logger = logging.getLogger("diffusers.pipelines.pipeline_utils")
    diffusers_utils_logger.setLevel(logging.ERROR)
    diffusers_attention_logger = logging.getLogger("diffusers.models.attention_dispatch")
    diffusers_attention_logger.setLevel(logging.ERROR)  # Suppress all debug/info/warning logs
    diffusers_dynamic_modules_logger = logging.getLogger("diffusers.utils.dynamic_modules_utils")
    diffusers_dynamic_modules_logger.setLevel(logging.ERROR)
    diffusers_modeling_logger = logging.getLogger("diffusers.models.modeling_utils")
    diffusers_modeling_logger.setLevel(logging.ERROR)

    torchdistlogger = logging.getLogger("torch.distributed.nn.jit.instantiator")
    torchdistlogger.setLevel(logging.WARNING)
    torch_utils_logger = logging.getLogger("diffusers.utils.torch_utils")
    torch_utils_logger.setLevel(logging.ERROR)
    torchao_intmm_logger = logging.getLogger("torchao.kernel.intmm")
    torchao_intmm_logger.setLevel(logging.WARNING)
    torch_autograd_logger = logging.getLogger("torch.autograd")
    torch_autograd_logger.setLevel(logging.WARNING)
    torch_autograd_graph_logger = logging.getLogger("torch.autograd.graph")
    torch_autograd_graph_logger.setLevel(logging.WARNING)
    torch_inductor_logger = logging.getLogger("torch._inductor")
    torch_inductor_logger.setLevel(logging.WARNING)
    torch_inductor_template_logger = logging.getLogger("torch._inductor.template_heuristics.registry")
    torch_inductor_template_logger.setLevel(logging.WARNING)

    starlette_sse_logger = logging.getLogger("sse_starlette.sse")
    starlette_sse_logger.setLevel(logging.WARNING)
    py_multipart_logger = logging.getLogger("python_multipart.multipart")
    py_multipart_logger.setLevel(logging.WARNING)

    urllib_logger = logging.getLogger("urllib3.connectionpool")
    urllib_logger.setLevel(logging.WARNING)

    huggingface_login_logger = logging.getLogger("huggingface_hub._login")
    huggingface_login_logger.setLevel(logging.ERROR)

    # Suppress uvicorn access logs (HTTP request logs)
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.setLevel(logging.WARNING)
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.setLevel(logging.WARNING)

    # Also clean up any handlers that might have been added to these loggers
    # and ensure they don't propagate if they have their own handlers
    for logger_name in [
        "diffusers.models.unet_2d_condition",
        "diffusers.models.attention_dispatch",
        "diffusers.utils.dynamic_modules_utils",
        "diffusers.models.modeling_utils",
        "PIL",
        "PIL.Image",
        "PIL.PngImagePlugin",
        "transformers.configuration_utils",
        "transformers.processing_utils",
        "transformers.image_processing_base",
        "transformers.video_processing_utils",
        "transformers.utils.import_utils",
        "transformers.tokenization_utils",
        "transformers.tokenization_utils_base",
        "transformers.generation_utils",
        "transformers.generation.configuration_utils",
        "transformers.modeling_utils",
        "diffusers.configuration_utils",
        "diffusers.pipelines.pipeline_utils",
        "torch.distributed.nn.jit.instantiator",
        "diffusers.utils.torch_utils",
        "torchao.kernel.intmm",
        "torch.autograd",
        "torch.autograd.graph",
        "torch._inductor",
        "torch._inductor.template_heuristics.registry",
        "sse_starlette.sse",
        "python_multipart.multipart",
        "urllib3.connectionpool",
        "huggingface_hub._login",
        "uvicorn.access",
        "uvicorn.error",
    ]:
        third_party_logger = logging.getLogger(logger_name)
        # Remove any handlers added directly to this logger
        for handler in list(third_party_logger.handlers):
            third_party_logger.removeHandler(handler)
        # Ensure it propagates to root so our level filtering works
        third_party_logger.propagate = True


def ensure_custom_handlers(verbose: bool = False) -> None:
    """
    Remove any third-party handlers (Accelerate, Deepspeed, etc.) that might prepend their own formatting.
    We keep only the two handlers defined in this module and make sure they are attached exactly once.

    Args:
        verbose: Print debug information about handler cleanup
    """
    root_logger = logging.getLogger()
    existing = {id(handler): handler for handler in root_logger.handlers}

    # Debug: print handler info if verbose
    if verbose and existing:
        import sys

        print(f"[log_format] Found {len(existing)} handlers before cleanup:", file=sys.stderr)
        for handler_id, handler in existing.items():
            handler_type = type(handler).__name__
            handler_level = getattr(handler, "level", "NOTSET")
            stream = getattr(handler, "stream", None)
            stream_name = getattr(stream, "name", "unknown") if stream else "N/A"
            print(
                f"  - {handler_type} (level={handler_level}, stream={stream_name}, id={handler_id}, is_custom={handler_id in _CUSTOM_HANDLERS})",
                file=sys.stderr,
            )

    # Remove unknown handlers that would reformat our messages.
    removed_count = 0
    for handler_id, handler in list(existing.items()):
        if handler_id not in _CUSTOM_HANDLERS:
            root_logger.removeHandler(handler)
            removed_count += 1

    if verbose and removed_count > 0:
        import sys

        print(f"[log_format] Removed {removed_count} non-custom handlers", file=sys.stderr)

    # Ensure our handlers are present exactly once.
    for handler in _CUSTOM_HANDLERS.values():
        if handler not in root_logger.handlers:
            root_logger.addHandler(handler)

    # Also deduplicate by type in case handlers were added multiple times.
    seen_types = set()
    for handler in list(root_logger.handlers):
        handler_type = (type(handler), getattr(handler, "stream", None), getattr(handler, "baseFilename", None))
        if handler_type in seen_types:
            root_logger.removeHandler(handler)
        else:
            seen_types.add(handler_type)

    # Ensure root logger is at DEBUG level (for file handler) and console handler is at INFO
    root_logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    if verbose:
        import sys

        print(
            f"[log_format] After cleanup: {len(root_logger.handlers)} handlers, console level={console_handler.level}, root level={root_logger.level}",
            file=sys.stderr,
        )

    # Always configure third-party loggers (including library utils after module import)
    configure_third_party_loggers(include_library_utils=True)


# Remove existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Configure third-party loggers at module import (but skip library utils to avoid circular imports)
# The library utils (transformers.utils.logging, diffusers.utils.logging) will be configured
# when ensure_custom_handlers() is called later from train.py, trainer.py, etc.
configure_third_party_loggers(include_library_utils=False)

# Suppress specific PIL warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="PIL",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="transformers.deepspeed",
    message="transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations",
)

# Ignore torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="torch.utils._pytree",
    message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.",
)

# Suppress torch autocast warnings for unsupported dtypes
warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp.autocast_mode")

warnings.filterwarnings(
    "ignore",
)
warnings.filterwarnings(
    "ignore",
    message=".*is deprecated.*",
)
