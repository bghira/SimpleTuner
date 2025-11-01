import logging
import os
import warnings

import torch

# Check if we're in web server mode - disable colors for web responses
DISABLE_COLORS = os.environ.get("SIMPLETUNER_WEB_MODE", "").lower() in ("1", "true", "yes") or os.environ.get(
    "SIMPLETUNER_DISABLE_COLORS", ""
).lower() in ("1", "true", "yes")

if not DISABLE_COLORS:
    try:
        from colorama import Back, Fore, Style, init

        COLORAMA_AVAILABLE = True
    except ImportError:
        COLORAMA_AVAILABLE = False
        DISABLE_COLORS = True
else:
    COLORAMA_AVAILABLE = False


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
    init(autoreset=True)

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set lowest level to capture everything

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Change to ERROR if you want to suppress INFO messages too
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


def ensure_custom_handlers() -> None:
    """
    Remove any third-party handlers (Accelerate, Deepspeed, etc.) that might prepend their own formatting.
    We keep only the two handlers defined in this module and make sure they are attached exactly once.
    """
    root_logger = logging.getLogger()
    existing = {id(handler): handler for handler in root_logger.handlers}

    # Remove unknown handlers that would reformat our messages.
    for handler_id, handler in list(existing.items()):
        if handler_id not in _CUSTOM_HANDLERS:
            root_logger.removeHandler(handler)

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


# Remove existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

ensure_custom_handlers()

forward_logger = logging.getLogger("diffusers.models.unet_2d_condition")
forward_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)
pil_logger = logging.getLogger("PIL.Image")
pil_logger.setLevel("ERROR")
pil_logger = logging.getLogger("PIL.PngImagePlugin")
pil_logger.setLevel("ERROR")
transformers_logger = logging.getLogger("transformers.configuration_utils")
transformers_logger.setLevel("ERROR")
transformers_logger = logging.getLogger("transformers.processing_utils")
transformers_logger.setLevel("ERROR")
diffusers_logger = logging.getLogger("diffusers.configuration_utils")
diffusers_logger.setLevel("ERROR")
diffusers_utils_logger = logging.getLogger("diffusers.pipelines.pipeline_utils")
diffusers_utils_logger.setLevel("ERROR")
torchdistlogger = logging.getLogger("torch.distributed.nn.jit.instantiator")
torchdistlogger.setLevel("WARNING")
torch_utils_logger = logging.getLogger("diffusers.utils.torch_utils")
torch_utils_logger.setLevel("ERROR")
starlette_sse_logger = logging.getLogger("sse_starlette.sse")
starlette_sse_logger.setLevel("WARNING")
py_multipart_logger = logging.getLogger("python_multipart.multipart")
py_multipart_logger.setLevel("WARNING")
urllib = logging.getLogger("urllib3.connectionpool")
urllib.setLevel("WARNING")

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
