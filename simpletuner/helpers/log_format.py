import logging
import os
import warnings

import torch
from colorama import Back, Fore, Style, init


class ColorizedFormatter(logging.Formatter):
    level_colors = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Back.WHITE + Style.BRIGHT,
    }

    def format(self, record):
        # Try to get torch rank if torch is available
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        except Exception:
            rank = 0

        level_color = self.level_colors.get(record.levelno, "")
        reset_color = Style.RESET_ALL
        message = super().format(record)
        return f"[RANK {rank}] {level_color}{message}{reset_color}"


# Initialize colorama
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

# Remove existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

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
