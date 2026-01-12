import os

quantised_precision_levels = [
    "no_change",
    "int8-quanto",
    "int4-quanto",
    "int2-quanto",
    "int8-torchao",
    # SDNQ: Works on AMD, Apple, and NVIDIA
    # Full finetune recommended: uint8, uint16, fp16 (int16 also available)
    "int8-sdnq",
    "uint8-sdnq",
    "int16-sdnq",
    "uint16-sdnq",
    "fp16-sdnq",
    # LoRA-only (frozen weights): lower precision options
    "int6-sdnq",
    "int5-sdnq",
    "uint5-sdnq",
    "uint4-sdnq",
    "uint3-sdnq",
    "uint2-sdnq",
]

# Skip torch import in CLI mode for fast startup
if os.environ.get("SIMPLETUNER_SKIP_TORCH", "").lower() not in ("1", "true", "yes"):
    import torch

    if torch.cuda.is_available():
        quantised_precision_levels.extend(
            [
                "nf4-bnb",
                "int4-torchao",
                # "fp4-bnb",
                # "fp8-bnb",
                "fp8-quanto",
                "fp8uz-quanto",
            ]
        )
        primary_device = torch.cuda.get_device_properties(0)
        if primary_device.major >= 8:
            # Hopper! Or blackwell+.
            quantised_precision_levels.append("fp8-torchao")

try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image

supported_extensions = Image.registered_extensions()
image_file_extensions = set(
    ext.lower().lstrip(".") for ext, img_format in supported_extensions.items() if img_format in Image.OPEN
)

video_file_extensions = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm",
    "flv",
    "wmv",
    "m4v",
    "mpeg",
    "mpg",
    "3gp",
    "ogv",
}
audio_file_extensions = {
    "wav",
    "wave",
    "flac",
    "mp3",
    "mp4a",
    "m4a",
    "aac",
    "ogg",
    "oga",
    "opus",
    "wma",
    "aiff",
    "aif",
    "aifc",
    "alac",
}
# we combine image and video extensions as image extensions because it's a hack that is used to list all files.
image_file_extensions = image_file_extensions.union(video_file_extensions)

from simpletuner.lycoris_defaults import lycoris_defaults

# Skip diffusers overrides in CLI mode (they require torch)
if os.environ.get("SIMPLETUNER_SKIP_TORCH", "").lower() not in ("1", "true", "yes"):
    from . import diffusers_overrides  # noqa: F401  Ensures FSDP and attention patches are registered on import


def steps_remaining_in_epoch(current_step: int, steps_per_epoch: int) -> int:
    """
    Calculate the number of steps remaining in the current epoch.

    Args:
        current_step (int): The current step within the epoch.
        steps_per_epoch (int): Total number of steps in the epoch.

    Returns:
        int: Number of steps remaining in the current epoch.
    """
    remaining_steps = steps_per_epoch - (current_step % steps_per_epoch)
    return remaining_steps


def _flatten_parameters(trainable_parameters):
    """
    Yield parameters from a potentially nested collection of parameter groups.
    """
    if trainable_parameters is None:
        return

    for entry in trainable_parameters:
        if entry is None:
            continue
        if isinstance(entry, dict):
            params = entry.get("params", [])
            if not isinstance(params, (list, tuple, set)):
                params = [params]
            yield from _flatten_parameters(params)
        elif isinstance(entry, (list, tuple, set)):
            yield from _flatten_parameters(entry)
        else:
            yield entry


def trainable_parameter_count(trainable_parameters):
    """
    Convert parameter count to human-readable format.

    Args:
        num_params (int): Number of trainable parameters

    Returns:
        str: Formatted string like '1.01M', '2.34B', etc.
    """
    num_params = sum(p.numel() for p in _flatten_parameters(trainable_parameters))
    if num_params < 1000:
        return str(num_params)
    elif num_params < 1_000_000:
        return f"{num_params / 1000:.2f}K".rstrip("0").rstrip(".")
    elif num_params < 1_000_000_000:
        return f"{num_params / 1_000_000:.2f}M".rstrip("0").rstrip(".")
    elif num_params < 1_000_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B".rstrip("0").rstrip(".")
    else:
        return f"{num_params / 1_000_000_000_000:.2f}T".rstrip("0").rstrip(".")
