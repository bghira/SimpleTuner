quantised_precision_levels = [
    "no_change",
    "int8-quanto",
    "int4-quanto",
    "int2-quanto",
    "int8-torchao",
]
import torch

if torch.cuda.is_available():
    quantised_precision_levels.extend(
        [
            "nf4-bnb",
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

video_file_extensions = set(["mp4", "avi", "gif", "mov", "webm"])
# we combine image and video extensions as image extensions because it's a hack that is used to list all files.
image_file_extensions = image_file_extensions.union(video_file_extensions)

lycoris_defaults = {
    "lora": {
        "algo": "lora",
        "multiplier": 1.0,
        "linear_dim": 64,
        "linear_alpha": 32,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": 16},
                "FeedForward": {"factor": 8},
            },
        },
    },
    "loha": {
        "algo": "loha",
        "multiplier": 1.0,
        "linear_dim": 32,
        "linear_alpha": 16,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": 16},
                "FeedForward": {"factor": 8},
            },
        },
    },
    "lokr": {
        "algo": "lokr",
        "multiplier": 1.0,
        "linear_dim": 10000,  # Full dimension
        "linear_alpha": 1,  # Ignored in full dimension
        "factor": 16,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": 16},
                "FeedForward": {"factor": 8},
            },
        },
    },
    "full": {
        "algo": "full",
        "multiplier": 1.0,
        "linear_dim": 1024,  # Example full matrix size
        "linear_alpha": 512,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
        },
    },
    "ia3": {
        "algo": "ia3",
        "multiplier": 1.0,
        "linear_dim": None,  # No network arguments
        "linear_alpha": None,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
        },
    },
    "dylora": {
        "algo": "dylora",
        "multiplier": 1.0,
        "linear_dim": 128,
        "linear_alpha": 64,
        "block_size": 1,  # Update one row/col per step
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": 16},
                "FeedForward": {"factor": 8},
            },
        },
    },
    "diag-oft": {
        "algo": "diag-oft",
        "multiplier": 1.0,
        "linear_dim": 64,  # Block size
        "constraint": False,
        "rescaled": False,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": 16},
                "FeedForward": {"factor": 8},
            },
        },
    },
    "boft": {
        "algo": "boft",
        "multiplier": 1.0,
        "linear_dim": 64,  # Block size
        "constraint": False,
        "rescaled": False,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": 16},
                "FeedForward": {"factor": 8},
            },
        },
    },
}


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


def trainable_parameter_count(trainable_parameters):
    """
    Convert parameter count to human-readable format.

    Args:
        num_params (int): Number of trainable parameters

    Returns:
        str: Formatted string like '1.01M', '2.34B', etc.
    """
    num_params = sum(p.numel() for p in trainable_parameters)
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
