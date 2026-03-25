# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/pipelines/anima/validation.py
# Adapted for SimpleTuner local imports.

"""Input validation helpers and compatibility constants for AnimaPipeline."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .constants import FORGE_BETA_ALPHA, FORGE_BETA_BETA
from .scheduler import AnimaFlowMatchEulerDiscreteScheduler

PromptInput = str | list[str] | tuple[str, ...]
ImageInput = Image.Image | np.ndarray | torch.Tensor
ImageBatchInput = ImageInput | list[ImageInput] | tuple[ImageInput, ...]

_IMAGE_INPUT_TYPES = (Image.Image, np.ndarray, torch.Tensor, list, tuple)
_IMAGE_BATCH_ITEM_TYPES = (Image.Image, np.ndarray, torch.Tensor)
_SUPPORTED_SAMPLERS = set(AnimaFlowMatchEulerDiscreteScheduler.SUPPORTED_SAMPLERS)
_SUPPORTED_SIGMA_SCHEDULES = set(AnimaFlowMatchEulerDiscreteScheduler.SUPPORTED_SIGMA_SCHEDULES)
_SUPPORTED_CFG_BATCH_MODES = {"split", "concat"}
_SUPPORTED_OUTPUT_TYPES = {"pil", "np", "latent"}
_DIFFUSERS_COMPAT_IGNORED_SINGLE_FILE_FROM_PRETRAINED_KEYS = {
    "custom_pipeline",
    "custom_revision",
    "from_flax",
    "load_connected_pipeline",
    "low_cpu_mem_usage",
    "max_memory",
    "mirror",
    "offload_folder",
    "offload_state_dict",
    "output_loading_info",
    "provider",
    "provider_options",
    "quantization_config",
    "sess_options",
    "subfolder",
    "use_onnx",
    "use_safetensors",
    "variant",
    "dduf_file",
}
_DIFFUSERS_COMPAT_IGNORED_FROM_SINGLE_FILE_KEYS = _DIFFUSERS_COMPAT_IGNORED_SINGLE_FILE_FROM_PRETRAINED_KEYS | {
    "config",
    "disable_mmap",
    "original_config",
    "original_config_file",
    "scaling_factor",
    "weight_name",
}
_ANIMA_COMPONENT_OVERRIDE_KEYS: set[str] = set()  # no longer user-configurable
_ANIMA_RUNTIME_OPTION_KEYS = {
    "device",
    "dtype",
    "torch_dtype",
    "text_encoder_dtype",
}
_ANIMA_REMOVED_FROM_PRETRAINED_RUNTIME_FEATURE_KEYS = {
    "enable_model_cpu_offload",
    "enable_vae_slicing",
    "enable_vae_tiling",
    "enable_vae_xformers",
}
_ANIMA_LOADER_OPTION_KEYS = {
    "local_files_only",
    "cache_dir",
    "force_download",
    "token",
    "revision",
    "proxies",
}
_ANIMA_COMPONENT_INSTANCE_KEYS = {"scheduler"}
_ANIMA_SINGLE_FILE_FROM_PRETRAINED_KEYS = (
    _ANIMA_RUNTIME_OPTION_KEYS | _ANIMA_LOADER_OPTION_KEYS | _ANIMA_COMPONENT_INSTANCE_KEYS
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_image_like_input(image: ImageBatchInput | None, *, input_name: str) -> None:
    if image is None:
        return

    if not isinstance(image, _IMAGE_INPUT_TYPES):
        raise ValueError(f"`{input_name}` must be a PIL image, numpy array, torch tensor, or a list/tuple of those types.")

    if isinstance(image, (list, tuple)):
        if len(image) == 0:
            raise ValueError(f"`{input_name}` list/tuple must not be empty.")
        if not all(isinstance(item, _IMAGE_BATCH_ITEM_TYPES) for item in image):
            raise ValueError(f"`{input_name}` list/tuple must contain PIL.Image.Image, numpy arrays, or torch tensors.")


def _validate_sampler_schedule(*, sampler: str, sigma_schedule: str) -> None:
    if sampler not in _SUPPORTED_SAMPLERS:
        raise ValueError("`sampler` must be one of: flowmatch_euler, euler, euler_a_rf, euler_ancestral_rf.")
    if sigma_schedule not in _SUPPORTED_SIGMA_SCHEDULES:
        raise ValueError("`sigma_schedule` must be one of: beta, uniform, simple, normal.")
    if sampler == "flowmatch_euler" and sigma_schedule != "uniform":
        raise ValueError("`flowmatch_euler` requires `sigma_schedule='uniform'`.")


def _validate_sampling_modes(
    *,
    sampler: str,
    sigma_schedule: str,
    cfg_batch_mode: str,
    output_type: str,
) -> None:
    _validate_sampler_schedule(sampler=sampler, sigma_schedule=sigma_schedule)
    if cfg_batch_mode not in _SUPPORTED_CFG_BATCH_MODES:
        raise ValueError("`cfg_batch_mode` must be one of: split, concat.")
    if output_type not in _SUPPORTED_OUTPUT_TYPES:
        raise ValueError("`output_type` must be one of: pil, np, latent.")


def _validate_callback_tensor_input_names(
    *,
    callback_on_step_end_tensor_inputs: list[str] | None,
    allowed_inputs: list[str],
) -> None:
    if callback_on_step_end_tensor_inputs is None:
        return

    invalid = [name for name in callback_on_step_end_tensor_inputs if name not in allowed_inputs]
    if invalid:
        raise ValueError("`callback_on_step_end_tensor_inputs` must be a subset of " f"{allowed_inputs}, but got {invalid}.")


def _warn_ignored_sampling_arguments(
    *,
    sampler: str,
    sigma_schedule: str,
    beta_alpha: float,
    beta_beta: float,
    eta: float,
    s_noise: float,
) -> None:
    ignored: list[str] = []

    if sigma_schedule != "beta":
        if not math.isclose(beta_alpha, FORGE_BETA_ALPHA):
            ignored.append("beta_alpha")
        if not math.isclose(beta_beta, FORGE_BETA_BETA):
            ignored.append("beta_beta")

    if sampler in {"flowmatch_euler", "euler"}:
        if not math.isclose(eta, 1.0):
            ignored.append("eta")
        if not math.isclose(s_noise, 1.0):
            ignored.append("s_noise")

    if ignored:
        warnings.warn(
            "Ignoring sampling arguments for "
            f"sampler='{sampler}', sigma_schedule='{sigma_schedule}': " + ", ".join(sorted(ignored)),
            stacklevel=3,
        )


def _raise_if_removed_from_pretrained_runtime_feature_kwargs(
    kwargs: dict[str, Any],
    *,
    api_name: str = "from_pretrained",
) -> None:
    removed = sorted(key for key in _ANIMA_REMOVED_FROM_PRETRAINED_RUNTIME_FEATURE_KEYS if key in kwargs)
    if len(removed) == 0:
        return
    raise ValueError(
        f"Unsupported `{api_name}` arguments: "
        + ", ".join(removed)
        + ". Configure runtime features after loading with standard pipeline methods, "
        + "for example `pipe.enable_model_cpu_offload()` and `pipe.enable_vae_slicing()`."
    )


def _pop_ignored_kwargs(
    kwargs: dict[str, Any],
    *,
    ignored_keys: set[str],
    api_name: str,
) -> None:
    ignored = sorted(key for key in kwargs if key in ignored_keys)
    for key in ignored:
        kwargs.pop(key, None)
    if ignored:
        warnings.warn(
            f"Ignoring unsupported {api_name} arguments for AnimaPipeline: {', '.join(ignored)}",
            stacklevel=3,
        )


def _partition_single_file_from_pretrained_kwargs(
    kwargs: dict[str, Any],
) -> tuple[list[str], list[str]]:
    ignored = sorted(key for key in kwargs if key in _DIFFUSERS_COMPAT_IGNORED_SINGLE_FILE_FROM_PRETRAINED_KEYS)
    unknown = sorted(key for key in kwargs if key not in _ANIMA_SINGLE_FILE_FROM_PRETRAINED_KEYS and key not in ignored)
    return ignored, unknown


def _looks_like_single_file_source(source: str) -> bool:
    if "::" in source and not source.startswith(("http://", "https://")):
        return True
    if source.startswith(("http://", "https://")):
        return True
    if source.lower().endswith(".safetensors"):
        return True
    path = Path(source)
    return path.is_file()
