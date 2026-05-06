from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch


class PEFTLoRAFormat(str, Enum):
    DIFFUSERS = "diffusers"
    COMFYUI = "comfyui"


def normalize_lora_format(value: Optional[Any]) -> PEFTLoRAFormat:
    """
    Coerce a user-provided LoRA format value into a PEFTLoRAFormat enum.
    Defaults to DIFFUSERS when the value is unrecognised or empty.
    """
    if isinstance(value, PEFTLoRAFormat):
        return value
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate == PEFTLoRAFormat.COMFYUI.value:
            return PEFTLoRAFormat.COMFYUI
    return PEFTLoRAFormat.DIFFUSERS


def detect_state_dict_format(state_dict: Dict[str, Any]) -> Optional[PEFTLoRAFormat]:
    """
    Heuristically detect whether a state dict looks like ComfyUI-style or Diffusers/PEFT-style.
    Returns None when no keys are present.
    """
    if not state_dict:
        return None

    keys = list(state_dict.keys())
    comfy_prefix_hits = sum(k.startswith("diffusion_model.") for k in keys)
    comfy_alpha_hits = sum(k.endswith(".alpha") for k in keys)
    comfy_ab_hits = sum(".lora_A" in k or ".lora_B" in k for k in keys)
    diffusers_down_up_hits = sum(".lora.down" in k or ".lora.up" in k for k in keys)

    if comfy_prefix_hits or (comfy_alpha_hits and diffusers_down_up_hits == 0):
        return PEFTLoRAFormat.COMFYUI
    if comfy_ab_hits and diffusers_down_up_hits == 0 and comfy_prefix_hits >= 0:
        return PEFTLoRAFormat.COMFYUI
    return PEFTLoRAFormat.DIFFUSERS


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def convert_comfyui_to_diffusers(
    state_dict: Dict[str, Any], target_prefix: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Convert a ComfyUI-style LoRA state dict (diffusion_model.* + lora_A/B + .alpha) to
    Diffusers-style keys (target_prefix.* + lora.down/up). Returns the converted
    state dict and a mapping of alpha values keyed by the converted module path + '.alpha'.
    """
    converted: Dict[str, Any] = {}
    alpha_map: Dict[str, float] = {}
    prefix = f"{target_prefix}." if target_prefix else ""

    for key, value in state_dict.items():
        stripped_key = key
        if stripped_key.startswith("diffusion_model."):
            stripped_key = stripped_key.replace("diffusion_model.", prefix, 1)
        elif prefix and stripped_key.startswith(prefix):
            # Already has the desired prefix
            pass
        elif target_prefix and not any(
            stripped_key.startswith(existing)
            for existing in ("text_encoder.", "text_encoder_2.", "controlnet.", "unet.", "transformer.")
        ):
            stripped_key = prefix + stripped_key

        if stripped_key.endswith(".alpha"):
            module_key = stripped_key[: -len(".alpha")]
            try:
                alpha_map[f"{module_key}.alpha"] = _as_float(value)
            except Exception:
                continue
            continue

        if stripped_key.endswith(".lora_A.weight"):
            stripped_key = stripped_key.replace(".lora_A.weight", ".lora.down.weight")
        elif stripped_key.endswith(".lora_B.weight"):
            stripped_key = stripped_key.replace(".lora_B.weight", ".lora.up.weight")

        converted[stripped_key] = value

    return converted, alpha_map


def _resolve_alpha_for_module(module_key: str, weight: Any, adapter_metadata: Optional[dict]) -> Optional[float]:
    rank = weight.shape[0] if hasattr(weight, "shape") and len(weight.shape) > 0 else None
    base_alpha = None
    alpha_pattern = {}
    if adapter_metadata:
        base_alpha = adapter_metadata.get("lora_alpha", None)
        alpha_pattern = adapter_metadata.get("alpha_pattern", {}) or {}

    if module_key in alpha_pattern:
        return _as_float(alpha_pattern[module_key])
    if base_alpha is not None:
        try:
            return _as_float(base_alpha)
        except Exception:
            return None
    if rank is not None:
        try:
            return float(rank)
        except Exception:
            return None
    return None


def _kohya_component_prefix(component_prefix: str, *, sdxl: bool) -> Optional[str]:
    component = component_prefix.removesuffix(".")
    if component == "unet":
        return "lora_unet"
    if component == "text_encoder":
        return "lora_te1" if sdxl else "lora_te"
    if component == "text_encoder_2":
        return "lora_te2"
    return None


def _component_metadata(
    component_prefix: str,
    adapter_metadata: Optional[dict],
    component_adapter_metadata: Optional[dict[str, dict]],
) -> Optional[dict]:
    component = component_prefix.removesuffix(".")
    if component_adapter_metadata and component in component_adapter_metadata:
        return component_adapter_metadata[component]
    return adapter_metadata


def _kohya_module_key(module_key: str, component_prefix: str, *, sdxl: bool) -> Optional[str]:
    kohya_prefix = _kohya_component_prefix(component_prefix, sdxl=sdxl)
    if kohya_prefix is None or not module_key.startswith(component_prefix):
        return None

    module_path = module_key.removeprefix(component_prefix)
    module_path = module_path.replace(".processor.", ".")
    return f"{kohya_prefix}_{module_path.replace('.', '_')}"


def convert_diffusers_to_comfyui_sd_lora(
    state_dict: Dict[str, Any],
    *,
    adapter_metadata: Optional[dict] = None,
    component_adapter_metadata: Optional[dict[str, dict]] = None,
    sdxl: bool = True,
) -> Dict[str, Any]:
    """
    Convert SD/SDXL Diffusers/PEFT LoRA keys to the Kohya-style names ComfyUI
    maps for UNet and CLIP LoRA loading.
    """
    converted: Dict[str, Any] = {}
    alpha_entries: Dict[str, torch.Tensor] = {}

    suffix_map = {
        ".lora.down.weight": ".lora_down.weight",
        ".lora.up.weight": ".lora_up.weight",
        ".lora_A.weight": ".lora_down.weight",
        ".lora_B.weight": ".lora_up.weight",
    }

    for key, weight in state_dict.items():
        component_prefix = next(
            (prefix for prefix in ("unet.", "text_encoder.", "text_encoder_2.") if key.startswith(prefix)),
            None,
        )
        if component_prefix is None:
            converted[key] = weight
            continue

        matched_suffix = next((suffix for suffix in suffix_map if key.endswith(suffix)), None)
        if matched_suffix is None:
            converted[key] = weight
            continue

        module_key = key[: -len(matched_suffix)]
        kohya_key = _kohya_module_key(module_key, component_prefix, sdxl=sdxl)
        if kohya_key is None:
            converted[key] = weight
            continue

        new_key = f"{kohya_key}{suffix_map[matched_suffix]}"
        converted[new_key] = weight

        if suffix_map[matched_suffix] == ".lora_down.weight" and kohya_key not in alpha_entries:
            metadata = _component_metadata(component_prefix, adapter_metadata, component_adapter_metadata)
            alpha_value = _resolve_alpha_for_module(
                module_key.removeprefix(component_prefix),
                weight,
                metadata,
            )
            if alpha_value is not None:
                alpha_entries[kohya_key] = torch.tensor(alpha_value, dtype=torch.float32)

    for module_key, alpha_value in alpha_entries.items():
        converted[f"{module_key}.alpha"] = alpha_value

    return converted


def convert_diffusers_to_comfyui(
    state_dict: Dict[str, Any],
    *,
    diffusion_prefix: str = "diffusion_model",
    adapter_metadata: Optional[dict] = None,
    preserve_component_prefixes: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """
    Convert a Diffusers/PEFT-style LoRA state dict to ComfyUI style with diffusion_model.* prefixes,
    lora_A/B weights, and .alpha tensors.
    """
    converted: Dict[str, Any] = {}
    alpha_entries: Dict[str, torch.Tensor] = {}
    preserve_component_prefixes = preserve_component_prefixes or set()

    for key, weight in state_dict.items():
        new_key = key
        for component_prefix in ("unet.", "transformer.", "controlnet."):
            if new_key.startswith(component_prefix):
                if component_prefix.removesuffix(".") not in preserve_component_prefixes:
                    new_key = new_key.replace(component_prefix, f"{diffusion_prefix}.", 1)
                break

        if ".lora.down." in new_key:
            new_key = new_key.replace(".lora.down.", ".lora_A.")
            module_key = new_key[: new_key.rfind(".lora_A.")]
            alpha_value = _resolve_alpha_for_module(
                module_key.removeprefix(f"{diffusion_prefix}."), weight, adapter_metadata
            )
            if alpha_value is not None and module_key not in alpha_entries:
                alpha_entries[module_key] = torch.tensor(alpha_value, dtype=torch.float32)
        elif new_key.endswith(".lora.down.weight"):
            new_key = new_key.replace(".lora.down.weight", ".lora_A.weight")
            module_key = new_key[: new_key.rfind(".lora_A.weight")]
            alpha_value = _resolve_alpha_for_module(
                module_key.removeprefix(f"{diffusion_prefix}."), weight, adapter_metadata
            )
            if alpha_value is not None and module_key not in alpha_entries:
                alpha_entries[module_key] = torch.tensor(alpha_value, dtype=torch.float32)
        elif ".lora.up." in new_key:
            new_key = new_key.replace(".lora.up.", ".lora_B.")
        elif new_key.endswith(".lora.up.weight"):
            new_key = new_key.replace(".lora.up.weight", ".lora_B.weight")

        converted[new_key] = weight

    for module_key, alpha_value in alpha_entries.items():
        alpha_key = f"{module_key}.alpha"
        converted[alpha_key] = alpha_value

    return converted
