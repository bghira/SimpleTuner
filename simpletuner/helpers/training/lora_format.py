from collections import Counter
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch

_LORA_DOWN_WEIGHT_SUFFIXES = (".lora.down.weight", ".lora_A.weight", ".lora_down.weight")
_LORA_UP_WEIGHT_SUFFIXES = (".lora.up.weight", ".lora_B.weight", ".lora_up.weight")
_LORA_ALPHA_SUFFIXES = (".alpha", ".lora_alpha")


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
    comfy_prefix_hits = sum(k.startswith(("diffusion_model.", "model.diffusion_model.")) for k in keys)
    comfy_alpha_hits = sum(k.endswith(".alpha") for k in keys)
    diffusers_down_up_hits = sum(".lora.down" in k or ".lora.up" in k for k in keys)

    if comfy_prefix_hits or (comfy_alpha_hits and diffusers_down_up_hits == 0):
        return PEFTLoRAFormat.COMFYUI
    return PEFTLoRAFormat.DIFFUSERS


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def _strip_prefix(key: str, prefix_to_strip: Optional[str]) -> str:
    if prefix_to_strip and key.startswith(prefix_to_strip):
        return key[len(prefix_to_strip) :]
    return key


def _most_common(values: list[Any]) -> Any:
    return Counter(values).most_common(1)[0][0]


def collect_lora_ranks(state_dict: Dict[str, Any], *, prefix_to_strip: Optional[str] = None) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    for key, value in state_dict.items():
        if not hasattr(value, "shape"):
            continue
        stripped_key = _strip_prefix(key, prefix_to_strip)
        module_key = None
        rank = None
        for suffix in _LORA_DOWN_WEIGHT_SUFFIXES:
            if stripped_key.endswith(suffix):
                module_key = stripped_key[: -len(suffix)]
                rank = int(value.shape[0])
                break
        if module_key is None:
            for suffix in _LORA_UP_WEIGHT_SUFFIXES:
                if stripped_key.endswith(suffix):
                    module_key = stripped_key[: -len(suffix)]
                    rank = int(value.shape[1])
                    break
        if module_key is None or rank is None:
            continue
        existing_rank = ranks.get(module_key)
        if existing_rank is not None and existing_rank != rank:
            raise ValueError(f"LoRA checkpoint has conflicting ranks for `{module_key}`: {existing_rank} and {rank}.")
        ranks[module_key] = rank
    return ranks


def collect_lora_alphas(state_dict: Dict[str, Any], *, prefix_to_strip: Optional[str] = None) -> Dict[str, float]:
    alphas: Dict[str, float] = {}
    for key, value in state_dict.items():
        stripped_key = _strip_prefix(key, prefix_to_strip)
        for suffix in _LORA_ALPHA_SUFFIXES:
            if stripped_key.endswith(suffix):
                alphas[stripped_key[: -len(suffix)]] = _as_float(value)
                break
    return alphas


def synthesize_missing_lora_alphas_from_ranks(
    state_dict: Dict[str, Any],
    *,
    existing_alphas: Optional[Dict[str, Any]] = None,
    prefix_to_strip: Optional[str] = None,
) -> Dict[str, float]:
    """
    For mixed-rank LoRAs with no alpha data, synthesize alpha=rank per module.
    Uniform-rank LoRAs keep the caller's configured/global alpha.
    """
    if existing_alphas:
        return {}
    if collect_lora_alphas(state_dict, prefix_to_strip=prefix_to_strip):
        return {}

    ranks = collect_lora_ranks(state_dict, prefix_to_strip=prefix_to_strip)
    if len(set(ranks.values())) <= 1:
        return {}
    return {f"{module_key}.alpha": float(rank) for module_key, rank in ranks.items()}


def peft_lora_config_kwargs_from_state_dict(
    state_dict: Dict[str, Any],
    *,
    prefix_to_strip: Optional[str] = None,
) -> Dict[str, Any]:
    ranks = collect_lora_ranks(state_dict, prefix_to_strip=prefix_to_strip)
    if not ranks:
        return {}

    default_rank = _most_common(list(ranks.values()))
    rank_pattern = {module_key: rank for module_key, rank in ranks.items() if rank != default_rank}
    alphas = collect_lora_alphas(state_dict, prefix_to_strip=prefix_to_strip)
    if alphas:
        default_alpha = _most_common(list(alphas.values()))
        alpha_pattern = {
            module_key: alpha for module_key, alpha in alphas.items() if module_key in ranks and alpha != default_alpha
        }
    else:
        default_alpha = float(default_rank)
        alpha_pattern = (
            {module_key: float(rank) for module_key, rank in ranks.items() if rank != default_rank}
            if len(set(ranks.values())) > 1
            else {}
        )

    kwargs: Dict[str, Any] = {
        "r": default_rank,
        "lora_alpha": default_alpha,
    }
    if rank_pattern:
        kwargs["rank_pattern"] = rank_pattern
    if alpha_pattern:
        kwargs["alpha_pattern"] = alpha_pattern
    return kwargs


def get_peft_kwargs(rank, network_alpha_dict=None, peft_state_dict=None, *args, **kwargs):
    from diffusers.utils import get_peft_kwargs as diffusers_get_peft_kwargs

    if peft_state_dict is not None and not network_alpha_dict:
        explicit_alphas = collect_lora_alphas(peft_state_dict)
        if explicit_alphas:
            network_alpha_dict = {f"{module_key}.alpha": alpha for module_key, alpha in explicit_alphas.items()}
        else:
            inferred_alphas = synthesize_missing_lora_alphas_from_ranks(peft_state_dict)
            if inferred_alphas:
                network_alpha_dict = inferred_alphas
    return diffusers_get_peft_kwargs(rank, network_alpha_dict, peft_state_dict, *args, **kwargs)


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
        elif ".lora.up." in new_key:
            new_key = new_key.replace(".lora.up.", ".lora_B.")

        if ".lora_A." in new_key:
            module_key = new_key[: new_key.rfind(".lora_A.")]
            alpha_value = _resolve_alpha_for_module(
                module_key.removeprefix(f"{diffusion_prefix}."), weight, adapter_metadata
            )
            if alpha_value is not None and module_key not in alpha_entries:
                alpha_entries[module_key] = torch.tensor(alpha_value, dtype=torch.float32)

        converted[new_key] = weight

    for module_key, alpha_value in alpha_entries.items():
        alpha_key = f"{module_key}.alpha"
        converted[alpha_key] = alpha_value

    return converted
