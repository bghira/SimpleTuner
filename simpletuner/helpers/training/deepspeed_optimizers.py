from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

SUPPORTED_OPTIMIZERS = {
    "adagrad": "Adagrad",
    "adam": "Adam",
    "adamw": "AdamW",
    "lamb": "Lamb",
    "onebitadam": "OneBitAdam",
    "onebitlamb": "OneBitLamb",
    "zerooneadam": "ZeroOneAdam",
    "muadam": "MuAdam",
    "muadamw": "MuAdamW",
    "musgd": "MuSGD",
    "lion": "Lion",
    "muon": "Muon",
}

UNSUPPORTED_OPTIMIZERS = {"cpuadam", "fusedadam"}
DEFAULT_OPTIMIZER = "AdamW"


def normalize_optimizer_name(raw_name: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return canonical DeepSpeed optimizer name and optional fallback source."""
    if raw_name is None:
        return None, None
    text = str(raw_name).strip()
    if not text:
        return None, None
    lowered = text.lower()
    if lowered in UNSUPPORTED_OPTIMIZERS:
        return DEFAULT_OPTIMIZER, text
    if lowered in SUPPORTED_OPTIMIZERS:
        return SUPPORTED_OPTIMIZERS[lowered], None
    return DEFAULT_OPTIMIZER, text


def sanitize_optimizer_block(config_section: Any) -> Tuple[Any, Optional[str]]:
    """Normalize optimizer entries inside a DeepSpeed configuration."""
    if not isinstance(config_section, dict):
        return config_section, None
    if "deepspeed_config_file" in config_section:
        return config_section, None

    fallback_source: Optional[str] = None
    optimizer_block = config_section.get("optimizer")
    if isinstance(optimizer_block, dict):
        normalized_name, fallback_source = normalize_optimizer_name(
            optimizer_block.get("type") or optimizer_block.get("name")
        )
        if normalized_name:
            optimizer_block["type"] = normalized_name
            optimizer_block.pop("name", None)
    elif isinstance(optimizer_block, str):
        normalized_name, fallback_source = normalize_optimizer_name(optimizer_block)
        if normalized_name:
            config_section["optimizer"] = {"type": normalized_name}
    return config_section, fallback_source


def sanitize_optimizer_mapping(target: Dict[str, Any]) -> Optional[str]:
    """Normalize optimizer entries across alias keys (with and without '--')."""
    if not isinstance(target, dict):
        return None

    detected_fallback: Optional[str] = None
    for variant in ("--deepspeed_config", "deepspeed_config"):
        if variant not in target:
            continue
        sanitized, fallback = sanitize_optimizer_block(target[variant])
        target[variant] = sanitized
        alias = variant.lstrip("-")
        if alias != variant:
            target[alias] = sanitized
        if fallback and not detected_fallback:
            detected_fallback = fallback
    return detected_fallback
