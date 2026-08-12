"""Helpers for resolving composed distillation configurations."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import simpletuner.helpers.distillation.factory  # noqa: F401
from simpletuner.helpers.distillation.registry import DistillationRegistry
from simpletuner.helpers.distillation.requirements import (
    EMPTY_PROFILE,
    DistillerRequirementProfile,
    merge_distiller_requirement_profiles,
)


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_method(value: Any) -> Optional[str]:
    if value in (None, "", False):
        return None
    method = str(value).strip().lower().replace("-", "_")
    if method in {"none", "false", "0"}:
        return None
    return method


def _method_config(method: str, distillation_config: Any) -> Mapping[str, Any]:
    if not isinstance(distillation_config, Mapping):
        return {}
    specific = distillation_config.get(method)
    if isinstance(specific, Mapping):
        return specific
    return distillation_config


def resolve_configured_distiller_requirement_profile(config: Any) -> DistillerRequirementProfile:
    """Resolve top-level and H3 drift inner-distiller data requirements from a config object."""
    method = _normalize_method(_config_value(config, "distillation_method"))
    if method is None:
        method = _normalize_method(_config_value(config, "--distillation_method"))
    if method is None:
        return EMPTY_PROFILE

    profile = DistillationRegistry.get_requirement_profile(method)
    if method != "h3_drift":
        return profile

    h3_config = _method_config(method, _config_value(config, "distillation_config"))
    inner_method = _normalize_method(h3_config.get("inner_distillation_method"))
    if inner_method is None:
        return profile
    if inner_method == "h3_drift":
        raise ValueError("H3 drift may not wrap another h3_drift distiller.")

    inner_profile = DistillationRegistry.get_requirement_profile(inner_method)
    return merge_distiller_requirement_profiles(profile, inner_profile)
