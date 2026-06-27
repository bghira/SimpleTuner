"""Replicate model profile definitions for SimpleTuner cloud training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

DEFAULT_REPLICATE_HARDWARE_PROFILE = "h100"


@dataclass(frozen=True)
class ReplicateHardwareProfile:
    """A Replicate-hosted SimpleTuner trainer profile."""

    id: str
    model: str
    label: str
    hardware_type: str


REPLICATE_HARDWARE_PROFILES: Dict[str, ReplicateHardwareProfile] = {
    "h100": ReplicateHardwareProfile(
        id="h100",
        model="simpletuner/advanced-trainer-h100",
        label="H100",
        hardware_type="H100",
    ),
    "h100-x2": ReplicateHardwareProfile(
        id="h100-x2",
        model="simpletuner/advanced-trainer-h100-x2",
        label="2x H100",
        hardware_type="2x H100",
    ),
    "h100-x4": ReplicateHardwareProfile(
        id="h100-x4",
        model="simpletuner/advanced-trainer-h100-x4",
        label="4x H100",
        hardware_type="4x H100",
    ),
    "h100-x8": ReplicateHardwareProfile(
        id="h100-x8",
        model="simpletuner/advanced-trainer-h100-x8",
        label="8x H100",
        hardware_type="8x H100",
    ),
    "l40s": ReplicateHardwareProfile(
        id="l40s",
        model="simpletuner/advanced-trainer-l40s",
        label="L40S",
        hardware_type="L40S",
    ),
    "l40s-x2": ReplicateHardwareProfile(
        id="l40s-x2",
        model="simpletuner/advanced-trainer-l40s-x2",
        label="2x L40S",
        hardware_type="2x L40S",
    ),
    "l40s-x4": ReplicateHardwareProfile(
        id="l40s-x4",
        model="simpletuner/advanced-trainer-l40s-x4",
        label="4x L40S",
        hardware_type="4x L40S",
    ),
    "l40s-x8": ReplicateHardwareProfile(
        id="l40s-x8",
        model="simpletuner/advanced-trainer-l40s-x8",
        label="8x L40S",
        hardware_type="8x L40S",
    ),
}


def list_replicate_hardware_profiles() -> List[Dict[str, Any]]:
    """Return profile metadata suitable for API/UI responses."""
    return [
        {
            "id": profile.id,
            "model": profile.model,
            "label": profile.label,
            "hardware_type": profile.hardware_type,
        }
        for profile in REPLICATE_HARDWARE_PROFILES.values()
    ]


def normalize_replicate_hardware_profile(profile: str | None) -> str:
    """Normalize and validate a Replicate hardware profile identifier."""
    value = (profile or DEFAULT_REPLICATE_HARDWARE_PROFILE).strip().lower().replace("_", "-")
    if value.startswith("advanced-trainer-"):
        value = value.removeprefix("advanced-trainer-")
    if value in {"1xh100", "h100-x1", "1x-h100"}:
        value = "h100"
    elif value in {"1xl40s", "l40s-x1", "1x-l40s"}:
        value = "l40s"

    if value not in REPLICATE_HARDWARE_PROFILES:
        valid = ", ".join(REPLICATE_HARDWARE_PROFILES)
        raise ValueError(f"Invalid Replicate hardware profile '{profile}'. Valid profiles: {valid}")
    return value


def get_replicate_hardware_profile(profile: str | None) -> ReplicateHardwareProfile:
    """Resolve a profile identifier to a profile definition."""
    return REPLICATE_HARDWARE_PROFILES[normalize_replicate_hardware_profile(profile)]
