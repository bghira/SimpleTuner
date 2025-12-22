"""AccelerationPreset dataclass for defining memory optimization presets."""

from dataclasses import dataclass, field
from typing import Any

from .backends import AccelerationBackend


@dataclass
class AccelerationPreset:
    """Single preset configuration for an acceleration backend.

    Models define their own presets with model-appropriate levels,
    target modules, block counts, and tradeoff descriptions.
    """

    backend: AccelerationBackend
    level: str  # e.g. "basic", "aggressive", "balanced"
    name: str  # Display name
    description: str  # What this level does
    tab: str  # "basic", "intermediate", "advanced"
    tradeoff_vram: str  # e.g. "Reduces VRAM by 40-60%"
    tradeoff_speed: str  # e.g. "Increases training time by 20-40%"
    tradeoff_notes: str  # Requirements, warnings
    requires_cuda: bool = False
    requires_min_system_ram_gb: int = 0
    config: dict[str, Any] = field(default_factory=dict)  # Config keys this preset sets
