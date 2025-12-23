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


def get_sdnq_presets(base_config: dict[str, Any] | None = None) -> list[AccelerationPreset]:
    """Generate standard SDNQ acceleration presets.

    SDNQ (SD.Next Quantization) works on AMD, Apple, and NVIDIA platforms.

    Args:
        base_config: Base configuration dict to merge with SDNQ settings.
                    Typically includes gradient_checkpointing, etc.

    Returns:
        List of AccelerationPreset objects for SDNQ backends.
    """
    if base_config is None:
        base_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }

    return [
        # Basic tab - SDNQ presets (works on AMD, Apple, NVIDIA)
        AccelerationPreset(
            backend=AccelerationBackend.SDNQ,
            level="int8",
            name="SDNQ - int8",
            description="8-bit quantization with stochastic rounding. Good for full finetune or LoRA.",
            tab="basic",
            tradeoff_vram="Reduces VRAM by ~50%",
            tradeoff_speed="Minimal training overhead",
            tradeoff_notes="Works on all platforms (CUDA, ROCm, MPS).",
            config={
                **base_config,
                "base_model_precision": "int8-sdnq",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.SDNQ,
            level="uint8",
            name="SDNQ - uint8",
            description="Unsigned 8-bit quantization. Slightly more aggressive than int8.",
            tab="basic",
            tradeoff_vram="Reduces VRAM by ~50%",
            tradeoff_speed="Minimal training overhead",
            tradeoff_notes="Works on all platforms (CUDA, ROCm, MPS).",
            config={
                **base_config,
                "base_model_precision": "uint8-sdnq",
            },
        ),
        # Advanced tab - SDNQ presets
        AccelerationPreset(
            backend=AccelerationBackend.SDNQ,
            level="fp16",
            name="SDNQ - fp16",
            description="16-bit float quantization. Minimal quality loss with SDNQ benefits.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~25%",
            tradeoff_speed="Minimal training overhead",
            tradeoff_notes="Best quality. Works on all platforms.",
            config={
                **base_config,
                "base_model_precision": "fp16-sdnq",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.SDNQ,
            level="uint4",
            name="SDNQ - uint4 (LoRA only)",
            description="4-bit quantization with SVD. Extreme VRAM savings for LoRA training.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~75%",
            tradeoff_speed="Moderate overhead from SVD operations",
            tradeoff_notes="LoRA training only (frozen weights). Works on all platforms.",
            config={
                **base_config,
                "base_model_precision": "uint4-sdnq",
            },
        ),
    ]
