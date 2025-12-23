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
    requires_model_type: list[str] | None = None  # If set, only show for these model types (e.g. ["full"])
    group: str | None = None  # Mutually exclusive group (e.g. "quantization" - only one can be selected)
    display_group: str | None = None  # UI grouping (e.g. "deepspeed" to show ZeRO 1/2/3 as one card)
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
            group="quantization",
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
            group="quantization",
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
            group="quantization",
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
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "uint4-sdnq",
            },
        ),
        # Advanced tab - SDNQ with quantized optimizer presets
        AccelerationPreset(
            backend=AccelerationBackend.SDNQ,
            level="int8-qopt",
            name="SDNQ - int8 + Quantized AdamW",
            description="8-bit model quantization with quantized optimizer state buffers.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~60% (model + optimizer)",
            tradeoff_speed="Minimal training overhead",
            tradeoff_notes="Optimizer states stored as uint8. Works on all platforms.",
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "int8-sdnq",
                "optimizer": "sdnq-adamw",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.SDNQ,
            level="uint8-qopt",
            name="SDNQ - uint8 + Quantized AdamW",
            description="Unsigned 8-bit model with quantized optimizer state buffers.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~60% (model + optimizer)",
            tradeoff_speed="Minimal training overhead",
            tradeoff_notes="Optimizer states stored as uint8. Works on all platforms.",
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "uint8-sdnq",
                "optimizer": "sdnq-adamw",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.SDNQ,
            level="int8-muon",
            name="SDNQ - int8 + Muon",
            description="8-bit model quantization with Muon optimizer and quantized matmul.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~65% (model + optimizer + matmul)",
            tradeoff_speed="Moderate overhead from quantized operations",
            tradeoff_notes="Uses int8 quantized matmul for Newton-Schulz iterations.",
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "int8-sdnq",
                "optimizer": "sdnq-muon+quantized_matmul",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.SDNQ,
            level="uint4-qopt",
            name="SDNQ - uint4 + Quantized AdamW (LoRA)",
            description="4-bit model with quantized optimizer. Maximum VRAM savings for LoRA.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~80% (model + optimizer)",
            tradeoff_speed="Moderate overhead from SVD and quantized ops",
            tradeoff_notes="LoRA training only. Optimizer states stored as uint8.",
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "uint4-sdnq",
                "optimizer": "sdnq-adamw",
            },
        ),
    ]


def get_torchao_presets(base_config: dict[str, Any] | None = None) -> list[AccelerationPreset]:
    """Generate TorchAO acceleration presets.

    TorchAO quantization requires NVIDIA GPUs.

    Args:
        base_config: Base configuration dict to merge with TorchAO settings.

    Returns:
        List of AccelerationPreset objects for TorchAO backends.
    """
    if base_config is None:
        base_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }

    return [
        # Basic tab - TorchAO presets
        AccelerationPreset(
            backend=AccelerationBackend.TORCHAO,
            level="int8",
            name="TorchAO - int8",
            description="8-bit integer quantization via TorchAO.",
            tab="basic",
            tradeoff_vram="Reduces VRAM by ~50%",
            tradeoff_speed="Minimal training overhead",
            tradeoff_notes="NVIDIA GPUs only.",
            requires_cuda=True,
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "int8-torchao",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.TORCHAO,
            level="fp8",
            name="TorchAO - fp8",
            description="8-bit float quantization. Good balance of quality and VRAM.",
            tab="basic",
            tradeoff_vram="Reduces VRAM by ~50%",
            tradeoff_speed="Minimal training overhead",
            tradeoff_notes="NVIDIA GPUs only (Ada/Hopper recommended).",
            requires_cuda=True,
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "fp8-torchao",
            },
        ),
        # Advanced tab - TorchAO with quantized optimizer
        AccelerationPreset(
            backend=AccelerationBackend.TORCHAO,
            level="int8-qopt",
            name="TorchAO - int8 + AdamW8bit",
            description="8-bit model with 8-bit optimizer states.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~60% (model + optimizer)",
            tradeoff_speed="Minimal training overhead",
            tradeoff_notes="NVIDIA GPUs only.",
            requires_cuda=True,
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "int8-torchao",
                "optimizer": "ao-adamw8bit",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.TORCHAO,
            level="fp8-qopt",
            name="TorchAO - fp8 + AdamW8bit",
            description="8-bit float model with 8-bit optimizer states.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~60% (model + optimizer)",
            tradeoff_speed="Minimal training overhead",
            tradeoff_notes="NVIDIA GPUs only (Ada/Hopper recommended).",
            requires_cuda=True,
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "fp8-torchao",
                "optimizer": "ao-adamw8bit",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.TORCHAO,
            level="nf4",
            name="TorchAO - nf4 (LoRA only)",
            description="4-bit NormalFloat quantization. Extreme VRAM savings for LoRA.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~75%",
            tradeoff_speed="Moderate overhead from dequantization",
            tradeoff_notes="LoRA training only (frozen weights). NVIDIA GPUs only.",
            requires_cuda=True,
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "nf4-torchao",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.TORCHAO,
            level="nf4-qopt",
            name="TorchAO - nf4 + AdamW8bit (LoRA)",
            description="4-bit model with 8-bit optimizer. Maximum VRAM savings for LoRA.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~80% (model + optimizer)",
            tradeoff_speed="Moderate overhead from dequantization",
            tradeoff_notes="LoRA training only. NVIDIA GPUs only.",
            requires_cuda=True,
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "nf4-torchao",
                "optimizer": "ao-adamw8bit",
            },
        ),
    ]


def get_quanto_presets(base_config: dict[str, Any] | None = None) -> list[AccelerationPreset]:
    """Generate Quanto acceleration presets.

    Quanto quantization works on AMD, Apple, and NVIDIA platforms.

    Args:
        base_config: Base configuration dict to merge with Quanto settings.

    Returns:
        List of AccelerationPreset objects for Quanto backends.
    """
    if base_config is None:
        base_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }

    return [
        # Basic tab - Quanto presets
        AccelerationPreset(
            backend=AccelerationBackend.QUANTO,
            level="int8",
            name="Quanto - int8",
            description="8-bit integer quantization via Optimum Quanto.",
            tab="basic",
            tradeoff_vram="Reduces VRAM by ~50%",
            tradeoff_speed="Minimal training overhead",
            tradeoff_notes="Works on all platforms (CUDA, ROCm, MPS).",
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "int8-quanto",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.QUANTO,
            level="int4",
            name="Quanto - int4 (LoRA only)",
            description="4-bit integer quantization. Good VRAM savings for LoRA.",
            tab="basic",
            tradeoff_vram="Reduces VRAM by ~70%",
            tradeoff_speed="Moderate overhead from dequantization",
            tradeoff_notes="LoRA training only. Works on all platforms.",
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "int4-quanto",
            },
        ),
        # Advanced tab - Quanto presets
        AccelerationPreset(
            backend=AccelerationBackend.QUANTO,
            level="int2",
            name="Quanto - int2 (LoRA only)",
            description="2-bit integer quantization. Extreme VRAM savings for LoRA.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~85%",
            tradeoff_speed="Significant overhead from dequantization",
            tradeoff_notes="LoRA training only. May impact quality. Works on all platforms.",
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "int2-quanto",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.QUANTO,
            level="fp8",
            name="Quanto - fp8",
            description="8-bit float quantization. Better quality than int8.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~50%",
            tradeoff_speed="Minimal training overhead",
            tradeoff_notes="Works on all platforms.",
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "fp8-quanto",
            },
        ),
    ]


def get_bitsandbytes_presets(base_config: dict[str, Any] | None = None) -> list[AccelerationPreset]:
    """Generate BitsAndBytes acceleration presets.

    BitsAndBytes quantization requires NVIDIA GPUs.

    Args:
        base_config: Base configuration dict to merge with BitsAndBytes settings.

    Returns:
        List of AccelerationPreset objects for BitsAndBytes backends.
    """
    if base_config is None:
        base_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }

    return [
        # Basic tab - BitsAndBytes presets
        AccelerationPreset(
            backend=AccelerationBackend.BITSANDBYTES,
            level="nf4",
            name="BitsAndBytes - nf4 (LoRA only)",
            description="4-bit NormalFloat quantization via BitsAndBytes.",
            tab="basic",
            tradeoff_vram="Reduces VRAM by ~75%",
            tradeoff_speed="Moderate overhead from dequantization",
            tradeoff_notes="LoRA training only (frozen weights). NVIDIA GPUs only.",
            requires_cuda=True,
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "nf4-bnb",
            },
        ),
        # Advanced tab - BitsAndBytes with quantized optimizer
        AccelerationPreset(
            backend=AccelerationBackend.BITSANDBYTES,
            level="nf4-adamw8bit",
            name="BitsAndBytes - nf4 + AdamW8bit (LoRA)",
            description="4-bit model with 8-bit AdamW optimizer.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~80% (model + optimizer)",
            tradeoff_speed="Moderate overhead from quantized ops",
            tradeoff_notes="LoRA training only. NVIDIA GPUs only.",
            requires_cuda=True,
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "nf4-bnb",
                "optimizer": "bnb-adamw8bit",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.BITSANDBYTES,
            level="nf4-adamw8bit-paged",
            name="BitsAndBytes - nf4 + Paged AdamW8bit (LoRA)",
            description="4-bit model with paged 8-bit AdamW. Offloads optimizer to CPU.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~85% (model + paged optimizer)",
            tradeoff_speed="Higher overhead from CPU paging",
            tradeoff_notes="LoRA training only. NVIDIA GPUs only.",
            requires_cuda=True,
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "nf4-bnb",
                "optimizer": "bnb-adamw8bit-paged",
            },
        ),
        AccelerationPreset(
            backend=AccelerationBackend.BITSANDBYTES,
            level="nf4-lion8bit",
            name="BitsAndBytes - nf4 + Lion8bit (LoRA)",
            description="4-bit model with 8-bit Lion optimizer. Memory efficient.",
            tab="advanced",
            tradeoff_vram="Reduces VRAM by ~82% (model + optimizer)",
            tradeoff_speed="Moderate overhead from quantized ops",
            tradeoff_notes="LoRA training only. NVIDIA GPUs only.",
            requires_cuda=True,
            group="quantization",
            config={
                **base_config,
                "base_model_precision": "nf4-bnb",
                "optimizer": "bnb-lion8bit",
            },
        ),
    ]


def get_deepspeed_presets(base_config: dict[str, Any] | None = None) -> list[AccelerationPreset]:
    """Generate DeepSpeed ZeRO acceleration presets.

    DeepSpeed requires NVIDIA GPUs and multi-GPU setups.

    Args:
        base_config: Base configuration dict to merge with DeepSpeed settings.

    Returns:
        List of AccelerationPreset objects for DeepSpeed backends.
    """
    if base_config is None:
        base_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }

    return [
        AccelerationPreset(
            backend=AccelerationBackend.DEEPSPEED_ZERO_1,
            level="zero1",
            name="DeepSpeed ZeRO Stage 1",
            description="Shards optimizer states across GPUs.",
            tab="advanced",
            tradeoff_vram="Reduces optimizer memory by ~75% per GPU",
            tradeoff_speed="Minimal overhead",
            tradeoff_notes="Multi-GPU only. Not compatible with FSDP.",
            requires_cuda=True,
            requires_model_type=["full"],
            group="quantization",
            display_group="deepspeed",
            config={**base_config, "deepspeed_config": "zero1"},
        ),
        AccelerationPreset(
            backend=AccelerationBackend.DEEPSPEED_ZERO_2,
            level="zero2",
            name="DeepSpeed ZeRO Stage 2",
            description="Shards optimizer states and gradients across GPUs.",
            tab="advanced",
            tradeoff_vram="Reduces memory by ~85% per GPU",
            tradeoff_speed="Moderate overhead from gradient sync",
            tradeoff_notes="Multi-GPU only. Not compatible with FSDP.",
            requires_cuda=True,
            requires_model_type=["full"],
            group="quantization",
            display_group="deepspeed",
            config={**base_config, "deepspeed_config": "zero2"},
        ),
        AccelerationPreset(
            backend=AccelerationBackend.DEEPSPEED_ZERO_3,
            level="zero3",
            name="DeepSpeed ZeRO Stage 3",
            description="Shards optimizer, gradients, and parameters across GPUs.",
            tab="advanced",
            tradeoff_vram="Reduces memory by ~90% per GPU",
            tradeoff_speed="Higher overhead from parameter gathering",
            tradeoff_notes="Multi-GPU only. Not compatible with FSDP. Most aggressive.",
            requires_cuda=True,
            requires_model_type=["full"],
            group="quantization",
            display_group="deepspeed",
            config={**base_config, "deepspeed_config": "zero3"},
        ),
    ]
