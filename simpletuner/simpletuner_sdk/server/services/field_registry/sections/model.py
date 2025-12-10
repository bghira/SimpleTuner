import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training import quantised_precision_levels

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ValidationRule, ValidationRuleType

if TYPE_CHECKING:
    from ..registry import FieldRegistry


logger = logging.getLogger(__name__)


def register_model_fields(registry: "FieldRegistry") -> None:
    """Add model configuration fields."""
    logger.debug("_add_model_config_fields called")
    # Model Family
    model_family_list = list(ModelRegistry.model_families().keys())

    def _family_label(key: str) -> str:
        model_cls = ModelRegistry.model_families().get(key)
        if model_cls is None:
            return key.replace("_", " ").title()
        return getattr(model_cls, "NAME", key.replace("_", " ").title())

    def _quant_label(value: str) -> str:
        if value == "no_change":
            return "No Change"
        if "-" in value:
            prefix, suffix = value.split("-", 1)
            prefix_label = prefix.upper()
            suffix_key = suffix.replace("_", " ")
            if suffix == "quanto":
                return f"{prefix_label} (Quanto)"
            if suffix == "bnb":
                return f"{prefix_label} (BitsAndBytes)"
            if suffix == "torchao":
                return f"{prefix_label} (TorchAO)"
            return f"{prefix_label} ({suffix_key.title()})"
        return value.replace("_", " ").title()

    quant_choice_defs = [{"value": opt, "label": _quant_label(opt)} for opt in quantised_precision_levels]
    registry._add_field(
        ConfigField(
            name="model_family",
            arg_name="--model_family",
            ui_label="Model Family",
            field_type=FieldType.SELECT,
            tab="model",
            section="model_config",
            subsection="architecture",
            choices=[{"value": f, "label": _family_label(f)} for f in ModelRegistry.model_families().keys()],
            validation_rules=[
                ValidationRule(ValidationRuleType.REQUIRED, message="Model family is required"),
                ValidationRule(ValidationRuleType.CHOICES, value=ModelRegistry.model_families().keys()),
            ],
            help_text="The overall architecture to train: SD 1.5, SDXL, Flux, Cosmos, HunyuanVideo, etc. Each family has different capabilities, requirements, and compatible tools.",
            tooltip="Different model families have different capabilities and requirements",
            importance=ImportanceLevel.ESSENTIAL,
            order=2,
            documentation="OPTIONS.md#--model_family",
        )
    )

    # Model Flavour
    registry._add_field(
        ConfigField(
            name="model_flavour",
            arg_name="--model_flavour",
            ui_label="Model Flavour",
            field_type=FieldType.SELECT,
            tab="model",
            section="model_config",
            subsection="architecture",
            default_value=None,
            choices=[],  # Dynamic based on model_family
            dependencies=[FieldDependency(field="model_family", operator="not_equals", value="")],
            help_text="Specific variant of the selected model family",
            tooltip="Some models have multiple variants with different sizes or capabilities",
            importance=ImportanceLevel.IMPORTANT,
            order=3,
            dynamic_choices=True,
        )
    )

    # ControlNet toggle
    registry._add_field(
        ConfigField(
            name="controlnet",
            arg_name="--controlnet",
            ui_label="Enable ControlNet Training",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="architecture",
            default_value=False,
            help_text="Train ControlNet (full or LoRA) branches alongside the primary network.",
            tooltip="When enabled, ControlNet datasets and conditioning tools become available in the Dataset Builder.",
            importance=ImportanceLevel.IMPORTANT,
            order=4,
        )
    )

    # Pretrained Model Path
    registry._add_field(
        ConfigField(
            name="pretrained_model_name_or_path",
            arg_name="--pretrained_model_name_or_path",
            ui_label="Base Model Path",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_config",
            subsection="advanced_paths",
            placeholder="Leave blank to use the default for the selected flavour",
            help_text="Optional override of the model checkpoint. Leave blank to use the default path for the selected model flavour.",
            tooltip="Provide a custom Hugging Face model ID or local directory. If omitted, the selected model flavour determines the path.",
            importance=ImportanceLevel.IMPORTANT,
            order=4,
            documentation="OPTIONS.md#--pretrained_model_name_or_path",
        )
    )

    # Output Directory
    registry._add_field(
        ConfigField(
            name="output_dir",
            arg_name="--output_dir",
            ui_label="Output Directory",
            field_type=FieldType.TEXT,
            tab="basic",
            section="essential_settings",
            subsection="paths",
            default_value="simpletuner-results",
            validation_rules=[ValidationRule(ValidationRuleType.REQUIRED, message="Output directory is required")],
            help_text="Directory where model checkpoints and logs will be saved",
            tooltip="All training outputs including checkpoints, logs, and samples will be saved here",
            importance=ImportanceLevel.ESSENTIAL,
            order=5,
        )
    )

    registry._add_field(
        ConfigField(
            name="configs_dir",
            arg_name="configs_dir",
            ui_label="Config Directory",
            field_type=FieldType.TEXT,
            tab="basic",
            section="essential_settings",
            subsection="paths",
            default_value="",
            help_text="Root folder SimpleTuner uses to store and load configuration files",
            tooltip="Overrides the default ~/.simpletuner/configs directory when managing saved configs.",
            importance=ImportanceLevel.IMPORTANT,
            webui_only=True,  # WebUI-specific field, not passed to trainer
            order=6,
        )
    )

    # Logging Directory
    registry._add_field(
        ConfigField(
            name="logging_dir",
            arg_name="--logging_dir",
            ui_label="Logging Directory",
            field_type=FieldType.TEXT,
            tab="basic",
            section="essential_settings",
            subsection="paths",
            default_value="./logs",
            help_text="Directory for TensorBoard and other logs",
            tooltip="Training metrics and TensorBoard logs will be saved here",
            importance=ImportanceLevel.IMPORTANT,
            order=7,
            documentation="OPTIONS.md#--logging_dir",
        )
    )

    registry._add_field(
        ConfigField(
            name="model_type",
            arg_name="--model_type",
            ui_label="Model Type",
            field_type=FieldType.SELECT,
            tab="model",
            section="model_config",
            subsection="architecture",
            default_value="full",
            choices=[
                {"value": "full", "label": "Full Model Training"},
                {"value": "lora", "label": "LoRA (Low-Rank Adaptation)"},
            ],
            validation_rules=[
                ValidationRule(ValidationRuleType.REQUIRED, message="Model type is required"),
                ValidationRule(ValidationRuleType.CHOICES, value=["full", "lora"]),
            ],
            help_text="Choose between full model training or LoRA adapter training",
            tooltip="Full training updates all model weights. LoRA only trains small adapter matrices, using less memory and producing smaller files.",
            importance=ImportanceLevel.ESSENTIAL,
            order=1,
            documentation="OPTIONS.md#--model_type",
        )
    )

    # Training Seed
    registry._add_field(
        ConfigField(
            name="seed",
            arg_name="--seed",
            ui_label="Training Seed",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="project",
            subsection="advanced",
            default_value=None,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Seed must be non-negative"),
                ValidationRule(ValidationRuleType.MAX, value=2147483647, message="Seed must fit in 32-bit integer"),
            ],
            help_text="Seed used for deterministic training behaviour",
            tooltip="Use the same seed to reproduce runs precisely. Leave blank to randomise per launch.",
            importance=ImportanceLevel.ADVANCED,
            order=9,
        )
    )

    # Resolution
    registry._add_field(
        ConfigField(
            name="resolution",
            arg_name="--resolution",
            ui_label="Training Resolution",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="training_config",
            default_value=1024,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=128, message="Resolution too low"),
                ValidationRule(ValidationRuleType.MAX, value=2048, message="Resolution too high for most GPUs"),
                ValidationRule(ValidationRuleType.DIVISIBLE_BY, value=8, message="Resolution must be divisible by 8"),
            ],
            help_text="Default image resolution (width × height) applied when a dataset entry does not specify its own.",
            tooltip="Higher resolutions require more VRAM. SD 1.5: 512, SDXL: 1024, Flux: 1024+",
            importance=ImportanceLevel.ESSENTIAL,
            order=10,
            dependencies=[
                FieldDependency(field="model_family", operator="equals", value="sd15", action="set_value", target_value=512),
                FieldDependency(
                    field="model_family",
                    operator="in",
                    values=["sdxl", "flux", "sd3"],
                    action="set_value",
                    target_value=1024,
                ),
            ],
            documentation="OPTIONS.md#--resolution",
        )
    )

    # Resume from Checkpoint
    registry._add_field(
        ConfigField(
            name="resume_from_checkpoint",
            arg_name="--resume_from_checkpoint",
            ui_label="Resume From Checkpoint",
            field_type=FieldType.SELECT,
            tab="basic",
            section="training_config",
            default_value="latest",
            choices=[
                {"value": "latest", "label": "Latest checkpoint (automatic)"},
                {"value": "", "label": "None (start from scratch)"},
            ],
            help_text="Select checkpoint to resume training from",
            tooltip="Checkpoints will be dynamically loaded based on output directory. Use 'latest' to auto-resume from the most recent checkpoint.",
            importance=ImportanceLevel.ADVANCED,
            order=11,
            dynamic_choices=True,  # Mark this field as having dynamic choices
            documentation="OPTIONS.md#--resume_from_checkpoint",
        )
    )

    # Prediction Type
    registry._add_field(
        ConfigField(
            name="prediction_type",
            arg_name="--prediction_type",
            ui_label="Prediction Type",
            field_type=FieldType.SELECT,
            tab="model",
            section="architecture",
            subsection="advanced",
            default_value=None,
            choices=[
                {"value": None, "label": "Auto-detect"},
                {"value": "epsilon", "label": "Epsilon"},
                {"value": "v_prediction", "label": "V-Prediction"},
                {"value": "sample", "label": "Sample"},
                {"value": "flow_matching", "label": "Flow Matching"},
            ],
            help_text="The parameterization type for the diffusion model",
            tooltip="Usually auto-detected from the model. Flow matching is used by Flux, SD3, and similar models.",
            importance=ImportanceLevel.ADVANCED,
            order=10,
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True)],
        )
    )

    # VAE Path
    registry._add_field(
        ConfigField(
            name="pretrained_vae_model_name_or_path",
            arg_name="--pretrained_vae_model_name_or_path",
            ui_label="Custom VAE Path",
            field_type=FieldType.TEXT,
            tab="model",
            section="vae_config",
            placeholder="madebyollin/sdxl-vae-fp16-fix",
            help_text="Optional: Override the default VAE with a custom one",
            tooltip="Can be a HuggingFace model ID or local path to a custom VAE",
            importance=ImportanceLevel.ADVANCED,
            order=11,
        )
    )

    # VAE Dtype
    registry._add_field(
        ConfigField(
            name="vae_dtype",
            arg_name="--vae_dtype",
            ui_label="VAE Precision",
            field_type=FieldType.SELECT,
            tab="model",
            section="vae_config",
            default_value="bf16",
            choices=[
                {"value": "default", "label": "Model Default"},
                {"value": "fp32", "label": "FP32"},
                {"value": "fp16", "label": "FP16"},
                {"value": "bf16", "label": "BF16"},
            ],
            help_text="Precision for VAE encoding/decoding. Lower precision saves memory.",
            tooltip="FP16/BF16 can reduce memory usage but may slightly affect image quality",
            importance=ImportanceLevel.ADVANCED,
            order=12,
        )
    )

    # VAE Cache On-Demand
    registry._add_field(
        ConfigField(
            name="vae_cache_ondemand",
            arg_name="--vae_cache_ondemand",
            ui_label="Use On-demand VAE Caching",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="vae_config",
            default_value=False,
            help_text="Process VAE latents during training instead of precomputing them",
            tooltip="Saves upfront time but slows each training step and increases memory pressure",
            importance=ImportanceLevel.ADVANCED,
            order=13,
        )
    )

    # VAE Cache Disable
    registry._add_field(
        ConfigField(
            name="vae_cache_disable",
            arg_name="--vae_cache_disable",
            ui_label="Disable VAE Cache Writing",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="vae_config",
            default_value=False,
            help_text="Implicitly enables on-demand caching and disables writing embeddings to disk.",
            tooltip="Useful for large datasets where writing embeddings to disk is impractical.",
            importance=ImportanceLevel.ADVANCED,
            order=14,
        )
    )

    # Accelerator Cache Clear Interval
    registry._add_field(
        ConfigField(
            name="accelerator_cache_clear_interval",
            arg_name="--accelerator_cache_clear_interval",
            ui_label="Accelerator Cache Clear Interval",
            field_type=FieldType.NUMBER,
            tab="training",
            section="memory_optimization",
            subsection="memory_optimization",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Interval must be at least 0")],
            help_text="Clear the cache from VRAM every X steps to prevent memory leaks (0 or blank disables)",
            tooltip="Higher values may cause memory leaks but train faster. Set 0/leave blank to disable periodic cache clears.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=14,
        )
    )

    # Aspect Bucket Rounding
    registry._add_field(
        ConfigField(
            name="aspect_bucket_rounding",
            arg_name="--aspect_bucket_rounding",
            ui_label="Aspect Bucket Rounding Precision",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="image_processing",
            subsection="advanced",
            default_value=2,
            choices=[{"value": i, "label": str(i)} for i in range(1, 10)],
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Rounding must be at least 1"),
                ValidationRule(ValidationRuleType.MAX, value=9, message="Rounding must be at most 9"),
            ],
            help_text="Number of decimal places to round aspect ratios to for bucket creation",
            tooltip="Higher precision creates more buckets but may reduce bucket efficiency. Lower values group more images together.",
            importance=ImportanceLevel.ADVANCED,
            order=15,
        )
    )

    # Base Model Precision
    registry._add_field(
        ConfigField(
            name="base_model_precision",
            arg_name="--base_model_precision",
            ui_label="Base Model Precision",
            field_type=FieldType.SELECT,
            tab="model",
            section="quantization",
            default_value="no_change",
            choices=quant_choice_defs,
            help_text="Precision for loading the base model. Lower precision saves memory.",
            tooltip="Quantization reduces memory usage but may impact quality. INT8/INT4 are commonly used.",
            importance=ImportanceLevel.ADVANCED,
            order=14,
            dependencies=[FieldDependency(field="model_type", operator="equals", value="lora", action="enable")],
            documentation="OPTIONS.md#--base_model_precision",
        )
    )

    for idx in range(1, 5):
        ui_label = "Text Encoder Precision" if idx == 1 else f"Text Encoder {idx} Precision"
        registry._add_field(
            ConfigField(
                name=f"text_encoder_{idx}_precision",
                arg_name=f"--text_encoder_{idx}_precision",
                ui_label=ui_label,
                field_type=FieldType.SELECT,
                tab="model",
                section="quantization",
                default_value="no_change",
                choices=quant_choice_defs,
                help_text="Precision for text encoders. Lower precision saves memory.",
                tooltip="Text encoder quantization has minimal impact on quality",
                importance=ImportanceLevel.ADVANCED,
                order=15 + idx - 1,
            )
        )

    # Gradient Checkpointing Interval
    registry._add_field(
        ConfigField(
            name="gradient_checkpointing_interval",
            arg_name="--gradient_checkpointing_interval",
            ui_label="Gradient Checkpointing Interval",
            field_type=FieldType.NUMBER,
            tab="model",
            section="memory_optimization",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Interval must be at least 1")],
            dependencies=[FieldDependency(field="gradient_checkpointing", operator="equals", value=True, action="enable")],
            help_text="Checkpoint every N transformer blocks (leave blank to disable)",
            tooltip="Higher values save more memory but increase compute. Clear this field to turn off partial checkpointing (supported for Flux, Sana, SDXL, SD3, and Chroma).",
            importance=ImportanceLevel.ADVANCED,
            order=16,
            documentation="OPTIONS.md#--gradient_checkpointing_interval",
        )
    )

    # Offload During Startup
    registry._add_field(
        ConfigField(
            name="offload_during_startup",
            arg_name="--offload_during_startup",
            ui_label="Offload During Startup",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="memory_optimization",
            default_value=False,
            help_text="Offload text encoders to CPU during VAE caching",
            tooltip="Useful for large models that OOM during startup. May significantly increase startup time.",
            importance=ImportanceLevel.ADVANCED,
            order=17,
            documentation="OPTIONS.md#--offload_during_startup",
        )
    )

    # Quantize Via
    registry._add_field(
        ConfigField(
            name="quantize_via",
            arg_name="--quantize_via",
            ui_label="Quantization Device",
            field_type=FieldType.SELECT,
            tab="model",
            section="quantization",
            default_value="accelerator",
            choices=[
                {"value": "cpu", "label": "CPU (Slower but safer)"},
                {"value": "accelerator", "label": "GPU/Accelerator (Faster)"},
            ],
            dependencies=[
                FieldDependency(field="model_type", operator="equals", value="lora", action="enable"),
            ],
            help_text="Where to perform model quantization",
            tooltip="CPU is safer for 24GB cards with large models. GPU is faster but may OOM.",
            importance=ImportanceLevel.ADVANCED,
            order=18,
            documentation="OPTIONS.md#--quantize_via",
        )
    )

    registry._add_field(
        ConfigField(
            name="wan_force_2_1_time_embedding",
            arg_name="--wan_force_2_1_time_embedding",
            ui_label="Force Wan 2.1 Time Embedding",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_config",
            subsection="wan_specific",
            default_value=False,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="wan", action="show")],
            help_text="Use Wan 2.1 style time embeddings even when running Wan 2.2 checkpoints.",
            tooltip="Enable this if Wan 2.2 checkpoints report shape mismatches in the time embedding layers.",
            importance=ImportanceLevel.ADVANCED,
            order=30,
        )
    )

    # Fused QKV Projections
    registry._add_field(
        ConfigField(
            name="fused_qkv_projections",
            arg_name="--fuse_qkv_projections",
            ui_label="Fused QKV Projections",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="architecture",
            default_value=False,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="flux")],
            help_text="Enables Flash Attention 3 when supported; otherwise falls back to PyTorch SDPA.",
            tooltip="Improves attention efficiency on modern NVIDIA GPUs. Uses native SDPA when Flash Attention 3 is unavailable.",
            importance=ImportanceLevel.EXPERIMENTAL,
            model_specific=["flux"],
            order=19,
            aliases=["--fused_qkv_projections"],
            documentation="OPTIONS.md#--fuse_qkv_projections",
        )
    )

    registry._add_field(
        ConfigField(
            name="rescale_betas_zero_snr",
            arg_name="--rescale_betas_zero_snr",
            ui_label="Rescale Betas Zero SNR",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="architecture",
            subsection="advanced",
            default_value=False,
            model_specific=["sd15", "sd20", "sdxl", "deepfloyd"],
            help_text="Required for V-prediction models (SD 2.x, NoobAI, Terminus XL, etc.). Modifies the noise schedule so the model can produce true blacks and whites, based on the 2023 ByteDance 'Common Diffusion Noise Schedules are Flawed' paper.",
            tooltip="Enable for V-prediction DDPM models to fix noise schedule issues that cause washed-out darks and blown-out highlights.",
            importance=ImportanceLevel.IMPORTANT,
            order=21,
        )
    )

    # Control
    registry._add_field(
        ConfigField(
            name="control",
            arg_name="--control",
            ui_label="Enable Control Training",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="architecture",
            subsection="advanced",
            default_value=False,
            disabled=True,
            help_text="Enable channel-wise control style training",
            tooltip="When enabled, requires conditioning input images alongside training data for control-style training. Currently disabled for all architectures.",
            importance=ImportanceLevel.ADVANCED,
            order=20,
        )
    )

    # ControlNet Custom Config
    registry._add_field(
        ConfigField(
            name="controlnet_custom_config",
            arg_name="--controlnet_custom_config",
            ui_label="ControlNet Custom Config",
            field_type=FieldType.TEXT,
            tab="model",
            section="architecture",
            subsection="advanced",
            default_value=None,
            placeholder='{"num_layers": 12, "num_single_layers": 6}',
            help_text="Custom configuration for ControlNet models",
            tooltip="JSON config with keys like num_layers for specific ControlNet models (e.g., HiDream).",
            importance=ImportanceLevel.IMPORTANT,
            order=21,
        )
    )

    # ControlNet Model Path
    registry._add_field(
        ConfigField(
            name="controlnet_model_name_or_path",
            arg_name="--controlnet_model_name_or_path",
            ui_label="ControlNet Model Path",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_config",
            subsection="advanced_paths",
            default_value=None,
            placeholder="path/to/controlnet",
            help_text="Path to ControlNet model weights to preload",
            tooltip="HuggingFace model ID or local path for ControlNet weights when using ControlNet training.",
            importance=ImportanceLevel.ADVANCED,
            order=22,
        )
    )

    # TREAD Config
    registry._add_field(
        ConfigField(
            name="tread_config",
            arg_name="--tread_config",
            ui_label="TREAD Configuration",
            field_type=FieldType.TEXT,
            tab="model",
            section="architecture",
            subsection="advanced",
            default_value=None,
            placeholder="path/to/tread_config.json",
            dependencies=[
                FieldDependency(
                    field="model_family",
                    operator="in",
                    values=[
                        "auraflow",
                        "cosmos2image",
                        "flux",
                        "flux2",
                        "hidream",
                        "wan",
                        "pixart",
                        "sana",
                        "sd3",
                        "kandinsky5-image",
                        "kandinsky5-video",
                    ],
                )
            ],
            help_text="Configuration for TREAD training method",
            tooltip="JSON config for TREAD method that can speed up training. Works with supported model architectures.",
            importance=ImportanceLevel.EXPERIMENTAL,
            model_specific=[
                "auraflow",
                "cosmos2image",
                "flux",
                "flux2",
                "hidream",
                "wan",
                "pixart",
                "sana",
                "sd3",
                "kandinsky5-image",
                "kandinsky5-video",
            ],
            order=23,
        )
    )

    # Pretrained Transformer Model Path
    registry._add_field(
        ConfigField(
            name="pretrained_transformer_model_name_or_path",
            arg_name="--pretrained_transformer_model_name_or_path",
            ui_label="Transformer Model Path",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_config",
            subsection="advanced_paths",
            default_value=None,
            placeholder="path/to/transformer",
            help_text="Path to pretrained transformer model",
            tooltip="HuggingFace model ID or local path for the transformer component.",
            importance=ImportanceLevel.ADVANCED,
            order=23,
        )
    )

    # Pretrained Transformer Subfolder
    registry._add_field(
        ConfigField(
            name="pretrained_transformer_subfolder",
            arg_name="--pretrained_transformer_subfolder",
            ui_label="Transformer Subfolder",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_config",
            subsection="advanced_paths",
            default_value="transformer",
            placeholder="transformer",
            help_text="Subfolder containing transformer model weights",
            tooltip="Subfolder within the model directory that contains transformer weights. Use 'none' for flat directory.",
            importance=ImportanceLevel.ADVANCED,
            order=24,
        )
    )

    # Pretrained UNet Model Path
    registry._add_field(
        ConfigField(
            name="pretrained_unet_model_name_or_path",
            arg_name="--pretrained_unet_model_name_or_path",
            ui_label="UNet Model Path",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_config",
            subsection="advanced_paths",
            default_value=None,
            placeholder="path/to/unet",
            help_text="Path to pretrained UNet model",
            tooltip="HuggingFace model ID or local path for the UNet component (used by some models).",
            importance=ImportanceLevel.ADVANCED,
            order=25,
        )
    )

    # Pretrained UNet Subfolder
    registry._add_field(
        ConfigField(
            name="pretrained_unet_subfolder",
            arg_name="--pretrained_unet_subfolder",
            ui_label="UNet Subfolder",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_config",
            subsection="advanced_paths",
            default_value="unet",
            placeholder="unet",
            help_text="Subfolder containing UNet model weights",
            tooltip="Subfolder within the model directory that contains UNet weights. Use 'none' for flat directory.",
            importance=ImportanceLevel.ADVANCED,
            order=26,
        )
    )

    # Pretrained VAE Model Path
    registry._add_field(
        ConfigField(
            name="pretrained_vae_model_name_or_path",
            arg_name="--pretrained_vae_model_name_or_path",
            ui_label="VAE Model Path",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_config",
            subsection="advanced_paths",
            default_value="madebyollin/sdxl-vae-fp16-fix",
            placeholder="path/to/vae",
            help_text="Path to pretrained VAE model",
            tooltip="HuggingFace model ID or local path for the VAE component. Default is a high-quality SDXL VAE.",
            importance=ImportanceLevel.ADVANCED,
            order=27,
        )
    )

    # Pretrained T5 Model Path
    registry._add_field(
        ConfigField(
            name="pretrained_t5_model_name_or_path",
            arg_name="--pretrained_t5_model_name_or_path",
            ui_label="T5 Model Path",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_config",
            subsection="advanced_paths",
            default_value=None,
            placeholder="path/to/t5",
            help_text="Path to pretrained T5 model",
            tooltip="HuggingFace model ID or local path for the T5 text encoder component (used by SD3/FLUX).",
            importance=ImportanceLevel.ADVANCED,
            order=28,
            documentation="OPTIONS.md#--pretrained_t5_model_name_or_path",
        )
    )

    # Revision
    registry._add_field(
        ConfigField(
            name="revision",
            arg_name="--revision",
            ui_label="Model Revision",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_config",
            subsection="advanced_paths",
            default_value=None,
            placeholder="main",
            help_text="Git branch/tag/commit for model version",
            tooltip="Specific version of the model to load from HuggingFace. Useful for reproducible training.",
            importance=ImportanceLevel.ADVANCED,
            order=29,
        )
    )

    # Variant
    registry._add_field(
        ConfigField(
            name="variant",
            arg_name="--variant",
            ui_label="Model Variant",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_config",
            subsection="advanced_paths",
            default_value=None,
            placeholder="fp16",
            help_text="Model variant (e.g., fp16, bf16)",
            tooltip="Specific variant of the model to load, such as precision variants.",
            importance=ImportanceLevel.ADVANCED,
            order=30,
        )
    )

    # Base Model Default Dtype
    registry._add_field(
        ConfigField(
            name="base_model_default_dtype",
            arg_name="--base_model_default_dtype",
            ui_label="Base Model Default Precision",
            field_type=FieldType.SELECT,
            tab="model",
            section="quantization",
            default_value="bf16",
            choices=[
                {"value": "bf16", "label": "BF16"},
                {"value": "fp32", "label": "FP32"},
            ],
            dependencies=[FieldDependency(field="model_type", value="lora")],
            help_text="Default precision for quantized base model weights",
            tooltip="Precision for non-quantized weights in quantized models. BF16 recommended for stability.",
            importance=ImportanceLevel.ADVANCED,
            order=32,
        )
    )

    # UNet Attention Slice
    registry._add_field(
        ConfigField(
            name="unet_attention_slice",
            arg_name="--unet_attention_slice",
            ui_label="UNet Attention Slicing",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="memory_optimization",
            default_value=False,
            dependencies=[
                FieldDependency(field="model_family", operator="in", values=["sd15", "sd20", "sdxl", "deepfloyd"])
            ],
            help_text="Enable attention slicing for UNet-based models",
            tooltip="Experimental feature for memory savings. May impact training quality. Only available for UNet-based architectures.",
            importance=ImportanceLevel.EXPERIMENTAL,
            model_specific=["sd15", "sd20", "sdxl", "deepfloyd"],
            order=33,
        )
    )
