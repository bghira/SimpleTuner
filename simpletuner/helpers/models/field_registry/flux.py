from simpletuner.simpletuner_sdk.server.services.field_registry.types import (
    ConfigField,
    FieldDependency,
    FieldType,
    ImportanceLevel,
    ValidationRule,
    ValidationRuleType,
)


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="validation_guidance_real",
            arg_name="--validation_guidance_real",
            ui_label="Real CFG (Distilled Models)",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_guidance",
            default_value=1.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1.0, message="Must be at least 1.0")],
            help_text="CFG value for distilled models (e.g., FLUX schnell)",
            tooltip="Use 1.0 for no CFG (distilled models). Higher values for real CFG sampling.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
            model_specific=["flux", "flux2"],
        )
    )

    registry._add_field(
        ConfigField(
            name="validation_no_cfg_until_timestep",
            arg_name="--validation_no_cfg_until_timestep",
            ui_label="Skip CFG Until Timestep",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_guidance",
            default_value=2,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Skip CFG for initial timesteps (Flux only)",
            tooltip="For Flux real CFG: skip CFG on these initial timesteps. Default: 2",
            importance=ImportanceLevel.ADVANCED,
            order=3,
            model_specific=["flux", "flux2"],
        )
    )

    flux_targets = [
        "mmdit",
        "context",
        "context+ffs",
        "all",
        "all+ffs",
        "ai-toolkit",
        "tiny",
        "nano",
        "controlnet",
        "all+ffs+embedder",
        "all+ffs+embedder+controlnet",
    ]
    registry._add_field(
        ConfigField(
            name="flux_lora_target",
            arg_name="--flux_lora_target",
            ui_label="Flux LoRA Target Layers",
            field_type=FieldType.SELECT,
            tab="model",
            section="lora_config",
            subsection="model_specific",
            default_value="all",
            choices=[{"value": t, "label": t} for t in flux_targets],
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="model_family", value="flux"),
            ],
            help_text="Which layers to train in Flux models",
            tooltip="'all' trains all attention layers. 'context' only trains text layers. '+ffs' includes feed-forward layers.",
            importance=ImportanceLevel.ADVANCED,
            model_specific=["flux"],
            order=10,
        )
    )

    registry._add_field(
        ConfigField(
            name="flux_guidance_mode",
            arg_name="--flux_guidance_mode",
            ui_label="Flux Guidance Mode",
            field_type=FieldType.SELECT,
            tab="model",
            section="model_specific",
            default_value="constant",
            choices=[{"value": "constant", "label": "Constant"}, {"value": "random-range", "label": "Random Range"}],
            help_text="Guidance mode for Flux training",
            tooltip="Constant uses same guidance for all samples. Random Range varies guidance per sample.",
            importance=ImportanceLevel.ADVANCED,
            order=40,
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="flux_attention_masked_training",
            arg_name="--flux_attention_masked_training",
            ui_label="Attention Masked Training",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            default_value=False,
            model_specific=["flux"],
            help_text="Enable attention masked training for Flux models",
            tooltip="Experimental feature for Flux models that masks certain attention patterns during training.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=10,
        )
    )

    registry._add_field(
        ConfigField(
            name="flux_fast_schedule",
            arg_name="--flux_fast_schedule",
            ui_label="Fast Training Schedule",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            default_value=False,
            model_specific=["flux"],
            help_text="Use experimental fast schedule for Flux training",
            tooltip="Experimental feature that may speed up Flux.1S training at the cost of quality.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=11,
        )
    )

    registry._add_field(
        ConfigField(
            name="flux_guidance_value",
            arg_name="--flux_guidance_value",
            ui_label="Flux Guidance Value",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            default_value=1.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Guidance value for constant mode",
            tooltip="1.0 preserves CFG distillation. Higher values require CFG at inference.",
            importance=ImportanceLevel.ADVANCED,
            order=41,
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="flux_guidance_min",
            arg_name="--flux_guidance_min",
            ui_label="Flux Guidance Min",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            default_value=0.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Minimum guidance value for random-range mode",
            tooltip="Lower bound of guidance range when using random-range mode.",
            importance=ImportanceLevel.ADVANCED,
            order=42,
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="flux_guidance_max",
            arg_name="--flux_guidance_max",
            ui_label="Flux Guidance Max",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            default_value=4.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Maximum guidance value for random-range mode",
            tooltip="Upper bound of guidance range when using random-range mode.",
            importance=ImportanceLevel.ADVANCED,
            order=43,
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show")],
        )
    )

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
            name="custom_text_encoder_intermediary_layers",
            arg_name="--custom_text_encoder_intermediary_layers",
            ui_label="Custom Text Encoder Layers",
            field_type=FieldType.TEXT,
            tab="model",
            section="architecture",
            subsection="advanced",
            default_value=None,
            placeholder="[10, 20, 30]",
            dependencies=[FieldDependency(field="model_family", operator="in", values=["flux2"])],
            help_text="Override which hidden state layers to extract from the text encoder. Provide as JSON array (e.g., [10, 20, 30]). Leave blank to use model defaults.",
            tooltip="FLUX.2-dev uses layers [10, 20, 30] from Mistral-3, Klein models use [9, 18, 27] from Qwen3. Override for experimentation.",
            importance=ImportanceLevel.EXPERIMENTAL,
            model_specific=["flux2"],
            order=34,
            documentation="OPTIONS.md#--custom_text_encoder_intermediary_layers",
        )
    )
