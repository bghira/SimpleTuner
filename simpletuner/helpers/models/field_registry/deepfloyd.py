from simpletuner.simpletuner_sdk.server.services.field_registry.types import (
    ConfigField,
    FieldType,
    ImportanceLevel,
    ValidationRule,
    ValidationRuleType,
)


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="deepfloyd_validation_pipeline_mode",
            arg_name="--deepfloyd_validation_pipeline_mode",
            ui_label="DeepFloyd Validation Pipeline",
            field_type=FieldType.SELECT,
            tab="validation",
            section="validation_options",
            default_value="auto",
            choices=[
                {"value": "auto", "label": "Auto"},
                {"value": "trained-stage", "label": "Trained Stage Only"},
                {"value": "full-pipeline", "label": "Full Pipeline"},
            ],
            help_text="Choose whether DeepFloyd validation runs only the trained stage or chains fixed peer stages.",
            tooltip="Auto uses the full DeepFloyd pipeline for prompt validation and the trained stage for dataset image validation.",
            importance=ImportanceLevel.ADVANCED,
            order=21,
            subsection="advanced",
            model_specific=["deepfloyd"],
            documentation="OPTIONS.md#--deepfloyd_validation_pipeline_mode",
        )
    )

    registry._add_field(
        ConfigField(
            name="deepfloyd_validation_stage1_model",
            arg_name="--deepfloyd_validation_stage1_model",
            ui_label="DeepFloyd Stage I Model",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_options",
            default_value=None,
            placeholder="DeepFloyd/IF-I-XL-v1.0",
            help_text="Fixed DeepFloyd stage I model used when validating a trained stage II model through the full pipeline.",
            tooltip="Leave blank to use DeepFloyd/IF-I-XL-v1.0.",
            importance=ImportanceLevel.ADVANCED,
            order=22,
            subsection="advanced",
            model_specific=["deepfloyd"],
            documentation="OPTIONS.md#--deepfloyd_validation_stage1_model",
        )
    )

    registry._add_field(
        ConfigField(
            name="deepfloyd_validation_stage2_model",
            arg_name="--deepfloyd_validation_stage2_model",
            ui_label="DeepFloyd Stage II Model",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_options",
            default_value=None,
            placeholder="DeepFloyd/IF-II-M-v1.0",
            help_text="Fixed DeepFloyd stage II model used when validating a trained stage I model through the full pipeline.",
            tooltip="Leave blank to use DeepFloyd/IF-II-M-v1.0.",
            importance=ImportanceLevel.ADVANCED,
            order=23,
            subsection="advanced",
            model_specific=["deepfloyd"],
            documentation="OPTIONS.md#--deepfloyd_validation_stage2_model",
        )
    )

    registry._add_field(
        ConfigField(
            name="deepfloyd_validation_stage3_mode",
            arg_name="--deepfloyd_validation_stage3_mode",
            ui_label="DeepFloyd Stage III Mode",
            field_type=FieldType.SELECT,
            tab="validation",
            section="validation_options",
            default_value="none",
            choices=[
                {"value": "none", "label": "None"},
                {"value": "sd-x4-upscaler", "label": "Stable Diffusion x4 Upscaler"},
            ],
            help_text="Optional terminal DeepFloyd validation upscaler after stage II.",
            tooltip="Stage III is not a released DeepFloyd model; this option can use the era-compatible SD x4 upscaler.",
            importance=ImportanceLevel.ADVANCED,
            order=24,
            subsection="advanced",
            model_specific=["deepfloyd"],
            documentation="OPTIONS.md#--deepfloyd_validation_stage3_mode",
        )
    )

    registry._add_field(
        ConfigField(
            name="deepfloyd_validation_stage3_model",
            arg_name="--deepfloyd_validation_stage3_model",
            ui_label="DeepFloyd Stage III Model",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_options",
            default_value=None,
            placeholder="stabilityai/stable-diffusion-x4-upscaler",
            help_text="Model repository used when DeepFloyd stage III mode is the SD x4 upscaler.",
            tooltip="Leave blank to use stabilityai/stable-diffusion-x4-upscaler.",
            importance=ImportanceLevel.ADVANCED,
            order=25,
            subsection="advanced",
            model_specific=["deepfloyd"],
            documentation="OPTIONS.md#--deepfloyd_validation_stage3_model",
        )
    )

    for field_name, label, arg_name, order in [
        (
            "deepfloyd_validation_stage1_num_inference_steps",
            "DeepFloyd Stage I Steps",
            "--deepfloyd_validation_stage1_num_inference_steps",
            26,
        ),
        (
            "deepfloyd_validation_stage2_num_inference_steps",
            "DeepFloyd Stage II Steps",
            "--deepfloyd_validation_stage2_num_inference_steps",
            27,
        ),
    ]:
        registry._add_field(
            ConfigField(
                name=field_name,
                arg_name=arg_name,
                ui_label=label,
                field_type=FieldType.NUMBER,
                tab="validation",
                section="validation_options",
                default_value=None,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
                help_text="Override the DeepFloyd per-stage validation step count.",
                tooltip="Leave blank to use the normal validation step count.",
                importance=ImportanceLevel.ADVANCED,
                order=order,
                subsection="advanced",
                model_specific=["deepfloyd"],
            )
        )

    for field_name, label, arg_name, order in [
        ("deepfloyd_validation_stage1_guidance", "DeepFloyd Stage I Guidance", "--deepfloyd_validation_stage1_guidance", 28),
        (
            "deepfloyd_validation_stage2_guidance",
            "DeepFloyd Stage II Guidance",
            "--deepfloyd_validation_stage2_guidance",
            29,
        ),
        (
            "deepfloyd_validation_stage3_guidance",
            "DeepFloyd Stage III Guidance",
            "--deepfloyd_validation_stage3_guidance",
            30,
        ),
    ]:
        registry._add_field(
            ConfigField(
                name=field_name,
                arg_name=arg_name,
                ui_label=label,
                field_type=FieldType.NUMBER,
                tab="validation",
                section="validation_options",
                default_value=None,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
                help_text="Override the DeepFloyd per-stage validation guidance scale.",
                tooltip="Leave blank to use the normal validation guidance value.",
                importance=ImportanceLevel.ADVANCED,
                order=order,
                subsection="advanced",
                model_specific=["deepfloyd"],
            )
        )

    registry._add_field(
        ConfigField(
            name="deepfloyd_validation_stage3_noise_level",
            arg_name="--deepfloyd_validation_stage3_noise_level",
            ui_label="DeepFloyd Stage III Noise",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_options",
            default_value=100,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Noise level passed to the SD x4 upscaler during DeepFloyd validation.",
            tooltip="Only used when DeepFloyd stage III mode is the SD x4 upscaler.",
            importance=ImportanceLevel.ADVANCED,
            order=31,
            subsection="advanced",
            model_specific=["deepfloyd"],
            documentation="OPTIONS.md#--deepfloyd_validation_stage3_noise_level",
        )
    )
