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
            name="ltx_train_mode",
            arg_name="--ltx_train_mode",
            ui_label="LTX Train Mode",
            field_type=FieldType.SELECT,
            tab="model",
            section="model_specific",
            model_specific=["ltx"],
            default_value="i2v",
            choices=[
                {"value": "t2v", "label": "Text-to-Video"},
                {"value": "i2v", "label": "Image-to-Video"},
            ],
            help_text="Training mode for LTX models",
            tooltip="Choose whether datasets default to text-to-video (t2v) or image-to-video (i2v) processing.",
            importance=ImportanceLevel.ADVANCED,
            order=22,
        )
    )

    registry._add_field(
        ConfigField(
            name="ltx_i2v_prob",
            arg_name="--ltx_i2v_prob",
            ui_label="LTX Image-to-Video Probability",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            model_specific=["ltx"],
            default_value=0.1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            help_text="Probability of using image-to-video training for LTX",
            tooltip="Fraction of training that uses image-to-video instead of video-to-video.",
            importance=ImportanceLevel.ADVANCED,
            order=23,
        )
    )

    registry._add_field(
        ConfigField(
            name="ltx_partial_noise_fraction",
            arg_name="--ltx_partial_noise_fraction",
            ui_label="LTX Partial Noise Fraction",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            model_specific=["ltx"],
            default_value=0.05,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            help_text="Fraction of noise to add for LTX partial training",
            tooltip="Controls how much noise to add during partial training steps.",
            importance=ImportanceLevel.ADVANCED,
            order=24,
        )
    )

    registry._add_field(
        ConfigField(
            name="ltx_protect_first_frame",
            arg_name="--ltx_protect_first_frame",
            ui_label="LTX Protect First Frame",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            model_specific=["ltx"],
            default_value=False,
            help_text="Protect the first frame from noise in LTX training",
            tooltip="When enabled, first frame is kept clean while subsequent frames get noise.",
            importance=ImportanceLevel.ADVANCED,
            order=25,
        )
    )
