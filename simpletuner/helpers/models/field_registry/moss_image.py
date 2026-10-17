from simpletuner.simpletuner_sdk.server.services.field_registry.types import ConfigField, FieldType, ImportanceLevel


def register_fields(registry):
    for name, label, help_text, default in (
        (
            "moss_image_auto_json",
            "MOSS-Image JSON Captions",
            "Encode positive captions in MOSS-Image's native summary/items JSON format.",
            False,
        ),
        (
            "moss_image_prompt_upsample",
            "MOSS-Image Prompt Upsampling",
            "Expand positive captions into native JSON layouts with a vision-language model before encoding.",
            False,
        ),
        (
            "moss_image_prompt_upsampler_model_id",
            "MOSS-Image Caption Model",
            "Hugging Face repository or local path for the vision-language model used to generate JSON captions.",
            "Qwen/Qwen3.5-27B",
        ),
    ):
        registry._add_field(
            ConfigField(
                name=name,
                arg_name=f"--{name}",
                ui_label=label,
                field_type=FieldType.CHECKBOX if isinstance(default, bool) else FieldType.TEXT,
                tab="model",
                section="model_specific",
                model_specific=["moss_image"],
                default_value=default,
                help_text=help_text,
                importance=ImportanceLevel.ADVANCED,
                documentation=f"OPTIONS.md#--{name}",
            )
        )
    registry._add_field(
        ConfigField(
            name="moss_image_validation_shift",
            arg_name="--moss_image_validation_shift",
            ui_label="MOSS-Image Validation Shift",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_options",
            model_specific=["moss_image"],
            default_value=5.0,
            help_text="Inference flow shift for MOSS-Image validation. The pretrained training grid remains unshifted.",
            importance=ImportanceLevel.ADVANCED,
            order=29,
            documentation="OPTIONS.md#--moss_image_validation_shift",
        )
    )
    registry._add_field(
        ConfigField(
            name="moss_image_uncond_skip_layers",
            arg_name="--moss_image_uncond_skip_layers",
            ui_label="MOSS-Image Negative-Branch Skip Layers",
            field_type=FieldType.TEXT_JSON,
            tab="validation",
            section="validation_options",
            model_specific=["moss_image"],
            default_value="[22]",
            help_text="JSON list of transformer layers to skip in the negative CFG branch. Use [] to disable skipping.",
            tooltip="OpenMOSS recommends layer 22 (zero based). These skips retain two forwards per CFG sampling step.",
            importance=ImportanceLevel.ADVANCED,
            order=30,
            documentation="OPTIONS.md#--moss_image_uncond_skip_layers",
        )
    )
