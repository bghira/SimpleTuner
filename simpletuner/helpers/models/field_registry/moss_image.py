from simpletuner.simpletuner_sdk.server.services.field_registry.types import ConfigField, FieldType, ImportanceLevel


def register_fields(registry):
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
