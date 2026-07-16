from simpletuner.simpletuner_sdk.server.services.field_registry.types import ConfigField, FieldType, ImportanceLevel


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="krea2_reference_latents",
            arg_name="--krea2_reference_latents",
            ui_label="Krea 2 Reference Latents",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            model_specific=["krea2"],
            default_value=False,
            help_text="Enable Krea 2 reference-dataset training with image-context prompt embeds and clean reference latents.",
            tooltip="When enabled, Krea 2 requires paired conditioning data, encodes prompts with the reference image through Qwen3VL, and appends the clean reference latents to the transformer input.",
            importance=ImportanceLevel.ADVANCED,
            order=36,
            documentation="OPTIONS.md#--krea2_reference_latents",
        )
    )
