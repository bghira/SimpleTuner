from simpletuner.simpletuner_sdk.server.services.field_registry.types import (
    ConfigField,
    FieldDependency,
    FieldType,
    ImportanceLevel,
)


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="sd3_clip_uncond_behaviour",
            arg_name="--sd3_clip_uncond_behaviour",
            ui_label="SD3 CLIP Unconditional Behavior",
            field_type=FieldType.SELECT,
            tab="training",
            section="text_encoder",
            subsection="advanced",
            default_value="empty_string",
            choices=[{"value": "empty_string", "label": "Empty String"}, {"value": "zero", "label": "Zero"}],
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True)],
            help_text="How SD3 handles unconditional prompts",
            tooltip="Affects how SD3 processes prompts without conditioning. Empty string is default.",
            importance=ImportanceLevel.ADVANCED,
            order=11,
        )
    )

    registry._add_field(
        ConfigField(
            name="sd3_t5_uncond_behaviour",
            arg_name="--sd3_t5_uncond_behaviour",
            ui_label="SD3 T5 Unconditional Behavior",
            field_type=FieldType.SELECT,
            tab="training",
            section="text_encoder",
            subsection="advanced",
            default_value=None,
            choices=[{"value": "empty_string", "label": "Empty String"}, {"value": "zero", "label": "Zero"}],
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True)],
            help_text="How SD3 T5 handles unconditional prompts",
            tooltip="Overrides CLIP behavior for T5. If not set, follows CLIP setting.",
            importance=ImportanceLevel.ADVANCED,
            order=12,
        )
    )
