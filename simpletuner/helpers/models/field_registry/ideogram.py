from simpletuner.simpletuner_sdk.server.services.field_registry.types import (
    ConfigField,
    FieldDependency,
    FieldType,
    ImportanceLevel,
)


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="ideogram_auto_json",
            arg_name="--ideogram_auto_json",
            ui_label="Ideogram Auto JSON",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            model_specific=["ideogram"],
            default_value=True,
            help_text="Convert non-JSON Ideogram 4 prompts into the structured JSON caption schema.",
            tooltip="When enabled, plain validation prompts are wrapped into Ideogram 4's JSON caption format.",
            importance=ImportanceLevel.ADVANCED,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="ideogram")],
            order=33,
        )
    )

    registry._add_field(
        ConfigField(
            name="ideogram_validation",
            arg_name="--ideogram_validation",
            ui_label="Ideogram Validation",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            model_specific=["ideogram"],
            default_value=False,
            help_text="Temporarily enable Ideogram validation by reusing the conditional transformer for CFG's unconditional pass.",
            tooltip="Validation is off by default for Ideogram until separate unconditional-model handling is implemented.",
            importance=ImportanceLevel.ADVANCED,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="ideogram")],
            order=34,
        )
    )

    registry._add_field(
        ConfigField(
            name="ideogram_prompt_upsample",
            arg_name="--ideogram_prompt_upsample",
            ui_label="Ideogram Prompt Upsample",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            model_specific=["ideogram"],
            default_value=False,
            help_text="Use Ideogram 4's prompt enhancer to rewrite prompts before text embedding cache generation.",
            tooltip="When enabled and supported by the loaded pipeline, captions are expanded with Ideogram's prompt upsampler before JSON conversion and encoding.",
            importance=ImportanceLevel.ADVANCED,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="ideogram")],
            order=35,
        )
    )

    registry._add_field(
        ConfigField(
            name="ideogram_prompt_enhancer_head_id",
            arg_name="--ideogram_prompt_enhancer_head_id",
            ui_label="Ideogram Prompt Enhancer Head",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_specific",
            model_specific=["ideogram"],
            default_value="diffusers/qwen3-vl-8b-instruct-lm-head",
            help_text="Hugging Face repo id for Ideogram 4's prompt upsampling LM head.",
            tooltip="Used when --ideogram_prompt_upsample is enabled to rewrite prompts into Ideogram's structured JSON caption schema.",
            importance=ImportanceLevel.ADVANCED,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="ideogram")],
            order=36,
        )
    )

    registry._add_field(
        ConfigField(
            name="ideogram_schedule_mu",
            arg_name="--ideogram_schedule_mu",
            ui_label="Ideogram Schedule Mu",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            model_specific=["ideogram"],
            default_value=0.0,
            help_text="Base mean used by Ideogram 4's resolution-aware logit-normal training timestep schedule.",
            tooltip="Matches the default mu used by the vendored Ideogram validation pipeline.",
            importance=ImportanceLevel.ADVANCED,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="ideogram")],
            order=37,
        )
    )

    registry._add_field(
        ConfigField(
            name="ideogram_schedule_std",
            arg_name="--ideogram_schedule_std",
            ui_label="Ideogram Schedule Std",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            model_specific=["ideogram"],
            default_value=1.5,
            help_text="Standard deviation used by Ideogram 4's logit-normal training timestep schedule.",
            tooltip="Matches the default std used by the vendored Ideogram validation pipeline.",
            importance=ImportanceLevel.ADVANCED,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="ideogram")],
            order=38,
        )
    )
