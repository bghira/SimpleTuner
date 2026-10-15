from simpletuner.simpletuner_sdk.server.services.field_registry.types import (
    ConfigField,
    FieldDependency,
    FieldType,
    ImportanceLevel,
)


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="flux3_prompt_upsample",
            arg_name="--flux3_prompt_upsample",
            ui_label="FLUX.3 Prompt Upsampling",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            model_specific=["flux3_image"],
            default_value=False,
            help_text="Expand positive captions with the FLUX.3 Image Klein prompt upsampler before text caching.",
            tooltip=(
                "Uses reference images for edits. Downloads a separate Qwen3.5-9B model; "
                "adds preprocessing time and memory use."
            ),
            importance=ImportanceLevel.ADVANCED,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="flux3_image")],
            order=35,
        )
    )
