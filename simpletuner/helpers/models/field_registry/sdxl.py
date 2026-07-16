from simpletuner.simpletuner_sdk.server.services.field_registry.types import (
    ConfigField,
    FieldDependency,
    FieldType,
    ImportanceLevel,
)


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="sdxl_refiner_uses_full_range",
            arg_name="--sdxl_refiner_uses_full_range",
            ui_label="SDXL Refiner Full Range",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            default_value=False,
            help_text="Use full timestep range for SDXL refiner",
            tooltip="When enabled, refiner uses full timestep range instead of just high timesteps.",
            importance=ImportanceLevel.ADVANCED,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="sdxl")],
            order=32,
        )
    )

    registry._add_field(
        ConfigField(
            name="sdxl_validation_pipeline_mode",
            arg_name="--sdxl_validation_pipeline_mode",
            ui_label="SDXL Validation Pipeline",
            field_type=FieldType.SELECT,
            tab="validation",
            section="validation_options",
            default_value="trained-stage",
            choices=[
                {"value": "trained-stage", "label": "Trained Stage Only"},
                {"value": "full-pipeline", "label": "Full Pipeline"},
            ],
            help_text="Choose whether SDXL validation runs only the trained stage or chains the base/refiner split pipeline.",
            tooltip="Full pipeline runs stage 1 to the refiner boundary as latents, then resumes through stage 2.",
            importance=ImportanceLevel.ADVANCED,
            order=33,
            subsection="advanced",
            model_specific=["sdxl"],
            documentation="OPTIONS.md#--sdxl_validation_pipeline_mode",
        )
    )

    registry._add_field(
        ConfigField(
            name="sdxl_validation_stage1_model",
            arg_name="--sdxl_validation_stage1_model",
            ui_label="SDXL Stage 1 Model",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_options",
            default_value=None,
            placeholder="stabilityai/stable-diffusion-xl-base-1.0",
            help_text="Fixed SDXL stage 1 model used when validating a trained refiner through the full pipeline.",
            tooltip="Leave blank to infer the matching SDXL base model from the selected flavour.",
            importance=ImportanceLevel.ADVANCED,
            order=34,
            subsection="advanced",
            model_specific=["sdxl"],
            documentation="OPTIONS.md#--sdxl_validation_stage1_model",
        )
    )

    registry._add_field(
        ConfigField(
            name="sdxl_validation_stage2_model",
            arg_name="--sdxl_validation_stage2_model",
            ui_label="SDXL Stage 2 Model",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_options",
            default_value=None,
            placeholder="stabilityai/stable-diffusion-xl-refiner-1.0",
            help_text="Fixed SDXL refiner model used when validating a trained stage 1 model through the full pipeline.",
            tooltip="Leave blank to infer the matching SDXL refiner from the selected flavour.",
            importance=ImportanceLevel.ADVANCED,
            order=35,
            subsection="advanced",
            model_specific=["sdxl"],
            documentation="OPTIONS.md#--sdxl_validation_stage2_model",
        )
    )
