from simpletuner.simpletuner_sdk.server.services.field_registry.types import ConfigField, FieldType, ImportanceLevel


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="pixart_validation_pipeline_mode",
            arg_name="--pixart_validation_pipeline_mode",
            ui_label="PixArt Validation Pipeline",
            field_type=FieldType.SELECT,
            tab="validation",
            section="validation_options",
            default_value="trained-stage",
            choices=[
                {"value": "trained-stage", "label": "Trained Stage Only"},
                {"value": "full-pipeline", "label": "Full Pipeline"},
            ],
            help_text="Choose whether PixArt validation runs only the trained stage or chains the v0.7 split pipeline.",
            tooltip="Full pipeline runs stage 1 to the refiner boundary as latents, then resumes through stage 2.",
            importance=ImportanceLevel.ADVANCED,
            order=33,
            subsection="advanced",
            model_specific=["pixart_sigma"],
            documentation="OPTIONS.md#--pixart_validation_pipeline_mode",
        )
    )

    registry._add_field(
        ConfigField(
            name="pixart_validation_stage1_model",
            arg_name="--pixart_validation_stage1_model",
            ui_label="PixArt Stage 1 Model",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_options",
            default_value=None,
            placeholder="terminusresearch/pixart-900m-1024-ft-v0.7-stage1",
            help_text="Fixed PixArt stage 1 model used when validating a trained stage 2 model through the full pipeline.",
            tooltip="Leave blank to use the PixArt 900M v0.7 stage 1 model.",
            importance=ImportanceLevel.ADVANCED,
            order=34,
            subsection="advanced",
            model_specific=["pixart_sigma"],
            documentation="OPTIONS.md#--pixart_validation_stage1_model",
        )
    )

    registry._add_field(
        ConfigField(
            name="pixart_validation_stage2_model",
            arg_name="--pixart_validation_stage2_model",
            ui_label="PixArt Stage 2 Model",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_options",
            default_value=None,
            placeholder="terminusresearch/pixart-900m-1024-ft-v0.7-stage2",
            help_text="Fixed PixArt stage 2 model used when validating a trained stage 1 model through the full pipeline.",
            tooltip="Leave blank to use the PixArt 900M v0.7 stage 2 model.",
            importance=ImportanceLevel.ADVANCED,
            order=35,
            subsection="advanced",
            model_specific=["pixart_sigma"],
            documentation="OPTIONS.md#--pixart_validation_stage2_model",
        )
    )
