from simpletuner.simpletuner_sdk.server.services.field_registry.types import (
    ConfigField,
    FieldDependency,
    FieldType,
    ImportanceLevel,
    ValidationRule,
    ValidationRuleType,
)


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="enable_chunked_feed_forward",
            arg_name="--enable_chunked_feed_forward",
            ui_label="Enable Feed-Forward Chunking",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="memory_optimization",
            default_value=False,
            help_text="Split Wan feed-forward layers into smaller chunks to reduce peak VRAM usage.",
            tooltip="Available for Wan models. Breaks long MLPs into mini-batches so checkpoint recomputes allocate less memory.",
            importance=ImportanceLevel.ADVANCED,
            model_specific=["wan", "wan_s2v"],
            order=8,
        )
    )

    registry._add_field(
        ConfigField(
            name="feed_forward_chunk_size",
            arg_name="--feed_forward_chunk_size",
            ui_label="Feed-Forward Chunk Size",
            field_type=FieldType.NUMBER,
            tab="model",
            section="memory_optimization",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Chunk size must be at least 1")],
            help_text="Number of samples processed per chunk when feed-forward chunking is enabled.",
            tooltip="Leave blank for auto. Lower values reduce memory further but increase wall-clock time.",
            importance=ImportanceLevel.ADVANCED,
            model_specific=["wan", "wan_s2v"],
            order=9,
            dependencies=[
                FieldDependency(field="enable_chunked_feed_forward", operator="equals", value=True, action="show")
            ],
            allow_empty=True,
        )
    )

    registry._add_field(
        ConfigField(
            name="wan_force_2_1_time_embedding",
            arg_name="--wan_force_2_1_time_embedding",
            ui_label="Force Wan 2.1 Time Embedding",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            default_value=False,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="wan", action="show")],
            help_text="Use Wan 2.1 style time embeddings even when running Wan 2.2 checkpoints.",
            tooltip="Enable this if Wan 2.2 checkpoints report shape mismatches in the time embedding layers.",
            importance=ImportanceLevel.ADVANCED,
            order=30,
        )
    )

    registry._add_field(
        ConfigField(
            name="wan_validation_load_other_stage",
            arg_name="--wan_validation_load_other_stage",
            ui_label="Wan Paired-Stage Validation",
            field_type=FieldType.CHECKBOX,
            tab="validation",
            section="validation_options",
            default_value=False,
            help_text="Load the opposite Wan 2.2 stage during validation so the pipeline can switch denoisers at the stage boundary.",
            tooltip="For Wan 2.2 and compatible staged flavours such as AnimeGen, this loads the fixed peer stage alongside the trained stage for validation renders.",
            importance=ImportanceLevel.ADVANCED,
            order=32,
            subsection="advanced",
            model_specific=["wan"],
            documentation="OPTIONS.md#--wan_validation_load_other_stage",
        )
    )
