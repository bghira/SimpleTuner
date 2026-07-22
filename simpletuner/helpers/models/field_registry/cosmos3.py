from simpletuner.simpletuner_sdk.server.services.field_registry.types import (
    ConfigField,
    FieldDependency,
    FieldType,
    ImportanceLevel,
)


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="cosmos3_reasoner_component",
            arg_name="--cosmos3_reasoner_component",
            ui_label="Cosmos3 Reasoner Component",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_specific",
            model_specific=["cosmos3"],
            default_value="auto",
            help_text="Hugging Face repo id or local path for the frozen Cosmos3 reasoning component used by text cache.",
            tooltip="Use auto to select the matching Edge, Nano, Super, Super-I2V, or Super-T2I reasoner component.",
            importance=ImportanceLevel.ADVANCED,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="cosmos3")],
            order=33,
        )
    )
    registry._add_field(
        ConfigField(
            name="cosmos3_generator_component",
            arg_name="--cosmos3_generator_component",
            ui_label="Cosmos3 Generator Component",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_specific",
            model_specific=["cosmos3"],
            default_value="auto",
            help_text="Hugging Face repo id or local path for the Cosmos3 generation-only transformer component.",
            tooltip="Use auto to select the matching Edge, Nano, Super, Super-I2V, or Super-T2I generator component.",
            importance=ImportanceLevel.ADVANCED,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="cosmos3")],
            order=34,
        )
    )
