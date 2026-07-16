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
            name="hidream_use_load_balancing_loss",
            arg_name="--hidream_use_load_balancing_loss",
            ui_label="Enable HiDream Load Balancing Loss",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=False,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="hidream")],
            help_text="Apply experimental load balancing loss when training HiDream models.",
            tooltip="Balances expert contributions during HiDream training. Only available for HiDream model family.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=6,
        )
    )

    registry._add_field(
        ConfigField(
            name="hidream_load_balancing_loss_weight",
            arg_name="--hidream_load_balancing_loss_weight",
            ui_label="HiDream Load Balancing Weight",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="hidream_use_load_balancing_loss", operator="equals", value=True)],
            help_text="Strength multiplier for HiDream load balancing loss.",
            tooltip="Adjust if you need stronger balancing between experts. Leave blank to use the trainer default.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=7,
        )
    )
