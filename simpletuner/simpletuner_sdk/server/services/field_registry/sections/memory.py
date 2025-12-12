import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ValidationRule, ValidationRuleType

if TYPE_CHECKING:
    from ..registry import FieldRegistry


logger = logging.getLogger(__name__)


def register_memory_fields(registry: "FieldRegistry") -> None:
    """Add memory and performance configuration fields."""
    # Gradient Accumulation Steps
    registry._add_field(
        ConfigField(
            name="gradient_accumulation_steps",
            arg_name="--gradient_accumulation_steps",
            ui_label="Gradient Accumulation Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="training_schedule",
            default_value=1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="Number of steps to accumulate gradients",
            tooltip="Simulates larger batch sizes. Effective batch = batch_size * accumulation_steps * num_gpus",
            importance=ImportanceLevel.IMPORTANT,
            order=4,
            documentation="OPTIONS.md#--gradient_accumulation_steps",
        )
    )
