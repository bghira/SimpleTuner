import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ValidationRule, ValidationRuleType

if TYPE_CHECKING:
    from ..registry import FieldRegistry


logger = logging.getLogger(__name__)


def register_loss_fields(registry: "FieldRegistry") -> None:
    """Add loss function configuration fields."""
    # Loss Type
    registry._add_field(
        ConfigField(
            name="loss_type",
            arg_name="--loss_type",
            ui_label="Loss Function",
            field_type=FieldType.SELECT,
            tab="training",
            section="loss_functions",
            default_value="l2",
            choices=[
                {"value": "l2", "label": "L2 (MSE)"},
                {"value": "huber", "label": "Huber"},
                {"value": "smooth_l1", "label": "Smooth L1"},
            ],
            help_text="How the model measures prediction errors. L2 (MSE) squares errors so large mistakes are penalized heavily. Huber/Smooth L1 treat large errors more gently, which can help when your dataset has unusual images.",
            tooltip="L2 is standard and works well for most cases. Huber/Smooth L1 reduce the influence of outliers (unusual samples) during training.",
            importance=ImportanceLevel.ADVANCED,
            order=1,
        )
    )

    # Huber Schedule
    registry._add_field(
        ConfigField(
            name="huber_schedule",
            arg_name="--huber_schedule",
            ui_label="Huber Schedule",
            field_type=FieldType.SELECT,
            tab="training",
            section="loss_functions",
            default_value="snr",
            choices=[
                {"value": "snr", "label": "SNR (Default)"},
                {"value": "exponential", "label": "Exponential"},
                {"value": "constant", "label": "Constant"},
            ],
            dependencies=[FieldDependency(field="loss_type", operator="equals", value="huber")],
            help_text="Schedule for Huber loss transition threshold",
            tooltip="Controls how huber_c evolves during training. Only applies when using Huber loss.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
        )
    )

    # Huber C Value
    registry._add_field(
        ConfigField(
            name="huber_c",
            arg_name="--huber_c",
            ui_label="Huber C Threshold",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0.1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="loss_type", operator="equals", value="huber")],
            help_text="Transition point between L2 and L1 regions for Huber loss",
            tooltip="Lower values emphasise L1 behaviour sooner; higher values behave more like L2.",
            importance=ImportanceLevel.ADVANCED,
            order=3,
        )
    )

    # SNR Gamma
    registry._add_field(
        ConfigField(
            name="snr_gamma",
            arg_name="--snr_gamma",
            ui_label="SNR Gamma",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            order=4,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="SNR gamma must be non-negative")],
            help_text="SNR weighting gamma value (0 = disabled). Try 5 when using epsilon/V-prediction models.",
            tooltip="Rebalances loss across timesteps. Recommended value: 5.0 for epsilon and V-Prediction models to curb overemphasis on easy timesteps.",
            importance=ImportanceLevel.ADVANCED,
            documentation="OPTIONS.md#--snr_gamma",
        )
    )

    # Masked Loss Probability
    registry._add_field(
        ConfigField(
            name="masked_loss_probability",
            arg_name="--masked_loss_probability",
            ui_label="Masked Loss Probability",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=1.0,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            help_text="Probability of applying masked loss weighting per batch",
            tooltip="Lower values reduce how often masked loss is applied, useful for datasets with sparse masks.",
            importance=ImportanceLevel.ADVANCED,
            order=5,
        )
    )

    # HiDream Load Balancing Loss Toggle
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

    # HiDream Load Balancing Weight
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
