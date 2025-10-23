import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ValidationRule, ValidationRuleType

if TYPE_CHECKING:
    from ..registry import FieldRegistry


logger = logging.getLogger(__name__)


def register_optimizer_fields(registry: "FieldRegistry") -> None:
    """Add optimizer configuration fields."""
    adam_like_optimizers = [
        "adamw",
        "adamw_bf16",
        "adamw8bit",
        "adam",
        "adam8bit",
        "ao-adamw8bit",
        "ao-adamw4bit",
        "ao-adamwfp8",
        "optimi-adamw",
        "StableAdamWUnfused",
        "deepspeed-adamw",
    ]
    optimi_optimizers = ["optimi-adamw", "optimi-lion", "optimi-stableadamw"]
    accelerate_optimizers = ["ao-adamw8bit", "ao-adamw4bit", "ao-adamwfp8"]

    # Adam Beta1
    registry._add_field(
        ConfigField(
            name="adam_beta1",
            arg_name="--adam_beta1",
            ui_label="Adam Beta 1",
            field_type=FieldType.NUMBER,
            tab="training",
            section="optimizer_config",
            subsection="advanced",
            default_value=0.9,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Beta1 must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1, message="Beta1 must be between 0 and 1"),
            ],
            dependencies=[FieldDependency(field="optimizer", values=adam_like_optimizers)],
            help_text="First moment decay rate for Adam optimizers",
            tooltip="Controls momentum. Default 0.9 works well. Lower values reduce momentum.",
            importance=ImportanceLevel.ADVANCED,
            order=1,
        )
    )

    # Adam Beta2
    registry._add_field(
        ConfigField(
            name="adam_beta2",
            arg_name="--adam_beta2",
            ui_label="Adam Beta 2",
            field_type=FieldType.NUMBER,
            tab="training",
            section="optimizer_config",
            subsection="advanced",
            default_value=0.999,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Beta2 must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1, message="Beta2 must be between 0 and 1"),
            ],
            dependencies=[FieldDependency(field="optimizer", values=adam_like_optimizers)],
            help_text="Second moment decay rate for Adam optimizers",
            tooltip="Controls adaptive learning rates. Default 0.999 is standard. Lower values make adaptation more aggressive.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
        )
    )

    # Optimizer Beta1 (alternative parameter)
    registry._add_field(
        ConfigField(
            name="optimizer_beta1",
            arg_name="--optimizer_beta1",
            ui_label="Optimizer Beta 1",
            field_type=FieldType.NUMBER,
            tab="training",
            section="optimizer_config",
            subsection="advanced",
            default_value=None,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Beta1 must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1, message="Beta1 must be between 0 and 1"),
            ],
            help_text="First moment decay rate for optimizers",
            tooltip="Alternative parameter for setting beta1. Use optimizer_config for more control.",
            importance=ImportanceLevel.ADVANCED,
            order=3,
        )
    )

    # Optimizer Beta2 (alternative parameter)
    registry._add_field(
        ConfigField(
            name="optimizer_beta2",
            arg_name="--optimizer_beta2",
            ui_label="Optimizer Beta 2",
            field_type=FieldType.NUMBER,
            tab="training",
            section="optimizer_config",
            subsection="advanced",
            default_value=None,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Beta2 must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1, message="Beta2 must be between 0 and 1"),
            ],
            help_text="Second moment decay rate for optimizers",
            tooltip="Alternative parameter for setting beta2. Use optimizer_config for more control.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
        )
    )

    # Optimizer CPU Offload Method
    registry._add_field(
        ConfigField(
            name="optimizer_cpu_offload_method",
            arg_name="--optimizer_cpu_offload_method",
            ui_label="Optimizer CPU Offload Method",
            field_type=FieldType.SELECT,
            tab="training",
            section="optimizer_config",
            subsection="advanced",
            default_value=None,
            choices=[
                {"value": "none", "label": "None"},
            ],
            help_text="Method for CPU offloading optimizer states",
            tooltip="Currently only the placeholder 'none' option is supported.",
            importance=ImportanceLevel.ADVANCED,
            order=5,
        )
    )

    # Gradient Precision
    registry._add_field(
        ConfigField(
            name="gradient_precision",
            arg_name="--gradient_precision",
            ui_label="Gradient Precision",
            field_type=FieldType.SELECT,
            tab="training",
            section="optimizer_config",
            subsection="advanced",
            default_value=None,
            choices=[
                {"value": "unmodified", "label": "Unmodified"},
                {"value": "fp32", "label": "FP32"},
            ],
            help_text="Precision for gradient computation",
            tooltip="'unmodified' keeps the framework default. FP32 uses higher precision for gradient accumulation.",
            importance=ImportanceLevel.ADVANCED,
            order=6,
        )
    )

    # Adam Weight Decay
    registry._add_field(
        ConfigField(
            name="adam_weight_decay",
            arg_name="--adam_weight_decay",
            ui_label="Weight Decay",
            field_type=FieldType.NUMBER,
            tab="training",
            section="optimizer_config",
            subsection="advanced",
            default_value=1e-2,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="optimizer", values=adam_like_optimizers)],
            help_text="L2 regularisation strength for Adam-family optimizers.",
            tooltip="Higher values encourage smaller weights. If unsure, keep at 0.01 or match your baseline config.",
            importance=ImportanceLevel.ADVANCED,
            order=3,
        )
    )

    # Adam Epsilon
    registry._add_field(
        ConfigField(
            name="adam_epsilon",
            arg_name="--adam_epsilon",
            ui_label="Adam Epsilon",
            field_type=FieldType.NUMBER,
            tab="training",
            section="optimizer_config",
            subsection="advanced",
            default_value=1e-8,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be positive")],
            dependencies=[FieldDependency(field="optimizer", values=adam_like_optimizers)],
            help_text="Small constant added for numerical stability.",
            tooltip="Tweak only if you encounter numerical instabilities. Typical range 1e-8 to 1e-6.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
        )
    )

    # Prodigy Steps
    registry._add_field(
        ConfigField(
            name="prodigy_steps",
            arg_name="--prodigy_steps",
            ui_label="Prodigy Adjustment Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="optimizer_config",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="optimizer", values=["prodigy"])],
            help_text="Number of steps Prodigy should spend adapting its learning rate.",
            tooltip="If unset, SimpleTuner estimates 25% of total training steps. Set explicitly for long or short runs.",
            importance=ImportanceLevel.ADVANCED,
            order=5,
        )
    )

    # Max Gradient Norm
    registry._add_field(
        ConfigField(
            name="max_grad_norm",
            arg_name="--max_grad_norm",
            ui_label="Max Gradient Norm",
            field_type=FieldType.NUMBER,
            tab="training",
            section="optimizer_config",
            default_value=2.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            help_text="Gradient clipping threshold to prevent exploding gradients.",
            tooltip="Set to 0 to disable clipping. Value of 2.0 is a balanced default for diffusion models.",
            importance=ImportanceLevel.IMPORTANT,
            order=6,
        )
    )

    # Optimizer Extra Settings
    registry._add_field(
        ConfigField(
            name="optimizer_config",
            arg_name="--optimizer_config",
            ui_label="Optimizer Extra Settings",
            field_type=FieldType.TEXT,
            tab="training",
            section="optimizer_config",
            subsection="advanced",
            placeholder="beta1=0.9,beta2=0.95,weight_decay=0.01",
            help_text="Comma-separated key=value pairs forwarded to the selected optimizer",
            tooltip="Example: beta1=0.9,beta2=0.95,weight_decay=0.01. Leave blank to use optimizer defaults.",
            importance=ImportanceLevel.ADVANCED,
            order=7,
        )
    )

    # Gradient Clip Method
    registry._add_field(
        ConfigField(
            name="grad_clip_method",
            arg_name="--grad_clip_method",
            ui_label="Gradient Clip Method",
            field_type=FieldType.SELECT,
            tab="training",
            section="optimizer_config",
            subsection="advanced",
            default_value="value",
            choices=[{"value": "value", "label": "Clip Individual Values"}, {"value": "norm", "label": "Clip By Norm"}],
            help_text="Strategy for applying max_grad_norm during clipping.",
            tooltip="'Value' clips offending gradients individually. 'Norm' rescales all gradients proportionally.",
            importance=ImportanceLevel.ADVANCED,
            order=7,
        )
    )

    # Optimizer Offload Gradients
    registry._add_field(
        ConfigField(
            name="optimizer_offload_gradients",
            arg_name="--optimizer_offload_gradients",
            ui_label="Offload Gradients to CPU",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="optimizer_config",
            default_value=False,
            dependencies=[FieldDependency(field="optimizer", values=accelerate_optimizers + optimi_optimizers)],
            help_text="Move optimizer gradients to CPU to save GPU memory.",
            tooltip="Useful on large models when paired with Accelerate/Optimi optimizers. Increases CPU pressure.",
            importance=ImportanceLevel.ADVANCED,
            order=8,
        )
    )

    # Fuse Optimizer Kernels
    registry._add_field(
        ConfigField(
            name="fuse_optimizer",
            arg_name="--fuse_optimizer",
            ui_label="Use Fused Optimizer",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="optimizer_config",
            default_value=False,
            dependencies=[FieldDependency(field="optimizer", values=accelerate_optimizers + optimi_optimizers)],
            help_text="Enable fused kernels when offloading to reduce memory overhead.",
            tooltip="May be slower but reduces memory when using CPU-offloaded optimizers.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=9,
        )
    )

    # Optimizer Release Gradients
    registry._add_field(
        ConfigField(
            name="optimizer_release_gradients",
            arg_name="--optimizer_release_gradients",
            ui_label="Release Gradients After Step",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="optimizer_config",
            default_value=False,
            dependencies=[FieldDependency(field="optimizer", values=optimi_optimizers)],
            help_text="Free gradient tensors immediately after optimizer step when using Optimi optimizers.",
            tooltip="Saves memory on Optimi optimizers at the cost of extra allocations in the next step.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=10,
        )
    )
