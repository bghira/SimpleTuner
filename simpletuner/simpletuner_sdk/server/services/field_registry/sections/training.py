import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from simpletuner.helpers.training.optimizer_param import optimizer_choices as _optimizer_choices_map

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ValidationRule, ValidationRuleType

if TYPE_CHECKING:
    from ..registry import FieldRegistry


logger = logging.getLogger(__name__)


def register_training_fields(registry: "FieldRegistry") -> None:
    """Add training parameter fields."""
    # Number of Training Epochs
    registry._add_field(
        ConfigField(
            name="num_train_epochs",
            arg_name="--num_train_epochs",
            ui_label="Number of Epochs",
            field_type=FieldType.NUMBER,
            tab="training",
            section="training_schedule",
            default_value=1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Epochs must be non-negative"),
                ValidationRule(ValidationRuleType.MAX, value=1000, message="Consider if you really need >1000 epochs"),
            ],
            help_text="Number of times to iterate through the entire dataset",
            tooltip="One epoch = one full pass through all training data. More epochs can improve quality but may cause overfitting.",
            importance=ImportanceLevel.ESSENTIAL,
            order=1,
        )
    )

    # Max Training Steps
    registry._add_field(
        ConfigField(
            name="max_train_steps",
            arg_name="--max_train_steps",
            ui_label="Max Training Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="training_schedule",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Steps must be non-negative")],
            help_text="Maximum number of training steps (0 = use epochs instead)",
            tooltip="If set to a positive value, training will stop after this many steps regardless of epochs",
            importance=ImportanceLevel.IMPORTANT,
            order=2,
        )
    )

    # Batch Size
    registry._add_field(
        ConfigField(
            name="train_batch_size",
            arg_name="--train_batch_size",
            ui_label="Training Batch Size",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="training_data",
            default_value=4,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Batch size must be at least 1"),
                ValidationRule(ValidationRuleType.MAX, value=128, message="Batch size >128 is unusual"),
            ],
            help_text="Number of samples processed per forward/backward pass (per device).",
            tooltip="Higher batch sizes can improve training stability but require more VRAM. Start with 1-4 for most GPUs.",
            importance=ImportanceLevel.ESSENTIAL,
            order=3,
        )
    )

    # Learning Rate
    registry._add_field(
        ConfigField(
            name="learning_rate",
            arg_name="--learning_rate",
            ui_label="Learning Rate",
            field_type=FieldType.NUMBER,
            tab="training",
            section="learning_rate",
            default_value=4e-7,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Learning rate must be positive"),
                ValidationRule(ValidationRuleType.MAX, value=1, message="Learning rate >1 is extremely high"),
            ],
            help_text="Base learning rate for training",
            tooltip="Controls how much model weights change per step. Lower = more stable but slower. Typical range: 1e-6 to 1e-4",
            importance=ImportanceLevel.ESSENTIAL,
            order=1,
        )
    )

    # Optimizer
    optimizer_choices = list(_optimizer_choices_map.keys())
    if not optimizer_choices:
        raise RuntimeError("optimizer_choices from optimizer_param is empty; expected at least one optimizer option")
    registry._add_field(
        ConfigField(
            name="optimizer",
            arg_name="--optimizer",
            ui_label="Optimizer",
            field_type=FieldType.SELECT,
            tab="training",
            section="optimizer_config",
            choices=[{"value": opt, "label": opt} for opt in optimizer_choices],
            validation_rules=[
                ValidationRule(ValidationRuleType.REQUIRED, message="Optimizer is required"),
                ValidationRule(ValidationRuleType.CHOICES, value=optimizer_choices),
            ],
            help_text="Optimization algorithm for training",
            tooltip="AdamW variants are most common. 8-bit versions save memory. Prodigy auto-adjusts learning rate.",
            importance=ImportanceLevel.ESSENTIAL,
            order=5,
        )
    )


    # LR Scheduler
    lr_scheduler_choices = [
        "linear",
        "sine",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ]
    registry._add_field(
        ConfigField(
            name="lr_scheduler",
            arg_name="--lr_scheduler",
            ui_label="Learning Rate Scheduler",
            field_type=FieldType.SELECT,
            tab="training",
            section="learning_rate",
            default_value="sine",
            choices=[{"value": s, "label": s.replace("_", " ").title()} for s in lr_scheduler_choices],
            validation_rules=[ValidationRule(ValidationRuleType.CHOICES, value=lr_scheduler_choices)],
            help_text="How learning rate changes during training",
            tooltip="Sine and cosine gradually reduce LR. Constant keeps it fixed. Warmup helps stability at start.",
            importance=ImportanceLevel.IMPORTANT,
            order=2,
        )
    )

    # Gradient Accumulation Steps
    registry._add_field(
        ConfigField(
            name="gradient_accumulation_steps",
            arg_name="--gradient_accumulation_steps",
            ui_label="Gradient Accumulation Steps",
            field_type=FieldType.NUMBER,
            tab="model",
            section="memory_optimization",
            default_value=1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="Number of steps to accumulate gradients before updating",
            tooltip="Simulates larger batch sizes without using more VRAM. Effective batch = batch_size * accumulation_steps",
            importance=ImportanceLevel.IMPORTANT,
            order=4,
        )
    )

    # LR Warmup Steps
    registry._add_field(
        ConfigField(
            name="lr_warmup_steps",
            arg_name="--lr_warmup_steps",
            ui_label="Learning Rate Warmup Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="learning_rate",
            default_value=500,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Number of steps to gradually increase LR from 0",
            tooltip="Helps training stability at start. Typically 5-10% of total steps",
            importance=ImportanceLevel.ADVANCED,
            order=3,
        )
    )

    # Max Checkpoints
    registry._add_field(
        ConfigField(
            name="checkpoints_total_limit",
            arg_name="--checkpoints_total_limit",
            ui_label="Maximum Checkpoints to Keep",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="checkpointing",
            default_value=5,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative (0 = unlimited)")
            ],
            help_text="Maximum number of checkpoints to keep on disk",
            tooltip="Older checkpoints are deleted when limit is exceeded. Set to 0 for unlimited",
            importance=ImportanceLevel.ADVANCED,
            order=2,
        )
    )

    # Gradient Checkpointing
    registry._add_field(
        ConfigField(
            name="gradient_checkpointing",
            arg_name="--gradient_checkpointing",
            ui_label="Enable Gradient Checkpointing",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="memory_optimization",
            default_value=True,
            help_text="Trade compute for memory by recomputing activations",
            tooltip="Reduces VRAM usage significantly but increases training time by ~20%",
            importance=ImportanceLevel.ADVANCED,
            order=1,
        )
    )

    # Train Text Encoder
    registry._add_field(
        ConfigField(
            name="train_text_encoder",
            arg_name="--train_text_encoder",
            ui_label="Train Text Encoder",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="text_encoder",
            default_value=False,
            help_text="Also train the text encoder (CLIP) model",
            tooltip="Can improve concept learning but uses more VRAM. Not recommended for LoRA",
            importance=ImportanceLevel.ADVANCED,
            order=1,
        )
    )

    # Text Encoder LR
    registry._add_field(
        ConfigField(
            name="text_encoder_lr",
            arg_name="--text_encoder_lr",
            ui_label="Text Encoder Learning Rate",
            field_type=FieldType.NUMBER,
            tab="training",
            section="text_encoder",
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be positive")],
            help_text="Separate learning rate for text encoder",
            tooltip="Usually lower than main LR. If not set, uses main learning rate",
            importance=ImportanceLevel.ADVANCED,
            order=4,
            dependencies=[FieldDependency(field="train_text_encoder", operator="equals", value=True, action="show")],
        )
    )

    # LR Number of Cycles
    registry._add_field(
        ConfigField(
            name="lr_num_cycles",
            arg_name="--lr_num_cycles",
            ui_label="LR Scheduler Cycles",
            field_type=FieldType.NUMBER,
            tab="training",
            section="learning_rate",
            default_value=1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must have at least 1 cycle")],
            dependencies=[
                FieldDependency(field="lr_scheduler", operator="equals", value="cosine_with_restarts", action="show")
            ],
            help_text="Number of cosine annealing cycles",
            tooltip="Only used with cosine_with_restarts scheduler. More cycles = more LR resets.",
            importance=ImportanceLevel.ADVANCED,
            order=5,
        )
    )

    # LR Power
    registry._add_field(
        ConfigField(
            name="lr_power",
            arg_name="--lr_power",
            ui_label="LR Polynomial Power",
            field_type=FieldType.NUMBER,
            tab="training",
            section="learning_rate",
            default_value=0.8,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.1, message="Power should be positive")],
            dependencies=[FieldDependency(field="lr_scheduler", operator="equals", value="polynomial", action="show")],
            help_text="Power for polynomial decay scheduler",
            tooltip="1.0 = linear decay, 2.0 = quadratic decay. Higher = stays high longer then drops faster.",
            importance=ImportanceLevel.ADVANCED,
            order=6,
        )
    )

    # Use Soft Min SNR
    registry._add_field(
        ConfigField(
            name="use_soft_min_snr",
            arg_name="--use_soft_min_snr",
            ui_label="Use Soft Min-SNR",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            subsection="advanced",
            default_value=False,
            dependencies=[FieldDependency(field="snr_gamma", operator="greater_than", value=0, action="show")],
            help_text="Use soft clamping instead of hard clamping for Min-SNR",
            tooltip="Smoother transition at the clamping boundary. May improve training stability.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=8,
        )
    )

    # Use EMA Toggle
    registry._add_field(
        ConfigField(
            name="use_ema",
            arg_name="--use_ema",
            ui_label="Enable EMA",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="ema_config",
            default_value=False,
            checkbox_label="Use EMA",
            help_text="Maintain an exponential moving average copy of the model during training.",
            tooltip="Improves convergence stability at the cost of extra memory and compute.",
            importance=ImportanceLevel.ADVANCED,
            order=0,
        )
    )

    # EMA Device
    registry._add_field(
        ConfigField(
            name="ema_device",
            arg_name="--ema_device",
            ui_label="EMA Device",
            field_type=FieldType.SELECT,
            tab="model",
            section="ema_config",
            default_value="cpu",
            choices=[
                {"value": "accelerator", "label": "Training Accelerator"},
                {"value": "cpu", "label": "CPU"},
            ],
            dependencies=[FieldDependency(field="use_ema", operator="equals", value=True, action="show")],
            help_text="Where to keep the EMA weights in-between updates.",
            tooltip="'Accelerator' keeps EMA on the training device for fastest updates. 'CPU' allows moving weights off-device.",
            importance=ImportanceLevel.ADVANCED,
            order=1,
        )
    )

    # EMA CPU Only
    registry._add_field(
        ConfigField(
            name="ema_cpu_only",
            arg_name="--ema_cpu_only",
            ui_label="EMA on CPU Only",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="ema_config",
            default_value=False,
            dependencies=[FieldDependency(field="use_ema", operator="equals", value=True, action="show")],
            checkbox_label="Keep EMA on CPU only",
            help_text="Keep EMA weights exclusively on CPU even when ema_device would normally move them.",
            tooltip="Combine with ema_device=cpu to avoid shuttling weights; trades speed for lower VRAM use.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
        )
    )

    # EMA Update Interval
    registry._add_field(
        ConfigField(
            name="ema_update_interval",
            arg_name="--ema_update_interval",
            ui_label="EMA Update Interval",
            field_type=FieldType.NUMBER,
            tab="model",
            section="ema_config",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must update at least every step")],
            dependencies=[FieldDependency(field="use_ema", operator="equals", value=True, action="show")],
            help_text="Update EMA weights every N optimizer steps",
            tooltip="Higher values = faster training but less smooth EMA. Default: 10",
            importance=ImportanceLevel.ADVANCED,
            order=3,
        )
    )

    # EMA Foreach Disable
    registry._add_field(
        ConfigField(
            name="ema_foreach_disable",
            arg_name="--ema_foreach_disable",
            ui_label="Disable EMA Foreach Ops",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="ema_config",
            default_value=False,
            dependencies=[FieldDependency(field="use_ema", operator="equals", value=True, action="show")],
            checkbox_label="Disable torch.foreach",
            help_text="Fallback to standard tensor ops instead of torch.foreach updates.",
            tooltip="Enable if your hardware or backend has issues with torch.foreach kernels.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
        )
    )

    # EMA Decay
    registry._add_field(
        ConfigField(
            name="ema_decay",
            arg_name="--ema_decay",
            ui_label="EMA Decay",
            field_type=FieldType.NUMBER,
            tab="model",
            section="ema_config",
            default_value=0.995,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be positive"),
                ValidationRule(ValidationRuleType.MAX, value=0.9999, message="Must be less than 1"),
            ],
            dependencies=[FieldDependency(field="use_ema", operator="equals", value=True, action="show")],
            help_text="Smoothing factor for EMA updates (closer to 1.0 = slower drift).",
            tooltip="Common values: 0.99 for responsive EMA, 0.999 for very smooth outputs.",
            importance=ImportanceLevel.ADVANCED,
            order=5,
        )
    )
