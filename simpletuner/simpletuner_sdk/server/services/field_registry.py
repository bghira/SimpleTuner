"""Field Registry for SimpleTuner Configuration Parameters.

This module provides a comprehensive registry of all 261+ configuration parameters
with metadata, validation rules, and interdependencies.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable

from simpletuner.helpers.models.all import model_families


class FieldType(Enum):
    """UI field types."""
    TEXT = "text"
    NUMBER = "number"
    SELECT = "select"
    CHECKBOX = "checkbox"
    TEXTAREA = "textarea"
    PASSWORD = "password"
    FILE = "file"
    MULTI_SELECT = "multi_select"


class ImportanceLevel(Enum):
    """Field importance levels for progressive disclosure."""
    ESSENTIAL = "essential"
    IMPORTANT = "important"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"


class ValidationRuleType(Enum):
    """Types of validation rules."""
    REQUIRED = "required"
    MIN = "min"
    MAX = "max"
    PATTERN = "pattern"
    CHOICES = "choices"
    CUSTOM = "custom"
    PATH_EXISTS = "path_exists"
    DIVISIBLE_BY = "divisible_by"


@dataclass
class ValidationRule:
    """Validation rule for a field."""
    type: ValidationRuleType
    value: Any = None
    message: str = ""
    pattern: Optional[str] = None  # For PATTERN type validation
    condition: Optional[Dict[str, Any]] = None  # Only validate if condition met


@dataclass
class FieldDependency:
    """Dependency rule for field visibility/availability."""
    field: str
    value: Any = None
    values: Optional[List[Any]] = None  # Multiple acceptable values
    operator: str = "equals"  # equals, not_equals, in, not_in, greater_than, less_than
    action: str = "show"  # show, hide, enable, disable, set_value
    condition_met_value: Any = None  # Value when condition is met (for set_value)
    condition_not_met_value: Any = None  # Value when condition is not met
    target_value: Any = None  # Target value for set_value action


@dataclass
class ConfigField:
    """Configuration field metadata."""
    name: str  # UI field name
    arg_name: str  # CLI argument name (e.g., --model_type)
    ui_label: str
    field_type: FieldType
    tab: str
    section: str
    subsection: Optional[str] = None
    default_value: Any = None
    choices: Optional[List[Dict[str, Any]]] = None  # {"value": ..., "label": ...}
    validation_rules: List[ValidationRule] = field(default_factory=list)
    dependencies: List[FieldDependency] = field(default_factory=list)
    help_text: str = ""
    tooltip: str = ""
    placeholder: str = ""
    importance: ImportanceLevel = ImportanceLevel.IMPORTANT
    model_specific: Optional[List[str]] = None  # Which models this applies to
    platform_specific: Optional[List[str]] = None  # cuda, mps, etc.
    warning: Optional[str] = None  # For experimental features
    group: Optional[str] = None  # For grouping related fields
    order: int = 0  # Display order within section


class FieldRegistry:
    """Central registry for all configuration fields."""

    def __init__(self):
        self._fields: Dict[str, ConfigField] = {}
        self._dependencies_map: Dict[str, List[str]] = {}  # field -> dependent fields
        self._initialize_fields()

    def _initialize_fields(self):
        """Initialize all configuration fields."""
        # Model Configuration Fields
        self._add_model_config_fields()
        # Training Parameter Fields
        self._add_training_parameter_fields()
        # LoRA Configuration Fields
        self._add_lora_config_fields()
        # Data Configuration Fields
        self._add_data_config_fields()
        # Validation Fields
        self._add_validation_fields()
        # Advanced Configuration Fields
        self._add_advanced_config_fields()
        # Loss Function Fields
        self._add_loss_config_fields()
        # Optimizer Fields
        self._add_optimizer_fields()
        # Memory & Performance Fields
        self._add_memory_performance_fields()
        # Logging & Monitoring Fields
        self._add_logging_fields()

    def _add_field(self, field: ConfigField):
        """Add a field to the registry and update dependency maps."""
        self._fields[field.name] = field

        # Update dependency map
        for dep in field.dependencies:
            if dep.field not in self._dependencies_map:
                self._dependencies_map[dep.field] = []
            self._dependencies_map[dep.field].append(field.name)

    def _add_model_config_fields(self):
        """Add model configuration fields."""
        # Model Type - The critical missing field
        self._add_field(ConfigField(
            name="model_type",
            arg_name="--model_type",
            ui_label="Model Type",
            field_type=FieldType.SELECT,
            tab="basic",
            section="model_config",
            subsection="architecture",
            default_value="full",
            choices=[
                {"value": "full", "label": "Full Model Training"},
                {"value": "lora", "label": "LoRA (Low-Rank Adaptation)"}
            ],
            validation_rules=[
                ValidationRule(ValidationRuleType.REQUIRED, message="Model type is required"),
                ValidationRule(ValidationRuleType.CHOICES, value=["full", "lora"])
            ],
            help_text="Choose between full model training or LoRA adapter training",
            tooltip="Full training updates all model weights. LoRA only trains small adapter matrices, using less memory and producing smaller files.",
            importance=ImportanceLevel.ESSENTIAL,
            order=1
        ))

        # Model Family
        model_family_list = list(model_families.keys())
        self._add_field(ConfigField(
            name="model_family",
            arg_name="--model_family",
            ui_label="Model Family",
            field_type=FieldType.SELECT,
            tab="basic",
            section="model_config",
            subsection="architecture",
            choices=[{"value": f, "label": f.upper()} for f in model_family_list],
            validation_rules=[
                ValidationRule(ValidationRuleType.REQUIRED, message="Model family is required"),
                ValidationRule(ValidationRuleType.CHOICES, value=model_family_list)
            ],
            help_text="The base model architecture family to train",
            tooltip="Different model families have different capabilities and requirements",
            importance=ImportanceLevel.ESSENTIAL,
            order=2
        ))

        # Model Flavour
        self._add_field(ConfigField(
            name="model_flavour",
            arg_name="--model_flavour",
            ui_label="Model Flavour",
            field_type=FieldType.SELECT,
            tab="basic",
            section="model_config",
            subsection="architecture",
            default_value="",
            choices=[],  # Dynamic based on model_family
            dependencies=[
                FieldDependency(field="model_family", operator="not_equals", value="")
            ],
            help_text="Specific variant of the selected model family",
            tooltip="Some models have multiple variants with different sizes or capabilities",
            importance=ImportanceLevel.IMPORTANT,
            order=3
        ))

        # Pretrained Model Path
        self._add_field(ConfigField(
            name="pretrained_model_name_or_path",
            arg_name="--pretrained_model_name_or_path",
            ui_label="Base Model Path",
            field_type=FieldType.TEXT,
            tab="basic",
            section="model_config",
            subsection="paths",
            placeholder="black-forest-labs/FLUX.1-dev",
            validation_rules=[
                ValidationRule(ValidationRuleType.REQUIRED, message="Model path is required")
            ],
            help_text="Hugging Face model ID or local path to the base model",
            tooltip="Can be a HuggingFace model ID (e.g., 'stabilityai/stable-diffusion-xl-base-1.0') or a local directory path",
            importance=ImportanceLevel.ESSENTIAL,
            order=4
        ))

        # Output Directory
        self._add_field(ConfigField(
            name="output_dir",
            arg_name="--output_dir",
            ui_label="Output Directory",
            field_type=FieldType.TEXT,
            tab="basic",
            section="essential_settings",
            subsection="paths",
            default_value="./output",
            validation_rules=[
                ValidationRule(ValidationRuleType.REQUIRED, message="Output directory is required")
            ],
            help_text="Directory where model checkpoints and logs will be saved",
            tooltip="All training outputs including checkpoints, logs, and samples will be saved here",
            importance=ImportanceLevel.ESSENTIAL,
            order=5
        ))

        # Project Name
        self._add_field(ConfigField(
            name="project_name",
            arg_name="--project_name",
            ui_label="Project Name",
            field_type=FieldType.TEXT,
            tab="basic",
            section="essential_settings",
            subsection="project",
            placeholder="my-training-run",
            validation_rules=[
                ValidationRule(ValidationRuleType.PATTERN, value=r"^[a-zA-Z0-9_-]+$",
                             message="Project name can only contain letters, numbers, hyphens, and underscores")
            ],
            help_text="Name for this training run (used in logging and outputs)",
            tooltip="A descriptive name for your training run. Used in file names and logging.",
            importance=ImportanceLevel.ESSENTIAL,
            order=6
        ))

        # Logging Directory
        self._add_field(ConfigField(
            name="logging_dir",
            arg_name="--logging_dir",
            ui_label="Logging Directory",
            field_type=FieldType.TEXT,
            tab="basic",
            section="essential_settings",
            subsection="paths",
            default_value="./logs",
            help_text="Directory for TensorBoard and other logs",
            tooltip="Training metrics and TensorBoard logs will be saved here",
            importance=ImportanceLevel.IMPORTANT,
            order=7
        ))

        # Mixed Precision
        self._add_field(ConfigField(
            name="mixed_precision",
            arg_name="--mixed_precision",
            ui_label="Mixed Precision",
            field_type=FieldType.SELECT,
            tab="basic",
            section="hardware_config",
            default_value="bf16",
            choices=[
                {"value": "no", "label": "Disabled (FP32)"},
                {"value": "fp16", "label": "FP16 (Half Precision)"},
                {"value": "bf16", "label": "BF16 (Brain Float)"},
                {"value": "fp8-e4m3fn", "label": "FP8 E4M3 (H100/Ada)"},
                {"value": "fp8-e5m2", "label": "FP8 E5M2 (H100/Ada)"}
            ],
            help_text="Precision mode for training (affects memory usage and speed)",
            tooltip="BF16 is recommended for most modern GPUs. FP16 saves memory but may have stability issues. FP8 requires H100 or newer.",
            importance=ImportanceLevel.IMPORTANT,
            order=8,
            dependencies=[
                FieldDependency(
                    field="model_family",
                    operator="equals",
                    value="flux",
                    action="show",
                    condition_met_value="fp8-e4m3fn",
                    condition_not_met_value="bf16"
                )
            ]
        ))

        # Random Seed
        self._add_field(ConfigField(
            name="seed",
            arg_name="--seed",
            ui_label="Random Seed",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="hardware_config",
            default_value=42,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Seed must be non-negative"),
                ValidationRule(ValidationRuleType.MAX, value=2147483647, message="Seed must fit in 32-bit integer")
            ],
            help_text="Random seed for reproducible training",
            tooltip="Use the same seed to get reproducible results across training runs",
            importance=ImportanceLevel.ADVANCED,
            order=9
        ))

        # Resolution
        self._add_field(ConfigField(
            name="resolution",
            arg_name="--resolution",
            ui_label="Training Resolution",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="training_config",
            default_value=1024,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=128, message="Resolution too low"),
                ValidationRule(ValidationRuleType.MAX, value=2048, message="Resolution too high for most GPUs"),
                ValidationRule(ValidationRuleType.DIVISIBLE_BY, value=8, message="Resolution must be divisible by 8")
            ],
            help_text="Image resolution for training (width and height)",
            tooltip="Higher resolutions require more VRAM. SD 1.5: 512, SDXL: 1024, Flux: 1024+",
            importance=ImportanceLevel.ESSENTIAL,
            order=10,
            dependencies=[
                FieldDependency(
                    field="model_family",
                    operator="equals",
                    value="sd15",
                    action="set_value",
                    target_value=512
                ),
                FieldDependency(
                    field="model_family",
                    operator="in",
                    values=["sdxl", "flux", "sd3"],
                    action="set_value",
                    target_value=1024
                )
            ]
        ))

        # Resume from Checkpoint
        self._add_field(ConfigField(
            name="resume_from_checkpoint",
            arg_name="--resume_from_checkpoint",
            ui_label="Resume From Checkpoint",
            field_type=FieldType.TEXT,
            tab="basic",
            section="training_config",
            placeholder="latest",
            help_text="Path to checkpoint to resume training from",
            tooltip="Use 'latest' to auto-resume from the most recent checkpoint, or provide a specific checkpoint path",
            importance=ImportanceLevel.ADVANCED,
            order=11
        ))

        # Prediction Type
        self._add_field(ConfigField(
            name="prediction_type",
            arg_name="--prediction_type",
            ui_label="Prediction Type",
            field_type=FieldType.SELECT,
            tab="model",
            section="architecture",
            default_value="",
            choices=[
                {"value": "", "label": "Auto-detect"},
                {"value": "epsilon", "label": "Epsilon"},
                {"value": "v_prediction", "label": "V-Prediction"},
                {"value": "sample", "label": "Sample"},
                {"value": "flow_matching", "label": "Flow Matching"}
            ],
            help_text="The parameterization type for the diffusion model",
            tooltip="Usually auto-detected from the model. Flow matching is used by Flux, SD3, and similar models.",
            importance=ImportanceLevel.ADVANCED,
            order=10
        ))

        # VAE Path
        self._add_field(ConfigField(
            name="vae_path",
            arg_name="--vae_path",
            ui_label="Custom VAE Path",
            field_type=FieldType.TEXT,
            tab="model",
            section="vae_config",
            placeholder="madebyollin/sdxl-vae-fp16-fix",
            help_text="Optional: Override the default VAE with a custom one",
            tooltip="Can be a HuggingFace model ID or local path to a custom VAE",
            importance=ImportanceLevel.ADVANCED,
            order=11
        ))

        # VAE Dtype
        self._add_field(ConfigField(
            name="vae_dtype",
            arg_name="--vae_dtype",
            ui_label="VAE Precision",
            field_type=FieldType.SELECT,
            tab="model",
            section="vae_config",
            default_value="default",
            choices=[
                {"value": "default", "label": "Default (Match training)"},
                {"value": "fp32", "label": "FP32"},
                {"value": "fp16", "label": "FP16"},
                {"value": "bf16", "label": "BF16"}
            ],
            help_text="Precision for VAE encoding/decoding. Lower precision saves memory.",
            tooltip="FP16/BF16 can reduce memory usage but may slightly affect image quality",
            importance=ImportanceLevel.ADVANCED,
            order=12
        ))

        # VAE Cache Preprocessing
        self._add_field(ConfigField(
            name="vae_cache_preprocess",
            arg_name="--vae_cache_preprocess",
            ui_label="Enable VAE Cache Preprocessing",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="vae_config",
            default_value=False,
            help_text="Pre-encode images with VAE to speed up training",
            tooltip="Requires additional disk space but significantly speeds up training",
            importance=ImportanceLevel.ADVANCED,
            order=13
        ))

        # Base Model Precision
        self._add_field(ConfigField(
            name="base_model_precision",
            arg_name="--base_model_precision",
            ui_label="Base Model Precision",
            field_type=FieldType.SELECT,
            tab="model",
            section="quantization",
            default_value="no_change",
            choices=[
                {"value": "no_change", "label": "No Change"},
                {"value": "int8-quanto", "label": "INT8 (Quanto)"},
                {"value": "int4-quanto", "label": "INT4 (Quanto)"},
                {"value": "int2-quanto", "label": "INT2 (Quanto)"},
                {"value": "fp8-quanto", "label": "FP8 (Quanto)"},
                {"value": "nf4-bnb", "label": "NF4 (BitsAndBytes)"}
            ],
            help_text="Precision for loading the base model. Lower precision saves memory.",
            tooltip="Quantization reduces memory usage but may impact quality. INT8/INT4 are commonly used.",
            importance=ImportanceLevel.ADVANCED,
            order=14,
            dependencies=[
                FieldDependency(
                    field="model_type",
                    operator="equals",
                    value="lora",
                    action="enable"
                )
            ]
        ))

        # Text Encoder Precision
        self._add_field(ConfigField(
            name="text_encoder_1_precision",
            arg_name="--text_encoder_1_precision",
            ui_label="Text Encoder Precision",
            field_type=FieldType.SELECT,
            tab="model",
            section="quantization",
            default_value="no_change",
            choices=[
                {"value": "no_change", "label": "No Change"},
                {"value": "int8-quanto", "label": "INT8 (Quanto)"},
                {"value": "int4-quanto", "label": "INT4 (Quanto)"}
            ],
            help_text="Precision for text encoders. Lower precision saves memory.",
            tooltip="Text encoder quantization has minimal impact on quality",
            importance=ImportanceLevel.ADVANCED,
            order=15
        ))

    def _add_training_parameter_fields(self):
        """Add training parameter fields."""
        # Number of Training Epochs
        self._add_field(ConfigField(
            name="num_train_epochs",
            arg_name="--num_train_epochs",
            ui_label="Number of Epochs",
            field_type=FieldType.NUMBER,
            tab="training",
            section="training_schedule",
            default_value=1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Epochs must be non-negative"),
                ValidationRule(ValidationRuleType.MAX, value=1000, message="Consider if you really need >1000 epochs")
            ],
            help_text="Number of times to iterate through the entire dataset",
            tooltip="One epoch = one full pass through all training data. More epochs can improve quality but may cause overfitting.",
            importance=ImportanceLevel.ESSENTIAL,
            order=1
        ))

        # Max Training Steps
        self._add_field(ConfigField(
            name="max_train_steps",
            arg_name="--max_train_steps",
            ui_label="Max Training Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="training_schedule",
            default_value=0,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Steps must be non-negative")
            ],
            help_text="Maximum number of training steps (0 = use epochs instead)",
            tooltip="If set to a positive value, training will stop after this many steps regardless of epochs",
            importance=ImportanceLevel.IMPORTANT,
            order=2
        ))

        # Batch Size
        self._add_field(ConfigField(
            name="train_batch_size",
            arg_name="--train_batch_size",
            ui_label="Batch Size",
            field_type=FieldType.NUMBER,
            tab="training",
            section="training_schedule",
            default_value=4,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Batch size must be at least 1"),
                ValidationRule(ValidationRuleType.MAX, value=128, message="Batch size >128 is unusual")
            ],
            help_text="Number of images to process in each training step",
            tooltip="Higher batch sizes can improve training stability but require more VRAM. Start with 1-4 for most GPUs.",
            importance=ImportanceLevel.ESSENTIAL,
            order=3
        ))

        # Learning Rate
        self._add_field(ConfigField(
            name="learning_rate",
            arg_name="--learning_rate",
            ui_label="Learning Rate",
            field_type=FieldType.NUMBER,
            tab="training",
            section="learning_rate",
            default_value=4e-7,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Learning rate must be positive"),
                ValidationRule(ValidationRuleType.MAX, value=1, message="Learning rate >1 is extremely high")
            ],
            help_text="Base learning rate for training",
            tooltip="Controls how much model weights change per step. Lower = more stable but slower. Typical range: 1e-6 to 1e-4",
            importance=ImportanceLevel.ESSENTIAL,
            order=1
        ))

        # Optimizer
        optimizer_choices = [
            "adamw_bf16", "ao-adamw8bit", "ao-adamw4bit", "ao-adamwfp8", "optimi-adamw",
            "optimi-lion", "optimi-stableadamw", "prodigy", "soap", "dadaptation", "dadaptsgd",
            "dadaptadam", "dadaptlion", "adafactor", "adamw8bit", "adamw", "adam8bit", "adam",
            "lion8bit", "lion", "rmsprop", "sgd", "StableAdamWUnfused", "deepspeed-adamw"
        ]
        self._add_field(ConfigField(
            name="optimizer",
            arg_name="--optimizer",
            ui_label="Optimizer",
            field_type=FieldType.SELECT,
            tab="basic",
            section="training_essentials",
            choices=[{"value": opt, "label": opt} for opt in optimizer_choices],
            validation_rules=[
                ValidationRule(ValidationRuleType.REQUIRED, message="Optimizer is required"),
                ValidationRule(ValidationRuleType.CHOICES, value=optimizer_choices)
            ],
            help_text="Optimization algorithm for training",
            tooltip="AdamW variants are most common. 8-bit versions save memory. Prodigy auto-adjusts learning rate.",
            importance=ImportanceLevel.ESSENTIAL,
            order=5
        ))

        # LR Scheduler
        lr_scheduler_choices = [
            "linear", "sine", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ]
        self._add_field(ConfigField(
            name="lr_scheduler",
            arg_name="--lr_scheduler",
            ui_label="Learning Rate Scheduler",
            field_type=FieldType.SELECT,
            tab="training",
            section="learning_rate",
            default_value="sine",
            choices=[{"value": s, "label": s.replace('_', ' ').title()} for s in lr_scheduler_choices],
            validation_rules=[
                ValidationRule(ValidationRuleType.CHOICES, value=lr_scheduler_choices)
            ],
            help_text="How learning rate changes during training",
            tooltip="Sine and cosine gradually reduce LR. Constant keeps it fixed. Warmup helps stability at start.",
            importance=ImportanceLevel.IMPORTANT,
            order=2
        ))

        # Gradient Accumulation Steps
        self._add_field(ConfigField(
            name="gradient_accumulation_steps",
            arg_name="--gradient_accumulation_steps",
            ui_label="Gradient Accumulation Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="training_schedule",
            default_value=1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")
            ],
            help_text="Number of steps to accumulate gradients before updating",
            tooltip="Simulates larger batch sizes without using more VRAM. Effective batch = batch_size * accumulation_steps",
            importance=ImportanceLevel.IMPORTANT,
            order=4
        ))

        # LR Warmup Steps
        self._add_field(ConfigField(
            name="lr_warmup_steps",
            arg_name="--lr_warmup_steps",
            ui_label="Learning Rate Warmup Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="learning_rate",
            default_value=0,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")
            ],
            help_text="Number of steps to gradually increase LR from 0",
            tooltip="Helps training stability at start. Typically 5-10% of total steps",
            importance=ImportanceLevel.ADVANCED,
            order=3
        ))

        # Save Steps
        self._add_field(ConfigField(
            name="save_every_n_steps",
            arg_name="--save_every_n_steps",
            ui_label="Save Checkpoint Every N Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="checkpointing",
            default_value=500,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative (0 disables)")
            ],
            help_text="How often to save model checkpoints",
            tooltip="Set to 0 to only save at end. Lower values = more frequent saves but more disk usage",
            importance=ImportanceLevel.IMPORTANT,
            order=1
        ))

        # Max Checkpoints
        self._add_field(ConfigField(
            name="save_total_limit",
            arg_name="--save_total_limit",
            ui_label="Maximum Checkpoints to Keep",
            field_type=FieldType.NUMBER,
            tab="training",
            section="checkpointing",
            default_value=10,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative (0 = unlimited)")
            ],
            help_text="Maximum number of checkpoints to keep on disk",
            tooltip="Older checkpoints are deleted when limit is exceeded. Set to 0 for unlimited",
            importance=ImportanceLevel.ADVANCED,
            order=2
        ))

        # Gradient Checkpointing
        self._add_field(ConfigField(
            name="gradient_checkpointing",
            arg_name="--gradient_checkpointing",
            ui_label="Enable Gradient Checkpointing",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="memory_optimization",
            default_value=False,
            help_text="Trade compute for memory by recomputing activations",
            tooltip="Reduces VRAM usage significantly but increases training time by ~20%",
            importance=ImportanceLevel.ADVANCED,
            order=1
        ))

        # Train Text Encoder
        self._add_field(ConfigField(
            name="train_text_encoder",
            arg_name="--train_text_encoder",
            ui_label="Train Text Encoder",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="training_components",
            default_value=False,
            help_text="Also train the text encoder (CLIP) model",
            tooltip="Can improve concept learning but uses more VRAM. Not recommended for LoRA",
            importance=ImportanceLevel.ADVANCED,
            order=1,
            dependencies=[
                FieldDependency(
                    field="model_type",
                    operator="equals",
                    value="full",
                    action="enable"
                )
            ]
        ))

        # Text Encoder LR
        self._add_field(ConfigField(
            name="text_encoder_lr",
            arg_name="--text_encoder_lr",
            ui_label="Text Encoder Learning Rate",
            field_type=FieldType.NUMBER,
            tab="training",
            section="learning_rate",
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Must be positive")
            ],
            help_text="Separate learning rate for text encoder",
            tooltip="Usually lower than main LR. If not set, uses main learning rate",
            importance=ImportanceLevel.ADVANCED,
            order=4,
            dependencies=[
                FieldDependency(
                    field="train_text_encoder",
                    operator="equals",
                    value=True,
                    action="show"
                )
            ]
        ))

    def _add_lora_config_fields(self):
        """Add LoRA configuration fields."""
        # LoRA Rank
        self._add_field(ConfigField(
            name="lora_rank",
            arg_name="--lora_rank",
            ui_label="LoRA Rank",
            field_type=FieldType.NUMBER,
            tab="model",
            section="lora_config",
            subsection="basic",
            default_value=16,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="LoRA rank must be at least 1"),
                ValidationRule(ValidationRuleType.MAX, value=256, message="LoRA rank >256 is very large")
            ],
            dependencies=[
                FieldDependency(field="model_type", value="lora")
            ],
            help_text="Dimension of LoRA update matrices",
            tooltip="Higher rank = more capacity but larger files. Common values: 4-32 for style, 64-128 for complex concepts",
            importance=ImportanceLevel.IMPORTANT,
            order=1
        ))

        # LoRA Alpha
        self._add_field(ConfigField(
            name="lora_alpha",
            arg_name="--lora_alpha",
            ui_label="LoRA Alpha",
            field_type=FieldType.NUMBER,
            tab="model",
            section="lora_config",
            subsection="basic",
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.1, message="LoRA alpha should be positive")
            ],
            dependencies=[
                FieldDependency(field="model_type", value="lora")
            ],
            help_text="Scaling factor for LoRA updates",
            tooltip="Usually set equal to rank. Controls the magnitude of LoRA's effect.",
            importance=ImportanceLevel.IMPORTANT,
            order=2
        ))

        # LoRA Type
        self._add_field(ConfigField(
            name="lora_type",
            arg_name="--lora_type",
            ui_label="LoRA Type",
            field_type=FieldType.SELECT,
            tab="model",
            section="lora_config",
            subsection="basic",
            default_value="standard",
            choices=[
                {"value": "standard", "label": "Standard PEFT LoRA"},
                {"value": "lycoris", "label": "LyCORIS (LoKr/LoHa/etc)"}
            ],
            dependencies=[
                FieldDependency(field="model_type", value="lora")
            ],
            help_text="LoRA implementation type",
            tooltip="Standard LoRA is most common. LyCORIS offers advanced decomposition methods.",
            importance=ImportanceLevel.ADVANCED,
            order=3
        ))

        # Flux LoRA Target
        flux_targets = [
            "mmdit", "context", "context+ffs", "all", "all+ffs", "ai-toolkit",
            "tiny", "nano", "controlnet", "all+ffs+embedder", "all+ffs+embedder+controlnet"
        ]
        self._add_field(ConfigField(
            name="flux_lora_target",
            arg_name="--flux_lora_target",
            ui_label="Flux LoRA Target Layers",
            field_type=FieldType.SELECT,
            tab="model",
            section="lora_config",
            subsection="model_specific",
            default_value="all",
            choices=[{"value": t, "label": t} for t in flux_targets],
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="model_family", value="flux")
            ],
            help_text="Which layers to train in Flux models",
            tooltip="'all' trains all attention layers. 'context' only trains text layers. '+ffs' includes feed-forward layers.",
            importance=ImportanceLevel.ADVANCED,
            model_specific=["flux"],
            order=10
        ))

        # Use DoRA
        self._add_field(ConfigField(
            name="use_dora",
            arg_name="--use_dora",
            ui_label="Use DoRA",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="lora_config",
            subsection="advanced",
            default_value=False,
            dependencies=[
                FieldDependency(field="model_type", value="lora")
            ],
            help_text="Enable DoRA (Weight-Decomposed LoRA)",
            tooltip="Experimental feature that decomposes weights into magnitude and direction. May improve quality but slower.",
            warning="Experimental feature - may slow down training",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=20
        ))

    def _add_data_config_fields(self):
        """Add data configuration fields."""
        # Resolution
        self._add_field(ConfigField(
            name="resolution",
            arg_name="--resolution",
            ui_label="Training Resolution",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="training_essentials",
            default_value=1024,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=256, message="Resolution must be at least 256"),
                ValidationRule(ValidationRuleType.MAX, value=4096, message="Resolution >4096 is very high"),
                ValidationRule(ValidationRuleType.DIVISIBLE_BY, value=64, message="Resolution must be divisible by 64")
            ],
            help_text="Resolution for training images",
            tooltip="All images will be resized to this resolution. Must be divisible by 64. Higher = better quality but more VRAM.",
            importance=ImportanceLevel.ESSENTIAL,
            order=3
        ))

        # Resolution Type
        self._add_field(ConfigField(
            name="resolution_type",
            arg_name="--resolution_type",
            ui_label="Resolution Type",
            field_type=FieldType.SELECT,
            tab="data",
            section="image_processing",
            subsection="resolution",
            default_value="pixel_area",
            choices=[
                {"value": "pixel", "label": "Pixel (shortest edge)"},
                {"value": "area", "label": "Area (megapixels)"},
                {"value": "pixel_area", "label": "Pixel as Area"}
            ],
            help_text="How to interpret the resolution value",
            tooltip="Pixel: resize shortest edge. Area: total pixel count. Pixel as Area: converts pixel to megapixels.",
            importance=ImportanceLevel.ADVANCED,
            order=2
        ))

        # Data Backend Config
        self._add_field(ConfigField(
            name="data_backend_config",
            arg_name="--data_backend_config",
            ui_label="Data Backend Config",
            field_type=FieldType.FILE,
            tab="basic",
            section="data_config",
            placeholder="config/multidatabackend.json",
            validation_rules=[
                ValidationRule(ValidationRuleType.REQUIRED, message="Data backend configuration is required")
            ],
            help_text="Path to multi-databackend configuration JSON",
            tooltip="Defines your training datasets, captions, and data loading settings",
            importance=ImportanceLevel.ESSENTIAL,
            order=1
        ))

        # Caption Strategy
        self._add_field(ConfigField(
            name="caption_strategy",
            arg_name="--caption_strategy",
            ui_label="Caption Strategy",
            field_type=FieldType.SELECT,
            tab="data",
            section="dataset_config",
            default_value="filename",
            choices=[
                {"value": "filename", "label": "Filename"},
                {"value": "textfile", "label": "Text Files"},
                {"value": "instance_prompt", "label": "Instance Prompt"},
                {"value": "parquet", "label": "Parquet Dataset"}
            ],
            help_text="How to load captions for images",
            tooltip="Filename: use image name. Textfile: .txt files. Parquet: structured dataset files.",
            importance=ImportanceLevel.IMPORTANT,
            order=2
        ))

        # VAE Batch Size
        self._add_field(ConfigField(
            name="vae_batch_size",
            arg_name="--vae_batch_size",
            ui_label="VAE Batch Size",
            field_type=FieldType.NUMBER,
            tab="advanced",
            section="memory_performance",
            subsection="vae_settings",
            default_value=4,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="VAE batch size must be at least 1")
            ],
            help_text="Batch size for VAE encoding during caching",
            tooltip="Higher values speed up VAE caching but use more VRAM. Reduce if getting OOM during cache creation.",
            importance=ImportanceLevel.ADVANCED,
            order=1
        ))

        # Caption Dropout
        self._add_field(ConfigField(
            name="caption_dropout_probability",
            arg_name="--caption_dropout_probability",
            ui_label="Caption Dropout Probability",
            field_type=FieldType.NUMBER,
            tab="data",
            section="caption_processing",
            default_value=0.0,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1, message="Must be between 0 and 1")
            ],
            help_text="Probability of dropping captions during training",
            tooltip="Helps model learn from images alone. 0.1 = 10% chance of no caption",
            importance=ImportanceLevel.ADVANCED,
            order=1
        ))

        # Image Transforms
        self._add_field(ConfigField(
            name="image_transform_steps",
            arg_name="--image_transform_steps",
            ui_label="Image Transform Steps",
            field_type=FieldType.TEXT,
            tab="data",
            section="image_processing",
            subsection="augmentation",
            placeholder="crop,resize",
            help_text="Comma-separated list of image transforms to apply",
            tooltip="Options: crop, resize, rotate, flip, color_jitter",
            importance=ImportanceLevel.ADVANCED,
            order=3
        ))

        # Aspect Ratio Bucketing
        self._add_field(ConfigField(
            name="aspect_ratio_bucketing",
            arg_name="--aspect_ratio_bucketing",
            ui_label="Enable Aspect Ratio Bucketing",
            field_type=FieldType.CHECKBOX,
            tab="data",
            section="image_processing",
            subsection="bucketing",
            default_value=True,
            help_text="Group images by aspect ratio to reduce padding",
            tooltip="Improves training efficiency and quality for datasets with varied aspect ratios",
            importance=ImportanceLevel.IMPORTANT,
            order=1
        ))

        # Minimum Aspect Ratio
        self._add_field(ConfigField(
            name="aspect_ratio_bucket_min",
            arg_name="--aspect_ratio_bucket_min",
            ui_label="Minimum Aspect Ratio",
            field_type=FieldType.NUMBER,
            tab="data",
            section="image_processing",
            subsection="bucketing",
            default_value=0.5,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.1, message="Must be greater than 0.1"),
                ValidationRule(ValidationRuleType.MAX, value=2.0, message="Must be less than 2.0")
            ],
            help_text="Minimum aspect ratio for bucketing (width/height)",
            tooltip="Images narrower than this will be padded or cropped",
            importance=ImportanceLevel.ADVANCED,
            order=2,
            dependencies=[
                FieldDependency(
                    field="aspect_ratio_bucketing",
                    operator="equals",
                    value=True,
                    action="show"
                )
            ]
        ))

        # Maximum Aspect Ratio
        self._add_field(ConfigField(
            name="aspect_ratio_bucket_max",
            arg_name="--aspect_ratio_bucket_max",
            ui_label="Maximum Aspect Ratio",
            field_type=FieldType.NUMBER,
            tab="data",
            section="image_processing",
            subsection="bucketing",
            default_value=2.0,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.5, message="Must be greater than 0.5"),
                ValidationRule(ValidationRuleType.MAX, value=3.0, message="Must be less than 3.0")
            ],
            help_text="Maximum aspect ratio for bucketing (width/height)",
            tooltip="Images wider than this will be padded or cropped",
            importance=ImportanceLevel.ADVANCED,
            order=3,
            dependencies=[
                FieldDependency(
                    field="aspect_ratio_bucketing",
                    operator="equals",
                    value=True,
                    action="show"
                )
            ]
        ))

        # Repeats
        self._add_field(ConfigField(
            name="repeats",
            arg_name="--repeats",
            ui_label="Dataset Repeats",
            field_type=FieldType.NUMBER,
            tab="data",
            section="dataset_config",
            default_value=1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")
            ],
            help_text="Number of times to repeat the dataset per epoch",
            tooltip="Useful for small datasets. Effectively multiplies dataset size",
            importance=ImportanceLevel.ADVANCED,
            order=1
        ))

        # Maximum Token Length
        self._add_field(ConfigField(
            name="maximum_caption_length",
            arg_name="--maximum_caption_length",
            ui_label="Maximum Caption Length",
            field_type=FieldType.NUMBER,
            tab="data",
            section="caption_processing",
            default_value=120,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1"),
                ValidationRule(ValidationRuleType.MAX, value=1024, message="Maximum reasonable length is 1024")
            ],
            help_text="Maximum number of tokens in captions",
            tooltip="Longer captions will be truncated. SD models typically use 77, SDXL/Flux can use more",
            importance=ImportanceLevel.ADVANCED,
            order=2
        ))

    def _add_validation_fields(self):
        """Add validation configuration fields."""
        # Validation Steps
        self._add_field(ConfigField(
            name="validation_steps",
            arg_name="--validation_steps",
            ui_label="Validation Steps",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_schedule",
            default_value=100,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Validation steps must be positive")
            ],
            help_text="Run validation every N training steps",
            tooltip="How often to generate validation images during training. Lower = more frequent validation.",
            importance=ImportanceLevel.IMPORTANT,
            order=1
        ))

        # Validation Prompt
        self._add_field(ConfigField(
            name="validation_prompt",
            arg_name="--validation_prompt",
            ui_label="Validation Prompt",
            field_type=FieldType.TEXTAREA,
            tab="validation",
            section="prompt_management",
            placeholder="a photo of a cat, highly detailed, 4k",
            help_text="Prompt to use for validation images",
            tooltip="This prompt will be used to generate images during training to monitor progress",
            importance=ImportanceLevel.IMPORTANT,
            order=1
        ))

        # Number of Validation Images
        self._add_field(ConfigField(
            name="num_validation_images",
            arg_name="--num_validation_images",
            ui_label="Validation Images Count",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_schedule",
            default_value=1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Must generate at least 1 validation image"),
                ValidationRule(ValidationRuleType.MAX, value=10, message="Generating >10 images may slow training")
            ],
            help_text="Number of images to generate per validation",
            tooltip="More images give better sense of model performance but take longer to generate",
            importance=ImportanceLevel.ADVANCED,
            order=2
        ))

        # Validation Guidance Scale
        self._add_field(ConfigField(
            name="validation_guidance",
            arg_name="--validation_guidance",
            ui_label="Guidance Scale",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="generation_settings",
            subsection="guidance",
            default_value=7.5,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Guidance must be at least 1"),
                ValidationRule(ValidationRuleType.MAX, value=30, message="Guidance >30 may cause artifacts")
            ],
            help_text="CFG guidance scale for validation images",
            tooltip="Higher values follow prompt more closely. 7-12 is typical. Set to 1 for distilled models.",
            importance=ImportanceLevel.ADVANCED,
            order=1
        ))

    def _add_advanced_config_fields(self):
        """Add advanced configuration fields."""
        # Mixed Precision
        self._add_field(ConfigField(
            name="mixed_precision",
            arg_name="--mixed_precision",
            ui_label="Mixed Precision",
            field_type=FieldType.SELECT,
            tab="advanced",
            section="memory_performance",
            subsection="precision",
            default_value="bf16",
            choices=[
                {"value": "no", "label": "No (FP32)"},
                {"value": "fp16", "label": "FP16"},
                {"value": "bf16", "label": "BF16 (Recommended)"},
                {"value": "fp8", "label": "FP8 (Experimental)"}
            ],
            help_text="Precision for training computations",
            tooltip="BF16 is recommended for stability. FP16 saves memory but less stable. FP8 is experimental.",
            importance=ImportanceLevel.IMPORTANT,
            order=1
        ))

        # Gradient Checkpointing
        self._add_field(ConfigField(
            name="gradient_checkpointing",
            arg_name="--gradient_checkpointing",
            ui_label="Gradient Checkpointing",
            field_type=FieldType.CHECKBOX,
            tab="advanced",
            section="memory_performance",
            subsection="memory_optimization",
            default_value=False,
            help_text="Trade compute for memory during training",
            tooltip="Reduces VRAM usage by recomputing activations. Slows training by ~20% but can enable larger batch sizes.",
            importance=ImportanceLevel.ADVANCED,
            order=1
        ))

        # Attention Mechanism
        attention_mechanisms = [
            "diffusers", "xformers", "sageattention", "sageattention-int8-fp16-triton",
            "sageattention-int8-fp16-cuda", "sageattention-int8-fp8-cuda"
        ]
        self._add_field(ConfigField(
            name="attention_mechanism",
            arg_name="--attention_mechanism",
            ui_label="Attention Implementation",
            field_type=FieldType.SELECT,
            tab="advanced",
            section="memory_performance",
            subsection="attention",
            default_value="diffusers",
            choices=[{"value": a, "label": a} for a in attention_mechanisms],
            help_text="Attention computation backend",
            tooltip="Xformers saves memory. SageAttention is faster but experimental. Diffusers is default.",
            importance=ImportanceLevel.ADVANCED,
            order=1
        ))

    def _add_loss_config_fields(self):
        """Add loss function configuration fields."""
        # Loss Type
        self._add_field(ConfigField(
            name="loss_type",
            arg_name="--loss_type",
            ui_label="Loss Function",
            field_type=FieldType.SELECT,
            tab="training",
            section="loss_config",
            default_value="l2",
            choices=[
                {"value": "l2", "label": "L2 (MSE)"},
                {"value": "huber", "label": "Huber"},
                {"value": "smooth_l1", "label": "Smooth L1"}
            ],
            help_text="Loss function for training",
            tooltip="L2 is standard. Huber/Smooth L1 are more robust to outliers but may train differently.",
            importance=ImportanceLevel.ADVANCED,
            order=1
        ))

        # SNR Gamma
        self._add_field(ConfigField(
            name="snr_gamma",
            arg_name="--snr_gamma",
            ui_label="SNR Gamma",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_config",
            subsection="snr_weighting",
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="SNR gamma must be non-negative")
            ],
            help_text="SNR weighting gamma value (0 = disabled)",
            tooltip="Rebalances loss across timesteps. Recommended value: 5.0. Helps prevent overtraining on easy timesteps.",
            importance=ImportanceLevel.ADVANCED,
            order=2
        ))

    def _add_optimizer_fields(self):
        """Add optimizer configuration fields."""
        # Adam Beta1
        self._add_field(ConfigField(
            name="adam_beta1",
            arg_name="--adam_beta1",
            ui_label="Adam Beta 1",
            field_type=FieldType.NUMBER,
            tab="training",
            section="optimizer_config",
            subsection="adam_params",
            default_value=0.9,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Beta1 must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1, message="Beta1 must be between 0 and 1")
            ],
            dependencies=[
                FieldDependency(field="optimizer", values=["adamw", "adamw_bf16", "adamw8bit", "adam", "adam8bit"])
            ],
            help_text="First moment decay rate for Adam optimizers",
            tooltip="Controls momentum. Default 0.9 works well. Lower values reduce momentum.",
            importance=ImportanceLevel.ADVANCED,
            order=1
        ))

        # Adam Beta2
        self._add_field(ConfigField(
            name="adam_beta2",
            arg_name="--adam_beta2",
            ui_label="Adam Beta 2",
            field_type=FieldType.NUMBER,
            tab="training",
            section="optimizer_config",
            subsection="adam_params",
            default_value=0.999,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Beta2 must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1, message="Beta2 must be between 0 and 1")
            ],
            dependencies=[
                FieldDependency(field="optimizer", values=["adamw", "adamw_bf16", "adamw8bit", "adam", "adam8bit"])
            ],
            help_text="Second moment decay rate for Adam optimizers",
            tooltip="Controls adaptive learning rates. Default 0.999 is standard. Lower values make adaptation more aggressive.",
            importance=ImportanceLevel.ADVANCED,
            order=2
        ))

    def _add_memory_performance_fields(self):
        """Add memory and performance configuration fields."""
        # Gradient Accumulation Steps
        self._add_field(ConfigField(
            name="gradient_accumulation_steps",
            arg_name="--gradient_accumulation_steps",
            ui_label="Gradient Accumulation Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="training_schedule",
            default_value=1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")
            ],
            help_text="Number of steps to accumulate gradients",
            tooltip="Simulates larger batch sizes. Effective batch = batch_size * accumulation_steps * num_gpus",
            importance=ImportanceLevel.IMPORTANT,
            order=4
        ))

        # Base Model Precision
        precision_choices = [
            "no_change", "fp8-quanto", "fp8-torchao", "int8-quanto", "int8-torchao",
            "int4-quanto", "int4-torchao", "int2-quanto", "fp4-bnb", "fp8-bnb", "int8-bnb", "int4-bnb"
        ]
        self._add_field(ConfigField(
            name="base_model_precision",
            arg_name="--base_model_precision",
            ui_label="Base Model Quantization",
            field_type=FieldType.SELECT,
            tab="advanced",
            section="memory_performance",
            subsection="quantization",
            default_value="no_change",
            choices=[{"value": p, "label": p} for p in precision_choices],
            help_text="Quantization for base model weights",
            tooltip="Reduces VRAM usage by quantizing model. int8 saves ~50%, int4 saves ~75% but may impact quality.",
            importance=ImportanceLevel.ADVANCED,
            order=1
        ))

    def _add_logging_fields(self):
        """Add logging and monitoring fields."""
        # Report To
        self._add_field(ConfigField(
            name="report_to",
            arg_name="--report_to",
            ui_label="Logging Platform",
            field_type=FieldType.SELECT,
            tab="monitoring",
            section="logging",
            default_value="wandb",
            choices=[
                {"value": "tensorboard", "label": "TensorBoard"},
                {"value": "wandb", "label": "Weights & Biases"},
                {"value": "comet_ml", "label": "Comet ML"},
                {"value": "all", "label": "All Platforms"},
                {"value": "none", "label": "None"}
            ],
            help_text="Where to log training metrics",
            tooltip="WandB provides cloud logging. TensorBoard is local. 'All' logs to all configured platforms.",
            importance=ImportanceLevel.IMPORTANT,
            order=1
        ))

        # Checkpointing Steps
        self._add_field(ConfigField(
            name="checkpointing_steps",
            arg_name="--checkpointing_steps",
            ui_label="Checkpoint Every N Steps",
            field_type=FieldType.NUMBER,
            tab="monitoring",
            section="checkpointing",
            default_value=500,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")
            ],
            help_text="Save model checkpoint every N steps",
            tooltip="Regular checkpoints let you resume training and test different training stages",
            importance=ImportanceLevel.IMPORTANT,
            order=1
        ))

        # Output Directory
        self._add_field(ConfigField(
            name="output_dir",
            arg_name="--output_dir",
            ui_label="Output Directory",
            field_type=FieldType.TEXT,
            tab="basic",
            section="training_essentials",
            default_value="simpletuner-results",
            validation_rules=[
                ValidationRule(ValidationRuleType.REQUIRED, message="Output directory is required")
            ],
            help_text="Where to save the trained model and checkpoints",
            tooltip="Directory will be created if it doesn't exist. Use absolute paths for clarity.",
            importance=ImportanceLevel.ESSENTIAL,
            order=2
        ))

    def get_field(self, field_name: str) -> Optional[ConfigField]:
        """Get a specific field by name."""
        return self._fields.get(field_name)

    def get_fields_for_tab(self, tab: str, context: Optional[Dict[str, Any]] = None) -> List[ConfigField]:
        """Get all fields for a specific tab, filtered by context."""
        fields = [f for f in self._fields.values() if f.tab == tab]

        if context:
            # Filter by dependencies
            fields = [f for f in fields if self._check_dependencies(f, context)]

            # Filter by model-specific
            model_family = context.get("model_family")
            if model_family:
                fields = [f for f in fields if not f.model_specific or model_family in f.model_specific]

            # Filter by platform-specific
            platform = context.get("platform", "cuda")
            fields = [f for f in fields if not f.platform_specific or platform in f.platform_specific]

        # Sort by section, subsection, and order
        fields.sort(key=lambda f: (f.section, f.subsection or "", f.order))
        return fields

    def get_fields_by_section(self, tab: str, section: str, context: Optional[Dict[str, Any]] = None) -> List[ConfigField]:
        """Get fields for a specific section within a tab."""
        fields = self.get_fields_for_tab(tab, context)
        return [f for f in fields if f.section == section]

    def _check_dependencies(self, field: ConfigField, context: Dict[str, Any]) -> bool:
        """Check if field dependencies are satisfied."""
        for dep in field.dependencies:
            dep_value = context.get(dep.field)

            if dep.operator == "equals":
                if dep_value != dep.value:
                    return False
            elif dep.operator == "not_equals":
                if dep_value == dep.value:
                    return False
            elif dep.operator == "in":
                if dep_value not in (dep.values or []):
                    return False
            elif dep.operator == "not_in":
                if dep_value in (dep.values or []):
                    return False
            elif dep.operator == "greater_than":
                if not dep_value or dep_value <= dep.value:
                    return False
            elif dep.operator == "less_than":
                if not dep_value or dep_value >= dep.value:
                    return False

        return True

    def get_dependent_fields(self, field_name: str) -> List[str]:
        """Get fields that depend on the given field."""
        return self._dependencies_map.get(field_name, [])

    def validate_field_value(self, field_name: str, value: Any, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate a field value and return error messages."""
        field = self.get_field(field_name)
        if not field:
            return [f"Unknown field: {field_name}"]

        errors = []

        for rule in field.validation_rules:
            # Check if rule applies based on condition
            if rule.condition and context:
                if not self._check_rule_condition(rule.condition, context):
                    continue

            # Apply validation rule
            if rule.rule_type == ValidationRuleType.REQUIRED:
                if value is None or (isinstance(value, str) and not value.strip()):
                    errors.append(rule.message or f"{field.ui_label} is required")

            elif rule.rule_type == ValidationRuleType.MIN:
                if value is not None and value < rule.value:
                    errors.append(rule.message or f"{field.ui_label} must be at least {rule.value}")

            elif rule.rule_type == ValidationRuleType.MAX:
                if value is not None and value > rule.value:
                    errors.append(rule.message or f"{field.ui_label} must be at most {rule.value}")

            elif rule.rule_type == ValidationRuleType.CHOICES:
                if value is not None and value not in rule.value:
                    errors.append(rule.message or f"{field.ui_label} must be one of: {', '.join(map(str, rule.value))}")

            elif rule.rule_type == ValidationRuleType.DIVISIBLE_BY:
                if value is not None and value % rule.value != 0:
                    errors.append(rule.message or f"{field.ui_label} must be divisible by {rule.value}")

            elif rule.rule_type == ValidationRuleType.PATTERN:
                import re
                if value is not None and not re.match(rule.value, str(value)):
                    errors.append(rule.message or f"{field.ui_label} has invalid format")

        return errors

    def _check_rule_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if a validation rule condition is met."""
        for field, expected in condition.items():
            if context.get(field) != expected:
                return False
        return True

    def get_sections_for_tab(self, tab: str) -> List[Dict[str, Any]]:
        """Get unique sections for a tab with metadata."""
        sections = {}
        fields = [f for f in self._fields.values() if f.tab == tab]

        for field in fields:
            if field.section not in sections:
                sections[field.section] = {
                    "id": field.section,
                    "title": field.section.replace("_", " ").title(),
                    "subsections": set()
                }
            if field.subsection:
                sections[field.section]["subsections"].add(field.subsection)

        # Convert sets to lists
        for section in sections.values():
            section["subsections"] = sorted(list(section["subsections"]))

        return list(sections.values())

    def export_field_metadata(self) -> Dict[str, Any]:
        """Export all field metadata for frontend consumption."""
        return {
            "fields": {
                name: {
                    "name": f.name,
                    "arg_name": f.arg_name,
                    "ui_label": f.ui_label,
                    "field_type": f.field_type.value,
                    "tab": f.tab,
                    "section": f.section,
                    "subsection": f.subsection,
                    "default_value": f.default_value,
                    "choices": f.choices,
                    "dependencies": [
                        {
                            "field": d.field,
                            "value": d.value,
                            "values": d.values,
                            "operator": d.operator
                        }
                        for d in f.dependencies
                    ],
                    "help_text": f.help_text,
                    "tooltip": f.tooltip,
                    "placeholder": f.placeholder,
                    "importance": f.importance.value,
                    "model_specific": f.model_specific,
                    "platform_specific": f.platform_specific,
                    "warning": f.warning,
                    "group": f.group,
                    "order": f.order
                }
                for name, f in self._fields.items()
            },
            "dependencies_map": self._dependencies_map,
            "tabs": self._get_tab_structure()
        }

    def _get_tab_structure(self) -> Dict[str, Any]:
        """Get the structure of all tabs and sections."""
        tabs = {}

        for field in self._fields.values():
            if field.tab not in tabs:
                tabs[field.tab] = {
                    "sections": {}
                }

            if field.section not in tabs[field.tab]["sections"]:
                tabs[field.tab]["sections"][field.section] = {
                    "title": field.section.replace("_", " ").title(),
                    "subsections": set(),
                    "field_count": 0
                }

            tabs[field.tab]["sections"][field.section]["field_count"] += 1
            if field.subsection:
                tabs[field.tab]["sections"][field.section]["subsections"].add(field.subsection)

        # Convert sets to lists
        for tab in tabs.values():
            for section in tab["sections"].values():
                section["subsections"] = sorted(list(section["subsections"]))

        return tabs


# Create a singleton instance
field_registry = FieldRegistry()