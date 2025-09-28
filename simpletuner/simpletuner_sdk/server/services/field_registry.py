"""Field Registry for SimpleTuner Configuration Parameters.

This module provides a comprehensive registry of all 261+ configuration parameters
with metadata, validation rules, and interdependencies.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    from simpletuner.helpers.models.all import model_families

    logger.debug(f"Successfully imported model_families with {len(model_families)} families")
except ImportError as e:
    logger.error(f"Failed to import model_families: {e}")
    model_families = {}

try:
    from .arg_parser_integration import arg_parser_integration

    logger.debug("Successfully imported arg_parser_integration")
except ImportError as e:
    logger.error(f"Failed to import arg_parser_integration: {e}")
    arg_parser_integration = None


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
    dynamic_choices: bool = False  # Whether choices are dynamically loaded
    cmd_args_help: Optional[str] = None  # Formatted help text from cmd_args.py
    step: Optional[float] = None  # Numeric input increment
    custom_component: Optional[str] = None  # Custom renderer identifier for UI
    checkbox_label: Optional[str] = None  # Alternate label for checkbox/toggle inputs
    webui_onboarding: bool = False  # treated as WebUI-only state, never persisted to config


class FieldRegistry:
    """Central registry for all configuration fields."""

    def __init__(self):
        self._fields: Dict[str, ConfigField] = {}
        self._dependencies_map: Dict[str, List[str]] = {}  # field -> dependent fields
        logger.debug("FieldRegistry.__init__ called")
        self._initialize_fields()
        logger.debug(f"FieldRegistry initialized with {len(self._fields)} fields")

    def _initialize_fields(self):
        """Initialize all configuration fields."""
        logger.debug("FieldRegistry._initialize_fields called")
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
        # Auto-populate help text from cmd_args.py if not provided
        if field.arg_name and arg_parser_integration:
            arg_help = arg_parser_integration.get_argument_help(field.arg_name)
            if arg_help:
                # Store cmd_args help separately for detailed tooltip
                field.cmd_args_help = arg_parser_integration.format_help_for_ui(arg_help)

                # Use cmd_args help as primary help text if not set
                if not field.help_text:
                    field.help_text = arg_help

        if field.field_type == FieldType.NUMBER and field.step is None:
            auto_step = self._compute_default_step(field.default_value)
            if auto_step is not None:
                field.step = auto_step

        self._fields[field.name] = field

        # Update dependency map
        for dep in field.dependencies:
            if dep.field not in self._dependencies_map:
                self._dependencies_map[dep.field] = []
            self._dependencies_map[dep.field].append(field.name)

    @staticmethod
    def _compute_default_step(default_value: Any) -> Optional[float]:
        """Derive a sensible numeric step based on the provided default value."""

        if default_value in (None, ""):
            return None

        if isinstance(default_value, bool):
            return None

        if isinstance(default_value, int):
            return 1.0

        if isinstance(default_value, float):
            if default_value == 0:
                return 1.0
            magnitude = math.pow(10, math.floor(math.log10(abs(default_value))))
            return float(magnitude)

        return None

    def _add_model_config_fields(self):
        """Add model configuration fields."""
        logger.debug("_add_model_config_fields called")
        # Model Family
        model_family_list = list(model_families.keys())
        self._add_field(
            ConfigField(
                name="model_family",
                arg_name="--model_family",
                ui_label="Model Family",
                field_type=FieldType.SELECT,
                tab="model",
                section="model_config",
                subsection="architecture",
                choices=[{"value": f, "label": f.upper()} for f in model_family_list],
                validation_rules=[
                    ValidationRule(ValidationRuleType.REQUIRED, message="Model family is required"),
                    ValidationRule(ValidationRuleType.CHOICES, value=model_family_list),
                ],
                help_text="The base model architecture family to train",
                tooltip="Different model families have different capabilities and requirements",
                importance=ImportanceLevel.ESSENTIAL,
                order=2,
            )
        )

        # Model Flavour
        self._add_field(
            ConfigField(
                name="model_flavour",
                arg_name="--model_flavour",
                ui_label="Model Flavour",
                field_type=FieldType.SELECT,
                tab="model",
                section="model_config",
                subsection="architecture",
                default_value="",
                choices=[],  # Dynamic based on model_family
                dependencies=[FieldDependency(field="model_family", operator="not_equals", value="")],
                help_text="Specific variant of the selected model family",
                tooltip="Some models have multiple variants with different sizes or capabilities",
                importance=ImportanceLevel.IMPORTANT,
                order=3,
            )
        )

        # Pretrained Model Path
        self._add_field(
            ConfigField(
                name="pretrained_model_name_or_path",
                arg_name="--pretrained_model_name_or_path",
                ui_label="Base Model Path",
                field_type=FieldType.TEXT,
                tab="model",
                section="model_config",
                subsection="paths",
                placeholder="Leave blank to use the default for the selected flavour",
                help_text="Optional override of the model checkpoint. Leave blank to use the default path for the selected model flavour.",
                tooltip="Provide a custom Hugging Face model ID or local directory. If omitted, the selected model flavour determines the path.",
                importance=ImportanceLevel.IMPORTANT,
                order=4,
            )
        )

        # Output Directory
        self._add_field(
            ConfigField(
                name="output_dir",
                arg_name="--output_dir",
                ui_label="Output Directory",
                field_type=FieldType.TEXT,
                tab="basic",
                section="essential_settings",
                subsection="paths",
                default_value="./output",
                validation_rules=[ValidationRule(ValidationRuleType.REQUIRED, message="Output directory is required")],
                help_text="Directory where model checkpoints and logs will be saved",
                tooltip="All training outputs including checkpoints, logs, and samples will be saved here",
                importance=ImportanceLevel.ESSENTIAL,
                order=5,
            )
        )

        self._add_field(
            ConfigField(
                name="configs_dir",
                arg_name="configs_dir",
                ui_label="Config Directory",
                field_type=FieldType.TEXT,
                tab="basic",
                section="essential_settings",
                subsection="paths",
                default_value="",
                help_text="Root folder SimpleTuner uses to store and load configuration files",
                tooltip="Overrides the default ~/.simpletuner/configs directory when managing saved configs.",
                importance=ImportanceLevel.IMPORTANT,
                order=6,
            )
        )

        # Logging Directory
        self._add_field(
            ConfigField(
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
                order=7,
            )
        )

        self._add_field(
            ConfigField(
                name="model_type",
                arg_name="--model_type",
                ui_label="Model Type",
                field_type=FieldType.SELECT,
                tab="model",
                section="model_config",
                subsection="architecture",
                default_value="full",
                choices=[
                    {"value": "full", "label": "Full Model Training"},
                    {"value": "lora", "label": "LoRA (Low-Rank Adaptation)"},
                ],
                validation_rules=[
                    ValidationRule(ValidationRuleType.REQUIRED, message="Model type is required"),
                    ValidationRule(ValidationRuleType.CHOICES, value=["full", "lora"]),
                ],
                help_text="Choose between full model training or LoRA adapter training",
                tooltip="Full training updates all model weights. LoRA only trains small adapter matrices, using less memory and producing smaller files.",
                importance=ImportanceLevel.ESSENTIAL,
                order=1,
            )
        )

        # Training Seed
        self._add_field(
            ConfigField(
                name="seed",
                arg_name="--seed",
                ui_label="Training Seed",
                field_type=FieldType.NUMBER,
                tab="basic",
                section="hardware_config",
                default_value=42,
                validation_rules=[
                    ValidationRule(ValidationRuleType.MIN, value=0, message="Seed must be non-negative"),
                    ValidationRule(ValidationRuleType.MAX, value=2147483647, message="Seed must fit in 32-bit integer"),
                ],
                help_text="Seed used for deterministic training behaviour",
                tooltip="Use the same seed to reproduce runs precisely. Leave blank to randomise per launch.",
                importance=ImportanceLevel.ADVANCED,
                order=9,
            )
        )

        # Resolution
        self._add_field(
            ConfigField(
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
                    ValidationRule(ValidationRuleType.DIVISIBLE_BY, value=8, message="Resolution must be divisible by 8"),
                ],
                help_text="Default image resolution (width × height) applied when a dataset entry does not specify its own.",
                tooltip="Higher resolutions require more VRAM. SD 1.5: 512, SDXL: 1024, Flux: 1024+",
                importance=ImportanceLevel.ESSENTIAL,
                order=10,
                dependencies=[
                    FieldDependency(
                        field="model_family", operator="equals", value="sd15", action="set_value", target_value=512
                    ),
                    FieldDependency(
                        field="model_family",
                        operator="in",
                        values=["sdxl", "flux", "sd3"],
                        action="set_value",
                        target_value=1024,
                    ),
                ],
            )
        )

        # Resume from Checkpoint
        self._add_field(
            ConfigField(
                name="resume_from_checkpoint",
                arg_name="--resume_from_checkpoint",
                ui_label="Resume From Checkpoint",
                field_type=FieldType.SELECT,
                tab="basic",
                section="training_config",
                default_value="latest",
                choices=[{"value": "", "label": "None (Start fresh)"}, {"value": "latest", "label": "Latest checkpoint"}],
                help_text="Select checkpoint to resume training from",
                tooltip="Checkpoints will be dynamically loaded based on output directory. Use 'latest' to auto-resume from the most recent checkpoint.",
                importance=ImportanceLevel.ADVANCED,
                order=11,
                dynamic_choices=True,  # Mark this field as having dynamic choices
            )
        )

        # Prediction Type
        self._add_field(
            ConfigField(
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
                    {"value": "flow_matching", "label": "Flow Matching"},
                ],
                help_text="The parameterization type for the diffusion model",
                tooltip="Usually auto-detected from the model. Flow matching is used by Flux, SD3, and similar models.",
                importance=ImportanceLevel.ADVANCED,
                order=10,
                dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True)],
            )
        )

        # VAE Path
        self._add_field(
            ConfigField(
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
                order=11,
            )
        )

        # VAE Dtype
        self._add_field(
            ConfigField(
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
                    {"value": "bf16", "label": "BF16"},
                ],
                help_text="Precision for VAE encoding/decoding. Lower precision saves memory.",
                tooltip="FP16/BF16 can reduce memory usage but may slightly affect image quality",
                importance=ImportanceLevel.ADVANCED,
                order=12,
            )
        )

        # VAE Cache On-Demand
        self._add_field(
            ConfigField(
                name="vae_cache_ondemand",
                arg_name="--vae_cache_ondemand",
                ui_label="Use On-demand VAE Caching",
                field_type=FieldType.CHECKBOX,
                tab="model",
                section="vae_config",
                default_value=False,
                help_text="Process VAE latents during training instead of precomputing them",
                tooltip="Saves upfront time but slows each training step and increases memory pressure",
                importance=ImportanceLevel.ADVANCED,
                order=13,
            )
        )

        # Base Model Precision
        self._add_field(
            ConfigField(
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
                    {"value": "nf4-bnb", "label": "NF4 (BitsAndBytes)"},
                ],
                help_text="Precision for loading the base model. Lower precision saves memory.",
                tooltip="Quantization reduces memory usage but may impact quality. INT8/INT4 are commonly used.",
                importance=ImportanceLevel.ADVANCED,
                order=14,
                dependencies=[FieldDependency(field="model_type", operator="equals", value="lora", action="enable")],
            )
        )

        text_encoder_precision_choices = [
            {"value": "no_change", "label": "No Change"},
            {"value": "int8-quanto", "label": "INT8 (Quanto)"},
            {"value": "int4-quanto", "label": "INT4 (Quanto)"},
            {"value": "int2-quanto", "label": "INT2 (Quanto)"},
            {"value": "int8-torchao", "label": "INT8 (TorchAO)"},
            {"value": "nf4-bnb", "label": "NF4 (BitsAndBytes)"},
            {"value": "fp8-quanto", "label": "FP8 (Quanto)"},
            {"value": "fp8uz-quanto", "label": "FP8UZ (Quanto)"},
            {"value": "fp8-torchao", "label": "FP8 (TorchAO)"},
        ]

        for idx in range(1, 5):
            ui_label = "Text Encoder Precision" if idx == 1 else f"Text Encoder {idx} Precision"
            self._add_field(
                ConfigField(
                    name=f"text_encoder_{idx}_precision",
                    arg_name=f"--text_encoder_{idx}_precision",
                    ui_label=ui_label,
                    field_type=FieldType.SELECT,
                    tab="model",
                    section="quantization",
                    default_value="no_change",
                    choices=text_encoder_precision_choices,
                    help_text="Precision for text encoders. Lower precision saves memory.",
                    tooltip="Text encoder quantization has minimal impact on quality",
                    importance=ImportanceLevel.ADVANCED,
                    order=15 + idx - 1,
                )
            )

        # Gradient Checkpointing Interval
        self._add_field(
            ConfigField(
                name="gradient_checkpointing_interval",
                arg_name="--gradient_checkpointing_interval",
                ui_label="Gradient Checkpointing Interval",
                field_type=FieldType.NUMBER,
                tab="model",
                section="memory_optimization",
                default_value=1,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Interval must be at least 1")],
                dependencies=[
                    FieldDependency(field="gradient_checkpointing", operator="equals", value=True, action="enable")
                ],
                help_text="Checkpoint every N transformer blocks",
                tooltip="Higher values save more memory but increase computation time. Only supported for SDXL and Flux models.",
                importance=ImportanceLevel.ADVANCED,
                order=16,
            )
        )

        # Offload During Startup
        self._add_field(
            ConfigField(
                name="offload_during_startup",
                arg_name="--offload_during_startup",
                ui_label="Offload During Startup",
                field_type=FieldType.CHECKBOX,
                tab="model",
                section="memory_optimization",
                default_value=False,
                help_text="Offload text encoders to CPU during VAE caching",
                tooltip="Useful for large models that OOM during startup. May significantly increase startup time.",
                importance=ImportanceLevel.ADVANCED,
                order=17,
            )
        )

        # Quantize Via
        self._add_field(
            ConfigField(
                name="quantize_via",
                arg_name="--quantize_via",
                ui_label="Quantization Device",
                field_type=FieldType.SELECT,
                tab="model",
                section="quantization",
                default_value="cpu",
                choices=[
                    {"value": "cpu", "label": "CPU (Slower but safer)"},
                    {"value": "accelerator", "label": "GPU/Accelerator (Faster)"},
                ],
                dependencies=[
                    FieldDependency(field="model_type", operator="equals", value="lora", action="enable"),
                    FieldDependency(field="base_model_precision", operator="not_equals", value="no_change", action="enable"),
                ],
                help_text="Where to perform model quantization",
                tooltip="CPU is safer for 24GB cards with large models. GPU is faster but may OOM.",
                importance=ImportanceLevel.ADVANCED,
                order=18,
            )
        )

        # Fused QKV Projections
        self._add_field(
            ConfigField(
                name="fused_qkv_projections",
                arg_name="--fused_qkv_projections",
                ui_label="Fused QKV Projections",
                field_type=FieldType.CHECKBOX,
                tab="model",
                section="architecture",
                default_value=False,
                platform_specific=["cuda"],
                help_text="Enables Flash Attention 3 when supported; otherwise falls back to PyTorch SDPA.",
                tooltip="Improves attention efficiency on modern NVIDIA GPUs. Uses native SDPA when Flash Attention 3 is unavailable.",
                importance=ImportanceLevel.EXPERIMENTAL,
                order=19,
            )
        )

    def _add_training_parameter_fields(self):
        """Add training parameter fields."""
        # Number of Training Epochs
        self._add_field(
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
        self._add_field(
            ConfigField(
                name="max_train_steps",
                arg_name="--max_train_steps",
                ui_label="Max Training Steps",
                field_type=FieldType.NUMBER,
                tab="training",
                section="training_schedule",
                default_value=0,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Steps must be non-negative")],
                help_text="Maximum number of training steps (0 = use epochs instead)",
                tooltip="If set to a positive value, training will stop after this many steps regardless of epochs",
                importance=ImportanceLevel.IMPORTANT,
                order=2,
            )
        )

        # Batch Size
        self._add_field(
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
        self._add_field(
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
        optimizer_choices = [
            "adamw_bf16",
            "ao-adamw8bit",
            "ao-adamw4bit",
            "ao-adamwfp8",
            "optimi-adamw",
            "optimi-lion",
            "optimi-stableadamw",
            "prodigy",
            "soap",
            "dadaptation",
            "dadaptsgd",
            "dadaptadam",
            "dadaptlion",
            "adafactor",
            "adamw8bit",
            "adamw",
            "adam8bit",
            "adam",
            "lion8bit",
            "lion",
            "rmsprop",
            "sgd",
            "StableAdamWUnfused",
            "deepspeed-adamw",
        ]
        self._add_field(
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

        self._add_field(
            ConfigField(
                name="optimizer_config",
                arg_name="--optimizer_config",
                ui_label="Optimizer Extra Settings",
                field_type=FieldType.TEXT,
                tab="training",
                section="optimizer_config",
                placeholder="beta1=0.9,beta2=0.95,weight_decay=0.01",
                help_text="Comma-separated key=value pairs forwarded to the selected optimizer",
                tooltip="Example: beta1=0.9,beta2=0.95,weight_decay=0.01. Leave blank to use optimizer defaults.",
                importance=ImportanceLevel.ADVANCED,
                order=6,
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
        self._add_field(
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
        self._add_field(
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
        self._add_field(
            ConfigField(
                name="lr_warmup_steps",
                arg_name="--lr_warmup_steps",
                ui_label="Learning Rate Warmup Steps",
                field_type=FieldType.NUMBER,
                tab="training",
                section="learning_rate",
                default_value=0,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
                help_text="Number of steps to gradually increase LR from 0",
                tooltip="Helps training stability at start. Typically 5-10% of total steps",
                importance=ImportanceLevel.ADVANCED,
                order=3,
            )
        )

        # Max Checkpoints
        self._add_field(
            ConfigField(
                name="checkpoints_total_limit",
                arg_name="--checkpoints_total_limit",
                ui_label="Maximum Checkpoints to Keep",
                field_type=FieldType.NUMBER,
                tab="basic",
                section="checkpointing",
                default_value=10,
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
        self._add_field(
            ConfigField(
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
                order=1,
            )
        )

        # Train Text Encoder
        self._add_field(
            ConfigField(
                name="train_text_encoder",
                arg_name="--train_text_encoder",
                ui_label="Train Text Encoder",
                field_type=FieldType.CHECKBOX,
                tab="training",
                section="text_encoder_training",
                default_value=False,
                help_text="Also train the text encoder (CLIP) model",
                tooltip="Can improve concept learning but uses more VRAM. Not recommended for LoRA",
                importance=ImportanceLevel.ADVANCED,
                order=1,
            )
        )

        # Text Encoder LR
        self._add_field(
            ConfigField(
                name="text_encoder_lr",
                arg_name="--text_encoder_lr",
                ui_label="Text Encoder Learning Rate",
                field_type=FieldType.NUMBER,
                tab="training",
                section="text_encoder_training",
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be positive")],
                help_text="Separate learning rate for text encoder",
                tooltip="Usually lower than main LR. If not set, uses main learning rate",
                importance=ImportanceLevel.ADVANCED,
                order=4,
                dependencies=[FieldDependency(field="train_text_encoder", operator="equals", value=True, action="show")],
            )
        )

        # LR Number of Cycles
        self._add_field(
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
        self._add_field(
            ConfigField(
                name="lr_power",
                arg_name="--lr_power",
                ui_label="LR Polynomial Power",
                field_type=FieldType.NUMBER,
                tab="training",
                section="learning_rate",
                default_value=1.0,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.1, message="Power should be positive")],
                dependencies=[FieldDependency(field="lr_scheduler", operator="equals", value="polynomial", action="show")],
                help_text="Power for polynomial decay scheduler",
                tooltip="1.0 = linear decay, 2.0 = quadratic decay. Higher = stays high longer then drops faster.",
                importance=ImportanceLevel.ADVANCED,
                order=6,
            )
        )

        # Use Soft Min SNR
        self._add_field(
            ConfigField(
                name="use_soft_min_snr",
                arg_name="--use_soft_min_snr",
                ui_label="Use Soft Min-SNR",
                field_type=FieldType.CHECKBOX,
                tab="training",
                section="loss_functions",
                default_value=False,
                dependencies=[FieldDependency(field="snr_gamma", operator="greater_than", value=0, action="show")],
                help_text="Use soft clamping instead of hard clamping for Min-SNR",
                tooltip="Smoother transition at the clamping boundary. May improve training stability.",
                importance=ImportanceLevel.EXPERIMENTAL,
                order=8,
            )
        )

        # Use EMA Toggle
        self._add_field(
            ConfigField(
                name="use_ema",
                arg_name="--use_ema",
                ui_label="Enable EMA",
                field_type=FieldType.CHECKBOX,
                tab="training",
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
        self._add_field(
            ConfigField(
                name="ema_device",
                arg_name="--ema_device",
                ui_label="EMA Device",
                field_type=FieldType.SELECT,
                tab="training",
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
        self._add_field(
            ConfigField(
                name="ema_cpu_only",
                arg_name="--ema_cpu_only",
                ui_label="EMA on CPU Only",
                field_type=FieldType.CHECKBOX,
                tab="training",
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
        self._add_field(
            ConfigField(
                name="ema_update_interval",
                arg_name="--ema_update_interval",
                ui_label="EMA Update Interval",
                field_type=FieldType.NUMBER,
                tab="training",
                section="ema_config",
                default_value=10,
                validation_rules=[
                    ValidationRule(ValidationRuleType.MIN, value=1, message="Must update at least every step")
                ],
                dependencies=[FieldDependency(field="use_ema", operator="equals", value=True, action="show")],
                help_text="Update EMA weights every N optimizer steps",
                tooltip="Higher values = faster training but less smooth EMA. Default: 10",
                importance=ImportanceLevel.ADVANCED,
                order=3,
            )
        )

        # EMA Foreach Disable
        self._add_field(
            ConfigField(
                name="ema_foreach_disable",
                arg_name="--ema_foreach_disable",
                ui_label="Disable EMA Foreach Ops",
                field_type=FieldType.CHECKBOX,
                tab="training",
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
        self._add_field(
            ConfigField(
                name="ema_decay",
                arg_name="--ema_decay",
                ui_label="EMA Decay",
                field_type=FieldType.NUMBER,
                tab="training",
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

    def _add_lora_config_fields(self):
        """Add LoRA configuration fields."""
        # LoRA Rank
        self._add_field(
            ConfigField(
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
                    ValidationRule(ValidationRuleType.MAX, value=256, message="LoRA rank >256 is very large"),
                ],
                dependencies=[FieldDependency(field="model_type", value="lora")],
                help_text="Dimension of LoRA update matrices",
                tooltip="Higher rank = more capacity but larger files. Common values: 4-32 for style, 64-128 for complex concepts",
                importance=ImportanceLevel.IMPORTANT,
                order=1,
            )
        )

        # LoRA Alpha
        self._add_field(
            ConfigField(
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
                dependencies=[FieldDependency(field="model_type", value="lora")],
                help_text="Scaling factor for LoRA updates",
                tooltip="Usually set equal to rank. Controls the magnitude of LoRA's effect.",
                importance=ImportanceLevel.IMPORTANT,
                order=2,
            )
        )

        # LoRA Type
        self._add_field(
            ConfigField(
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
                    {"value": "lycoris", "label": "LyCORIS (LoKr/LoHa/etc)"},
                ],
                dependencies=[FieldDependency(field="model_type", value="lora")],
                help_text="LoRA implementation type",
                tooltip="Standard LoRA is most common. LyCORIS offers advanced decomposition methods.",
                importance=ImportanceLevel.ADVANCED,
                order=3,
            )
        )

        # Flux LoRA Target
        flux_targets = [
            "mmdit",
            "context",
            "context+ffs",
            "all",
            "all+ffs",
            "ai-toolkit",
            "tiny",
            "nano",
            "controlnet",
            "all+ffs+embedder",
            "all+ffs+embedder+controlnet",
        ]
        self._add_field(
            ConfigField(
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
                    FieldDependency(field="model_family", value="flux"),
                ],
                help_text="Which layers to train in Flux models",
                tooltip="'all' trains all attention layers. 'context' only trains text layers. '+ffs' includes feed-forward layers.",
                importance=ImportanceLevel.ADVANCED,
                model_specific=["flux"],
                order=10,
            )
        )

        # Use DoRA
        self._add_field(
            ConfigField(
                name="use_dora",
                arg_name="--use_dora",
                ui_label="Use DoRA",
                field_type=FieldType.CHECKBOX,
                tab="model",
                section="lora_config",
                subsection="advanced",
                default_value=False,
                dependencies=[FieldDependency(field="model_type", value="lora")],
                help_text="Enable DoRA (Weight-Decomposed LoRA)",
                tooltip="Experimental feature that decomposes weights into magnitude and direction. May improve quality but slower.",
                warning="Experimental feature - may slow down training",
                importance=ImportanceLevel.EXPERIMENTAL,
                order=20,
            )
        )

    def _add_data_config_fields(self):
        """Add data configuration fields."""
        # Resolution
        self._add_field(
            ConfigField(
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
                    ValidationRule(ValidationRuleType.DIVISIBLE_BY, value=64, message="Resolution must be divisible by 64"),
                ],
                help_text="Resolution for training images",
                tooltip="All images will be resized to this resolution. Must be divisible by 64. Higher = better quality but more VRAM.",
                importance=ImportanceLevel.ESSENTIAL,
                order=3,
            )
        )

        # Resolution Type
        self._add_field(
            ConfigField(
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
                    {"value": "pixel_area", "label": "Pixel as Area"},
                ],
                help_text="How to interpret the resolution value",
                tooltip="Pixel: resize shortest edge. Area: total pixel count. Pixel as Area: converts pixel to megapixels.",
                importance=ImportanceLevel.ADVANCED,
                order=2,
            )
        )

        # Data Backend Config
        self._add_field(
            ConfigField(
                name="data_backend_config",
                arg_name="--data_backend_config",
                ui_label="Data Backend Config",
                field_type=FieldType.SELECT,
                tab="basic",
                section="data_config",
                default_value="config/multidatabackend.json",
                choices=[],
                validation_rules=[
                    ValidationRule(ValidationRuleType.REQUIRED, message="Select a data backend configuration")
                ],
                help_text="Select a saved dataset configuration (managed in Datasets & Environments tabs)",
                tooltip="Pick which dataset plan to use. Create or edit datasets in the Datasets tab; manage saved plans from Environments.",
                importance=ImportanceLevel.ESSENTIAL,
                order=1,
                dynamic_choices=True,
            )
        )

        # Caption Strategy
        self._add_field(
            ConfigField(
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
                    {"value": "parquet", "label": "Parquet Dataset"},
                ],
                help_text="How to load captions for images",
                tooltip="Filename: use image name. Textfile: .txt files. Parquet: structured dataset files.",
                importance=ImportanceLevel.IMPORTANT,
                order=2,
            )
        )

        # VAE Batch Size
        self._add_field(
            ConfigField(
                name="vae_batch_size",
                arg_name="--vae_batch_size",
                ui_label="VAE Batch Size",
                field_type=FieldType.NUMBER,
                tab="model",
                section="vae_config",
                default_value=4,
                validation_rules=[
                    ValidationRule(ValidationRuleType.MIN, value=1, message="VAE batch size must be at least 1")
                ],
                help_text="Batch size for VAE encoding during caching",
                tooltip="Higher values speed up VAE caching but use more VRAM. Reduce if getting OOM during cache creation.",
                importance=ImportanceLevel.ADVANCED,
                order=1,
            )
        )

        # Caption Dropout
        self._add_field(
            ConfigField(
                name="caption_dropout_probability",
                arg_name="--caption_dropout_probability",
                ui_label="Caption Dropout Probability",
                field_type=FieldType.NUMBER,
                tab="data",
                section="caption_processing",
                default_value=0.0,
                validation_rules=[
                    ValidationRule(ValidationRuleType.MIN, value=0, message="Must be between 0 and 1"),
                    ValidationRule(ValidationRuleType.MAX, value=1, message="Must be between 0 and 1"),
                ],
                help_text="Probability of dropping captions during training",
                tooltip="Helps model learn from images alone. 0.1 = 10% chance of no caption",
                importance=ImportanceLevel.ADVANCED,
                order=1,
            )
        )

        # Image Transforms
        self._add_field(
            ConfigField(
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
                order=3,
            )
        )

        # Tokenizer Max Length (Danger mode only)
        self._add_field(
            ConfigField(
                name="tokenizer_max_length",
                arg_name="--tokenizer_max_length",
                ui_label="Tokenizer Max Length",
                field_type=FieldType.NUMBER,
                tab="data",
                section="caption_processing",
                default_value=120,
                validation_rules=[
                    ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1"),
                    ValidationRule(ValidationRuleType.MAX, value=1024, message="Maximum reasonable length is 1024"),
                ],
                help_text="Override the tokenizer sequence length (advanced).",
                tooltip="Only adjust when you understand the model's tokenizer limits.",
                dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show")],
                importance=ImportanceLevel.EXPERIMENTAL,
                order=2,
            )
        )

    def _add_validation_fields(self):
        """Add validation configuration fields."""
        # Validation Steps
        self._add_field(
            ConfigField(
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
                order=1,
            )
        )

        self._add_field(
            ConfigField(
                name="disable_benchmark",
                arg_name="--disable_benchmark",
                ui_label="Skip Baseline Benchmark",
                field_type=FieldType.CHECKBOX,
                tab="validation",
                section="validation_schedule",
                default_value=False,
                help_text="Skip generating baseline comparison images before training starts",
                tooltip="Disable if you want to reduce startup time; recommended to keep enabled for qualitative comparisons.",
                importance=ImportanceLevel.ADVANCED,
                order=2,
            )
        )

        # Validation Prompt
        self._add_field(
            ConfigField(
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
                order=1,
            )
        )

        # Number of Validation Images
        self._add_field(
            ConfigField(
                name="num_validation_images",
                arg_name="--num_validation_images",
                ui_label="Validation Images Count",
                field_type=FieldType.NUMBER,
                tab="validation",
                section="validation_schedule",
                default_value=1,
                validation_rules=[
                    ValidationRule(ValidationRuleType.MIN, value=1, message="Must generate at least 1 validation image"),
                    ValidationRule(ValidationRuleType.MAX, value=10, message="Generating >10 images may slow training"),
                ],
                help_text="Number of images to generate per validation",
                tooltip="More images give better sense of model performance but take longer to generate",
                importance=ImportanceLevel.ADVANCED,
                order=3,
            )
        )

        # Validation Guidance Scale
        self._add_field(
            ConfigField(
                name="validation_guidance",
                arg_name="--validation_guidance",
                ui_label="Guidance Scale",
                field_type=FieldType.NUMBER,
                tab="validation",
                section="validation_guidance",
                default_value=7.5,
                validation_rules=[
                    ValidationRule(ValidationRuleType.MIN, value=1, message="Guidance must be at least 1"),
                    ValidationRule(ValidationRuleType.MAX, value=30, message="Guidance >30 may cause artifacts"),
                ],
                help_text="CFG guidance scale for validation images",
                tooltip="Higher values follow prompt more closely. 7-12 is typical. Set to 1 for distilled models.",
                importance=ImportanceLevel.ADVANCED,
                order=1,
            )
        )

        # Validation Inference Steps
        self._add_field(
            ConfigField(
                name="validation_num_inference_steps",
                arg_name="--validation_num_inference_steps",
                ui_label="Inference Steps",
                field_type=FieldType.NUMBER,
                tab="validation",
                section="validation_schedule",
                default_value=20,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1 step")],
                help_text="Number of diffusion steps for validation renders",
                tooltip="Lower values speed up validation at the cost of quality. Typical range: 20-30.",
                importance=ImportanceLevel.ADVANCED,
                order=4,
            )
        )

        # Validation on Startup
        self._add_field(
            ConfigField(
                name="validation_on_startup",
                arg_name="--validation_on_startup",
                ui_label="Validation on Startup",
                field_type=FieldType.CHECKBOX,
                tab="validation",
                section="validation_schedule",
                default_value=False,
                help_text="Run validation on the base model before training starts",
                tooltip="Useful for comparing before/after results",
                importance=ImportanceLevel.ADVANCED,
                order=5,
            )
        )

        # Validation Using Datasets
        self._add_field(
            ConfigField(
                name="validation_using_datasets",
                arg_name="--validation_using_datasets",
                ui_label="Validate Using Dataset Images",
                field_type=FieldType.CHECKBOX,
                tab="validation",
                section="validation_options",
                default_value=False,
                help_text="Use random images from training datasets for validation",
                tooltip="Alternative to validation prompts. Be mindful of privacy when sharing results.",
                importance=ImportanceLevel.ADVANCED,
                order=3,
            )
        )

        # Validation Torch Compile
        self._add_field(
            ConfigField(
                name="validation_torch_compile",
                arg_name="--validation_torch_compile",
                ui_label="Compile Validation Pipeline",
                field_type=FieldType.CHECKBOX,
                tab="validation",
                section="validation_options",
                default_value=False,
                help_text="Use torch.compile() on validation pipeline for speed",
                tooltip="Can significantly speed up validation but may error on some setups",
                importance=ImportanceLevel.EXPERIMENTAL,
                order=4,
            )
        )

        # Validation Guidance Real
        self._add_field(
            ConfigField(
                name="validation_guidance_real",
                arg_name="--validation_guidance_real",
                ui_label="Real CFG (Distilled Models)",
                field_type=FieldType.NUMBER,
                tab="validation",
                section="validation_guidance",
                default_value=1.0,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1.0, message="Must be at least 1.0")],
                help_text="CFG value for distilled models (e.g., FLUX schnell)",
                tooltip="Use 1.0 for no CFG (distilled models). Higher values for real CFG sampling.",
                importance=ImportanceLevel.ADVANCED,
                order=2,
            )
        )

        # Validation No CFG Until Timestep
        self._add_field(
            ConfigField(
                name="validation_no_cfg_until_timestep",
                arg_name="--validation_no_cfg_until_timestep",
                ui_label="Skip CFG Until Timestep",
                field_type=FieldType.NUMBER,
                tab="validation",
                section="validation_guidance",
                default_value=2,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
                help_text="Skip CFG for initial timesteps (Flux only)",
                tooltip="For Flux real CFG: skip CFG on these initial timesteps. Default: 2",
                importance=ImportanceLevel.ADVANCED,
                order=3,
            )
        )

        # Validation Negative Prompt
        self._add_field(
            ConfigField(
                name="validation_negative_prompt",
                arg_name="--validation_negative_prompt",
                ui_label="Negative Prompt",
                field_type=FieldType.TEXTAREA,
                tab="validation",
                section="prompt_management",
                default_value="blurry, cropped, ugly",
                help_text="Negative prompt for validation images",
                tooltip="What to avoid in generated images. Set to empty string to disable.",
                importance=ImportanceLevel.ADVANCED,
                order=2,
            )
        )

        # Validation Randomize
        self._add_field(
            ConfigField(
                name="validation_randomize",
                arg_name="--validation_randomize",
                ui_label="Randomize Seeds",
                field_type=FieldType.CHECKBOX,
                tab="validation",
                section="validation_options",
                default_value=False,
                help_text="Use random seeds for each validation",
                tooltip="Ignores validation_seed and generates different images each time",
                importance=ImportanceLevel.ADVANCED,
                order=5,
            )
        )

        # Validation Seed
        self._add_field(
            ConfigField(
                name="validation_seed",
                arg_name="--validation_seed",
                ui_label="Validation Seed",
                field_type=FieldType.NUMBER,
                tab="validation",
                section="validation_options",
                default_value=42,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
                dependencies=[FieldDependency(field="validation_randomize", operator="equals", value=False, action="show")],
                help_text="Fixed seed for reproducible validation images",
                tooltip="Use the same seed to compare training progress consistently",
                importance=ImportanceLevel.ADVANCED,
                order=6,
            )
        )

        # Validation Disable
        self._add_field(
            ConfigField(
                name="validation_disable",
                arg_name="--validation_disable",
                ui_label="Disable Validation",
                field_type=FieldType.CHECKBOX,
                tab="validation",
                section="validation_schedule",
                default_value=False,
                help_text="Completely disable validation image generation",
                tooltip="Saves time and VRAM but you won't see progress during training",
                importance=ImportanceLevel.ADVANCED,
                order=5,
            )
        )

        # Validation Prompt Library
        self._add_field(
            ConfigField(
                name="validation_prompt_library",
                arg_name="--validation_prompt_library",
                ui_label="Use Prompt Library",
                field_type=FieldType.CHECKBOX,
                tab="validation",
                section="prompt_management",
                default_value=False,
                help_text="Use SimpleTuner's built-in prompt library",
                tooltip="Generates multiple diverse validation images automatically",
                importance=ImportanceLevel.ADVANCED,
                order=3,
            )
        )

        # User Prompt Library
        self._add_field(
            ConfigField(
                name="user_prompt_library",
                arg_name="--user_prompt_library",
                ui_label="Custom Prompt Library Path",
                field_type=FieldType.TEXT,
                tab="validation",
                section="prompt_management",
                placeholder="/path/to/prompt_library.json",
                help_text="Path to custom JSON prompt library",
                tooltip="See user_prompt_library.json.example for format",
                importance=ImportanceLevel.ADVANCED,
                order=4,
            )
        )

        # Eval Dataset ID
        self._add_field(
            ConfigField(
                name="eval_dataset_id",
                arg_name="--eval_dataset_id",
                ui_label="Evaluation Dataset ID",
                field_type=FieldType.TEXT,
                tab="validation",
                section="evaluation",
                placeholder="dataset_name",
                help_text="Specific dataset to use for evaluation metrics",
                tooltip="If not set, uses all datasets. Useful for img2img validation.",
                importance=ImportanceLevel.ADVANCED,
                order=1,
            )
        )

        # Validation Stitch Input Location
        self._add_field(
            ConfigField(
                name="validation_stitch_input_location",
                arg_name="--validation_stitch_input_location",
                ui_label="Input Image Position",
                field_type=FieldType.SELECT,
                tab="validation",
                section="validation_options",
                default_value="left",
                choices=[{"value": "left", "label": "Left"}, {"value": "right", "label": "Right"}],
                help_text="Where to place input image in img2img validations",
                tooltip="For img2img models like DeepFloyd Stage II",
                importance=ImportanceLevel.ADVANCED,
                order=8,
            )
        )

        # Validation Guidance Rescale
        self._add_field(
            ConfigField(
                name="validation_guidance_rescale",
                arg_name="--validation_guidance_rescale",
                ui_label="Guidance Rescale",
                field_type=FieldType.NUMBER,
                tab="validation",
                section="validation_guidance",
                default_value=0.0,
                validation_rules=[
                    ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative"),
                    ValidationRule(ValidationRuleType.MAX, value=1.0, message="Maximum is 1.0"),
                ],
                help_text="CFG rescale value for validation",
                tooltip="Reduces oversaturation from high CFG. 0.0 = disabled, 0.7 = recommended if needed",
                importance=ImportanceLevel.ADVANCED,
                order=4,
            )
        )

    def _add_advanced_config_fields(self):
        """Add advanced configuration fields."""
        # Danger Mode Toggle
        self._add_field(
            ConfigField(
                name="i_know_what_i_am_doing",
                arg_name="--i_know_what_i_am_doing",
                ui_label="I Know What I'm Doing",
                field_type=FieldType.CHECKBOX,
                tab="advanced",
                section="safety_overrides",
                default_value=False,
                help_text="Unlock experimental overrides and bypass built-in safety limits.",
                tooltip="Only enable if you understand the implications. Required for editing prediction type and other safeguards.",
                importance=ImportanceLevel.EXPERIMENTAL,
                order=0,
            )
        )

        # Mixed Precision
        self._add_field(
            ConfigField(
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
                    {"value": "fp8", "label": "FP8 (Experimental)"},
                ],
                help_text="Precision for training computations",
                tooltip="BF16 is recommended for stability. FP16 saves memory but less stable. FP8 is experimental.",
                importance=ImportanceLevel.IMPORTANT,
                order=1,
                dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True)],
            )
        )

        # Gradient Checkpointing
        self._add_field(
            ConfigField(
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
                order=1,
            )
        )

        # Attention Mechanism
        attention_mechanisms = [
            "diffusers",
            "xformers",
            "sageattention",
            "sageattention-int8-fp16-triton",
            "sageattention-int8-fp16-cuda",
            "sageattention-int8-fp8-cuda",
        ]
        self._add_field(
            ConfigField(
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
                order=1,
            )
        )

        # Disable TF32
        self._add_field(
            ConfigField(
                name="disable_tf32",
                arg_name="--disable_tf32",
                ui_label="Disable TF32",
                field_type=FieldType.CHECKBOX,
                tab="advanced",
                section="memory_performance",
                subsection="precision",
                default_value=False,
                platform_specific=["cuda"],
                help_text="Disable TF32 precision on Ampere GPUs",
                tooltip="TF32 is enabled by default on RTX 3000/4000 series. Disabling may reduce performance but increase precision.",
                importance=ImportanceLevel.ADVANCED,
                order=3,
            )
        )

        # Set Grads to None
        self._add_field(
            ConfigField(
                name="set_grads_to_none",
                arg_name="--set_grads_to_none",
                ui_label="Set Gradients to None",
                field_type=FieldType.CHECKBOX,
                tab="advanced",
                section="memory_performance",
                subsection="memory_optimization",
                default_value=False,
                help_text="Set gradients to None instead of zero",
                tooltip="Can save memory and improve performance. May cause issues with some optimizers.",
                importance=ImportanceLevel.EXPERIMENTAL,
                order=2,
            )
        )

        # Noise Offset
        self._add_field(
            ConfigField(
                name="noise_offset",
                arg_name="--noise_offset",
                ui_label="Noise Offset",
                field_type=FieldType.NUMBER,
                tab="advanced",
                section="noise_settings",
                default_value=0.0,
                validation_rules=[
                    ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative"),
                    ValidationRule(ValidationRuleType.MAX, value=1.0, message="Values above 1.0 are extreme"),
                ],
                help_text="Add noise offset to training",
                tooltip="Helps generate darker/lighter images. Common values: 0.05-0.1. 0 = disabled.",
                importance=ImportanceLevel.ADVANCED,
                order=1,
            )
        )

        # Noise Offset Probability
        self._add_field(
            ConfigField(
                name="noise_offset_probability",
                arg_name="--noise_offset_probability",
                ui_label="Noise Offset Probability",
                field_type=FieldType.NUMBER,
                tab="advanced",
                section="noise_settings",
                default_value=0.25,
                validation_rules=[
                    ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                    ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
                ],
                help_text="Probability of applying noise offset",
                tooltip="Apply noise offset this fraction of the time. Default: 25%",
                importance=ImportanceLevel.ADVANCED,
                order=2,
            )
        )

    def _add_loss_config_fields(self):
        """Add loss function configuration fields."""
        # Loss Type
        self._add_field(
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
                help_text="Loss function for training",
                tooltip="L2 is standard. Huber/Smooth L1 are more robust to outliers but may train differently.",
                importance=ImportanceLevel.ADVANCED,
                order=1,
            )
        )

        # Huber Schedule
        self._add_field(
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
        self._add_field(
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
        self._add_field(
            ConfigField(
                name="snr_gamma",
                arg_name="--snr_gamma",
                ui_label="SNR Gamma",
                field_type=FieldType.NUMBER,
                tab="training",
                section="loss_functions",
                order=4,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="SNR gamma must be non-negative")],
                help_text="SNR weighting gamma value (0 = disabled)",
                tooltip="Rebalances loss across timesteps. Recommended value: 5.0. Helps prevent overtraining on easy timesteps.",
                importance=ImportanceLevel.ADVANCED,
                dependencies=[
                    FieldDependency(
                        field="model_family",
                        operator="in",
                        values=["sd1x", "sdxl", "kolors", "deepfloyd", "pixart_sigma"],
                        action="show",
                    )
                ],
            )
        )

        # Masked Loss Probability
        self._add_field(
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
        self._add_field(
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
        self._add_field(
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

    def _add_optimizer_fields(self):
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
        self._add_field(
            ConfigField(
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
        self._add_field(
            ConfigField(
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
                    ValidationRule(ValidationRuleType.MAX, value=1, message="Beta2 must be between 0 and 1"),
                ],
                dependencies=[FieldDependency(field="optimizer", values=adam_like_optimizers)],
                help_text="Second moment decay rate for Adam optimizers",
                tooltip="Controls adaptive learning rates. Default 0.999 is standard. Lower values make adaptation more aggressive.",
                importance=ImportanceLevel.ADVANCED,
                order=2,
            )
        )

        # Adam Weight Decay
        self._add_field(
            ConfigField(
                name="adam_weight_decay",
                arg_name="--adam_weight_decay",
                ui_label="Weight Decay",
                field_type=FieldType.NUMBER,
                tab="training",
                section="optimizer_config",
                subsection="adam_params",
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
        self._add_field(
            ConfigField(
                name="adam_epsilon",
                arg_name="--adam_epsilon",
                ui_label="Adam Epsilon",
                field_type=FieldType.NUMBER,
                tab="training",
                section="optimizer_config",
                subsection="adam_params",
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
        self._add_field(
            ConfigField(
                name="prodigy_steps",
                arg_name="--prodigy_steps",
                ui_label="Prodigy Adjustment Steps",
                field_type=FieldType.NUMBER,
                tab="training",
                section="optimizer_config",
                default_value=0,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
                dependencies=[FieldDependency(field="optimizer", values=["prodigy"])],
                help_text="Number of steps Prodigy should spend adapting its learning rate.",
                tooltip="If unset, SimpleTuner estimates 25% of total training steps. Set explicitly for long or short runs.",
                importance=ImportanceLevel.ADVANCED,
                order=5,
            )
        )

        # Max Gradient Norm
        self._add_field(
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

        # Gradient Clip Method
        self._add_field(
            ConfigField(
                name="grad_clip_method",
                arg_name="--grad_clip_method",
                ui_label="Gradient Clip Method",
                field_type=FieldType.SELECT,
                tab="training",
                section="optimizer_config",
                default_value="value",
                choices=[{"value": "value", "label": "Clip Individual Values"}, {"value": "norm", "label": "Clip By Norm"}],
                help_text="Strategy for applying max_grad_norm during clipping.",
                tooltip="'Value' clips offending gradients individually. 'Norm' rescales all gradients proportionally.",
                importance=ImportanceLevel.ADVANCED,
                order=7,
            )
        )

        # Optimizer Offload Gradients
        self._add_field(
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
        self._add_field(
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
        self._add_field(
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

    def _add_memory_performance_fields(self):
        """Add memory and performance configuration fields."""
        # Gradient Accumulation Steps
        self._add_field(
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
            )
        )

    def _add_logging_fields(self):
        """Add logging and monitoring fields."""
        # Report To
        self._add_field(
            ConfigField(
                name="report_to",
                arg_name="--report_to",
                ui_label="Logging Platform",
                field_type=FieldType.SELECT,
                tab="basic",
                section="logging",
                default_value="wandb",
                choices=[
                    {"value": "tensorboard", "label": "TensorBoard"},
                    {"value": "wandb", "label": "Weights & Biases"},
                    {"value": "comet_ml", "label": "Comet ML"},
                    {"value": "all", "label": "All Platforms"},
                    {"value": "none", "label": "None"},
                ],
                help_text="Where to log training metrics",
                tooltip="WandB provides cloud logging. TensorBoard is local. 'All' logs to all configured platforms.",
                importance=ImportanceLevel.IMPORTANT,
                order=1,
            )
        )

        # Checkpointing Steps
        self._add_field(
            ConfigField(
                name="checkpointing_steps",
                arg_name="--checkpointing_steps",
                ui_label="Checkpoint Every N Steps",
                field_type=FieldType.NUMBER,
                tab="basic",
                section="checkpointing",
                default_value=500,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
                help_text="Save model checkpoint every N steps",
                tooltip="Regular checkpoints let you resume training and test different training stages",
                importance=ImportanceLevel.IMPORTANT,
                order=1,
            )
        )

        # Tracker Run Name
        self._add_field(
            ConfigField(
                name="tracker_run_name",
                arg_name="--tracker_run_name",
                ui_label="Run Name",
                field_type=FieldType.TEXT,
                tab="basic",
                section="project",
                default_value="simpletuner-testing",
                placeholder="simpletuner-testing",
                help_text="Name for this training run in tracking platforms",
                tooltip="Identifies this specific run in WandB/TensorBoard. If not set, uses a generated name.",
                importance=ImportanceLevel.ESSENTIAL,
                order=2,
                dependencies=[FieldDependency(field="report_to", operator="not_equals", value="none", action="show")],
            )
        )

        # Merge environment configuration toggle
        self._add_field(
            ConfigField(
                name="merge_environment_config",
                arg_name="merge_environment_config",
                ui_label="Merge active environment defaults",
                field_type=FieldType.CHECKBOX,
                tab="basic",
                section="project",
                default_value=False,
                help_text="When enabled, values from the Default configuration fill in any unset training options.",
                tooltip="Disable this if you want to have a standalone configuration instead of inheriting the Default configuration.",
                importance=ImportanceLevel.ESSENTIAL,
                order=0,
                custom_component="merge_environment_toggle",
                checkbox_label="Use environment defaults",
                webui_onboarding=True,
            )
        )

        # Tracker Project Name
        self._add_field(
            ConfigField(
                name="tracker_project_name",
                arg_name="--tracker_project_name",
                ui_label="Project Name",
                field_type=FieldType.TEXT,
                tab="basic",
                section="project",
                default_value="simpletuner",
                placeholder="simpletuner",
                help_text="Project name in tracking platforms",
                tooltip="Groups related training runs together in WandB/logging platforms",
                importance=ImportanceLevel.ESSENTIAL,
                order=1,
            )
        )

        # Tracker Image Layout
        self._add_field(
            ConfigField(
                name="tracker_image_layout",
                arg_name="--tracker_image_layout",
                ui_label="Image Layout Style",
                field_type=FieldType.SELECT,
                tab="basic",
                section="logging",
                default_value="gallery",
                choices=[
                    {"value": "gallery", "label": "Gallery (with slider)"},
                    {"value": "table", "label": "Table (row-wise)"},
                ],
                help_text="How validation images are displayed in trackers",
                tooltip="Gallery mode allows easy historical comparison. Table mode shows all at once.",
                importance=ImportanceLevel.ADVANCED,
                order=4,
                dependencies=[FieldDependency(field="report_to", operator="not_equals", value="none", action="show")],
            )
        )

        # Logging Directory
        self._add_field(
            ConfigField(
                name="logging_dir",
                arg_name="--logging_dir",
                ui_label="Local Logging Directory",
                field_type=FieldType.TEXT,
                tab="basic",
                section="logging",
                default_value="logs",
                help_text="Directory for TensorBoard logs",
                tooltip="Local directory where training metrics are saved. Used by TensorBoard.",
                importance=ImportanceLevel.ADVANCED,
                order=5,
                dependencies=[
                    FieldDependency(field="report_to", operator="in", values=["tensorboard", "all"], action="show")
                ],
            )
        )

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
        for dependency_field, expected in condition.items():
            if context.get(dependency_field) != expected:
                return False
        return True

    def get_sections_for_tab(self, tab: str) -> List[Dict[str, Any]]:
        """Get unique sections for a tab with metadata."""
        sections = {}
        fields = [f for f in self._fields.values() if f.tab == tab]

        for cfg_field in fields:
            if cfg_field.section not in sections:
                sections[cfg_field.section] = {
                    "id": cfg_field.section,
                    "title": cfg_field.section.replace("_", " ").title(),
                    "subsections": set(),
                    "empty_message": None,
                }
            if cfg_field.subsection:
                sections[cfg_field.section]["subsections"].add(cfg_field.subsection)

        # Convert sets to lists
        for section_id, section in sections.items():
            section["subsections"] = sorted(list(section["subsections"]))
            if section_id == "text_encoder_training":
                section["empty_message"] = "This model does not support text encoder training."

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
                        {"field": d.field, "value": d.value, "values": d.values, "operator": d.operator}
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
                    "order": f.order,
                }
                for name, f in self._fields.items()
            },
            "dependencies_map": self._dependencies_map,
            "tabs": self._get_tab_structure(),
        }

    def _get_tab_structure(self) -> Dict[str, Any]:
        """Get the structure of all tabs and sections."""
        tabs = {}

        for cfg_field in self._fields.values():
            if cfg_field.tab not in tabs:
                tabs[cfg_field.tab] = {"sections": {}}

            if cfg_field.section not in tabs[cfg_field.tab]["sections"]:
                tabs[cfg_field.tab]["sections"][cfg_field.section] = {
                    "title": cfg_field.section.replace("_", " ").title(),
                    "subsections": set(),
                    "field_count": 0,
                }

            tabs[cfg_field.tab]["sections"][cfg_field.section]["field_count"] += 1
            if cfg_field.subsection:
                tabs[cfg_field.tab]["sections"][cfg_field.section]["subsections"].add(cfg_field.subsection)

        # Convert sets to lists
        for tab in tabs.values():
            for section in tab["sections"].values():
                section["subsections"] = sorted(list(section["subsections"]))

        return tabs

    def get_webui_onboarding_fields(self) -> List[ConfigField]:
        """Return fields that should be treated as WebUI onboarding state only."""

        return [field for field in self._fields.values() if getattr(field, "webui_onboarding", False)]


# Create a singleton instance
field_registry = FieldRegistry()
