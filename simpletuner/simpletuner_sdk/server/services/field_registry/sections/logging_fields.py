import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ValidationRule, ValidationRuleType

if TYPE_CHECKING:
    from ..registry import FieldRegistry


logger = logging.getLogger(__name__)


def register_logging_fields(registry: "FieldRegistry") -> None:
    """Add logging and monitoring fields."""
    # Report To
    registry._add_field(
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
    registry._add_field(
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

    # Checkpointing Rolling Steps
    registry._add_field(
        ConfigField(
            name="checkpointing_rolling_steps",
            arg_name="--checkpointing_rolling_steps",
            ui_label="Rolling Checkpoint Window",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="checkpointing",
            default_value=0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="Rolling checkpoint window size for continuous checkpointing",
            tooltip="When set, maintains a rolling window of recent checkpoints instead of individual files. Higher values keep more history.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
        )
    )

    # Checkpointing Use Tempdir
    registry._add_field(
        ConfigField(
            name="checkpointing_use_tempdir",
            arg_name="--checkpointing_use_tempdir",
            ui_label="Use Temporary Directory for Checkpoints",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="checkpointing",
            default_value=False,
            help_text="Use temporary directory for checkpoint files before final save",
            tooltip="Reduces I/O overhead during checkpointing by writing to temp dir first, then moving to final location.",
            importance=ImportanceLevel.ADVANCED,
            order=3,
        )
    )

    # Checkpoints Rolling Total Limit
    registry._add_field(
        ConfigField(
            name="checkpoints_rolling_total_limit",
            arg_name="--checkpoints_rolling_total_limit",
            ui_label="Rolling Checkpoint Limit",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="checkpointing",
            default_value=1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="Maximum number of rolling checkpoints to keep",
            tooltip="When using rolling checkpoints, limit the number of checkpoints in the rolling window.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
        )
    )

    # Tracker Run Name
    registry._add_field(
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
    registry._add_field(
        ConfigField(
            name="merge_environment_config",
            arg_name="merge_environment_config",
            ui_label="Merge active environment defaults",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="project",
            default_value=None,
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
    registry._add_field(
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
    registry._add_field(
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
    registry._add_field(
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
            dependencies=[FieldDependency(field="report_to", operator="in", values=["tensorboard", "all"], action="show")],
        )
    )

    # Hub Model ID
    registry._add_field(
        ConfigField(
            name="hub_model_id",
            arg_name="--hub_model_id",
            ui_label="Hugging Face Hub Model ID",
            field_type=FieldType.TEXT,
            tab="basic",
            section="project",
            default_value=None,
            placeholder="username/model-name",
            help_text="Model identifier for Hugging Face Hub",
            tooltip="When pushing to Hub, this determines the model URL. If not set, derived from output_dir.",
            importance=ImportanceLevel.ADVANCED,
            order=6,
        )
    )

    # Push to Hub
    registry._add_field(
        ConfigField(
            name="push_to_hub",
            arg_name="--push_to_hub",
            ui_label="Push to Hugging Face Hub",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="project",
            default_value=False,
            help_text="Automatically upload model to Hugging Face Hub",
            tooltip="Requires HF token. Model will be uploaded when training completes or at checkpoints.",
            importance=ImportanceLevel.ADVANCED,
            order=7,
        )
    )

    # Push Checkpoints to Hub
    registry._add_field(
        ConfigField(
            name="push_checkpoints_to_hub",
            arg_name="--push_checkpoints_to_hub",
            ui_label="Push Checkpoints to Hub",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="project",
            default_value=False,
            dependencies=[FieldDependency(field="push_to_hub", operator="equals", value=True, action="show")],
            help_text="Upload intermediate checkpoints to Hub",
            tooltip="Uploads each checkpoint instead of just the final model. May use significant storage.",
            importance=ImportanceLevel.ADVANCED,
            order=8,
        )
    )

    # Model Card Note
    registry._add_field(
        ConfigField(
            name="model_card_note",
            arg_name="--model_card_note",
            ui_label="Model Card Note",
            field_type=FieldType.SELECT,
            tab="basic",
            section="project",
            default_value=None,
            choices=[
                {"value": "", "label": "None"},
                {"value": "private", "label": "Private"},
                {"value": "duplicate", "label": "Duplicate"},
            ],
            help_text="Note for model card on Hub",
            tooltip="Special notes for model card. 'Private' = not publicly visible. 'Duplicate' = copy of existing model.",
            importance=ImportanceLevel.ADVANCED,
            order=9,
        )
    )

    # Model Card Safe for Work
    registry._add_field(
        ConfigField(
            name="model_card_safe_for_work",
            arg_name="--model_card_safe_for_work",
            ui_label="Safe for Work",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="project",
            default_value=False,
            help_text="Mark the uploaded model as safe for work on Hugging Face Hub",
            tooltip="Check if the model should be flagged as safe for work. Leave unchecked to include the default safety warning.",
            importance=ImportanceLevel.ADVANCED,
            order=10,
        )
    )

    # Enable Watermark
    registry._add_field(
        ConfigField(
            name="enable_watermark",
            arg_name="--enable_watermark",
            ui_label="Enable Watermark",
            field_type=FieldType.CHECKBOX,
            tab="advanced",
            section="model_specific",
            default_value=False,
            help_text="Add invisible watermark to generated images",
            tooltip="Experimental feature for tracking model outputs. May affect image quality slightly.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=11,
        )
    )

    # Framerate
    registry._add_field(
        ConfigField(
            name="framerate",
            arg_name="--framerate",
            ui_label="Video Framerate",
            field_type=FieldType.NUMBER,
            tab="advanced",
            section="model_specific",
            default_value=None,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1"),
                ValidationRule(ValidationRuleType.MAX, value=120, message="Must be <= 120"),
            ],
            help_text="Framerate for video model training",
            tooltip="Frames per second for video generation models. Higher = smoother but more compute.",
            importance=ImportanceLevel.ADVANCED,
            order=12,
        )
    )

    # Seed for Each Device
    registry._add_field(
        ConfigField(
            name="seed_for_each_device",
            arg_name="--seed_for_each_device",
            ui_label="Seed Per Device",
            field_type=FieldType.NUMBER,
            tab="advanced",
            section="hardware_config",
            default_value=True,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Additional seed offset per GPU device",
            tooltip="When using multiple GPUs, adds this value to seed per device for more variation.",
            importance=ImportanceLevel.ADVANCED,
            order=13,
        )
    )

    # SNR Weight
    registry._add_field(
        ConfigField(
            name="snr_weight",
            arg_name="--snr_weight",
            ui_label="SNR Weight",
            field_type=FieldType.NUMBER,
            tab="advanced",
            section="loss_functions",
            default_value=0.1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            help_text="Weight factor for SNR-based loss scaling",
            tooltip="Alternative to snr_gamma. Controls how much SNR affects loss weighting.",
            importance=ImportanceLevel.ADVANCED,
            order=14,
        )
    )

    # Rescale Betas Zero SNR
    registry._add_field(
        ConfigField(
            name="rescale_betas_zero_snr",
            arg_name="--rescale_betas_zero_snr",
            ui_label="Rescale Betas Zero SNR",
            field_type=FieldType.CHECKBOX,
            tab="advanced",
            section="loss_functions",
            default_value=False,
            help_text="Rescale betas for zero terminal SNR",
            tooltip="Experimental feature for improved noise scheduling. May affect training dynamics.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=15,
        )
    )

    # Webhook Config
    registry._add_field(
        ConfigField(
            name="webhook_config",
            arg_name="--webhook_config",
            ui_label="Webhook Configuration",
            field_type=FieldType.TEXT,
            tab="advanced",
            section="monitoring",
            default_value=None,
            placeholder="path/to/webhook_config.json",
            help_text="Path to webhook configuration file",
            tooltip="JSON config for external monitoring webhooks. See docs for format.",
            importance=ImportanceLevel.ADVANCED,
            order=16,
        )
    )

    # Webhook Reporting Interval
    registry._add_field(
        ConfigField(
            name="webhook_reporting_interval",
            arg_name="--webhook_reporting_interval",
            ui_label="Webhook Reporting Interval",
            field_type=FieldType.NUMBER,
            tab="advanced",
            section="monitoring",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=30, message="Must be at least 30 seconds")],
            help_text="Interval for webhook reports (seconds)",
            tooltip="How often to send status updates via webhook. Minimum: 30 seconds.",
            importance=ImportanceLevel.ADVANCED,
            order=17,
        )
    )

    # Distillation Method
    registry._add_field(
        ConfigField(
            name="distillation_method",
            arg_name="--distillation_method",
            ui_label="Distillation Method",
            field_type=FieldType.SELECT,
            tab="advanced",
            section="model_specific",
            default_value=None,
            choices=[
                {"value": None, "label": "None"},
                {"value": "lora", "label": "LoRA"},
                {"value": "full", "label": "Full"},
            ],
            help_text="Method for model distillation",
            tooltip="Distillation method to use when converting models. LoRA = lightweight, Full = complete.",
            importance=ImportanceLevel.ADVANCED,
            order=18,
        )
    )

    # Distillation Config
    registry._add_field(
        ConfigField(
            name="distillation_config",
            arg_name="--distillation_config",
            ui_label="Distillation Configuration",
            field_type=FieldType.TEXT,
            tab="advanced",
            section="model_specific",
            default_value=None,
            placeholder="path/to/distillation_config.json",
            help_text="Path to distillation configuration file",
            tooltip="JSON config for distillation parameters and settings.",
            importance=ImportanceLevel.ADVANCED,
            order=19,
        )
    )

    # EMA Validation
    registry._add_field(
        ConfigField(
            name="ema_validation",
            arg_name="--ema_validation",
            ui_label="EMA Validation",
            field_type=FieldType.CHECKBOX,
            tab="advanced",
            section="model_specific",
            default_value="comparison",
            help_text="Use EMA weights for validation instead of main weights",
            tooltip="When enabled, validation uses the EMA copy of the model for more stable results.",
            importance=ImportanceLevel.ADVANCED,
            order=20,
        )
    )

    # Fused QKV Projections (already exists but let's ensure it's correct)
    # This field was already added, so no need to add it again

    # Local Rank
    registry._add_field(
        ConfigField(
            name="local_rank",
            arg_name="--local_rank",
            ui_label="Local Rank",
            field_type=FieldType.NUMBER,
            tab="advanced",
            section="hardware_config",
            default_value=-1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Local rank for distributed training",
            tooltip="For multi-GPU training, specifies the rank of this process. Usually set automatically.",
            importance=ImportanceLevel.ADVANCED,
            order=21,
        )
    )

    # LTX Model Configuration
    registry._add_field(
        ConfigField(
            name="ltx_train_mode",
            arg_name="--ltx_train_mode",
            ui_label="LTX Train Mode",
            field_type=FieldType.SELECT,
            tab="advanced",
            section="model_specific",
            model_specific=["ltx"],
            default_value="video",
            choices=[
                {"value": "video", "label": "Video"},
                {"value": "image", "label": "Image"},
            ],
            help_text="Training mode for LTX models",
            tooltip="Video mode for video generation, Image mode for image generation.",
            importance=ImportanceLevel.ADVANCED,
            order=22,
        )
    )

    registry._add_field(
        ConfigField(
            name="ltx_i2v_prob",
            arg_name="--ltx_i2v_prob",
            ui_label="LTX Image-to-Video Probability",
            field_type=FieldType.NUMBER,
            tab="advanced",
            section="model_specific",
            model_specific=["ltx"],
            default_value=0.1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            help_text="Probability of using image-to-video training for LTX",
            tooltip="Fraction of training that uses image-to-video instead of video-to-video.",
            importance=ImportanceLevel.ADVANCED,
            order=23,
        )
    )

    registry._add_field(
        ConfigField(
            name="ltx_partial_noise_fraction",
            arg_name="--ltx_partial_noise_fraction",
            ui_label="LTX Partial Noise Fraction",
            field_type=FieldType.NUMBER,
            tab="advanced",
            section="model_specific",
            model_specific=["ltx"],
            default_value=0.1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            help_text="Fraction of noise to add for LTX partial training",
            tooltip="Controls how much noise to add during partial training steps.",
            importance=ImportanceLevel.ADVANCED,
            order=24,
        )
    )

    registry._add_field(
        ConfigField(
            name="ltx_protect_first_frame",
            arg_name="--ltx_protect_first_frame",
            ui_label="LTX Protect First Frame",
            field_type=FieldType.CHECKBOX,
            tab="advanced",
            section="model_specific",
            model_specific=["ltx"],
            default_value=False,
            help_text="Protect the first frame from noise in LTX training",
            tooltip="When enabled, first frame is kept clean while subsequent frames get noise.",
            importance=ImportanceLevel.ADVANCED,
            order=25,
        )
    )

    # Offload Parameter Path
    registry._add_field(
        ConfigField(
            name="offload_param_path",
            arg_name="--offload_param_path",
            ui_label="Offload Parameter Path",
            field_type=FieldType.TEXT,
            tab="advanced",
            section="memory_optimization",
            default_value=None,
            placeholder="path/to/offload_params",
            help_text="Path to offloaded parameter files",
            tooltip="Directory where offloaded model parameters are stored for CPU offloading.",
            importance=ImportanceLevel.ADVANCED,
            order=26,
        )
    )

    # Offset Noise
    registry._add_field(
        ConfigField(
            name="offset_noise",
            arg_name="--offset_noise",
            ui_label="Offset Noise",
            field_type=FieldType.CHECKBOX,
            tab="advanced",
            section="noise_settings",
            default_value=False,
            help_text="Enable offset-noise training",
            tooltip="Applies the offset noise technique described by Cross Labs (requires --noise_offset for magnitude)",
            importance=ImportanceLevel.ADVANCED,
            order=27,
        )
    )

    # Quantize Activations
    registry._add_field(
        ConfigField(
            name="quantize_activations",
            arg_name="--quantize_activations",
            ui_label="Quantize Activations",
            field_type=FieldType.CHECKBOX,
            tab="advanced",
            section="quantization",
            default_value=False,
            help_text="Quantize model activations during training",
            tooltip="Experimental feature that quantizes activations to save memory. May affect training quality.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=28,
        )
    )

    # Refiner Training
    registry._add_field(
        ConfigField(
            name="refiner_training",
            arg_name="--refiner_training",
            ui_label="Refiner Training",
            field_type=FieldType.CHECKBOX,
            tab="advanced",
            section="model_specific",
            default_value=False,
            help_text="Enable refiner model training mode",
            tooltip="When enabled, trains the refiner component of SDXL models for higher quality outputs.",
            importance=ImportanceLevel.ADVANCED,
            order=29,
        )
    )

    registry._add_field(
        ConfigField(
            name="refiner_training_invert_schedule",
            arg_name="--refiner_training_invert_schedule",
            ui_label="Invert Refiner Schedule",
            field_type=FieldType.CHECKBOX,
            tab="advanced",
            section="model_specific",
            default_value=False,
            dependencies=[FieldDependency(field="refiner_training", operator="equals", value=True, action="show")],
            help_text="Invert the noise schedule for refiner training",
            tooltip="When enabled, inverts the timestep schedule for refiner training.",
            importance=ImportanceLevel.ADVANCED,
            order=30,
        )
    )

    registry._add_field(
        ConfigField(
            name="refiner_training_strength",
            arg_name="--refiner_training_strength",
            ui_label="Refiner Training Strength",
            field_type=FieldType.NUMBER,
            tab="advanced",
            section="model_specific",
            default_value=0.2,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be <= 1.0"),
            ],
            dependencies=[FieldDependency(field="refiner_training", operator="equals", value=True, action="show")],
            help_text="Strength of refiner training",
            tooltip="Controls how much the refiner affects the final output. Higher = stronger effect.",
            importance=ImportanceLevel.ADVANCED,
            order=31,
        )
    )

    # SDXL Refiner Full Range
    registry._add_field(
        ConfigField(
            name="sdxl_refiner_uses_full_range",
            arg_name="--sdxl_refiner_uses_full_range",
            ui_label="SDXL Refiner Full Range",
            field_type=FieldType.CHECKBOX,
            tab="advanced",
            section="model_specific",
            default_value=False,
            help_text="Use full timestep range for SDXL refiner",
            tooltip="When enabled, refiner uses full timestep range instead of just high timesteps.",
            importance=ImportanceLevel.ADVANCED,
            order=32,
        )
    )

    # Sana Complex Human Instruction
    registry._add_field(
        ConfigField(
            name="sana_complex_human_instruction",
            arg_name="--sana_complex_human_instruction",
            ui_label="Sana Complex Human Instruction",
            field_type=FieldType.TEXT,
            tab="advanced",
            section="model_specific",
            model_specific=["sana"],
            default_value=[
                "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
                "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
                "- Do not add detail that is not in the original prompt! Do not mention colors if the User Prompt does not!",
                "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
                "- If the prompt contains explicit material, do not remove this information from the output, which would damage the performance of downstream classifiers.",
                "Here are examples of how to transform or refine prompts:",
                "- User Prompt: A cat sleeping -> Enhanced: A cat sleeping peacefully, showcasing the joy of pet ownership. Cute floof kitty cat gatto.",
                "- User Prompt: A busy city street -> Enhanced: A bustling city street scene featuring a crowd of people.",
                "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
                "User Prompt: ",
            ],
            placeholder="complex human instruction",
            help_text="Complex human instruction for Sana model training",
            tooltip="Special instruction format for Sana model training with complex human prompts.",
            importance=ImportanceLevel.ADVANCED,
            order=33,
        )
    )

    # Help Field (for CLI help display)
    registry._add_field(
        ConfigField(
            name="help",
            arg_name="--help",
            ui_label="Show Help",
            field_type=FieldType.CHECKBOX,
            tab="advanced",
            section="system",
            default_value="==SUPPRESS==",
            help_text="Display help information and exit",
            tooltip="When enabled, shows the help message for all available command line options and exits.",
            importance=ImportanceLevel.ADVANCED,
            order=34,
        )
    )
