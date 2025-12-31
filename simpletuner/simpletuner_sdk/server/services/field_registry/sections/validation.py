import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ParserType, ValidationRule, ValidationRuleType

if TYPE_CHECKING:
    from ..registry import FieldRegistry


logger = logging.getLogger(__name__)


def register_validation_fields(registry: "FieldRegistry") -> None:
    """Add validation configuration fields."""
    # Validation Step Interval
    registry._add_field(
        ConfigField(
            name="validation_step_interval",
            arg_name="--validation_step_interval",
            aliases=["--validation_steps"],
            ui_label="Validation Step Interval",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_schedule",
            default_value=100,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Validation steps must be positive")],
            help_text="Run validation every N training steps",
            tooltip="How often to generate validation images during training. Lower = more frequent validation.",
            importance=ImportanceLevel.IMPORTANT,
            order=1,
        )
    )

    # Validation Epoch Interval
    registry._add_field(
        ConfigField(
            name="validation_epoch_interval",
            arg_name="--validation_epoch_interval",
            ui_label="Validation Epoch Interval",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_schedule",
            default_value=None,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Validation epoch interval must be positive")
            ],
            help_text="Run validation every N training epochs (leave blank to disable)",
            tooltip="Schedule validation runs based on completed epochs. Combine with step interval for finer control.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
        )
    )

    registry._add_field(
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
            order=3,
            subsection="advanced",
            documentation="OPTIONS.md#--disable_benchmark",
        )
    )

    registry._add_field(
        ConfigField(
            name="validation_preview",
            arg_name="--validation_preview",
            ui_label="Stream Validation Previews",
            field_type=FieldType.CHECKBOX,
            tab="validation",
            section="validation_schedule",
            default_value=False,
            help_text="Decode intermediate validation latents with Tiny AutoEncoders and stream them over webhook callbacks.",
            tooltip="Enables live previews via webhook for supported models. Requires a webhook configuration.",
            warning="Only available on model families with Tiny AutoEncoder support.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
            subsection="advanced",
            documentation="OPTIONS.md#--validation_preview",
        )
    )

    registry._add_field(
        ConfigField(
            name="validation_preview_steps",
            arg_name="--validation_preview_steps",
            ui_label="Preview Step Interval",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_schedule",
            default_value=1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Preview interval must be >= 1")],
            help_text="Always emit the first preview, then emit every N sampling steps thereafter.",
            tooltip="Set >1 to throttle preview decoding after the initial preview if the Tiny AutoEncoder adds overhead.",
            importance=ImportanceLevel.ADVANCED,
            order=5,
            subsection="advanced",
            documentation="OPTIONS.md#--validation_preview_steps",
        )
    )

    # Validation Prompt
    registry._add_field(
        ConfigField(
            name="validation_prompt",
            arg_name="--validation_prompt",
            ui_label="Validation Prompt",
            field_type=FieldType.TEXTAREA,
            tab="validation",
            section="prompt_management",
            placeholder="e.g. a photo of a cat, highly detailed, 4k (leave empty to skip)",
            help_text="Prompt to use for validation images",
            tooltip="This prompt will be used to generate images during training to monitor progress. Leave empty to skip prompt-based validation.",
            importance=ImportanceLevel.IMPORTANT,
            order=1,
            allow_empty=True,
        )
    )

    # Validation Lyrics
    registry._add_field(
        ConfigField(
            name="validation_lyrics",
            arg_name="--validation_lyrics",
            ui_label="Validation Lyrics",
            field_type=FieldType.TEXTAREA,
            tab="validation",
            section="prompt_management",
            placeholder="Enter lyrics for audio validation",
            help_text="Lyrics to use for audio validation",
            tooltip="Provide lyrics for music generation validation. Only used by audio models.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
            allow_empty=True,
            model_specific=["ace_step"],
        )
    )

    # Validation Audio Duration
    registry._add_field(
        ConfigField(
            name="validation_audio_duration",
            arg_name="--validation_audio_duration",
            ui_label="Validation Audio Duration",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_schedule",
            default_value=30.0,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1.0, message="Duration must be at least 1 second"),
                ValidationRule(ValidationRuleType.MAX, value=300.0, message="Duration recommended to be under 300s"),
            ],
            help_text="Duration of generated audio for validation (seconds)",
            tooltip="Length of the audio clip to generate during validation runs.",
            importance=ImportanceLevel.ADVANCED,
            order=6,
            model_specific=["ace_step"],
        )
    )

    # Number of Validation Images
    registry._add_field(
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
            subsection="advanced",
        )
    )

    # Number of Eval Images
    registry._add_field(
        ConfigField(
            name="num_eval_images",
            arg_name="--num_eval_images",
            ui_label="Evaluation Images Count",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="evaluation",
            default_value=4,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Must generate at least 1 evaluation image"),
                ValidationRule(ValidationRuleType.MAX, value=10, message="Generating >10 images may slow training"),
            ],
            help_text="Number of images to generate for evaluation metrics",
            tooltip="More images give better evaluation metrics but take longer to generate",
            importance=ImportanceLevel.ADVANCED,
            order=3,
        )
    )

    # Eval Steps Interval
    registry._add_field(
        ConfigField(
            name="eval_steps_interval",
            arg_name="--eval_steps_interval",
            ui_label="Evaluation Steps Interval",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="evaluation",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="Run evaluation every N training steps (leave blank to disable)",
            tooltip="How often to run evaluation metrics during training. Clear this field to skip scheduled evaluation runs.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
        )
    )

    registry._add_field(
        ConfigField(
            name="eval_epoch_interval",
            arg_name="--eval_epoch_interval",
            ui_label="Evaluation Epoch Interval",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="evaluation",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0001, message="Must be greater than 0")],
            help_text="Run evaluation every N epochs (decimals run multiple times per epoch)",
            tooltip="Supports fractional epochs, e.g. 0.5 evaluates twice per epoch. Leave blank to disable.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
            parser_type=ParserType.FLOAT,
        )
    )

    # Eval Timesteps
    registry._add_field(
        ConfigField(
            name="eval_timesteps",
            arg_name="--eval_timesteps",
            ui_label="Number of Evaluation Steps",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="evaluation",
            default_value=28,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="Number of timesteps for evaluation",
            tooltip="Lower values speed up evaluation at the cost of quality. Typical range: 20-30.",
            importance=ImportanceLevel.ADVANCED,
            order=5,
        )
    )

    # Eval Dataset Pooling
    registry._add_field(
        ConfigField(
            name="eval_dataset_pooling",
            arg_name="--eval_dataset_pooling",
            ui_label="Evaluation Dataset Pooling",
            field_type=FieldType.CHECKBOX,
            tab="validation",
            section="evaluation",
            default_value=False,
            help_text="Combine evaluation metrics from all evaluation datasets into a single chart",
            importance=ImportanceLevel.ADVANCED,
            subsection="advanced",
            order=6,
        )
    )

    # Disable Validation Loss
    registry._add_field(
        ConfigField(
            name="eval_loss_disable",
            arg_name="--eval_loss_disable",
            ui_label="Disable Validation Loss",
            field_type=FieldType.CHECKBOX,
            tab="validation",
            section="evaluation",
            default_value=False,
            help_text="Disable validation loss computation on eval datasets",
            tooltip="Skip computing loss on evaluation datasets during training. CLIP scoring (if enabled) still runs.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
        )
    )

    # Evaluation Type
    registry._add_field(
        ConfigField(
            name="evaluation_type",
            arg_name="--evaluation_type",
            ui_label="Evaluation Type",
            field_type=FieldType.SELECT,
            tab="validation",
            section="evaluation",
            default_value="none",
            choices=[
                {"value": "none", "label": "None"},
                {"value": "clip", "label": "CLIP Score"},
            ],
            help_text="Type of evaluation metrics to compute",
            tooltip="Choose evaluation metric for validation runs. CLIP measures text-image alignment.",
            importance=ImportanceLevel.ADVANCED,
            order=1,
            documentation="OPTIONS.md#--evaluation_type",
        )
    )

    # Pretrained Evaluation Model Path
    registry._add_field(
        ConfigField(
            name="pretrained_evaluation_model_name_or_path",
            arg_name="--pretrained_evaluation_model_name_or_path",
            ui_label="Evaluation Model Path",
            field_type=FieldType.TEXT,
            tab="validation",
            section="evaluation",
            default_value="openai/clip-vit-large-patch14-336",
            placeholder="path/to/evaluation_model",
            help_text="Path to pretrained model for evaluation metrics",
            tooltip="HuggingFace model ID or local path for evaluation model (e.g., CLIP for CLIP score).",
            importance=ImportanceLevel.ADVANCED,
            subsection="advanced",
            order=7,
        )
    )

    # Validation Guidance Scale
    registry._add_field(
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
    registry._add_field(
        ConfigField(
            name="validation_num_inference_steps",
            arg_name="--validation_num_inference_steps",
            ui_label="Inference Steps",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_schedule",
            default_value=30,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1 step")],
            help_text="Number of diffusion steps for validation renders",
            tooltip="Lower values speed up validation at the cost of quality. Typical range: 20-30.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
            subsection="advanced",
        )
    )

    # Validation on Startup
    registry._add_field(
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
            subsection="advanced",
        )
    )

    # Validation Using Datasets
    registry._add_field(
        ConfigField(
            name="validation_method",
            arg_name="--validation_method",
            ui_label="Validation Method",
            field_type=FieldType.SELECT,
            tab="validation",
            section="validation_options",
            default_value="simpletuner-local",
            choices=[
                {"label": "SimpleTuner (local pipeline)", "value": "simpletuner-local"},
                {"label": "External script", "value": "external-script"},
            ],
            help_text="Choose how validation runs are executed.",
            tooltip="Default runs validations locally. Select external-script to call a custom executable instead of the built-in pipeline.",
            importance=ImportanceLevel.IMPORTANT,
            order=1,
            documentation="OPTIONS.md#--validation_method",
        )
    )

    registry._add_field(
        ConfigField(
            name="validation_external_script",
            arg_name="--validation_external_script",
            ui_label="External Validation Script",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_options",
            default_value=None,
            placeholder="/path/to/script {local_checkpoint_path}",
            help_text="Executable to run when validation_method is external-script.",
            tooltip="Supports placeholders like {local_checkpoint_path} to pass the latest checkpoint directory to your script.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
            dependencies=[
                FieldDependency(field="validation_method", operator="equals", value="external-script", action="show")
            ],
            documentation="OPTIONS.md#--validation_external_script",
        )
    )

    registry._add_field(
        ConfigField(
            name="validation_external_background",
            arg_name="--validation_external_background",
            ui_label="Run External Validation in Background",
            field_type=FieldType.CHECKBOX,
            tab="validation",
            section="validation_options",
            default_value=False,
            help_text="Start the external validation script and immediately return without waiting.",
            tooltip="When enabled, the external script runs without blocking training and its exit status is not checked.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
            dependencies=[
                FieldDependency(field="validation_method", operator="equals", value="external-script", action="show")
            ],
            documentation="OPTIONS.md#--validation_external_background",
        )
    )

    # Validation Using Datasets
    registry._add_field(
        ConfigField(
            name="validation_using_datasets",
            arg_name="--validation_using_datasets",
            ui_label="Validate Using Dataset Images",
            field_type=FieldType.CHECKBOX,
            tab="validation",
            section="validation_options",
            default_value=None,
            help_text="Use random images from training datasets for validation",
            tooltip="Alternative to validation prompts. Be mindful of privacy when sharing results.",
            importance=ImportanceLevel.ADVANCED,
            order=3,
            subsection="advanced",
        )
    )

    # Validation Guidance Real
    registry._add_field(
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
            model_specific=["flux"],
        )
    )

    # Validation No CFG Until Timestep
    registry._add_field(
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
            model_specific=["flux"],
        )
    )

    # Validation Negative Prompt
    registry._add_field(
        ConfigField(
            name="validation_negative_prompt",
            arg_name="--validation_negative_prompt",
            ui_label="Negative Prompt",
            field_type=FieldType.TEXTAREA,
            tab="validation",
            section="prompt_management",
            default_value="blurry, cropped, ugly",
            placeholder="e.g. blurry, cropped, ugly (leave empty to disable)",
            help_text="Negative prompt for validation images",
            tooltip="What to avoid in generated images. Set to empty string to disable.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
            allow_empty=True,
        )
    )

    # Validation Randomize
    registry._add_field(
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
    registry._add_field(
        ConfigField(
            name="validation_seed",
            arg_name="--validation_seed",
            ui_label="Validation Seed",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_options",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="validation_randomize", operator="equals", value=False, action="show")],
            help_text="Fixed seed for reproducible validation images",
            tooltip="Use the same seed to compare training progress consistently",
            importance=ImportanceLevel.ADVANCED,
            order=6,
        )
    )

    # Validation Multi-GPU Mode
    registry._add_field(
        ConfigField(
            name="validation_multigpu",
            arg_name="--validation_multigpu",
            ui_label="Validation Multi-GPU Mode",
            field_type=FieldType.SELECT,
            tab="validation",
            section="validation_options",
            default_value="batch-parallel",
            choices=[
                {"label": "Single GPU (rank 0 only)", "value": "single-gpu"},
                {"label": "Batch Parallel (all ranks)", "value": "batch-parallel"},
            ],
            help_text="Select how validation workloads are distributed across processes.",
            tooltip="Batch-parallel runs validation on every process and gathers the results. Single-GPU keeps the legacy main-rank behaviour.",
            importance=ImportanceLevel.ADVANCED,
            order=7,
        )
    )

    # Validation Disable
    registry._add_field(
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
    registry._add_field(
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
    registry._add_field(
        ConfigField(
            name="user_prompt_library",
            arg_name="--user_prompt_library",
            ui_label="Custom Prompt Library Path",
            field_type=FieldType.TEXT,
            tab="validation",
            section="prompt_management",
            placeholder="/path/to/prompt_library.json",
            tooltip="See user_prompt_library.json.example for format",
            importance=ImportanceLevel.ADVANCED,
            order=4,
            custom_component="prompt_library_path",
        )
    )

    # Eval Dataset ID
    registry._add_field(
        ConfigField(
            name="eval_dataset_id",
            arg_name="--eval_dataset_id",
            ui_label="Evaluation Dataset ID",
            field_type=FieldType.SELECT,
            tab="validation",
            section="evaluation",
            default_value=None,
            help_text="Specific evaluation dataset to use (leave unselected to use all)",
            tooltip="Choose an evaluation dataset configured on the Datasets tab. Leave blank to evaluate all datasets.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
            dynamic_choices=True,
        )
    )

    # Validation Stitch Input Location
    registry._add_field(
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
            subsection="advanced",
        )
    )

    # Validation Guidance Rescale
    registry._add_field(
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
            model_specific=["sdxl", "sd1x", "sd2x"],
        )
    )

    # Validation Disable Unconditional
    registry._add_field(
        ConfigField(
            name="validation_disable_unconditional",
            arg_name="--validation_disable_unconditional",
            ui_label="Disable Unconditional Generation",
            field_type=FieldType.CHECKBOX,
            tab="validation",
            section="validation_options",
            default_value=False,
            help_text="Disable unconditional image generation during validation",
            tooltip="When enabled, only generates conditional images (with prompts) during validation.",
            importance=ImportanceLevel.ADVANCED,
            order=9,
        )
    )

    # Validation Guidance Skip Layers
    registry._add_field(
        ConfigField(
            name="validation_guidance_skip_layers",
            arg_name="--validation_guidance_skip_layers",
            ui_label="Skip Guidance Layers",
            field_type=FieldType.TEXTAREA,
            tab="validation",
            section="validation_options",
            default_value=None,
            placeholder="e.g. [7, 8, 9]",
            help_text="JSON list of transformer layers to skip during classifier-free guidance",
            tooltip="Provide a JSON array such as [7, 8, 9] to disable guidance on specific layers.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=10,
            subsection="advanced",
            dependencies=[
                FieldDependency(field="model_family", operator="in", values=["auraflow", "chroma", "sd3", "wan"]),
            ],
        )
    )

    # Validation Guidance Skip Layers Start
    registry._add_field(
        ConfigField(
            name="validation_guidance_skip_layers_start",
            arg_name="--validation_guidance_skip_layers_start",
            ui_label="Skip Layers Start",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_options",
            default_value=0.01,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Starting layer index to skip guidance",
            tooltip="Skip guidance computation from this layer onwards.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=11,
            subsection="advanced",
            dependencies=[
                FieldDependency(field="model_family", operator="in", values=["auraflow", "chroma", "sd3", "wan"]),
            ],
        )
    )

    # Validation Guidance Skip Layers Stop
    registry._add_field(
        ConfigField(
            name="validation_guidance_skip_layers_stop",
            arg_name="--validation_guidance_skip_layers_stop",
            ui_label="Skip Layers Stop",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_options",
            default_value=0.2,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Ending layer index to skip guidance",
            tooltip="Skip guidance computation up to this layer.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=12,
            subsection="advanced",
            dependencies=[
                FieldDependency(field="model_family", operator="in", values=["auraflow", "chroma", "sd3", "wan"]),
            ],
        )
    )

    # Validation Guidance Skip Scale
    registry._add_field(
        ConfigField(
            name="validation_guidance_skip_scale",
            arg_name="--validation_guidance_skip_scale",
            ui_label="Skip Guidance Scale",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_options",
            default_value=2.8,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            help_text="Scale guidance strength when applying layer skipping",
            tooltip="Increase this when skipping more layers to maintain output quality.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=13,
            subsection="advanced",
            dependencies=[
                FieldDependency(field="model_family", operator="in", values=["auraflow", "chroma", "sd3", "wan"]),
            ],
        )
    )

    # Validation LyCORIS Strength
    registry._add_field(
        ConfigField(
            name="validation_lycoris_strength",
            arg_name="--validation_lycoris_strength",
            ui_label="LyCORIS Validation Strength",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_options",
            default_value=1.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            help_text="Strength multiplier for LyCORIS validation",
            tooltip="Adjust the strength of LyCORIS effects during validation.",
            importance=ImportanceLevel.ADVANCED,
            order=14,
            subsection="advanced",
            dependencies=[
                FieldDependency(field="model_type", operator="equals", value="lora", action="show"),
                FieldDependency(field="lora_type", operator="equals", value="lycoris", action="show"),
            ],
        )
    )

    # Validation Noise Scheduler
    registry._add_field(
        ConfigField(
            name="validation_noise_scheduler",
            arg_name="--validation_noise_scheduler",
            ui_label="Validation Noise Scheduler",
            field_type=FieldType.SELECT,
            tab="validation",
            section="validation_options",
            default_value=None,
            choices=[
                {"value": None, "label": "Model Default"},
                {"value": "ddim", "label": "DDIM"},
                {"value": "ddpm", "label": "DDPM"},
                {"value": "euler", "label": "Euler"},
                {"value": "euler-a", "label": "Euler A"},
                {"value": "unipc", "label": "UniPC"},
                {"value": "dpm++", "label": "DPM++"},
                {"value": "perflow", "label": "PerFlow"},
            ],
            help_text="Noise scheduler for validation",
            tooltip="Override the scheduler used for validation renders, otherwise use the model default.",
            importance=ImportanceLevel.ADVANCED,
            order=15,
        )
    )

    # Validation Num Video Frames
    registry._add_field(
        ConfigField(
            name="validation_num_video_frames",
            arg_name="--validation_num_video_frames",
            ui_label="Validation Video Frames",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_options",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="Number of frames for video validation",
            tooltip="For video models, number of frames to generate during validation.",
            importance=ImportanceLevel.ADVANCED,
            order=16,
        )
    )

    # Validation Resolution
    registry._add_field(
        ConfigField(
            name="validation_resolution",
            arg_name="--validation_resolution",
            ui_label="Validation Resolution",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_options",
            default_value="256",
            help_text="Override resolution for validation images (pixels or megapixels)",
            tooltip="Enter a pixel size like 512 or a megapixel value like 0.5 to auto-convert.",
            importance=ImportanceLevel.ADVANCED,
            order=17,
            documentation="OPTIONS.md#--validation_resolution",
        )
    )

    # Validation Seed Source
    registry._add_field(
        ConfigField(
            name="validation_seed_source",
            arg_name="--validation_seed_source",
            ui_label="Validation Seed Source",
            field_type=FieldType.SELECT,
            tab="validation",
            section="validation_options",
            default_value="cpu",
            choices=[
                {"value": "cpu", "label": "CPU"},
                {"value": "gpu", "label": "GPU"},
            ],
            help_text="Source device used to generate validation seeds",
            tooltip="Use CPU-based or GPU-based RNG when deriving validation seeds.",
            importance=ImportanceLevel.ADVANCED,
            order=18,
            subsection="advanced",
        )
    )

    registry._add_field(
        ConfigField(
            name="validation_adapter_path",
            arg_name="--validation_adapter_path",
            ui_label="Validation Adapter Path",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_adapters",
            default_value=None,
            placeholder="repo/id:weights.safetensors or /path/to/adapter.safetensors",
            help_text="Temporarily load a single LoRA adapter during validation from a local file or Hugging Face repo.",
            tooltip="Formats: 'org/repo:weights.safetensors', 'org/repo' (defaults to pytorch_lora_weights.safetensors) or a local path.",
            importance=ImportanceLevel.ADVANCED,
            order=1,
            documentation="OPTIONS.md#--validation_adapter_path",
        )
    )

    registry._add_field(
        ConfigField(
            name="validation_adapter_name",
            arg_name="--validation_adapter_name",
            ui_label="Validation Adapter Name",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_adapters",
            default_value=None,
            placeholder="my_adapter_name",
            help_text="Optional adapter identifier to use when loading LoRA weights for validation.",
            tooltip="If left blank, SimpleTuner generates a unique adapter name automatically.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
            documentation="OPTIONS.md#--validation_adapter_name",
        )
    )

    registry._add_field(
        ConfigField(
            name="validation_adapter_strength",
            arg_name="--validation_adapter_strength",
            ui_label="Validation Adapter Strength",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_adapters",
            default_value=1.0,
            help_text="Strength multiplier applied when activating the validation adapter.",
            tooltip="Values greater than 1 increase the LoRA influence; values between 0 and 1 reduce it.",
            importance=ImportanceLevel.ADVANCED,
            order=3,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Strength must be greater than 0"),
            ],
            documentation="OPTIONS.md#--validation_adapter_strength",
        )
    )

    registry._add_field(
        ConfigField(
            name="validation_adapter_mode",
            arg_name="--validation_adapter_mode",
            ui_label="Validation Adapter Comparison",
            field_type=FieldType.SELECT,
            tab="validation",
            section="validation_adapters",
            default_value="adapter_only",
            choices=[
                {"value": "adapter_only", "label": "Adapter Only"},
                {"value": "comparison", "label": "Comparison"},
                {"value": "none", "label": "Disabled"},
            ],
            help_text="Select whether to sample only the adapter, compare against the base model, or skip loading it.",
            tooltip="Comparison renders both with and without the adapter so you can review differences.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
            documentation="OPTIONS.md#--validation_adapter_mode",
        )
    )

    registry._add_field(
        ConfigField(
            name="validation_adapter_config",
            arg_name="--validation_adapter_config",
            ui_label="Validation Adapter Config",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_adapters",
            default_value=None,
            placeholder="/path/to/validation_adapters.json",
            help_text="JSON file or inline JSON describing multiple adapter combinations to evaluate during validation.",
            tooltip="Each entry can define 'label' and a list of adapter paths so multiple validation runs are automated.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=5,
            documentation="OPTIONS.md#--validation_adapter_config",
        )
    )
