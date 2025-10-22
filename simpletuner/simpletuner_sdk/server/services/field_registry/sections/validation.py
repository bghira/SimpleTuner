import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ValidationRule, ValidationRuleType

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
            placeholder="a photo of a cat, highly detailed, 4k",
            help_text="Prompt to use for validation images",
            tooltip="This prompt will be used to generate images during training to monitor progress",
            importance=ImportanceLevel.IMPORTANT,
            order=1,
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
            webui_only=True,  # WebUI-specific field, not passed to trainer
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

    # Validation Torch Compile
    registry._add_field(
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
            subsection="advanced",
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show")],
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
            help_text="Negative prompt for validation images",
            tooltip="What to avoid in generated images. Set to empty string to disable.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
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
            help_text="Path to custom JSON prompt library",
            tooltip="See user_prompt_library.json.example for format",
            importance=ImportanceLevel.ADVANCED,
            order=4,
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

    # Validation Torch Compile Mode
    registry._add_field(
        ConfigField(
            name="validation_torch_compile_mode",
            arg_name="--validation_torch_compile_mode",
            ui_label="Torch Compile Mode",
            field_type=FieldType.SELECT,
            tab="validation",
            section="validation_options",
            default_value="max-autotune",
            choices=[
                {"value": "default", "label": "Default"},
                {"value": "reduce-overhead", "label": "Reduce Overhead"},
                {"value": "max-autotune", "label": "Max Autotune"},
            ],
            help_text="Torch compile mode for validation",
            tooltip="Different compilation modes for torch.compile() during validation.",
            importance=ImportanceLevel.ADVANCED,
            order=19,
            subsection="advanced",
        )
    )
