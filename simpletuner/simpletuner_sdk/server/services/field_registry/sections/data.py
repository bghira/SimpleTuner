import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from simpletuner.helpers.image_manipulation.nsfw_classifier import DEFAULT_NSFW_CHECK_MODELS_CSV

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ValidationRule, ValidationRuleType

if TYPE_CHECKING:
    from ..registry import FieldRegistry


logger = logging.getLogger(__name__)


def register_data_fields(registry: "FieldRegistry") -> None:
    """Add data configuration fields."""
    # Resolution
    registry._add_field(
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
    registry._add_field(
        ConfigField(
            name="resolution_type",
            arg_name="--resolution_type",
            ui_label="Resolution Type",
            field_type=FieldType.SELECT,
            tab="basic",
            section="image_processing",
            subsection="advanced",
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
            documentation="OPTIONS.md#--resolution_type",
        )
    )

    # Data Backend Config
    registry._add_field(
        ConfigField(
            name="data_backend_config",
            arg_name="--data_backend_config",
            ui_label="Data Backend Config",
            field_type=FieldType.SELECT,
            tab="basic",
            section="data_config",
            default_value=None,
            choices=[],
            validation_rules=[ValidationRule(ValidationRuleType.REQUIRED, message="Select a data backend configuration")],
            help_text="Select a saved dataset configuration (managed in Datasets & Environments tabs)",
            tooltip="Pick which dataset plan to use. Create or edit datasets in the Datasets tab; manage saved plans from Environments.",
            importance=ImportanceLevel.ESSENTIAL,
            order=1,
            dynamic_choices=True,
            documentation="OPTIONS.md#--data_backend_config",
        )
    )

    # Caption Strategy
    registry._add_field(
        ConfigField(
            name="caption_strategy",
            arg_name="--caption_strategy",
            ui_label="Caption Strategy",
            field_type=FieldType.SELECT,
            tab="basic",
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
            documentation="OPTIONS.md#--caption_strategy",
        )
    )

    # Conditioning Multidataset Sampling
    registry._add_field(
        ConfigField(
            name="conditioning_multidataset_sampling",
            arg_name="--conditioning_multidataset_sampling",
            ui_label="Conditioning Multidataset Sampling",
            field_type=FieldType.SELECT,
            tab="basic",
            section="dataset_config",
            default_value="random",
            choices=[
                {"value": "combined", "label": "Combined"},
                {"value": "random", "label": "Random"},
            ],
            help_text="How to sample from multiple conditioning datasets",
            tooltip="'combined': Use all conditioning images. 'random': Select one dataset per sample.",
            importance=ImportanceLevel.ADVANCED,
            order=3,
        )
    )

    # Instance Prompt
    registry._add_field(
        ConfigField(
            name="instance_prompt",
            arg_name="--instance_prompt",
            ui_label="Instance Prompt",
            field_type=FieldType.TEXT,
            tab="basic",
            section="dataset_config",
            default_value=None,
            placeholder="a photo of sks",
            help_text="Default caption for datasets using 'instance_prompt' caption strategy without their own prompt",
            tooltip="Used when caption_strategy is 'instance_prompt'. Defines the concept being trained.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
        )
    )

    # Disable Multiline Split
    registry._add_field(
        ConfigField(
            name="disable_multiline_split",
            arg_name=None,
            ui_label="Disable Multiline Caption Split",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="dataset_defaults",
            default_value=False,
            help_text="Keep text file captions as a single string instead of splitting by newlines",
            tooltip="When enabled, captions from .txt files will not be split into multiple variants by newlines. Useful for captions that contain intentional line breaks.",
            importance=ImportanceLevel.ADVANCED,
            order=5,
        )
    )

    # Parquet Caption Column
    registry._add_field(
        ConfigField(
            name="parquet_caption_column",
            arg_name="--parquet_caption_column",
            ui_label="Parquet Caption Column",
            field_type=FieldType.TEXT,
            tab="basic",
            section="dataset_config",
            default_value=None,
            placeholder="caption",
            help_text="Column name containing captions in parquet files",
            tooltip="When using parquet datasets, specifies which column contains the text captions.",
            importance=ImportanceLevel.ADVANCED,
            order=5,
        )
    )

    # Parquet Filename Column
    registry._add_field(
        ConfigField(
            name="parquet_filename_column",
            arg_name="--parquet_filename_column",
            ui_label="Parquet Filename Column",
            field_type=FieldType.TEXT,
            tab="basic",
            section="dataset_config",
            default_value=None,
            placeholder="image_path",
            help_text="Column name containing image paths in parquet files",
            tooltip="When using parquet datasets, specifies which column contains the image file paths.",
            importance=ImportanceLevel.ADVANCED,
            order=6,
        )
    )

    # Ignore Missing Files
    registry._add_field(
        ConfigField(
            name="ignore_missing_files",
            arg_name="--ignore_missing_files",
            ui_label="Keep missing images in buckets",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="training_data",
            subsection="advanced",
            default_value=False,
            help_text="Prevent trainer from removing files from buckets if they're missing from disk, allowing removal of files from disk without breaking training",
            tooltip="When enabled, missing files are logged but don't stop training. Useful for large datasets.",
            importance=ImportanceLevel.ADVANCED,
            order=7,
        )
    )

    # VAE Cache Scan Behaviour
    registry._add_field(
        ConfigField(
            name="vae_cache_scan_behaviour",
            arg_name="--vae_cache_scan_behaviour",
            ui_label="VAE Cache Scan Behaviour",
            field_type=FieldType.SELECT,
            tab="model",
            section="vae_config",
            default_value="recreate",
            choices=[
                {"value": "recreate", "label": "Recreate"},
                {"value": "sync", "label": "Sync"},
            ],
            help_text="How to scan VAE cache for missing files",
            tooltip="'recreate': rebuild inconsistent cache entries. 'sync': adjust bucket metadata to match existing latents.",
            importance=ImportanceLevel.ADVANCED,
            order=20,
            documentation="OPTIONS.md#--vae_cache_scan_behaviour",
        )
    )

    nsfw_check_enabled = [FieldDependency(field="enable_nsfw_check", operator="equals", value=True, action="show")]

    # NSFW Classifier Checks
    registry._add_field(
        ConfigField(
            name="enable_nsfw_check",
            arg_name="--enable_nsfw_check",
            ui_label="Enable NSFW Classifier Check",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="vae_config",
            default_value=False,
            help_text="Scan uncached VAE preprocessing samples with Transformers image classifiers and reject NSFW samples",
            tooltip="Runs during VAE cache preprocessing for uncached image, video, and conditioning samples. Cached datasets and skip_file_discovery=vae are not rescanned.",
            importance=ImportanceLevel.ADVANCED,
            order=25,
            documentation="OPTIONS.md#--enable_nsfw_check",
        )
    )

    registry._add_field(
        ConfigField(
            name="nsfw_check_models",
            arg_name="--nsfw_check_models",
            ui_label="NSFW Classifier Models",
            field_type=FieldType.TEXT,
            tab="model",
            section="vae_config",
            default_value=DEFAULT_NSFW_CHECK_MODELS_CSV,
            placeholder="Falconsai/nsfw_image_detection:threshold=0.5,AdamCodd/vit-base-nsfw-detector:threshold=0.5",
            help_text="CSV list of Hugging Face Transformers image-classification models with optional :threshold= values",
            tooltip="Only standard Transformers image-classification models are supported; trust_remote_code is not enabled.",
            dependencies=nsfw_check_enabled,
            importance=ImportanceLevel.ADVANCED,
            order=26,
            documentation="OPTIONS.md#--nsfw_check_models",
        )
    )

    registry._add_field(
        ConfigField(
            name="nsfw_check_min_votes",
            arg_name="--nsfw_check_min_votes",
            ui_label="NSFW Minimum Votes",
            field_type=FieldType.NUMBER,
            tab="model",
            section="vae_config",
            default_value=2,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="At least one classifier vote is required")
            ],
            help_text="Minimum number of configured classifiers that must flag a frame before the sample is rejected",
            tooltip="Must be between 1 and the number of configured classifier models.",
            dependencies=nsfw_check_enabled,
            importance=ImportanceLevel.ADVANCED,
            order=27,
            documentation="OPTIONS.md#--nsfw_check_min_votes",
        )
    )

    registry._add_field(
        ConfigField(
            name="nsfw_check_backend_types",
            arg_name="--nsfw_check_backend_types",
            ui_label="NSFW Backend Types",
            field_type=FieldType.TEXT,
            tab="model",
            section="vae_config",
            default_value="all",
            placeholder="all",
            help_text="CSV list of data backend type values eligible for NSFW checks",
            tooltip="Use all, or values such as local,huggingface,csv,aws.",
            dependencies=nsfw_check_enabled,
            importance=ImportanceLevel.ADVANCED,
            order=28,
            documentation="OPTIONS.md#--nsfw_check_backend_types",
        )
    )

    registry._add_field(
        ConfigField(
            name="nsfw_check_sample_types",
            arg_name="--nsfw_check_sample_types",
            ui_label="NSFW Sample Types",
            field_type=FieldType.TEXT,
            tab="model",
            section="vae_config",
            default_value="image,conditioning",
            placeholder="image,conditioning",
            help_text="CSV list of dataset_type values eligible for NSFW checks",
            tooltip="Defaults to training image and conditioning samples. Include video to scan video datasets; eval datasets are skipped.",
            dependencies=nsfw_check_enabled,
            importance=ImportanceLevel.ADVANCED,
            order=29,
            documentation="OPTIONS.md#--nsfw_check_sample_types",
        )
    )

    registry._add_field(
        ConfigField(
            name="delete_nsfw_images",
            arg_name="--delete_nsfw_images",
            ui_label="Delete NSFW Images",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="vae_config",
            default_value=False,
            help_text="Delete samples rejected by the NSFW classifier when the data backend supports deletion",
            tooltip="When disabled, rejected samples are removed from metadata for the current run but left in the source dataset.",
            dependencies=nsfw_check_enabled,
            importance=ImportanceLevel.ADVANCED,
            order=30,
            documentation="OPTIONS.md#--delete_nsfw_images",
        )
    )

    registry._add_field(
        ConfigField(
            name="nsfw_check_video_frame_count",
            arg_name="--nsfw_check_video_frame_count",
            ui_label="NSFW Video Frame Count",
            field_type=FieldType.NUMBER,
            tab="model",
            section="vae_config",
            default_value=3,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="At least one video frame must be checked")
            ],
            help_text="Number of frames to sample from each video-like input for NSFW classification",
            tooltip="Used when the classifier receives video or multi-frame conditioning samples.",
            dependencies=nsfw_check_enabled,
            importance=ImportanceLevel.ADVANCED,
            order=31,
            documentation="OPTIONS.md#--nsfw_check_video_frame_count",
        )
    )

    registry._add_field(
        ConfigField(
            name="nsfw_check_video_frame_selection",
            arg_name="--nsfw_check_video_frame_selection",
            ui_label="NSFW Video Frame Selection",
            field_type=FieldType.SELECT,
            tab="model",
            section="vae_config",
            default_value="uniform",
            choices=[
                {"value": "uniform", "label": "Uniform"},
                {"value": "first", "label": "First frames"},
                {"value": "middle", "label": "Middle frames"},
            ],
            help_text="Frame selection strategy for video NSFW checks",
            tooltip="Uniform samples across the clip; first checks the opening frames; middle checks the centered frame window.",
            dependencies=nsfw_check_enabled,
            importance=ImportanceLevel.ADVANCED,
            order=32,
            documentation="OPTIONS.md#--nsfw_check_video_frame_selection",
        )
    )

    registry._add_field(
        ConfigField(
            name="nsfw_check_video_min_flagged_frames",
            arg_name="--nsfw_check_video_min_flagged_frames",
            ui_label="NSFW Video Min Flagged Frames",
            field_type=FieldType.NUMBER,
            tab="model",
            section="vae_config",
            default_value=1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="At least one flagged frame is required")
            ],
            help_text="Number of checked video frames that must be flagged before rejecting the sample",
            tooltip="A frame is flagged when nsfw_check_min_votes classifiers mark it as NSFW.",
            dependencies=nsfw_check_enabled,
            importance=ImportanceLevel.ADVANCED,
            order=33,
            documentation="OPTIONS.md#--nsfw_check_video_min_flagged_frames",
        )
    )

    # VAE Enable Slicing
    registry._add_field(
        ConfigField(
            name="vae_enable_slicing",
            arg_name="--vae_enable_slicing",
            ui_label="Enable VAE Slicing",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="vae_config",
            default_value=False,
            help_text="Enable VAE attention slicing for memory efficiency",
            tooltip="Reduces VAE memory usage by processing attention in slices. May slightly slow encoding.",
            importance=ImportanceLevel.ADVANCED,
            order=21,
        )
    )

    # VAE Enable Tiling
    registry._add_field(
        ConfigField(
            name="vae_enable_tiling",
            arg_name="--vae_enable_tiling",
            ui_label="Enable VAE Tiling",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="vae_config",
            default_value=False,
            help_text="Enable VAE tiling for large images",
            tooltip="Process large images in tiles to reduce memory usage. Useful for very high resolution images.",
            importance=ImportanceLevel.ADVANCED,
            order=22,
        )
    )

    # VAE Enable Patch Convolution
    registry._add_field(
        ConfigField(
            name="vae_enable_patch_conv",
            arg_name="--vae_enable_patch_conv",
            ui_label="Enable VAE Patch Convolution",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="vae_config",
            default_value=False,
            help_text="Process VAE conv layers in temporal patches to lower memory usage.",
            tooltip="Useful for HunyuanVideo VAE on limited VRAM; may slightly reduce throughput.",
            importance=ImportanceLevel.ADVANCED,
            order=23,
        )
    )

    # VAE Temporal Rolling
    registry._add_field(
        ConfigField(
            name="vae_enable_temporal_roll",
            arg_name="--vae_enable_temporal_roll",
            ui_label="Enable VAE Temporal Rolling",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="vae_config",
            default_value=False,
            help_text="Stream VAE convs across time with frame carry to reduce peak VRAM.",
            tooltip="Useful for HunyuanVideo/Kandinsky5 video VAEs on long sequences; trades some speed for memory.",
            importance=ImportanceLevel.ADVANCED,
            order=24,
        )
    )

    # VAE Batch Size
    registry._add_field(
        ConfigField(
            name="vae_batch_size",
            arg_name="--vae_batch_size",
            ui_label="VAE Batch Size",
            field_type=FieldType.NUMBER,
            tab="model",
            section="vae_config",
            default_value=4,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="VAE batch size must be at least 1")],
            help_text="Batch size for VAE encoding during caching",
            tooltip="Higher values speed up VAE caching but use more VRAM. Reduce if getting OOM during cache creation.",
            importance=ImportanceLevel.ADVANCED,
            order=1,
        )
    )

    # Max Upscale Threshold
    registry._add_field(
        ConfigField(
            name="max_upscale_threshold",
            arg_name="--max_upscale_threshold",
            ui_label="Maximum Upscale Threshold",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="training_data",
            subsection="advanced",
            default_value=None,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1, message="Must be between 0 and 1"),
            ],
            help_text="Limit upscaling of small images to prevent quality degradation (opt-in)",
            tooltip="When set, filters out aspect buckets requiring upscaling beyond this threshold. Example: 0.2 allows up to 20% upscaling. Default (None) allows unlimited upscaling.",
            importance=ImportanceLevel.ADVANCED,
            order=8,
        )
    )

    # Caption Dropout
    registry._add_field(
        ConfigField(
            name="caption_dropout_probability",
            arg_name="--caption_dropout_probability",
            ui_label="Caption Dropout Probability",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="caption_processing",
            default_value=None,
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

    # Tokenizer Max Length (Danger mode only)
    registry._add_field(
        ConfigField(
            name="tokenizer_max_length",
            arg_name="--tokenizer_max_length",
            ui_label="Tokenizer Max Length",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="caption_processing",
            default_value=None,
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

    # Audio Max Duration
    registry._add_field(
        ConfigField(
            name="audio_max_duration_seconds",
            arg_name="--audio_max_duration_seconds",
            ui_label="Audio Max Duration (s)",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="dataset_config",
            subsection="audio",
            default_value=None,
            help_text="Maximum duration for audio samples in seconds",
            tooltip="Longer samples will be truncated or skipped depending on configuration.",
            importance=ImportanceLevel.IMPORTANT,
            order=10,
        )
    )

    # Audio Min Duration
    registry._add_field(
        ConfigField(
            name="audio_min_duration_seconds",
            arg_name="--audio_min_duration_seconds",
            ui_label="Audio Min Duration (s)",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="dataset_config",
            subsection="audio",
            default_value=None,
            help_text="Minimum duration for audio samples in seconds",
            tooltip="Samples shorter than this will be skipped.",
            importance=ImportanceLevel.IMPORTANT,
            order=11,
        )
    )

    # Audio Channels
    registry._add_field(
        ConfigField(
            name="audio_channels",
            arg_name="--audio_channels",
            ui_label="Audio Channels",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="dataset_config",
            subsection="audio",
            default_value=1,
            help_text="Number of audio channels (1=mono, 2=stereo)",
            tooltip="Target channel count for training data. Inputs will be mixed/expanded to match.",
            importance=ImportanceLevel.IMPORTANT,
            order=12,
        )
    )

    # Audio Duration Interval
    registry._add_field(
        ConfigField(
            name="audio_duration_interval",
            arg_name="--audio_duration_interval",
            ui_label="Audio Duration Interval (s)",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="dataset_config",
            subsection="audio",
            default_value=3.0,
            help_text="Bucket interval for audio durations",
            tooltip="Audio samples will be bucketed/truncated to multiples of this duration.",
            importance=ImportanceLevel.IMPORTANT,
            order=13,
        )
    )

    # Audio Truncation Mode
    registry._add_field(
        ConfigField(
            name="audio_truncation_mode",
            arg_name="--audio_truncation_mode",
            ui_label="Audio Truncation Mode",
            field_type=FieldType.SELECT,
            tab="basic",
            section="dataset_config",
            subsection="audio",
            default_value="beginning",
            choices=[
                {"value": "beginning", "label": "Beginning (Keep Start)"},
                {"value": "end", "label": "End (Keep End)"},
                {"value": "random", "label": "Random"},
            ],
            help_text="How to truncate audio that exceeds bucket length",
            tooltip="Which part of the audio file to keep when truncating to fit a bucket.",
            importance=ImportanceLevel.IMPORTANT,
            order=14,
        )
    )
