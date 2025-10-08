import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

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
            help_text="Instance prompt for training",
            tooltip="Used when caption_strategy is 'instance_prompt'. Defines the concept being trained.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
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
            ui_label="Ignore Missing Files",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="data_config",
            default_value=False,
            help_text="Continue training even if some files are missing",
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
            tab="basic",
            section="data_config",
            default_value="recreate",
            choices=[
                {"value": "recreate", "label": "Recreate"},
                {"value": "sync", "label": "Sync"},
            ],
            help_text="How to scan VAE cache for missing files",
            tooltip="'recreate': rebuild inconsistent cache entries. 'sync': adjust bucket metadata to match existing latents.",
            importance=ImportanceLevel.ADVANCED,
            order=8,
        )
    )

    # VAE Enable Slicing
    registry._add_field(
        ConfigField(
            name="vae_enable_slicing",
            arg_name="--vae_enable_slicing",
            ui_label="Enable VAE Slicing",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="data_config",
            default_value=False,
            help_text="Enable VAE attention slicing for memory efficiency",
            tooltip="Reduces VAE memory usage by processing attention in slices. May slightly slow encoding.",
            importance=ImportanceLevel.ADVANCED,
            order=9,
        )
    )

    # VAE Enable Tiling
    registry._add_field(
        ConfigField(
            name="vae_enable_tiling",
            arg_name="--vae_enable_tiling",
            ui_label="Enable VAE Tiling",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="data_config",
            default_value=False,
            help_text="Enable VAE tiling for large images",
            tooltip="Process large images in tiles to reduce memory usage. Useful for very high resolution images.",
            importance=ImportanceLevel.ADVANCED,
            order=10,
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
