"""Dataset blueprint metadata exposed to the Web UI."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel


class DatasetFieldOption(BaseModel):
    value: str
    label: str
    hint: Optional[str] = None


class DatasetField(BaseModel):
    id: str
    label: str
    type: Literal["text", "number", "select", "toggle", "textarea"]
    description: Optional[str] = None
    required: bool = False
    placeholder: Optional[str] = None
    defaultValue: Optional[Any] = None
    options: Optional[List[DatasetFieldOption]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    advanced: bool = False


class BlueprintMetadata(BaseModel):
    tags: Optional[List[str]] = None
    docsUrl: Optional[str] = None


class BackendBlueprint(BaseModel):
    id: str
    backendType: str
    datasetTypes: List[str]
    label: str
    description: str
    defaults: Dict[str, Any]
    fields: List[DatasetField]
    metadata: Optional[BlueprintMetadata] = None


_BLUEPRINTS: List[BackendBlueprint] = [
    BackendBlueprint.model_validate(
        {
            "id": "local-image",
            "backendType": "local",
            "datasetTypes": ["image", "conditioning", "eval"],
            "label": "local media backend",
            "description": "use filesystem folders or network mounts for primary training data",
            "defaults": {
                "type": "local",
                "metadata_backend": "discovery",
                "caption_strategy": "textfile",
                "resolution": 1024,
                "resolution_type": "pixel_area",
                "probability": 1,
                "repeats": 0,
                "cache_dir_vae": "cache/vae",
            },
            "fields": [
                {
                    "id": "id",
                    "label": "dataset id",
                    "description": "unique identifier for this dataset",
                    "type": "text",
                    "required": True,
                    "placeholder": "my-dataset",
                },
                {
                    "id": "type",
                    "label": "backend type",
                    "description": "storage backend type",
                    "type": "select",
                    "defaultValue": "local",
                    "options": [
                        {"value": "local", "label": "local filesystem"},
                        {"value": "aws", "label": "AWS S3"},
                        {"value": "csv", "label": "CSV URL list"},
                        {"value": "huggingface", "label": "Hugging Face"},
                    ],
                },
                {
                    "id": "dataset_type",
                    "label": "dataset type",
                    "description": "type of data in this dataset",
                    "type": "select",
                    "defaultValue": "image",
                    "options": [
                        {"value": "image", "label": "image"},
                        {"value": "video", "label": "video"},
                        {"value": "conditioning", "label": "conditioning"},
                        {"value": "text_embeds", "label": "text embeds"},
                        {"value": "image_embeds", "label": "image embeds"},
                        {"value": "audio", "label": "audio"},
                        {"value": "eval", "label": "eval"},
                    ],
                },
                {
                    "id": "instance_data_dir",
                    "label": "instance data path",
                    "description": "absolute path containing images or videos",
                    "type": "text",
                    "required": True,
                    "placeholder": "/data/training/images",
                },
                {
                    "id": "metadata_backend",
                    "label": "metadata backend",
                    "description": "select how captions and annotations are resolved",
                    "type": "select",
                    "defaultValue": "discovery",
                    "options": [
                        {"value": "discovery", "label": "discovery json"},
                        {"value": "json", "label": "static json"},
                        {"value": "parquet", "label": "parquet index"},
                        {"value": "csv", "label": "CSV index"},
                        {"value": "huggingface", "label": "Hugging Face dataset"},
                        {"value": "none", "label": "none"},
                    ],
                },
                {
                    "id": "caption_strategy",
                    "label": "caption strategy",
                    "description": "controls caption lookup priority",
                    "type": "select",
                    "defaultValue": "textfile",
                    "options": [
                        {"value": "textfile", "label": "text file"},
                        {"value": "csv", "label": "CSV"},
                        {"value": "parquet", "label": "parquet"},
                        {"value": "jsonl", "label": "JSON lines"},
                        {"value": "instanceprompt", "label": "instance prompt only"},
                    ],
                },
                {
                    "id": "resolution",
                    "label": "base resolution",
                    "description": "pixel edge or megapixel target depending on resolution type",
                    "type": "number",
                    "defaultValue": 1024,
                    "min": 256,
                    "step": 64,
                },
                {
                    "id": "resolution_type",
                    "label": "resolution type",
                    "type": "select",
                    "defaultValue": "pixel",
                    "options": [
                        {"value": "pixel", "label": "pixel edge"},
                        {"value": "pixel_area", "label": "pixel area"},
                        {"value": "area", "label": "megapixels"},
                    ],
                },
                {
                    "id": "probability",
                    "label": "sampling probability",
                    "description": "relative weight applied when mixing datasets",
                    "type": "number",
                    "defaultValue": 1,
                    "min": 0,
                    "step": 0.05,
                },
                {
                    "id": "repeats",
                    "label": "repeats",
                    "description": "optional repeat count when building the epoch plan",
                    "type": "number",
                    "defaultValue": 0,
                    "min": 0,
                    "step": 1,
                    "advanced": True,
                },
                {
                    "id": "cache_dir_vae",
                    "label": "vae cache directory",
                    "description": "directory used for cached vae latents",
                    "type": "text",
                    "placeholder": "cache/vae",
                    "advanced": True,
                },
                {
                    "id": "minimum_image_size",
                    "label": "min image size",
                    "description": "discard images smaller than this (match resolution type)",
                    "type": "number",
                    "advanced": True,
                    "min": 0,
                    "step": 1,
                },
                {
                    "id": "maximum_image_size",
                    "label": "max image size",
                    "description": "optional clamp before downsampling (match resolution type)",
                    "type": "number",
                    "advanced": True,
                    "min": 0,
                    "step": 1,
                },
                {
                    "id": "target_downsample_size",
                    "label": "target downsample",
                    "description": "secondary clamp for very large assets",
                    "type": "number",
                    "advanced": True,
                    "min": 0,
                    "step": 1,
                },
                {
                    "id": "crop",
                    "label": "enable cropping",
                    "description": "apply crop before resizing",
                    "type": "toggle",
                    "defaultValue": False,
                    "advanced": True,
                },
                {
                    "id": "crop_style",
                    "label": "crop style",
                    "type": "select",
                    "defaultValue": "random",
                    "options": [
                        {"value": "random", "label": "random"},
                        {"value": "center", "label": "center"},
                        {"value": "corner", "label": "corner"},
                        {"value": "face", "label": "face detection"},
                    ],
                    "advanced": True,
                },
                {
                    "id": "crop_aspect",
                    "label": "crop aspect",
                    "description": "aspect ratio behavior when cropping",
                    "type": "select",
                    "defaultValue": "square",
                    "options": [
                        {"value": "square", "label": "square"},
                        {"value": "preserve", "label": "preserve"},
                        {"value": "closest", "label": "closest bucket"},
                        {"value": "random", "label": "weighted buckets"},
                    ],
                    "advanced": True,
                },
                {
                    "id": "max_upscale_threshold",
                    "label": "max upscale",
                    "description": "maximum upscale factor (e.g. 0.5 = max 50% upscale)",
                    "type": "number",
                    "min": 0,
                    "max": 2,
                    "step": 0.1,
                    "advanced": True,
                },
                {
                    "id": "minimum_aspect_ratio",
                    "label": "min aspect ratio",
                    "description": "exclude images with aspect ratio below this value",
                    "type": "number",
                    "min": 0,
                    "step": 0.01,
                    "advanced": True,
                },
                {
                    "id": "maximum_aspect_ratio",
                    "label": "max aspect ratio",
                    "description": "exclude images with aspect ratio above this value",
                    "type": "number",
                    "min": 0,
                    "step": 0.01,
                    "advanced": True,
                },
                {
                    "id": "is_regularisation_data",
                    "label": "regularisation data",
                    "description": "use for prior preservation to prevent model drift",
                    "type": "toggle",
                    "defaultValue": False,
                    "advanced": True,
                },
                {
                    "id": "slider_strength",
                    "label": "slider strength",
                    "description": "slider LoRA concept strength (positive = add, negative = remove)",
                    "type": "number",
                    "step": 0.1,
                    "advanced": True,
                },
                {
                    "id": "vae_cache_clear_each_epoch",
                    "label": "clear VAE cache each epoch",
                    "description": "regenerate latents per epoch for random crop/aspect training",
                    "type": "toggle",
                    "defaultValue": False,
                    "advanced": True,
                },
                {
                    "id": "preserve_data_backend_cache",
                    "label": "preserve backend cache",
                    "description": "keep data backend cache between runs",
                    "type": "toggle",
                    "defaultValue": False,
                    "advanced": True,
                },
                {
                    "id": "instance_prompt",
                    "label": "instance prompt",
                    "description": "trigger word or prompt prepended to captions",
                    "type": "text",
                    "advanced": True,
                },
                {
                    "id": "prepend_instance_prompt",
                    "label": "prepend instance prompt",
                    "description": "add trigger word before each caption",
                    "type": "toggle",
                    "defaultValue": False,
                    "advanced": True,
                },
                {
                    "id": "text_embeds",
                    "label": "text embeds dataset",
                    "description": "link to text embeddings dataset",
                    "type": "text",
                    "advanced": True,
                },
                {
                    "id": "image_embeds",
                    "label": "image embeds dataset",
                    "description": "link to image embeddings dataset",
                    "type": "text",
                    "advanced": True,
                },
                {
                    "id": "conditioning_image_embeds",
                    "label": "conditioning image embeds",
                    "description": "link to conditioning image embeddings dataset (for I2V models)",
                    "type": "text",
                    "advanced": True,
                },
                {
                    "id": "cache_dir_conditioning_image_embeds",
                    "label": "conditioning embed cache dir",
                    "description": "cache directory for conditioning image embeddings",
                    "type": "text",
                    "advanced": True,
                },
                {
                    "id": "video_bucket_strategy",
                    "label": "video bucket strategy",
                    "description": "how to group videos into buckets (video datasets only)",
                    "type": "select",
                    "defaultValue": "aspect_ratio",
                    "options": [
                        {"value": "aspect_ratio", "label": "aspect ratio only"},
                        {"value": "resolution_frames", "label": "resolution@frames (WxH@F)"},
                    ],
                    "advanced": True,
                },
                {
                    "id": "video_frame_interval",
                    "label": "video frame interval",
                    "description": "frame count rounding interval for resolution_frames bucketing",
                    "type": "number",
                    "min": 1,
                    "step": 1,
                    "advanced": True,
                },
                {
                    "id": "video_num_frames",
                    "label": "video num frames",
                    "description": "target frame count for training (sets fixed bucket if using resolution_frames)",
                    "type": "number",
                    "min": 1,
                    "step": 1,
                    "advanced": True,
                },
                {
                    "id": "video_min_frames",
                    "label": "video min frames",
                    "description": "minimum frame count for videos (shorter videos are discarded)",
                    "type": "number",
                    "min": 1,
                    "step": 1,
                    "advanced": True,
                },
                {
                    "id": "video_max_frames",
                    "label": "video max frames",
                    "description": "maximum frame count for videos (longer videos are skipped during scan)",
                    "type": "number",
                    "min": 1,
                    "step": 1,
                    "advanced": True,
                },
                {
                    "id": "video_is_i2v",
                    "label": "video I2V mode",
                    "description": "enable image-to-video training mode",
                    "type": "toggle",
                    "defaultValue": False,
                    "advanced": True,
                },
            ],
            "metadata": {
                "tags": ["image", "local"],
                "docsUrl": "https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md",
            },
        }
    ),
    BackendBlueprint.model_validate(
        {
            "id": "local-audio",
            "backendType": "local",
            "datasetTypes": ["audio"],
            "label": "local audio backend",
            "description": "use filesystem folders for audio training data",
            "defaults": {
                "type": "local",
                "metadata_backend": "discovery",
                "caption_strategy": "textfile",
                "repeats": 0,
                "cache_dir_vae": "cache/audio/vae",
            },
            "fields": [
                {
                    "id": "id",
                    "label": "dataset id",
                    "description": "unique identifier for this dataset",
                    "type": "text",
                    "required": True,
                    "placeholder": "my-audio-dataset",
                },
                {
                    "id": "type",
                    "label": "backend type",
                    "description": "storage backend type",
                    "type": "select",
                    "defaultValue": "local",
                    "options": [
                        {"value": "local", "label": "local filesystem"},
                    ],
                },
                {
                    "id": "dataset_type",
                    "label": "dataset type",
                    "description": "type of data in this dataset",
                    "type": "select",
                    "defaultValue": "audio",
                    "options": [
                        {"value": "audio", "label": "audio"},
                    ],
                },
                {
                    "id": "instance_data_dir",
                    "label": "instance data path",
                    "description": "absolute path containing audio files",
                    "type": "text",
                    "required": True,
                    "placeholder": "/data/training/audio",
                },
                {
                    "id": "metadata_backend",
                    "label": "metadata backend",
                    "description": "select how captions and annotations are resolved",
                    "type": "select",
                    "defaultValue": "discovery",
                    "options": [
                        {"value": "discovery", "label": "discovery json"},
                        {"value": "json", "label": "static json"},
                        {"value": "none", "label": "none"},
                    ],
                },
                {
                    "id": "caption_strategy",
                    "label": "caption strategy",
                    "description": "controls caption lookup priority",
                    "type": "select",
                    "defaultValue": "textfile",
                    "options": [
                        {"value": "textfile", "label": "text file"},
                        {"value": "filename", "label": "filename"},
                        {"value": "csv", "label": "csv column"},
                        {"value": "huggingface", "label": "huggingface field"},
                        {"value": "parquet", "label": "parquet"},
                    ],
                },
                {
                    "id": "repeats",
                    "label": "repeats",
                    "description": "optional repeat count when building the epoch plan",
                    "type": "number",
                    "defaultValue": 0,
                    "min": 0,
                    "step": 1,
                    "advanced": True,
                },
                {
                    "id": "audio_min_duration_seconds",
                    "label": "min duration (s)",
                    "description": "minimum audio duration in seconds",
                    "type": "number",
                    "min": 0.1,
                    "step": 0.1,
                },
                {
                    "id": "audio_max_duration_seconds",
                    "label": "max duration (s)",
                    "description": "maximum audio duration in seconds",
                    "type": "number",
                    "min": 1,
                    "step": 0.5,
                },
                {
                    "id": "audio_channels",
                    "label": "channels",
                    "description": "number of audio channels (1 for mono, 2 for stereo)",
                    "type": "number",
                    "defaultValue": 1,
                    "min": 1,
                    "step": 1,
                    "advanced": True,
                },
                {
                    "id": "audio_duration_interval",
                    "label": "duration interval",
                    "description": "audio will be truncated to the nearest multiple of this value (seconds)",
                    "type": "number",
                    "defaultValue": 3.0,
                    "min": 0.1,
                    "step": 0.1,
                    "advanced": True,
                },
                {
                    "id": "audio_truncation_mode",
                    "label": "truncation mode",
                    "type": "select",
                    "defaultValue": "beginning",
                    "options": [
                        {"value": "beginning", "label": "beginning"},
                        {"value": "end", "label": "end"},
                        {"value": "random", "label": "random"},
                    ],
                    "advanced": True,
                },
                {
                    "id": "cache_dir_vae",
                    "label": "vae cache directory",
                    "description": "directory used for cached audio latents",
                    "type": "text",
                    "placeholder": "cache/audio/vae",
                    "advanced": True,
                },
            ],
            "metadata": {
                "tags": ["audio", "local"],
                "docsUrl": "https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md#audio-datasets",
            },
        }
    ),
    BackendBlueprint.model_validate(
        {
            "id": "local-text-embeds",
            "backendType": "local",
            "datasetTypes": ["text_embeds"],
            "label": "local text embeds",
            "description": "use cached text embeddings stored on disk",
            "defaults": {
                "type": "local",
                "metadata_backend": "json",
                "cache_dir": "cache/text_embeds",
                "default": False,
            },
            "fields": [
                {
                    "id": "id",
                    "label": "dataset id",
                    "description": "unique identifier for this dataset",
                    "type": "text",
                    "required": True,
                    "placeholder": "my-text-embeds",
                },
                {
                    "id": "type",
                    "label": "backend type",
                    "description": "storage backend type",
                    "type": "select",
                    "defaultValue": "local",
                    "options": [
                        {"value": "local", "label": "local filesystem"},
                    ],
                },
                {
                    "id": "dataset_type",
                    "label": "dataset type",
                    "description": "type of data in this dataset",
                    "type": "select",
                    "defaultValue": "text_embeds",
                    "options": [
                        {"value": "text_embeds", "label": "text embeds"},
                    ],
                },
                {
                    "id": "cache_dir",
                    "label": "cache directory",
                    "description": "path containing cached text embeddings",
                    "type": "text",
                    "required": True,
                    "placeholder": "cache/text_embeds",
                },
                {
                    "id": "default",
                    "label": "mark as default",
                    "description": "toggle if this dataset should provide captions when multiple text embeds exist",
                    "type": "toggle",
                    "defaultValue": False,
                },
                {
                    "id": "caption_filter_list",
                    "label": "caption filter list",
                    "description": "optional filter list file applied during sampling",
                    "type": "text",
                    "placeholder": "config/filter_list.json",
                    "advanced": True,
                },
                {
                    "id": "skip_file_discovery",
                    "label": "skip file discovery",
                    "description": "assume cache directory already contains valid manifests",
                    "type": "toggle",
                    "defaultValue": False,
                    "advanced": True,
                },
                {
                    "id": "write_batch_size",
                    "label": "write batch size",
                    "description": "batch size when writing embeddings to disk",
                    "type": "number",
                    "defaultValue": 128,
                    "min": 1,
                    "step": 1,
                    "advanced": True,
                },
                {
                    "id": "preserve_data_backend_cache",
                    "label": "preserve backend cache",
                    "description": "keep data backend cache between runs",
                    "type": "toggle",
                    "defaultValue": False,
                    "advanced": True,
                },
            ],
            "metadata": {
                "tags": ["text_embeds", "local"],
                "docsUrl": "https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md#text-embeds",
            },
        }
    ),
    BackendBlueprint.model_validate(
        {
            "id": "huggingface-image",
            "backendType": "huggingface",
            "datasetTypes": ["image"],
            "label": "hugging face dataset",
            "description": "stream images from the hugging face hub",
            "defaults": {
                "type": "huggingface",
                "metadata_backend": "huggingface",
                "caption_strategy": "huggingface",
                "split": "train",
            },
            "fields": [
                {
                    "id": "id",
                    "label": "dataset id",
                    "description": "unique identifier for this dataset",
                    "type": "text",
                    "required": True,
                    "placeholder": "my-hf-dataset",
                },
                {
                    "id": "type",
                    "label": "backend type",
                    "description": "storage backend type",
                    "type": "select",
                    "defaultValue": "huggingface",
                    "options": [
                        {"value": "huggingface", "label": "Hugging Face"},
                    ],
                },
                {
                    "id": "dataset_type",
                    "label": "dataset type",
                    "description": "type of data in this dataset",
                    "type": "select",
                    "defaultValue": "image",
                    "options": [
                        {"value": "image", "label": "image"},
                    ],
                },
                {
                    "id": "dataset_name",
                    "label": "dataset repository",
                    "description": "hugging face repo id, eg. stabilityai/stable-diffusion",
                    "type": "text",
                    "required": True,
                    "placeholder": "owner/dataset",
                },
                {
                    "id": "split",
                    "label": "split",
                    "type": "text",
                    "defaultValue": "train",
                },
                {
                    "id": "image_column",
                    "label": "image column",
                    "type": "text",
                    "defaultValue": "image",
                },
                {
                    "id": "caption_column",
                    "label": "caption column",
                    "description": "column containing captions",
                    "type": "text",
                    "defaultValue": "text",
                },
                {
                    "id": "dataset_config",
                    "label": "dataset config",
                    "description": "dataset configuration/subset name",
                    "type": "text",
                    "placeholder": "default",
                },
                {
                    "id": "streaming",
                    "label": "enable streaming",
                    "description": "toggle streaming to iterate without full download",
                    "type": "toggle",
                    "defaultValue": True,
                },
                {
                    "id": "auth_token",
                    "label": "auth token",
                    "description": "hugging face authentication token for private datasets",
                    "type": "text",
                    "advanced": True,
                },
                {
                    "id": "cache_dir",
                    "label": "dataset cache",
                    "description": "optional hugging face cache directory override",
                    "type": "text",
                    "placeholder": "cache/huggingface",
                    "advanced": True,
                },
                {
                    "id": "revision",
                    "label": "revision",
                    "description": "pin to a specific git revision or commit",
                    "type": "text",
                    "advanced": True,
                },
            ],
            "metadata": {
                "tags": ["image", "huggingface"],
                "docsUrl": "https://huggingface.co/docs",
            },
        }
    ),
    BackendBlueprint.model_validate(
        {
            "id": "csv-image",
            "backendType": "csv",
            "datasetTypes": ["image"],
            "label": "csv manifest",
            "description": "drive image datasets via url/caption csv manifests",
            "defaults": {
                "type": "csv",
                "metadata_backend": "csv",
                "caption_strategy": "csv",
            },
            "fields": [
                {
                    "id": "id",
                    "label": "dataset id",
                    "description": "unique identifier for this dataset",
                    "type": "text",
                    "required": True,
                    "placeholder": "my-csv-dataset",
                },
                {
                    "id": "type",
                    "label": "backend type",
                    "description": "storage backend type",
                    "type": "select",
                    "defaultValue": "csv",
                    "options": [
                        {"value": "csv", "label": "CSV URL list"},
                    ],
                },
                {
                    "id": "dataset_type",
                    "label": "dataset type",
                    "description": "type of data in this dataset",
                    "type": "select",
                    "defaultValue": "image",
                    "options": [
                        {"value": "image", "label": "image"},
                    ],
                },
                {
                    "id": "csv_file",
                    "label": "csv file path",
                    "description": "path to the csv manifest",
                    "type": "text",
                    "required": True,
                    "placeholder": "config/dataset.csv",
                },
                {
                    "id": "csv_caption_column",
                    "label": "caption column",
                    "type": "text",
                    "defaultValue": "caption",
                },
                {
                    "id": "csv_url_column",
                    "label": "asset column",
                    "type": "text",
                    "defaultValue": "url",
                },
                {
                    "id": "csv_cache_dir",
                    "label": "download cache",
                    "description": "directory used when caching remote assets",
                    "type": "text",
                    "placeholder": "cache/csv",
                    "advanced": True,
                },
            ],
            "metadata": {
                "tags": ["image", "csv"],
                "docsUrl": "https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md#csv-datasets",
            },
        }
    ),
    BackendBlueprint.model_validate(
        {
            "id": "aws-image",
            "backendType": "aws",
            "datasetTypes": ["image"],
            "label": "aws s3 bucket",
            "description": "pull media from an s3 bucket using botocore",
            "defaults": {
                "type": "aws",
                "metadata_backend": "discovery",
                "caption_strategy": "textfile",
            },
            "fields": [
                {
                    "id": "id",
                    "label": "dataset id",
                    "description": "unique identifier for this dataset",
                    "type": "text",
                    "required": True,
                    "placeholder": "my-s3-dataset",
                },
                {
                    "id": "type",
                    "label": "backend type",
                    "description": "storage backend type",
                    "type": "select",
                    "defaultValue": "aws",
                    "options": [
                        {"value": "aws", "label": "AWS S3"},
                    ],
                },
                {
                    "id": "dataset_type",
                    "label": "dataset type",
                    "description": "type of data in this dataset",
                    "type": "select",
                    "defaultValue": "image",
                    "options": [
                        {"value": "image", "label": "image"},
                    ],
                },
                {
                    "id": "aws_bucket_name",
                    "label": "bucket name",
                    "type": "text",
                    "required": True,
                    "placeholder": "simpletuner-datasets",
                },
                {
                    "id": "aws_data_prefix",
                    "label": "bucket prefix",
                    "type": "text",
                    "placeholder": "train/images",
                },
                {
                    "id": "aws_region_name",
                    "label": "region",
                    "type": "text",
                    "placeholder": "us-east-1",
                },
                {
                    "id": "aws_endpoint_url",
                    "label": "endpoint url",
                    "type": "text",
                    "placeholder": "https://s3.us-east-1.amazonaws.com",
                    "advanced": True,
                },
                {
                    "id": "aws_access_key_id",
                    "label": "access key id",
                    "type": "text",
                    "advanced": True,
                },
                {
                    "id": "aws_secret_access_key",
                    "label": "secret access key",
                    "type": "text",
                    "advanced": True,
                },
                {
                    "id": "aws_max_pool_connections",
                    "label": "max pool connections",
                    "type": "number",
                    "advanced": True,
                    "min": 1,
                    "step": 1,
                },
            ],
            "metadata": {
                "tags": ["image", "aws"],
                "docsUrl": "https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md#aws-datasets",
            },
        }
    ),
]


def get_dataset_blueprints() -> List[BackendBlueprint]:
    """Return a deep copy of available dataset blueprints."""
    return [blueprint.model_copy(deep=True) for blueprint in _BLUEPRINTS]


def get_blueprint_lookup() -> Dict[Tuple[str, str], BackendBlueprint]:
    """Return a mapping of (backendType, datasetType) -> blueprint."""
    lookup: Dict[Tuple[str, str], BackendBlueprint] = {}
    for blueprint in _BLUEPRINTS:
        for dataset_type in blueprint.datasetTypes:
            lookup[(blueprint.backendType, dataset_type)] = blueprint.model_copy(deep=True)
    return lookup


def find_blueprint(backend_type: str, dataset_type: str) -> Optional[BackendBlueprint]:
    """Lookup a blueprint for the given backend and dataset type."""
    for blueprint in _BLUEPRINTS:
        if blueprint.backendType != backend_type:
            continue
        if dataset_type in blueprint.datasetTypes:
            return blueprint.model_copy(deep=True)
    return None
