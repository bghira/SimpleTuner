"""Shared validation functions for backend configurations."""
from typing import Any, Dict, List, Union, Optional
import os

from simpletuner.helpers.training.state_tracker import StateTracker


def validate_crop_aspect(
    crop_aspect: str,
    crop_aspect_buckets: Optional[List[Union[float, Dict[str, Any]]]] = None,
    backend_id: str = "",
) -> None:
    valid_crop_aspects = ["square", "preserve", "random", "closest"]
    if crop_aspect not in valid_crop_aspects:
        raise ValueError(f"(id={backend_id}) crop_aspect must be one of {valid_crop_aspects}.")


def validate_crop_aspect_buckets(
    crop_aspect: str,
    crop_aspect_buckets: Optional[List[Union[float, Dict[str, Any]]]],
    backend_id: str
) -> None:
    if crop_aspect in ["random", "closest"]:
        if not crop_aspect_buckets or not isinstance(crop_aspect_buckets, list):
            raise ValueError(
                f"(id={backend_id}) crop_aspect_buckets must be provided when crop_aspect is set to 'random'."
                " This should be a list of float values or a list of dictionaries following the format: {{'aspect_bucket': float, 'weight': float}}."
                " The weight represents how likely this bucket is to be chosen, and all weights should add up to 1.0 collectively."
            )

        for bucket in crop_aspect_buckets:
            if type(bucket) not in [float, int, dict]:
                raise ValueError(
                    f"(id={backend_id}) crop_aspect_buckets must be a list of float values or a list of dictionaries following the format: {{'aspect_bucket': float, 'weight': float}}."
                    " The weight represents how likely this bucket is to be chosen, and all weights should add up to 1.0 collectively."
                )


def validate_crop_style(crop_style: str, backend_id: str = "") -> None:
    valid_crop_styles = ["random", "corner", "center", "centre", "face"]
    if crop_style not in valid_crop_styles:
        raise ValueError(f"(id={backend_id}) crop_style must be one of {valid_crop_styles}.")


def validate_dataset_type(dataset_type: str, valid_types: List[str], backend_id: str) -> None:
    if dataset_type not in valid_types:
        raise ValueError(f"(id={backend_id}) dataset_type must be one of {valid_types}.")


def validate_resolution_type(resolution_type: str, backend_id: str) -> None:
    valid_resolution_types = ["pixel", "area", "pixel_area"]
    if resolution_type not in valid_resolution_types:
        raise ValueError(f"(id={backend_id}) resolution_type must be one of {valid_resolution_types}.")


def validate_image_size_constraints(
    maximum_image_size: Optional[Union[int, float]],
    target_downsample_size: Optional[Union[int, float]],
    resolution_type: str,
    model_type: str,
    backend_id: str
) -> None:
    if maximum_image_size and not target_downsample_size:
        raise ValueError(
            "When a data backend is configured to use `maximum_image_size`, you must also provide a value for `target_downsample_size`."
        )

    # Validate maximum_image_size
    if maximum_image_size:
        if (
            resolution_type == "area"
            and maximum_image_size > 10
            and not os.environ.get("SIMPLETUNER_MAXIMUM_IMAGE_SIZE_OVERRIDE", False)
        ):
            raise ValueError(
                f"(id={backend_id}) maximum_image_size must be less than 10 megapixels when resolution_type is 'area'."
            )
        elif (
            resolution_type == "pixel"
            and maximum_image_size < 512
            and "deepfloyd" not in model_type
        ):
            raise ValueError(
                f"(id={backend_id}) maximum_image_size must be at least 512 pixels when resolution_type is 'pixel'."
            )

    # Validate target_downsample_size
    if target_downsample_size:
        if (
            resolution_type == "area"
            and target_downsample_size > 10
            and not os.environ.get("SIMPLETUNER_MAXIMUM_IMAGE_SIZE_OVERRIDE", False)
        ):
            raise ValueError(
                f"(id={backend_id}) target_downsample_size must be less than 10 megapixels when resolution_type is 'area'."
            )
        elif (
            resolution_type == "pixel"
            and target_downsample_size < 512
            and "deepfloyd" not in model_type
        ):
            raise ValueError(
                f"(id={backend_id}) target_downsample_size must be at least 512 pixels when resolution_type is 'pixel'."
            )


def validate_resolution_constraints(*args, **kwargs) -> None:
    # backward-compatible alias

    validate_image_size_constraints(*args, **kwargs)


def validate_caption_strategy_compatibility(
    caption_strategy: str,
    metadata_backend: str,
    backend_type: str,
    backend_id: str
) -> None:
    # Check parquet + json/discovery incompatibility
    if caption_strategy == "parquet" and metadata_backend in ["json", "discovery"]:
        raise ValueError(
            f"(id={backend_id}) Cannot use caption_strategy=parquet with metadata_backend={metadata_backend}. Instead, it is recommended to use the textfile strategy and extract your captions into txt files."
        )

    # Check huggingface caption strategy compatibility
    if caption_strategy == "huggingface" and backend_type != "huggingface":
        raise ValueError(
            f"(id={backend_id}) caption_strategy='huggingface' can only be used with type='huggingface' backends"
        )


def validate_huggingface_backend_settings(
    backend_type: str,
    metadata_backend: Optional[str],
    caption_strategy: Optional[str],
    backend_id: str
) -> Dict[str, str]:
    if backend_type != "huggingface":
        return {"metadata_backend": metadata_backend, "caption_strategy": caption_strategy}

    # Validate metadata_backend for huggingface
    if metadata_backend is not None and metadata_backend != "huggingface":
        raise ValueError(
            f"(id={backend_id}) When using a huggingface data backend, metadata_backend must be set to 'huggingface'."
        )

    # Validate caption_strategy for huggingface
    if caption_strategy is not None and caption_strategy not in ["huggingface", "instanceprompt"]:
        raise ValueError(
            f"(id={backend_id}) When using a huggingface data backend, caption_strategy must be set to 'huggingface'."
        )

    # Set defaults
    return {
        "metadata_backend": metadata_backend or "huggingface",
        "caption_strategy": caption_strategy or "huggingface"
    }


def validate_video_frame_settings(
    min_frames: int,
    num_frames: int,
    backend_id: str
) -> None:
    # Check if both are integers
    if not all([isinstance(min_frames, int), isinstance(num_frames, int)]):
        raise ValueError(
            f"video->min_frames and video->num_frames must be integers. Received min_frames={min_frames} and num_frames={num_frames}."
        )

    # Check if both are positive
    if min_frames < 1 or num_frames < 1:
        raise ValueError(
            f"video->min_frames and video->num_frames must be greater than 0. Received min_frames={min_frames} and num_frames={num_frames}."
        )

    # Check relationship between min and target frames
    if min_frames < num_frames:
        raise ValueError(
            f"video->min_frames must be greater than or equal to video->num_frames. Received min_frames={min_frames} and num_frames={num_frames}."
        )


def validate_backend_id(backend_id: str) -> None:
    if not backend_id or backend_id == "":
        raise ValueError("Backend configuration must have a non-empty 'id' field.")


def check_for_caption_filter_list_misuse(
    dataset_type: str,
    has_caption_filter_list: bool,
    backend_id: str
) -> None:
    if has_caption_filter_list and dataset_type != "text_embeds":
        raise ValueError(
            f"caption_filter_list is only a valid setting for text datasets. It is currently set for the {dataset_type} dataset {backend_id}."
        )
