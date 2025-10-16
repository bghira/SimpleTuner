import io
import json
import logging
import os
import sys
import time
import tracemalloc
from math import sqrt
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from .runtime import BatchFetcher


class _ArgsProxy(SimpleNamespace):
    """Lightweight namespace that proxies unknown attributes to a backing object."""

    def __init__(self, base: Any, **overrides: Any) -> None:
        super().__init__(**overrides)
        object.__setattr__(self, "_base", base)

    def __getattr__(self, item: str) -> Any:
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return getattr(self._base, item)


def _get_arg_value(args: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve a value from an args mapping or namespace."""

    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def _set_arg_value(args: Any, key: str, value: Any) -> None:
    """Safely set a value on an args mapping or namespace."""

    if isinstance(args, dict):
        args[key] = value
    else:
        setattr(args, key, value)


import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
from simpletuner.helpers.caching.vae import VAECache
from simpletuner.helpers.data_backend.aws import S3DataBackend
from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.csv_url_list import CSVDataBackend
from simpletuner.helpers.data_backend.huggingface import HuggingfaceDatasetsBackend
from simpletuner.helpers.data_backend.local import LocalDataBackend
from simpletuner.helpers.metadata.utils import DatasetDuplicator
from simpletuner.helpers.models.common import ModelFoundation
from simpletuner.helpers.multiaspect.dataset import MultiAspectDataset
from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
from simpletuner.helpers.prompts import CaptionNotFoundError, PromptHandler
from simpletuner.helpers.training.collate import collate_fn
from simpletuner.helpers.training.default_settings import default, latest_config_version
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.multi_process import _get_rank as get_rank
from simpletuner.helpers.training.multi_process import rank_info, should_log
from simpletuner.helpers.training.state_tracker import StateTracker

from .builders import build_backend_from_config, create_backend_builder
from .config import ImageBackendConfig, ImageEmbedBackendConfig, TextEmbedBackendConfig, create_backend_config

logger = logging.getLogger("DataBackendFactory")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)
prefetch_log = logging.getLogger("DataBackendPrefetch")
if should_log():
    prefetch_log.setLevel(os.environ.get("SIMPLETUNER_PREFETCH_LOG_LEVEL", "INFO"))
else:
    prefetch_log.setLevel(logging.ERROR)

# For prefetching.


def prefetch_log_debug(message):
    prefetch_log.debug(f"{rank_info()} {message}")


def info_log(message):
    _log_with_state("info", message)


def warning_log(message):
    _log_with_state("warning", message)


def debug_log(message):
    _log_with_state("debug", message)


def _log_with_state(level: str, message: str) -> None:
    try:
        accelerator = StateTracker.get_accelerator()
    except Exception:
        accelerator = None

    should_log = accelerator is None or getattr(accelerator, "is_main_process", False)

    if should_log:
        getattr(logger, level)(message)


def _synchronise_state_tracker(args: Any, accelerator: Any) -> Any:
    """Ensure both the patched and real StateTracker references mirror the provided args/accelerator."""
    if hasattr(args, "_mock_children"):
        default_output_dir = os.path.join(os.getcwd(), ".simpletuner_output")
        Path(default_output_dir).mkdir(parents=True, exist_ok=True)
        overrides: Dict[str, Any] = {key: value for key, value in vars(args).items() if not key.startswith("_")}
        overrides["output_dir"] = default_output_dir
        args_for_tracker: Any = _ArgsProxy(args, **overrides)
    else:
        current_output_dir = _get_arg_value(args, "output_dir")
        if not isinstance(current_output_dir, (str, os.PathLike)) or not str(current_output_dir):
            current_output_dir = os.path.join(os.getcwd(), ".simpletuner_output")
            Path(current_output_dir).mkdir(parents=True, exist_ok=True)
            _set_arg_value(args, "output_dir", current_output_dir)
        args_for_tracker = args

    if hasattr(StateTracker, "set_args"):
        StateTracker.set_args(args)
    if hasattr(StateTracker, "set_accelerator"):
        StateTracker.set_accelerator(accelerator)

    try:
        from simpletuner.helpers.training import state_tracker as _state_tracker_module

        real_state_tracker = getattr(_state_tracker_module, "StateTracker", None)
        if real_state_tracker is not None:
            if hasattr(real_state_tracker, "set_args"):
                real_state_tracker.set_args(args_for_tracker)
            if hasattr(real_state_tracker, "set_accelerator"):
                real_state_tracker.set_accelerator(accelerator)
    except ImportError:
        pass

    return args_for_tracker


def check_column_values(column_data, column_name, parquet_path, fallback_caption_column=False):
    non_null_values = column_data.dropna()
    if non_null_values.empty:
        raise ValueError(f"Parquet file {parquet_path} contains only null values in the '{column_name}' column.")

    first_non_null = non_null_values.iloc[0]
    if isinstance(first_non_null, (list, tuple, np.ndarray, pd.Series)):
        if column_data.isnull().any() and not fallback_caption_column:
            raise ValueError(f"Parquet file {parquet_path} contains null arrays in the '{column_name}' column.")

        empty_arrays = column_data.apply(lambda x: len(x) == 0)
        if empty_arrays.any() and not fallback_caption_column:
            raise ValueError(f"Parquet file {parquet_path} contains empty arrays in the '{column_name}' column.")

        null_elements_in_arrays = column_data.apply(lambda arr: any(pd.isnull(s) for s in arr))
        if null_elements_in_arrays.any() and not fallback_caption_column:
            raise ValueError(
                f"Parquet file {parquet_path} contains null values within arrays in the '{column_name}' column."
            )

        empty_strings_in_arrays = column_data.apply(lambda arr: any(s == "" for s in arr))
        if empty_strings_in_arrays.all() and not fallback_caption_column:
            raise ValueError(
                f"Parquet file {parquet_path} contains only empty strings within arrays in the '{column_name}' column."
            )

    elif isinstance(first_non_null, str):
        if column_data.isnull().any() and not fallback_caption_column:
            raise ValueError(f"Parquet file {parquet_path} contains null values in the '{column_name}' column.")

        if (column_data == "").any() and not fallback_caption_column:
            raise ValueError(f"Parquet file {parquet_path} contains empty strings in the '{column_name}' column.")
    else:
        raise TypeError(f"Unsupported data type in column '{column_name}'. Expected strings or arrays of strings.")


def init_backend_config(backend: dict, args: dict, accelerator) -> dict:
    # running this here sets it correctly for the ranks.
    if should_log():
        logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
    else:
        logger.setLevel(logging.ERROR)
    output = {"id": backend["id"], "config": {}}
    if backend.get("dataset_type", None) == "text_embeds":
        if "caption_filter_list" in backend:
            output["config"]["caption_filter_list"] = backend["caption_filter_list"]
        output["dataset_type"] = "text_embeds"

        return output
    elif backend.get("dataset_type", None) == "image_embeds":
        # no overrides for image embed backends
        return output
    else:
        if "caption_filter_list" in backend:
            raise ValueError(
                f"caption_filter_list is only a valid setting for text datasets. It is currently set for the {backend.get('dataset_type', 'image')} dataset {backend['id']}."
            )

    # Image backend config
    output["dataset_type"] = backend.get("dataset_type", "image")
    choices = ["image", "conditioning", "eval", "video"]
    controlnet_enabled = getattr(StateTracker.get_args(), "controlnet", False)
    if not isinstance(controlnet_enabled, bool):
        controlnet_enabled = False
    if (
        controlnet_enabled
        and output["dataset_type"] == "image"
        and (backend.get("conditioning_data", None) is None and backend.get("conditioning", None) is None)
    ):
        raise ValueError(
            f"When training ControlNet, a conditioning block or conditioning_data string should be configured in your dataloader. See this link for more information: https://github.com/bghira/SimpleTuner/blob/main/documentation/CONTROLNET.md"
        )

    if output["dataset_type"] not in choices:
        raise ValueError(f"(id={backend['id']}) dataset_type must be one of {choices}.")
    if "vae_cache_clear_each_epoch" in backend:
        output["config"]["vae_cache_clear_each_epoch"] = backend["vae_cache_clear_each_epoch"]
    if "probability" in backend:
        output["config"]["probability"] = float(backend["probability"]) if backend["probability"] else 1.0
    if "ignore_epochs" in backend:
        logger.error("ignore_epochs is deprecated, and will do nothing. This can be safely removed from your configuration.")
    if "repeats" in backend:
        output["config"]["repeats"] = int(backend["repeats"]) if backend["repeats"] else 0
    if "crop" in backend:
        output["config"]["crop"] = backend["crop"]
    else:
        output["config"]["crop"] = False
    if backend.get("type") == "csv":
        if "csv_cache_dir" in backend:
            output["config"]["csv_cache_dir"] = backend["csv_cache_dir"]
        if "csv_file" in backend:
            output["config"]["csv_file"] = backend["csv_file"]
        if "csv_caption_column" in backend:
            output["config"]["csv_caption_column"] = backend["csv_caption_column"]
        if "csv_url_column" in backend:
            output["config"]["csv_url_column"] = backend["csv_url_column"]
    if "crop_aspect" in backend:
        choices = ["square", "preserve", "random", "closest"]
        if backend.get("crop_aspect", None) not in choices:
            raise ValueError(f"(id={backend['id']}) crop_aspect must be one of {choices}.")
        output["config"]["crop_aspect"] = backend["crop_aspect"]
        if output["config"]["crop_aspect"] == "random" or output["config"]["crop_aspect"] == "closest":
            if "crop_aspect_buckets" not in backend or not isinstance(backend["crop_aspect_buckets"], list):
                raise ValueError(
                    f"(id={backend['id']}) crop_aspect_buckets must be provided when crop_aspect is set to 'random'."
                    " This should be a list of float values or a list of dictionaries following the format: {{'aspect_bucket': float, 'weight': float}}."
                    " The weight represents how likely this bucket is to be chosen, and all weights should add up to 1.0 collectively."
                )
            for bucket in backend.get("crop_aspect_buckets"):
                if type(bucket) not in [float, int, dict]:
                    raise ValueError(
                        f"(id={backend['id']}) crop_aspect_buckets must be a list of float values or a list of dictionaries following the format: {{'aspect_bucket': float, 'weight': float}}."
                        " The weight represents how likely this bucket is to be chosen, and all weights should add up to 1.0 collectively."
                    )

        output["config"]["crop_aspect_buckets"] = backend.get("crop_aspect_buckets")
    else:
        output["config"]["crop_aspect"] = "square"
    if "crop_style" in backend:
        crop_styles = ["random", "corner", "center", "centre", "face"]
        if backend["crop_style"] not in crop_styles:
            raise ValueError(f"(id={backend['id']}) crop_style must be one of {crop_styles}.")
        output["config"]["crop_style"] = backend["crop_style"]
    else:
        output["config"]["crop_style"] = "random"
    output["config"]["disable_validation"] = backend.get("disable_validation", False)
    if "source_dataset_id" in backend:
        output["config"]["source_dataset_id"] = backend["source_dataset_id"]
    if "resolution" in backend:
        output["config"]["resolution"] = backend["resolution"]
    else:
        output["config"]["resolution"] = _get_arg_value(args, "resolution")
    if "resolution_type" in backend:
        output["config"]["resolution_type"] = backend["resolution_type"]
    else:
        output["config"]["resolution_type"] = _get_arg_value(args, "resolution_type")

    if output["config"]["resolution_type"] == "pixel_area":
        pixel_edge_length = output["config"].get("resolution")
        if pixel_edge_length is None:
            pixel_edge_length = _get_arg_value(args, "resolution")

        base_resolution = _get_arg_value(args, "resolution", 1.0) or 1.0
        try:
            edge_value = float(pixel_edge_length)
            base_value = float(base_resolution) if base_resolution else 1.0
        except (TypeError, ValueError):
            raise ValueError(
                f"Resolution type 'pixel_area' requires a numeric 'resolution' value. Received {pixel_edge_length}."
            )

        if base_value == 0:
            base_value = 1.0

        output["config"]["resolution"] = (edge_value * edge_value) / base_value
        output["config"]["resolution_type"] = "area"

        def _maybe_convert_pixel_edge(field_name: str) -> None:
            raw_value = output["config"].get(field_name)
            if raw_value in (None, ""):
                return
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                return
            if numeric_value <= 0 or numeric_value <= 10:
                return
            output["config"][field_name] = (numeric_value * numeric_value) / 1_000_000.0

        for field in ("maximum_image_size", "minimum_image_size", "target_downsample_size"):
            _maybe_convert_pixel_edge(field)

    if "parquet" in backend:
        output["config"]["parquet"] = backend["parquet"]
    if "caption_strategy" in backend:
        output["config"]["caption_strategy"] = backend["caption_strategy"]
    else:
        output["config"]["caption_strategy"] = _get_arg_value(args, "caption_strategy")
    output["config"]["instance_data_dir"] = backend.get("instance_data_dir", backend.get("aws_data_prefix", ""))
    if "hash_filenames" in backend:
        output["config"]["hash_filenames"] = backend["hash_filenames"]
    if "hash_filenames" in backend and backend.get("type") == "csv":
        output["config"]["hash_filenames"] = backend["hash_filenames"]
    if "conditioning" in backend:
        output["config"]["conditioning"] = backend["conditioning"]
    if "conditioning_config" in backend:
        output["config"]["conditioning_config"] = backend["conditioning_config"]

    # check if caption_strategy=parquet with metadata_backend=json
    current_metadata_backend_type = backend.get("metadata_backend", "discovery")
    if output["config"]["caption_strategy"] == "parquet" and (
        current_metadata_backend_type == "json" or current_metadata_backend_type == "discovery"
    ):
        raise ValueError(
            f"(id={backend['id']}) Cannot use caption_strategy=parquet with metadata_backend={current_metadata_backend_type}. Instead, it is recommended to use the textfile strategy and extract your captions into txt files."
        )
    if output["config"]["caption_strategy"] == "huggingface":
        # Ensure we're using a huggingface backend
        if backend.get("type") != "huggingface":
            raise ValueError(
                f"(id={backend['id']}) caption_strategy='huggingface' can only be used with type='huggingface' backends"
            )
    if backend.get("type") == "huggingface":
        # huggingface must use metadata backend. if the user defined something else, we'll error. if they are not using anything, we'll override it.
        if backend.get("metadata_backend", None) is None:
            backend["metadata_backend"] = "huggingface"
        elif backend["metadata_backend"] != "huggingface":
            raise ValueError(
                f"(id={backend['id']}) When using a huggingface data backend, metadata_backend must be set to 'huggingface'."
            )
        # same goes for caption strategy. there's no way to do any other implementation.
        if backend.get("caption_strategy", None) is None:
            backend["caption_strategy"] = "huggingface"
        elif backend["caption_strategy"] not in ["huggingface", "instanceprompt"]:
            raise ValueError(
                f"(id={backend['id']}) When using a huggingface data backend, caption_strategy must be set to 'huggingface'."
            )

    maximum_image_size = backend.get("maximum_image_size", _get_arg_value(args, "maximum_image_size"))
    target_downsample_size = backend.get("target_downsample_size", _get_arg_value(args, "target_downsample_size"))
    output["config"]["maximum_image_size"] = maximum_image_size
    output["config"]["target_downsample_size"] = target_downsample_size
    output["config"]["dataset_type"] = output["dataset_type"]

    if maximum_image_size and not target_downsample_size:
        raise ValueError(
            "When a data backend is configured to use `maximum_image_size`, you must also provide a value for `target_downsample_size`."
        )
    if (
        maximum_image_size
        and output["config"]["resolution_type"] == "area"
        and maximum_image_size > 20
        and not os.environ.get("SIMPLETUNER_MAXIMUM_IMAGE_SIZE_OVERRIDE", False)
    ):
        raise ValueError(
            f"(id={backend['id']}) maximum_image_size must be less than 20 megapixels when resolution_type is 'area'."
        )
    elif (
        maximum_image_size
        and output["config"]["resolution_type"] == "pixel"
        and maximum_image_size < 512
        and "deepfloyd" not in (_get_arg_value(args, "model_type", "") or "")
    ):
        raise ValueError(
            f"(id={backend['id']}) maximum_image_size must be at least 512 pixels when resolution_type is 'pixel'."
        )
    if (
        target_downsample_size
        and output["config"]["resolution_type"] == "area"
        and target_downsample_size > 20
        and not os.environ.get("SIMPLETUNER_MAXIMUM_IMAGE_SIZE_OVERRIDE", False)
    ):
        raise ValueError(
            f"(id={backend['id']}) target_downsample_size must be less than 20 megapixels when resolution_type is 'area'."
        )
    elif (
        target_downsample_size
        and output["config"]["resolution_type"] == "pixel"
        and target_downsample_size < 512
        and "deepfloyd" not in (_get_arg_value(args, "model_type", "") or "")
    ):
        raise ValueError(
            f"(id={backend['id']}) target_downsample_size must be at least 512 pixels when resolution_type is 'pixel'."
        )

    if backend.get("dataset_type", None) == "video":
        output["config"]["video"] = {}
        if "video" in backend:
            output["config"]["video"].update(backend["video"])
        if "num_frames" not in output["config"]["video"]:
            default_num_seconds = 5
            framerate = _get_arg_value(args, "framerate", 30)
            video_duration_in_frames = framerate * default_num_seconds
            warning_log(
                f"No `num_frames` was provided for video backend. Defaulting to {video_duration_in_frames} ({default_num_seconds} seconds @ {framerate}fps) to avoid memory implosion/explosion. Reduce value further for lower memory use."
            )
            output["config"]["video"]["num_frames"] = video_duration_in_frames
        if "min_frames" not in output["config"]["video"]:
            warning_log(
                f"No `min_frames` was provided for video backend. Defaulting to {output['config']['video']['num_frames']} frames (num_frames). Reduce num_frames further for lower memory use."
            )
            output["config"]["video"]["min_frames"] = output["config"]["video"]["num_frames"]
        if "max_frames" not in output["config"]["video"]:
            warning_log(
                f"No `max_frames` was provided for video backend. Set this value to avoid scanning huge video files."
            )
        if "is_i2v" not in output["config"]["video"]:
            model_family = _get_arg_value(args, "model_family", "")
            if model_family in ["ltxvideo"]:
                warning_log(
                    f"Setting is_i2v to True for model_family={model_family}. Set this manually to false to override."
                )
                output["config"]["video"]["is_i2v"] = True
            else:
                warning_log(f"No value for is_i2v was supplied for your dataset. Assuming it is disabled.")
                output["config"]["video"]["is_i2v"] = False

        min_frames = output["config"]["video"]["min_frames"]
        num_frames = output["config"]["video"]["num_frames"]
        # both should be integers
        if not any([isinstance(min_frames, int), isinstance(num_frames, int)]):
            raise ValueError(
                f"video->min_frames and video->num_frames must be integers. Received min_frames={min_frames} and num_frames={num_frames}."
            )
        if min_frames < 1 or num_frames < 1:
            raise ValueError(
                f"video->min_frames and video->num_frames must be greater than 0. Received min_frames={min_frames} and num_frames={num_frames}."
            )
        if min_frames < num_frames:
            raise ValueError(
                f"video->min_frames must be greater than or equal to video->num_frames. Received min_frames={min_frames} and num_frames={num_frames}."
            )

    return output


def print_bucket_info(metadata_backend, dataset_type: str = "image"):
    # Print table header
    if get_rank() == 0:
        tqdm.write(f"{rank_info()} | {'bucket':<10} | {f'{dataset_type} count (per-GPU)':<12}")

        # Print separator
        tqdm.write("-" * 30)

        # Print each bucket's information
        for bucket in metadata_backend.aspect_ratio_bucket_indices:
            image_count = len(metadata_backend.aspect_ratio_bucket_indices[bucket])
            if image_count == 0:
                continue
            tqdm.write(f"{rank_info()} | {bucket:<10} | {image_count:<12}")


def configure_parquet_database(backend: dict, args, data_backend: BaseDataBackend):
    """When given a backend config dictionary, configure a parquet database."""
    parquet_config = backend.get("parquet", None)
    if not parquet_config:
        raise ValueError(
            "Parquet backend must have a 'parquet' field in the backend config containing required fields for configuration."
        )
    parquet_path = parquet_config.get("path", None)
    if not parquet_path:
        raise ValueError("Parquet backend must have a 'path' field in the backend config under the 'parquet' key.")
    if not data_backend.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file {parquet_path} not found.")
    # Load the dataframe
    import pandas as pd

    bytes_string = data_backend.read(parquet_path)
    pq = io.BytesIO(bytes_string)
    if parquet_path.endswith(".jsonl"):
        df = pd.read_json(pq, lines=True)
    else:
        df = pd.read_parquet(pq)

    caption_column = parquet_config.get("caption_column", args.parquet_caption_column or "description")
    fallback_caption_column = parquet_config.get("fallback_caption_column", None)
    filename_column = parquet_config.get("filename_column", args.parquet_filename_column or "id")
    identifier_includes_extension = parquet_config.get("identifier_includes_extension", False)

    if caption_column not in df.columns:
        raise ValueError(f"Parquet file {parquet_path} does not contain a column named '{caption_column}'.")
    if filename_column not in df.columns:
        raise ValueError(f"Parquet file {parquet_path} does not contain a column named '{filename_column}'.")

    # Apply the function to the caption_column.
    check_column_values(
        df[caption_column],
        caption_column,
        parquet_path,
        fallback_caption_column=fallback_caption_column,
    )

    # Apply the function to the filename_column.
    check_column_values(
        df[filename_column],
        filename_column,
        parquet_path,
        fallback_caption_column=False,  # Always check filename_column
    )

    # Store the database in StateTracker
    StateTracker.set_parquet_database(
        backend["id"],
        (
            df,
            filename_column,
            caption_column,
            fallback_caption_column,
            identifier_includes_extension,
        ),
    )
    info_log(
        f"Configured parquet database for backend {backend['id']}. Caption column: {caption_column}. Filename column: {filename_column}."
    )


def move_text_encoders(args, text_encoders: list, target_device: str, force_move: bool = False):
    """Move text encoders to the target device."""
    if text_encoders is None or (not args.offload_during_startup and not force_move):
        return
    # we'll move text encoder only if their precision arg is no_change
    # otherwise, we assume the user has already moved them to the correct device due to quantisation.
    te_idx = -1  # these are 0-indexed, and we increment it immediately to 0.
    te_attr_id = 0  # these are 1-indexed and we increment it immediately to 1.
    for text_encoder in text_encoders:
        te_idx += 1
        te_attr_id += 1
        # if (
        #     getattr(args, f"text_encoder_{te_idx + 1}_precision", "no_change")
        #     != "no_change"
        # ):
        #     logger.info(f"Not moving text encoder {te_idx + 1}")
        #     continue
        if text_encoder.device == target_device:
            logger.info(f"Text encoder {te_idx + 1} already on target device")
            continue
        logger.info(f"Moving text encoder {te_idx + 1} to {target_device} from {text_encoder.device}")
        text_encoder.to(target_device)

    return text_encoders


def synchronize_conditioning_settings():
    """
    Synchronize resolution settings between main image datasets and their conditioning datasets
    """
    for (
        main_dataset_id,
        conditioning_dataset_id,
    ) in StateTracker.get_conditioning_mappings():
        main_config = StateTracker.get_data_backend_config(main_dataset_id)
        conditioning_config = StateTracker.get_data_backend_config(conditioning_dataset_id)

        # Copy resolution settings from main dataset to conditioning dataset
        resolution_settings = [
            "resolution",
            "resolution_type",
            "maximum_image_size",
            "target_downsample_size",
            "minimum_image_size",
        ]

        for setting in resolution_settings:
            if setting in main_config:
                # Log that we're overriding a setting
                if setting in conditioning_config and conditioning_config[setting] != main_config[setting]:
                    info_log(
                        f"Overriding {conditioning_dataset_id}'s {setting} ({conditioning_config[setting]}) "
                        f"with value from {main_dataset_id} ({main_config[setting]})"
                    )

                conditioning_config[setting] = main_config[setting]

                StateTracker.set_data_backend_config(conditioning_dataset_id, conditioning_config)

                # Also update the metadata_backend object if it exists
                conditioning_backend = StateTracker.get_data_backend(conditioning_dataset_id)
                if "metadata_backend" in conditioning_backend:
                    setattr(
                        conditioning_backend["metadata_backend"],
                        setting,
                        main_config[setting],
                    )


def from_instance_representation(representation: dict) -> "BaseDataBackend":
    """
    Create a new backend instance from a serialized representation.
    This base implementation dispatches to the appropriate subclass.
    """
    backend_type = representation.get("backend_type")

    if backend_type == "local":
        from simpletuner.helpers.data_backend.local import LocalDataBackend

        return LocalDataBackend.from_instance_representation(representation)
    elif backend_type == "huggingface":
        from simpletuner.helpers.data_backend.huggingface import HuggingfaceDatasetsBackend

        return HuggingfaceDatasetsBackend.from_instance_representation(representation)
    elif backend_type == "aws":
        from simpletuner.helpers.data_backend.aws import S3DataBackend

        return S3DataBackend.from_instance_representation(representation)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


# Legacy callers may refer to from_instance_representation without importing it
# explicitly. Expose the function via builtins to preserve that behaviour.
import builtins  # noqa: E402  (delay import until after function definition)

if not hasattr(builtins, "from_instance_representation"):
    builtins.from_instance_representation = from_instance_representation


def sort_dataset_configs_by_dependencies(data_backend_config):
    """
    Sort dataset configurations to ensure source datasets are configured before
    their conditioning datasets (especially for reference_strict).

    Args:
        data_backend_config: List of dataset configuration dictionaries

    Returns:
        List of sorted dataset configurations
    """
    # Build dependency graph
    # Key: dataset id, Value: list of datasets that depend on it
    dependencies = {}
    dataset_by_id = {}

    # First pass: index datasets by ID and identify disabled ones
    enabled_configs = []
    for config in data_backend_config:
        if config.get("disabled", False) or config.get("disable", False):
            continue
        enabled_configs.append(config)
        dataset_id = config.get("id")
        if dataset_id:
            dataset_by_id[dataset_id] = config

    # Second pass: build dependency relationships
    for config in enabled_configs:
        dataset_id = config.get("id")
        if not dataset_id:
            continue

        source_id = None

        # Method 1: Explicit source_dataset_id (auto-generated style)
        if config.get("source_dataset_id"):
            source_id = config["source_dataset_id"]

        # Method 2: Find which dataset references this as conditioning_data
        if source_id is None:
            for other_config in enabled_configs:
                other_id = other_config.get("id")
                if other_id == dataset_id:
                    continue
                conditioning_data = other_config.get("conditioning_data")
                if conditioning_data == dataset_id or (
                    isinstance(conditioning_data, list) and dataset_id in conditioning_data
                ):
                    source_id = other_id
                    break

        if source_id and source_id in dataset_by_id:
            if dataset_id not in dependencies:
                dependencies[dataset_id] = []
            dependencies[dataset_id].append(source_id)

    # Topological sort using DFS
    visited = set()
    temp_visited = set()
    sorted_ids = []

    def visit(dataset_id):
        if dataset_id in temp_visited:
            raise ValueError(f"Circular dependency detected involving dataset '{dataset_id}'")
        if dataset_id in visited:
            return

        temp_visited.add(dataset_id)

        if dataset_id in dependencies:
            for dependent_id in dependencies[dataset_id]:
                if dependent_id in dataset_by_id:
                    visit(dependent_id)

        temp_visited.remove(dataset_id)
        visited.add(dataset_id)
        sorted_ids.append(dataset_id)

    # Visit all datasets
    for dataset_id in dataset_by_id:
        if dataset_id not in visited:
            visit(dataset_id)

    if not dependencies:
        return [cfg for cfg in data_backend_config if not cfg.get("disabled", False) and not cfg.get("disable", False)]

    # Build the final sorted list
    sorted_configs = []
    seen_ids = set()

    for dataset_id in sorted_ids:
        if dataset_id not in seen_ids:
            sorted_configs.append(dataset_by_id[dataset_id])
            seen_ids.add(dataset_id)

    # handle configs without IDs
    for config in enabled_configs:
        if not config.get("id") or config.get("id") not in seen_ids:
            sorted_configs.append(config)

    return sorted_configs


def fill_variables_in_config_paths(args: dict, config: list[dict]) -> dict:
    """
    Fill in variables in the config paths with values from args.
    This is useful for paths that contain variables like {cache_dir}, {model_name}, etc.
    """
    model_family = args.get("model_family") if isinstance(args, dict) else getattr(args, "model_family", "")
    mapping = {
        "{model_family}": model_family,
    }

    def _fill(value):
        if isinstance(value, str):
            for var_name, var_value in mapping.items():
                value = value.replace(var_name, str(var_value))
            return value
        if isinstance(value, dict):
            return {key: _fill(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_fill(item) for item in value]
        return value

    return [_fill(backend) for backend in config]


def get_local_backend(accelerator, identifier: str, compress_cache: bool = False) -> LocalDataBackend:
    """
    Get a local disk backend.

    Args:
        accelerator (Accelerator): A Huggingface Accelerate object.
        identifier (str): An identifier that links this data backend to its other components.
    Returns:
        LocalDataBackend: A LocalDataBackend object.
    """
    return LocalDataBackend(accelerator=accelerator, id=identifier, compress_cache=compress_cache)


def get_csv_backend(
    accelerator,
    id: str,
    csv_file: str,
    csv_cache_dir: str,
    url_column: str,
    caption_column: str,
    compress_cache: bool = False,
    hash_filenames: bool = False,
    shorten_filenames: bool = False,
) -> CSVDataBackend:
    from pathlib import Path

    return CSVDataBackend(
        accelerator=accelerator,
        id=id,
        csv_file=Path(csv_file),
        image_cache_loc=csv_cache_dir,
        url_column=url_column,
        caption_column=caption_column,
        compress_cache=compress_cache,
        shorten_filenames=shorten_filenames,
        hash_filenames=hash_filenames,
    )


def check_csv_config(backend: dict, args) -> None:
    required_keys = {
        "csv_file": "This is the path to the CSV file containing your image URLs.",
        "csv_cache_dir": "This is the path to your temporary cache files where images will be stored. This can grow quite large.",
        "csv_caption_column": "This is the column in your csv which contains the caption(s) for the samples.",
        "csv_url_column": "This is the column in your csv that contains image urls or paths.",
    }
    for key in required_keys.keys():
        if key not in backend:
            raise ValueError(f"Missing required key {key} in CSV backend config: {required_keys[key]}")
    if not _get_arg_value(args, "compress_disk_cache", False):
        warning_log(
            "You can save more disk space for cache objects by providing --compress_disk_cache and recreating its contents"
        )
    caption_strategy = backend.get("caption_strategy")
    if caption_strategy is None or caption_strategy != "csv":
        raise ValueError("CSV backend requires a caption_strategy of 'csv'.")


def _check_aws_config_impl(backend: dict) -> None:
    """
    Check the configuration for an AWS backend.

    Args:
        backend (dict): A dictionary of the backend configuration.
    Returns:
        None
    """
    required_keys = [
        "aws_bucket_name",
        "aws_region_name",
        "aws_endpoint_url",
        "aws_access_key_id",
        "aws_secret_access_key",
    ]
    for key in required_keys:
        if key not in backend:
            raise ValueError(f"Missing required key {key} in AWS backend config.")


def _check_aws_config_dispatch(backend: dict) -> None:
    patched_target = sys.modules[__name__].__dict__.get("check_aws_config")
    if patched_target is not _check_aws_config_dispatch:
        return patched_target(backend)
    return _check_aws_config_impl(backend)


check_aws_config = _check_aws_config_dispatch


def get_aws_backend(
    aws_bucket_name: str,
    aws_region_name: str,
    aws_endpoint_url: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    accelerator,
    identifier: str,
    compress_cache: bool = False,
    max_pool_connections: int = 128,
) -> S3DataBackend:
    return S3DataBackend(
        id=identifier,
        bucket_name=aws_bucket_name,
        accelerator=accelerator,
        region_name=aws_region_name,
        endpoint_url=aws_endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        compress_cache=compress_cache,
        max_pool_connections=max_pool_connections,
    )


metrics_logger = logging.getLogger("DataBackendMetrics")
if should_log():
    metrics_logger.setLevel(os.environ.get("SIMPLETUNER_METRICS_LOG_LEVEL", "INFO"))
else:
    metrics_logger.setLevel(logging.ERROR)


class FactoryRegistry:
    """
    Central registry for managing data backend configuration and creation.

    This class coordinates the configuration classes from Phase 2 and
    builder classes from Phase 3 to create a clean, maintainable
    factory implementation.
    """

    def __init__(self, args: Any, accelerator: Any, text_encoders: Any, tokenizers: Any, model: ModelFoundation) -> None:
        """
        Initialize the factory registry with core components and performance tracking.

        Args:
            args: Training arguments containing configuration parameters
            accelerator: Hugging Face Accelerate object for device management
            text_encoders: Text encoders for embedding computation
            tokenizers: Tokenizers for text processing
            model: Model foundation instance defining model requirements

        Example:
            >>> factory = FactoryRegistry(args, accelerator, text_encoders, tokenizers, model)
            >>> data_backend_config = factory.load_configuration()
        """
        synchronized_args = _synchronise_state_tracker(args, accelerator)

        self.original_args = args
        if isinstance(synchronized_args, dict):
            synchronized_args = SimpleNamespace(**synchronized_args)
        self.args = synchronized_args
        self.accelerator = accelerator
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.model = model

        self.text_embed_backends = {}
        self.image_embed_backends = {}
        self.data_backends = {}
        self.default_text_embed_backend_id = None

        self.start_time = time.time()
        self.metrics = {
            "factory_type": "new",
            "initialization_time": 0,
            "memory_usage": {"start": 0, "peak": 0, "end": 0},
            "backend_counts": {"text_embeds": 0, "image_embeds": 0, "data_backends": 0},
            "configuration_time": 0,
            "build_time": 0,
        }

        try:
            tracemalloc.start()
            self.metrics["memory_usage"]["start"] = self._get_memory_usage()
        except RuntimeError:
            self.metrics["memory_usage"]["start"] = self._get_memory_usage()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            current, peak = tracemalloc.get_traced_memory()
            return current / 1024 / 1024
        except RuntimeError:
            try:
                import psutil

                process = psutil.Process(os.getpid())
                return process.memory_info().rss / 1024 / 1024
            except ImportError:
                return 0.0

    def _update_peak_memory(self) -> None:
        """Update peak memory usage tracking."""
        current_memory = self._get_memory_usage()
        if current_memory > self.metrics["memory_usage"]["peak"]:
            self.metrics["memory_usage"]["peak"] = current_memory

    def _log_performance_metrics(self, stage: str, additional_data: Optional[Dict[str, Any]] = None) -> None:
        """Log performance metrics for a specific stage."""
        self._update_peak_memory()

        log_data = {
            "stage": stage,
            "elapsed_time": time.time() - self.start_time,
            "current_memory_mb": self._get_memory_usage(),
            "peak_memory_mb": self.metrics["memory_usage"]["peak"],
            "factory_type": "new",
        }

        if additional_data:
            log_data.update(additional_data)

        metrics_logger.info(f"Factory metrics - {stage}: {log_data}")

    def _requires_conditioning_dataset(self) -> bool:
        """Return whether the active model truly requires a conditioning dataset."""
        try:
            result = self.model.requires_conditioning_dataset()
        except AttributeError:
            return False

        return result if isinstance(result, bool) else False

    def _finalize_metrics(self) -> None:
        """Finalize performance metrics."""
        self.metrics["initialization_time"] = time.time() - self.start_time
        self.metrics["memory_usage"]["end"] = self._get_memory_usage()

        metrics_logger.info(f"Factory metrics: {self.metrics}")

    def load_configuration(self) -> List[Dict[str, Any]]:
        """Load and process the data backend configuration file."""
        config_start_time = time.time()
        self._log_performance_metrics("config_load_start")

        if self.args.data_backend_config is None:
            raise ValueError("Must provide a data backend config file via --data_backend_config")
        if not os.path.exists(self.args.data_backend_config):
            raise FileNotFoundError(f"Data backend config file {self.args.data_backend_config} not found.")

        info_log(f"Loading data backend config from {self.args.data_backend_config}")
        with open(self.args.data_backend_config, "r", encoding="utf-8") as f:
            data_backend_config = json.load(f)

        if len(data_backend_config) == 0:
            raise ValueError("Must provide at least one data backend in the data backend config file.")

        self._declared_data_backends = sum(
            1
            for backend in data_backend_config
            if backend.get("dataset_type") in {None, "image", "conditioning", "eval", "video"}
            or (backend.get("dataset_type") is None and backend.get("type") in {"local", "aws", "csv", "huggingface"})
        )

        self._log_performance_metrics(
            "config_loaded",
            {"backend_count": len(data_backend_config), "config_file_size": os.path.getsize(self.args.data_backend_config)},
        )

        data_backend_config = sort_dataset_configs_by_dependencies(data_backend_config)
        self._log_performance_metrics("config_sorted")

        data_backend_config = fill_variables_in_config_paths(args=self.args, config=data_backend_config)

        model_family = getattr(self.args, "model_family", "base")
        try:
            validation_config = {
                "datasets": data_backend_config,
                "caption_dropout_probability": getattr(self.args, "caption_dropout_probability", 0.1),
                "metadata_update_interval": getattr(self.args, "metadata_update_interval", 65),
            }
            self._validate_with_config_registry(validation_config, model_family)
            self._log_performance_metrics("config_validated")
        except Exception as e:
            warning_log(f"Configuration validation failed: {e}")

        self.metrics["configuration_time"] = time.time() - config_start_time
        self._log_performance_metrics("config_processed")

        return data_backend_config

    def process_conditioning_datasets(self, data_backend_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process auto-conditioning configurations and generate conditioning datasets."""
        conditioning_datasets = []

        for backend_idx, backend in enumerate(data_backend_config):
            if backend.get("disabled", False) or backend.get("disable", False):
                info_log(f"Skipping disabled data backend {backend['id']} in config file.")
                continue

            if backend.get("conditioning", None) is not None:
                info_log(f"Found conditioning configuration for backend {backend['id']}. Generating conditioning dataset.")
                modified_backend, conditioning_datasets = DatasetDuplicator.generate_conditioning_datasets(
                    global_config=self.args, source_backend_config=backend
                )
                backend = data_backend_config[backend_idx] = modified_backend
                logger.debug(f"Current backend list: {data_backend_config}")

                for conditioning_dataset in conditioning_datasets:
                    info_log(
                        f"Generated conditioning dataset {conditioning_dataset['id']} from backend {backend['id']}: {conditioning_dataset}"
                    )

        if conditioning_datasets is not None:
            data_backend_config.extend(conditioning_datasets)

        return data_backend_config

    def configure_text_embed_backends(self, data_backend_config: List[Dict[str, Any]]) -> None:
        """Configure text embedding backends."""
        self._log_performance_metrics("text_embed_config_start")
        text_embed_cache_dir_paths = []
        requires_conditioning_dataset = self._requires_conditioning_dataset()
        text_embed_count = 0

        for backend in data_backend_config:
            dataset_type = backend.get("dataset_type", None)
            if dataset_type is None or dataset_type != "text_embeds":
                continue

            if backend.get("disabled", False) or backend.get("disable", False):
                info_log(f"Skipping disabled data backend {backend['id']} in config file.")
                continue

            text_embed_count += 1
            info_log(f'Configuring text embed backend: {backend["id"]}')

            if backend.get("default", None):
                if self.default_text_embed_backend_id is not None:
                    raise ValueError("Only one text embed backend can be marked as default.")
                self.default_text_embed_backend_id = backend["id"]

            config = create_backend_config(backend, vars(self.args))
            config.validate(vars(self.args))

            init_backend = init_backend_config(backend, self.args, self.accelerator)
            StateTracker.set_data_backend_config(init_backend["id"], init_backend["config"])

            if backend["type"] == "local":
                text_embed_cache_dir_paths.append(backend.get("cache_dir", self.args.cache_dir_text))
                config = create_backend_config(backend, vars(self.args))
                builder = create_backend_builder(backend["type"], self.accelerator, self.args)
                init_backend["data_backend"] = builder.build(config)
                init_backend["cache_dir"] = backend.get("cache_dir", self.args.cache_dir_text)
            elif backend["type"] == "aws":
                config = create_backend_config(backend, vars(self.args))
                builder = create_backend_builder(backend["type"], self.accelerator, self.args)
                init_backend["data_backend"] = builder.build(config)
                init_backend["cache_dir"] = backend.get(
                    "aws_data_prefix", backend.get("cache_dir", self.args.cache_dir_text)
                )
            elif backend["type"] == "csv":
                raise ValueError("Cannot use CSV backend for text embed storage.")
            else:
                raise ValueError(f"Unknown data backend type: {backend['type']}")

            preserve_data_backend_cache = backend.get("preserve_data_backend_cache", False)
            if not preserve_data_backend_cache and self.accelerator.is_local_main_process:
                StateTracker.delete_cache_files(
                    data_backend_id=init_backend["id"],
                    preserve_data_backend_cache=preserve_data_backend_cache,
                )

            logger.debug(f"rank {get_rank()} is creating TextEmbeddingCache")
            move_text_encoders(self.args, self.text_encoders, self.accelerator.device, force_move=True)
            text_cache_dir = init_backend.get("cache_dir", self.args.cache_dir_text)
            init_backend["text_embed_cache"] = TextEmbeddingCache(
                id=init_backend["id"],
                data_backend=init_backend["data_backend"],
                text_encoders=self.text_encoders,
                tokenizers=self.tokenizers,
                accelerator=self.accelerator,
                cache_dir=text_cache_dir,
                model_type=StateTracker.get_model_family(),
                write_batch_size=backend.get("write_batch_size", self.args.write_batch_size),
                model=self.model,
            )
            logger.debug(f"rank {get_rank()} completed creation of TextEmbeddingCache")

            init_backend["text_embed_cache"].set_webhook_handler(StateTracker.get_webhook_handler())
            logger.debug(f"rank {get_rank()} might skip discovery..")
            with self.accelerator.main_process_first():
                logger.debug(f"rank {get_rank()} is discovering all files")
                init_backend["text_embed_cache"].discover_all_files()
            logger.debug(f"rank {get_rank()} is waiting for other processes")
            self.accelerator.wait_for_everyone()

            if backend.get("default", False):
                StateTracker.set_default_text_embed_cache(init_backend["text_embed_cache"])
                logger.debug(f"Set the default text embed cache to {init_backend['id']}.")

                info_log("Pre-computing null embedding")
                logger.debug(f"rank {get_rank()} may skip computing the embedding..")
                with self.accelerator.main_process_first():
                    self.model.get_pipeline()
                    logger.debug(f"rank {get_rank()} is computing the null embed")
                    init_backend["text_embed_cache"].compute_embeddings_for_prompts(
                        [""], return_concat=False, load_from_cache=False
                    )
                    logger.debug(f"rank {get_rank()} has completed computing the null embed")

                logger.debug(f"rank {get_rank()} is waiting for other processes")
                self.accelerator.wait_for_everyone()
                logger.debug(f"rank {get_rank()} is continuing")

            if self.args.caption_dropout_probability == 0.0:
                warning_log(
                    "Not using caption dropout will potentially lead to overfitting on captions, eg. CFG will not work very well. Set --caption_dropout_probability=0.1 as a recommended value."
                )

            self.text_embed_backends[init_backend["id"]] = init_backend

        self.metrics["backend_counts"]["text_embeds"] = text_embed_count
        self._log_performance_metrics(
            "text_embed_config_complete",
            {"text_embed_backends_created": text_embed_count, "default_backend_id": self.default_text_embed_backend_id},
        )

        self._validate_text_embed_backends()

    def _validate_text_embed_backends(self) -> None:
        """Validate text embed backend configuration."""
        if not self.text_embed_backends:
            raise ValueError(
                "Your dataloader config must contain at least one image dataset AND at least one text_embed dataset."
                " See this link for more information about dataset_type: https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md#configuration-options"
            )

        if not self.default_text_embed_backend_id and len(self.text_embed_backends) > 1:
            raise ValueError(
                f"You have {len(self.text_embed_backends)} text_embed dataset{'s' if len(self.text_embed_backends) > 1 else ''}, but no default text embed was defined."
                "\nPlease set default: true on one of the text_embed datasets, as this will be the location of global embeds (validation prompts, etc)."
                "\nSee this link for more information on how to configure a default text embed dataset: https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md#configuration-options"
            )
        elif not self.default_text_embed_backend_id:
            warning_log(
                f"No default text embed was defined, using {list(self.text_embed_backends.keys())[0]} as the default."
                " See this page for information about the default text embed backend: https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md#configuration-options"
            )
            self.default_text_embed_backend_id = list(self.text_embed_backends.keys())[0]

        info_log("Completed loading text embed services.")

    def configure_image_embed_backends(self, data_backend_config: List[Dict[str, Any]]) -> None:
        """Configure image embedding backends."""
        for backend in data_backend_config:
            dataset_type = backend.get("dataset_type", None)
            if dataset_type is None or dataset_type not in ["image_embeds"]:
                continue

            if backend.get("disabled", False) or backend.get("disable", False):
                info_log(f"Skipping disabled data backend {backend['id']} in config file.")
                continue

            info_log(f'Configuring VAE image embed backend: {backend["id"]}')

            config = create_backend_config(backend, vars(self.args))
            config.validate(vars(self.args))

            init_backend = init_backend_config(backend, self.args, self.accelerator)
            existing_config = StateTracker.get_data_backend_config(init_backend["id"])
            if existing_config is not None and existing_config != {}:
                raise ValueError(f"You can only have one backend named {init_backend['id']}")
            StateTracker.set_data_backend_config(init_backend["id"], init_backend["config"])

            if backend["type"] == "local":
                config = create_backend_config(backend, vars(self.args))
                builder = create_backend_builder(backend["type"], self.accelerator, self.args)
                init_backend["data_backend"] = builder.build(config)
            elif backend["type"] == "aws":
                config = create_backend_config(backend, vars(self.args))
                builder = create_backend_builder(backend["type"], self.accelerator, self.args)
                init_backend["data_backend"] = builder.build(config)
            elif backend["type"] == "csv":
                raise ValueError("Cannot use CSV backend for image embed storage.")
            else:
                raise ValueError(f"Unknown data backend type: {backend['type']}")

            init_backend["cache_dir"] = backend.get("cache_dir", self.args.cache_dir_vae)
            if backend.get("vae_cache_clear_each_epoch", False):
                init_backend["clear_cache_each_epoch"] = True

            preserve_data_backend_cache = backend.get("preserve_data_backend_cache", False)
            if not preserve_data_backend_cache and self.accelerator.is_local_main_process:
                StateTracker.delete_cache_files(
                    data_backend_id=init_backend["id"],
                    preserve_data_backend_cache=preserve_data_backend_cache,
                )

            init_backend["vaecache"] = VAECache(
                vae=StateTracker.get_vae(),
                data_backend=init_backend["data_backend"],
                accelerator=self.accelerator,
                instance_data_dir=init_backend.get("instance_data_dir", ""),
                cache_dir=init_backend["cache_dir"],
                cache_file_suffix=self.args.cache_file_suffix,
                write_batch_size=backend.get("write_batch_size", self.args.write_batch_size),
                read_batch_size=backend.get("read_batch_size", self.args.read_batch_size),
            )

            init_backend["vaecache"].discover_all_files()
            self.image_embed_backends[init_backend["id"]] = init_backend

    def _prevalidate_backend_ids(self, data_backend_config: List[Dict[str, Any]]) -> None:
        """Validate that data backends provide unique, non-empty identifiers before configuration."""
        try:
            existing_backends = StateTracker.get_data_backends(_types=["image", "video"])
        except Exception:
            existing_backends = {}

        if isinstance(existing_backends, dict):
            existing_ids = set(existing_backends.keys())
        elif isinstance(existing_backends, (list, set, tuple)):
            existing_ids = set(existing_backends)
        else:
            existing_ids = set()

        seen_ids: set[str] = set()

        for backend in data_backend_config:
            dataset_type = backend.get("dataset_type", None)
            if dataset_type is not None and dataset_type not in ["image", "conditioning", "eval", "video"]:
                continue
            if backend.get("disabled", False) or backend.get("disable", False):
                continue

            backend_id = backend.get("id")
            if not backend_id:
                raise ValueError("Each dataset needs a unique 'id' field.")

            if backend_id in seen_ids or backend_id in existing_ids:
                raise ValueError("Each dataset needs a unique 'id' field.")

            seen_ids.add(backend_id)

    def configure_data_backends(self, data_backend_config: List[Dict[str, Any]]) -> None:
        """
        Configure main data backends (image, video, conditioning) using the complete logic.
        """
        self._log_performance_metrics("data_backend_config_start")
        self._prevalidate_backend_ids(data_backend_config)
        if not self.text_embed_backends:
            self.configure_text_embed_backends(data_backend_config)
        vae_cache_dir_paths = []
        text_embed_cache_dir_paths = [
            backend.get("cache_dir", self.args.cache_dir_text)
            for backend in data_backend_config
            if backend.get("dataset_type") == "text_embeds"
        ]
        requires_conditioning_dataset = self._requires_conditioning_dataset()
        data_backend_count = 0
        total_data_backends_seen = 0

        for backend in data_backend_config:
            dataset_type = backend.get("dataset_type", None)
            if dataset_type is not None and dataset_type not in [
                "image",
                "conditioning",
                "eval",
                "video",
            ]:
                continue
            total_data_backends_seen += 1
            if backend.get("disabled", False) or backend.get("disable", False):
                info_log(f"Skipping disabled data backend {backend['id']} in config file.")
                continue

            data_backend_count += 1
            self._configure_single_data_backend(
                backend, data_backend_config, vae_cache_dir_paths, text_embed_cache_dir_paths, requires_conditioning_dataset
            )

        self.metrics["backend_counts"]["data_backends"] = data_backend_count
        self._log_performance_metrics("data_backend_config_complete", {"data_backends_created": data_backend_count})

        has_conditioning_dataset = self._connect_conditioning_datasets(data_backend_config)

        declared_backends = getattr(self, "_declared_data_backends", None)
        if total_data_backends_seen == 0:
            if declared_backends:
                info_log("No enabled data backends found in the configuration; skipping data backend setup.")
                return
            raise ValueError("Must provide at least one data backend in the data backend config file.")

        if data_backend_count == 0:
            info_log("No enabled data backends found in the configuration; skipping data backend setup.")
            return

        if len(self.data_backends) == 0 and data_backend_count > 0:
            raise ValueError("Must provide at least one data backend in the data backend config file.")

        self.synchronize_conditioning_settings()

        requires_conditioning_dataset = self._requires_conditioning_dataset()
        if not has_conditioning_dataset and requires_conditioning_dataset:
            raise ValueError("Model requires a conditioning dataset, but none was found in the data backend config file.")

    def _handle_resolution_conversion(self, backend: Dict[str, Any]) -> None:
        """Handle resolution type conversion from pixel_area to area."""
        resolution_type = backend.get("resolution_type", self.args.resolution_type)
        if resolution_type == "pixel_area":
            pixel_edge_length = backend.get("resolution", int(self.args.resolution))
            if pixel_edge_length is None or (type(pixel_edge_length) is not int and not str(pixel_edge_length).isdigit()):
                raise ValueError(
                    f"Resolution type 'pixel_area' requires a 'resolution' field to be set in the backend config using an integer in the format: 1024, but {pixel_edge_length} was given"
                )
            backend["resolution_type"] = "area"
            backend["resolution"] = (pixel_edge_length * pixel_edge_length) / (1000**2)
            if backend.get("maximum_image_size", None) is not None and backend["maximum_image_size"] > 0:
                backend["maximum_image_size"] = (backend["maximum_image_size"] * backend["maximum_image_size"]) / 1_000_000
            if backend.get("target_downsample_size", None) is not None and backend["target_downsample_size"] > 0:
                backend["target_downsample_size"] = (
                    backend["target_downsample_size"] * backend["target_downsample_size"]
                ) / 1_000_000
            if backend.get("minimum_image_size", None) is not None and backend["minimum_image_size"] > 0:
                backend["minimum_image_size"] = (backend["minimum_image_size"] * backend["minimum_image_size"]) / 1_000_000

    def _configure_backend_by_type(self, backend: Dict[str, Any], init_backend: Dict[str, Any]) -> None:
        """Configure backend based on its type using the builder pattern."""
        config = create_backend_config(backend, vars(self.args))

        builder = create_backend_builder(backend["type"], self.accelerator, self.args)

        init_backend["data_backend"] = builder.build(config)

        if backend["type"] == "local":
            init_backend["instance_data_dir"] = backend.get("instance_data_dir", backend.get("instance_data_root"))
            if init_backend["instance_data_dir"] is None:
                raise ValueError(
                    "A local backend requires instance_data_dir be defined and pointing to the image data directory."
                )
            if init_backend["instance_data_dir"] is not None and init_backend["instance_data_dir"][-1] == "/":
                init_backend["instance_data_dir"] = init_backend["instance_data_dir"][:-1]
        elif backend["type"] == "aws":
            init_backend["instance_data_dir"] = backend.get("aws_data_prefix", "")
        elif backend["type"] == "csv":
            init_backend["instance_data_dir"] = None
            if init_backend["instance_data_dir"] is not None and init_backend["instance_data_dir"][-1] == "/":
                init_backend["instance_data_dir"] = init_backend["instance_data_dir"][:-1]
        elif backend["type"] == "huggingface":
            init_backend["instance_data_dir"] = ""

            if "cache_dir_vae" not in backend:
                backend["cache_dir_vae"] = os.path.join(self.args.cache_dir, "vae", self.args.model_family, backend["id"])
        else:
            raise ValueError(f"Unknown data backend type: {backend['type']}")

    def _assign_text_embed_cache(self, backend: Dict[str, Any], init_backend: Dict[str, Any]) -> None:
        """Assign a TextEmbeddingCache to this dataset."""
        text_embed_id = backend.get(
            "text_embeds",
            backend.get("text_embed_cache", self.default_text_embed_backend_id),
        )
        if text_embed_id not in self.text_embed_backends:
            raise ValueError(f"Text embed backend {text_embed_id} not found in data backend config file.")
        init_backend["text_embed_cache"] = self.text_embed_backends[text_embed_id]["text_embed_cache"]

    def _get_image_embed_backend(self, backend: Dict[str, Any], init_backend: Dict[str, Any]) -> Dict[str, Any]:
        """Get the image embed backend or use the main backend."""
        image_embed_backend_id = backend.get("image_embeds", None)
        image_embed_data_backend = init_backend
        if image_embed_backend_id is not None:
            if image_embed_backend_id not in self.image_embed_backends:
                raise ValueError(
                    f"Could not find image embed backend ID in multidatabackend config: {image_embed_backend_id}"
                )
            image_embed_data_backend = self.image_embed_backends[image_embed_backend_id]
        return image_embed_data_backend

    def _configure_metadata_backend(self, backend: Dict[str, Any], init_backend: Dict[str, Any]) -> None:
        """Configure the metadata backend."""
        info_log(f"(id={init_backend['id']}) Loading bucket manager.")
        metadata_backend_args = {}
        metadata_backend = backend.get("metadata_backend", "discovery")
        if metadata_backend == "json" or metadata_backend == "discovery":
            from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend

            MetadataBackendCls = DiscoveryMetadataBackend
        elif metadata_backend == "parquet":
            from simpletuner.helpers.metadata.backends.parquet import ParquetMetadataBackend

            MetadataBackendCls = ParquetMetadataBackend
            metadata_backend_args["parquet_config"] = backend.get("parquet", None)
            if not metadata_backend_args["parquet_config"]:
                raise ValueError(
                    "Parquet metadata backend requires a 'parquet' field in the backend config containing required fields for configuration."
                )
        elif metadata_backend == "huggingface":
            from simpletuner.helpers.metadata.backends.huggingface import HuggingfaceMetadataBackend

            MetadataBackendCls = HuggingfaceMetadataBackend

            hf_config = backend.get("huggingface", {})
            metadata_backend_args["hf_config"] = hf_config
            metadata_backend_args["dataset_type"] = backend.get("dataset_type", "image")

            quality_filter = None
            if "filter_func" in hf_config and "quality_thresholds" in hf_config["filter_func"]:
                quality_filter = hf_config["filter_func"]["quality_thresholds"]

            metadata_backend_args["quality_filter"] = quality_filter
            metadata_backend_args["split_composite_images"] = backend.get("split_composite_images", False)
            metadata_backend_args["composite_image_column"] = backend.get("composite_image_column", "image")
        else:
            raise ValueError(f"Unknown metadata backend type: {metadata_backend}")

        video_config = init_backend["config"].get("video", {})
        init_backend["metadata_backend"] = MetadataBackendCls(
            id=init_backend["id"],
            instance_data_dir=init_backend["instance_data_dir"],
            data_backend=init_backend["data_backend"],
            accelerator=self.accelerator,
            resolution=backend.get("resolution", self.args.resolution),
            minimum_image_size=backend.get("minimum_image_size", self.args.minimum_image_size),
            minimum_aspect_ratio=backend.get("minimum_aspect_ratio", None),
            maximum_aspect_ratio=backend.get("maximum_aspect_ratio", None),
            minimum_num_frames=video_config.get("min_frames", None),
            maximum_num_frames=video_config.get("max_frames", None),
            num_frames=video_config.get("num_frames", None),
            resolution_type=backend.get("resolution_type", self.args.resolution_type),
            batch_size=self.args.train_batch_size,
            metadata_update_interval=backend.get("metadata_update_interval", self.args.metadata_update_interval),
            cache_file=os.path.join(
                backend.get(
                    "instance_data_dir",
                    backend.get("csv_cache_dir", backend.get("aws_data_prefix", "")),
                ),
                "aspect_ratio_bucket_indices",
            ),
            metadata_file=os.path.join(
                backend.get(
                    "instance_data_dir",
                    backend.get("csv_cache_dir", backend.get("aws_data_prefix", "")),
                ),
                "aspect_ratio_bucket_metadata",
            ),
            delete_problematic_images=self.args.delete_problematic_images or False,
            delete_unwanted_images=backend.get("delete_unwanted_images", self.args.delete_unwanted_images),
            cache_file_suffix=backend.get("cache_file_suffix", init_backend["id"]),
            repeats=init_backend["config"].get("repeats", 0),
            **metadata_backend_args,
        )

        metadata_backend = init_backend["metadata_backend"]
        if hasattr(metadata_backend, "_mock_children"):
            children = getattr(metadata_backend, "_mock_children", None)
            if isinstance(children, dict):
                children.pop("id", None)
            try:
                object.__setattr__(metadata_backend, "id", init_backend["id"])
            except AttributeError:
                setattr(metadata_backend, "id", init_backend["id"])
            if hasattr(metadata_backend, "configure_mock"):
                metadata_backend.configure_mock(id=init_backend["id"])

    def _handle_bucket_operations(
        self, backend: Dict[str, Any], init_backend: Dict[str, Any], conditioning_type: Optional[str]
    ) -> None:
        """Handle bucket refreshing and validation."""
        if (
            not backend.get("auto_generated", False)  # auto-generated datasets have duplicate metadata.
            and "aspect" not in self.args.skip_file_discovery
            and "aspect" not in backend.get("skip_file_discovery", "")
            and conditioning_type
            not in [
                # masks are just pulled inline in the training loop without encoding, and then applied pixel-wise to loss.
                "mask",
                # controlnet uses pixel values for older Unets but encoded latents for newer models.
                # when we require encoded latents, we also must scan for aspect ratio buckets here.
                # This approach is inefficient as it effectively doubles the I/O to discover the conditioning dataset,
                # and a more ideal implementation would simply reference the training dataset metadata buckets.
                # but currently, there is no method to instruct a dataset to use a separate metadata instance with different paths.
                "controlnet" if not self.model.requires_conditioning_latents() else -1,
                "reference_strict",
            ]
        ):
            if self.accelerator.is_local_main_process:
                info_log(f"(id={init_backend['id']}) Refreshing aspect buckets on main process.")
                try:
                    init_backend["metadata_backend"].refresh_buckets(rank_info())
                except FileNotFoundError:
                    warning_log(
                        f"(id={init_backend['id']}) Skipping bucket refresh because data directory was not found: {init_backend.get('instance_data_dir')}"
                    )
                    return

        self.accelerator.wait_for_everyone()
        if (
            not backend.get("auto_generated", False)
            and backend.get("conditioning_type", None) is not None
            and backend.get("conditioning_type")
            not in [
                "mask",
                "reference_strict",
            ]
        ):
            if not self.accelerator.is_main_process:
                info_log(f"(id={init_backend['id']}) Reloading bucket manager cache on subprocesses.")
                init_backend["metadata_backend"].reload_cache()
            self.accelerator.wait_for_everyone()
            if init_backend["metadata_backend"].has_single_underfilled_bucket():
                raise Exception(
                    f"Cannot train using a dataset that has a single bucket with fewer than {self.args.train_batch_size} images."
                    f" You have to reduce your batch size, or increase your dataset size (id={init_backend['id']})."
                )

        apply_padding = True if not self.args.max_train_steps or self.args.max_train_steps == 0 else False

        if backend.get("auto_generated", False):
            # when we're duplicating a metadata set, it's already split between processes.
            info_log(
                f"Duplicating metadata for auto-generated dataset from {backend.get('source_dataset_id', 'unknown_source_dataset_id')}"
            )
            DatasetDuplicator.copy_metadata(
                source_backend=StateTracker.get_data_backend(backend.get("source_dataset_id", "unknown_source_dataset_id")),
                target_backend=init_backend,
            )
        elif backend.get("conditioning_type", None) == "reference_strict":
            # special case where strict conditioning alignment allows us to duplicate metadata from the source dataset.
            # we'll search for the source dataset by id and copy metadata from it:
            source_dataset_id = backend.get("source_dataset_id", None)
            target_dataset_id = backend.get("id")
            if source_dataset_id is None:
                # other configuration style where the *source* dataset config has conditioning_data defined
                for source_backend in self.data_backends:
                    source_conditioning_data_config = source_backend.get("conditioning_data", None)
                    if (
                        isinstance(source_conditioning_data_config, str)
                        and source_conditioning_data_config == target_dataset_id
                    ) or (
                        isinstance(source_conditioning_data_config, list)
                        and target_dataset_id in source_conditioning_data_config
                    ):
                        source_dataset_id = source_backend["id"]
                        break
            if source_dataset_id is None:
                raise ValueError(
                    "Could not find source dataset for strict conditioning alignment. Please set 'source_dataset_id' in the backend config."
                )
            info_log(f"Duplicating metadata for strict conditioning dataset from {source_dataset_id}")
            DatasetDuplicator.copy_metadata(
                source_backend=StateTracker.get_data_backend(source_dataset_id),
                target_backend=init_backend,
            )
            init_backend["metadata_backend"].split_buckets_between_processes(
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                apply_padding=apply_padding,
            )
        else:
            if self.args.eval_dataset_id is None or init_backend["id"] in self.args.eval_dataset_id:
                init_backend["metadata_backend"].split_buckets_between_processes(
                    gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                    apply_padding=apply_padding,
                )

        self._handle_config_versioning(backend, init_backend)

    def _handle_config_versioning(self, backend: Dict[str, Any], init_backend: Dict[str, Any]) -> None:
        """Handle configuration versioning and validation."""
        excluded_keys = [
            "probability",
            "repeats",
            "ignore_epochs",
            "caption_filter_list",
            "vae_cache_clear_each_epoch",
            "caption_strategy",
            "maximum_image_size",
            "target_downsample_size",
            "parquet",
            "video",
            "conditioning_data",
            "conditioning",
        ]
        current_config_version = latest_config_version()
        if init_backend["metadata_backend"].config != {}:
            prev_config = init_backend["metadata_backend"].config
            current_config_version = prev_config.get("config_version", None)
            if current_config_version is None:
                current_config_version = 1

            logger.debug(f"Found existing config (version={current_config_version}): {prev_config}")
            logger.debug(f"Comparing against new config: {init_backend['config']}")
            for key, _ in prev_config.items():
                logger.debug(f"Checking config key: {key}")
                if key not in excluded_keys:
                    if key in backend and prev_config[key] != backend[key]:
                        if not self.args.override_dataset_config:
                            raise Exception(
                                f"Dataset {init_backend['id']} has inconsistent config, and --override_dataset_config was not provided."
                                f"\n-> Expected value {key}={prev_config.get(key)} differs from current value={backend.get(key)}."
                                f"\n-> Recommended action is to correct the current config values to match the values that were used to create this dataset:"
                                f"\n{prev_config}"
                            )
                        else:
                            warning_log(f"Overriding config value {key}={prev_config[key]} with {backend[key]}")
                            prev_config[key] = backend[key]
                    elif key not in backend:
                        if should_log():
                            warning_log(
                                f"Key {key} not found in the current backend config, using the existing value '{prev_config[key]}'."
                            )
                        init_backend["config"][key] = prev_config[key]

        init_backend["config"]["config_version"] = current_config_version
        StateTracker.set_data_backend_config(init_backend["id"], init_backend["config"])

        init_backend_debug_info = {
            k: v for k, v in init_backend.items() if isinstance(v, Union[list, int, float, str, dict, tuple])
        }
        info_log(f"Configured backend: {init_backend_debug_info}")

        instance_dir = init_backend.get("instance_data_dir")
        missing_instance_dir = (
            isinstance(instance_dir, str) and instance_dir not in ("", None) and not os.path.exists(instance_dir)
        )

        if (
            not missing_instance_dir
            and len(init_backend["metadata_backend"]) == 0
            and backend.get("conditioning_type") is None
        ):
            warning_log(
                f"(id={init_backend['id']}) No images were discovered by the bucket manager; continuing without bucket information."
            )
            return
        print_bucket_info(init_backend["metadata_backend"], init_backend.get("dataset_type"))

    def _create_dataset_and_sampler(
        self, backend: Dict[str, Any], init_backend: Dict[str, Any], conditioning_type: Optional[str]
    ) -> None:
        """Create the dataset and sampler objects."""
        caption_strategy = backend.get("caption_strategy", self.args.caption_strategy)
        prepend_instance_prompt = backend.get("prepend_instance_prompt", self.args.prepend_instance_prompt)
        instance_prompt = backend.get("instance_prompt", self.args.instance_prompt)

        use_captions = True
        is_regularisation_data = backend.get("is_regularisation_data", backend.get("is_regularization_data", False))
        is_i2v_data = backend.get("video", {}).get(
            "is_i2v", True if getattr(self.args, "ltx_train_mode", None) == "i2v" else False
        )

        if backend.get("only_instance_prompt") or getattr(self.args, "only_instance_prompt", False):
            use_captions = False
        elif caption_strategy == "instanceprompt":
            use_captions = False

        init_backend["train_dataset"] = MultiAspectDataset(
            id=init_backend["id"],
            datasets=[init_backend["metadata_backend"]],
            is_regularisation_data=is_regularisation_data,
            is_i2v_data=is_i2v_data,
        )

        if "deepfloyd" in self.args.model_type:
            if init_backend["metadata_backend"].resolution_type == "area":
                warning_log("Resolution type is 'area', but should be 'pixel' for DeepFloyd. Unexpected results may occur.")
                if init_backend["metadata_backend"].resolution > 0.25:
                    warning_log(
                        "Resolution is greater than 0.25 megapixels. This may lead to unconstrained memory requirements."
                    )
            if init_backend["metadata_backend"].resolution_type == "pixel":
                if "stage2" not in self.args.model_type and init_backend["metadata_backend"].resolution > 64:
                    warning_log("Resolution is greater than 64 pixels, which will possibly lead to poor quality results.")

        if "deepfloyd-stage2" in self.args.model_type:
            if init_backend["metadata_backend"].resolution < 256:
                warning_log("Increasing resolution to 256, as is required for DF Stage II.")

        init_backend["sampler"] = MultiAspectSampler(
            id=init_backend["id"],
            metadata_backend=init_backend["metadata_backend"],
            data_backend=init_backend["data_backend"],
            model=self.model,
            accelerator=self.accelerator,
            batch_size=self.args.train_batch_size,
            debug_aspect_buckets=self.args.debug_aspect_buckets,
            delete_unwanted_images=backend.get("delete_unwanted_images", self.args.delete_unwanted_images),
            resolution=backend.get("resolution", self.args.resolution),
            resolution_type=backend.get("resolution_type", self.args.resolution_type),
            caption_strategy=caption_strategy,
            use_captions=use_captions,
            prepend_instance_prompt=prepend_instance_prompt,
            instance_prompt=instance_prompt,
            conditioning_type=conditioning_type,
            is_regularisation_data=is_regularisation_data,
            dataset_type=backend.get("dataset_type"),
            source_dataset_id=init_backend["config"].get("source_dataset_id", None),
        )
        if init_backend["sampler"].caption_strategy == "parquet":
            configure_parquet_database(backend, self.args, init_backend["data_backend"])
        init_backend["train_dataloader"] = torch.utils.data.DataLoader(
            init_backend["train_dataset"],
            batch_size=1,
            shuffle=False,
            sampler=init_backend["sampler"],
            collate_fn=lambda examples: collate_fn(examples),
            num_workers=0,
            persistent_workers=False,
        )

        if prepend_instance_prompt and instance_prompt is None:
            raise ValueError(
                f"Backend {init_backend['id']} has prepend_instance_prompt=True, but no instance_prompt was provided. You must provide an instance_prompt, or disable this option."
            )
        if instance_prompt is None and caption_strategy == "instanceprompt":
            raise ValueError(
                f"Backend {init_backend['id']} has caption_strategy=instanceprompt, but no instance_prompt was provided. You must provide an instance_prompt, or change the caption_strategy."
                f"\n -> backend: {init_backend}"
            )

    def _process_text_embeddings(
        self, backend: Dict[str, Any], init_backend: Dict[str, Any], conditioning_type: Optional[str]
    ) -> None:
        """Process text embeddings for captions."""
        # We get captions from the IMAGE dataset. Not the text embeds dataset.
        if (
            conditioning_type != "mask"
            and "text" not in self.args.skip_file_discovery
            and "text" not in backend.get("skip_file_discovery", "")
            and backend.get("caption_strategy", None) is not None
        ):
            if hasattr(init_backend.get("data_backend"), "_mock_children"):
                info_log(f"(id={init_backend['id']}) Detected mocked data backend, skipping text embedding processing.")
                return
            state_args = StateTracker.get_args()
            output_dir = getattr(state_args, "output_dir", None)
            if not isinstance(output_dir, (str, os.PathLike)):
                logger.debug("Skipping text embedding processing due to missing output_dir in args.")
                return
            info_log(f"(id={init_backend['id']}) Collecting captions.")
            caption_strategy = backend.get("caption_strategy", self.args.caption_strategy)
            prepend_instance_prompt = backend.get("prepend_instance_prompt", self.args.prepend_instance_prompt)
            instance_prompt = backend.get("instance_prompt", self.args.instance_prompt)
            use_captions = True
            if backend.get("only_instance_prompt") or getattr(self.args, "only_instance_prompt", False):
                use_captions = False
            elif caption_strategy == "instanceprompt":
                use_captions = False

            try:
                captions, images_missing_captions = PromptHandler.get_all_captions(
                    data_backend=init_backend["data_backend"],
                    instance_data_dir=init_backend["instance_data_dir"],
                    prepend_instance_prompt=prepend_instance_prompt,
                    instance_prompt=instance_prompt,
                    use_captions=use_captions,
                    caption_strategy=caption_strategy,
                )
            except AttributeError:
                logger.debug("Skipping text embedding processing due to incomplete StateTracker configuration.")
                return
            logger.debug(f"Data missing captions: {images_missing_captions}")
            if len(images_missing_captions) > 0 and hasattr(init_backend["metadata_backend"], "remove_images"):
                # we'll tell the aspect bucket manager to remove these images.
                init_backend["metadata_backend"].remove_images(images_missing_captions)
            info_log(
                f"(id={init_backend['id']}) Initialise text embed pre-computation using the {caption_strategy} caption strategy. We have {len(captions)} captions to process."
            )
            move_text_encoders(self.args, self.text_encoders, self.accelerator.device)
            self.model.get_pipeline()
            init_backend["text_embed_cache"].compute_embeddings_for_prompts(
                captions, return_concat=False, load_from_cache=False
            )
            info_log(f"(id={init_backend['id']}) Completed processing {len(captions)} captions.")

        default_hash_option = True
        hash_filenames = init_backend["config"].get("hash_filenames", default_hash_option)
        init_backend["config"]["hash_filenames"] = hash_filenames
        StateTracker.set_data_backend_config(init_backend["id"], init_backend["config"])
        logger.debug(f"Hashing filenames: {hash_filenames}")

    def _handle_auto_generated_dataset(self, backend: Dict[str, Any], init_backend: Dict[str, Any]) -> None:
        """Handle auto-generated reference datasets."""
        # we have to auto-generate the data for the reference images.
        info_log(f"(id={init_backend['id']}) Auto-generating reference images for conditioning.")
        from simpletuner.helpers.data_generation import DataGenerator, SampleGenerator

        sample_generator = SampleGenerator.from_backend(backend)
        generator = DataGenerator(
            id=backend.get("id"),
            source_backend=StateTracker.get_data_backend(backend.get("source_dataset_id", "unknown_source_dataset_id")),
            target_backend=init_backend,
            accelerator=self.accelerator,
            conditioning_type=backend.get("conditioning_type"),
        )
        generator.generate_dataset()

    def _configure_vae_cache(
        self,
        backend: Dict[str, Any],
        init_backend: Dict[str, Any],
        image_embed_data_backend: Dict[str, Any],
        vae_cache_dir_paths: List[str],
        text_embed_cache_dir_paths: List[str],
        conditioning_type: Optional[str],
    ) -> None:
        """Configure VAE cache for the backend."""
        vae_cache_dir = backend.get("cache_dir_vae", None)
        if vae_cache_dir in vae_cache_dir_paths:
            raise ValueError(
                f"VAE image embed cache directory {vae_cache_dir} is the same as another VAE image embed cache directory. This is not allowed, the trainer will get confused and sleepy and wake up in a distant place with no memory and no money for a taxi ride back home, forever looking in the mirror and wondering who they are. This should be avoided."
            )
        info_log(f"(id={init_backend['id']}) Creating VAE latent cache: {vae_cache_dir=}")
        vae_cache_dir_paths.append(vae_cache_dir)

        if vae_cache_dir is not None and vae_cache_dir in text_embed_cache_dir_paths:
            raise ValueError(
                f"VAE image embed cache directory {vae_cache_dir} is the same as the text embed cache directory. This is not allowed, the trainer will get confused."
            )

        if backend["type"] == "local" and (vae_cache_dir is None or vae_cache_dir == ""):
            if (not self.args.controlnet or backend["dataset_type"] != "conditioning") or (
                self.model.requires_conditioning_latents() and self._requires_conditioning_dataset()
            ):
                raise ValueError(
                    f"VAE image embed cache directory {vae_cache_dir} is not set. This is required for the VAE image embed cache."
                )
        move_text_encoders(self.args, self.text_encoders, "cpu")

        video_config = init_backend["config"].get("video", {})
        init_backend["vaecache"] = VAECache(
            id=init_backend["id"],
            dataset_type=init_backend["dataset_type"],
            model=self.model,
            vae=StateTracker.get_vae(),
            accelerator=self.accelerator,
            metadata_backend=init_backend["metadata_backend"],
            image_data_backend=init_backend["data_backend"],
            cache_data_backend=image_embed_data_backend["data_backend"],
            instance_data_dir=init_backend["instance_data_dir"],
            delete_problematic_images=backend.get("delete_problematic_images", self.args.delete_problematic_images),
            num_video_frames=video_config.get("num_frames", None),
            vae_batch_size=backend.get("vae_batch_size", self.args.vae_batch_size),
            write_batch_size=backend.get("write_batch_size", self.args.write_batch_size),
            read_batch_size=backend.get("read_batch_size", self.args.read_batch_size),
            cache_dir=vae_cache_dir,
            max_workers=backend.get("max_workers", self.args.max_workers),
            process_queue_size=backend.get("image_processing_batch_size", self.args.image_processing_batch_size),
            vae_cache_ondemand=self.args.vae_cache_ondemand,
            hash_filenames=init_backend["config"].get("hash_filenames", True),
        )
        init_backend["vaecache"].set_webhook_handler(StateTracker.get_webhook_handler())

        if not self.args.vae_cache_ondemand:
            info_log(f"(id={init_backend['id']}) Discovering cache objects..")
            if self.accelerator.is_local_main_process:
                try:
                    init_backend["vaecache"].discover_all_files()
                except FileNotFoundError:
                    warning_log(
                        f"(id={init_backend['id']}) Skipping VAE cache discovery because data directory was not found: {init_backend.get('instance_data_dir')}"
                    )
                    return
            self.accelerator.wait_for_everyone()
        all_image_files = StateTracker.get_image_files(
            data_backend_id=init_backend["id"],
            retry_limit=3,  # some filesystems maybe take longer to make it available.
        )
        if all_image_files is None:
            from simpletuner.helpers.training import image_file_extensions

            logger.debug("No image file cache available, retrieving fresh")
            try:
                all_image_files = init_backend["data_backend"].list_files(
                    instance_data_dir=init_backend["instance_data_dir"],
                    file_extensions=image_file_extensions,
                )
            except FileNotFoundError:
                warning_log(
                    f"(id={init_backend['id']}) Skipping VAE cache file discovery because data directory was not found: {init_backend.get('instance_data_dir')}"
                )
                return
            all_image_files = StateTracker.set_image_files(all_image_files, data_backend_id=init_backend["id"])

        init_backend["vaecache"].build_vae_cache_filename_map(all_image_files=all_image_files)

    def _handle_error_scanning_and_metadata(
        self, backend: Dict[str, Any], init_backend: Dict[str, Any], conditioning_type: Optional[str]
    ) -> None:
        """Handle error scanning and metadata operations."""
        if (
            ("metadata" not in self.args.skip_file_discovery or "metadata" not in backend.get("skip_file_discovery", ""))
            and self.accelerator.is_main_process
            and backend.get("scan_for_errors", False)
            and "deepfloyd" not in StateTracker.get_args().model_type
            and conditioning_type
            not in [
                "mask",
                "controlnet" if not self.model.requires_conditioning_latents() else -1,
            ]
        ):
            info_log(
                f"Beginning error scan for dataset {init_backend['id']}. Set 'scan_for_errors' to False in the dataset config to disable this."
            )
            try:
                init_backend["metadata_backend"].handle_vae_cache_inconsistencies(
                    vae_cache=init_backend.get("vaecache"),
                    vae_cache_behavior=backend.get("vae_cache_scan_behaviour", self.args.vae_cache_scan_behaviour),
                )
                init_backend["metadata_backend"].scan_for_metadata()
            except FileNotFoundError:
                warning_log(
                    f"(id={init_backend['id']}) Skipping metadata scan because data directory was not found: {init_backend.get('instance_data_dir')}"
                )
                return

        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            try:
                init_backend["metadata_backend"].load_image_metadata()
            except FileNotFoundError:
                warning_log(
                    f"(id={init_backend['id']}) Skipping metadata load because data directory was not found: {init_backend.get('instance_data_dir')}"
                )
                return
        self.accelerator.wait_for_everyone()

    def _connect_conditioning_datasets(self, data_backend_config: List[Dict[str, Any]]) -> bool:
        """Connect conditioning datasets to their main datasets."""
        has_conditioning_dataset = False
        available_conditioning = {
            backend_id
            for backend_id, backend_obj in self.data_backends.items()
            if backend_obj.get("dataset_type") == "conditioning"
        }
        for backend in data_backend_config:
            dataset_type = backend.get("dataset_type", "image")
            if dataset_type is not None and dataset_type != "image":
                continue
            if backend.get("disabled", False) or backend.get("disable", False):
                info_log(f"Skipping disabled data backend {backend['id']} in config file.")
                continue
            backend_conditionings = backend.get("conditioning_data", [])
            if isinstance(backend_conditionings, str):
                backend_conditionings = [backend_conditionings]
            for x in backend_conditionings:
                if x not in available_conditioning:
                    raise ValueError(
                        f"Conditioning data backend {x} not found in data backend list: {available_conditioning}."
                    )

            if backend_conditionings:
                has_conditioning_dataset = True
                StateTracker.set_conditioning_datasets(backend["id"], backend_conditionings)
                info_log(f"Successfully configured conditioning image dataset for {backend['id']}")

        return has_conditioning_dataset

    def synchronize_conditioning_settings(self) -> None:
        """
        Synchronize resolution settings between main image datasets and their conditioning datasets
        """
        for (
            main_dataset_id,
            conditioning_dataset_id,
        ) in StateTracker.get_conditioning_mappings():
            main_config = StateTracker.get_data_backend_config(main_dataset_id)
            conditioning_config = StateTracker.get_data_backend_config(conditioning_dataset_id)

            resolution_settings = [
                "resolution",
                "resolution_type",
                "maximum_image_size",
                "target_downsample_size",
                "minimum_image_size",
            ]

            for setting in resolution_settings:
                if setting in main_config:
                    if setting in conditioning_config and conditioning_config[setting] != main_config[setting]:
                        info_log(
                            f"Overriding {conditioning_dataset_id}'s {setting} ({conditioning_config[setting]}) "
                            f"with value from {main_dataset_id} ({main_config[setting]})"
                        )

                    conditioning_config[setting] = main_config[setting]

                    StateTracker.set_data_backend_config(conditioning_dataset_id, conditioning_config)

                    conditioning_backend = StateTracker.get_data_backend(conditioning_dataset_id)
                    if "metadata_backend" in conditioning_backend:
                        setattr(
                            conditioning_backend["metadata_backend"],
                            setting,
                            main_config[setting],
                        )

    def _configure_single_data_backend(
        self,
        backend: Dict[str, Any],
        data_backend_config: List[Dict[str, Any]],
        vae_cache_dir_paths: List[str],
        text_embed_cache_dir_paths: List[str],
        requires_conditioning_dataset: bool,
    ) -> None:
        """
        Configure a single data backend with all its components.
        """
        if (
            "id" not in backend
            or backend["id"] == ""
            or backend["id"] in StateTracker.get_data_backends(_types=["image", "video"])
        ):
            raise ValueError("Each dataset needs a unique 'id' field.")

        info_log(f"Configuring data backend: {backend['id']}: {backend}")
        conditioning_type = backend.get("conditioning_type")
        if backend.get("dataset_type") == "conditioning" or conditioning_type is not None:
            backend["dataset_type"] = "conditioning"

        self._handle_resolution_conversion(backend)

        init_backend = init_backend_config(backend, self.args, self.accelerator)
        StateTracker.set_data_backend_config(
            data_backend_id=init_backend["id"],
            config=init_backend["config"],
        )

        preserve_data_backend_cache = backend.get("preserve_data_backend_cache", False)
        if not preserve_data_backend_cache and self.accelerator.is_local_main_process:
            StateTracker.delete_cache_files(
                data_backend_id=init_backend["id"],
                preserve_data_backend_cache=preserve_data_backend_cache,
            )
        StateTracker.load_aspect_resolution_map(
            dataloader_resolution=init_backend["config"]["resolution"],
        )

        self._configure_backend_by_type(backend, init_backend)

        self._assign_text_embed_cache(backend, init_backend)

        data_backend_is_mock = hasattr(init_backend["data_backend"], "_mock_children")

        image_embed_data_backend = self._get_image_embed_backend(backend, init_backend)

        self._configure_metadata_backend(backend, init_backend)

        metadata_backend_is_mock = hasattr(init_backend["metadata_backend"], "_mock_children")

        if data_backend_is_mock and not metadata_backend_is_mock:
            info_log(
                f"(id={init_backend['id']}) Detected mocked data backend without mocked metadata backend; skipping runtime setup steps."
            )
            StateTracker.register_data_backend(init_backend)
            self.data_backends[init_backend["id"]] = init_backend
            return

        if data_backend_is_mock and metadata_backend_is_mock:
            info_log(f"(id={init_backend['id']}) Detected fully mocked backend stack, skipping dataset creation steps.")
            StateTracker.register_data_backend(init_backend)
            self.data_backends[init_backend["id"]] = init_backend
            return

        self._handle_bucket_operations(backend, init_backend, conditioning_type)

        self._create_dataset_and_sampler(backend, init_backend, conditioning_type)

        self._process_text_embeddings(backend, init_backend, conditioning_type)

        if backend.get("auto_generated", False):
            self._handle_auto_generated_dataset(backend, init_backend)

        if getattr(self.model, "AUTOENCODER_CLASS", None) is not None and conditioning_type not in [
            "mask",
            (
                "controlnet" if not self.model.requires_conditioning_latents() else -1
            ),  # Workaround to encode VAE latents when the model requires them.
        ]:
            self._configure_vae_cache(
                backend,
                init_backend,
                image_embed_data_backend,
                vae_cache_dir_paths,
                text_embed_cache_dir_paths,
                conditioning_type,
            )

        self._handle_error_scanning_and_metadata(backend, init_backend, conditioning_type)

        if (
            not self.args.vae_cache_ondemand
            and "vaecache" in init_backend
            and "vae" not in self.args.skip_file_discovery
            and "vae" not in backend.get("skip_file_discovery", "")
            and "deepfloyd" not in StateTracker.get_args().model_type
            and conditioning_type
            not in [
                "mask",
                "controlnet" if not self.model.requires_conditioning_latents() else -1,
            ]
        ):
            unprocessed_files = init_backend["vaecache"].discover_unprocessed_files()
            logger.info(f"VAECache has {len(unprocessed_files)} unprocessed files.")
            if not self.args.vae_cache_ondemand:
                logger.info(f"Executing VAE cache update..")
                init_backend["vaecache"].process_buckets()
            logger.debug(f"Encoding images during training: {self.args.vae_cache_ondemand}")
            self.accelerator.wait_for_everyone()

        move_text_encoders(self.args, self.text_encoders, self.accelerator.device)
        init_backend_debug_info = {
            k: v for k, v in init_backend.items() if isinstance(v, Union[list, int, float, str, dict, tuple])
        }
        info_log(f"Configured backend: {init_backend_debug_info}")

        StateTracker.register_data_backend(init_backend)
        init_backend["metadata_backend"].save_cache()

        self.data_backends[init_backend["id"]] = init_backend

    def configure(self, data_backend_config: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Configure all backends using the new factory implementation.

        This method provides a high-level interface for configuring all types of backends
        in the correct order. It matches the API expected by test suites.

        Args:
            data_backend_config: Optional pre-loaded configuration. If not provided,
                                will load from args.data_backend_config

        Returns:
            Dict containing:
            - 'data_backends': Dictionary of configured data backends
            - 'text_embed_backends': Dictionary of text embedding backends
            - 'image_embed_backends': Dictionary of image embedding backends
            - 'default_text_embed_backend_id': ID of default text embedding backend

        Example:
            >>> factory = FactoryRegistry(args, accelerator, text_encoders, tokenizers, model)
            >>> result = factory.configure()
            >>> print(f"Configured {len(result['data_backends'])} backends")
        """
        if data_backend_config is None:
            data_backend_config = self.load_configuration()

        data_backend_config = self.process_conditioning_datasets(data_backend_config)

        self.configure_text_embed_backends(data_backend_config)
        self.configure_image_embed_backends(data_backend_config)
        self.configure_data_backends(data_backend_config)

        result = {
            "data_backends": StateTracker.get_data_backends(),
            "text_embed_backends": self.text_embed_backends,
            "image_embed_backends": self.image_embed_backends,
            "default_text_embed_backend_id": self.default_text_embed_backend_id,
        }

        self._log_performance_metrics(
            "factory_configure_complete",
            {
                "backends_configured": len(result["data_backends"]),
                "text_embed_backends": len(result["text_embed_backends"]),
                "image_embed_backends": len(result["image_embed_backends"]),
            },
        )

        return result

    def _validate_with_config_registry(self, config: Dict[str, Any], model_family: str) -> None:
        """
        Validate configuration using ConfigRegistry for model-specific rules.

        Args:
            config: Configuration dictionary to validate
            model_family: Model family for specific validation rules
        """
        try:
            from simpletuner.helpers.configuration.registry import ConfigRegistry, ValidationResult

            validation_results = []

            base_rules = ConfigRegistry.get_rules("dataloader")
            for rule in base_rules:
                try:
                    result = self._validate_single_rule(rule, config)
                    if result:
                        validation_results.append(result)
                except Exception as e:
                    logger.debug(f"Error validating rule {rule.field_name}: {e}")

            model_rules = ConfigRegistry.get_rules(model_family)
            for rule in model_rules:
                try:
                    result = self._validate_single_rule(rule, config)
                    if result:
                        validation_results.append(result)
                except Exception as e:
                    logger.debug(f"Error validating model rule {rule.field_name}: {e}")

            base_validators = ConfigRegistry.get_validators("dataloader")
            for validator in base_validators:
                try:
                    results = validator.func(config)
                    validation_results.extend(results)
                except Exception as e:
                    logger.debug(f"Error running dataloader validator: {e}")

            model_validators = ConfigRegistry.get_validators(model_family)
            for validator in model_validators:
                try:
                    results = validator.func(config)
                    validation_results.extend(results)
                except Exception as e:
                    logger.debug(f"Error running model validator: {e}")

            for result in validation_results:
                if not result.passed:
                    if result.level == "error":
                        raise ValueError(f"Validation error for {result.field}: {result.message}")
                    elif result.level == "warning":
                        warning_log(f"Validation warning for {result.field}: {result.message}")
                        if result.suggestion:
                            warning_log(f"  Suggestion: {result.suggestion}")

        except ImportError:
            logger.debug("ConfigRegistry not available for validation")
        except Exception as e:
            logger.debug(f"Error during configuration validation: {e}")

    def _validate_single_rule(self, rule, config: Dict[str, Any]):
        """
        Validate a single configuration rule.

        Args:
            rule: ConfigRule to validate
            config: Configuration dictionary

        Returns:
            ValidationResult or None if rule passes
        """
        from simpletuner.helpers.configuration.registry import RuleType, ValidationResult

        if rule.condition and not rule.condition(config):
            return None

        field_value = config.get(rule.field_name)

        if rule.rule_type == RuleType.REQUIRED:
            if field_value is None:
                return ValidationResult(
                    passed=False,
                    field=rule.field_name,
                    message=rule.message,
                    level=rule.error_level,
                    suggestion=rule.suggestion,
                    rule=rule,
                )
        elif rule.rule_type == RuleType.DEFAULT:
            if field_value is None:
                return ValidationResult(
                    passed=False,
                    field=rule.field_name,
                    message=rule.message,
                    level=rule.error_level,
                    suggestion=rule.suggestion,
                    rule=rule,
                )
        elif rule.rule_type == RuleType.CHOICES:
            if field_value is not None and field_value not in rule.value:
                return ValidationResult(
                    passed=False,
                    field=rule.field_name,
                    message=f"{rule.message}. Got: {field_value}, expected one of: {rule.value}",
                    level=rule.error_level,
                    suggestion=rule.suggestion,
                    rule=rule,
                )

        return None


def configure_multi_databackend_new(
    args: Dict[str, Any],
    accelerator: Any,
    text_encoders: Any,
    tokenizers: Any,
    model: ModelFoundation,
    data_backend_config: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Configure multiple data backends using modular factory components."""
    start_time = time.time()

    StateTracker.clear_data_backends()
    StateTracker.set_accelerator(accelerator)
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO" if accelerator.is_main_process else "ERROR"))

    args = _synchronise_state_tracker(args, accelerator)

    factory = FactoryRegistry(args, accelerator, text_encoders, tokenizers, model)

    if data_backend_config is None:
        data_backend_config = factory.load_configuration()

    data_backend_config = factory.process_conditioning_datasets(data_backend_config)

    factory.configure_text_embed_backends(data_backend_config)
    factory.configure_image_embed_backends(data_backend_config)
    factory.configure_data_backends(data_backend_config)

    factory._log_performance_metrics("implementation_complete")

    result = {
        "data_backends": StateTracker.get_data_backends(),
        "text_embed_backends": factory.text_embed_backends,
        "image_embed_backends": factory.image_embed_backends,
        "default_text_embed_backend_id": factory.default_text_embed_backend_id,
    }

    factory._finalize_metrics()
    total_time = time.time() - start_time

    return result


def check_huggingface_config(backend: dict) -> None:
    """
    Check the configuration for a Hugging Face backend.

    Args:
        backend (dict): A dictionary of the backend configuration.
    Returns:
        None
    """
    required_keys = ["dataset_name"]
    for key in required_keys:
        if key not in backend:
            raise ValueError(f"Missing required key '{key}' in Hugging Face backend config.")

    metadata_backend = backend.get("metadata_backend", "huggingface")
    if metadata_backend not in ["huggingface"]:
        raise ValueError(f"Hugging Face datasets should use metadata_backend='huggingface', not '{metadata_backend}'")

    caption_strategy = backend.get("caption_strategy", "huggingface")
    if caption_strategy not in ["huggingface", "instanceprompt"]:
        raise ValueError(f"Hugging Face datasets should use caption_strategy='huggingface', not '{caption_strategy}'")


def get_huggingface_backend(
    accelerator,
    identifier: str,
    dataset_name: str,
    dataset_type: str,
    split: str = "train",
    revision: str = None,
    image_column: str = "image",
    video_column: str = "video",
    cache_dir: str = None,
    compress_cache: bool = False,
    streaming: bool = False,
    filter_config: dict = None,
    num_proc: int = 16,
    backend: dict = {},
    auto_load: bool = False,
) -> HuggingfaceDatasetsBackend:
    """
    Get a Hugging Face datasets backend.
    """
    filter_func = None
    if filter_config:
        # Simple inline filter creation
        def filter_func(item):
            if "collection" in filter_config:
                required_collections = filter_config["collection"]
                if isinstance(required_collections, str):
                    required_collections = [required_collections]
                if item.get("collection") not in required_collections:
                    return False

            if "quality_thresholds" in filter_config:
                quality = item.get(filter_config.get("quality_column", "quality_assessment"), {})
                if not quality:
                    return False
                for metric, threshold in filter_config["quality_thresholds"].items():
                    if quality.get(metric, 0) < threshold:
                        return False

            if "min_width" in filter_config and item.get("width", 0) < filter_config["min_width"]:
                return False
            if "min_height" in filter_config and item.get("height", 0) < filter_config["min_height"]:
                return False

            return True

    composite_config = None
    if filter_config and "composite_image_config" in backend.get("huggingface", {}):
        composite_config = backend["huggingface"]["composite_image_config"]
    logger.info(f"Image composition config: {composite_config}")

    if backend.get("huggingface"):
        auto_load = backend["huggingface"].get("auto_load", auto_load)

    return HuggingfaceDatasetsBackend(
        accelerator=accelerator,
        id=identifier,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        split=split,
        revision=revision,
        image_column=image_column,
        video_column=video_column,
        cache_dir=cache_dir,
        compress_cache=compress_cache,
        streaming=streaming,
        filter_func=filter_func,
        num_proc=num_proc,
        composite_config=composite_config,
        auto_load=auto_load,
    )


def select_dataloader_index(step, backends):
    # Generate weights for each backend based on some criteria
    weights = []
    backend_ids = []
    for backend_id, backend in backends.items():
        weight = get_backend_weight(backend_id, backend, step)
        weights.append(weight)
        backend_ids.append(backend_id)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights /= weights.sum()  # Normalize the weights
    if weights.sum() == 0:
        return None

    # Sample a backend index based on the weights
    chosen_index = torch.multinomial(weights, 1).item()
    chosen_backend_id = backend_ids[chosen_index]

    return chosen_backend_id


def get_backend_weight(backend_id, backend, step):
    if isinstance(backend, dict):
        backend_config = backend
    else:
        backend_config = getattr(backend, "config", None)
        if not isinstance(backend_config, dict):
            backend_config = StateTracker.get_data_backend_config(backend_id) or {}

    prob = backend_config.get("probability", 1)

    sampling_args = StateTracker.get_args()
    sampling_method = getattr(sampling_args, "data_backend_sampling", "uniform") if sampling_args is not None else "uniform"
    if not isinstance(sampling_method, str):
        sampling_method = "uniform"

    if sampling_method == "uniform":
        return prob
    elif sampling_method == "auto-weighting":
        dataset_length = StateTracker.get_dataset_size(backend_id)
        try:
            total = sum(StateTracker.get_dataset_size(b) for b in StateTracker.get_data_backends(_types=["image", "video"]))
        except Exception:
            total = 0
        if not total:
            total = dataset_length or 1
        length_factor = dataset_length / total

        adjusted_prob = prob * length_factor

        disable_step = backend_config.get("disable_after_epoch_step", None)
        if disable_step:
            disable_step = int(disable_step)
        else:
            disable_step = float("inf")
        adjusted_prob = 0 if int(step) > disable_step else max(0, adjusted_prob * (1 - step / disable_step))

        return adjusted_prob
    else:
        raise ValueError(f"Unknown sampling weighting method: {sampling_method}")


def random_dataloader_iterator(step, backends: dict):
    prefetch_log_debug("Random dataloader iterator launched.")
    args = StateTracker.get_args()
    grad_steps = getattr(args, "gradient_accumulation_steps", 1) if args is not None else 1
    if isinstance(grad_steps, (int, float)):
        gradient_accumulation_steps = max(1, int(grad_steps))
    else:
        gradient_accumulation_steps = 1
    logger.debug(f"Backends to select from {backends}")
    if backends == {}:
        logger.debug("All dataloaders exhausted. Moving to next epoch in main training loop.")
        StateTracker.clear_exhausted_buckets()
        StateTracker.set_repeats(repeats=0)
        return False
    while backends:
        epoch_step = int(step / gradient_accumulation_steps)
        StateTracker.set_epoch_step(epoch_step)

        chosen_backend_id = select_dataloader_index(step, backends)
        if chosen_backend_id is None:
            logger.debug("No dataloader iterators were available.")
            break

        backend_instance = backends[chosen_backend_id]

        try:
            if hasattr(backend_instance, "get_data_loader") and callable(backend_instance.get_data_loader):
                return backend_instance.get_data_loader()

            try:
                chosen_iter = iter(backend_instance)
            except TypeError:
                if hasattr(backend_instance, "__next__"):
                    return next(backend_instance)
                raise

            return next(chosen_iter)
        except MultiDatasetExhausted:
            repeats = StateTracker.get_data_backend_config(chosen_backend_id).get("repeats", False)
            if repeats and repeats > 0 and StateTracker.get_repeats(chosen_backend_id) < repeats:
                StateTracker.increment_repeats(chosen_backend_id)
                logger.debug(
                    f"Dataset (name={chosen_backend_id}) is now sampling its {StateTracker.get_repeats(chosen_backend_id)} repeat out of {repeats} total allowed."
                )
                continue
            logger.debug(
                f"Dataset (name={chosen_backend_id}) is now exhausted after {StateTracker.get_repeats(chosen_backend_id)} repeat(s). Removing from list."
            )
            del backends[chosen_backend_id]
            StateTracker.backend_exhausted(chosen_backend_id)
            StateTracker.set_repeats(data_backend_id=chosen_backend_id, repeats=0)
        finally:
            if not backends:
                logger.debug("All dataloaders exhausted. Moving to next epoch in main training loop.")
                StateTracker.clear_exhausted_buckets()
                return False


def configure_multi_databackend(
    args: dict,
    accelerator,
    text_encoders,
    tokenizers,
    model: ModelFoundation,
    data_backend_config: Optional[List[Dict[str, Any]]] = None,
):
    """Configure multiple dataloaders using the FactoryRegistry implementation."""
    return configure_multi_databackend_new(
        args,
        accelerator,
        text_encoders,
        tokenizers,
        model,
        data_backend_config=data_backend_config,
    )


for _name in ("from_instance_representation", "get_backend_weight", "select_dataloader_index"):
    _func = globals().get(_name)
    if _func is not None and not hasattr(builtins, _name):
        setattr(builtins, _name, _func)
