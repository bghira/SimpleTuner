import concurrent.futures
import logging
import os
from collections import Counter, defaultdict
from os import environ

import numpy as np
import torch
from PIL import Image

from simpletuner.helpers.data_backend.dataset_runtime import text_embed_request
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.models.common import TextEmbedCacheKey
from simpletuner.helpers.training.multi_process import _get_rank, rank_info
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.utils.pathing import normalize_data_path

logger = logging.getLogger("collate_fn")
logger.setLevel(environ.get("SIMPLETUNER_COLLATE_LOG_LEVEL", "INFO") if _get_rank() == 0 else "ERROR")
rank_text = rank_info()
from torchvision.transforms import ToTensor

# Convert PIL Image to PyTorch Tensor
to_tensor = ToTensor()

# Conditioning type categories for routing
REFERENCE_CONDITIONING_TYPES = {"reference_strict", "reference_loose"}
MASK_CONDITIONING_TYPES = {"mask", "segmentation"}


def _flow_dpo_requires_reference_latents() -> bool:
    method = StateTracker.get_distillation_method()
    if method is None:
        method = getattr(StateTracker.get_args(), "distillation_method", None)
    method = getattr(method, "value", method)
    return isinstance(method, str) and method.lower() == "flow_dpo"


def _distillation_training_batch_requirements() -> set[str]:
    method = getattr(StateTracker.get_args(), "distillation_method", None)
    if method is None:
        method = StateTracker.get_distillation_method()
    method = getattr(method, "value", method)
    if method in (None, "", "none"):
        return set()

    from simpletuner.helpers.distillation.factory import DistillerFactory

    return DistillerFactory.training_batch_requirements(
        method,
        {"distillation_config": getattr(StateTracker.get_args(), "distillation_config", {})},
    )


def debug_log(msg: str):
    logger.debug(f"{rank_text}{msg}")


def compute_time_ids(
    intermediary_size: tuple,
    target_size: tuple,
    weight_dtype,
    vae_downscale_factor: int = 8,
    crop_coordinates: list = None,
):
    if intermediary_size is None or target_size is None:
        raise Exception(
            f"Cannot continue, the intermediary_size or target_size were not provided: {intermediary_size}, {target_size}"
        )
    logger.debug(
        f"Computing time ids for:" f"\n-> intermediary_size = {intermediary_size}" f"\n-> target_size = {target_size}"
    )
    # The dimensions of tensors are "transposed", as:
    # (batch_size, height, width)
    # An image would look like:
    # (width, height)
    # SDXL conditions are:
    # [h, w, h, w, h, w]
    original_width = intermediary_size[0]
    original_height = intermediary_size[1]
    target_width = int(target_size[2] * vae_downscale_factor)
    target_height = int(target_size[1] * vae_downscale_factor)
    final_target_size = (target_height, target_width)
    if original_width is None:
        raise ValueError("Original width must be specified.")
    if original_height is None:
        raise ValueError("Original height must be specified.")
    if crop_coordinates is None:
        raise ValueError("Crop coordinates were not collected during collate.")
    if StateTracker.is_sdxl_refiner():
        fake_aesthetic_score = StateTracker.get_args().data_aesthetic_score
        add_time_ids = list((original_height, original_width) + tuple(crop_coordinates) + (fake_aesthetic_score,))
    else:
        add_time_ids = list((original_height, original_width) + tuple(crop_coordinates) + final_target_size)

    add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype)
    logger.debug(f"compute_time_ids returning {add_time_ids.shape} shaped time ids: {add_time_ids}")
    return add_time_ids


def extract_filepaths(examples):
    filepaths = []
    for example in examples:
        filepaths.append(example["image_path"])
    return filepaths


def describe_missing_conditioning_pairs(
    examples,
    conditioning_examples,
    conditioning_backends,
    training_backend_id,
    training_root=None,
):
    if not examples or not conditioning_backends or not training_backend_id:
        return []
    if any(example.get("data_backend_id") != training_backend_id for example in examples):
        return ["Unable to list missing pairs because multiple training data backends are present in the batch."]

    expected_counter = Counter()
    for example in examples:
        identifier = normalize_data_path(example.get("image_path"), training_root)
        if identifier is not None:
            expected_counter[identifier] += 1
    if not expected_counter:
        return []

    actual_counts = defaultdict(Counter)
    resolution_errors = []
    for cond_example in conditioning_examples:
        backend_id = getattr(cond_example, "_source_dataset_id", getattr(cond_example, "data_backend_id", None))
        if backend_id is None:
            continue
        identifier = None
        if hasattr(cond_example, "training_sample_path"):
            try:
                identifier = normalize_data_path(
                    cond_example.training_sample_path(training_backend_id),
                    training_root,
                )
            except Exception as exc:
                resolution_errors.append(
                    f"{backend_id}: failed to resolve training pair for "
                    f"{getattr(cond_example, '_image_path', 'unknown')}: {exc}"
                )
        if identifier is not None:
            actual_counts[backend_id][identifier] += 1

    messages = []
    for backend_cfg in conditioning_backends:
        backend_id = backend_cfg.get("id")
        backend_actual = actual_counts.get(backend_id, Counter())
        missing_paths = []
        for identifier, expected_count in expected_counter.items():
            actual_count = backend_actual.get(identifier, 0)
            if actual_count < expected_count:
                missing_paths.extend([identifier] * (expected_count - actual_count))
        if missing_paths:
            preview = ", ".join(missing_paths[:3])
            if len(missing_paths) > 3:
                preview += ", ..."
            messages.append(f"{backend_id} missing {len(missing_paths)} pair(s): {preview}")

    messages.extend(resolution_errors)
    return messages


def fetch_latent(fp, data_backend_id: str, vae_cache, view=None):
    """Worker method to fetch latent for a single image."""
    debug_log(f" -> pull latents for fp {fp} from cache via data backend {data_backend_id}")
    latent = vae_cache.retrieve_from_cache(fp, view=view)

    # Move to CPU and pin memory if it's not on the GPU
    if not torch.backends.mps.is_available():
        debug_log(" -> push latents to GPU via pinned memory")
        if isinstance(latent, dict):
            latent["latents"] = latent["latents"].to("cpu").pin_memory()
        else:
            latent = latent.to("cpu").pin_memory()
    return latent


def deepfloyd_pixels(filepaths, data_backend_id: str, model, *, dataset=None, view=None):
    """DeepFloyd doesn't use the VAE. We retrieve, normalise, and stack the pixel tensors directly."""
    if dataset is None:
        dataset = StateTracker.get_dataset(data_backend_id)
    pixels = []
    for record in dataset.read_samples(filepaths, fields=("image", "metadata"), view=view):
        image = record.pop("image")
        sample = TrainingSample(
            image=image,
            image_path=record["image_path"],
            image_metadata=record,
            data_backend_id=data_backend_id,
            data_backend_config=dataset.config,
            model=model,
        )
        pixels.append(sample.prepare(return_tensor=True).image)
    return torch.stack(pixels).to(memory_format=torch.contiguous_format).float()


_REQUIRED_GEOMETRY_KEYS = (
    "original_size",
    "target_size",
    "intermediary_size",
    "crop_coordinates",
    "aspect_ratio",
)


def _build_training_sample_for_conditioning(training_fp, training_example, training_data_backend_id: str) -> TrainingSample:
    """Prefer metadata from the collated training example when preparing conditioning samples."""
    metadata = None
    if isinstance(training_example, dict):
        metadata = training_example.copy()
    elif isinstance(training_example, TrainingSample):
        metadata = training_example.image_metadata.copy() if training_example.image_metadata else None
        if metadata is None:
            return training_example
    elif hasattr(training_example, "image_metadata"):
        metadata = getattr(training_example, "image_metadata")
        if isinstance(metadata, dict):
            metadata = metadata.copy()

    if metadata and all(metadata.get(key) is not None for key in _REQUIRED_GEOMETRY_KEYS):
        metadata.setdefault("image_path", training_fp)
        return TrainingSample(
            image=None,
            data_backend_id=training_data_backend_id,
            image_metadata=metadata,
            image_path=metadata.get("image_path", training_fp),
        )

    if not training_fp:
        raise ValueError("Training sample path was not provided for conditioning alignment.")

    return TrainingSample.from_image_path(
        image_path=training_fp,
        data_backend_id=training_data_backend_id,
    )


def fetch_conditioning_pixel_values(
    fp,
    training_fp,
    training_example,
    conditioning_data_backend_id: str,
    training_data_backend_id: str,
):
    """Worker method to fetch pixel values for a single image."""
    # Prefer metadata captured during the original training sample preparation
    training_sample = _build_training_sample_for_conditioning(training_fp, training_example, training_data_backend_id)

    conditioning_sample = TrainingSample.from_image_path(
        image_path=fp,
        data_backend_id=conditioning_data_backend_id,
    )

    cond_image = conditioning_sample.image
    if isinstance(cond_image, np.ndarray) and cond_image.ndim >= 4:
        conditioning_sample.image = cond_image[0]
    elif isinstance(cond_image, list) and len(cond_image) > 0:
        conditioning_sample.image = cond_image[0]

    if isinstance(conditioning_sample.image, np.ndarray):
        frame = conditioning_sample.image
        if frame.ndim == 3:
            conditioning_sample.image = Image.fromarray(frame.astype(np.uint8))
        elif frame.ndim > 3:
            conditioning_sample.image = Image.fromarray(frame[0].astype(np.uint8))

    if conditioning_sample.model is not None and getattr(conditioning_sample.model, "_is_i2v_like_flavour", lambda: False)():
        conditioning_sample.transforms = conditioning_sample.model.get_transforms(dataset_type="image")

    prepared_like = conditioning_sample.prepare_like(training_sample, return_tensor=True).image

    return prepared_like


def conditioning_pixels(
    filepaths,
    training_filepaths,
    training_examples,
    conditioning_data_backend_id: str,
    training_data_backend_id: str,
):
    """For pixel-based conditioning images that must be prepared matching a paired image's metadata.."""
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            pixels = list(
                executor.map(
                    fetch_conditioning_pixel_values,
                    filepaths,
                    training_filepaths,
                    training_examples,
                    [conditioning_data_backend_id] * len(filepaths),
                    [training_data_backend_id] * len(filepaths),
                )
            )
    except Exception as e:
        logger.error(
            f"(conditioning_data_backend_id={conditioning_data_backend_id}) Error while retrieving or transforming pixels (training data id={training_data_backend_id}): {e}"
        )
        raise
    pixels = torch.stack(pixels)
    pixels = pixels.to(memory_format=torch.contiguous_format).float()

    return pixels


def compute_latents(filepaths, data_backend_id: str, model, *, vae_cache=None, dataset=None, view=None):
    # Use a thread pool to fetch latents concurrently
    try:
        if "deepfloyd" in StateTracker.get_args().model_family:
            latents = deepfloyd_pixels(filepaths, data_backend_id, model, dataset=dataset, view=view)

            return latents
        if vae_cache is None:
            vae_cache = StateTracker.get_dataset(data_backend_id).vae_cache
        if getattr(vae_cache, "vae_cache_ondemand", False):
            latents = vae_cache.encode_images([None] * len(filepaths), filepaths, view=view)
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                latents = list(
                    executor.map(
                        fetch_latent,
                        filepaths,
                        [data_backend_id] * len(filepaths),
                        [vae_cache] * len(filepaths),
                        [view] * len(filepaths),
                    )
                )
    except Exception as e:
        logger.error(f"(id={data_backend_id}) Error while computing latents: {e}")
        raise

    return latents


def _training_text_request(prompt_entry, text_embed_cache, model=None):
    """Resolve the store and key for an occurrence, including caption dropout."""
    if not isinstance(prompt_entry, dict):
        prompt_entry = {"prompt": prompt_entry, "key": prompt_entry, "metadata": {}}
    prompt_value = prompt_entry.get("prompt")
    if prompt_value == "" or not prompt_value:
        # Grab the default text embed backend for null caption.
        text_embed_cache = StateTracker.get_default_text_embed_cache()
        # Use sentinel key for filename-based caches to match encode_dropout_caption()
        cache_model = model if model is not None else getattr(text_embed_cache, "model", None)
        use_dropout_sentinel = getattr(cache_model, "use_text_cache_dropout_sentinel", lambda: True)()
        if text_embed_cache._requires_path_based_keys and use_dropout_sentinel:
            prompt_entry = {**prompt_entry, "key": "__caption_dropout__"}
        debug_log(
            f"Hashing caption '{prompt_value}' on text embed cache: {text_embed_cache.id} using data backend {text_embed_cache.data_backend.id}"
        )
    return text_embed_cache, prompt_entry


def compute_prompt_embeddings(prompt_entries, text_embed_cache, model):
    """Read and collate one embedding per occurrence through the bound stores."""
    requests = [_training_text_request(entry, text_embed_cache, model) for entry in prompt_entries]
    text_encoder_output = [None] * len(requests)
    positions_by_store = defaultdict(list)
    for index, (store, _) in enumerate(requests):
        positions_by_store[store].append(index)
    # Keep precomputed training batches on the host as before. Mixed batches
    # need a common device as the encoder may return accelerator tensors.
    device = next((store.accelerator.device for store in positions_by_store if store.text_cache_ondemand), None)
    for store, positions in positions_by_store.items():
        outputs = store.read_training_batch([requests[index][1] for index in positions], device=device)
        for index, output in zip(positions, outputs, strict=True):
            text_encoder_output[index] = output

    def _collate_tensors(tensors):
        """
        Intelligently collate a list of tensors, handling both 2D and 3D cases.

        - If tensors are 2D [seq, dim], stack to get [batch, seq, dim]
        - If tensors are 3D [1, seq, dim], concatenate along dim=0 to get [batch, seq, dim]
        - If tensors have inconsistent dimensions, normalize them first
        """
        tensors = [tensor for tensor in tensors if tensor is not None]
        if not tensors:
            return None

        first_tensor = tensors[0]
        dims = first_tensor.dim()

        # Check if all tensors have the same number of dimensions
        all_same_dims = all(t.dim() == dims for t in tensors)

        if dims == 2:
            # 2D tensors: [seq, dim] - use stack
            return torch.stack(tensors)
        elif dims == 3 and all_same_dims:
            # 3D tensors: [batch, seq, dim] - use cat along batch dimension
            # This handles cached embeddings that already include batch dimension
            return torch.cat(tensors, dim=0)
        elif dims == 1:
            # 1D tensors - use stack
            return torch.stack(tensors)
        else:
            # Mixed dimensions - normalize to 3D then concatenate
            normalized = []
            for t in tensors:
                if t.dim() == 2:
                    # Add batch dimension
                    normalized.append(t.unsqueeze(0))
                elif t.dim() == 3:
                    normalized.append(t)
                elif t.dim() == 1:
                    # Add batch dimension
                    normalized.append(t.unsqueeze(0))
                else:
                    raise ValueError(f"Unexpected tensor dimension: {t.dim()} with shape {t.shape}")
            return torch.cat(normalized, dim=0)

    # Is there a better way to do this?
    transformed_encoder_output = model.collate_prompt_embeds(text_encoder_output)
    if transformed_encoder_output == {}:
        if "prompt_embeds" in text_encoder_output[0]:
            transformed_encoder_output["prompt_embeds"] = _collate_tensors([t["prompt_embeds"] for t in text_encoder_output])
        if "pooled_prompt_embeds" in text_encoder_output[0]:
            transformed_encoder_output["pooled_prompt_embeds"] = _collate_tensors(
                [t["pooled_prompt_embeds"] for t in text_encoder_output]
            )
        # compatibility for old style
        if "attention_mask" in text_encoder_output[0]:
            transformed_encoder_output["attention_masks"] = _collate_tensors(
                [t["attention_mask"] for t in text_encoder_output]
            )
        if "prompt_attention_mask" in text_encoder_output[0]:
            transformed_encoder_output["attention_masks"] = _collate_tensors(
                [t["prompt_attention_mask"] for t in text_encoder_output]
            )
        # new style
        if "attention_masks" in text_encoder_output[0]:
            transformed_encoder_output["attention_masks"] = _collate_tensors(
                [t["attention_masks"] for t in text_encoder_output]
            )

        if "time_ids" in text_encoder_output[0]:
            transformed_encoder_output["time_ids"] = _collate_tensors([t["time_ids"] for t in text_encoder_output])

    if transformed_encoder_output == {}:
        raise Exception(f"Could not compute text encoder output: {text_encoder_output}")

    logger.debug(f"Transformed text encoder output: {transformed_encoder_output.keys()}")
    return transformed_encoder_output


def gather_conditional_pixart_size_features(examples, latents, weight_dtype):
    bsz = len(examples)
    # 1/8th scale VAE
    LATENT_COMPRESSION_F = 8
    batch_height = latents.shape[2] * LATENT_COMPRESSION_F
    batch_width = latents.shape[3] * LATENT_COMPRESSION_F
    resolution = torch.tensor([batch_height, batch_width]).repeat(bsz, 1)
    aspect_ratio = torch.tensor([float(batch_height / batch_width)]).repeat(bsz, 1)
    resolution = resolution.to(dtype=weight_dtype, device=StateTracker.get_accelerator().device)
    aspect_ratio = aspect_ratio.to(dtype=weight_dtype, device=StateTracker.get_accelerator().device)

    return {"resolution": resolution, "aspect_ratio": aspect_ratio}


def gather_conditional_sdxl_size_features(examples, latents, weight_dtype):
    batch_time_ids_list = []
    if len(examples) != len(latents):
        raise ValueError(f"Number of examples ({len(examples)}) and latents ({len(latents)}) must match.")

    for idx, example in enumerate(examples):
        # Compute time IDs for all examples
        # - We use the intermediary size as the original size for SDXL.
        # - This is because we first resize to intermediary_size before cropping.
        time_ids = compute_time_ids(
            intermediary_size=tuple(example.get("intermediary_size", example.get("original_size"))),
            target_size=latents[idx].shape,
            crop_coordinates=example["crop_coordinates"],
            weight_dtype=weight_dtype,
        )

        # Overwrite with zeros if conditioning is to be dropped
        if example["drop_conditioning"]:
            time_ids = torch.zeros_like(time_ids)

        batch_time_ids_list.append(time_ids)

    return torch.stack(batch_time_ids_list, dim=0)


def check_latent_shapes(latents, filepaths, data_backend_id, batch, is_conditioning=False, *, vae_cache=None, view=None):
    # Validate shapes
    test_shape = latents[0].shape
    # 5D tensors (B, F, C, H, W) are for LTX Video currently, and we'll just test the C, H, W shape
    if len(test_shape) == 5:
        test_shape = test_shape[1:]

    # For conditioning latents with multiple backends, we might have different aspect ratios
    # Only enforce same aspect ratio for training latents
    if not is_conditioning:
        # Check all "aspect_ratio" values and raise error if any differ
        first_aspect_ratio = None
        for example in batch:
            aspect_ratio = None
            if isinstance(example, dict):
                aspect_ratio = example.get("aspect_ratio")
            elif isinstance(example, TrainingSample):
                if hasattr(example, "aspect_ratio"):
                    aspect_ratio = example.aspect_ratio
            if first_aspect_ratio is None and aspect_ratio is not None:
                first_aspect_ratio = aspect_ratio
            if aspect_ratio is not None and first_aspect_ratio is not None and aspect_ratio != first_aspect_ratio:
                error_msg = f"(id=({data_backend_id}) Aspect ratio mismatch: {aspect_ratio} != {first_aspect_ratio}"
                logger.error(error_msg)
                logger.error(f"Erroneous batch: {batch}")
                raise ValueError(error_msg)

    # Rest of the validation remains the same
    for idx, latent in enumerate(latents):
        if latent is None:
            logger.debug(f"Error batch: {batch}")
            error_msg = f"(id={data_backend_id}) File {filepaths[idx]} latent is None."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            cache = StateTracker.get_dataset(data_backend_id).vae_cache if vae_cache is None else vae_cache
            cache_filepath, _ = cache.generate_vae_cache_filename(filepaths[idx], view=view)
            cache.cache_data_backend.delete(cache_filepath)
            raise ValueError(f"(id={data_backend_id}) Deleted cache file {cache_filepath}: contains NaN or Inf values")

        # For conditioning latents, allow different shapes
        if not is_conditioning:
            if len(latent.shape) == 5:
                if latent.shape[1:] != test_shape:
                    raise ValueError(
                        f"(id={data_backend_id}) File {filepaths[idx]} latent shape mismatch: {latent.shape[1:]} != {test_shape}"
                    )
            elif latent.shape != test_shape:
                raise ValueError(
                    f"(id={data_backend_id}) File {filepaths[idx]} latent shape mismatch: {latent.shape} != {test_shape}"
                )

    # Don't stack if shapes differ (for conditioning with multiple aspect ratios)
    if is_conditioning and len(set(_latent.shape for _latent in latents)) > 1:
        # Return list of tensors instead of stacked tensor
        return [_latent.to(StateTracker.get_accelerator().device) for _latent in latents]
    else:
        # Stack normally if all shapes match
        return torch.stack(
            [_latent.to(StateTracker.get_accelerator().device) for _latent in latents],
            dim=0,
        )


def collate_fn(batch, *, dataset=None):
    if len(batch) != 1:
        raise ValueError("This trainer is not designed to handle multiple batches in a single collate.")
    debug_log("Begin collate_fn on batch")

    # SDXL Dropout
    dropout_probability = StateTracker.get_args().caption_dropout_probability
    batch = batch[0]
    examples = batch["training_samples"]
    continuation = examples[0].get("dataset_continuation") if examples else None
    dropout_seed = examples[0].get("caption_dropout_seed") if examples else None
    random_dropout = np.random.default_rng(dropout_seed).random if dropout_seed is not None else np.random.rand
    conditioning_examples = batch["conditioning_samples"]
    has_conditioning_captions = False
    if any([sample.caption is not None for sample in conditioning_examples]):
        # we can use the conditioning captions instead of the principle captions, since they're available.
        has_conditioning_captions = True
    is_regularisation_data = batch.get("is_regularisation_data", False)
    is_i2v_data = batch.get("is_i2v_data", False)
    if StateTracker.get_args().controlnet and len(examples) != len(conditioning_examples):
        raise ValueError(
            "Number of training samples and conditioning samples must match for ControlNet."
            f"\n-> Training samples: {examples}"
            f"\n-> Conditioning samples: {conditioning_examples}"
        )

    # Randomly drop captions/conditioning based on dropout_probability
    data_backend_id = None
    for example in examples:
        data_backend_id = example["data_backend_id"]
        if dropout_probability is not None and dropout_probability > 0 and random_dropout() < dropout_probability:
            example["instance_prompt_text"] = ""  # Drop caption
            example["drop_conditioning"] = True  # Flag to drop conditioning
        else:
            example["drop_conditioning"] = False

    assert isinstance(data_backend_id, str)
    batch_backend_id = data_backend_id
    debug_log("Extract filepaths")
    filepaths = extract_filepaths(examples)
    if dataset is None:
        dataset = StateTracker.get_dataset(batch_backend_id)
    source_config = dataset.config
    dataset_type = source_config["dataset_type"]
    training_data_root = source_config.get("instance_data_dir")
    references = {item.id: item for item in dataset.conditioning_datasets}
    conditioning_ids = tuple(references)

    is_audio_only = dataset_type == "audio" and bool(source_config.get("audio", {}).get("audio_only", False))

    model = StateTracker.get_model()
    uses_audio_tokens = False
    if model is not None:
        try:
            audio_token_flag = model.uses_audio_tokens()
            if isinstance(audio_token_flag, (bool, np.bool_)):
                uses_audio_tokens = bool(audio_token_flag)
            else:
                uses_audio_tokens = False
        except AttributeError:
            uses_audio_tokens = False
    if uses_audio_tokens:
        token_payload = model.collate_audio_tokens(examples)
        if not isinstance(token_payload, dict):
            raise ValueError("collate_audio_tokens must return a dictionary payload.")
        token_payload.setdefault("prompts", None)
        token_payload["is_regularisation_data"] = is_regularisation_data
        token_payload["is_i2v_data"] = is_i2v_data
        return token_payload

    uses_text_embeddings_cache = True
    if model is not None:
        try:
            uses_text_embeddings_cache = bool(model.uses_text_embeddings_cache())
        except AttributeError:
            uses_text_embeddings_cache = True

    debug_log("Compute latents")
    view = examples[0].get("dataset_read_view")
    batch_data = compute_latents(filepaths, batch_backend_id, model, vae_cache=dataset.vae_cache, dataset=dataset, view=view)
    latent_metadata = None
    if isinstance(batch_data[0], dict):
        latent_metadata = []
        for idx, entry in enumerate(batch_data):
            metadata_entry = {k: v for k, v in entry.items() if k != "latents"}
            metadata_entry.setdefault("filepath", filepaths[idx])
            metadata_entry.setdefault("data_backend_id", examples[idx]["data_backend_id"])
            latent_metadata.append(metadata_entry)
        latent_batch = [v["latents"] for v in batch_data]
    else:
        latent_batch = batch_data
        # Fallback: collect metadata so audio models (e.g. ACE-Step) can build attention masks from lengths.
        if latent_metadata is None and StateTracker.get_args().model_family == "ace_step":
            latent_metadata = []
            for path, metadata in zip(filepaths, dataset.metadata_backend.get_metadata_batch(filepaths), strict=True):
                record = dict(metadata or {})
                record.setdefault("filepath", path)
                record.setdefault("data_backend_id", batch_backend_id)
                latent_metadata.append(record)
    if "deepfloyd" not in StateTracker.get_args().model_family:
        debug_log("Check latents")
        latent_batch = check_latent_shapes(
            latent_batch, filepaths, batch_backend_id, examples, vae_cache=dataset.vae_cache, view=view
        )

    conditioning_image_embeds = None
    conditioning_captions = [
        (
            sample.caption
            if getattr(sample, "caption", None)
            else getattr(sample, "image_metadata", {}).get("instance_prompt_text", "")
        )
        for sample in conditioning_examples
    ]
    if model.requires_conditioning_image_embeds():

        def _prepare_embed_tensor(embed_tensor):
            if isinstance(embed_tensor, dict):
                processed_entry = {}
                for key, value in embed_tensor.items():
                    if torch.is_tensor(value) and not torch.backends.mps.is_available():
                        processed_entry[key] = value.to("cpu").pin_memory()
                    else:
                        processed_entry[key] = value
                return processed_entry
            if torch.is_tensor(embed_tensor) and not torch.backends.mps.is_available():
                return embed_tensor.to("cpu").pin_memory()
            return embed_tensor

        embed_tensors = []
        use_reference_embeds = bool(
            conditioning_examples and getattr(model, "conditioning_image_embeds_use_reference_dataset", lambda: False)()
        )
        if use_reference_embeds:
            for sample, caption in zip(conditioning_examples, conditioning_captions):
                cache = references[sample.data_backend_id].conditioning_image_embed_cache
                if cache is None:
                    raise ValueError(
                        f"Conditioning dataset {sample.data_backend_id} is missing a conditioning image embed cache."
                    )
                embed_tensor = cache.retrieve_from_cache(sample.image_path(basename_only=False), caption=caption or None)
                embed_tensors.append(_prepare_embed_tensor(embed_tensor))
        else:
            cache = dataset.conditioning_image_embed_cache
            if cache is None:
                raise ValueError("Conditioning image embed cache is required but was not configured.")
            for path in filepaths:
                embed_tensor = cache.retrieve_from_cache(path, caption=None)
                embed_tensors.append(_prepare_embed_tensor(embed_tensor))

        if embed_tensors:
            if isinstance(embed_tensors[0], dict):
                conditioning_image_embeds = embed_tensors
            else:
                conditioning_image_embeds = torch.stack(embed_tensors, dim=0)

    conditioning_pairs_by_backend: dict[str, list[tuple[TrainingSample, str, dict | TrainingSample]]] = defaultdict(list)
    conditioning_types_by_backend: dict[str, str] = {}
    conditioning_type = None
    loss_mask_type = None
    conditioning_pixel_values = None
    conditioning_latents = None
    conditioning_latents_type = None
    reference_pixels = {}

    sampling_mode = getattr(StateTracker.get_args(), "conditioning_multidataset_sampling", "combined")
    is_combined_mode = sampling_mode == "combined"
    is_random_mode = sampling_mode == "random" and len(conditioning_ids) > 1
    if len(conditioning_examples) > 0:
        # check the # of conditioning backends
        logger.debug(f"Found {len(conditioning_examples)} conditioning examples.")

        # In "random" mode with multiple backends, only ONE conditioning sample per training example
        # In "combined" mode, one sample from EACH backend per training example
        if is_random_mode:
            expected_conditioning_total = len(examples)
        else:
            expected_conditioning_total = len(examples) * len(conditioning_ids)
        if len(conditioning_examples) != expected_conditioning_total:
            missing_pairs = describe_missing_conditioning_pairs(
                examples,
                conditioning_examples,
                [{"id": key} for key in conditioning_ids],
                data_backend_id,
                training_data_root,
            )
            detail_suffix = ""
            if missing_pairs:
                preview = "; ".join(missing_pairs[:3])
                if len(missing_pairs) > 3:
                    preview += "; ..."
                detail_suffix = f" Missing pairs: {preview}"
            raise ValueError(
                "Each conditioning backend must supply one sample per training example "
                f"(expected {expected_conditioning_total}, got {len(conditioning_examples)})."
                f"{detail_suffix}"
            )

        normalized_training_examples: dict[str, list[dict | TrainingSample]] = defaultdict(list)
        for training_example in examples:
            training_image_path = (
                training_example.get("image_path")
                if isinstance(training_example, dict)
                else getattr(training_example, "image_path", None)
            )
            identifier = normalize_data_path(training_image_path, training_data_root)
            if identifier:
                normalized_training_examples[identifier].append(training_example)

        occurrence_positions = Counter()

        def _next_training_example_for_path(match_path: str, backend_id: str):
            if not match_path:
                return None
            candidates = [match_path]
            stripped = match_path.lstrip("/\\")
            if stripped and stripped != match_path:
                candidates.append(stripped)
            if training_data_root and stripped:
                joined = os.path.join(training_data_root, stripped)
                candidates.append(joined)
            seen_keys = set()
            for candidate in candidates:
                normalized_candidate = normalize_data_path(candidate, training_data_root)
                if not normalized_candidate or normalized_candidate in seen_keys:
                    continue
                seen_keys.add(normalized_candidate)
                entries = normalized_training_examples.get(normalized_candidate, ())
                key = (None if is_random_mode else backend_id, normalized_candidate)
                position = occurrence_positions[key]
                if position < len(entries):
                    occurrence_positions[key] += 1
                    return entries[position]
            return None

        training_resolution_errors: list[str] = []
        for cond_example in conditioning_examples:
            backend_id = getattr(cond_example, "_source_dataset_id", getattr(cond_example, "data_backend_id", None))
            if backend_id is None or backend_id not in conditioning_ids:
                logger.debug(f"Skipping conditioning sample because backend {backend_id} is not registered for this batch.")
                continue

            cond_type = cond_example.get_conditioning_type()
            conditioning_types_by_backend[backend_id] = cond_type

            try:
                training_pair_path = cond_example.training_sample_path(training_dataset_id=data_backend_id)
            except Exception as exc:  # pragma: no cover - defensive guard
                training_resolution_errors.append(
                    f"{backend_id}: unable to resolve training pair for {cond_example.image_path(basename_only=False)} ({exc})"
                )
                continue

            paired_training_example = _next_training_example_for_path(training_pair_path, backend_id)
            if paired_training_example is None:
                training_resolution_errors.append(
                    f"{backend_id}: missing training metadata for {cond_example.image_path(basename_only=False)} (paired path: {training_pair_path})"
                )
                continue

            conditioning_pairs_by_backend[backend_id].append((cond_example, training_pair_path, paired_training_example))

        if training_resolution_errors:
            preview = "; ".join(training_resolution_errors[:3])
            if len(training_resolution_errors) > 3:
                preview += "; ..."
            raise ValueError(f"Failed to resolve conditioning pairs: {preview}")

        backend_pair_counts = {backend_id: len(pairs) for backend_id, pairs in conditioning_pairs_by_backend.items()}
        debug_log(f"Counted conditioning pairs per backend: {backend_pair_counts}")

        # Separate backends by conditioning type category
        reference_backends = [
            b for b in conditioning_ids if conditioning_types_by_backend.get(b) in REFERENCE_CONDITIONING_TYPES
        ]
        mask_backends = [b for b in conditioning_ids if conditioning_types_by_backend.get(b) in MASK_CONDITIONING_TYPES]
        # Backends with None or unrecognized conditioning types are treated as untyped
        # and processed based on model requirements (backwards compatibility)
        untyped_backends = [
            b
            for b in conditioning_ids
            if conditioning_types_by_backend.get(b) not in REFERENCE_CONDITIONING_TYPES
            and conditioning_types_by_backend.get(b) not in MASK_CONDITIONING_TYPES
        ]
        debug_log(
            f"Reference backends: {reference_backends}, "
            f"Mask backends: {mask_backends}, "
            f"Untyped backends: {untyped_backends}"
        )

        # Determine the conditioning types for model input and loss masking
        # Use explicit precedence order when multiple types of same category exist
        ref_types_present = {
            cond_type for cond_type in conditioning_types_by_backend.values() if cond_type in REFERENCE_CONDITIONING_TYPES
        }
        mask_types_present = {
            cond_type for cond_type in conditioning_types_by_backend.values() if cond_type in MASK_CONDITIONING_TYPES
        }

        if len(ref_types_present) > 1:
            logger.warning(
                f"Multiple reference conditioning types in batch: {sorted(ref_types_present)}. "
                "Using first by precedence order."
            )
        if len(mask_types_present) > 1:
            logger.warning(
                f"Multiple loss-mask conditioning types in batch: {sorted(mask_types_present)}. "
                "Using first by precedence order."
            )

        # Select types by precedence (order defined in constants)
        conditioning_type = next((t for t in REFERENCE_CONDITIONING_TYPES if t in ref_types_present), None)
        loss_mask_type = next((t for t in MASK_CONDITIONING_TYPES if t in mask_types_present), None)

        assert model is not None
        has_conditioning_backends = bool(conditioning_types_by_backend) or model.requires_conditioning_dataset()
        if has_conditioning_backends:
            requires_conditioning_latents = (
                model.requires_conditioning_latents() or is_i2v_data or _flow_dpo_requires_reference_latents()
            )
            needs_reference_pixels = getattr(
                model, "requires_text_embed_conditioning_pixels", model.requires_text_embed_image_context
            )()

            # Combine reference and untyped backends for latent/pixel collection
            # Untyped backends are processed like reference backends for backwards compatibility
            latent_source_backends = reference_backends + untyped_backends

            # Collect latents from reference/untyped backends (for model input)
            if latent_source_backends and requires_conditioning_latents:
                conditioning_latents = []
                conditioning_latents_type = []
                debug_log("Compute conditioning latents from reference/untyped backends")
                for backend_id in latent_source_backends:
                    backend_pairs = conditioning_pairs_by_backend.get(backend_id, [])
                    if not backend_pairs:
                        continue
                    _examples = [pair[0] for pair in backend_pairs]
                    _filepaths = [sample.image_path(basename_only=False) for sample in _examples]
                    _latents = compute_latents(
                        _filepaths,
                        backend_id,
                        model,
                    )
                    debug_log(
                        f"Conditioning latents computed: {len(_latents)} items with shapes: {[_latent.shape for _latent in _latents]}"
                    )

                    # unpack from dicts (vae-cache style) & shape-check
                    if isinstance(_latents[0], dict):
                        _latents = [v["latents"] for v in _latents]

                    backend_cond_type = conditioning_types_by_backend.get(backend_id)
                    _latents = check_latent_shapes(
                        _latents,
                        _filepaths,
                        backend_id,
                        _examples,
                        is_conditioning=(backend_cond_type == "reference_loose"),
                    )
                    conditioning_latents.append(_latents)
                    conditioning_latents_type.append(backend_cond_type)

            # Preserve the existing group order while recording exact text context per occurrence.
            if latent_source_backends and needs_reference_pixels:
                conditioning_pixel_values = []
                debug_log("Collect conditioning pixel values from reference/untyped backends for text embed image context")
                for backend_id in latent_source_backends:
                    backend_pairs = conditioning_pairs_by_backend.get(backend_id, [])
                    if not backend_pairs:
                        continue
                    _examples = [pair[0] for pair in backend_pairs]
                    _filepaths = [sample.image_path(basename_only=False) for sample in _examples]
                    paired_training_paths = [pair[1] for pair in backend_pairs]
                    paired_training_examples = [pair[2] for pair in backend_pairs]
                    _pixel_values = conditioning_pixels(
                        _filepaths,
                        paired_training_paths,
                        paired_training_examples,
                        backend_id,
                        data_backend_id,
                    )
                    debug_log(f"Found {len(_pixel_values)} reference pixel values for image context.")
                    conditioning_pixel_values.append(
                        torch.stack([pixels.to(StateTracker.get_accelerator().device) for pixels in _pixel_values])
                    )
                    for paired, pixels in zip(paired_training_examples, conditioning_pixel_values[-1], strict=True):
                        reference_pixels.setdefault(id(paired), pixels)

            # Collect pixels from mask backends (for loss masking)
            if mask_backends:
                if conditioning_pixel_values is None:
                    conditioning_pixel_values = []
                debug_log("Collect conditioning pixel values from mask backends for loss masking")
                for backend_id in mask_backends:
                    backend_pairs = conditioning_pairs_by_backend.get(backend_id, [])
                    if not backend_pairs:
                        continue
                    _examples = [pair[0] for pair in backend_pairs]
                    _filepaths = [sample.image_path(basename_only=False) for sample in _examples]
                    paired_training_paths = [pair[1] for pair in backend_pairs]
                    paired_training_examples = [pair[2] for pair in backend_pairs]
                    _pixel_values = conditioning_pixels(
                        _filepaths,
                        paired_training_paths,
                        paired_training_examples,
                        backend_id,
                        data_backend_id,
                    )
                    debug_log(f"Found {len(_pixel_values)} mask pixel values.")
                    conditioning_pixel_values.append(
                        torch.stack([pixels.to(StateTracker.get_accelerator().device) for pixels in _pixel_values])
                    )

    def _conditioning_pixel_value_for_example(example):
        pixel_tensor = reference_pixels.get(id(example))
        if pixel_tensor is None:
            return None
        if pixel_tensor.dim() == 4 and pixel_tensor.size(0) == 1:
            pixel_tensor = pixel_tensor.squeeze(0)
        if pixel_tensor.dim() != 3:
            return None
        pixel_tensor = pixel_tensor.to(torch.float32)
        tensor_max = pixel_tensor.max().item()
        tensor_min = pixel_tensor.min().item()
        if tensor_max > 1.0 or tensor_min < 0.0:
            # Most datasets store conditioning pixels in [-1, 1]
            if tensor_max <= 1.0 and tensor_min >= -1.0:
                pixel_tensor = (pixel_tensor + 1.0) / 2.0
            else:
                pixel_tensor = pixel_tensor / 255.0
        pixel_tensor = pixel_tensor.clamp(0.0, 1.0)
        return pixel_tensor.detach().to("cpu")

    # Compute embeddings and handle dropped conditionings
    debug_log(f"Extract captions. {is_combined_mode=}, {is_random_mode=}, {has_conditioning_captions=}")

    if has_conditioning_captions and is_random_mode:
        paired_captions = {
            id(example): sample.caption for pairs in conditioning_pairs_by_backend.values() for sample, _, example in pairs
        }
        captions = [
            (paired_captions.get(id(example)) if not example["drop_conditioning"] else None)
            or example["instance_prompt_text"]
            for example in examples
        ]
        debug_log(f"Pull cached text embeds. conditioning captions: {captions}")
    else:
        # Use training captions (default behavior)
        captions = [example["instance_prompt_text"] for example in examples]
        debug_log(f"Pull cached text embeds. Using training set captions: {captions}")
    text_embed_cache = dataset.text_embed_cache if uses_text_embeddings_cache else None
    prompt_requests = []
    unconditional_prompt_requests = []
    load_unconditional_embeddings = "unconditional_text_embeddings" in _distillation_training_batch_requirements()
    key_type = TextEmbedCacheKey.CAPTION
    getter = getattr(model, "text_embed_cache_key", None)
    if callable(getter):
        try:
            key_type = getter()
        except Exception as exc:
            debug_log(f"text_embed_cache_key() lookup failed on model {type(model)}: {exc}")

    if uses_text_embeddings_cache:
        for idx, caption in enumerate(captions):
            example = examples[idx]
            example_path = example.get("image_path")
            example_backend_id = example.get("data_backend_id")
            normalized_identifier = normalize_data_path(example_path, training_data_root)
            metadata_builder = getattr(model, "text_embed_cache_metadata_for_sample", None)

            def _build_prompt_metadata(prompt):
                prompt_metadata = {
                    "image_path": example_path,
                    "data_backend_id": example_backend_id,
                    "prompt": prompt,
                    "dataset_relative_path": normalized_identifier,
                }
                if conditioning_examples:
                    reference_paths, reference_backend_ids = [], []
                    for backend_id in reference_backends + untyped_backends:
                        pairs = conditioning_pairs_by_backend[backend_id]
                        for sample, _, paired_example in pairs:
                            if paired_example is example:
                                reference_paths.append(sample.image_path(basename_only=False))
                                reference_backend_ids.append(backend_id)
                    if reference_paths:
                        prompt_metadata["image_paths"] = reference_paths
                        prompt_metadata["data_backend_ids"] = reference_backend_ids
                if callable(metadata_builder):
                    prompt_metadata.update(
                        metadata_builder(
                            example=example,
                            latent=latent_batch[idx],
                            prompt=prompt,
                            data_backend_id=example_backend_id,
                            dataset_relative_path=normalized_identifier,
                        )
                    )
                return prompt_metadata

            metadata = _build_prompt_metadata(caption)
            # Only include conditioning pixels for text embedding when using a single
            # conditioning image. With multiple backends in combined mode, skip image
            # context in embeddings and rely solely on latent references.
            # (In random mode with multiple backends, only one image is selected, so
            # we can still use it for text embedding context.)
            has_multiple_combined_refs = len(conditioning_ids) > 1 and is_combined_mode
            pixel_value = None
            if (
                not has_multiple_combined_refs
                and getattr(model, "requires_text_embed_conditioning_pixels", model.requires_text_embed_image_context)()
            ):
                pixel_value = _conditioning_pixel_value_for_example(example)
                if pixel_value is not None:
                    metadata["conditioning_pixel_values"] = pixel_value
                    metadata["conditioning_pixel_values_range"] = (0.0, 1.0)
            prompt_requests.append(
                text_embed_request(
                    model, key_type, caption, metadata, example_backend_id, example_path, normalized_identifier
                )
            )
            if load_unconditional_embeddings:
                unconditional_metadata = _build_prompt_metadata("")
                if pixel_value is not None:
                    unconditional_metadata.update(
                        conditioning_pixel_values=pixel_value, conditioning_pixel_values_range=(0.0, 1.0)
                    )
                unconditional_prompt_requests.append(
                    text_embed_request(
                        model, key_type, "", unconditional_metadata, example_backend_id, example_path, normalized_identifier
                    )
                )

    if text_embed_cache is not None and not text_embed_cache.disabled:
        all_text_encoder_outputs = compute_prompt_embeddings(prompt_requests, text_embed_cache, StateTracker.get_model())
        if unconditional_prompt_requests:
            unconditional_text_encoder_outputs = compute_prompt_embeddings(
                unconditional_prompt_requests,
                text_embed_cache,
                StateTracker.get_model(),
            )
        else:
            unconditional_text_encoder_outputs = {}
    else:
        all_text_encoder_outputs = {}
        unconditional_text_encoder_outputs = {}
    # TODO: Remove model-specific logic from collate.
    if StateTracker.get_model_family() in ["sdxl", "kolors"]:
        debug_log("Compute and stack SDXL time ids")
        all_text_encoder_outputs["batch_time_ids"] = gather_conditional_sdxl_size_features(
            examples, latent_batch, StateTracker.get_weight_dtype()
        )
        debug_log(
            f"Time ids stacked to {all_text_encoder_outputs['batch_time_ids'].shape}: {all_text_encoder_outputs['batch_time_ids']}"
        )
    elif StateTracker.get_model_family() == "pixart_sigma":
        debug_log("Compute and stack PixArt time ids")
        all_text_encoder_outputs["batch_time_ids"] = gather_conditional_pixart_size_features(
            examples, latent_batch, StateTracker.get_weight_dtype()
        )

    # Extract S2V audio paths for speech-to-video models
    s2v_audio_paths = []
    s2v_audio_backend_ids = []
    for example in examples:
        audio_path = None
        audio_backend_id = None
        if isinstance(example, dict):
            audio_path = example.get("s2v_audio_path")
            audio_backend_id = example.get("s2v_audio_backend_id")
        elif hasattr(example, "image_metadata") and example.image_metadata:
            audio_path = example.image_metadata.get("s2v_audio_path")
            audio_backend_id = example.image_metadata.get("s2v_audio_backend_id")
        elif hasattr(example, "_s2v_audio_path"):
            audio_path = example._s2v_audio_path
            audio_backend_id = getattr(example, "_s2v_audio_backend_id", None)
        s2v_audio_paths.append(audio_path)
        s2v_audio_backend_ids.append(audio_backend_id)

    def single_audio_source():
        if len(dataset.audio_datasets) == 1:
            audio = dataset.audio_datasets[0]
            return audio.id, audio.config, audio.metadata_backend
        return None, {}, None

    if not any(s2v_audio_paths):

        def resolve_local_source_audio_path(path: str | None) -> str | None:
            if not path:
                return None
            candidates = [path]
            if training_data_root and not os.path.isabs(path):
                candidates.insert(0, os.path.join(training_data_root, path.lstrip(os.sep)))
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
            return None

        if dataset_type == "video":
            audio_backend_id, s2v_config, metadata_backend = single_audio_source()
            if s2v_config:
                audio_config = s2v_config.get("audio", {})
                if audio_config.get("source_from_video", False):
                    if audio_backend_id is not None:
                        if metadata_backend is None:
                            s2v_audio_paths = [resolve_local_source_audio_path(path) for path in filepaths]
                            s2v_audio_backend_ids = [
                                audio_backend_id if path is not None else None for path in s2v_audio_paths
                            ]
                        else:
                            s2v_audio_paths = []
                            s2v_audio_backend_ids = []
                            for path in filepaths:
                                audio_path = resolve_local_source_audio_path(path)
                                if metadata_backend.get_metadata_by_filepath(path) is None or audio_path is None:
                                    s2v_audio_paths.append(None)
                                    s2v_audio_backend_ids.append(None)
                                else:
                                    s2v_audio_paths.append(audio_path)
                                    s2v_audio_backend_ids.append(audio_backend_id)

    audio_latent_batch = None
    audio_latent_mask = None
    uses_audio_latents = False
    if model is not None:
        try:
            uses_audio_latents = bool(model.uses_audio_latents_for_data_backend(batch_backend_id))
        except AttributeError:
            uses_audio_latents = False
    if uses_audio_latents and any(s2v_audio_paths):
        if dataset_type == "video":
            default_audio_backend_id, _, _ = single_audio_source()
            grouped: dict[str, list[int]] = {}
            for idx, path in enumerate(s2v_audio_paths):
                if path is None:
                    continue
                backend_id = s2v_audio_backend_ids[idx] or default_audio_backend_id
                if backend_id is None:
                    raise ValueError(
                        f"Missing s2v audio backend id for {path}. Ensure the audio dataset is linked via s2v_datasets."
                    )
                grouped.setdefault(backend_id, []).append(idx)

            audio_latents_by_index: list[torch.Tensor | None] = [None] * len(s2v_audio_paths)
            for backend_id, indices in grouped.items():
                audio_paths = [s2v_audio_paths[i] for i in indices]
                latents = compute_latents(audio_paths, backend_id, model)
                for offset, latent in zip(indices, latents):
                    if isinstance(latent, dict):
                        latent = latent.get("latents")
                    audio_latents_by_index[offset] = latent

            first_latent = next((latent for latent in audio_latents_by_index if torch.is_tensor(latent)), None)
            if first_latent is not None:
                filled = []
                for latent in audio_latents_by_index:
                    if latent is None:
                        filled.append(torch.zeros_like(first_latent))
                    else:
                        if latent.shape != first_latent.shape:
                            raise ValueError(
                                f"S2V audio latent shape mismatch. Expected {first_latent.shape} but got {latent.shape}."
                            )
                        filled.append(latent)
                audio_latent_batch = torch.stack(filled, dim=0)
                audio_latent_mask = torch.tensor(
                    [1 if path is not None else 0 for path in s2v_audio_paths], dtype=torch.float32
                )

    # Handle audio-only mode: transfer audio latents and set video mask
    video_latent_mask = None
    if is_audio_only:
        # For audio-only datasets, the latent_batch contains audio latents from the audio VAE cache
        # We need to move them to audio_latent_batch and set video latents to None
        if audio_latent_batch is None and latent_batch is not None:
            # The latent_batch contains audio latents - move them
            if isinstance(latent_batch, list):
                audio_latent_batch = torch.stack(latent_batch, dim=0) if len(latent_batch) > 0 else None
            else:
                audio_latent_batch = latent_batch
            audio_latent_mask = torch.ones(len(examples), dtype=torch.float32)
            latent_batch = None  # Signal that video latents need to be built by the model

        # Set video latent mask to zeros (mask out video loss)
        batch_size = len(examples)
        video_latent_mask = torch.zeros(batch_size, dtype=torch.float32)

    # Grounding annotations
    grounding_batch = None
    max_grounding_entities = getattr(StateTracker.get_args(), "max_grounding_entities", None)
    if max_grounding_entities and max_grounding_entities > 0:
        if text_embed_cache is None:
            raise ValueError("Grounding annotations require a text embedding cache.")
        from simpletuner.helpers.training.grounding.collate import GroundingCollate

        grounding_collate = GroundingCollate(
            max_entities=max_grounding_entities,
            vae_scale_factor=model.vae_output_scaling_factor if hasattr(model, "vae_output_scaling_factor") else 8,
        )
        # Detect video batches: latent_batch has 5 dims (B, C, T, H, W)
        grounding_num_frames = 1
        if latent_batch is not None and latent_batch.dim() == 5:
            grounding_num_frames = latent_batch.shape[2]
        grounding_batch = grounding_collate.build_batch(
            examples=examples,
            dataset_root=dataset.metadata_backend.instance_data_dir,
            text_embed_cache=text_embed_cache,
            grounding_image_cache=dataset.grounding_image_embed_cache,
            num_frames=grounding_num_frames,
        )

    return {
        **({"dataset_continuation": continuation} if continuation is not None else {}),
        "latent_batch": latent_batch,
        "latent_metadata": latent_metadata,
        "filepaths": filepaths,
        "data_backend_id": batch_backend_id,
        "prompts": captions,
        "text_conditioning_mask": torch.tensor([bool(caption) for caption in captions], dtype=torch.bool),
        "text_encoder_output": all_text_encoder_outputs,
        "prompt_embeds": all_text_encoder_outputs.get("prompt_embeds"),
        "negative_prompt_embeds": unconditional_text_encoder_outputs.get("prompt_embeds"),
        "negative_text_token_tags": unconditional_text_encoder_outputs.get("text_token_tags"),
        "negative_encoder_attention_mask": unconditional_text_encoder_outputs.get("attention_masks"),
        "negative_add_text_embeds": unconditional_text_encoder_outputs.get("pooled_prompt_embeds"),
        "add_text_embeds": all_text_encoder_outputs.get("pooled_prompt_embeds"),
        "t5xxl_ids": all_text_encoder_outputs.get("t5xxl_ids"),
        "t5xxl_weights": all_text_encoder_outputs.get("t5xxl_weights"),
        "batch_time_ids": all_text_encoder_outputs.get("batch_time_ids"),
        "conditioning_pixel_values": conditioning_pixel_values,
        "conditioning_latents": conditioning_latents,
        "conditioning_latents_type": conditioning_latents_type,
        "conditioning_image_embeds": conditioning_image_embeds,
        "conditioning_captions": conditioning_captions,
        "encoder_attention_mask": all_text_encoder_outputs.get("attention_masks"),
        "is_regularisation_data": is_regularisation_data,
        "is_i2v_data": is_i2v_data,
        "conditioning_type": conditioning_type,
        "loss_mask_type": loss_mask_type,
        "audio_latent_batch": audio_latent_batch,
        "audio_latent_mask": audio_latent_mask,
        "video_latent_mask": video_latent_mask,
        "is_audio_only": is_audio_only,
        "s2v_audio_paths": s2v_audio_paths if any(s2v_audio_paths) else None,
        "s2v_audio_backend_ids": s2v_audio_backend_ids if any(s2v_audio_backend_ids) else None,
        "grounding_batch": grounding_batch,
        "slider_strength": batch.get("slider_strength"),
    }
