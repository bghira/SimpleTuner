import concurrent.futures
import logging
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from os import environ

import numpy as np
import torch
from PIL import Image

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


def fetch_pixel_values(fp, data_backend_id: str, model):
    """Worker method to fetch pixel values for a single image."""
    debug_log(f" -> pull pixels for fp {fp} from cache via data backend {data_backend_id}")
    data_backend = StateTracker.get_data_backend(data_backend_id)
    image = data_backend["data_backend"].read_image(fp)
    training_sample = TrainingSample(image=image, data_backend_id=data_backend_id, model=model)
    return training_sample.prepare(return_tensor=True).image


def fetch_latent(fp, data_backend_id: str):
    """Worker method to fetch latent for a single image."""
    debug_log(f" -> pull latents for fp {fp} from cache via data backend {data_backend_id}")
    latent = StateTracker.get_vaecache(id=data_backend_id).retrieve_from_cache(fp)

    # Move to CPU and pin memory if it's not on the GPU
    if not torch.backends.mps.is_available():
        debug_log(" -> push latents to GPU via pinned memory")
        if isinstance(latent, dict):
            latent["latents"] = latent["latents"].to("cpu").pin_memory()
        else:
            latent = latent.to("cpu").pin_memory()
    return latent


def deepfloyd_pixels(filepaths, data_backend_id: str, model):
    """DeepFloyd doesn't use the VAE. We retrieve, normalise, and stack the pixel tensors directly."""
    # Use a thread pool to fetch latents concurrently
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            pixels = list(
                executor.map(
                    fetch_pixel_values,
                    filepaths,
                    [data_backend_id] * len(filepaths),
                    [model] * len(filepaths),
                )
            )
    except Exception as e:
        logger.error(f"(id={data_backend_id}) Error while computing pixels: {e}")
        raise
    pixels = torch.stack(pixels)
    pixels = pixels.to(memory_format=torch.contiguous_format).float()

    return pixels


def fetch_conditioning_pixel_values(fp, training_fp, conditioning_data_backend_id: str, training_data_backend_id: str):
    """Worker method to fetch pixel values for a single image."""
    # Retrieve data backends
    conditioning_data_backend = StateTracker.get_data_backend(conditioning_data_backend_id)
    training_data_backend = StateTracker.get_data_backend(training_data_backend_id)

    # Use the provided training file path directly
    training_sample = TrainingSample.from_image_path(
        image_path=training_fp,
        data_backend_id=training_data_backend_id,
    )

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


def compute_latents(filepaths, data_backend_id: str, model):
    # Use a thread pool to fetch latents concurrently
    try:
        if "deepfloyd" in StateTracker.get_args().model_family:
            latents = deepfloyd_pixels(filepaths, data_backend_id, model)

            return latents
        if StateTracker.get_args().vae_cache_ondemand:
            latents = StateTracker.get_vaecache(id=data_backend_id).encode_images([None] * len(filepaths), filepaths)
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                latents = list(executor.map(fetch_latent, filepaths, [data_backend_id] * len(filepaths)))
    except Exception as e:
        logger.error(f"(id={data_backend_id}) Error while computing latents: {e}")
        raise

    return latents


def compute_single_embedding(prompt_entry, text_embed_cache):
    """Worker function to compute embedding for a single caption."""
    if not isinstance(prompt_entry, dict):
        prompt_entry = {"prompt": prompt_entry, "key": prompt_entry, "metadata": {}}
    prompt_value = prompt_entry.get("prompt")
    if prompt_value == "" or not prompt_value:
        # Grab the default text embed backend for null caption.
        text_embed_cache = StateTracker.get_default_text_embed_cache()
        debug_log(
            f"Hashing caption '{prompt_value}' on text embed cache: {text_embed_cache.id} using data backend {text_embed_cache.data_backend.id}"
        )
    text_encoder_output = text_embed_cache.compute_prompt_embeddings_with_model(prompt_records=[prompt_entry])
    logger.debug(f"Keys: {text_encoder_output.keys()}")
    for key, val in text_encoder_output.items():
        if isinstance(val, torch.Tensor):
            logger.debug(f"{key} shape: {val.shape}")
        else:
            logger.debug(f"Value type: {type(val)}")
    return text_encoder_output


def compute_prompt_embeddings(prompt_entries, text_embed_cache, model):
    """
    Retrieve / compute text embeds in parallel.
    Args:
        prompt_entries: List of strings or prompt records
        text_embed_cache: TextEmbedCache instance

    Returns:
        prompt_embeds_all: Tensor of shape (batch_size, 512)
        add_text_embeds_all: Tensor of shape (batch_size, 512)
    """
    debug_log(" -> get embed from cache")
    # Use a thread pool to compute embeddings concurrently
    normalized_entries = []
    for entry in prompt_entries:
        if isinstance(entry, dict):
            normalized_entries.append(entry)
        else:
            normalized_entries.append({"prompt": entry, "key": entry, "metadata": {}})
    with ThreadPoolExecutor() as executor:
        text_encoder_output = list(
            executor.map(
                compute_single_embedding,
                normalized_entries,
                [text_embed_cache] * len(normalized_entries),
            )
        )
    prompt_embeds, pooled_prompt_embeds, attn_masks, time_ids = [], [], [], []
    # Is there a better way to do this?
    transformed_encoder_output = model.collate_prompt_embeds(text_encoder_output)
    if transformed_encoder_output == {}:
        if "prompt_embeds" in text_encoder_output[0]:
            transformed_encoder_output["prompt_embeds"] = torch.stack([t["prompt_embeds"] for t in text_encoder_output])
        if "pooled_prompt_embeds" in text_encoder_output[0]:
            transformed_encoder_output["pooled_prompt_embeds"] = torch.stack(
                [t["pooled_prompt_embeds"] for t in text_encoder_output]
            )
        # compatibility for old style
        if "attention_mask" in text_encoder_output[0]:
            transformed_encoder_output["attention_masks"] = torch.stack([t["attention_mask"] for t in text_encoder_output])
        if "prompt_attention_mask" in text_encoder_output[0]:
            transformed_encoder_output["attention_masks"] = torch.stack(
                [t["prompt_attention_mask"] for t in text_encoder_output]
            )
        # new style
        if "attention_masks" in text_encoder_output[0]:
            transformed_encoder_output["attention_masks"] = torch.stack([t["attention_masks"] for t in text_encoder_output])

        if "time_ids" in text_encoder_output[0]:
            transformed_encoder_output["time_ids"] = torch.stack([t["time_ids"] for t in text_encoder_output])

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


def check_latent_shapes(latents, filepaths, data_backend_id, batch, is_conditioning=False):
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
                aspect_ratio = example["aspect_ratio"]
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
            data_backend = StateTracker.get_data_backend(data_backend_id)
            data_backend["vaecache"].cache_data_backend.delete(filepaths[idx])
            raise ValueError(f"(id={data_backend_id}) Deleted cache file {filepaths[idx]}: contains NaN or Inf values")

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


def collate_fn(batch):
    if len(batch) != 1:
        raise ValueError("This trainer is not designed to handle multiple batches in a single collate.")
    debug_log("Begin collate_fn on batch")

    # SDXL Dropout
    dropout_probability = StateTracker.get_args().caption_dropout_probability
    batch = batch[0]
    examples = batch["training_samples"]
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
        if dropout_probability is not None and dropout_probability > 0 and np.random.rand() < dropout_probability:
            example["instance_prompt_text"] = ""  # Drop caption
            example["drop_conditioning"] = True  # Flag to drop conditioning
        else:
            example["drop_conditioning"] = False

    assert isinstance(data_backend_id, str)
    debug_log("Collect luminance values")
    if "luminance" in examples[0]:
        batch_luminance = [example["luminance"] for example in examples]
    else:
        batch_luminance = [0] * len(examples)
    # average it
    batch_luminance = sum(batch_luminance) / len(batch_luminance)
    debug_log("Extract filepaths")
    filepaths = extract_filepaths(examples)
    data_backend = StateTracker.get_data_backend(data_backend_id)
    training_data_root = data_backend.get("config", {}).get("instance_data_dir")

    debug_log("Compute latents")
    model = StateTracker.get_model()
    batch_data = compute_latents(filepaths, data_backend_id, model)
    if isinstance(batch_data[0], dict):
        latent_batch = [v["latents"] for v in batch_data]
    else:
        latent_batch = batch_data
    if "deepfloyd" not in StateTracker.get_args().model_family:
        debug_log("Check latents")
        latent_batch = check_latent_shapes(latent_batch, filepaths, data_backend_id, examples)

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
                cond_backend = StateTracker.get_data_backend(sample.data_backend_id)
                cache = cond_backend.get("conditioning_image_embed_cache")
                if cache is None:
                    raise ValueError(
                        f"Conditioning dataset {sample.data_backend_id} is missing a conditioning image embed cache."
                    )
                embed_tensor = cache.retrieve_from_cache(sample.image_path(basename_only=False), caption=caption or None)
                embed_tensors.append(_prepare_embed_tensor(embed_tensor))
        else:
            cache = data_backend.get("conditioning_image_embed_cache")
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

    training_filepaths = []
    conditioning_type = None
    conditioning_pixel_values = None
    conditioning_latents = None

    # get multiple backend ids
    conditioning_backends = data_backend.get("conditioning_data", [])
    if len(conditioning_examples) > 0:
        # check the # of conditioning backends
        logger.debug(f"Found {len(conditioning_examples)} conditioning examples.")

        expected_conditioning_total = len(examples) * len(conditioning_backends)
        if len(conditioning_examples) != expected_conditioning_total:
            missing_pairs = describe_missing_conditioning_pairs(
                examples,
                conditioning_examples,
                conditioning_backends,
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

        conditioning_map = defaultdict(list)
        for i, cond_example in enumerate(conditioning_examples):
            train_example = examples[i % len(examples)]
            cond_backend = conditioning_backends[i // len(examples)]
            # Ensure conditioning types match
            cond_type = cond_example.get_conditioning_type()
            if conditioning_type is None:
                conditioning_type = cond_type
            elif cond_type != conditioning_type:
                # todo: allow each cond backend to have a different type?
                raise ValueError(
                    f"Conditioning type mismatch: {conditioning_type} != {cond_type}"
                    "\n-> Ensure all conditioning samples are of the same type."
                )

            # Collect conditioning and training file paths
            conditioning_map[cond_backend["id"]].append(cond_example)
            training_filepaths.append(train_example["image_path"])
        debug_log(
            f"Counted {len(conditioning_map)} conditioning filepaths and {len(training_filepaths)} training filepaths."
        )

        assert model is not None
        if conditioning_type is not None or model.requires_conditioning_dataset():
            conditioning_latents = []
            needs_conditioning_pixels = (
                not model.requires_conditioning_latents()
                or getattr(model, "requires_text_embed_image_context", lambda: False)()
            )

            if model.requires_conditioning_latents():
                # Kontext / other latent-conditioned models / adapters
                debug_log("Compute conditioning latents")
                for _backend_id, _examples in conditioning_map.items():
                    _filepaths = [cond_example.image_path(basename_only=False) for cond_example in _examples]
                    _latents = compute_latents(
                        _filepaths,
                        _backend_id,
                        model,
                    )
                    debug_log(
                        f"Conditioning latents computed: {len(_latents)} items with shapes: {[_latent.shape for _latent in _latents]}"
                    )

                    # unpack from dicts (vae-cache style) & shape-check
                    if isinstance(_latents[0], dict):
                        _latents = [v["latents"] for v in _latents]

                    _latents = check_latent_shapes(
                        _latents,
                        _filepaths,
                        _backend_id,
                        _examples,
                    )
                    conditioning_latents.append(_latents)
            else:
                needs_conditioning_pixels = True

            if needs_conditioning_pixels:
                debug_log("Collect conditioning pixel values for prompt encoding.")
                conditioning_pixel_values = []
                for _backend_id, _examples in conditioning_map.items():
                    _filepaths = [cond_example.image_path(basename_only=False) for cond_example in _examples]
                    _pixel_values = conditioning_pixels(
                        _filepaths,
                        training_filepaths,
                        _backend_id,
                        data_backend_id,
                    )
                    debug_log(f"Found {len(_pixel_values)} conditioning pixel values.")
                    conditioning_pixel_values.append(
                        torch.stack([pixels.to(StateTracker.get_accelerator().device) for pixels in _pixel_values])
                    )

    def _conditioning_pixel_value_for_example(example_idx: int):
        if not conditioning_pixel_values:
            return None
        first_backend = conditioning_pixel_values[0]
        if not torch.is_tensor(first_backend):
            return None
        if example_idx >= first_backend.shape[0]:
            return None
        pixel_tensor = first_backend[example_idx]
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
        pixel_tensor = pixel_tensor.clamp_(0.0, 1.0)
        return pixel_tensor.detach().to("cpu")

    # Check if we're in combined mode with multiple conditioning datasets
    sampling_mode = getattr(StateTracker.get_args(), "conditioning_multidataset_sampling")
    is_combined_mode = sampling_mode == "combined"
    is_random_mode = sampling_mode == "random" and len(conditioning_backends) > 1

    # Compute embeddings and handle dropped conditionings
    debug_log(f"Extract captions. {is_combined_mode=}, {is_random_mode=}, {has_conditioning_captions=}")

    if has_conditioning_captions and is_random_mode:
        # Only use conditioning captions in random mode
        captions = [
            example.caption if example.caption else example["instance_prompt_text"] for example in conditioning_examples
        ]
        # If the caption is empty, we use the instance prompt text.
        captions = [caption if caption else example["instance_prompt_text"] for caption, example in zip(captions, examples)]
        debug_log(f"Pull cached text embeds. conditioning captions: {captions}")

        # Get the appropriate text_embed_cache
        if conditioning_backends:
            text_embed_cache = conditioning_backends[0]["text_embed_cache"]
        else:
            text_embed_cache = StateTracker.get_data_backend(data_backend_id)["text_embed_cache"]
    else:
        # Use training captions (default behavior)
        captions = [example["instance_prompt_text"] for example in examples]
        debug_log(f"Pull cached text embeds. Using training set captions: {captions}")
        text_embed_cache = StateTracker.get_data_backend(data_backend_id)["text_embed_cache"]
    prompt_requests = []
    key_type = TextEmbedCacheKey.CAPTION
    getter = getattr(model, "text_embed_cache_key", None)
    if callable(getter):
        try:
            key_type = getter()
        except Exception as exc:
            debug_log(f"text_embed_cache_key() lookup failed on model {type(model)}: {exc}")

    for idx, caption in enumerate(captions):
        example = examples[idx]
        example_path = example.get("image_path")
        data_backend_id = example.get("data_backend_id")
        backend_config = StateTracker.get_data_backend_config(data_backend_id) if data_backend_id else {}
        backend_config = backend_config or {}
        dataset_root = backend_config.get("instance_data_dir")
        normalized_identifier = normalize_data_path(example_path, dataset_root)
        metadata = {
            "image_path": example_path,
            "data_backend_id": data_backend_id,
            "prompt": caption,
            "dataset_relative_path": normalized_identifier,
        }
        pixel_value = _conditioning_pixel_value_for_example(idx)
        if pixel_value is not None:
            metadata["conditioning_pixel_values"] = pixel_value
        if key_type is TextEmbedCacheKey.DATASET_AND_FILENAME and data_backend_id and example_path:
            key_value = f"{data_backend_id}:{normalized_identifier}"
        elif key_type is TextEmbedCacheKey.FILENAME and example_path:
            key_value = normalize_data_path(example_path, None)
        else:
            key_value = caption
        prompt_requests.append({"prompt": caption, "key": key_value, "metadata": metadata})

    if not text_embed_cache.disabled:
        all_text_encoder_outputs = compute_prompt_embeddings(prompt_requests, text_embed_cache, StateTracker.get_model())
    else:
        all_text_encoder_outputs = {}
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

    return {
        "latent_batch": latent_batch,
        "prompts": captions,
        "text_encoder_output": all_text_encoder_outputs,
        "prompt_embeds": all_text_encoder_outputs.get("prompt_embeds"),
        "add_text_embeds": all_text_encoder_outputs.get("pooled_prompt_embeds"),
        "batch_time_ids": all_text_encoder_outputs.get("batch_time_ids"),
        "batch_luminance": batch_luminance,
        "conditioning_pixel_values": conditioning_pixel_values,
        "conditioning_latents": conditioning_latents,
        "conditioning_image_embeds": conditioning_image_embeds,
        "conditioning_captions": conditioning_captions,
        "encoder_attention_mask": all_text_encoder_outputs.get("attention_masks"),
        "is_regularisation_data": is_regularisation_data,
        "is_i2v_data": is_i2v_data,
        "conditioning_type": conditioning_type,
    }
