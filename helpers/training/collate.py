import torch
import logging
import concurrent.futures
import numpy as np
from os import environ
from helpers.training.state_tracker import StateTracker
from helpers.training.multi_process import rank_info, _get_rank
from helpers.image_manipulation.training_sample import TrainingSample
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

logger = logging.getLogger("collate_fn")
logger.setLevel(
    environ.get("SIMPLETUNER_COLLATE_LOG_LEVEL", "INFO")
    if _get_rank() == 0
    else "ERROR"
)
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
        f"Computing time ids for:"
        f"\n-> intermediary_size = {intermediary_size}"
        f"\n-> target_size = {target_size}"
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
        add_time_ids = list(
            (original_height, original_width)
            + tuple(crop_coordinates)
            + (fake_aesthetic_score,)
        )
    else:
        add_time_ids = list(
            (original_height, original_width)
            + tuple(crop_coordinates)
            + final_target_size
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype)
    logger.debug(
        f"compute_time_ids returning {add_time_ids.shape} shaped time ids: {add_time_ids}"
    )
    return add_time_ids


def extract_filepaths(examples):
    filepaths = []
    for example in examples:
        filepaths.append(example["image_path"])
    return filepaths


def fetch_pixel_values(fp, data_backend_id: str, model):
    """Worker method to fetch pixel values for a single image."""
    debug_log(
        f" -> pull pixels for fp {fp} from cache via data backend {data_backend_id}"
    )
    data_backend = StateTracker.get_data_backend(data_backend_id)
    image = data_backend["data_backend"].read_image(fp)
    training_sample = TrainingSample(
        image=image, data_backend_id=data_backend_id, model=model
    )
    return training_sample.prepare(return_tensor=True).image


def fetch_latent(fp, data_backend_id: str):
    """Worker method to fetch latent for a single image."""
    debug_log(
        f" -> pull latents for fp {fp} from cache via data backend {data_backend_id}"
    )
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


def fetch_conditioning_pixel_values(
    fp, training_fp, conditioning_data_backend_id: str, training_data_backend_id: str
):
    """Worker method to fetch pixel values for a single image."""
    # Retrieve data backends
    conditioning_data_backend = StateTracker.get_data_backend(
        conditioning_data_backend_id
    )
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

    # Prepare the conditioning sample to match the training sample
    prepared_like = conditioning_sample.prepare_like(
        training_sample, return_tensor=True
    ).image

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
            latents = StateTracker.get_vaecache(id=data_backend_id).encode_images(
                [None] * len(filepaths), filepaths
            )
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                latents = list(
                    executor.map(
                        fetch_latent, filepaths, [data_backend_id] * len(filepaths)
                    )
                )
    except Exception as e:
        logger.error(f"(id={data_backend_id}) Error while computing latents: {e}")
        raise

    return latents


def compute_single_embedding(caption, text_embed_cache):
    """Worker function to compute embedding for a single caption."""
    if caption == "" or not caption:
        # Grab the default text embed backend for null caption.
        text_embed_cache = StateTracker.get_default_text_embed_cache()
        debug_log(
            f"Hashing caption '{caption}' on text embed cache: {text_embed_cache.id} using data backend {text_embed_cache.data_backend.id}"
        )
    text_encoder_output = text_embed_cache.compute_prompt_embeddings_with_model(
        prompts=[caption]
    )
    logger.debug(f"Keys: {text_encoder_output.keys()}")
    for key, val in text_encoder_output.items():
        if isinstance(val, torch.Tensor):
            logger.debug(f"{key} shape: {val.shape}")
        else:
            logger.debug(f"Value type: {type(val)}")
    return text_encoder_output


def compute_prompt_embeddings(captions, text_embed_cache, model):
    """
    Retrieve / compute text embeds in parallel.
    Args:
        captions: List of strings
        text_embed_cache: TextEmbedCache instance

    Returns:
        prompt_embeds_all: Tensor of shape (batch_size, 512)
        add_text_embeds_all: Tensor of shape (batch_size, 512)
    """
    debug_log(" -> get embed from cache")
    # Use a thread pool to compute embeddings concurrently
    with ThreadPoolExecutor() as executor:
        text_encoder_output = list(
            executor.map(
                compute_single_embedding,
                captions,
                [text_embed_cache] * len(captions),
            )
        )
    prompt_embeds, pooled_prompt_embeds, attn_masks, time_ids = [], [], [], []
    # Is there a better way to do this?
    transformed_encoder_output = model.collate_prompt_embeds(text_encoder_output)
    if transformed_encoder_output == {}:
        if "prompt_embeds" in text_encoder_output[0]:
            transformed_encoder_output["prompt_embeds"] = torch.stack(
                [t["prompt_embeds"] for t in text_encoder_output]
            )
        if "pooled_prompt_embeds" in text_encoder_output[0]:
            transformed_encoder_output["pooled_prompt_embeds"] = torch.stack(
                [t["pooled_prompt_embeds"] for t in text_encoder_output]
            )
        if "attention_mask" in text_encoder_output[0]:
            transformed_encoder_output["attention_masks"] = torch.stack(
                [t["attention_mask"] for t in text_encoder_output]
            )
        if "time_ids" in text_encoder_output[0]:
            transformed_encoder_output["time_ids"] = torch.stack(
                [t["time_ids"] for t in text_encoder_output]
            )

    if transformed_encoder_output == {}:
        raise Exception(f"Could not compute text encoder output: {text_encoder_output}")

    logger.debug(
        f"Transformed text encoder output: {transformed_encoder_output.keys()}"
    )
    return transformed_encoder_output


def gather_conditional_pixart_size_features(examples, latents, weight_dtype):
    bsz = len(examples)
    # 1/8th scale VAE
    LATENT_COMPRESSION_F = 8
    batch_height = latents.shape[2] * LATENT_COMPRESSION_F
    batch_width = latents.shape[3] * LATENT_COMPRESSION_F
    resolution = torch.tensor([batch_height, batch_width]).repeat(bsz, 1)
    aspect_ratio = torch.tensor([float(batch_height / batch_width)]).repeat(bsz, 1)
    resolution = resolution.to(
        dtype=weight_dtype, device=StateTracker.get_accelerator().device
    )
    aspect_ratio = aspect_ratio.to(
        dtype=weight_dtype, device=StateTracker.get_accelerator().device
    )

    return {"resolution": resolution, "aspect_ratio": aspect_ratio}


def gather_conditional_sdxl_size_features(examples, latents, weight_dtype):
    batch_time_ids_list = []
    if len(examples) != len(latents):
        raise ValueError(
            f"Number of examples ({len(examples)}) and latents ({len(latents)}) must match."
        )

    for idx, example in enumerate(examples):
        # Compute time IDs for all examples
        # - We use the intermediary size as the original size for SDXL.
        # - This is because we first resize to intermediary_size before cropping.
        time_ids = compute_time_ids(
            intermediary_size=tuple(
                example.get("intermediary_size", example.get("original_size"))
            ),
            target_size=latents[idx].shape,
            crop_coordinates=example["crop_coordinates"],
            weight_dtype=weight_dtype,
        )

        # Overwrite with zeros if conditioning is to be dropped
        if example["drop_conditioning"]:
            time_ids = torch.zeros_like(time_ids)

        batch_time_ids_list.append(time_ids)

    return torch.stack(batch_time_ids_list, dim=0)


def check_latent_shapes(latents, filepaths, data_backend_id, batch):
    # Validate shapes
    test_shape = latents[0].shape
    # 5D tensors (B, F, C, H, W) are for LTX Video currently, and we'll just test the C, H, W shape
    if len(test_shape) == 5:
        test_shape = test_shape[1:]
    # Check all "aspect_ratio" values and raise error if any differ, with the two differing values:
    first_aspect_ratio = None
    for example in batch:
        if isinstance(example, dict):
            aspect_ratio = example["aspect_ratio"]
        elif isinstance(example, TrainingSample):
            aspect_ratio = example.aspect_ratio
        if first_aspect_ratio is None:
            first_aspect_ratio = aspect_ratio
        if aspect_ratio != first_aspect_ratio:
            error_msg = f"(id=({data_backend_id}) Aspect ratio mismatch: {aspect_ratio} != {first_aspect_ratio}"
            logger.error(error_msg)
            logger.error(f"Erroneous batch: {batch}")
            raise ValueError(error_msg)
    for idx, latent in enumerate(latents):
        # Are there any inf or nan positions?
        if latent is None:
            logger.debug(f"Error batch: {batch}")
            error_msg = f"(id={data_backend_id}) File {filepaths[idx]} latent is None. Filepath: {filepaths[idx]}, data_backend_id: {data_backend_id}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            # get the data_backend
            data_backend = StateTracker.get_data_backend(data_backend_id)
            # remove the object
            data_backend["vaecache"].cache_data_backend.delete(filepaths[idx])
            raise ValueError(
                f"(id={data_backend_id}) Deleted cache file {filepaths[idx]}: contains NaN or Inf values: {latent}"
            )
        if len(latent.shape) == 5:
            if latent.shape[1:] != test_shape:
                raise ValueError(
                    f"(id={data_backend_id}) File {filepaths[idx]} latent shape mismatch: {latent.shape[1:]} != {test_shape}"
                )
        elif latent.shape != test_shape:
            raise ValueError(
                f"(id={data_backend_id}) File {filepaths[idx]} latent shape mismatch: {latent.shape} != {test_shape}"
            )

    debug_log(f" -> stacking {len(latents)} latents: {latents}")
    return torch.stack(
        [latent.to(StateTracker.get_accelerator().device) for latent in latents], dim=0
    )


def collate_fn(batch):
    if len(batch) != 1:
        raise ValueError(
            "This trainer is not designed to handle multiple batches in a single collate."
        )
    debug_log("Begin collate_fn on batch")

    # SDXL Dropout
    dropout_probability = StateTracker.get_args().caption_dropout_probability
    batch = batch[0]
    examples = batch["training_samples"]
    conditioning_examples = batch["conditioning_samples"]
    is_regularisation_data = batch.get("is_regularisation_data", False)
    is_i2v_data = batch.get("is_i2v_data", False)
    if StateTracker.get_args().controlnet and len(examples) != len(
        conditioning_examples
    ):
        raise ValueError(
            "Number of training samples and conditioning samples must match for ControlNet."
            f"\n-> Training samples: {examples}"
            f"\n-> Conditioning samples: {conditioning_examples}"
        )

    # Randomly drop captions/conditioning based on dropout_probability
    data_backend_id = None
    for example in examples:
        data_backend_id = example["data_backend_id"]
        if (
            dropout_probability is not None
            and dropout_probability > 0
            and np.random.rand() < dropout_probability
        ):
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
    debug_log("Compute latents")
    model = StateTracker.get_model()
    batch_data = compute_latents(filepaths, data_backend_id, model)
    if isinstance(batch_data[0], dict):
        latent_batch = [v["latents"] for v in batch_data]
    else:
        latent_batch = batch_data
    if "deepfloyd" not in StateTracker.get_args().model_family:
        debug_log("Check latents")
        latent_batch = check_latent_shapes(
            latent_batch, filepaths, data_backend_id, examples
        )

    training_filepaths = []
    conditioning_type = None
    conditioning_pixel_values = None
    conditioning_latents = None

    if len(conditioning_examples) > 0:
        # check the # of conditioning backends
        logger.debug(f"Found {len(conditioning_examples)} conditioning examples.")

        # get multiple backend ids
        data_backend = StateTracker.get_data_backend(data_backend_id)
        conditioning_backends = data_backend.get("conditioning_data", [])

        if len(conditioning_examples) != len(examples) * len(conditioning_backends):
            raise ValueError(
                "The number of conditioning examples must be divisible by the number of training samples."
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
        if model.requires_conditioning_dataset():
            if model.requires_conditioning_latents():
                conditioning_latents = []
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
                        f"Conditioning latents computed: {len(_latents)} items."
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
                debug_log("Model may require conditioning pixels.")
                conditioning_pixel_values = []
                for _backend_id, _examples in conditioning_map.items():
                    _filepaths = [cond_example.image_path(basename_only=False) for cond_example in _examples]
                    _pixel_values = conditioning_pixels(
                        _filepaths,
                        training_filepaths,
                        _backend_id,
                        data_backend_id,
                    )
                    debug_log(
                        f"Found {len(_pixel_values)} conditioning pixel values."
                    )
                    # stack up that pixel values list
                    conditioning_pixel_values.append(torch.stack(
                        [
                            pixels.to(StateTracker.get_accelerator().device)
                            for pixels in _pixel_values
                        ]
                    )
                    )

    # Compute embeddings and handle dropped conditionings
    debug_log("Extract captions")
    captions = [example["instance_prompt_text"] for example in examples]
    debug_log("Pull cached text embeds")
    text_embed_cache = StateTracker.get_data_backend(data_backend_id)[
        "text_embed_cache"
    ]

    if not text_embed_cache.disabled:
        all_text_encoder_outputs = compute_prompt_embeddings(
            captions, text_embed_cache, StateTracker.get_model()
        )
    else:
        all_text_encoder_outputs = {}
    # TODO: Remove model-specific logic from collate.
    if StateTracker.get_model_family() in ["sdxl", "kolors"]:
        debug_log("Compute and stack SDXL time ids")
        all_text_encoder_outputs["batch_time_ids"] = (
            gather_conditional_sdxl_size_features(
                examples, latent_batch, StateTracker.get_weight_dtype()
            )
        )
        debug_log(
            f"Time ids stacked to {all_text_encoder_outputs['batch_time_ids'].shape}: {all_text_encoder_outputs['batch_time_ids']}"
        )
    elif StateTracker.get_model_family() == "pixart_sigma":
        debug_log("Compute and stack PixArt time ids")
        all_text_encoder_outputs["batch_time_ids"] = (
            gather_conditional_pixart_size_features(
                examples, latent_batch, StateTracker.get_weight_dtype()
            )
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
        "encoder_attention_mask": all_text_encoder_outputs.get("attention_masks"),
        "is_regularisation_data": is_regularisation_data,
        "is_i2v_data": is_i2v_data,
        "conditioning_type": conditioning_type,
    }
