import torch
import logging
import concurrent.futures
import numpy as np
from os import environ
from helpers.training.state_tracker import StateTracker
from helpers.training.multi_process import rank_info
from helpers.image_manipulation.training_sample import TrainingSample
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("collate_fn")
logger.setLevel(environ.get("SIMPLETUNER_COLLATE_LOG_LEVEL", "INFO"))
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


def fetch_pixel_values(fp, data_backend_id: str):
    """Worker method to fetch pixel values for a single image."""
    debug_log(
        f" -> pull pixels for fp {fp} from cache via data backend {data_backend_id}"
    )
    data_backend = StateTracker.get_data_backend(data_backend_id)
    image = data_backend["data_backend"].read_image(fp)
    training_sample = TrainingSample(
        image=image,
        data_backend_id=data_backend_id,
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
        latent = latent.to("cpu").pin_memory()
    return latent


def deepfloyd_pixels(filepaths, data_backend_id: str):
    """DeepFloyd doesn't use the VAE. We retrieve, normalise, and stack the pixel tensors directly."""
    # Use a thread pool to fetch latents concurrently
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            pixels = list(
                executor.map(
                    fetch_pixel_values, filepaths, [data_backend_id] * len(filepaths)
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


def compute_latents(filepaths, data_backend_id: str):
    # Use a thread pool to fetch latents concurrently
    try:
        if "deepfloyd" in StateTracker.get_args().model_type:
            latents = deepfloyd_pixels(filepaths, data_backend_id)

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


def compute_single_embedding(
    caption, text_embed_cache, is_sdxl, is_sd3: bool = False, is_flux: bool = False
):
    """Worker function to compute embedding for a single caption."""
    if caption == "" or not caption:
        # Grab the default text embed backend for null caption.
        text_embed_cache = StateTracker.get_default_text_embed_cache()
        debug_log(
            f"Hashing caption '{caption}' on text embed cache: {text_embed_cache.id} using data backend {text_embed_cache.data_backend.id}"
        )
    if is_sdxl:
        (
            prompt_embeds,
            pooled_prompt_embeds,
        ) = text_embed_cache.compute_embeddings_for_sdxl_prompts([caption])
        return (
            prompt_embeds[0],
            pooled_prompt_embeds[0],
        )  # Unpack the first (and only) element
    elif is_sd3:
        prompt_embeds, pooled_prompt_embeds = (
            text_embed_cache.compute_embeddings_for_sd3_prompts(prompts=[caption])
        )
        return prompt_embeds[0], pooled_prompt_embeds[0]
    elif is_flux:
        prompt_embeds, pooled_prompt_embeds, time_ids, masks = (
            text_embed_cache.compute_embeddings_for_flux_prompts(prompts=[caption])
        )
        return (
            prompt_embeds[0],
            pooled_prompt_embeds[0],
            time_ids[0],
            masks[0] if masks is not None else None,
        )
    else:
        prompt_embeds = text_embed_cache.compute_embeddings_for_legacy_prompts(
            [caption]
        )
        if type(prompt_embeds) == tuple:
            if StateTracker.get_model_family() in ["pixart_sigma", "smoldit"]:
                # PixArt requires the attn mask be returned, too.
                prompt_embeds, attn_mask = prompt_embeds

                return prompt_embeds, attn_mask
            elif "deepfloyd" in StateTracker.get_args().model_type:
                # DeepFloyd doesn't use the attn mask on the unet inputs, we discard it
                prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[0]
        result = torch.squeeze(prompt_embeds[0])
        debug_log(f"Torch shape: {result}")
        return result, None  # Unpack and return None for the second element


def compute_prompt_embeddings(captions, text_embed_cache):
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
    is_sdxl = (
        text_embed_cache.model_type == "sdxl" or text_embed_cache.model_type == "kolors"
    )
    is_sd3 = text_embed_cache.model_type == "sd3"
    is_pixart_sigma = text_embed_cache.model_type == "pixart_sigma"
    is_smoldit = text_embed_cache.model_type == "smoldit"
    is_flux = text_embed_cache.model_type == "flux"

    # Use a thread pool to compute embeddings concurrently
    with ThreadPoolExecutor() as executor:
        embeddings = list(
            executor.map(
                compute_single_embedding,
                captions,
                [text_embed_cache] * len(captions),
                [is_sdxl] * len(captions),
                [is_sd3] * len(captions),
                [is_flux] * len(captions),
            )
        )

    debug_log(f"Got embeddings: {embeddings}")
    if is_sdxl:
        # Separate the tuples
        prompt_embeds = [t[0] for t in embeddings]
        add_text_embeds = [t[1] for t in embeddings]
        return (torch.stack(prompt_embeds), torch.stack(add_text_embeds))
    elif is_sd3:
        # Separate the tuples
        prompt_embeds = [t[0] for t in embeddings]
        add_text_embeds = [t[1] for t in embeddings]
        return (torch.stack(prompt_embeds), torch.stack(add_text_embeds))
    elif is_pixart_sigma or is_smoldit:
        # the tuples here are the text encoder hidden states and the attention masks
        prompt_embeds, attn_masks = [], []
        for embed in embeddings:
            prompt_embeds.append(embed[0][0])
            attn_masks.append(embed[1][0])
        if len(prompt_embeds[0].shape) == 3:
            # some tensors are already expanded due to the way they were saved
            prompt_embeds = [t.squeeze(0) for t in prompt_embeds]
        return (torch.stack(prompt_embeds), torch.stack(attn_masks))
    elif is_flux:
        # Separate the tuples
        prompt_embeds = [t[0] for t in embeddings]
        add_text_embeds = [t[1] for t in embeddings]
        time_ids = [t[2] for t in embeddings]
        masks = [t[3] for t in embeddings]
        return (
            torch.stack(prompt_embeds),
            torch.stack(add_text_embeds),
            torch.stack(time_ids),
            torch.stack(masks) if None not in masks else None,
        )
    else:
        # Separate the tuples
        prompt_embeds = [t[0] for t in embeddings]
        return (torch.stack(prompt_embeds), None)


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
    # Check all "aspect_ratio" values and raise error if any differ, with the two differing values:
    for example in batch:
        if example["aspect_ratio"] != batch[0]["aspect_ratio"]:
            error_msg = f"(id=({data_backend_id}) Aspect ratio mismatch: {example['aspect_ratio']} != {batch[0][0]['aspect_ratio']}"
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
        if latent.shape != test_shape:
            raise ValueError(
                f"(id={data_backend_id}) File {filepaths[idx]} latent shape mismatch: {latent.shape} != {test_shape}"
            )

    debug_log(f" -> stacking {len(latents)} latents")
    return torch.stack(
        [latent.to(StateTracker.get_accelerator().device) for latent in latents]
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
    if StateTracker.get_args().controlnet and len(examples) != len(
        conditioning_examples
    ):
        raise ValueError(
            "Number of training samples and conditioning samples must match for ControlNet."
            f"\n-> Training samples: {examples}"
            f"\n-> Conditioning samples: {conditioning_examples}"
        )

    # Randomly drop captions/conditioning based on dropout_probability
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
    latent_batch = compute_latents(filepaths, data_backend_id)
    if "deepfloyd" not in StateTracker.get_args().model_type:
        debug_log("Check latents")
        latent_batch = check_latent_shapes(
            latent_batch, filepaths, data_backend_id, examples
        )

    conditioning_filepaths = []
    training_filepaths = []
    conditioning_type = None
    conditioning_pixel_values = None

    if len(conditioning_examples) > 0:
        if len(conditioning_examples) != len(examples):
            raise ValueError(
                "The number of conditioning examples must match the number of training examples."
            )

        data_backend = StateTracker.get_data_backend(data_backend_id)
        conditioning_data_backend_id = data_backend.get("conditioning_data", {}).get(
            "id"
        )

        for cond_example, train_example in zip(conditioning_examples, examples):
            # Ensure conditioning types match
            cond_type = cond_example.get_conditioning_type()
            if conditioning_type is None:
                conditioning_type = cond_type
            elif cond_type != conditioning_type:
                raise ValueError(
                    f"Conditioning type mismatch: {conditioning_type} != {cond_type}"
                    "\n-> Ensure all conditioning samples are of the same type."
                )

            # Collect conditioning and training file paths
            conditioning_filepaths.append(cond_example.image_path(basename_only=False))
            training_filepaths.append(train_example["image_path"])

        # Pass both file paths to `conditioning_pixels`
        conditioning_pixel_values = conditioning_pixels(
            conditioning_filepaths,
            training_filepaths,
            conditioning_data_backend_id,
            data_backend_id,
        )

        conditioning_pixel_values = torch.stack(
            [
                latent.to(StateTracker.get_accelerator().device)
                for latent in conditioning_pixel_values
            ]
        )

    # Compute embeddings and handle dropped conditionings
    debug_log("Extract captions")
    captions = [example["instance_prompt_text"] for example in examples]
    debug_log("Pull cached text embeds")
    text_embed_cache = StateTracker.get_data_backend(data_backend_id)[
        "text_embed_cache"
    ]

    attn_mask = None
    batch_time_ids = None
    if StateTracker.get_model_family() == "flux":
        debug_log("Compute and stack Flux time ids")
        prompt_embeds_all, add_text_embeds_all, batch_time_ids, attn_mask = (
            compute_prompt_embeddings(captions, text_embed_cache)
        )
    else:
        prompt_embeds_all, add_text_embeds_all = compute_prompt_embeddings(
            captions, text_embed_cache
        )

    if (
        StateTracker.get_model_family() == "sdxl"
        or StateTracker.get_model_family() == "kolors"
    ):
        debug_log("Compute and stack SDXL time ids")
        batch_time_ids = gather_conditional_sdxl_size_features(
            examples, latent_batch, StateTracker.get_weight_dtype()
        )
        debug_log(f"Time ids stacked to {batch_time_ids.shape}: {batch_time_ids}")
    elif StateTracker.get_model_family() == "pixart_sigma":
        debug_log("Compute and stack PixArt time ids")
        batch_time_ids = gather_conditional_pixart_size_features(
            examples, latent_batch, StateTracker.get_weight_dtype()
        )
        attn_mask = add_text_embeds_all
    elif StateTracker.get_model_family() == "smoldit":
        attn_mask = add_text_embeds_all

    return {
        "latent_batch": latent_batch,
        "prompt_embeds": prompt_embeds_all,
        "add_text_embeds": add_text_embeds_all,
        "batch_time_ids": batch_time_ids,
        "batch_luminance": batch_luminance,
        "conditioning_pixel_values": conditioning_pixel_values,
        "encoder_attention_mask": attn_mask,
        "is_regularisation_data": is_regularisation_data,
        "conditioning_type": conditioning_type,
    }
