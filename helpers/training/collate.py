import torch, logging
from os import environ
from helpers.training.state_tracker import StateTracker
from helpers.image_manipulation.brightness import calculate_batch_luminance

logger = logging.getLogger("Collate")
logger.setLevel(environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

from torchvision.transforms import ToTensor

# Convert PIL Image to PyTorch Tensor
to_tensor = ToTensor()


def compute_time_ids(
    original_size: tuple,
    target_size: tuple,
    weight_dtype,
    vae_downscale_factor: int = 8,
    crop_coordinates: list = None,
):
    if original_size is None or target_size is None:
        raise Exception(
            f"Cannot continue, the original_size or target_size were not provided: {original_size}, {target_size}"
        )
    logger.debug(
        f"Computing time ids for:"
        f"\n-> original_size = {original_size}"
        f"\n-> target_size = {target_size}"
    )
    # The dimensions of tensors are "transposed", as:
    # (batch_size, height, width)
    # An image would look like:
    # (width, height)
    # SDXL conditions are:
    # [h, w, h, w, h, w]
    original_width = original_size[0]
    original_height = original_size[1]
    target_width = int(target_size[2] * vae_downscale_factor)
    target_height = int(target_size[1] * vae_downscale_factor)
    final_target_size = (target_height, target_width)
    if original_width is None:
        raise ValueError("Original width must be specified.")
    if original_height is None:
        raise ValueError("Original height must be specified.")
    if crop_coordinates is None:
        crop_coordinates = (
            StateTracker.get_args().crops_coords_top_left_h,
            StateTracker.get_args().crops_coords_top_left_w,
        )
    add_time_ids = list(
        (original_height, original_width) + tuple(crop_coordinates) + final_target_size
    )
    add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype)
    logger.debug(
        f"compute_time_ids returning {add_time_ids.shape} shaped time ids: {add_time_ids}"
    )
    return add_time_ids


def extract_pixel_values(examples):
    pixel_values = []
    for example in examples:
        image_data = example["image_data"]
        pixel_values.append(
            to_tensor(image_data).to(
                memory_format=torch.contiguous_format,
                dtype=StateTracker.get_vae_dtype(),
            )
        )
    return pixel_values


def extract_filepaths(examples):
    filepaths = []
    for example in examples:
        filepaths.append(example["image_path"])
    return filepaths


def compute_latents(filepaths):
    latents = [StateTracker.get_vaecache().encode_image(None, fp) for fp in filepaths]

    test_shape = latents[0].shape
    idx = 0
    for latent in latents:
        # Move to CPU and pin memory if it's not on the GPU
        latent = latent.to("cpu").pin_memory()
        if latent.shape != test_shape:
            raise ValueError(
                f"File {filepaths[idx]} latent shape mismatch: {latent.shape} != {test_shape}"
            )
        idx += 1
    return torch.stack(latents)


def compute_prompt_embeddings(captions):
    (
        prompt_embeds_all,
        add_text_embeds_all,
    ) = StateTracker.get_embedcache().compute_embeddings_for_sdxl_prompts(captions)
    prompt_embeds_all = torch.concat([prompt_embeds_all for _ in range(1)], dim=0)
    add_text_embeds_all = torch.concat([add_text_embeds_all for _ in range(1)], dim=0)
    return prompt_embeds_all, add_text_embeds_all


def gather_conditional_size_features(examples, latents, weight_dtype):
    batch_time_ids_list = [
        compute_time_ids(
            original_size=tuple(example["original_size"]),
            target_size=latents[idx].shape,
            crop_coordinates=example["crop_coordinates"],
            weight_dtype=weight_dtype,
        )
        for idx, example in enumerate(examples)
    ]
    return torch.stack(batch_time_ids_list, dim=0)


def check_latent_shapes(latents, filepaths):
    reference_shape = latents[0].shape
    for idx, latent in enumerate(latents):
        if latent.shape != reference_shape:
            print(f"Latent shape mismatch for file: {filepaths[idx]}")


def collate_fn(batch):
    if len(batch) != 1:
        raise ValueError(
            "This trainer is not designed to handle multiple batches in a single collate."
        )
    examples = batch[0]
    batch_luminance = [example["luminance"] for example in examples]
    filepaths = extract_filepaths(examples)
    latent_batch = compute_latents(filepaths)
    check_latent_shapes(latent_batch, filepaths)

    # Extract the captions from the examples.
    captions = [example["instance_prompt_text"] for example in examples]
    prompt_embeds_all, add_text_embeds_all = compute_prompt_embeddings(captions)

    batch_time_ids = gather_conditional_size_features(
        examples, latent_batch, StateTracker.get_weight_dtype()
    )
    logger.debug(f"Stacked to {batch_time_ids.shape}: {batch_time_ids}")

    result = {
        "latent_batch": latent_batch,
        "prompt_embeds": prompt_embeds_all,
        "add_text_embeds": add_text_embeds_all,
        "batch_time_ids": batch_time_ids,
    }
    if StateTracker.tracking_luminance():
        result["luminance"] = batch_luminance

    return result
