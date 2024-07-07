import logging, os, time, torch, numpy as np
from tqdm import tqdm
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from helpers.multiaspect.image import MultiaspectImage
from helpers.image_manipulation.brightness import calculate_luminance
from helpers.training.state_tracker import StateTracker
from helpers.training.wrappers import unwrap_model
from helpers.prompts import PromptHandler
from helpers.sdxl.pipeline import StableDiffusionXLPipeline
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
)

if is_wandb_available():
    import wandb

from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline


logger = logging.getLogger("validation")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")


def resize_validation_images(validation_images, edge_length):
    # we have to scale all the inputs to a stage4 image down to 64px smaller edge.
    resized_validation_samples = []
    for _sample in validation_images:
        validation_shortname, validation_prompt, training_sample_image = _sample
        resize_to, crop_to, new_aspect_ratio = (
            MultiaspectImage.calculate_new_size_by_pixel_edge(
                aspect_ratio=MultiaspectImage.calculate_image_aspect_ratio(
                    training_sample_image
                ),
                resolution=edge_length,
                original_size=training_sample_image.size,
            )
        )
        # we can be less precise here
        training_sample_image = training_sample_image.resize(crop_to)
        resized_validation_samples.append(
            (validation_shortname, validation_prompt, training_sample_image)
        )
    return resized_validation_samples


def retrieve_validation_images():
    """
    From each data backend, collect the top 5 images for validation, such that
    we select the same images on each startup, unless the dataset changes.

    Returns:
        dict: A dictionary of shortname to image paths.
    """
    args = StateTracker.get_args()
    data_backends = StateTracker.get_data_backends(
        _type="conditioning" if args.controlnet else "image"
    )
    validation_data_backend_id = args.eval_dataset_id
    validation_set = []
    logger.info("Collecting validation images")
    for _data_backend in data_backends:
        data_backend = StateTracker.get_data_backend(_data_backend)
        data_backend_config = data_backend.get("config", {})
        should_skip_dataset = data_backend_config.get("disable_validation", False)
        logger.debug(f"Backend {_data_backend}: {data_backend}")
        if "id" not in data_backend or (
            args.controlnet and data_backend.get("dataset_type", None) != "conditioning"
        ):
            logger.debug(
                f"Skipping data backend: {_data_backend} dataset_type {data_backend.get('dataset_type', None)}"
            )
            continue
        logger.debug(f"Checking data backend: {data_backend['id']}")
        if (
            validation_data_backend_id is not None
            and data_backend["id"] != validation_data_backend_id
        ) or should_skip_dataset:
            logger.warning(f"Not collecting images from {data_backend['id']}")
            continue
        if "sampler" in data_backend:
            validation_samples_from_sampler = data_backend[
                "sampler"
            ].retrieve_validation_set(batch_size=args.num_eval_images)
            if "stage2" in args.model_type:
                validation_samples_from_sampler = resize_validation_images(
                    validation_samples_from_sampler, edge_length=64
                )

            validation_set.extend(validation_samples_from_sampler)
        else:
            logger.warning(
                f"Data backend {data_backend['id']} does not have a sampler. Skipping."
            )
    return validation_set


def prepare_validation_prompt_list(args, embed_cache):
    validation_negative_prompt_embeds = None
    validation_negative_pooled_embeds = None
    validation_prompts = (
        [""] if not StateTracker.get_args().validation_disable_unconditional else []
    )
    validation_shortnames = (
        ["unconditional"]
        if not StateTracker.get_args().validation_disable_unconditional
        else []
    )
    if not hasattr(embed_cache, "model_type"):
        raise ValueError(
            f"The default text embed cache backend was not found. You must specify 'default: true' on your text embed data backend via {StateTracker.get_args().multidatabackend_config}."
        )
    model_type = embed_cache.model_type
    validation_sample_images = None
    if (
        "deepfloyd-stage2" in args.model_type
        or args.controlnet
        or args.validation_using_datasets
    ):
        # Now, we prepare the DeepFloyd upscaler image inputs so that we can calculate their prompts.
        # If we don't do it here, they won't be available at inference time.
        validation_sample_images = retrieve_validation_images()
        if len(validation_sample_images) > 0:
            StateTracker.set_validation_sample_images(validation_sample_images)
            # Collect the prompts for the validation images.
            for _validation_sample in tqdm(
                validation_sample_images,
                ncols=100,
                desc="Precomputing validation image embeds",
            ):
                _, validation_prompt, _ = _validation_sample
                embed_cache.compute_embeddings_for_prompts(
                    [validation_prompt], load_from_cache=False
                )
            time.sleep(5)

    if args.validation_prompt_library:
        # Use the SimpleTuner prompts library for validation prompts.
        from helpers.prompts import prompts as prompt_library

        # Iterate through the prompts with a progress bar
        for shortname, prompt in tqdm(
            prompt_library.items(),
            leave=False,
            ncols=100,
            desc="Precomputing validation prompt embeddings",
        ):
            embed_cache.compute_embeddings_for_prompts(
                [prompt], is_validation=True, load_from_cache=False
            )
            validation_prompts.append(prompt)
            validation_shortnames.append(shortname)
    if args.user_prompt_library is not None:
        user_prompt_library = PromptHandler.load_user_prompts(args.user_prompt_library)
        for shortname, prompt in tqdm(
            user_prompt_library.items(),
            leave=False,
            ncols=100,
            desc="Precomputing user prompt library embeddings",
        ):
            embed_cache.compute_embeddings_for_prompts(
                [prompt], is_validation=True, load_from_cache=False
            )
            validation_prompts.append(prompt)
            validation_shortnames.append(shortname)
    if args.validation_prompt is not None:
        # Use a single prompt for validation.
        # This will add a single prompt to the prompt library, if in use.
        validation_prompts = validation_prompts + [args.validation_prompt]
        validation_shortnames = validation_shortnames + ["validation"]
        embed_cache.compute_embeddings_for_prompts(
            [args.validation_prompt], is_validation=True, load_from_cache=False
        )

    # Compute negative embed for validation prompts, if any are set.
    if validation_prompts:
        logger.info("Precomputing the negative prompt embed for validations.")
        if model_type == "sdxl" or model_type == "sd3":
            (
                validation_negative_prompt_embeds,
                validation_negative_pooled_embeds,
            ) = embed_cache.compute_embeddings_for_prompts(
                [StateTracker.get_args().validation_negative_prompt],
                is_validation=True,
                load_from_cache=False,
            )
            return (
                validation_prompts,
                validation_shortnames,
                validation_negative_prompt_embeds,
                validation_negative_pooled_embeds,
            )
        elif model_type == "legacy":
            validation_negative_prompt_embeds = (
                embed_cache.compute_embeddings_for_prompts(
                    [StateTracker.get_args().validation_negative_prompt],
                    load_from_cache=False,
                )
            )

            return (
                validation_prompts,
                validation_shortnames,
                validation_negative_prompt_embeds,
            )
        elif model_type == "pixart_sigma" or model_type == "aura_flow":
            # we use the legacy encoder but we return no pooled embeds.
            validation_negative_prompt_embeds = (
                embed_cache.compute_embeddings_for_prompts(
                    [StateTracker.get_args().validation_negative_prompt],
                    load_from_cache=False,
                )
            )

            return (
                validation_prompts,
                validation_shortnames,
                validation_negative_prompt_embeds,
                None,
            )
        else:
            raise ValueError(f"Unknown model type '{model_type}'")


def parse_validation_resolution(input_str: str) -> tuple:
    """
    If the args.validation_resolution:
     - is an int, we'll treat it as height and width square aspect
     - if it has an x in it, we will split and treat as WIDTHxHEIGHT
     - if it has comma, we will split and treat each value as above
    """
    if isinstance(input_str, int) or input_str.isdigit():
        if (
            "deepfloyd-stage2" in StateTracker.get_args().model_type
            and int(input_str) < 256
        ):
            raise ValueError(
                "Cannot use less than 256 resolution for DeepFloyd stage 2."
            )
        return (input_str, input_str)
    if "x" in input_str:
        pieces = input_str.split("x")
        if "deepfloyd-stage2" in StateTracker.get_args().model_type and (
            int(pieces[0]) < 256 or int(pieces[1]) < 256
        ):
            raise ValueError(
                "Cannot use less than 256 resolution for DeepFloyd stage 2."
            )
        return (int(pieces[0]), int(pieces[1]))


def get_validation_resolutions():
    """
    If the args.validation_resolution:
     - is an int, we'll treat it as height and width square aspect
     - if it has an x in it, we will split and treat as WIDTHxHEIGHT
     - if it has comma, we will split and treat each value as above
    """
    validation_resolution_parameter = StateTracker.get_args().validation_resolution
    if (
        type(validation_resolution_parameter) is str
        and "," in validation_resolution_parameter
    ):
        return [
            parse_validation_resolution(res)
            for res in validation_resolution_parameter.split(",")
        ]
    return [parse_validation_resolution(validation_resolution_parameter)]
