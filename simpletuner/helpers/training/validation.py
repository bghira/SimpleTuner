import base64
import inspect
import logging
import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Union

import diffusers
import numpy as np
import torch
from tqdm import tqdm

import wandb
from simpletuner.helpers.models.common import ImageModelFoundation, ModelFoundation, VideoModelFoundation
from simpletuner.helpers.training.wrappers import unwrap_model

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils.torch_utils import is_compiled_module
from PIL import Image, ImageDraw, ImageFont
from transformers.utils import ContextManagers

from simpletuner.helpers.image_manipulation.brightness import calculate_luminance
from simpletuner.helpers.models.common import PipelineTypes, PredictionTypes
from simpletuner.helpers.models.hidream.schedule import FlowUniPCMultistepScheduler
from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.training.deepspeed import deepspeed_zero_init_disabled_context_manager, prepare_model_for_deepspeed
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.validation_adapters import (
    ValidationAdapterRun,
    ValidationAdapterSpec,
    build_validation_adapter_runs,
)

logger = logging.getLogger("Validation")
from simpletuner.helpers.training.multi_process import gather_across_processes, should_log, split_across_processes

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


SCHEDULER_NAME_MAP = {
    "euler": EulerDiscreteScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "flow_matching": FlowMatchEulerDiscreteScheduler,
    "unipc": UniPCMultistepScheduler,
    "flow_unipc": FlowUniPCMultistepScheduler,
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "dpm++": DPMSolverMultistepScheduler,
    "sana": FlowMatchEulerDiscreteScheduler,
}

import logging
import os
import time

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import is_wandb_available

from simpletuner.helpers.prompts import PromptHandler

if is_wandb_available():
    import wandb


def resize_validation_images(validation_images, edge_length):
    # we have to scale all the inputs to a stage4 image down to 64px smaller edge.
    resized_validation_samples = []
    for _sample in validation_images:
        if len(_sample) == 4:
            validation_shortname, validation_prompt, _, training_sample_image = _sample
        elif len(_sample) == 3:
            validation_shortname, validation_prompt, training_sample_image = _sample
        resize_to, crop_to, new_aspect_ratio = MultiaspectImage.calculate_new_size_by_pixel_edge(
            aspect_ratio=MultiaspectImage.calculate_image_aspect_ratio(training_sample_image),
            resolution=int(edge_length),
            original_size=training_sample_image.size,
        )
        # we can be less precise here
        training_sample_image = training_sample_image.resize(crop_to)
        resized_validation_samples.append((validation_shortname, validation_prompt, training_sample_image))
    return resized_validation_samples


def reset_eval_datasets():
    eval_datasets = StateTracker.get_data_backends(_type="eval", _types=None)
    for dataset_name, dataset in eval_datasets.items():
        if "train_dataset" not in dataset:
            logger.debug(f"Skipping eval set {dataset_name} because it lacks a dataloader.")
        try:
            dataset["sampler"]._reset_buckets(raise_exhaustion_signal=False)
        except MultiDatasetExhausted as e:
            pass


def retrieve_eval_images(dataset_name=None):
    """
    If `dataset_name` is provided, only fetch samples from that specific dataset.
    Otherwise, we iterate over *all* eval datasets until we find a valid sample.

    Returns:
        A collated batch from the eval dataset(s), or raises MultiDatasetExhausted.
    """
    eval_datasets = StateTracker.get_data_backends(_type="eval", _types=None)
    output = {}
    eval_samples = None
    from simpletuner.helpers.training.collate import collate_fn

    new_sample = None
    # We loop until we successfully retrieve one collated batch or exhaust the data.
    while new_sample is None:
        # Decide which dataset(s) to pull from
        if dataset_name is not None:
            # Only attempt to pull from the requested dataset
            dataset_keys = [dataset_name]
        else:
            # Fallback: iterate over *all* eval datasets
            dataset_keys = list(eval_datasets.keys())

        for ds_name in dataset_keys:
            dataset = eval_datasets.get(ds_name)
            if not dataset or "train_dataset" not in dataset:
                logger.debug(f"Skipping eval set {ds_name} because it lacks a dataloader.")
                continue
            try:
                new_sample = next(dataset["sampler"].__iter__())
                data_loaded = dataset["train_dataset"].__getitem__(new_sample)
                if data_loaded:
                    output = collate_fn([data_loaded])
                # Indicate that we've found a batch and can break out of the loop
                new_sample = False
                break

            except MultiDatasetExhausted as e:
                logger.debug(f"Ran out of evaluation samples for dataset {ds_name}. Resetting buckets.")
                dataset["sampler"]._reset_buckets(raise_exhaustion_signal=False)
                # We re-raise if we've exhausted this dataset. If `dataset_name` is set,
                # we effectively stop; if it's None, we move to the next dataset.
                if dataset_name is not None:
                    raise e

    return output


def _coerce_validation_image_input(image_data):
    """
    Convert validation conditioning inputs into formats compatible with downstream pipelines.
    """
    if isinstance(image_data, (list, tuple)):
        coerced = [_coerce_validation_image_input(item) for item in image_data]
        return coerced if isinstance(image_data, list) else tuple(coerced)

    if torch.is_tensor(image_data):
        tensor = image_data.detach().cpu()
        if tensor.ndim >= 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        image_data = tensor.numpy()

    if isinstance(image_data, np.ndarray):
        if image_data.ndim == 4:
            if image_data.shape[0] == 0:
                raise ValueError("Validation conditioning video contains no frames.")
            frame = image_data[0]
        elif image_data.ndim == 3:
            frame = image_data
        else:
            raise ValueError(f"Unsupported validation image array shape: {image_data.shape}")
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0) * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return Image.fromarray(frame)

    return image_data


def _normalise_validation_sample(sample):
    """
    Ensure validation samples carry Image inputs (or lists/tuples of Images) instead of raw tensors/arrays.
    """
    if isinstance(sample, tuple):
        if len(sample) == 4:
            shortname, prompt, image_path, image_data = sample
            return shortname, prompt, image_path, _coerce_validation_image_input(image_data)
        if len(sample) == 3:
            shortname, prompt, image_data = sample
            return shortname, prompt, _coerce_validation_image_input(image_data)
    return sample


def _assert_eval_dataset_exists(eval_dataset_config, available_backends: dict[str, dict], context: str) -> set[str]:
    """
    Make sure that every dataset requested via --eval_dataset_id exists among the loaded validation backends.

    Args:
        eval_dataset_config: The raw value provided via --eval_dataset_id (string, iterable, or None).
        available_backends: Mapping of backend_id -> backend metadata that represents selectable datasets.
        context: Short description used to clarify the error message (e.g. "image validation").

    Raises:
        ValueError: If one or more requested dataset IDs are missing.
    """
    if eval_dataset_config is None:
        return set()

    if isinstance(eval_dataset_config, (list, tuple, set)):
        requested_ids = {str(item) for item in eval_dataset_config}
    else:
        requested_ids = {str(eval_dataset_config)}

    available_ids = {backend_id for backend_id in available_backends.keys()}

    missing_ids = sorted(requested_ids - available_ids)
    if missing_ids:
        formatted_missing = ", ".join(missing_ids)
        formatted_available = ", ".join(sorted(available_ids)) or "none"
        raise ValueError(
            f"--eval_dataset_id references unknown {context} dataset(s): {formatted_missing}. "
            f"Loaded datasets: {formatted_available}. "
            "Update the configuration or remove --eval_dataset_id."
        )
    return requested_ids


def retrieve_validation_images():
    """
    From each data backend, collect the top 5 images for validation, such that
    we select the same images on each startup, unless the dataset changes.

    Returns:
        dict: A dictionary of shortname to image paths.
    """
    model = StateTracker.get_model()
    if model.requires_validation_edit_captions() or model.requires_validation_i2v_samples():
        return retrieve_validation_edit_images()

    args = StateTracker.get_args()
    requires_cond_input = any(
        [
            StateTracker.get_model().requires_conditioning_validation_inputs(),
            args.controlnet,
            args.control,
        ]
    )
    dataset_type = StateTracker.get_model().conditioning_validation_dataset_type() if requires_cond_input else "image"
    data_backends = StateTracker.get_data_backends(
        _type=dataset_type, _types=None if requires_cond_input else ["image", "video"]
    )
    dataset_type_name = getattr(dataset_type, "value", str(dataset_type))
    selected_eval_backend_ids = _assert_eval_dataset_exists(
        args.eval_dataset_id, data_backends, f"{dataset_type_name} validation"
    )
    validation_set = []
    logger.info("Collecting validation images")
    for _data_backend in data_backends:
        data_backend = StateTracker.get_data_backend(_data_backend)
        data_backend_config = data_backend.get("config", {})
        should_skip_dataset = data_backend_config.get("disable_validation", False)
        logger.debug(f"Backend {_data_backend}: {data_backend}")
        if "id" not in data_backend or (
            requires_cond_input
            and data_backend.get("dataset_type", None) != StateTracker.get_model().conditioning_validation_dataset_type()
        ):
            logger.debug(f"Skipping data backend: {_data_backend} dataset_type {data_backend.get('dataset_type', None)}")
            continue
        logger.debug(f"Checking data backend: {data_backend['id']}")
        if (selected_eval_backend_ids and data_backend["id"] not in selected_eval_backend_ids) or should_skip_dataset:
            logger.warning(f"Not collecting images from {data_backend['id']}")
            continue
        if "sampler" in data_backend:
            validation_samples_from_sampler = data_backend["sampler"].retrieve_validation_set(
                batch_size=args.num_eval_images
            )
            validation_samples_from_sampler = [
                _normalise_validation_sample(sample) for sample in validation_samples_from_sampler
            ]
            validation_input_image_pixel_edge_len = StateTracker.get_model().validation_image_input_edge_length()
            if validation_input_image_pixel_edge_len is not None:
                logger.debug(
                    f"Resized validation image so that pixel edge length is {validation_input_image_pixel_edge_len}."
                )
                validation_samples_from_sampler = resize_validation_images(
                    validation_samples_from_sampler,
                    edge_length=validation_input_image_pixel_edge_len,
                )

            validation_set.extend(validation_samples_from_sampler)
        else:
            logger.warning(f"Data backend {data_backend['id']} does not have a sampler. Skipping.")
    logger.info(f"Collected {len(validation_set)} validation image inputs.")
    return validation_set


def retrieve_validation_edit_images() -> list[tuple[str, str, list[Image.Image]]]:
    """
    Returns [(shortname, *edited-scene caption*, reference_image), ...]
    for models that need **edit** validation (including I2V variants).

    Logic
    -----
    • loop over *image* datasets that have a sampler
    • for every deterministic validation sample returned by the sampler
      – grab its original file path from metadata
      – ask the dataset's *registered* conditioning backend for the
        counterpart via `get_conditioning_sample()`
      – add the trio to output
    """
    model = StateTracker.get_model()
    if not (model.requires_validation_edit_captions() or model.requires_validation_i2v_samples()):
        return []  # no-op for ordinary models

    args = StateTracker.get_args()
    # Respect the user's selected validation dataset via --eval_dataset_id and
    # honour any `disable_validation: true` flags in the backend configuration.
    validation_set = []

    image_backends = StateTracker.get_data_backends(_type="image")
    selected_eval_backend_ids = _assert_eval_dataset_exists(args.eval_dataset_id, image_backends, "image validation")

    # ---------- iterate over IMAGE datasets ---------------------------------
    for backend_id, backend in image_backends.items():
        backend_config = backend.get("config", {})
        should_skip_dataset = backend_config.get("disable_validation", False)

        # Skip datasets that the user has explicitly disabled for validation or
        # that do not match the requested `--eval_dataset_id`.
        if (selected_eval_backend_ids and backend.get("id") not in selected_eval_backend_ids) or should_skip_dataset:
            logger.debug(f"Not collecting edit-validation images from {backend.get('id', backend_id)}")
            continue
        sampler = backend.get("sampler")
        if sampler is None:  # nothing to iterate over
            continue

        # each backend should know which conditioning dataset is linked to it
        cond_backends = StateTracker.get_conditioning_datasets(backend_id)
        if not cond_backends:
            logger.debug("No conditioning backend configured for this image dataset.")
            continue

        # deterministic slice for validation
        for sample in sampler.retrieve_validation_set(batch_size=args.num_eval_images):
            # sample == (shortname, edited_prompt, pil_image)
            shortname, edited_prompt, sample_path, _ = sample

            # original relative file-path comes from metadata backend
            try:
                meta = StateTracker.get_metadata_by_filepath(
                    sample_path,
                    data_backend_id=backend_id,
                    search_dataset_types=["conditioning"],
                )
                image_dataset_dir_prefix = sampler.metadata_backend.instance_data_dir
                if image_dataset_dir_prefix is not None and image_dataset_dir_prefix in sample_path:
                    rel_path = sample_path.replace(image_dataset_dir_prefix, "")
                    # remove trailing '/'
                    rel_path = rel_path.lstrip("/")
                    logger.debug(f"Removed prefix, got relative path: {rel_path}")
                logger.debug(f"Metadata: {meta}")
            except Exception:
                continue  # metadata missing → skip

            reference_imgs = []
            for cond_backend in cond_backends:
                cond_sample = cond_backend["sampler"].get_conditioning_sample(rel_path)

                if cond_sample is None:
                    continue

                reference_imgs.append(cond_sample.image)
            if len(reference_imgs) != len(cond_backends):
                logger.warning(f"Didn't find enough conditioning samples for {rel_path}.")
                continue
            validation_set.append((shortname, edited_prompt, reference_imgs))

    logger.info(f"Collected {len(validation_set)} edit-validation samples.")
    return validation_set


def prepare_validation_prompt_list(args, embed_cache, model):
    validation_prompts = [""] if not StateTracker.get_args().validation_disable_unconditional else []
    validation_shortnames = ["unconditional"] if not StateTracker.get_args().validation_disable_unconditional else []
    if not hasattr(embed_cache, "model_type"):
        raise ValueError(
            f"The default text embed cache backend was not found. You must specify 'default: true' on your text embed data backend via {StateTracker.get_args().data_backend_config}."
        )
    model_type = embed_cache.model_type
    validation_sample_images = None
    if (
        ("deepfloyd" in args.model_family and str(args.model_flavour).startswith("ii-"))
        or model.requires_conditioning_validation_inputs()
        or args.controlnet
        or args.control
        or args.validation_using_datasets
    ):
        # Now, we prepare the DeepFloyd upscaler image inputs so that we can calculate their prompts.
        # If we don't do it here, they won't be available at inference time.
        validation_sample_images = retrieve_validation_images()
        if len(validation_sample_images) > 0:
            StateTracker.set_validation_sample_images(validation_sample_images)
            sample_prompts: list[str] = []
            sample_shortnames: list[str] = []
            # Collect the prompts for the validation images.
            for _validation_sample in tqdm(
                validation_sample_images,
                ncols=125,
                desc="Precomputing validation image embeds",
            ):
                validation_prompt = None
                shortname = None
                if isinstance(_validation_sample, tuple):
                    if len(_validation_sample) == 3:
                        shortname, validation_prompt, _ = _validation_sample
                    elif len(_validation_sample) == 4:
                        shortname, validation_prompt, *_ = _validation_sample
                if not validation_prompt:
                    logger.debug("Skipping validation sample without prompt while preparing embeds.")
                    continue
                embed_cache.compute_embeddings_for_prompts([validation_prompt], load_from_cache=False)
                if shortname is None:
                    shortname = f"validation_{len(sample_shortnames)}"
                sample_prompts.append(validation_prompt)
                sample_shortnames.append(shortname)
            if sample_prompts:
                validation_prompts.extend(sample_prompts)
                validation_shortnames.extend(sample_shortnames)
            time.sleep(5)

    allow_prompt_library = not (
        isinstance(model, VideoModelFoundation)
        and getattr(model, "requires_validation_i2v_samples", lambda: False)()
        and StateTracker.get_validation_sample_images()
    )

    if allow_prompt_library and args.validation_prompt_library:
        # Use the SimpleTuner prompts library for validation prompts.
        from simpletuner.helpers.prompts import prompts as prompt_library

        # Iterate through the prompts with a progress bar
        for shortname, prompt in tqdm(
            prompt_library.items(),
            leave=False,
            ncols=125,
            desc="Precomputing validation prompt embeddings",
        ):
            embed_cache.compute_embeddings_for_prompts([prompt], is_validation=True, load_from_cache=False)
            validation_prompts.append(prompt)
            validation_shortnames.append(shortname)

    if allow_prompt_library and args.user_prompt_library is not None:
        user_prompt_library = PromptHandler.load_user_prompts(args.user_prompt_library)
        for shortname, prompt in tqdm(
            user_prompt_library.items(),
            leave=False,
            ncols=125,
            desc="Precomputing user prompt library embeddings",
        ):
            # move_text_encoders(embed_cache.text_encoders, embed_cache.accelerator.device)
            embed_cache.compute_embeddings_for_prompts([prompt], is_validation=True, load_from_cache=False)
            # move_text_encoders(embed_cache.text_encoders, "cpu")
            validation_prompts.append(prompt)
            validation_shortnames.append(shortname)
    if allow_prompt_library and args.validation_prompt is not None and args.validation_prompt != "None":
        # Use a single prompt for validation.
        # This will add a single prompt to the prompt library, if in use.
        validation_prompts = validation_prompts + [args.validation_prompt]
        validation_shortnames = validation_shortnames + ["validation"]
        embed_cache.compute_embeddings_for_prompts([args.validation_prompt], is_validation=True, load_from_cache=False)
    # Compute negative embed for validation prompts, if any are set, so that it's stored before we unload the text encoder.
    if validation_prompts:
        negative_prompt = StateTracker.get_args().validation_negative_prompt
        logger.info(f"Precomputing the negative prompt embed for validations: {negative_prompt}")
        model.log_model_devices()
        validation_negative_prompt_text_encoder_output = embed_cache.compute_embeddings_for_prompts(
            [negative_prompt],
            is_validation=True,
            load_from_cache=False,
        )

    logger.info("Completed validation prompt gathering.")
    return {
        "validation_prompts": validation_prompts,
        "validation_shortnames": validation_shortnames,
        "validation_sample_images": validation_sample_images,
    }


def get_validation_resolutions():
    """
    If the args.validation_resolution:
     - is an int, we'll treat it as height and width square aspect
     - if it has an x in it, we will split and treat as WIDTHxHEIGHT
     - if it has comma, we will split and treat each value as above
    """
    validation_resolution_parameter = StateTracker.get_args().validation_resolution
    if type(validation_resolution_parameter) is str and "," in validation_resolution_parameter:
        return [parse_validation_resolution(res) for res in validation_resolution_parameter.split(",")]
    return [parse_validation_resolution(validation_resolution_parameter)]


def parse_validation_resolution(input_str: str) -> tuple:
    """
    If the args.validation_resolution:
     - is an int, we'll treat it as height and width square aspect
     - if it has an x in it, we will split and treat as WIDTHxHEIGHT
     - if it has comma, we will split and treat each value as above
    """
    is_df_ii = True if str(StateTracker.get_args().model_flavour).startswith("ii-") else False
    if isinstance(input_str, int) or input_str.isdigit():
        if is_df_ii and int(input_str) < 256:
            raise ValueError("Cannot use less than 256 resolution for DeepFloyd stage 2.")
        return (input_str, input_str)
    if "x" in input_str:
        pieces = input_str.split("x")
        if is_df_ii and (int(pieces[0]) < 256 or int(pieces[1]) < 256):
            raise ValueError("Cannot use less than 256 resolution for DeepFloyd stage 2.")
        return (int(pieces[0]), int(pieces[1]))


def load_video_frames(video_path):
    """Load video frames from a file."""
    try:
        import imageio

        reader = imageio.get_reader(video_path, "ffmpeg")
        frames = []
        for frame in reader:
            # Convert numpy array to PIL Image
            frames.append(Image.fromarray(frame))
        reader.close()
        return frames
    except ImportError:
        # Fallback to opencv if imageio not available
        try:
            import trainingsample as tsr

            cap = tsr.PyVideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB and then to PIL
                frame_rgb = tsr.cvt_color_py(frame, 4)  # 4 = COLOR_BGR2RGB
                frames.append(Image.fromarray(frame_rgb))
            cap.release()
            return frames
        except ImportError:
            logger.error("Neither imageio nor opencv-python is installed. Cannot load video frames.")
            return None


def apply_to_image_or_video(func):
    """
    Decorator that allows image manipulation functions to work on both single images and video frames.
    If input is a list (video frames), applies the function to each frame.
    If input is a single image, applies the function directly.
    """

    def wrapper(image_or_frames, *args, **kwargs):
        if isinstance(image_or_frames, list):
            # It's a video - apply to each frame
            return [func(frame, *args, **kwargs) for frame in image_or_frames]
        else:
            # It's a single image
            return func(image_or_frames, *args, **kwargs)

    return wrapper


def get_font_for_labels(font_size=28):
    """
    Helper function to get a font for labels with fallback options.

    Args:
        font_size (int): The desired font size

    Returns:
        ImageFont object or None if no font could be loaded
    """
    font = None
    font_candidates = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "Arial.ttf",
        "arial.ttf",
        "FreeSans.ttf",
        "NotoSans-Regular.ttf",
    ]

    for font_name in font_candidates:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except IOError:
            continue

    if font is None:
        # As a fallback, create a simple default font
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None  # Last resort

    return font


@apply_to_image_or_video
def draw_text_on_image(
    image,
    text,
    font=None,
    position=None,
    fill=(255, 255, 255),
    stroke_width=2,
    stroke_fill=(0, 0, 0),
):
    """Draw text on a single image."""
    draw = ImageDraw.Draw(image)

    if font is None:
        font = ImageFont.load_default()

    if position is None:
        # Center the text at the bottom
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = font.getsize(text)

        margin = 10
        x = (image.width - text_width) // 2
        y = image.height - text_height - margin
        position = (x, y)

    draw.text(
        position,
        text,
        fill=fill,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )
    return image


def stitch_images_or_videos(left, right, separator_width=5, labels=None):
    """
    Stitch two images or two videos side by side.
    If inputs are lists (video frames), stitches frame by frame.
    """
    if isinstance(left, list) and isinstance(right, list):
        # Both are videos - stitch frame by frame
        if len(left) != len(right):
            raise ValueError(f"Videos must have the same number of frames. Got {len(left)} and {len(right)}")
        return [_stitch_single_pair(left[i], right[i], separator_width, labels) for i in range(len(left))]
    elif not isinstance(left, list) and not isinstance(right, list):
        # Both are single images
        return _stitch_single_pair(left, right, separator_width, labels)
    else:
        raise ValueError("Both inputs must be either single images or lists of frames")


def _stitch_single_pair(left_image, right_image, separator_width=5, labels=None):
    """Helper to stitch a single pair of images."""
    # Your existing stitching logic here
    left_width, left_height = left_image.size
    right_width, right_height = right_image.size

    new_width = left_width + separator_width + right_width
    new_height = max(left_height, right_height)

    new_image = Image.new("RGB", (new_width, new_height), color="white")

    # Center vertically if needed
    left_y = (new_height - left_height) // 2
    right_y = (new_height - right_height) // 2

    new_image.paste(left_image, (0, left_y))
    new_image.paste(right_image, (left_width + separator_width, right_y))

    # Draw separator
    draw = ImageDraw.Draw(new_image)
    line_color = (200, 200, 200)
    for i in range(separator_width):
        x = left_width + i
        draw.line([(x, 0), (x, new_height)], fill=line_color)

    # Add labels if provided
    if labels:
        font = get_font_for_labels()
        if labels[0] is not None:
            draw.text(
                (10, 10),
                labels[0],
                fill=(255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
        if len(labels) > 1 and labels[1] is not None:
            draw.text(
                (left_width + separator_width + 10, 10),
                labels[1],
                fill=(255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

    return new_image


@dataclass(frozen=True)
class _ValidationWorkItem:
    index: int
    shortname: str
    prompt: str
    conditioning: Any


@dataclass(frozen=True)
class _PreviewMetadata:
    shortname: str
    prompt: str
    resolution: tuple[int, int]
    validation_type: str | None
    total_steps: int | None = None


class ValidationPreviewer:
    """
    Handles Tiny AutoEncoder powered previews emitted during validation.
    """

    def __init__(self, model: ModelFoundation, accelerator, config):
        self.model = model
        self.accelerator = accelerator
        self.config = config
        self.enabled = bool(getattr(config, "validation_preview", False))
        interval = getattr(config, "validation_preview_steps", 1)
        try:
            interval = int(interval)
        except (ValueError, TypeError):
            interval = 1
        self.step_interval = max(1, interval)
        self._decoder = None
        self._decoder_failed = False
        self._warned_unsupported = False
        self._warned_callback = False
        self._webhook_handler = StateTracker.get_webhook_handler()
        if self.enabled and not self._webhook_handler:
            logger.warning("validation_preview is enabled but no webhook is configured; previews are disabled.")
            self.enabled = False
        if self.enabled and not self.model.supports_validation_preview():
            logger.warning(
                "validation_preview requested but %s does not provide Tiny AutoEncoder metadata.", self.model.NAME
            )
            self.enabled = False

    def _is_main_process(self) -> bool:
        if self.accelerator is None:
            return True
        return bool(getattr(self.accelerator, "is_main_process", True))

    def _ensure_decoder(self) -> bool:
        if not self.enabled or self._decoder_failed:
            return False
        if self._decoder is None:
            self._decoder = self.model.get_validation_preview_decoder()
        if self._decoder is None:
            if not self._warned_unsupported:
                logger.warning("validation_preview requested but no Tiny AutoEncoder could be loaded.")
                self._warned_unsupported = True
            return False
        return True

    def _pipeline_supports_preview(self, pipeline) -> bool:
        if pipeline is None:
            return False
        signature = inspect.signature(pipeline.__call__)
        return "callback_on_step_end" in signature.parameters

    @contextmanager
    def attach(self, pipeline, pipeline_kwargs: dict, metadata: _PreviewMetadata):
        if not self.enabled or not self._is_main_process():
            yield
            return
        if not self._ensure_decoder():
            yield
            return
        if not self._pipeline_supports_preview(pipeline):
            if not self._warned_callback:
                logger.warning(
                    "validation_preview is enabled but pipeline %s does not expose callback_on_step_end.",
                    type(pipeline).__name__,
                )
                self._warned_callback = True
            yield
            return

        def _callback(pipe, step, timestep, callback_kwargs):
            self._handle_callback(step, timestep, callback_kwargs, metadata)
            return callback_kwargs

        pipeline_kwargs["callback_on_step_end"] = _callback
        pipeline_kwargs.setdefault("callback_on_step_end_tensor_inputs", ["latents"])
        try:
            yield
        finally:
            pipeline_kwargs.pop("callback_on_step_end", None)
            if pipeline_kwargs.get("callback_on_step_end_tensor_inputs") == ["latents"]:
                pipeline_kwargs.pop("callback_on_step_end_tensor_inputs", None)

    def _handle_callback(self, step: int, timestep, callback_kwargs: dict, metadata: _PreviewMetadata):
        if not self.enabled or self._decoder_failed:
            return
        if not self._should_emit_for_step(step):
            return
        latents = callback_kwargs.get("latents")
        if latents is None:
            return
        try:
            image_payloads, video_payloads = self._decode_preview(latents)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.warning("Disabling validation preview due to decode failure: %s", exc)
            self._decoder_failed = True
            return
        if not image_payloads and not video_payloads:
            return
        timestep_value = timestep.item() if torch.is_tensor(timestep) else timestep
        self._emit_event(image_payloads, video_payloads, metadata, step, timestep_value)

    def _decode_preview(self, latents: torch.Tensor):
        decoder = self._decoder
        latents = latents.detach()
        dtype = getattr(decoder, "dtype", torch.float32)
        device = getattr(decoder, "device", latents.device)
        latents = latents.to(device=device, dtype=torch.float32)
        if getattr(decoder, "requires_vae_rescaling", False):
            latents = self.model.denormalize_latents_for_preview(latents)
        latents = latents.to(dtype=dtype)
        decoded = decoder.decode(latents)
        if self._decoder.is_video:
            frames = decoded[0]
            pil_frames = [self._tensor_to_pil(frame) for frame in frames]
            video_payload = self._frames_to_gif(pil_frames)
            first_frame = pil_frames[0] if pil_frames else None
            images = [first_frame] if first_frame else []
            return images, video_payload
        else:
            image = decoded[0]
            return [self._tensor_to_pil(image)], None

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        array = tensor.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(array)

    def _frames_to_gif(self, frames):
        if not frames:
            return None
        import imageio

        buffer = BytesIO()
        imageio.mimsave(buffer, [np.array(frame) for frame in frames], format="GIF", duration=0.08)
        data_uri = f"data:image/gif;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
        return [{"src": data_uri, "mime_type": "image/gif"}]

    def _emit_event(self, images, videos, metadata: _PreviewMetadata, step: int, timestep):
        if self._webhook_handler is None:
            return
        total_steps = metadata.total_steps
        if not total_steps:
            total_steps = getattr(self.config, "validation_num_inference_steps", None)
        if total_steps:
            step_label = f"{int(step + 1)}/{int(total_steps)}"
        else:
            step_label = f"{int(step + 1)}"
        message_text = f"Validation (step {step_label}): {metadata.shortname or '(validation)'}"
        payload = {
            "type": "validation.image",
            "title": message_text,
            "message": message_text,
            "body": metadata.prompt or "",
            "data": {
                "step": int(step + 1),
                "timestep": float(timestep) if timestep is not None else None,
                "resolution": list(metadata.resolution),
                "validation_type": metadata.validation_type,
                "prompt": metadata.prompt,
                "step_label": step_label,
            },
        }
        self._webhook_handler.send_raw(
            structured_data=payload,
            message_type="validation.image",
            images=images,
            videos=videos,
            job_id=StateTracker.get_job_id(),
        )

    def _should_emit_for_step(self, step: int) -> bool:
        """
        Return True if the preview should be emitted for this sampling step.

        The first sampling step always emits so that users see an immediate preview.
        Subsequent emissions follow the configured interval, which is specified in
        one-indexed units to match scheduler step counts.
        """

        if self.step_interval <= 1 or step == 0:
            return True
        return ((step + 1) % self.step_interval) == 0


class Validation:
    def __init__(
        self,
        accelerator,
        model: ModelFoundation,
        distiller,
        args,
        validation_prompt_metadata,
        vae_path,
        weight_dtype,
        embed_cache,
        ema_model,
        is_deepspeed: bool = False,
        is_fsdp: bool = False,
        model_evaluator=None,
        trainable_parameters=None,
    ):
        self.trainable_parameters = trainable_parameters
        self.accelerator = accelerator
        self.prompt_handler = None
        self.unet, self.transformer = None, None
        self.model = model
        self.distiller = distiller
        if args.controlnet:
            self.controlnet = model.get_trained_component(unwrap_model=False)
        elif "unet" in str(self.model.get_trained_component().__class__).lower():
            self.unet = self.model.get_trained_component(unwrap_model=False)
        elif "transformer" in str(self.model.get_trained_component().__class__).lower():
            self.transformer = self.model.get_trained_component(unwrap_model=False)
        self.config = args
        self.save_dir = os.path.join(args.output_dir, "validation_images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.global_step = None
        self.global_resume_step = None
        self.validation_prompt_metadata = validation_prompt_metadata
        self.validation_images = None
        self.validation_video_paths: dict[str, list[str]] = {}
        self.weight_dtype = weight_dtype
        self.embed_cache = embed_cache
        self.ema_model = ema_model
        self.ema_enabled = False
        self.deepfloyd = True if "deepfloyd" in self.config.model_family else False
        self.deepfloyd_stage2 = True if str(self.config.model_flavour).startswith("ii-") else False
        preset_validation_inputs = None
        if isinstance(self.validation_prompt_metadata, dict):
            preset_validation_inputs = self.validation_prompt_metadata.get("validation_sample_images")
        if preset_validation_inputs:
            self.validation_image_inputs = [_normalise_validation_sample(sample) for sample in preset_validation_inputs]
            logger.debug(f"Loaded {len(self.validation_image_inputs)} validation image inputs from metadata.")
        else:
            self._discover_validation_input_samples()
        self.validation_resolutions = get_validation_resolutions() if not self.deepfloyd_stage2 else [(256, 256)]
        self.flow_matching = True if self.model.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING else False
        self.deepspeed = is_deepspeed
        self.fsdp = is_fsdp
        self._epoch_validations_completed: set[int] = set()
        self._pending_epoch_validation: int | None = None
        if is_deepspeed:
            if args.use_ema:
                if args.ema_validation != "none":
                    logger.error(
                        "EMA validation is not supported via DeepSpeed."
                        " Please use --ema_validation=none or disable DeepSpeed."
                    )
                    sys.exit(1)
        self.inference_device = accelerator.device if not is_deepspeed else "cuda" if torch.cuda.is_available() else "cpu"
        self.model_evaluator = model_evaluator
        if self.model_evaluator is not None:
            logger.debug(f"Using model evaluator: {self.model_evaluator}")
        self._update_state()
        self.eval_scores = {}
        self.validation_adapter_runs = build_validation_adapter_runs(
            getattr(args, "validation_adapter_path", None),
            getattr(args, "validation_adapter_config", None),
            adapter_name=getattr(args, "validation_adapter_name", None),
            adapter_strength=float(getattr(args, "validation_adapter_strength", 1.0) or 1.0),
            adapter_mode=getattr(args, "validation_adapter_mode", None),
        )
        self._active_adapter_run: ValidationAdapterRun | None = None
        self.preview = ValidationPreviewer(self.model, self.accelerator, self.config)

    def _validation_multigpu_mode(self) -> str:
        """
        Return the requested validation multi-GPU behaviour while accounting for single-process runs.
        """
        raw_mode = getattr(self.config, "validation_multigpu", "batch-parallel")
        mode = str(raw_mode).strip().lower() if raw_mode is not None else "batch-parallel"
        if mode not in {"single-gpu", "batch-parallel"}:
            mode = "single-gpu"
        if getattr(self.accelerator, "num_processes", 1) <= 1:
            return "single-gpu"
        return mode

    def _use_distributed_validation(self) -> bool:
        return self._validation_multigpu_mode() == "batch-parallel"

    def _validation_seed_source(self):
        if self.config.validation_seed_source == "gpu":
            return self.inference_device
        elif self.config.validation_seed_source == "cpu":
            return "cpu"
        else:
            raise Exception("Unknown validation seed source. Options: cpu, gpu")

    def _get_generator(self):
        _validation_seed_source = self._validation_seed_source()
        _generator = torch.Generator(device=_validation_seed_source).manual_seed(
            self.config.validation_seed or self.config.seed or 0
        )
        return _generator

    def clear_text_encoders(self):
        """
        Sets all text encoders to None.

        Returns:
            None
        """
        self.model.unload_text_encoder()

    def _discover_validation_input_samples(self):
        """
        If we have some workflow that requires image inputs for validation, we'll bind those now.

        Returns:
            Validation object (self)
        """
        self.validation_image_inputs = None
        if (
            self.deepfloyd_stage2
            or self.config.validation_using_datasets
            or self.config.controlnet
            or self.config.control
            or self.model.requires_conditioning_validation_inputs()
        ):
            self.validation_image_inputs = retrieve_validation_images()
            # Validation inputs are in the format of a list of tuples:
            # [(shortname, prompt, image), ...]
            logger.debug(f"Image inputs discovered for validation: {self.validation_image_inputs}")

    def _pipeline_cls(self):
        if self.model is not None:
            if self.config.validation_using_datasets:
                if PipelineTypes.IMG2IMG not in self.model.PIPELINE_CLASSES:
                    raise ValueError(f"Cannot run {self.model.MODEL_CLASS} in Img2Img mode for validation.")
            if self.config.controlnet:
                if PipelineTypes.CONTROLNET not in self.model.PIPELINE_CLASSES:
                    raise ValueError(f"Cannot run {self.model.MODEL_CLASS} in ControlNet mode for validation.")
            if self.config.control:
                if PipelineTypes.CONTROL not in self.model.PIPELINE_CLASSES:
                    raise ValueError(f"Cannot run {self.model.MODEL_CLASS} in Control mode for validation.")

        if self.config.validation_using_datasets:
            return self.model.PIPELINE_CLASSES[PipelineTypes.IMG2IMG]
        if self.config.controlnet:
            return self.model.PIPELINE_CLASSES[PipelineTypes.CONTROLNET]
        if self.config.control:
            return self.model.PIPELINE_CLASSES[PipelineTypes.CONTROL]
        return self.model.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]

    def _gather_prompt_embeds(self, validation_prompt: str):
        prompt_embed = self.embed_cache.compute_embeddings_for_prompts([validation_prompt])
        if prompt_embed is None:
            return

        prompt_embed = {k: v.to(self.inference_device) if hasattr(v, "to") else v for k, v in prompt_embed.items()}

        return self.model.convert_text_embed_for_pipeline(prompt_embed)

    def _benchmark_image(self, shortname, resolution):
        """
        We will retrieve the benchmark image/video for the shortname.
        """
        if not self.benchmark_exists():
            return None

        base_model_benchmark = self._benchmark_path("base_model")

        # Check if this is a video model
        if isinstance(self.model, VideoModelFoundation):
            # Look for video with resolution in filename
            video_pattern = f"{shortname}_{resolution[0]}x{resolution[1]}_"
            for filename in os.listdir(base_model_benchmark):
                if filename.startswith(video_pattern) and filename.endswith(".mp4"):
                    video_path = os.path.join(base_model_benchmark, filename)
                    frames = load_video_frames(video_path)
                    if frames:
                        logger.debug(f"Loaded {len(frames)} frames from benchmark video: {filename}")
                        return frames
                    else:
                        logger.warning(f"Failed to load benchmark video: {filename}")
                        return None

            # Fallback: try without resolution (old format)
            for filename in os.listdir(base_model_benchmark):
                if filename.startswith(f"{shortname}_") and filename.endswith(".mp4"):
                    video_path = os.path.join(base_model_benchmark, filename)
                    frames = load_video_frames(video_path)
                    if frames:
                        logger.debug(f"Loaded {len(frames)} frames from benchmark video: {filename}")
                        return frames

            # NEW: If no video found, fall back to image (video model might output images)
            logger.debug(f"No video benchmark found for {shortname}, checking for image...")

        # Image logic (now also used as fallback for video models)
        image_filename = f"{shortname}_{resolution[0]}x{resolution[1]}.png"
        image_path = os.path.join(base_model_benchmark, image_filename)
        if os.path.exists(image_path):
            logger.debug(f"Found image benchmark: {image_filename}")
            return Image.open(image_path)

        return None

    def stitch_benchmark_image(
        self,
        validation_image_result,
        benchmark_image,
        separator_width=5,
        labels=["base model", "checkpoint"],
    ):
        """
        Stitch benchmark and validation images/videos side by side.
        Handles both single images and lists of frames (videos).
        """
        # Check if both are videos (lists)
        if isinstance(validation_image_result, list) and isinstance(benchmark_image, list):
            # Stitch frame by frame
            stitched_frames = []
            max_frames = min(len(validation_image_result), len(benchmark_image))

            for i in range(max_frames):
                stitched_frame = self._stitch_single_images(
                    benchmark_image[i],
                    validation_image_result[i],
                    separator_width,
                    labels,
                )
                stitched_frames.append(stitched_frame)

            return stitched_frames

        # Both are single images
        elif hasattr(validation_image_result, "size") and hasattr(benchmark_image, "size"):
            return self._stitch_single_images(benchmark_image, validation_image_result, separator_width, labels)

        # Type mismatch - can't stitch
        else:
            logger.warning("Cannot stitch benchmark: type mismatch between video and image")
            return validation_image_result

    def _stitch_single_images(
        self,
        left_image,
        right_image,
        separator_width=5,
        labels=["base model", "checkpoint"],
    ):
        """Helper method to stitch two single images."""
        # Calculate dimensions
        raw_width = left_image.size[0] + right_image.size[0] + separator_width
        raw_height = max(left_image.size[1], right_image.size[1])

        # Ensure dimensions are divisible by 16 for video encoding
        new_width = raw_width if raw_width % 16 == 0 else raw_width + 1
        new_height = raw_height if raw_height % 16 == 0 else raw_height + 1

        new_image = Image.new("RGB", (new_width, new_height), color="white")

        # Center vertically if heights differ
        left_y = (new_height - left_image.size[1]) // 2
        right_y = (new_height - right_image.size[1]) // 2

        new_image.paste(left_image, (0, left_y))
        new_image.paste(right_image, (left_image.size[0] + separator_width, right_y))

        draw = ImageDraw.Draw(new_image)

        # Draw separator
        line_color = (200, 200, 200)
        for i in range(separator_width):
            x = left_image.size[0] + i
            draw.line([(x, 0), (x, new_height)], fill=line_color)

        # Add labels
        font = get_font_for_labels()

        if labels[0] is not None:
            draw.text(
                (10, 10),
                labels[0],
                fill=(255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

        if labels[1] is not None:
            draw.text(
                (left_image.size[0] + separator_width + 10, 10),
                labels[1],
                fill=(255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

        return new_image

    def _benchmark_images(self):
        """
        We will retrieve the benchmark images so they can be stitched to the validation outputs.
        """
        if not self.benchmark_exists():
            return None
        benchmark_images = []
        base_model_benchmark = self._benchmark_path("base_model")
        for _benchmark_image in os.listdir(base_model_benchmark):
            if _benchmark_image.endswith(".png"):
                benchmark_images.append(
                    (
                        _benchmark_image.replace(".png", ""),
                        f"Base model benchmark image {_benchmark_image}",
                        Image.open(os.path.join(base_model_benchmark, _benchmark_image)),
                    )
                )

        return benchmark_images

    def benchmark_exists(self, benchmark: str = "base_model"):
        """
        Determines whether the base model benchmark outputs already exist.
        """
        base_model_benchmark = self._benchmark_path()

        return os.path.exists(base_model_benchmark)

    def _benchmark_path(self, benchmark: str = "base_model"):
        # does the benchmark directory exist?
        if not os.path.exists(os.path.join(self.config.output_dir, "benchmarks")):
            os.makedirs(os.path.join(self.config.output_dir, "benchmarks"), exist_ok=True)
        return os.path.join(self.config.output_dir, "benchmarks", benchmark)

    def save_benchmark(self, benchmark: str = "base_model"):
        """
        Saves the benchmark outputs for the base model.
        """
        base_model_benchmark = self._benchmark_path(benchmark=benchmark)
        if not os.path.exists(base_model_benchmark):
            os.makedirs(base_model_benchmark, exist_ok=True)
        if self.validation_images is None:
            return

        for shortname, image_list in self.validation_images.items():
            for idx, image in enumerate(image_list):
                # Get the validation resolution that was used for this index
                if idx < len(self.validation_resolutions):
                    logger.debug(f"Validation saving image for resolution idx {idx} of {len(self.validation_resolutions)}")
                    resolution = self.validation_resolutions[idx]
                    if isinstance(resolution, str):
                        # Parse resolution string if needed
                        if "x" in resolution:
                            width, height = map(int, resolution.split("x"))
                        else:
                            width = height = int(resolution)
                    elif isinstance(resolution, tuple):
                        width, height = resolution
                    else:
                        width = height = resolution
                else:
                    # Fallback to actual size if we somehow have more images than resolutions
                    logger.warning(
                        f"Image index {idx} exceeds validation resolutions list {len(self.validation_resolutions)}, using actual size"
                    )
                    if hasattr(image, "size"):
                        width, height = image.size
                    elif isinstance(image, list) and len(image) > 0 and hasattr(image[0], "size"):
                        width, height = image[0].size
                    else:
                        logger.error(f"Could not determine size for image at index {idx}")
                        continue

                if hasattr(image, "size"):
                    # Single image - save with validation resolution in filename
                    image.save(os.path.join(base_model_benchmark, f"{shortname}_{width}x{height}.png"))
                elif type(image) is list:
                    # Video frames - save with validation resolution in filename
                    from diffusers.utils.export_utils import export_to_video

                    filename = f"{shortname}_{width}x{height}_{idx}.mp4"
                    export_to_video(
                        image,
                        os.path.join(base_model_benchmark, filename),
                        fps=self.config.framerate,
                    )

    def _update_state(self):
        """Updates internal state with the latest from StateTracker."""
        self.global_step = StateTracker.get_global_step()
        self.global_resume_step = StateTracker.get_global_resume_step() or 1
        self.current_epoch = StateTracker.get_epoch()
        self.current_epoch_step = StateTracker.get_epoch_step()

    def would_validate(
        self,
        step: int = 0,
        validation_type="intermediary",
        force_evaluation: bool = False,
    ):
        # a wrapper for should_perform_intermediary_validation that can run in the training loop
        self._update_state()
        return self.should_perform_intermediary_validation(step, self.validation_prompt_metadata, validation_type) or (
            step == 0 and validation_type == "base_model"
        )

    def run_validations(
        self,
        step: int = 0,
        validation_type="intermediary",
        force_evaluation: bool = False,
        skip_execution: bool = False,
    ):
        self._update_state()
        if self.validation_prompt_metadata is None:
            return self
        content = self.validation_prompt_metadata.get("validation_prompts", None)
        has_validation_prompts = content is not None and len(content) > 0
        current_step_aligns_with_interval = self.should_perform_intermediary_validation(
            step, self.validation_prompt_metadata, validation_type
        )
        is_base_model_benchmark = step == 0 and validation_type == "base_model"
        epoch_validation_pending = (
            validation_type == "intermediary"
            and current_step_aligns_with_interval
            and self._pending_epoch_validation is not None
            and self._pending_epoch_validation == self.current_epoch
        )
        current_validation_will_execute = has_validation_prompts and (
            current_step_aligns_with_interval or is_base_model_benchmark
        )
        logger.debug(
            f"Should evaluate: {current_validation_will_execute}, force evaluation: {force_evaluation}, skip execution: {skip_execution}"
        )
        if (not current_validation_will_execute and not force_evaluation) or not has_validation_prompts:
            return self
        if current_validation_will_execute and validation_type == "final":
            # If the validation would have fired off, we'll skip it.
            # This is useful at the end of training so we don't validate 2x.
            logger.debug("Not running validation because intermediary might have already fired off.")
            return self
        webhook_handler = StateTracker.get_webhook_handler()
        if webhook_handler is not None and (self.accelerator.is_main_process or not self._use_distributed_validation()):
            webhook_handler.send(
                message="Validations are generating.. this might take a minute! 🖼️",
                message_level="info",
            )
            webhook_handler.send_lifecycle_stage(
                stage_key="validation",
                stage_label="Running Validation",
                stage_status="running",
                message="Validation is starting.",
            )

        should_execute_locally = self.accelerator.is_main_process or self.deepspeed or self._use_distributed_validation()
        if should_execute_locally:
            logger.debug("Starting validation process...")
            diffusers.utils.logging._tqdm_active = False
            self.setup_pipeline(validation_type)
            if self.model.pipeline is None:
                logger.error("Not able to run validations, we did not obtain a valid pipeline.")
                self.validation_images = None
                return self
            self.setup_scheduler()
            master_validation_images: dict = {}
            self.validation_prompt_dict = {}
            self.validation_video_paths.clear()
            self.eval_scores = {}
            for adapter_run in self.validation_adapter_runs:
                self._log_adapter_run(adapter_run)
                with self._temporary_validation_adapters(adapter_run):
                    self.process_prompts(
                        validation_type=validation_type,
                        adapter_run=adapter_run,
                        image_accumulator=master_validation_images,
                    )
            self.validation_images = master_validation_images
            self.finalize_validation(validation_type)
            if self.evaluation_result is not None:
                logger.info(f"Evaluation result: {self.evaluation_result}")
            logger.debug("Validation process completed.")
            if webhook_handler is not None and (self.accelerator.is_main_process or not self._use_distributed_validation()):
                webhook_handler.send_lifecycle_stage(
                    stage_key="validation",
                    stage_label="Running Validation",
                    stage_status="completed",
                    message="Validation completed.",
                )
            self.clean_pipeline()

        if epoch_validation_pending and current_validation_will_execute and validation_type == "intermediary":
            self._epoch_validations_completed.add(self._pending_epoch_validation)
            self._pending_epoch_validation = None

        return self

    def should_perform_intermediary_validation(self, step, validation_prompts, validation_type):
        if validation_prompts is None or (isinstance(validation_prompts, list) and len(validation_prompts) == 0):
            return False
        step_interval_value = getattr(self.config, "validation_step_interval", None)
        if step_interval_value in ("", "None"):
            step_interval_value = None
        if step_interval_value is None:
            step_interval_value = getattr(self.config, "validation_steps", None)
        try:
            step_interval = int(step_interval_value) if step_interval_value is not None else None
        except (TypeError, ValueError):
            step_interval = None

        epoch_interval_value = getattr(self.config, "validation_epoch_interval", None)
        if epoch_interval_value in ("", "None"):
            epoch_interval_value = None
        try:
            epoch_interval = int(epoch_interval_value) if epoch_interval_value is not None else None
        except (TypeError, ValueError):
            epoch_interval = None

        should_do_step_validation = False
        if (
            validation_prompts
            and step_interval is not None
            and step_interval > 0
            and self.global_step > self.global_resume_step
        ):
            should_do_step_validation = self.global_step % step_interval == 0 and (
                step % self.config.gradient_accumulation_steps == 0
            )

        should_do_epoch_validation = False
        if (
            validation_prompts
            and epoch_interval is not None
            and epoch_interval > 0
            and self.global_step > self.global_resume_step
            and self.current_epoch is not None
            and self.current_epoch > 0
        ):
            if self._pending_epoch_validation is not None and self._pending_epoch_validation == self.current_epoch:
                should_do_epoch_validation = True
            elif self.current_epoch % epoch_interval == 0 and (self.current_epoch not in self._epoch_validations_completed):
                self._pending_epoch_validation = self.current_epoch
                should_do_epoch_validation = True

        should_validate = (should_do_step_validation or should_do_epoch_validation) and (
            self.accelerator.is_main_process or self.deepspeed
        )
        return should_validate

    def setup_scheduler(self):
        if self.distiller is not None:
            distillation_scheduler = self.distiller.get_scheduler()
            if distillation_scheduler is not None:
                self.model.pipeline.scheduler = distillation_scheduler
                return distillation_scheduler
        scheduler_args = {
            "prediction_type": self.config.prediction_type,
        }
        if self.config.model_family == "sana":
            self.config.validation_noise_scheduler = "sana"
        elif (
            getattr(self.model, "DEFAULT_NOISE_SCHEDULER", None) is not None
            and self.config.validation_noise_scheduler is None
        ):
            # set the default
            self.config.validation_noise_scheduler = self.model.DEFAULT_NOISE_SCHEDULER
        if self.model.PREDICTION_TYPE.value == "flow_matching":
            # some flow-matching adjustments should be made for euler and unipc video model generations.
            if self.config.validation_noise_scheduler in ["flow_matching", "euler"]:
                if self.config.validation_noise_scheduler == "euler":
                    self.config.validation_noise_scheduler = "flow_matching"
                # The Beta schedule looks WAY better...
                if not self.model.pipeline.scheduler.config.get("use_karras_sigmas", False):
                    scheduler_args["use_beta_sigmas"] = True
                scheduler_args["shift"] = self.config.flow_schedule_shift
            if self.config.validation_noise_scheduler in ["flow_unipc", "unipc"]:
                scheduler_args["prediction_type"] = "flow_prediction"
                scheduler_args["use_flow_sigmas"] = True
                scheduler_args["num_train_timesteps"] = 1000
                scheduler_args["flow_shift"] = self.config.flow_schedule_shift

        if self.config.validation_noise_scheduler is None:
            # if the user or model config has not supplied one, we just allow pipeline to do defaults.
            return

        if self.config.prediction_type is not None:
            scheduler_args["prediction_type"] = self.config.prediction_type

        if self.model.pipeline is not None and "variance_type" in self.model.pipeline.scheduler.config:
            variance_type = self.model.pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        scheduler = SCHEDULER_NAME_MAP[self.config.validation_noise_scheduler].from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="scheduler",
            revision=self.config.revision,
            timestep_spacing=self.config.inference_scheduler_timestep_spacing,
            rescale_betas_zero_snr=self.config.rescale_betas_zero_snr,
            **scheduler_args,
        )
        if self.model.pipeline is not None:
            self.model.pipeline.scheduler = scheduler
        return scheduler

    def setup_pipeline(self, validation_type):
        if hasattr(self.accelerator, "_lycoris_wrapped_network"):
            self.accelerator._lycoris_wrapped_network.set_multiplier(
                float(getattr(self.config, "validation_lycoris_strength", 1.0))
            )

        pipeline_type = (
            PipelineTypes.CONTROLNET
            if self.config.controlnet
            else (PipelineTypes.CONTROL if self.config.control else self.model.DEFAULT_PIPELINE_TYPE)
        )
        self.model.pipeline = self.model.get_pipeline(
            pipeline_type=pipeline_type,
            load_base_model=False,
        )

        self.model.move_models(self.accelerator.device)
        # Remove text encoders on 'meta' device to avoid move errors
        for attr in [
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "text_encoder_4",
        ]:
            te = getattr(self.model.pipeline, attr, None)
            if getattr(te, "device", None) and te.device.type == "meta":
                setattr(self.model.pipeline, attr, None)

        # For FSDP models, skip .to() call - DTensor parameters are already device-aware
        # and calling .to() causes: "RuntimeError: Attempted to set the storage of a tensor
        # on device 'cpu' to a storage on different device 'cuda:0'"
        pipeline_model = getattr(self.model.pipeline, self.model.MODEL_TYPE.value, None)
        is_fsdp = FSDP_AVAILABLE and pipeline_model is not None and isinstance(pipeline_model, FSDP)

        if not is_fsdp:
            self.model.pipeline.to(self.accelerator.device)

        self.model.pipeline.set_progress_bar_config(disable=True)

    def clean_pipeline(self):
        """Remove the pipeline."""
        if hasattr(self.accelerator, "_lycoris_wrapped_network"):
            self.accelerator._lycoris_wrapped_network.set_multiplier(1.0)
        if self.model.pipeline is not None:
            del self.model.pipeline
            self.model.pipeline = None

    def _has_adapter_variants(self) -> bool:
        return any(run.adapters for run in self.validation_adapter_runs if not run.is_base)

    def _decorate_shortname(self, shortname: str, adapter_run: ValidationAdapterRun | None) -> str:
        if adapter_run is None or adapter_run.is_base:
            return shortname
        suffix = adapter_run.slug
        if not shortname:
            return suffix
        return f"{shortname}__{suffix}"

    def _log_adapter_run(self, adapter_run: ValidationAdapterRun):
        if adapter_run.is_base and not self._has_adapter_variants():
            return
        if adapter_run.is_base:
            logger.info("Running validation without additional adapters.")
            return
        logger.info(
            "Running validation with adapter set '%s' containing %d adapter(s).",
            adapter_run.label,
            len(adapter_run.adapters),
        )

    def _next_adapter_name(
        self, adapter_run: ValidationAdapterRun, adapter_spec: ValidationAdapterSpec, idx: int, existing: list[str]
    ) -> str:
        base_name = adapter_spec.adapter_name or (adapter_run.slug or f"validation_adapter_{idx}")
        candidate = base_name.strip() or f"validation_adapter_{idx}"
        suffix = 2
        while candidate in existing:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        return candidate

    @contextmanager
    def _temporary_validation_adapters(self, adapter_run: ValidationAdapterRun):
        if adapter_run is None or not adapter_run.adapters:
            yield
            return
        pipeline = getattr(self.model, "pipeline", None)
        if pipeline is None:
            yield
            return
        if not hasattr(pipeline, "load_lora_weights"):
            raise ValueError(
                "The current pipeline does not support loading LoRA adapters. "
                "Remove --validation_adapter_path/--validation_adapter_config to continue."
            )
        adapter_names: list[str] = []
        adapter_scales: list[float] = []
        for idx, adapter in enumerate(adapter_run.adapters):
            adapter_name = self._next_adapter_name(adapter_run, adapter, idx, adapter_names)
            load_kwargs = {"adapter_name": adapter_name}
            if adapter.weight_name:
                load_kwargs["weight_name"] = adapter.weight_name
            try:
                if adapter.is_local:
                    pipeline.load_lora_weights(adapter.location, **load_kwargs)
                else:
                    pipeline.load_lora_weights(adapter.repo_id, **load_kwargs)
            except Exception as exc:  # pragma: no cover - defensive log
                logger.error("Failed to load validation adapter '%s': %s", adapter.location, exc)
                raise
            adapter_names.append(adapter_name)
            adapter_scales.append(adapter.strength)
        self._set_validation_adapter_weights(pipeline, adapter_names, adapter_scales)
        try:
            yield
        finally:
            self._remove_validation_adapters(pipeline, adapter_names)

    def _set_validation_adapter_weights(self, pipeline, adapter_names: list[str], adapter_scales: list[float]):
        if not adapter_names:
            return
        if hasattr(pipeline, "set_adapters"):
            names = adapter_names if len(adapter_names) > 1 else adapter_names[0]
            scales = adapter_scales if len(adapter_scales) > 1 else adapter_scales[0]
            pipeline.set_adapters(names, scales)
        elif hasattr(pipeline, "set_adapter"):
            pipeline.set_adapter(adapter_names[0], adapter_scales[0])
        else:
            logger.warning("Pipeline does not expose set_adapters; using adapter defaults.")

    def _remove_validation_adapters(self, pipeline, adapter_names: list[str]):
        if not adapter_names:
            return
        if hasattr(pipeline, "delete_adapters"):
            names = adapter_names if len(adapter_names) > 1 else adapter_names[0]
            pipeline.delete_adapters(names)
        else:
            logger.warning("Could not delete temporary validation adapters: %s", adapter_names)
        self._assert_adapters_detached(pipeline, adapter_names)

    def _assert_adapters_detached(self, pipeline, adapter_names: list[str]):
        if pipeline is None or not adapter_names:
            return
        lingering: set[str] = set()
        components = []
        if hasattr(pipeline, "components") and isinstance(pipeline.components, dict):
            components.extend(pipeline.components.values())
        for attr in (
            "transformer",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "text_encoder_4",
            "controlnet",
            "unet",
        ):
            component = getattr(pipeline, attr, None)
            if component is not None:
                components.append(component)
        for module in components:
            config = getattr(module, "peft_config", None)
            if not isinstance(config, dict):
                continue
            for name in adapter_names:
                if name in config:
                    lingering.add(name)
        if lingering:
            raise RuntimeError(
                f"Failed to detach temporary validation adapters: {', '.join(sorted(lingering))}. "
                "Please ensure your pipeline supports adapter removal."
            )

    def _prepare_validation_work_items(self, content: list[Any] | None) -> list[_ValidationWorkItem]:
        if content is None:
            return []
        metadata_shortnames = []
        if isinstance(self.validation_prompt_metadata, dict):
            metadata_shortnames = self.validation_prompt_metadata.get("validation_shortnames", []) or []
        work_items: list[_ValidationWorkItem] = []
        for idx, entry in enumerate(content):
            prompt_text: Any = entry
            conditioning: Any = None
            shortname: str | None = None
            if isinstance(entry, tuple):
                if len(entry) == 3 and isinstance(entry[2], list):
                    shortname, prompt_text, conditioning = entry
                    if isinstance(conditioning, list) and len(conditioning) == 1:
                        conditioning = conditioning[0]
                elif len(entry) == 3 and isinstance(entry[2], Image.Image):
                    shortname, prompt_text, conditioning = entry
                elif len(entry) == 4 and isinstance(entry[3], Image.Image):
                    shortname, prompt_text, _, conditioning = entry
                else:
                    candidate_shortname = entry[0] if len(entry) > 0 else None
                    candidate_prompt = entry[1] if len(entry) > 1 else prompt_text
                    shortname = candidate_shortname if isinstance(candidate_shortname, str) else None
                    prompt_text = candidate_prompt
            if shortname is None:
                if idx < len(metadata_shortnames):
                    shortname = metadata_shortnames[idx]
                else:
                    shortname = f"validation_{idx}"
            if not isinstance(prompt_text, str):
                prompt_text = str(prompt_text)
            work_items.append(
                _ValidationWorkItem(
                    index=idx,
                    shortname=shortname,
                    prompt=prompt_text,
                    conditioning=conditioning,
                )
            )
        return work_items

    @staticmethod
    def _serialise_media(media: Any):
        if media is None:
            return {"type": "none"}
        if isinstance(media, list):
            return {
                "type": "sequence",
                "items": [Validation._serialise_media(item) for item in media],
            }
        if not isinstance(media, Image.Image):
            coerced = _coerce_validation_image_input(media)
            if isinstance(coerced, Image.Image):
                media = coerced
            else:
                raise TypeError(f"Unsupported media type for validation serialisation: {type(media)}")
        buffer = BytesIO()
        media.save(buffer, format="PNG")
        return {
            "type": "image",
            "data": buffer.getvalue(),
        }

    @staticmethod
    def _serialise_media_list(media_list: list[Any]) -> list[Any]:
        return [Validation._serialise_media(item) for item in media_list or []]

    @staticmethod
    def _deserialise_media(serialised: Any):
        if serialised is None:
            return None
        media_type = serialised.get("type")
        if media_type == "none":
            return None
        if media_type == "sequence":
            return [Validation._deserialise_media(item) for item in serialised.get("items", [])]
        if media_type == "image":
            data = serialised.get("data", b"")
            with BytesIO(data) as buffer:
                with Image.open(buffer) as image:
                    image.load()
                    return image.copy()
        raise ValueError(f"Unknown media type '{media_type}' in serialised validation payload.")

    @staticmethod
    def _deserialise_media_list(serialised_list: list[Any]) -> list[Any]:
        return [Validation._deserialise_media(item) for item in serialised_list or []]

    def _execute_validation_work_item(
        self,
        item: _ValidationWorkItem,
        *,
        decorated_shortname: str,
        validation_type: str | None,
    ) -> dict[str, Any]:
        (
            stitched_validation_images,
            checkpoint_validation_images,
            _ema_validation_images,
        ) = self.validate_prompt(
            item.prompt,
            decorated_shortname,
            item.conditioning,
            validation_type,
        )
        return {
            "index": item.index,
            "shortname": item.shortname,
            "decorated_shortname": decorated_shortname,
            "prompt": item.prompt,
            "stitched": self._serialise_media_list(stitched_validation_images.get(decorated_shortname, [])),
            "checkpoint": self._serialise_media_list(checkpoint_validation_images.get(decorated_shortname, [])),
        }

    def _apply_serialised_validation_result(
        self,
        *,
        payload: dict[str, Any],
        validation_images: dict,
        validation_type: str | None,
    ) -> None:
        decorated_shortname: str = payload["decorated_shortname"]
        prompt: str = payload["prompt"]
        stitched_results = self._deserialise_media_list(payload.get("stitched", []))
        checkpoint_results = self._deserialise_media_list(payload.get("checkpoint", []))
        self.validation_prompt_dict[decorated_shortname] = prompt
        logger.debug(f"Completed generating image: {prompt}")
        validation_images.setdefault(decorated_shortname, []).extend(stitched_results)
        if isinstance(self.model, VideoModelFoundation):
            self._save_videos(validation_images, decorated_shortname, prompt)
        else:
            self._save_images(validation_images, decorated_shortname, prompt)
        checkpoint_payload = {decorated_shortname: checkpoint_results}
        self.evaluation_result = self.evaluate_images(checkpoint_payload)
        self._log_validations_to_webhook(validation_images, decorated_shortname, prompt)

    def process_prompts(
        self,
        validation_type: str = None,
        adapter_run: ValidationAdapterRun | None = None,
        image_accumulator: dict | None = None,
    ):
        """Processes each validation prompt and logs the result."""
        self.evaluation_result = None
        if self.validation_prompt_dict is None:
            self.validation_prompt_dict = {}
        validation_images = image_accumulator if image_accumulator is not None else {}
        _content = self.validation_prompt_metadata.get("validation_prompts", []) if self.validation_prompt_metadata else []
        total_samples = len(_content) if _content is not None else 0
        if self.validation_image_inputs:
            # Override the pipeline inputs to be entirely based upon the validation image inputs.
            _content = self.validation_image_inputs
            if "DeepFloyd" in self.model.NAME:
                resize_edge_length = 64
                # Resize validation input to 64px area
                _content = resize_validation_images(_content, resize_edge_length)
            total_samples = len(_content) if _content is not None else 0

        logger.debug(f"Processing content: {_content}")
        if total_samples == 0:
            logger.debug("No validation prompts to process.")
            return

        work_items = self._prepare_validation_work_items(_content)
        use_distributed = self._use_distributed_validation()
        num_processes = getattr(self.accelerator, "num_processes", 1)

        # Disable batch-parallel if we don't have enough prompts to meaningfully split
        # This avoids collective communication deadlocks when some processes get empty work
        if use_distributed and len(work_items) < num_processes:
            use_distributed = False
            logger.info(
                f"Disabling batch-parallel for validation: {len(work_items)} prompt(s) < {num_processes} processes. "
                f"Only main process will execute."
            )
            # When falling back to single-process, only main process should continue
            if not self.accelerator.is_main_process:
                return

        local_work_items = split_across_processes(self.accelerator, work_items) if use_distributed else work_items
        rank = getattr(self.accelerator, "process_index", 0)
        logger.info(
            f"[Rank {rank}] Processing {len(local_work_items)} local validation work items "
            f"(distributed_mode={use_distributed}, total={len(work_items)}, "
            f"work_item_prompts={[item.prompt for item in local_work_items]})"
        )
        progress_disable = use_distributed and not self.accelerator.is_main_process
        local_payloads: list[dict[str, Any]] = []
        for item in tqdm(
            local_work_items,
            desc="Processing validation prompts",
            total=len(local_work_items),
            leave=False,
            position=1,
            disable=progress_disable,
        ):
            decorated_shortname = self._decorate_shortname(item.shortname, adapter_run)
            self.validation_prompt_dict[decorated_shortname] = item.prompt
            logger.debug(f"validation prompt (shortname={decorated_shortname}): '{item.prompt}'")
            logger.debug(f"Processing validation for prompt: {item.prompt}")
            payload = self._execute_validation_work_item(
                item=item,
                decorated_shortname=decorated_shortname,
                validation_type=validation_type,
            )
            local_payloads.append(payload)

            # In non-distributed mode, apply results immediately after each prompt completes
            if not use_distributed:
                self._apply_serialised_validation_result(
                    payload=payload,
                    validation_images=validation_images,
                    validation_type=validation_type,
                )

        if use_distributed:
            logger.info(f"[Rank {rank}] Gathering {len(local_payloads)} local payloads")
            gathered_payloads = gather_across_processes(local_payloads)
            if not self.accelerator.is_main_process:
                return
            logger.info(
                f"[Rank {rank}] Gathered {len(gathered_payloads)} payload groups: {[len(p) for p in gathered_payloads]}"
            )
            aggregated_payloads = [payload for worker_payloads in gathered_payloads for payload in worker_payloads]
            logger.info(f"[Rank {rank}] Total aggregated payloads: {len(aggregated_payloads)}")

            aggregated_payloads.sort(key=lambda payload: payload["index"])
            for payload in aggregated_payloads:
                self._apply_serialised_validation_result(
                    payload=payload,
                    validation_images=validation_images,
                    validation_type=validation_type,
                )
        self.validation_images = validation_images
        if not use_distributed or self.accelerator.is_main_process:
            try:
                self._log_validations_to_trackers(validation_images)
            except Exception as e:
                logger.error(f"Error logging validation images: {e}")
                import traceback

                logger.error(traceback.format_exc())
            else:
                logger.debug("Validation tracker update complete.")

    def get_eval_result(self):
        return self.evaluation_result or {}

    def clear_eval_result(self):
        self.evaluation_result = None

    def stitch_three_images(
        self,
        left_image,
        middle_image,
        right_image,
        separator_width=5,
        labels=[None, None, None],
    ):
        """
        Stitch three images horizontally with proper spacing and optional labels.
        Handles different sizes by centering vertically.

        Args:
            left_image: First image (e.g., input)
            middle_image: Second image (e.g., output)
            right_image: Third image (e.g., benchmark)
            separator_width: Width of separator between images
            labels: Text labels for [left, middle, right] images
        """

        def _is_frame_sequence(media):
            return isinstance(media, list) and len(media) > 0 and all(isinstance(frame, Image.Image) for frame in media)

        def _ensure_sequence(media, name):
            if _is_frame_sequence(media):
                return list(media)
            if isinstance(media, Image.Image):
                return [media]
            raise TypeError(f"Unsupported media type for {name}: {type(media)}")

        def _pad_sequence(frames, target_len, name):
            if not frames:
                raise ValueError(f"Cannot stitch empty {name} sequence.")
            if len(frames) == target_len:
                return frames
            if len(frames) > target_len:
                return frames[:target_len]
            padding = [frames[-1]] * (target_len - len(frames))
            return frames + padding

        def _stitch_single_triplet(single_left, single_middle, single_right):
            left_width, left_height = single_left.size
            middle_width, middle_height = single_middle.size
            right_width, right_height = single_right.size

            new_width = left_width + separator_width + middle_width + separator_width + right_width
            new_height = max(left_height, middle_height, right_height)

            new_image = Image.new("RGB", (new_width, new_height), color="white")

            left_y = (new_height - left_height) // 2
            middle_y = (new_height - middle_height) // 2
            right_y = (new_height - right_height) // 2

            left_x = 0
            middle_x = left_width + separator_width
            right_x = middle_x + middle_width + separator_width

            new_image.paste(single_left, (left_x, left_y))
            new_image.paste(single_middle, (middle_x, middle_y))
            new_image.paste(single_right, (right_x, right_y))

            draw = ImageDraw.Draw(new_image)
            line_color = (200, 200, 200)
            for i in range(separator_width):
                x = left_width + i
                draw.line([(x, 0), (x, new_height)], fill=line_color)
            for i in range(separator_width):
                x = middle_x + middle_width + i
                draw.line([(x, 0), (x, new_height)], fill=line_color)

            font = get_font_for_labels()
            if len(labels) > 0 and labels[0] is not None:
                draw.text(
                    (left_x + 10, 10),
                    labels[0],
                    fill=(255, 255, 255),
                    font=font,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )

            if len(labels) > 1 and labels[1] is not None:
                draw.text(
                    (middle_x + 10, 10),
                    labels[1],
                    fill=(255, 255, 255),
                    font=font,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )

            if len(labels) > 2 and labels[2] is not None:
                draw.text(
                    (right_x + 10, 10),
                    labels[2],
                    fill=(255, 255, 255),
                    font=font,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )

            return new_image

        # if multi condition images ,we need concat they as left image firstly
        if isinstance(left_image, list) and not (_is_frame_sequence(middle_image) or _is_frame_sequence(right_image)):
            if all(isinstance(img, Image.Image) for img in left_image):
                widths, heights = zip(*(img.size for img in left_image))
                total_width = sum(widths)
                max_height = max(heights)
                new_image = Image.new("RGB", (total_width, max_height))
                x_offset = 0
                for img in left_image:
                    new_image.paste(img, (x_offset, 0))
                    x_offset += img.size[0]
                left_image = new_image
            else:
                logger.error(f"Condition in left_image are not all PIL image format")
        left_is_sequence = _is_frame_sequence(left_image)
        middle_is_sequence = _is_frame_sequence(middle_image)
        right_is_sequence = _is_frame_sequence(right_image)

        if left_is_sequence or middle_is_sequence or right_is_sequence:
            left_frames = _ensure_sequence(left_image, "left")
            middle_frames = _ensure_sequence(middle_image, "middle")
            right_frames = _ensure_sequence(right_image, "right")

            target_length = max(len(left_frames), len(middle_frames), len(right_frames))

            left_frames = _pad_sequence(left_frames, target_length, "left")
            middle_frames = _pad_sequence(middle_frames, target_length, "middle")
            right_frames = _pad_sequence(right_frames, target_length, "right")

            stitched_frames = []
            for idx in range(target_length):
                stitched_frames.append(_stitch_single_triplet(left_frames[idx], middle_frames[idx], right_frames[idx]))
            return stitched_frames if target_length > 1 else stitched_frames[0]

        return _stitch_single_triplet(left_image, middle_image, right_image)

    def stitch_conditioning_images(self, validation_image_results, conditioning_image):
        """
        For each image/video, make a new canvas and place conditioning image on the LEFT side.
        """
        stitched_results = []

        for idx, result in enumerate(validation_image_results):
            if isinstance(result, list):
                # It's a video - stitch conditioning image to each frame
                stitched_frames = [
                    _stitch_single_pair(conditioning_image, frame, separator_width=5, labels=None) for frame in result
                ]
                stitched_results.append(stitched_frames)
            else:
                # It's a single image
                stitched_results.append(_stitch_single_pair(conditioning_image, result, separator_width=5, labels=None))

        return stitched_results

    def stitch_validation_input_image(
        self,
        validation_image_result,
        validation_input_image,
        separator_width=5,
        labels=["input", "checkpoint"],
    ):
        """
        Stitch validation input image to the left of the validation output.
        Handles different sizes by centering vertically when heights differ.

        Args:
            validation_image_result: The generated validation image
            validation_input_image: The input/context image used for validation
            separator_width: Width of separator between images
            labels: Text labels for [left, right] images
        """
        if validation_input_image is None:
            return validation_image_result

        # Handle list of input images (for multi-input reference models)
        if isinstance(validation_input_image, list):
            # Create a composite image from the list - stack them horizontally
            total_height = max(img.size[1] for img in validation_input_image)
            total_width = sum(img.size[0] for img in validation_input_image) + separator_width * (
                len(validation_input_image) - 1
            )

            composite_input = Image.new("RGB", (total_width, total_height), color="white")
            x_offset = 0
            for i, img in enumerate(validation_input_image):
                # Center each image vertically if needed
                y_offset = (total_height - img.size[1]) // 2
                composite_input.paste(img, (x_offset, y_offset))
                x_offset += img.size[0]

                # Add separator between images (except after last one)
                if i < len(validation_input_image) - 1:
                    draw = ImageDraw.Draw(composite_input)
                    for j in range(separator_width):
                        draw.line(
                            [(x_offset + j, 0), (x_offset + j, total_height)],
                            fill=(200, 200, 200),
                        )
                    x_offset += separator_width

            validation_input_image = composite_input
            labels[0] = "inputs" if labels[0] == "input" else labels[0]  # Pluralize

        input_width, input_height = validation_input_image.size
        output_width, output_height = validation_image_result.size

        # Calculate new canvas dimensions
        new_width = input_width + separator_width + output_width
        new_height = max(input_height, output_height)

        # Create new image with white background
        new_image = Image.new("RGB", (new_width, new_height), color="white")

        # Calculate vertical positions for centering if heights differ
        input_y_offset = (new_height - input_height) // 2 if input_height < new_height else 0
        output_y_offset = (new_height - output_height) // 2 if output_height < new_height else 0

        # Paste input image on the left
        new_image.paste(validation_input_image, (0, input_y_offset))

        # Paste output image on the right
        new_image.paste(validation_image_result, (input_width + separator_width, output_y_offset))

        # Create drawing object for text and separator
        draw = ImageDraw.Draw(new_image)

        # Use a default font
        font = get_font_for_labels()

        # Add text labels if provided
        if labels[0] is not None:
            draw.text(
                (10, 10),
                labels[0],
                fill=(255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

        if labels[1] is not None:
            draw.text(
                (input_width + separator_width + 10, 10),
                labels[1],
                fill=(255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

        # Draw vertical separator line
        line_color = (200, 200, 200)  # Light gray
        for i in range(separator_width):
            x = input_width + i
            draw.line([(x, 0), (x, new_height)], fill=line_color)

        return new_image

    def _validation_types(self):
        types = ["checkpoint"]
        if self.config.use_ema:
            # ema has different validations we can add or overwrite.
            if self.config.ema_validation == "ema_only":
                # then we do not sample the base ckpt being trained, only the EMA weights.
                types = ["ema"]
            if self.config.ema_validation == "comparison":
                # then we sample both.
                types.append("ema")

        return types

    def validate_prompt(
        self,
        prompt,
        validation_shortname,
        validation_input_image=None,
        validation_type=None,
    ):
        """Generate validation images for a single prompt."""
        # Placeholder for actual image generation and logging
        logger.debug(f"Validating ({validation_shortname}) prompt: {prompt}")
        # benchmarked / stitched validation images
        stitched_validation_images = {}
        # untouched / un-stitched validation images
        checkpoint_validation_images = {}
        ema_validation_images = {}
        benchmark_image = None
        for resolution in self.validation_resolutions:
            extra_validation_kwargs = {}
            if validation_input_image is not None:
                extra_validation_kwargs["image"] = validation_input_image
                if self.deepfloyd_stage2:
                    # deepfloyd-if stage 2 requires 4x size declaration on the outputs.
                    # the input is 64px edge len.
                    validation_resolution_width, validation_resolution_height = (
                        val * 4 for val in extra_validation_kwargs["image"].size
                    )
                elif (
                    self.config.controlnet
                    or self.config.validation_using_datasets
                    or self.model.requires_conditioning_validation_inputs()
                ):
                    # for most conditioned models, we want the input size to remain.
                    validation_image_edge_len = self.model.validation_image_input_edge_length()
                    if validation_image_edge_len is not None:
                        # calculate the megapixels value (eg ~0.25 for 512px)
                        validation_image_megapixels = (validation_image_edge_len**2) / 1_000_000
                        validation_resolution, _, validation_aspect_ratio = (
                            MultiaspectImage.calculate_new_size_by_pixel_area(
                                aspect_ratio=MultiaspectImage.calculate_image_aspect_ratio(extra_validation_kwargs["image"]),
                                megapixels=validation_image_megapixels,
                                original_size=extra_validation_kwargs["image"].size,
                            )
                        )
                        extra_validation_kwargs["image"] = extra_validation_kwargs["image"].resize(
                            validation_resolution, Image.Resampling.LANCZOS
                        )
                        validation_resolution_width, validation_resolution_height = validation_resolution
                    else:
                        if isinstance(extra_validation_kwargs["image"], list):
                            (
                                validation_resolution_width,
                                validation_resolution_height,
                            ) = resolution
                        else:
                            (
                                validation_resolution_width,
                                validation_resolution_height,
                            ) = extra_validation_kwargs["image"].size
                else:
                    raise ValueError("Validation input images are not supported for this model type.")
                extra_validation_kwargs["control_image"] = extra_validation_kwargs["image"]
            else:
                validation_resolution_width, validation_resolution_height = resolution

            if type(self.config.validation_guidance_skip_layers) is list:
                extra_validation_kwargs["skip_layer_guidance_start"] = float(
                    self.config.validation_guidance_skip_layers_start
                )
                extra_validation_kwargs["skip_layer_guidance_stop"] = float(self.config.validation_guidance_skip_layers_stop)
                extra_validation_kwargs["skip_layer_guidance_scale"] = float(self.config.validation_guidance_skip_scale)
                extra_validation_kwargs["skip_guidance_layers"] = list(self.config.validation_guidance_skip_layers)

            extra_validation_kwargs["guidance_rescale"] = self.config.validation_guidance_rescale

            if StateTracker.get_args().validation_using_datasets:
                extra_validation_kwargs["strength"] = getattr(self.config, "validation_strength", 0.2)
                logger.debug(f"Set validation image denoise strength to {extra_validation_kwargs['strength']}")

            logger.debug(f"Processing width/height: {validation_resolution_width}x{validation_resolution_height}")
            if validation_shortname not in stitched_validation_images:
                stitched_validation_images[validation_shortname] = []
                checkpoint_validation_images[validation_shortname] = []
                ema_validation_images[validation_shortname] = []
            try:
                _embed = self._gather_prompt_embeds(prompt)
                if _embed is not None:
                    extra_validation_kwargs.update(_embed)
                else:
                    extra_validation_kwargs["prompt"] = prompt
            except Exception as e:
                import traceback

                logger.error(
                    f"Error gathering text embed for validation prompt {prompt}: {e}, traceback: {traceback.format_exc()}"
                )
                continue

            try:
                pipeline_kwargs = {
                    "prompt": None,
                    "negative_prompt": None,
                    "num_images_per_prompt": self.config.num_validation_images,
                    "num_inference_steps": self.config.validation_num_inference_steps,
                    "guidance_scale": self.config.validation_guidance,
                    "height": MultiaspectImage._round_to_nearest_multiple(int(validation_resolution_height), 16),
                    "width": MultiaspectImage._round_to_nearest_multiple(int(validation_resolution_width), 16),
                    **extra_validation_kwargs,
                }
                if self.model.VALIDATION_USES_NEGATIVE_PROMPT:
                    if StateTracker.get_args().validation_negative_prompt is None:
                        StateTracker.get_args().validation_negative_prompt = ""
                    _negative_embed = self.embed_cache.compute_embeddings_for_prompts(
                        [StateTracker.get_args().validation_negative_prompt],
                        is_validation=True,
                        load_from_cache=True,
                    )
                    if _negative_embed is not None:
                        negative_embed_data = {
                            k: (
                                v.to(
                                    device=self.inference_device,
                                    dtype=self.config.weight_dtype,
                                )
                                if hasattr(v, "to")
                                else v
                            )
                            for k, v in _negative_embed.items()
                        }
                        pipeline_kwargs.update(
                            self.model.convert_negative_text_embed_for_pipeline(
                                prompt=StateTracker.get_args().validation_negative_prompt,
                                text_embedding=negative_embed_data,
                            )
                        )
                    else:
                        pipeline_kwargs["negative_prompt"] = StateTracker.get_args().validation_negative_prompt
                # TODO: Refactor the rest so that it uses model class to update kwargs more generally.
                if self.config.validation_guidance_real > 1.0:
                    pipeline_kwargs["guidance_scale_real"] = float(self.config.validation_guidance_real)
                if isinstance(self.config.validation_no_cfg_until_timestep, int) and self.config.model_family == "flux":
                    pipeline_kwargs["no_cfg_until_timestep"] = self.config.validation_no_cfg_until_timestep

                pipeline_kwargs = self.model.update_pipeline_call_kwargs(pipeline_kwargs)
                logger.debug(f"Image being generated with parameters: {pipeline_kwargs}")
                if self.config.model_family == "sana":
                    pipeline_kwargs["complex_human_instruction"] = self.config.sana_complex_human_instruction

                validation_types = self._validation_types()
                all_validation_type_results = {}
                for current_validation_type in validation_types:
                    if not self.config.validation_randomize:
                        pipeline_kwargs["generator"] = self._get_generator()
                        logger.debug(f"Using a generator? {pipeline_kwargs['generator']}")
                    if current_validation_type == "ema":
                        self.enable_ema_for_inference()
                    pipeline_kwargs = {
                        k: (
                            v.to(
                                device=self.inference_device,
                                dtype=self.config.weight_dtype,
                            )
                            if hasattr(v, "to") and v.dtype in (torch.bfloat16, torch.float16, torch.float32)
                            else v
                        )
                        for k, v in pipeline_kwargs.items()
                    }

                    call_kwargs = inspect.signature(self.model.pipeline.__call__).parameters
                    logger.debug(f"Possible parameters for {type(self.model.pipeline)}: {call_kwargs}")
                    # remove any kwargs that are not in the pipeline call
                    pipeline_kwargs = {k: v for k, v in pipeline_kwargs.items() if k in call_kwargs}
                    removed_kwargs = [k for k in pipeline_kwargs.keys() if k not in call_kwargs]
                    logger.debug(f"Running validations with inputs: {pipeline_kwargs.keys()}")
                    if removed_kwargs:
                        logger.warning(f"Removed the following kwargs from validation pipeline: {removed_kwargs}")
                    # run in autocast ctx
                    preview_ctx = nullcontext()
                    if self.preview and current_validation_type == "checkpoint":
                        preview_ctx = self.preview.attach(
                            self.model.pipeline,
                            pipeline_kwargs,
                            _PreviewMetadata(
                                shortname=validation_shortname,
                                prompt=prompt,
                                resolution=(int(validation_resolution_width), int(validation_resolution_height)),
                                validation_type=validation_type,
                                total_steps=getattr(self.config, "validation_num_inference_steps", None),
                            ),
                        )
                    with preview_ctx:
                        with torch.amp.autocast(
                            self.inference_device.type,
                            dtype=self.config.weight_dtype,
                        ):
                            pipeline_result = self.model.pipeline(**pipeline_kwargs)
                        if hasattr(pipeline_result, "frames"):
                            all_validation_type_results[current_validation_type] = pipeline_result.frames
                        elif hasattr(pipeline_result, "images"):
                            all_validation_type_results[current_validation_type] = pipeline_result.images
                        else:
                            logger.error(f"Pipeline result does not have 'frames' or 'images': {pipeline_result}")
                            all_validation_type_results[current_validation_type] = []
                    if current_validation_type == "ema":
                        self.disable_ema_for_inference()

                # Keep the original unstitched results for checkpoint storage and benchmark comparison
                # Retrieve the default image result for stitching
                ema_image_results = all_validation_type_results.get("ema")
                validation_image_results = all_validation_type_results.get("checkpoint", ema_image_results)
                original_validation_image_results = validation_image_results
                display_validation_results = validation_image_results.copy()

                # Store EMA results separately
                if self.config.use_ema and ema_image_results is not None:
                    if ema_validation_images[validation_shortname] is None:
                        ema_validation_images[validation_shortname] = []
                    ema_validation_images[validation_shortname].extend(ema_image_results)

                if validation_type != "base_model":
                    # The logic flow is:
                    # 1. If controlnet/control: stitch conditioning image to the left
                    # 2. Otherwise check for input stitching + benchmark:
                    #    - Both: create [input | output | benchmark] with labels only on outer images
                    #    - Input only: create [input | output] with labels
                    #    - Benchmark only: create [output | benchmark] with labels
                    # 3. Original images are always preserved for checkpoint storage

                    # First, check if we need validation input stitching
                    has_input_stitching = (
                        hasattr(self.config, "validation_stitch_input_location")
                        and self.config.validation_stitch_input_location == "left"
                        and validation_input_image is not None
                    )

                    # Check if we'll be adding benchmark
                    will_add_benchmark = False
                    benchmark_image = None
                    if not self.config.disable_benchmark and self.benchmark_exists("base_model"):
                        benchmark_image = self._benchmark_image(validation_shortname, resolution)
                        will_add_benchmark = benchmark_image is not None

                    if has_input_stitching and not will_add_benchmark:
                        # Only input stitching, no benchmark
                        display_validation_results = [
                            self.stitch_validation_input_image(
                                validation_image_result=img,
                                validation_input_image=validation_input_image,
                                labels=(["input", f"step {StateTracker.get_global_step()}"]),
                            )
                            for img in display_validation_results
                        ]

                    # Apply controlnet stitching if needed (using original results)
                    if any([self.config.controlnet, self.config.control]):
                        # For controlnet, we stitch to the original results
                        display_validation_results = self.stitch_conditioning_images(
                            original_validation_image_results,
                            extra_validation_kwargs["control_image"],
                        )

                    # Apply benchmark stitching if we determined we have a benchmark
                    if will_add_benchmark:
                        if has_input_stitching:
                            # Three-way stitch: input | output | benchmark
                            for idx, original_img in enumerate(original_validation_image_results):
                                labels_to_use = [
                                    "input",
                                    "base model",
                                    f"step {StateTracker.get_global_step()}",
                                ]

                                display_validation_results[idx] = self.stitch_three_images(
                                    left_image=validation_input_image,
                                    middle_image=benchmark_image,
                                    right_image=original_img,
                                    labels=labels_to_use,
                                )
                        else:
                            # No input stitching, just stitch benchmark to output
                            for idx, original_img in enumerate(original_validation_image_results):
                                display_validation_results[idx] = self.stitch_benchmark_image(
                                    validation_image_result=original_img,
                                    benchmark_image=benchmark_image,
                                    labels=[
                                        "base model",
                                        f"step {StateTracker.get_global_step()}",
                                    ],
                                )

                    # Handle EMA comparison stitching
                    if self.config.use_ema and self.config.ema_validation == "comparison" and ema_image_results is not None:
                        # Create new display results that include EMA comparison
                        ema_display_results = []
                        for idx, display_img in enumerate(display_validation_results):
                            # Get the original EMA result
                            ema_img = ema_image_results[idx]

                            # Stitch EMA to the already-stitched display image
                            ema_stitched = self.stitch_benchmark_image(
                                validation_image_result=ema_img,
                                benchmark_image=display_img,
                                labels=[None, "EMA"],
                            )
                            ema_display_results.append(ema_stitched)

                        # Replace display results with EMA comparison results
                        display_validation_results = ema_display_results

                    # Add the validation prompt to the bottom of the entire image using a decent looking font, in a margin, centred width-wise.
                    # Scan for fonts to use available.
                    font = get_font_for_labels()
                    if font is not None:
                        # Add the validation prompt text to the bottom of each image
                        for idx, validation_result in enumerate(display_validation_results):
                            display_validation_results[idx] = draw_text_on_image(
                                validation_result, f"Prompt: {prompt}", font=font
                            )

                # Use original results for checkpoint storage, display results for viewing
                checkpoint_validation_images[validation_shortname].extend(original_validation_image_results)
                stitched_validation_images[validation_shortname].extend(display_validation_results)

            except Exception as e:
                import traceback

                logger.error(f"Error generating validation image: {e}, {traceback.format_exc()}")
                continue

        return (
            stitched_validation_images,
            checkpoint_validation_images,
            ema_validation_images,
        )

    def _save_videos(self, validation_images, validation_shortname, validation_prompt):
        validation_img_idx = 0
        from diffusers.utils.export_utils import export_to_video

        video_paths: list[str] = []
        for validation_image in validation_images[validation_shortname]:
            # Get the validation resolution for this index
            if validation_img_idx < len(self.validation_resolutions):
                resolution = self.validation_resolutions[validation_img_idx]
                if isinstance(resolution, str):
                    if "x" in resolution:
                        res_label = resolution
                    else:
                        res_label = f"{resolution}x{resolution}"
                elif isinstance(resolution, tuple):
                    res_label = f"{resolution[0]}x{resolution[1]}"
                else:
                    res_label = f"{resolution}x{resolution}"
            else:
                # Fallback to actual size if somehow out of bounds
                logger.warning(f"Image index {validation_img_idx} exceeds validation resolutions list")
                if type(validation_image) is list:
                    size_x, size_y = validation_image[0].size
                else:
                    size_x, size_y = validation_image.size
                res_label = f"{size_x}x{size_y}"

            # convert array of numpy to array of pil:
            validation_image = MultiaspectImage.numpy_list_to_pil(validation_image)
            if type(validation_image) is not list:
                # save as single image instead
                validation_image.save(
                    os.path.join(
                        self.save_dir,
                        f"step_{StateTracker.get_global_step()}_{validation_shortname}_{validation_img_idx}_{res_label}.png",
                    )
                )
                validation_img_idx += 1
                continue

            video_path = os.path.join(
                self.save_dir,
                f"step_{StateTracker.get_global_step()}_{validation_shortname}_{validation_img_idx}_{res_label}.mp4",
            )
            export_to_video(
                validation_image,
                video_path,
                fps=self.config.framerate,
            )
            video_paths.append(video_path)
            validation_img_idx += 1
        if video_paths:
            self.validation_video_paths[validation_shortname] = video_paths

    def _save_images(self, validation_images, validation_shortname, validation_prompt):
        validation_img_idx = 0
        resolutions = [_res for res in self.validation_resolutions for _res in [res] * self.config.num_eval_images]
        for validation_image in validation_images[validation_shortname]:
            res = resolutions[validation_img_idx]
            if "x" in res:
                res_label = str(res)
            elif type(res) is tuple:
                res_label = f"{res[0]}x{res[1]}"
            else:
                res_label = f"{res}x{res}"
            validation_image.save(
                os.path.join(
                    self.save_dir,
                    f"step_{StateTracker.get_global_step()}_{validation_shortname}_{validation_img_idx}_{res_label}.png",
                )
            )
            validation_img_idx += 1

    def _log_validations_to_webhook(self, validation_images, validation_shortname, validation_prompt):
        webhook_handler = StateTracker.get_webhook_handler()
        if webhook_handler is None:
            return

        media_word = "video" if isinstance(self.model, VideoModelFoundation) else "image"
        message = (
            f"Validation {media_word} for `{validation_shortname if validation_shortname != '' else '(blank shortname)'}`"
            f"\nValidation prompt: `{validation_prompt if validation_prompt != '' else '(blank prompt)'}`"
            f"\nEvaluation score: {self.eval_scores.get(validation_shortname, 'N/A')}"
        )

        images_payload = validation_images.get(validation_shortname, [])
        videos_for_discord = None
        videos_for_raw = None

        if isinstance(self.model, VideoModelFoundation):
            video_paths = self.validation_video_paths.get(validation_shortname, [])
            if video_paths:
                videos_for_discord = []
                videos_for_raw = []
                for path in video_paths:
                    try:
                        with open(path, "rb") as handle:
                            video_bytes = handle.read()
                    except Exception as exc:  # pragma: no cover - defensive logging
                        logger.warning("Failed to read validation video %s: %s", path, exc)
                        continue
                    video_buffer = BytesIO(video_bytes)
                    video_buffer.name = os.path.basename(path)
                    videos_for_discord.append(video_buffer)
                    data_uri = f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode('utf-8')}"
                    videos_for_raw.append({"src": data_uri, "mime_type": "video/mp4"})
                if videos_for_discord:
                    images_payload = None

        webhook_handler.send(
            message,
            images=images_payload,
            videos=videos_for_discord,
        )

        webhook_handler.send_raw(
            structured_data={"message": f"Validation: {validation_shortname}"},
            message_type="training.validation",
            message_level="info",
            job_id=StateTracker.get_job_id(),
            images=images_payload,
            videos=videos_for_raw,
        )

    def _log_validations_to_trackers(self, validation_images):
        for tracker in self.accelerator.trackers:
            if tracker.name == "comet_ml":
                experiment = self.accelerator.get_tracker("comet_ml").tracker
                for shortname, image_list in validation_images.items():
                    for idx, image in enumerate(image_list):
                        experiment.log_image(
                            image,
                            name=f"{shortname} - {self.validation_resolutions[idx]}",
                        )
            elif tracker.name == "tensorboard":
                tracker = self.accelerator.get_tracker("tensorboard")
                for shortname, image_list in validation_images.items():
                    tracker.log_images(
                        {
                            f"{shortname} - {self.validation_resolutions[idx]}": np.moveaxis(np.array(image), -1, 0)[
                                np.newaxis, ...
                            ]
                            for idx, image in enumerate(image_list)
                        },
                        step=StateTracker.get_global_step(),
                    )
            elif tracker.name == "wandb":
                resolution_list = [f"{res[0]}x{res[1]}" for res in get_validation_resolutions()]

                if self.config.tracker_image_layout == "table":
                    columns = [
                        "Prompt",
                        *resolution_list,
                        "Mean Luminance",
                    ]
                    table = wandb.Table(columns=columns)

                    # Process each prompt and its associated images
                    for prompt_shortname, image_list in validation_images.items():
                        wandb_images = []
                        luminance_values = []
                        logger.debug(f"Prompt {prompt_shortname} has {len(image_list)} images")
                        for image in image_list:
                            logger.debug(f"Adding to table: {image}")
                            wandb_image = wandb.Image(image)
                            wandb_images.append(wandb_image)
                            luminance = calculate_luminance(image)
                            luminance_values.append(luminance)
                        mean_luminance = torch.tensor(luminance_values).mean().item()
                        while len(wandb_images) < len(resolution_list):
                            # any missing images will crash it. use None so they are indexed.
                            logger.debug("Found a missing image - masking with a None")
                            wandb_images.append(None)
                        table.add_data(prompt_shortname, *wandb_images, mean_luminance)

                    # Log the table to Weights & Biases
                    tracker.log(
                        {"Validation Gallery": table},
                        step=StateTracker.get_global_step(),
                    )

                elif self.config.tracker_image_layout == "gallery":
                    gallery_images = {}
                    for prompt_shortname, image_list in validation_images.items():
                        logger.debug(f"Prompt {prompt_shortname} has {len(image_list)} images")
                        for idx, image in enumerate(image_list):
                            # if it's a list of images, make a grid
                            if isinstance(image, list) and isinstance(image[0], Image.Image):
                                image = image[0]
                            wandb_image = wandb.Image(
                                image,
                                caption=f"{prompt_shortname} - {resolution_list[idx]}",
                            )
                            gallery_images[f"{prompt_shortname} - {resolution_list[idx]}"] = wandb_image

                    # Log all images in one call to prevent the global step from ticking
                    tracker.log(gallery_images, step=StateTracker.get_global_step())

    def enable_ema_for_inference(self, pipeline=None):
        if self.ema_enabled:
            logger.debug("EMA already enabled. Not enabling EMA.")
            return
        if self.config.use_ema:
            logger.debug("Enabling EMA.")
            self.ema_enabled = True
            if self.config.model_type == "lora":
                if self.config.lora_type.lower() == "lycoris":
                    logger.debug("Setting Lycoris multiplier to 1.0")
                    self.accelerator._lycoris_wrapped_network.set_multiplier(1.0)
                    logger.debug("Storing Lycoris weights for later recovery.")
                    self.ema_model.store(self.accelerator._lycoris_wrapped_network.parameters())
                    logger.debug("Storing the EMA weights into the Lycoris adapter for inference.")
                    self.ema_model.copy_to(self.accelerator._lycoris_wrapped_network.parameters())
                elif self.config.lora_type.lower() == "standard":
                    _trainable_parameters = [
                        x for x in self.model.get_trained_component(unwrap_model=False).parameters() if x.requires_grad
                    ]
                    self.ema_model.store(_trainable_parameters)
                    self.ema_model.copy_to(_trainable_parameters)
            else:
                logger.debug("Storing EMA weights for later recovery.")
                self.ema_model.store(self.trainable_parameters())
                logger.debug("Storing the EMA weights into the model for inference.")
                self.ema_model.copy_to(self.trainable_parameters())
        else:
            logger.debug("Skipping EMA model setup for validation, as we are not using EMA.")

    def disable_ema_for_inference(self):
        if not self.ema_enabled:
            logger.debug("EMA was not enabled. Not disabling EMA.")
            return
        if self.config.use_ema:
            logger.debug("Disabling EMA.")
            self.ema_enabled = False
            if self.config.model_type == "lora" and self.config.lora_type.lower() == "lycoris":
                logger.debug("Setting Lycoris network multiplier to 1.0.")
                self.accelerator._lycoris_wrapped_network.set_multiplier(1.0)
                logger.debug("Restoring Lycoris weights.")
                self.ema_model.restore(self.accelerator._lycoris_wrapped_network.parameters())
            else:
                logger.debug("Restoring trainable parameters.")
                self.ema_model.restore(self.trainable_parameters())
            if self.config.ema_device != "accelerator":
                logger.debug("Moving EMA weights to CPU for storage.")
                self.ema_model.to(self.config.ema_device)
                self.model.get_trained_component(unwrap_model=False).to(self.inference_device)

        else:
            logger.debug("Skipping EMA model restoration for validation, as we are not using EMA.")

    def finalize_validation(self, validation_type):
        """Cleans up and restores original state if necessary."""
        if not self.config.keep_vae_loaded and not self.config.vae_cache_ondemand:
            self.model.unload_vae()
        self.model.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate_images(self, images: list = None):
        if self.model_evaluator is None:
            return None
        for shortname, image_list in images.items():
            if shortname in self.eval_scores:
                continue
            prompt = self.validation_prompt_dict.get(shortname, "")
            for image in image_list:
                evaluation_score = self.model_evaluator.evaluate([image], [prompt])
                self.eval_scores[shortname] = round(float(evaluation_score), 4)
        if len(self.eval_scores) == 0:
            logger.warning("No evaluation scores were calculated. Please check your evaluation settings.")
            return {}
        # Log the scores into dict: {"min", "max", "mean", "std"}
        result = {
            "clip/min": min(self.eval_scores.values()),
            "clip/max": max(self.eval_scores.values()),
            "clip/mean": np.mean(list(self.eval_scores.values())),
            "clip/std": np.std(list(self.eval_scores.values())),
        }

        return result


class Evaluation:
    """
    A class for running eval loss calculations on prepared batches..
    """

    def __init__(self, accelerator):
        self.config = StateTracker.get_args()
        self.accelerator = accelerator

    def would_evaluate(self, training_state: dict):
        if not self.accelerator.is_main_process:
            return
        if self.config.eval_steps_interval is None:
            return False
        if self.config.eval_steps_interval == 0:
            return False
        if (
            training_state["global_step"] % self.config.eval_steps_interval == 0
            and training_state["global_step"] > training_state["global_resume_step"]
        ):
            return True

        return False

    def total_eval_batches(self, dataset_name=None):
        """
        Return the total number of eval batches across:
          - all eval datasets if dataset_name is None
          - the specific dataset if dataset_name is given
        """
        eval_datasets = StateTracker.get_data_backends(_type="eval", _types=None)
        if dataset_name is not None:
            ds = eval_datasets.get(dataset_name)
            return len(ds["sampler"]) if ds else 0
        return sum(len(x["sampler"]) for _, x in eval_datasets.items())

    def get_timestep_schedule(self, noise_scheduler):
        accept_mu = "mu" in set(inspect.signature(noise_scheduler.set_timesteps).parameters.keys())
        scheduler_kwargs = {}
        if accept_mu and self.config.flow_schedule_auto_shift:
            from simpletuner.helpers.models.sd3.pipeline import calculate_shift

            scheduler_kwargs["mu"] = calculate_shift(StateTracker.get_model().get_trained_component().config.max_seq)

        noise_scheduler.set_timesteps(self.config.eval_timesteps, **scheduler_kwargs)
        timesteps = noise_scheduler.timesteps
        return timesteps

    def _evaluate_dataset_pass(
        self,
        dataset_name,
        prepare_batch,
        model_predict,
        calculate_loss,
        get_prediction_target,
        noise_scheduler,
    ):
        """
        Evaluate on exactly one dataset (if dataset_name is not None),
        or across *all* eval datasets (if dataset_name is None).

        Returns a dictionary: {
            timestep_value -> [list of losses at that timestep]
        }
        """
        if not self.accelerator.is_main_process:
            return {}

        accumulated_eval_losses = {}
        eval_batch = True
        evaluated_sample_count = 0

        # Figure out how many total batches for this pass
        total_batches = self.total_eval_batches(dataset_name=dataset_name)
        if self.config.num_eval_images is not None:
            total_batches = min(self.config.num_eval_images, total_batches)

        main_progress_bar = tqdm(
            total=total_batches,
            desc=f"Calculate validation loss ({dataset_name or 'ALL'})",
            position=0,
            leave=True,
        )

        # Save and restore RNG states so that eval doesn't disturb training RNG
        cpu_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()

        eval_timestep_list = self.get_timestep_schedule(noise_scheduler)
        logger.debug(f"Evaluation timesteps: {eval_timestep_list}")

        while eval_batch is not False and evaluated_sample_count < total_batches:
            try:
                evaluated_sample_count += 1
                if self.config.num_eval_images is not None and evaluated_sample_count > self.config.num_eval_images:
                    reset_eval_datasets()
                    raise MultiDatasetExhausted("Max eval samples reached, resetting evaluations.")
                # Pass the dataset_name so we fetch from the correct place
                eval_batch = retrieve_eval_images(dataset_name=dataset_name)

            except MultiDatasetExhausted:
                logger.info(f"Evaluation loss calculation completed for dataset: {dataset_name}")
                eval_batch = False

            if eval_batch is not None and eval_batch is not False:
                # Fix a known seed so noise is consistent across eval
                torch.manual_seed(0)
                prepared_eval_batch = prepare_batch(eval_batch)
                if "latents" not in prepared_eval_batch:
                    raise ValueError("Error calculating eval batch: no 'latents' found.")

                bsz = prepared_eval_batch["latents"].shape[0]
                sample_text_str = "samples" if bsz > 1 else "sample"

                with torch.no_grad():
                    for eval_timestep in tqdm(
                        eval_timestep_list,
                        total=len(eval_timestep_list),
                        desc=f"Evaluating batch of {bsz} {sample_text_str}",
                        position=1,
                        leave=False,
                    ):
                        if eval_timestep not in accumulated_eval_losses:
                            accumulated_eval_losses[eval_timestep] = []

                        torch.manual_seed(0)
                        current_eval_timestep_tensor = (
                            torch.Tensor([eval_timestep])
                            .expand(prepared_eval_batch["noisy_latents"].shape[0])
                            .to(
                                dtype=self.config.weight_dtype,
                                device=prepared_eval_batch["noisy_latents"].device,
                            )
                        )
                        eval_prediction = model_predict(
                            prepared_batch=prepared_eval_batch,
                            custom_timesteps=current_eval_timestep_tensor,
                        )
                        eval_loss = calculate_loss(
                            prepared_batch=prepared_eval_batch,
                            model_output=eval_prediction,
                            apply_conditioning_mask=False,
                        )
                        accumulated_eval_losses[eval_timestep].append(eval_loss)

                    main_progress_bar.update(1)

        try:
            reset_eval_datasets()
        except:
            pass

        # Restore RNG
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)

        return accumulated_eval_losses

    def execute_eval(
        self,
        prepare_batch,
        model_predict,
        calculate_loss,
        get_prediction_target,
        noise_scheduler,
    ):
        """
        Either run a pooled pass (all eval datasets at once) or
        run each dataset individually (and also produce a pooled result).
        """
        if not self.accelerator.is_main_process:
            return {}

        # Decide if we pool or do separate passes
        pooling = getattr(self.config, "eval_dataset_pooling")
        eval_datasets = StateTracker.get_data_backends(_type="eval", _types=None)  # dict of {name: ...}
        if len(eval_datasets) == 0:
            return {}

        if pooling:
            # Single pass across ALL eval datasets
            logger.debug("Running a single pooled eval pass across all datasets.")
            pooled_losses = self._evaluate_dataset_pass(
                dataset_name=None,
                prepare_batch=prepare_batch,
                model_predict=model_predict,
                calculate_loss=calculate_loss,
                get_prediction_target=get_prediction_target,
                noise_scheduler=noise_scheduler,
            )
            # We'll store everything under a "pooled" key for consistency
            accumulated = {"pooled": pooled_losses}

        else:
            # Multiple passes: one per dataset
            logger.debug("Running separate eval passes for each dataset + pooled results.")
            accumulated = {}
            # We'll also keep an aggregator for the final 'pooled' pass
            from collections import defaultdict

            pooled_collector = defaultdict(list)

            for ds_name in eval_datasets.keys():
                ds_losses = self._evaluate_dataset_pass(
                    dataset_name=ds_name,
                    prepare_batch=prepare_batch,
                    model_predict=model_predict,
                    calculate_loss=calculate_loss,
                    get_prediction_target=get_prediction_target,
                    noise_scheduler=noise_scheduler,
                )
                accumulated[ds_name] = ds_losses

                # Collect them into the global "pooled" aggregator
                for tstep, losses in ds_losses.items():
                    pooled_collector[tstep].extend(losses)

        return accumulated

    def generate_tracker_table(self, all_accumulated_losses: dict):
        """
        all_accumulated_losses is expected to be:
        {
          dataset_name_1: { timestep_1: [losses], timestep_2: [losses], ... },
          dataset_name_2: { ... },
          ...
          "pooled": { timestep_x: [losses], ... }
        }

        If config.eval_dataset_pooling = True, then typically you'll only see a "pooled" key.
        If config.eval_dataset_pooling = False, you'll see multiple datasets plus "pooled".
        """
        if not self.accelerator.is_main_process:
            return {}
        if all_accumulated_losses == {} or all_accumulated_losses is None:
            return {}

        results = {}

        # Helper to flatten timesteps->loss arrays into a single (ts, loss) table
        def flatten_timestep_losses(timestep_dict):
            data_rows = []
            for ts, loss_list in timestep_dict.items():
                for loss_val in loss_list:
                    data_rows.append((ts, loss_val))
            return data_rows

        logger.debug("Generating evaluation tracker tables...")
        for ds_name, timestep_dict in all_accumulated_losses.items():
            data_rows = flatten_timestep_losses(timestep_dict)
            if not data_rows:
                continue
            total_loss = sum(x[1] for x in data_rows)
            num_items = len(data_rows)
            mean_loss = total_loss / num_items

            # By default, store a minimal result
            results_key = f"loss/val/{ds_name}"
            results[results_key] = mean_loss

            if self.config.report_to == "wandb":
                # Create a small wandb table for these data
                table = wandb.Table(data=data_rows, columns=["timestep", "eval_loss"])
                chart = wandb.plot.line(
                    table,
                    x="timestep",
                    y="eval_loss",
                    title=f"{results_key} by timestep",
                )
                results[f"chart/{results_key}"] = chart

        return results
