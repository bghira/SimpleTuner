import torch
import diffusers
import os
import wandb
import logging
import inspect
import sys
import numpy as np
from tqdm import tqdm
from helpers.training.wrappers import unwrap_model
from helpers.models.common import VideoModelFoundation, ImageModelFoundation
from helpers.models.common import ModelFoundation

try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image
from helpers.training.state_tracker import StateTracker
from helpers.models.common import PredictionTypes, PipelineTypes
from helpers.training.exceptions import MultiDatasetExhausted
from diffusers.schedulers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    DDPMScheduler,
)
from helpers.models.hidream.schedule import FlowUniPCMultistepScheduler
from diffusers.utils.torch_utils import is_compiled_module
from helpers.multiaspect.image import MultiaspectImage
from helpers.image_manipulation.brightness import calculate_luminance
from PIL import Image, ImageDraw, ImageFont
from helpers.training.deepspeed import (
    deepspeed_zero_init_disabled_context_manager,
    prepare_model_for_deepspeed,
)
from transformers.utils import ContextManagers

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")


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
from diffusers.utils import is_wandb_available
from helpers.prompts import PromptHandler
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
)

if is_wandb_available():
    import wandb


logger = logging.getLogger("validation")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")


def resize_validation_images(validation_images, edge_length):
    # we have to scale all the inputs to a stage4 image down to 64px smaller edge.
    resized_validation_samples = []
    for _sample in validation_images:
        validation_shortname, validation_prompt, _, training_sample_image = _sample
        resize_to, crop_to, new_aspect_ratio = (
            MultiaspectImage.calculate_new_size_by_pixel_edge(
                aspect_ratio=MultiaspectImage.calculate_image_aspect_ratio(
                    training_sample_image
                ),
                resolution=int(edge_length),
                original_size=training_sample_image.size,
            )
        )
        # we can be less precise here
        training_sample_image = training_sample_image.resize(crop_to)
        resized_validation_samples.append(
            (validation_shortname, validation_prompt, training_sample_image)
        )
    return resized_validation_samples


def reset_eval_datasets():
    eval_datasets = StateTracker.get_data_backends(_type="eval", _types=None)
    for dataset_name, dataset in eval_datasets.items():
        if "train_dataset" not in dataset:
            logger.debug(
                f"Skipping eval set {dataset_name} because it lacks a dataloader."
            )
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
    from helpers.training.collate import collate_fn

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
                logger.debug(
                    f"Skipping eval set {ds_name} because it lacks a dataloader."
                )
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
                logger.debug(
                    f"Ran out of evaluation samples for dataset {ds_name}. Resetting buckets."
                )
                dataset["sampler"]._reset_buckets(raise_exhaustion_signal=False)
                # We re-raise if we've exhausted this dataset. If `dataset_name` is set,
                # we effectively stop; if it's None, we move to the next dataset.
                if dataset_name is not None:
                    raise e

    return output


def retrieve_validation_images():
    """
    From each data backend, collect the top 5 images for validation, such that
    we select the same images on each startup, unless the dataset changes.

    Returns:
        dict: A dictionary of shortname to image paths.
    """
    if StateTracker.get_model().requires_validation_edit_captions():
        return retrieve_validation_edit_images()

    args = StateTracker.get_args()
    requires_cond_input = any(
        [
            StateTracker.get_model().requires_conditioning_validation_inputs(),
            args.controlnet,
            args.control,
        ]
    )
    data_backends = StateTracker.get_data_backends(
        _type=(
            StateTracker.get_model().conditioning_validation_dataset_type()
            if requires_cond_input
            else "image"
        )
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
            requires_cond_input
            and data_backend.get("dataset_type", None)
            != StateTracker.get_model().conditioning_validation_dataset_type()
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
            validation_input_image_pixel_edge_len = (
                StateTracker.get_model().validation_image_input_edge_length()
            )
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
            logger.warning(
                f"Data backend {data_backend['id']} does not have a sampler. Skipping."
            )
    logger.info(f"Collected {len(validation_set)} validation image inputs.")
    return validation_set


def retrieve_validation_edit_images() -> list[tuple[str, str, Image.Image]]:
    """
    Returns [(shortname, *edited-scene caption*, reference_image), ...]
    for models that need **edit** validation.

    Logic
    -----
    • loop over *image* datasets that have a sampler
    • for every deterministic validation sample returned by the sampler
      – grab its original file path from metadata
      – ask the dataset’s *registered* conditioning backend for the
        counterpart via `get_conditioning_sample()`
      – add the trio to output
    """
    model = StateTracker.get_model()
    if not model.requires_validation_edit_captions():
        return []  # no-op for ordinary models

    args = StateTracker.get_args()
    validation_set = []

    # ---------- iterate over IMAGE datasets ---------------------------------
    for backend_id, backend in StateTracker.get_data_backends(_type="image").items():
        sampler = backend.get("sampler")
        if sampler is None:  # nothing to iterate over
            continue

        # each backend should know which conditioning dataset is linked to it
        cond_backend = backend.get("conditioning_data")
        cond_backend_id = cond_backend.get("id") if cond_backend else None
        if cond_backend_id is None:
            logger.debug("No conditioning backend configured for this image dataset.")
            continue

        logger.debug(f"Backend id: {backend_id}, cond_backend_id: {cond_backend_id}")
        cond_backend = StateTracker.get_data_backend(cond_backend_id)
        if cond_backend is None:
            logger.debug(f"Conditioning backend {cond_backend_id} not found.")
            continue  # mis-configuration → skip

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
                if (
                    image_dataset_dir_prefix is not None
                    and image_dataset_dir_prefix in sample_path
                ):
                    rel_path = sample_path.replace(image_dataset_dir_prefix, "")
                    # remove trailing '/'
                    rel_path = rel_path.lstrip("/")
                    logger.debug(f"Removed prefix, got relative path: {rel_path}")
                logger.debug(f"Metadata: {meta}")
            except Exception as e:
                continue  # metadata missing → skip

            conditioning_dir_prefix = cond_backend[
                "sampler"
            ].metadata_backend.instance_data_dir
            cond_sample_path = os.path.join(conditioning_dir_prefix, rel_path)
            cond_sample = cond_backend["sampler"].get_conditioning_sample(rel_path)

            if cond_sample is None:
                logger.warning(
                    f"Conditioning sample for {rel_path} not found in {cond_backend_id}."
                )
                continue

            reference_img = cond_sample.image
            logger.debug(f"Validation sample path: {cond_sample_path}")
            validation_set.append((shortname, edited_prompt, reference_img))

    logger.info(f"Collected {len(validation_set)} edit-validation samples.")
    return validation_set


def prepare_validation_prompt_list(args, embed_cache, model):
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
            # Collect the prompts for the validation images.
            for _validation_sample in tqdm(
                validation_sample_images,
                ncols=125,
                desc="Precomputing validation image embeds",
            ):
                if (
                    isinstance(_validation_sample, tuple)
                    and len(_validation_sample) == 3
                ):
                    _, validation_prompt, _ = _validation_sample
                elif (
                    isinstance(_validation_sample, tuple)
                    and len(_validation_sample) == 4
                ):
                    _, validation_prompt, _, _ = _validation_sample
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
            ncols=125,
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
            ncols=125,
            desc="Precomputing user prompt library embeddings",
        ):
            # move_text_encoders(embed_cache.text_encoders, embed_cache.accelerator.device)
            embed_cache.compute_embeddings_for_prompts(
                [prompt], is_validation=True, load_from_cache=False
            )
            # move_text_encoders(embed_cache.text_encoders, "cpu")
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
    # Compute negative embed for validation prompts, if any are set, so that it's stored before we unload the text encoder.
    if validation_prompts:
        logger.info("Precomputing the negative prompt embed for validations.")
        model.log_model_devices()
        validation_negative_prompt_text_encoder_output = (
            embed_cache.compute_embeddings_for_prompts(
                [StateTracker.get_args().validation_negative_prompt],
                is_validation=True,
                load_from_cache=False,
            )
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
    if (
        type(validation_resolution_parameter) is str
        and "," in validation_resolution_parameter
    ):
        return [
            parse_validation_resolution(res)
            for res in validation_resolution_parameter.split(",")
        ]
    return [parse_validation_resolution(validation_resolution_parameter)]


def parse_validation_resolution(input_str: str) -> tuple:
    """
    If the args.validation_resolution:
     - is an int, we'll treat it as height and width square aspect
     - if it has an x in it, we will split and treat as WIDTHxHEIGHT
     - if it has comma, we will split and treat each value as above
    """
    is_df_ii = (
        True if str(StateTracker.get_args().model_flavour).startswith("ii-") else False
    )
    if isinstance(input_str, int) or input_str.isdigit():
        if is_df_ii and int(input_str) < 256:
            raise ValueError(
                "Cannot use less than 256 resolution for DeepFloyd stage 2."
            )
        return (input_str, input_str)
    if "x" in input_str:
        pieces = input_str.split("x")
        if is_df_ii and (int(pieces[0]) < 256 or int(pieces[1]) < 256):
            raise ValueError(
                "Cannot use less than 256 resolution for DeepFloyd stage 2."
            )
        return (int(pieces[0]), int(pieces[1]))


class Validation:
    def __init__(
        self,
        accelerator,
        model: ModelFoundation,
        args,
        validation_prompt_metadata,
        vae_path,
        weight_dtype,
        embed_cache,
        ema_model,
        is_deepspeed: bool = False,
        model_evaluator=None,
        trainable_parameters=None,
    ):
        self.trainable_parameters = trainable_parameters
        self.accelerator = accelerator
        self.prompt_handler = None
        self.unet, self.transformer = None, None
        self.model = model
        if args.controlnet:
            self.controlnet = model.get_trained_component()
        elif "unet" in str(self.model.get_trained_component().__class__).lower():
            self.unet = self.model.get_trained_component()
        elif "transformer" in str(self.model.get_trained_component().__class__).lower():
            self.transformer = self.model.get_trained_component()
        self.config = args
        self.save_dir = os.path.join(args.output_dir, "validation_images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.global_step = None
        self.global_resume_step = None
        self.validation_prompt_metadata = validation_prompt_metadata
        self.validation_images = None
        self.weight_dtype = weight_dtype
        self.embed_cache = embed_cache
        self.ema_model = ema_model
        self.ema_enabled = False
        self.deepfloyd = True if "deepfloyd" in self.config.model_family else False
        self.deepfloyd_stage2 = (
            True if str(self.config.model_flavour).startswith("ii-") else False
        )
        self._discover_validation_input_samples()
        self.validation_resolutions = (
            get_validation_resolutions() if not self.deepfloyd_stage2 else [(256, 256)]
        )
        self.flow_matching = (
            True
            if self.model.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING
            else False
        )
        self.deepspeed = is_deepspeed
        if is_deepspeed:
            if args.use_ema:
                if args.ema_validation != "none":
                    logger.error(
                        "EMA validation is not supported via DeepSpeed."
                        " Please use --ema_validation=none or disable DeepSpeed."
                    )
                    sys.exit(1)
        self.inference_device = (
            accelerator.device
            if not is_deepspeed
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_evaluator = model_evaluator
        if self.model_evaluator is not None:
            logger.debug(f"Using model evaluator: {self.model_evaluator}")
        self._update_state()
        self.eval_scores = {}

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
            logger.debug(
                f"Image inputs discovered for validation: {self.validation_image_inputs}"
            )

    def _pipeline_cls(self):
        if self.model is not None:
            if self.config.validation_using_datasets:
                if PipelineTypes.IMG2IMG not in self.model.PIPELINE_CLASSES:
                    raise ValueError(
                        f"Cannot run {self.model.MODEL_CLASS} in Img2Img mode for validation."
                    )
            if self.config.controlnet:
                if PipelineTypes.CONTROLNET not in self.model.PIPELINE_CLASSES:
                    raise ValueError(
                        f"Cannot run {self.model.MODEL_CLASS} in ControlNet mode for validation."
                    )
            if self.config.control:
                if PipelineTypes.CONTROL not in self.model.PIPELINE_CLASSES:
                    raise ValueError(
                        f"Cannot run {self.model.MODEL_CLASS} in Control mode for validation."
                    )

        if self.config.validation_using_datasets:
            return self.model.PIPELINE_CLASSES[PipelineTypes.IMG2IMG]
        if self.config.controlnet:
            return self.model.PIPELINE_CLASSES[PipelineTypes.CONTROLNET]
        if self.config.control:
            return self.model.PIPELINE_CLASSES[PipelineTypes.CONTROL]
        return self.model.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]

    def _gather_prompt_embeds(self, validation_prompt: str):
        prompt_embed = self.embed_cache.compute_embeddings_for_prompts(
            [validation_prompt]
        )
        if prompt_embed is None:
            return

        prompt_embed = {
            k: v.to(self.inference_device) if hasattr(v, "to") else v
            for k, v in prompt_embed.items()
        }

        return self.model.convert_text_embed_for_pipeline(prompt_embed)

    def _benchmark_path(self, benchmark: str = "base_model"):
        # does the benchmark directory exist?
        if not os.path.exists(os.path.join(self.config.output_dir, "benchmarks")):
            os.makedirs(
                os.path.join(self.config.output_dir, "benchmarks"), exist_ok=True
            )
        return os.path.join(self.config.output_dir, "benchmarks", benchmark)

    def stitch_benchmark_image(
        self,
        validation_image_result,
        benchmark_image,
        separator_width=5,
        labels=["base model", "checkpoint"],
    ):
        """
        For each image, make a new canvas and place it side by side with its equivalent from {self.validation_image_inputs}
        Add "base model" text to the left image and "checkpoint" text to the right image
        Include a separator between the images
        """

        # Calculate new dimensions
        new_width = (
            benchmark_image.size[0] + validation_image_result.size[0] + separator_width
        )
        new_height = benchmark_image.size[1]

        # Create a new image with a white background
        new_image = Image.new("RGB", (new_width, new_height), color="white")

        # Paste the images with a gap between them
        new_image.paste(benchmark_image, (0, 0))
        new_image.paste(
            validation_image_result, (benchmark_image.size[0] + separator_width, 0)
        )

        # Create a drawing object
        draw = ImageDraw.Draw(new_image)

        # Use a default font
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except IOError:
            font = ImageFont.load_default()

        # Add text to the left image
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
            # Add text to the right image
            draw.text(
                (benchmark_image.size[0] + separator_width + 10, 10),
                labels[1],
                fill=(255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

        # Draw a vertical line as a separator
        line_color = (200, 200, 200)  # Light gray
        for i in range(separator_width):
            x = validation_image_result.size[0] + i
            draw.line([(x, 0), (x, new_height)], fill=line_color)

        return new_image

    def _benchmark_image(self, shortname, resolution):
        """
        We will retrieve the benchmark image for the shortname.
        """
        if not self.benchmark_exists():
            return None
        base_model_benchmark = self._benchmark_path("base_model")
        benchmark_image = None
        _test_filename = f"{shortname}_{resolution[0]}x{resolution[1]}.png"
        for _benchmark_image in os.listdir(base_model_benchmark):
            _basename = os.path.basename(_benchmark_image)
            if _basename == _test_filename:
                benchmark_image = Image.open(
                    os.path.join(base_model_benchmark, _benchmark_image)
                )
                break

        return benchmark_image

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
                        Image.open(
                            os.path.join(base_model_benchmark, _benchmark_image)
                        ),
                    )
                )

        return benchmark_images

    def benchmark_exists(self, benchmark: str = "base_model"):
        """
        Determines whether the base model benchmark outputs already exist.
        """
        base_model_benchmark = self._benchmark_path()

        return os.path.exists(base_model_benchmark)

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
                if hasattr(image, "size"):
                    width, height = image.size
                    image.save(
                        os.path.join(
                            base_model_benchmark, f"{shortname}_{width}x{height}.png"
                        )
                    )
                elif type(image) is list:
                    from diffusers.utils.export_utils import export_to_video

                    export_to_video(
                        image,
                        os.path.join(base_model_benchmark, f"{shortname}_{idx}.mp4"),
                        fps=self.config.framerate,
                    )

    def _update_state(self):
        """Updates internal state with the latest from StateTracker."""
        self.global_step = StateTracker.get_global_step()
        self.global_resume_step = StateTracker.get_global_resume_step() or 1

    def would_validate(
        self,
        step: int = 0,
        validation_type="intermediary",
        force_evaluation: bool = False,
    ):
        # a wrapper for should_perform_intermediary_validation that can run in the training loop
        self._update_state()
        return self.should_perform_intermediary_validation(
            step, self.validation_prompt_metadata, validation_type
        ) or (step == 0 and validation_type == "base_model")

    def run_validations(
        self,
        step: int = 0,
        validation_type="intermediary",
        force_evaluation: bool = False,
        skip_execution: bool = False,
    ):
        self._update_state()
        would_do_intermediary_validation = self.should_perform_intermediary_validation(
            step, self.validation_prompt_metadata, validation_type
        ) or (step == 0 and validation_type == "base_model")
        logger.debug(
            f"Should evaluate: {would_do_intermediary_validation}, force evaluation: {force_evaluation}, skip execution: {skip_execution}"
        )
        if not would_do_intermediary_validation and not force_evaluation:
            return self
        if would_do_intermediary_validation and validation_type == "final":
            # If the validation would have fired off, we'll skip it.
            # This is useful at the end of training so we don't validate 2x.
            logger.debug(
                "Not running validation because intermediary might have already fired off."
            )
            return self
        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message="Validations are generating.. this might take a minute! 🖼️",
                message_level="info",
            )

        if self.accelerator.is_main_process or self.deepspeed:
            logger.debug("Starting validation process...")
            diffusers.utils.logging._tqdm_active = False
            self.setup_pipeline(validation_type)
            if self.model.pipeline is None:
                logger.error(
                    "Not able to run validations, we did not obtain a valid pipeline."
                )
                self.validation_images = None
                return self
            self.setup_scheduler()
            self.process_prompts()
            self.finalize_validation(validation_type)
            if self.evaluation_result is not None:
                logger.info(f"Evaluation result: {self.evaluation_result}")
            logger.debug("Validation process completed.")
            self.clean_pipeline()

        return self

    def should_perform_intermediary_validation(
        self, step, validation_prompts, validation_type
    ):
        should_do_intermediary_validation = (
            validation_prompts
            and self.global_step % self.config.validation_steps == 0
            and step % self.config.gradient_accumulation_steps == 0
            and self.global_step > self.global_resume_step
        )
        return should_do_intermediary_validation and (
            self.accelerator.is_main_process or self.deepseed
        )

    def setup_scheduler(self):
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
                # The Beta schedule looks WAY better...
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

        if (
            self.model.pipeline is not None
            and "variance_type" in self.model.pipeline.scheduler.config
        ):
            variance_type = self.model.pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        scheduler = SCHEDULER_NAME_MAP[
            self.config.validation_noise_scheduler
        ].from_pretrained(
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

        if getattr(self.model, "pipeline", None) is None:
            self.model.pipeline = self.model.get_pipeline(
                pipeline_type=(
                    PipelineTypes.CONTROLNET
                    if self.config.controlnet
                    else (
                        PipelineTypes.CONTROL
                        if self.config.control
                        else self.model.DEFAULT_PIPELINE_TYPE
                    )
                ),
            )

        self.model.move_models(self.accelerator.device)
        # self.model = self.model.pipeline.to(self.inference_device)
        self.model.pipeline.set_progress_bar_config(disable=True)

    def clean_pipeline(self):
        """Remove the pipeline."""
        if hasattr(self.accelerator, "_lycoris_wrapped_network"):
            self.accelerator._lycoris_wrapped_network.set_multiplier(1.0)
        if self.model.pipeline is not None:
            del self.model.pipeline
            self.model.pipeline = None

    def process_prompts(self):
        """Processes each validation prompt and logs the result."""
        self.validation_prompt_dict = {}
        self.evaluation_result = None
        validation_images = {}
        _content = self.validation_prompt_metadata["validation_prompts"]
        total_samples = len(_content) if _content is not None else 0
        self.eval_scores = {}
        if self.validation_image_inputs:
            # Override the pipeline inputs to be entirely based upon the validation image inputs.
            _content = self.validation_image_inputs
            if "DeepFloyd" in self.model.NAME:
                resize_edge_length = 64
                # Resize validation input to 64px area
                _content = resize_validation_images(_content, resize_edge_length)
            total_samples = len(_content) if _content is not None else 0

        logger.debug(f"Processing content: {_content}")
        idx = 0
        for prompt in tqdm(
            _content if _content else [],
            desc="Processing validation prompts",
            total=total_samples,
            leave=False,
            position=1,
        ):
            validation_input_image = None
            if len(prompt) == 3 and isinstance(prompt[2], Image.Image):
                # DeepFloyd stage II inputs.
                shortname, prompt, validation_input_image = prompt
            elif len(prompt) == 4 and isinstance(prompt[3], Image.Image):
                (
                    shortname,
                    prompt,
                    validation_input_image_path,
                    validation_input_image,
                ) = prompt
            else:
                shortname = self.validation_prompt_metadata["validation_shortnames"][
                    idx
                ]
            logger.debug(f"validation prompt (shortname={shortname}): '{prompt}'")
            self.validation_prompt_dict[shortname] = prompt
            logger.debug(f"Processing validation for prompt: {prompt}")
            (
                stitched_validation_images,
                checkpoint_validation_images,
                ema_validation_images,
            ) = self.validate_prompt(prompt, shortname, validation_input_image)
            validation_images.update(stitched_validation_images)
            if isinstance(self.model, VideoModelFoundation):
                self._save_videos(validation_images, shortname, prompt)
            else:
                self._save_images(validation_images, shortname, prompt)
            logger.debug(f"Completed generating image: {prompt}")
            self.validation_images = validation_images
            self.evaluation_result = self.evaluate_images(checkpoint_validation_images)
            self._log_validations_to_webhook(validation_images, shortname, prompt)
            idx += 1
        try:
            self._log_validations_to_trackers(validation_images)
        except Exception as e:
            logger.error(f"Error logging validation images: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def get_eval_result(self):
        return self.evaluation_result or {}

    def clear_eval_result(self):
        self.evaluation_result = None

    def stitch_conditioning_images(self, validation_image_results, conditioning_image):
        """
        For each image, make a new canvas and place it side by side with its equivalent from {self.validation_image_inputs}
        """
        stitched_validation_images = []
        for idx, image in enumerate(validation_image_results):
            new_width = image.size[0] * 2
            new_height = image.size[1]
            new_image = Image.new("RGB", (new_width, new_height))
            new_image.paste(image, (0, 0))
            new_image.paste(conditioning_image, (image.size[0], 0))
            stitched_validation_images.append(new_image)

        return stitched_validation_images

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
        self, prompt, validation_shortname, validation_input_image=None
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
                    validation_image_edge_len = (
                        self.model.validation_image_input_edge_length()
                    )
                    if validation_image_edge_len is not None:
                        # calculate the megapixels value (eg ~0.25 for 512px)
                        validation_image_megapixels = (
                            validation_image_edge_len**2
                        ) / 1_000_000
                        validation_resolution, _, validation_aspect_ratio = (
                            MultiaspectImage.calculate_new_size_by_pixel_area(
                                aspect_ratio=MultiaspectImage.calculate_image_aspect_ratio(
                                    extra_validation_kwargs["image"]
                                ),
                                megapixels=validation_image_megapixels,
                                original_size=extra_validation_kwargs["image"].size,
                            )
                        )
                        extra_validation_kwargs["image"] = extra_validation_kwargs[
                            "image"
                        ].resize(validation_resolution, Image.Resampling.LANCZOS)
                        validation_resolution_width, validation_resolution_height = (
                            validation_resolution
                        )
                    else:
                        validation_resolution_width, validation_resolution_height = (
                            extra_validation_kwargs["image"].size
                        )
                else:
                    raise ValueError(
                        "Validation input images are not supported for this model type."
                    )
                extra_validation_kwargs["control_image"] = extra_validation_kwargs[
                    "image"
                ]
            else:
                validation_resolution_width, validation_resolution_height = resolution

            if type(self.config.validation_guidance_skip_layers) is list:
                extra_validation_kwargs["skip_layer_guidance_start"] = float(
                    self.config.validation_guidance_skip_layers_start
                )
                extra_validation_kwargs["skip_layer_guidance_stop"] = float(
                    self.config.validation_guidance_skip_layers_stop
                )
                extra_validation_kwargs["skip_layer_guidance_scale"] = float(
                    self.config.validation_guidance_skip_scale
                )
                extra_validation_kwargs["skip_guidance_layers"] = list(
                    self.config.validation_guidance_skip_layers
                )

            extra_validation_kwargs["guidance_rescale"] = (
                self.config.validation_guidance_rescale
            )

            if StateTracker.get_args().validation_using_datasets:
                extra_validation_kwargs["strength"] = getattr(
                    self.config, "validation_strength", 0.2
                )
                logger.debug(
                    f"Set validation image denoise strength to {extra_validation_kwargs['strength']}"
                )

            logger.debug(
                f"Processing width/height: {validation_resolution_width}x{validation_resolution_height}"
            )
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
                    "height": MultiaspectImage._round_to_nearest_multiple(
                        int(validation_resolution_height), 16
                    ),
                    "width": MultiaspectImage._round_to_nearest_multiple(
                        int(validation_resolution_width), 16
                    ),
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
                        pipeline_kwargs["negative_prompt"] = (
                            StateTracker.get_args().validation_negative_prompt
                        )
                # TODO: Refactor the rest so that it uses model class to update kwargs more generally.
                if self.config.validation_guidance_real > 1.0:
                    pipeline_kwargs["guidance_scale_real"] = float(
                        self.config.validation_guidance_real
                    )
                if (
                    isinstance(self.config.validation_no_cfg_until_timestep, int)
                    and self.config.model_family == "flux"
                ):
                    pipeline_kwargs["no_cfg_until_timestep"] = (
                        self.config.validation_no_cfg_until_timestep
                    )

                pipeline_kwargs = self.model.update_pipeline_call_kwargs(
                    pipeline_kwargs
                )
                logger.debug(
                    f"Image being generated with parameters: {pipeline_kwargs}"
                )
                # Print the device attr of any parameters that have one
                # for key, value in pipeline_kwargs.items():
                #     if hasattr(value, "device"):
                #         logger.debug(f"Device for {key}: {value.device}")
                # for key, value in self.model.pipeline.components.items():
                #     if hasattr(value, "device"):
                #         logger.debug(f"Device for {key}: {value.device}")
                if self.config.model_family == "sana":
                    pipeline_kwargs["complex_human_instruction"] = (
                        self.config.sana_complex_human_instruction
                    )

                validation_types = self._validation_types()
                all_validation_type_results = {}
                for current_validation_type in validation_types:
                    if not self.config.validation_randomize:
                        pipeline_kwargs["generator"] = self._get_generator()
                        logger.debug(
                            f"Using a generator? {pipeline_kwargs['generator']}"
                        )
                    if current_validation_type == "ema":
                        self.enable_ema_for_inference()
                    pipeline_kwargs = {
                        k: (
                            v.to(
                                device=self.inference_device,
                                dtype=self.config.weight_dtype,
                            )
                            if hasattr(v, "to")
                            else v
                        )
                        for k, v in pipeline_kwargs.items()
                    }
                    import inspect

                    call_kwargs = inspect.signature(
                        self.model.pipeline.__call__
                    ).parameters
                    logger.debug(
                        f"Possible parameters for {type(self.model.pipeline)}: {call_kwargs}"
                    )
                    # remove any kwargs that are not in the pipeline call
                    pipeline_kwargs = {
                        k: v for k, v in pipeline_kwargs.items() if k in call_kwargs
                    }
                    removed_kwargs = [
                        k for k in pipeline_kwargs.keys() if k not in call_kwargs
                    ]
                    logger.debug(
                        f"Running validations with inputs: {pipeline_kwargs.keys()}"
                    )
                    if removed_kwargs:
                        logger.warning(
                            f"Removed the following kwargs from validation pipeline: {removed_kwargs}"
                        )
                    if isinstance(self.model, VideoModelFoundation):
                        all_validation_type_results[current_validation_type] = (
                            self.model.pipeline(**pipeline_kwargs).frames
                        )
                    elif isinstance(self.model, ImageModelFoundation):
                        all_validation_type_results[current_validation_type] = (
                            self.model.pipeline(**pipeline_kwargs).images
                        )
                    if current_validation_type == "ema":
                        self.disable_ema_for_inference()

                # retrieve the default image result for stitching to controlnet inputs.
                ema_image_results = all_validation_type_results.get("ema")
                validation_image_results = all_validation_type_results.get(
                    "checkpoint", ema_image_results
                )
                original_validation_image_results = validation_image_results
                benchmark_image = None
                if any([self.config.controlnet, self.config.control]):
                    validation_image_results = self.stitch_conditioning_images(
                        original_validation_image_results,
                        extra_validation_kwargs["control_image"],
                    )
                elif not self.config.disable_benchmark and self.benchmark_exists(
                    "base_model"
                ):
                    benchmark_image = self._benchmark_image(
                        validation_shortname, resolution
                    )
                    if benchmark_image is not None:
                        for idx, validation_image in enumerate(
                            validation_image_results
                        ):
                            validation_image_results[idx] = self.stitch_benchmark_image(
                                validation_image_result=validation_image,
                                benchmark_image=benchmark_image,
                            )

                checkpoint_validation_images[validation_shortname].extend(
                    original_validation_image_results
                )
                stitched_validation_images[validation_shortname].extend(
                    validation_image_results
                )
                if self.config.use_ema:
                    if (
                        validation_shortname in ema_validation_images
                        and ema_image_results is not None
                    ):
                        if ema_validation_images[validation_shortname] is None:
                            # init the value
                            ema_validation_images[validation_shortname] = []
                        if isinstance(
                            ema_validation_images[validation_shortname], list
                        ):
                            # if we have a list of images, we can stitch them.
                            ema_validation_images[validation_shortname].extend(
                                ema_image_results
                            )

            except Exception as e:
                import traceback

                logger.error(
                    f"Error generating validation image: {e}, {traceback.format_exc()}"
                )
                continue
        if (
            self.config.use_ema
            and self.config.ema_validation == "comparison"
            and benchmark_image is not None
        ):
            for idx, validation_image in enumerate(
                stitched_validation_images[validation_shortname]
            ):
                stitched_validation_images[validation_shortname][idx] = (
                    self.stitch_benchmark_image(
                        validation_image_result=ema_validation_images[
                            validation_shortname
                        ][idx],
                        benchmark_image=stitched_validation_images[
                            validation_shortname
                        ][idx],
                        labels=[None, "EMA"],
                    )
                )

        return (
            stitched_validation_images,
            checkpoint_validation_images,
            ema_validation_images,
        )

    def _save_videos(self, validation_images, validation_shortname, validation_prompt):
        validation_img_idx = 0
        from diffusers.utils.export_utils import export_to_video

        for validation_image in validation_images[validation_shortname]:
            # convert array of numpy to array of pil:
            validation_image = MultiaspectImage.numpy_list_to_pil(validation_image)
            size_x, size_y = validation_image[0].size
            res_label = f"{size_x}x{size_y}"
            export_to_video(
                validation_image,
                os.path.join(
                    self.save_dir,
                    f"step_{StateTracker.get_global_step()}_{validation_shortname}_{validation_img_idx}_{res_label}.mp4",
                ),
                fps=self.config.framerate,
            )
            validation_img_idx += 1

    def _save_images(self, validation_images, validation_shortname, validation_prompt):
        validation_img_idx = 0
        for validation_image in validation_images[validation_shortname]:
            res = self.validation_resolutions[validation_img_idx]
            if "x" in res:
                res_label = str(res)
            elif type(res) is tuple:
                res_label = f"{res[0]}x{res[1]}"
            else:
                res_label = f"{res}x{res}"
            validation_image.save(
                os.path.join(
                    self.save_dir,
                    f"step_{StateTracker.get_global_step()}_{validation_shortname}_{res_label}.png",
                )
            )
            validation_img_idx += 1

    def _log_validations_to_webhook(
        self, validation_images, validation_shortname, validation_prompt
    ):
        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                (
                    f"Validation {'image' if StateTracker.get_webhook_handler().send_video is False else 'video'} for `{validation_shortname if validation_shortname != '' else '(blank shortname)'}`"
                    f"\nValidation prompt: `{validation_prompt if validation_prompt != '' else '(blank prompt)'}`"
                    f"\nEvaluation score: {self.eval_scores.get(validation_shortname, 'N/A')}"
                ),
                images=validation_images[validation_shortname],
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
                            f"{shortname} - {self.validation_resolutions[idx]}": np.moveaxis(
                                np.array(image), -1, 0
                            )[
                                np.newaxis, ...
                            ]
                            for idx, image in enumerate(image_list)
                        },
                        step=StateTracker.get_global_step(),
                    )
            elif tracker.name == "wandb":
                resolution_list = [
                    f"{res[0]}x{res[1]}" for res in get_validation_resolutions()
                ]

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
                        logger.debug(
                            f"Prompt {prompt_shortname} has {len(image_list)} images"
                        )
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
                        logger.debug(
                            f"Prompt {prompt_shortname} has {len(image_list)} images"
                        )
                        for idx, image in enumerate(image_list):
                            # if it's a list of images, make a grid
                            if isinstance(image, list) and isinstance(
                                image[0], Image.Image
                            ):
                                image = image[0]
                            wandb_image = wandb.Image(
                                image,
                                caption=f"{prompt_shortname} - {resolution_list[idx]}",
                            )
                            gallery_images[
                                f"{prompt_shortname} - {resolution_list[idx]}"
                            ] = wandb_image

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
                    self.ema_model.store(
                        self.accelerator._lycoris_wrapped_network.parameters()
                    )
                    logger.debug(
                        "Storing the EMA weights into the Lycoris adapter for inference."
                    )
                    self.ema_model.copy_to(
                        self.accelerator._lycoris_wrapped_network.parameters()
                    )
                elif self.config.lora_type.lower() == "standard":
                    _trainable_parameters = [
                        x
                        for x in self.model.get_trained_component().parameters()
                        if x.requires_grad
                    ]
                    self.ema_model.store(_trainable_parameters)
                    self.ema_model.copy_to(_trainable_parameters)
            else:
                # if self.config.ema_device != "accelerator":
                #     logger.info("Moving checkpoint to CPU for storage.")
                #     self.model.get_trained_component().to("cpu")
                logger.debug("Storing EMA weights for later recovery.")
                self.ema_model.store(self.trainable_parameters())
                logger.debug("Storing the EMA weights into the model for inference.")
                self.ema_model.copy_to(self.trainable_parameters())
            # if self.config.ema_device != "accelerator":
            #     logger.debug("Moving checkpoint to CPU for storage.")
            #     self.model.get_trained_component().to("cpu")
            #     logger.debug("Moving EMA weights to GPU for inference.")
            #     self.ema_model.to(self.inference_device)
        else:
            logger.debug(
                "Skipping EMA model setup for validation, as we are not using EMA."
            )

    def disable_ema_for_inference(self):
        if not self.ema_enabled:
            logger.debug("EMA was not enabled. Not disabling EMA.")
            return
        if self.config.use_ema:
            logger.debug("Disabling EMA.")
            self.ema_enabled = False
            if (
                self.config.model_type == "lora"
                and self.config.lora_type.lower() == "lycoris"
            ):
                logger.debug("Setting Lycoris network multiplier to 1.0.")
                self.accelerator._lycoris_wrapped_network.set_multiplier(1.0)
                logger.debug("Restoring Lycoris weights.")
                self.ema_model.restore(
                    self.accelerator._lycoris_wrapped_network.parameters()
                )
            else:
                logger.debug("Restoring trainable parameters.")
                self.ema_model.restore(self.trainable_parameters())
            if self.config.ema_device != "accelerator":
                logger.debug("Moving EMA weights to CPU for storage.")
                self.ema_model.to(self.config.ema_device)
                self.model.get_trained_component().to(self.inference_device)

        else:
            logger.debug(
                "Skipping EMA model restoration for validation, as we are not using EMA."
            )

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
            logger.warning(
                "No evaluation scores were calculated. Please check your evaluation settings."
            )
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
        accept_mu = "mu" in set(
            inspect.signature(noise_scheduler.set_timesteps).parameters.keys()
        )
        scheduler_kwargs = {}
        if accept_mu and self.config.flow_schedule_auto_shift:
            from helpers.models.sd3.pipeline import calculate_shift

            scheduler_kwargs["mu"] = calculate_shift(
                StateTracker.get_model().get_trained_component().config.max_seq
            )

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
                if (
                    self.config.num_eval_images is not None
                    and evaluated_sample_count > self.config.num_eval_images
                ):
                    reset_eval_datasets()
                    raise MultiDatasetExhausted(
                        "Max eval samples reached, resetting evaluations."
                    )
                # Pass the dataset_name so we fetch from the correct place
                eval_batch = retrieve_eval_images(dataset_name=dataset_name)

            except MultiDatasetExhausted:
                logger.info(
                    f"Evaluation loss calculation completed for dataset: {dataset_name}"
                )
                eval_batch = False

            if eval_batch is not None and eval_batch is not False:
                # Fix a known seed so noise is consistent across eval
                torch.manual_seed(0)
                prepared_eval_batch = prepare_batch(eval_batch)
                if "latents" not in prepared_eval_batch:
                    raise ValueError(
                        "Error calculating eval batch: no 'latents' found."
                    )

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
        eval_datasets = StateTracker.get_data_backends(
            _type="eval", _types=None
        )  # dict of {name: ...}
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
            logger.debug(
                "Running separate eval passes for each dataset + pooled results."
            )
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
