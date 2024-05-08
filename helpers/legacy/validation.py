import logging, os, torch, numpy as np
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


def deepfloyd_validation_images():
    """
    From each data backend, collect the top 5 images for validation, such that
    we select the same images on each startup, unless the dataset changes.

    Returns:
        dict: A dictionary of shortname to image paths.
    """
    data_backends = StateTracker.get_data_backends()
    validation_data_backend_id = StateTracker.get_args().eval_dataset_id
    validation_set = []
    logger.info("Collecting DF-II validation images")
    for _data_backend in data_backends:
        data_backend = StateTracker.get_data_backend(_data_backend)
        if "id" not in data_backend:
            continue
        logger.info(f"Checking data backend: {data_backend['id']}")
        if (
            validation_data_backend_id is not None
            and data_backend["id"] != validation_data_backend_id
        ):
            logger.warning(f"Not collecting images from {data_backend['id']}")
            continue
        if "sampler" in data_backend:
            validation_set.extend(
                data_backend["sampler"].retrieve_validation_set(
                    batch_size=StateTracker.get_args().num_eval_images
                )
            )
        else:
            logger.warning(
                f"Data backend {data_backend['id']} does not have a sampler. Skipping."
            )
    return validation_set


def prepare_validation_prompt_list(args, embed_cache):
    validation_negative_prompt_embeds = None
    validation_negative_pooled_embeds = None
    validation_prompts = [""]
    validation_shortnames = ["unconditional"]
    if not hasattr(embed_cache, "model_type"):
        raise ValueError(
            f"Embed cache engine did not contain a model_type. Cannot continue."
        )
    model_type = embed_cache.model_type
    validation_sample_images = None
    if "deepfloyd-stage2" in StateTracker.get_args().model_type:
        # Now, we prepare the DeepFloyd upscaler image inputs so that we can calculate their prompts.
        # If we don't do it here, they won't be available at inference time.
        validation_sample_images = deepfloyd_validation_images()
        if len(validation_sample_images) > 0:
            StateTracker.set_validation_sample_images(validation_sample_images)
            # Collect the prompts for the validation images.
            for _validation_sample in tqdm(
                validation_sample_images,
                ncols=100,
                desc="Precomputing DeepFloyd stage 2 eval prompt embeds",
            ):
                _, validation_prompt, _ = _validation_sample
                embed_cache.compute_embeddings_for_prompts([validation_prompt])

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
        if model_type == "sdxl":
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


def log_validations(
    accelerator,
    prompt_handler,
    unet,
    args,
    validation_prompts,
    validation_shortnames,
    step,
    text_encoder_1,
    tokenizer,
    vae_path: str,
    weight_dtype,
    embed_cache,
    validation_negative_pooled_embeds,
    validation_negative_prompt_embeds,
    text_encoder_2=None,
    tokenizer_2=None,
    ema_unet=None,
    vae=None,
    SCHEDULER_NAME_MAP: dict = {},
    validation_type: str = "training",
    pipeline: DiffusionPipeline = None,
):
    global_step = StateTracker.get_global_step()
    global_resume_step = StateTracker.get_global_resume_step() or global_step
    should_do_intermediary_validation = (
        validation_prompts
        and global_step % args.validation_steps == 0
        and step % args.gradient_accumulation_steps == 0
        and global_step > global_resume_step
    )
    if accelerator.is_main_process:
        if validation_type == "finish" or should_do_intermediary_validation:
            if (
                validation_prompts is None
                or validation_prompts == []
                or args.num_validation_images is None
                or args.num_validation_images <= 0
            ):
                return
            if validation_type == "finish" and should_do_intermediary_validation:
                # 382 - don't run final validation when we'd also have run the intermediary validation.
                return
            logger.debug(f"We have valid prompts to process.")
            if StateTracker.get_webhook_handler() is not None:
                StateTracker.get_webhook_handler().send(
                    message=f"Validations are generating.. this might take a minute! 🖼️",
                    message_level="info",
                )

            # create pipeline
            if validation_type == "validation" and args.use_ema:
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            vae_subfolder_path = "vae"
            if args.pretrained_vae_model_name_or_path is not None:
                vae_subfolder_path = None
            if vae is None and "deepfloyd" not in args.model_type:
                vae = AutoencoderKL.from_pretrained(
                    vae_path,
                    subfolder=vae_subfolder_path,
                    revision=args.revision,
                    force_upcast=False,
                )
            # The models need unwrapping because for compatibility in distributed training mode.
            if not pipeline:
                if StateTracker.get_model_type() == "sdxl":
                    pipeline_cls = StableDiffusionXLPipeline
                    pipeline = pipeline_cls.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=unwrap_model(accelerator, unet),
                        text_encoder=None,
                        text_encoder_2=None,
                        tokenizer=None,
                        tokenizer_2=None,
                        vae=vae,
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                        add_watermarker=args.enable_watermark,
                    )
                elif StateTracker.get_model_type() == "legacy":
                    pipeline_cls = DiffusionPipeline
                    if "deepfloyd-stage2" in args.model_type:
                        from diffusers.pipelines import IFSuperResolutionPipeline

                        pipeline_cls = IFSuperResolutionPipeline
                    pipeline = pipeline_cls.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=unwrap_model(accelerator, unet),
                        text_encoder=text_encoder_1,
                        tokenizer=None,
                        vae=vae,
                        revision=args.revision,
                        safety_checker=None,
                        torch_dtype=(
                            torch.bfloat16
                            if torch.backends.mps.is_available()
                            or torch.cuda.is_available()
                            else torch.bfloat16
                        ),
                    )
                scheduler_args = {}

                if "variance_type" in pipeline.scheduler.config:
                    variance_type = pipeline.scheduler.config.variance_type

                    if variance_type in ["learned", "learned_range"]:
                        variance_type = "fixed_small"

                    scheduler_args["variance_type"] = variance_type
                if "deepfloyd" in args.model_type:
                    args.validation_noise_scheduler = "ddpm"

                pipeline.scheduler = SCHEDULER_NAME_MAP[
                    args.validation_noise_scheduler
                ].from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder="scheduler",
                    prediction_type=args.prediction_type,
                    timestep_spacing=args.inference_scheduler_timestep_spacing,
                    rescale_betas_zero_snr=args.rescale_betas_zero_snr,
                    **scheduler_args,
                )
            if args.validation_torch_compile and not is_compiled_module(pipeline.unet):
                logger.warning(
                    f"Compiling the UNet for validation ({args.validation_torch_compile})"
                )
                pipeline.unet = torch.compile(
                    pipeline.unet,
                    mode=args.validation_torch_compile_mode,
                    fullgraph=False,
                )
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            # Save validation images
            val_save_dir = os.path.join(args.output_dir, "validation_images")
            if not os.path.exists(val_save_dir):
                os.makedirs(val_save_dir)

            validation_images = {}
            pipeline = pipeline.to(accelerator.device)
            extra_validation_kwargs = {}
            if not args.validation_randomize:
                extra_validation_kwargs["generator"] = torch.Generator(
                    device=accelerator.device
                ).manual_seed(args.validation_seed or args.seed or 0)
            _content = zip(
                validation_shortnames,
                validation_prompts,
                [None] * len(validation_prompts),
            )
            if "deepfloyd-stage2" in args.model_type:
                _content = StateTracker.get_validation_sample_images()
                logger.info(
                    f"Processing {len(_content)} DeepFloyd stage 2 validation images."
                )

            for _validation_prompt in tqdm(
                _content,
                leave=False,
                ncols=125,
                desc="Generating validation images",
            ):
                validation_shortname, validation_prompt, validation_sample = (
                    _validation_prompt
                )
                logger.debug(f"Validation image: {validation_prompt}")
                # Each validation prompt needs its own embed.
                if StateTracker.get_model_type() == "sdxl":
                    (
                        current_validation_prompt_embeds,
                        current_validation_pooled_embeds,
                    ) = embed_cache.compute_embeddings_for_prompts([validation_prompt])
                    if prompt_handler is not None:
                        for text_encoder in prompt_handler.text_encoders:
                            text_encoder = text_encoder.to(accelerator.device)
                        [
                            current_validation_prompt_embeds,
                            validation_negative_prompt_embeds,
                        ] = prompt_handler.compel.pad_conditioning_tensors_to_same_length(
                            [
                                current_validation_prompt_embeds,
                                validation_negative_prompt_embeds,
                            ]
                        )
                        for text_encoder in prompt_handler.text_encoders:
                            text_encoder = text_encoder.to("cpu")
                    current_validation_pooled_embeds = (
                        current_validation_pooled_embeds.to(
                            device=accelerator.device, dtype=weight_dtype
                        )
                    )
                    validation_negative_pooled_embeds = (
                        validation_negative_pooled_embeds.to(
                            device=accelerator.device, dtype=weight_dtype
                        )
                    )
                elif StateTracker.get_model_type() == "legacy":
                    validation_negative_pooled_embeds = None
                    current_validation_pooled_embeds = None
                    current_validation_prompt_embeds = (
                        embed_cache.compute_embeddings_for_prompts([validation_prompt])
                    )[0]
                    logger.debug(
                        f"Validations received the prompt embed: positive={current_validation_prompt_embeds.shape}, negative={validation_negative_prompt_embeds.shape}"
                    )
                    if (
                        prompt_handler is not None
                        and "deepfloyd" not in args.model_type
                    ):
                        for text_encoder in prompt_handler.text_encoders:
                            if text_encoder:
                                text_encoder = text_encoder.to(accelerator.device)
                        [
                            current_validation_prompt_embeds,
                            validation_negative_prompt_embeds,
                        ] = prompt_handler.compel.pad_conditioning_tensors_to_same_length(
                            [
                                current_validation_prompt_embeds,
                                validation_negative_prompt_embeds,
                            ]
                        )
                        for text_encoder in prompt_handler.text_encoders:
                            if text_encoder:
                                text_encoder = text_encoder.to(accelerator.device)
                current_validation_prompt_embeds = current_validation_prompt_embeds.to(
                    device=accelerator.device, dtype=weight_dtype
                )
                validation_negative_prompt_embeds = (
                    validation_negative_prompt_embeds.to(
                        device=accelerator.device, dtype=weight_dtype
                    )
                )

                # logger.debug(
                #     f"Generating validation image: {validation_prompt}"
                #     "\n Device allocations:"
                #     f"\n -> unet on {pipeline.unet.device}"
                #     f"\n -> text_encoder on {pipeline.text_encoder.device if pipeline.text_encoder is not None else None}"
                #     f"\n -> vae on {pipeline.vae.device if hasattr(pipeline, 'vae') else None}"
                #     f"\n -> current_validation_prompt_embeds on {current_validation_prompt_embeds.device}"
                #     f"\n -> current_validation_pooled_embeds on {current_validation_pooled_embeds.device if current_validation_pooled_embeds is not None else None}"
                #     f"\n -> validation_negative_prompt_embeds on {validation_negative_prompt_embeds.device}"
                #     f"\n -> validation_negative_pooled_embeds on {validation_negative_pooled_embeds.device if validation_negative_pooled_embeds is not None else None}"
                # )

                # logger.debug(
                #     f"Generating validation image: {validation_prompt}"
                #     f"\n Weight dtypes:"
                #     f"\n -> unet: {pipeline.unet.dtype}"
                #     f"\n -> text_encoder: {pipeline.text_encoder.dtype if pipeline.text_encoder is not None else None}"
                #     f"\n -> vae: {pipeline.vae.dtype}"
                #     f"\n -> current_validation_prompt_embeds: {current_validation_prompt_embeds.dtype}"
                #     f"\n -> current_validation_pooled_embeds: {current_validation_pooled_embeds.dtype}"
                #     f"\n -> validation_negative_prompt_embeds: {validation_negative_prompt_embeds.dtype}"
                #     f"\n -> validation_negative_pooled_embeds: {validation_negative_pooled_embeds.dtype}"
                # )
                # logger.debug(
                #     f"Generating validation image: {validation_prompt}"
                #     f"\n -> Number of images: {args.num_validation_images}"
                #     f"\n -> Number of inference steps: {args.validation_num_inference_steps}"
                #     f"\n -> Guidance scale: {args.validation_guidance}"
                #     f"\n -> Guidance rescale: {args.validation_guidance_rescale}"
                #     f"\n -> Resolution: {args.validation_resolution}"
                #     f"\n -> Extra validation kwargs: {extra_validation_kwargs}"
                # )
                if "deepfloyd" not in args.model_type:
                    extra_validation_kwargs["pooled_prompt_embeds"] = (
                        current_validation_pooled_embeds
                    )
                    extra_validation_kwargs["negative_pooled_prompt_embeds"] = (
                        validation_negative_pooled_embeds
                    )
                    extra_validation_kwargs["guidance_rescale"] = (
                        args.validation_guidance_rescale
                    )

                if validation_sample is not None:
                    # Resize the input sample so that we can validate the model's upscaling performance.
                    target_size, intermediary_size, aspect_ratio = (
                        MultiaspectImage.calculate_new_size_by_pixel_edge(
                            MultiaspectImage.calculate_image_aspect_ratio(
                                validation_sample.size[0] / validation_sample.size[1]
                            ),
                            64,
                        )
                    )
                    extra_validation_kwargs["image"] = validation_sample.resize(
                        target_size
                    )

                validation_resolutions = (
                    get_validation_resolutions()
                    if "deepfloyd-stage2" not in args.model_type
                    else ["base-256"]
                )
                logger.debug(f"Resolutions for validation: {validation_resolutions}")
                if validation_shortname not in validation_images:
                    validation_images[validation_shortname] = []

                for resolution in validation_resolutions:
                    if "deepfloyd-stage2" not in args.model_type:
                        validation_resolution_width, validation_resolution_height = (
                            resolution
                        )
                    else:
                        validation_resolution_width, validation_resolution_height = (
                            val * 4 for val in extra_validation_kwargs["image"].size
                        )
                    logger.info(
                        f"Processing width/height: {validation_resolution_width}x{validation_resolution_height}"
                    )
                    validation_images[validation_shortname].extend(
                        pipeline(
                            prompt_embeds=current_validation_prompt_embeds,
                            negative_prompt_embeds=validation_negative_prompt_embeds,
                            num_images_per_prompt=args.num_validation_images,
                            num_inference_steps=args.validation_num_inference_steps,
                            guidance_scale=args.validation_guidance,
                            height=MultiaspectImage._round_to_nearest_multiple(
                                int(validation_resolution_height)
                            ),
                            width=MultiaspectImage._round_to_nearest_multiple(
                                int(validation_resolution_width)
                            ),
                            **extra_validation_kwargs,
                        ).images
                    )
                validation_img_idx = 0
                for validation_image in validation_images[validation_shortname]:
                    validation_image.save(
                        os.path.join(
                            val_save_dir,
                            f"step_{global_step}_{validation_shortname}_{str(validation_resolutions[validation_img_idx])}.png",
                        )
                    )
                if StateTracker.get_webhook_handler() is not None:
                    StateTracker.get_webhook_handler().send(
                        f"Validation image for `{validation_shortname if validation_shortname != '' else '(blank shortname)'}`"
                        f"\nValidation prompt: `{validation_prompt if validation_prompt != '' else '(blank prompt)'}`",
                        images=validation_images[validation_shortname],
                    )

                logger.debug(f"Completed generating image: {validation_prompt}")

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    resolution_list = [
                        f"{res[0]}x{res[1]}" for res in get_validation_resolutions()
                    ]

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
                            logger.debug(f"Found a missing image - masking with a None")
                            wandb_images.append(None)
                        table.add_data(prompt_shortname, *wandb_images, mean_luminance)

                    # Log the table to Weights & Biases
                    tracker.log({"Validation Gallery": table}, step=global_step)

            if validation_type == "validation" and args.use_ema:
                # Switch back to the original UNet parameters.
                ema_unet.restore(unet.parameters())
            if not args.keep_vae_loaded and args.vae_cache_preprocess:
                # only delete the vae if we're not encoding embeds during training
                del vae
                vae = None
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
