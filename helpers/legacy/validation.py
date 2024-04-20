import logging, os, torch, numpy as np
from tqdm import tqdm
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
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


def log_validations(
    accelerator,
    prompt_handler,
    unet,
    args,
    validation_prompts,
    validation_shortnames,
    global_step,
    resume_global_step,
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
    if accelerator.is_main_process:
        logger.debug(
            f"Performing validation every {args.validation_steps} steps."
            f" We are on step {global_step} and have {len(validation_prompts)} validation prompts."
            f" We have {global_step} steps of progress done and are resuming from {resume_global_step}."
            f" We are on step {step} of the current epoch. We have {len(validation_prompts)} validation prompts."
            f" We have {step % args.gradient_accumulation_steps} gradient accumulation steps remaining."
        )
        if validation_type == "finish" or (
            validation_prompts
            and global_step % args.validation_steps == 0
            and step % args.gradient_accumulation_steps == 0
            and StateTracker.get_global_step() > resume_global_step
        ):
            logger.debug(
                f"We might want to process validations, because we have {len(validation_prompts)} validation prompts,"
                f" and we are on step {global_step} which meshes with our specified interval of {args.validation_steps} steps."
            )
            if (
                validation_prompts is None
                or validation_prompts == []
                or args.num_validation_images is None
                or args.num_validation_images <= 0
            ):
                logger.warning(
                    f"Not generating any validation images for this checkpoint. Live dangerously and prosper, pal!"
                )
                return
            logger.debug(
                f"We have valid prompts to process, this is looking better for our decision tree.."
            )

            logger.info(
                f"Running validation... \n Generating {len(validation_prompts)} images."
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
                    pipeline = pipeline_cls.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=unwrap_model(accelerator, unet),
                        text_encoder=text_encoder_1,
                        tokenizer=None,
                        vae=vae,
                        revision=args.revision,
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

            validation_images = []
            pipeline = pipeline.to(accelerator.device)
            extra_validation_kwargs = {}
            if not args.validation_randomize:
                extra_validation_kwargs["generator"] = torch.Generator(
                    device=accelerator.device
                ).manual_seed(args.validation_seed or args.seed or 0)
            for validation_prompt in tqdm(
                validation_prompts,
                leave=False,
                ncols=125,
                desc="Generating validation images",
            ):
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
                        args.validation_guidance_rescale,
                    )
                validation_images.extend(
                    pipeline(
                        prompt_embeds=current_validation_prompt_embeds,
                        negative_prompt_embeds=validation_negative_prompt_embeds,
                        num_images_per_prompt=args.num_validation_images,
                        num_inference_steps=args.validation_num_inference_steps,
                        guidance_scale=args.validation_guidance,
                        height=int(args.validation_resolution),
                        width=int(args.validation_resolution),
                        **extra_validation_kwargs,
                    ).images
                )
                validation_images[-1].save(
                    os.path.join(
                        val_save_dir,
                        f"step_{global_step}_val_img_{len(validation_images)}.png",
                    )
                )

                logger.debug(f"Completed generating image: {validation_prompt}")

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    validation_document = {}
                    validation_luminance = []
                    for idx, validation_image in enumerate(validation_images):
                        # Create a WandB entry containing each image.
                        validation_document[validation_shortnames[idx]] = wandb.Image(
                            validation_image
                        )
                        # Compute the luminance of each image.
                        validation_luminance.append(
                            calculate_luminance(validation_image)
                        )
                    # Compute the mean luminance across all samples:
                    validation_luminance = torch.tensor(validation_luminance)
                    validation_document["validation_luminance"] = (
                        validation_luminance.mean()
                    )
                    del validation_luminance
                    tracker.log(validation_document, step=global_step)

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
