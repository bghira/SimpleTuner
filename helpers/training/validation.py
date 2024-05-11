import torch, os, wandb, logging
from pathlib import Path
from tqdm import tqdm
from helpers.training.state_tracker import StateTracker
from diffusers.pipelines import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.training_utils import EMAModel
from diffusers.schedulers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
    DDIMScheduler,
    DDPMScheduler,
)
from diffusers.utils.torch_utils import is_compiled_module
from helpers.multiaspect.image import MultiaspectImage
from helpers.image_manipulation.brightness import calculate_luminance

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")

SCHEDULER_NAME_MAP = {
    "euler": EulerDiscreteScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "unipc": UniPCMultistepScheduler,
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
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


class Validation:
    def __init__(
        self,
        accelerator,
        prompt_handler,
        unet,
        args,
        validation_prompts,
        validation_shortnames,
        text_encoder_1,
        tokenizer,
        vae_path,
        weight_dtype,
        embed_cache,
        validation_negative_pooled_embeds,
        validation_negative_prompt_embeds,
        text_encoder_2,
        tokenizer_2,
        ema_unet,
        vae,
    ):
        self.accelerator = accelerator
        self.prompt_handler = prompt_handler
        self.unet = unet
        self.args = args
        self.save_dir = os.path.join(args.output_dir, "validation_images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.global_step = None
        self.global_resume_step = None
        self.text_encoder_1 = text_encoder_1
        self.tokenizer_1 = tokenizer
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2
        self.vae_path = vae_path
        self.validation_prompts = validation_prompts
        self.validation_shortnames = validation_shortnames
        self.validation_images = None
        self.weight_dtype = weight_dtype
        self.embed_cache = embed_cache
        self.validation_negative_pooled_embeds = validation_negative_pooled_embeds
        self.validation_negative_prompt_embeds = validation_negative_prompt_embeds
        self.ema_unet = ema_unet
        self.vae = vae
        self.pipeline = None
        self.validation_resolutions = (
            get_validation_resolutions()
            if "deepfloyd-stage2" not in args.model_type
            else ["base-256"]
        )

        self._update_state()

    def _pipeline_cls(self):
        model_type = StateTracker.get_model_type()
        if model_type == "sdxl":
            return StableDiffusionXLPipeline
        elif model_type == "legacy":
            if "deepfloyd-stage2" in self.args.model_type:
                from diffusers.pipelines import IFSuperResolutionPipeline

                return IFSuperResolutionPipeline
            else:
                return DiffusionPipeline

    def _gather_prompt_embeds(self, validation_prompt: str):
        prompt_embeds = {}
        if StateTracker.get_model_type() == "sdxl":
            (
                current_validation_prompt_embeds,
                current_validation_pooled_embeds,
            ) = self.embed_cache.compute_embeddings_for_prompts([validation_prompt])
            if self.prompt_handler is not None:
                for text_encoder in self.prompt_handler.text_encoders:
                    # Can't remember why we move this to the GPU right here..
                    text_encoder = text_encoder.to(self.accelerator.device)
                [
                    current_validation_prompt_embeds,
                    self.validation_negative_prompt_embeds,
                ] = self.prompt_handler.compel.pad_conditioning_tensors_to_same_length(
                    [
                        current_validation_prompt_embeds,
                        self.validation_negative_prompt_embeds,
                    ]
                )
                for text_encoder in self.prompt_handler.text_encoders:
                    # Or why we move it back...maybe it's a Compel oddity :(
                    text_encoder = text_encoder.to("cpu")
            current_validation_pooled_embeds = current_validation_pooled_embeds.to(
                device=self.accelerator.device, dtype=self.weight_dtype
            )
            self.validation_negative_pooled_embeds = (
                self.validation_negative_pooled_embeds.to(
                    device=self.accelerator.device, dtype=self.weight_dtype
                )
            )
            prompt_embeds["pooled_prompt_embeds"] = current_validation_pooled_embeds
            prompt_embeds["negative_pooled_prompt_embeds"] = (
                self.validation_negative_pooled_embeds
            )
        elif StateTracker.get_model_type() == "legacy":
            self.validation_negative_pooled_embeds = None
            current_validation_pooled_embeds = None
            current_validation_prompt_embeds = (
                self.embed_cache.compute_embeddings_for_prompts([validation_prompt])
            )[0]
            logger.debug(
                f"Validations received the prompt embed: positive={current_validation_prompt_embeds.shape}, negative={self.validation_negative_prompt_embeds.shape}"
            )
            if (
                self.prompt_handler is not None
                and "deepfloyd" not in self.args.model_type
            ):
                for text_encoder in self.prompt_handler.text_encoders:
                    if text_encoder:
                        text_encoder = text_encoder.to(self.accelerator.device)
                [
                    current_validation_prompt_embeds,
                    self.validation_negative_prompt_embeds,
                ] = self.prompt_handler.compel.pad_conditioning_tensors_to_same_length(
                    [
                        current_validation_prompt_embeds,
                        self.validation_negative_prompt_embeds,
                    ]
                )
                for text_encoder in self.prompt_handler.text_encoders:
                    if text_encoder:
                        text_encoder = text_encoder.to(self.accelerator.device)
                current_validation_prompt_embeds = current_validation_prompt_embeds.to(
                    device=self.accelerator.device, dtype=self.weight_dtype
                )
                self.validation_negative_prompt_embeds = (
                    self.validation_negative_prompt_embeds.to(
                        device=self.accelerator.device, dtype=self.weight_dtype
                    )
                )
        current_validation_prompt_embeds = current_validation_prompt_embeds.to(
            device=self.accelerator.device, dtype=self.weight_dtype
        )
        self.validation_negative_prompt_embeds = (
            self.validation_negative_prompt_embeds.to(
                device=self.accelerator.device, dtype=self.weight_dtype
            )
        )
        prompt_embeds["prompt_embeds"] = current_validation_prompt_embeds
        prompt_embeds["negative_prompt_embeds"] = self.validation_negative_prompt_embeds
        # If the prompt is an empty string, zero out all of the embeds:
        if validation_prompt == "":
            prompt_embeds = {
                key: torch.zeros_like(value).to(self.accelerator.device)
                for key, value in prompt_embeds.items()
            }

        return prompt_embeds

    def _update_state(self):
        """Updates internal state with the latest from StateTracker."""
        self.global_step = StateTracker.get_global_step()
        self.global_resume_step = StateTracker.get_global_resume_step() or 1

    def run_validations(
        self,
        step: int = 0,
        validation_type="intermediary",
        force_evaluation: bool = False,
        skip_execution: bool = False,
    ):
        self._update_state()
        should_validate = self.should_perform_validation(
            step, self.validation_prompts, validation_type
        )
        logger.debug(
            f"Should evaluate: {should_validate}, force evaluation: {force_evaluation}, skip execution: {skip_execution}"
        )
        if not should_validate and not force_evaluation:
            return self
        if should_validate and skip_execution:
            # If the validation would have fired off, we'll skip it.
            # This is useful at the end of training so we don't validate 2x.
            return self
        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message=f"Validations are generating.. this might take a minute! 🖼️",
                message_level="info",
            )

        if self.accelerator.is_main_process:
            logger.info("Starting validation process...")
            self.setup_pipeline(validation_type)
            self.process_prompts()
            self.finalize_validation(validation_type)
            logger.info("Validation process completed.")

        return self

    def should_perform_validation(self, step, validation_prompts, validation_type):
        should_do_intermediary_validation = (
            validation_prompts
            and self.global_step % self.args.validation_steps == 0
            and step % self.args.gradient_accumulation_steps == 0
            and self.global_step > self.global_resume_step
        )
        is_final_validation = validation_type == "final"
        return (
            is_final_validation or should_do_intermediary_validation
        ) and self.accelerator.is_main_process

    def setup_scheduler(self):
        scheduler_args = {}
        if "variance_type" in self.pipeline.scheduler.config:
            variance_type = self.pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type
        if "deepfloyd" in self.args.model_type:
            self.args.validation_noise_scheduler = "ddpm"

        self.pipeline.scheduler = SCHEDULER_NAME_MAP[
            self.args.validation_noise_scheduler
        ].from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="scheduler",
            prediction_type=self.args.prediction_type,
            timestep_spacing=self.args.inference_scheduler_timestep_spacing,
            rescale_betas_zero_snr=self.args.rescale_betas_zero_snr,
            **scheduler_args,
        )

    def setup_pipeline(self, validation_type):
        if validation_type == "intermediary" and self.args.use_ema:
            self.ema_unet.store(self.unet.parameters())
            self.ema_unet.copy_to(self.unet.parameters())

        if not self.pipeline:
            pipeline_cls = self._pipeline_cls()
            extra_pipeline_kwargs = {
                "text_encoder": self.text_encoder_1,
                "tokenizer": self.tokenizer_1,
                "vae": self.vae,
                "safety_checker": None,
            }
            if type(pipeline_cls) is StableDiffusionXLPipeline:
                del extra_pipeline_kwargs["safety_checker"]
                del extra_pipeline_kwargs["text_encoder"]
                del extra_pipeline_kwargs["tokenizer"]
                extra_pipeline_kwargs["text_encoder_1"] = self.text_encoder_1
                extra_pipeline_kwargs["text_encoder_2"] = self.text_encoder_2
                extra_pipeline_kwargs["tokenizer_1"] = self.tokenizer_1
                extra_pipeline_kwargs["tokenizer_2"] = self.tokenizer_2
            pipeline_kwargs = {
                "pretrained_model_name_or_path": self.args.pretrained_model_name_or_path,
                "unet": self.unet,
                "revision": self.args.revision,
                "torch_dtype": self.weight_dtype,
                **extra_pipeline_kwargs,
            }
            self.pipeline = pipeline_cls.from_pretrained(**pipeline_kwargs)
            if self.args.validation_torch_compile and not is_compiled_module(
                self.pipeline.unet
            ):
                logger.warning(
                    f"Compiling the UNet for validation ({self.args.validation_torch_compile})"
                )
                self.pipeline.unet = torch.compile(
                    self.pipeline.unet,
                    mode=self.args.validation_torch_compile_mode,
                    fullgraph=False,
                )
        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

    def process_prompts(self):
        """Processes each validation prompt and logs the result."""
        validation_images = {}
        for shortname, prompt in zip(
            self.validation_shortnames, self.validation_prompts
        ):
            logger.debug(f"Processing validation for prompt: {prompt}")
            validation_images.update(self.validate_prompt(prompt, shortname))
            self._save_images(validation_images, shortname, prompt)
            self._log_validations_to_webhook(validation_images, shortname, prompt)
            logger.debug(f"Completed generating image: {prompt}")
        self.validation_images = validation_images
        self._log_validations_to_trackers(validation_images)

    def validate_prompt(self, prompt, validation_shortname):
        """Generate validation images for a single prompt."""
        # Placeholder for actual image generation and logging
        logger.info(f"Validating prompt: {prompt}")
        validation_images = {}
        for resolution in self.validation_resolutions:
            extra_validation_kwargs = {}
            if not self.args.validation_randomize:
                extra_validation_kwargs["generator"] = torch.Generator(
                    device=self.accelerator.device
                ).manual_seed(self.args.validation_seed or self.args.seed or 0)
                logger.info(
                    f"Using a generator? {extra_validation_kwargs['generator']}"
                )
            if "deepfloyd-stage2" not in self.args.model_type:
                validation_resolution_width, validation_resolution_height = resolution
            else:
                validation_resolution_width, validation_resolution_height = (
                    val * 4 for val in extra_validation_kwargs["image"].size
                )
            if "deepfloyd" not in self.args.model_type:
                extra_validation_kwargs["guidance_rescale"] = (
                    self.args.validation_guidance_rescale
                )

            logger.debug(
                f"Processing width/height: {validation_resolution_width}x{validation_resolution_height}"
            )
            extra_validation_kwargs.update(self._gather_prompt_embeds(prompt))
            if validation_shortname not in validation_images:
                validation_images[validation_shortname] = []
            try:
                pipeline_kwargs = {
                    "num_images_per_prompt": self.args.num_validation_images,
                    "num_inference_steps": self.args.validation_num_inference_steps,
                    "guidance_scale": self.args.validation_guidance,
                    "height": MultiaspectImage._round_to_nearest_multiple(
                        int(validation_resolution_height)
                    ),
                    "width": MultiaspectImage._round_to_nearest_multiple(
                        int(validation_resolution_width)
                    ),
                    **extra_validation_kwargs,
                }
                logger.info(f"Image being generated with parameters: {pipeline_kwargs}")
                # Print the device attr of any parameters that have one
                for key, value in pipeline_kwargs.items():
                    if hasattr(value, "device"):
                        logger.info(f"Device for {key}: {value.device}")
                for key, value in self.pipeline.components.items():
                    if hasattr(value, "device"):
                        logger.info(f"Device for {key}: {value.device}")
                validation_images[validation_shortname].extend(
                    self.pipeline(**pipeline_kwargs).images
                )
            except Exception as e:
                logger.error(f"Error generating validation image: {e}")
                continue

        return validation_images

    def _save_images(self, validation_images, validation_shortname, validation_prompt):
        validation_img_idx = 0
        for validation_image in validation_images[validation_shortname]:
            validation_image.save(
                os.path.join(
                    self.save_dir,
                    f"step_{StateTracker.get_global_step()}_{validation_shortname}_{str(self.validation_resolutions[validation_img_idx])}.png",
                )
            )

    def _log_validations_to_webhook(
        self, validation_images, validation_shortname, validation_prompt
    ):
        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                f"Validation image for `{validation_shortname if validation_shortname != '' else '(blank shortname)'}`"
                f"\nValidation prompt: `{validation_prompt if validation_prompt != '' else '(blank prompt)'}`",
                images=validation_images[validation_shortname],
            )

    def _log_validations_to_trackers(self, validation_images):
        for tracker in self.accelerator.trackers:
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
                tracker.log(
                    {"Validation Gallery": table}, step=StateTracker.get_global_step()
                )

    def finalize_validation(self, validation_type):
        """Cleans up and restores original state if necessary."""
        if validation_type == "intermediary" and self.args.use_ema:
            self.ema_unet.restore(self.unet.parameters())
        if not self.args.keep_vae_loaded and self.args.vae_cache_preprocess:
            self.vae = None
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
