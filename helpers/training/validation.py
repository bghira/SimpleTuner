import torch, os, wandb, logging
from pathlib import Path
from tqdm import tqdm
from helpers.training.wrappers import unwrap_model
from PIL import Image
from helpers.training.state_tracker import StateTracker
from helpers.sdxl.pipeline import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from helpers.legacy.pipeline import StableDiffusionPipeline
from helpers.legacy.validation import retrieve_validation_images
from diffusers.training_utils import EMAModel
from diffusers.schedulers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    DDIMScheduler,
    DDPMScheduler,
)
from diffusers.utils.torch_utils import is_compiled_module
from helpers.multiaspect.image import MultiaspectImage
from helpers.image_manipulation.brightness import calculate_luminance

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")

try:
    from helpers.sd3.pipeline import (
        StableDiffusion3Pipeline,
        StableDiffusion3Img2ImgPipeline,
    )
except ImportError:
    logger.error(
        f"Stable Diffusion 3 not available in this release of Diffusers. Please upgrade."
    )
    raise ImportError()

SCHEDULER_NAME_MAP = {
    "euler": EulerDiscreteScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "flow-match": FlowMatchEulerDiscreteScheduler,
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
        transformer,
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
        ema_model,
        vae,
        controlnet=None,
        text_encoder_3=None,
        tokenizer_3=None,
    ):
        self.accelerator = accelerator
        self.prompt_handler = prompt_handler
        self.unet = unet
        self.transformer = transformer
        self.controlnet = controlnet
        self.args = args
        self.save_dir = os.path.join(args.output_dir, "validation_images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
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
        self.validation_negative_prompt_mask = None
        self.validation_negative_pooled_embeds = validation_negative_pooled_embeds
        self.validation_negative_prompt_embeds = (
            validation_negative_prompt_embeds
            if (
                type(validation_negative_prompt_embeds) is not list
                and type(validation_negative_prompt_embeds) is not tuple
            )
            else validation_negative_prompt_embeds[0]
        )
        self.ema_model = ema_model
        self.vae = vae
        self.pipeline = None
        self._discover_validation_input_samples()
        self.validation_resolutions = (
            get_validation_resolutions()
            if "deepfloyd-stage2" not in args.model_type
            else ["base-256"]
        )
        self.text_encoder_3 = text_encoder_3
        self.tokenizer_3 = tokenizer_3

        self._update_state()

    def _validation_seed_source(self):
        if self.args.validation_seed_source == "gpu":
            return self.accelerator.device
        elif self.args.validation_seed_source == "cpu":
            return "cpu"
        else:
            raise Exception("Unknown validation seed source. Options: cpu, gpu")

    def _get_generator(self):
        _validation_seed_source = self._validation_seed_source()
        _generator = torch.Generator(device=_validation_seed_source).manual_seed(
            self.args.validation_seed or self.args.seed or 0
        )
        return _generator

    def clear_text_encoders(self):
        """
        Sets all text encoders to None.

        Returns:
            None
        """
        self.text_encoder_1 = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None

    def init_vae(self):
        from diffusers import AutoencoderKL

        args = StateTracker.get_args()
        vae_path = (
            args.pretrained_model_name_or_path
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        precached_vae = StateTracker.get_vae()
        logger.debug(
            f"Was the VAE loaded? {precached_vae if precached_vae is None else 'Yes'}"
        )
        self.vae = precached_vae or AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            force_upcast=False,
        ).to(self.accelerator.device)
        StateTracker.set_vae(self.vae)

        return self.vae

    def _discover_validation_input_samples(self):
        """
        If we have some workflow that requires image inputs for validation, we'll bind those now.

        Returns:
            Validation object (self)
        """
        self.validation_image_inputs = None
        if (
            "deepfloyd-stage2" in self.args.model_type
            or self.args.validation_using_datasets
            or self.args.controlnet
        ):
            self.validation_image_inputs = retrieve_validation_images()
            # Validation inputs are in the format of a list of tuples:
            # [(shortname, prompt, image), ...]
            logger.debug(
                f"Image inputs discovered for validation: {self.validation_image_inputs}"
            )

    def _pipeline_cls(self):
        model_type = StateTracker.get_model_type()
        if model_type == "sdxl":
            if self.args.controlnet:
                from diffusers.pipelines import StableDiffusionXLControlNetPipeline

                return StableDiffusionXLControlNetPipeline
            if self.args.validation_using_datasets:
                return StableDiffusionXLImg2ImgPipeline
            return StableDiffusionXLPipeline
        elif model_type == "legacy":
            if "deepfloyd-stage2" in self.args.model_type:
                from diffusers.pipelines import IFSuperResolutionPipeline

                return IFSuperResolutionPipeline
            return StableDiffusionPipeline
        elif model_type == "sd3":
            if self.args.controlnet:
                raise Exception(f"SD3 ControlNet is not yet supported.")
            if self.args.validation_using_datasets:
                return StableDiffusion3Img2ImgPipeline
            return StableDiffusion3Pipeline
        elif model_type == "pixart_sigma":
            if self.args.controlnet:
                raise Exception(
                    "PixArt Sigma ControlNet inference validation is not yet supported."
                )
            if self.args.validation_using_datasets:
                raise Exception(
                    "PixArt Sigma inference validation using img2img is not yet supported. Please remove --validation_using_datasets."
                )
            from helpers.pixart.pipeline import PixArtSigmaPipeline

            return PixArtSigmaPipeline
        elif model_type == "aura_diffusion":
            if self.args.controlnet:
                raise Exception(
                    "Aura Diffusion ControlNet inference validation is not yet supported."
                )
            if self.args.validation_using_datasets:
                raise Exception(
                    "Aura Diffusion inference validation using img2img is not yet supported. Please remove --validation_using_datasets."
                )
            try:
                from helpers.aura_diffusion.pipeline import AuraFlowPipeline
            except Exception as e:
                logger.error(
                    f"Could not import Aura Diffusion pipeline. Perhaps you need a git-source version of Diffusers."
                )
                raise NotImplementedError("Aura Diffusion pipeline not available.")

            return AuraFlowPipeline

    def _gather_prompt_embeds(self, validation_prompt: str):
        prompt_embeds = {}
        current_validation_prompt_mask = None
        if (
            StateTracker.get_model_type() == "sdxl"
            or StateTracker.get_model_type() == "sd3"
        ):
            (
                current_validation_prompt_embeds,
                current_validation_pooled_embeds,
            ) = self.embed_cache.compute_embeddings_for_prompts([validation_prompt])
            if (
                self.prompt_handler is not None
                and not StateTracker.get_model_type() == "sd3"
            ):
                for text_encoder in self.prompt_handler.text_encoders:
                    # Can't remember why we move this to the GPU right here..
                    if text_encoder is not None:
                        text_encoder = text_encoder.to(
                            device=self.accelerator.device, dtype=self.weight_dtype
                        )
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
                    if text_encoder is not None:
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
        elif (
            StateTracker.get_model_type() == "legacy"
            or StateTracker.get_model_type() == "pixart_sigma"
            or StateTracker.get_model_type() == "aura_diffusion"
        ):
            self.validation_negative_pooled_embeds = None
            current_validation_pooled_embeds = None
            current_validation_prompt_embeds = (
                self.embed_cache.compute_embeddings_for_prompts([validation_prompt])
            )
            if self.args.pixart_sigma:
                current_validation_prompt_embeds, current_validation_prompt_mask = (
                    current_validation_prompt_embeds
                )
                current_validation_prompt_embeds = current_validation_prompt_embeds[0]
                if (
                    type(self.validation_negative_prompt_embeds) is tuple
                    or type(self.validation_negative_prompt_embeds) is list
                ):
                    (
                        self.validation_negative_prompt_embeds,
                        self.validation_negative_prompt_mask,
                    ) = self.validation_negative_prompt_embeds[0]
            elif self.args.aura_diffusion:
                raise NotImplementedError(
                    "Aura Diffusion validation prompt gathering is not yet implemented."
                )
            else:
                current_validation_prompt_embeds = current_validation_prompt_embeds[0]
            # logger.debug(
            #     f"Validations received the prompt embed: ({type(current_validation_prompt_embeds)}) positive={current_validation_prompt_embeds.shape if type(current_validation_prompt_embeds) is not list else current_validation_prompt_embeds[0].shape},"
            #     f" ({type(self.validation_negative_prompt_embeds)}) negative={self.validation_negative_prompt_embeds.shape if type(self.validation_negative_prompt_embeds) is not list else self.validation_negative_prompt_embeds[0].shape}"
            # )
            # logger.debug(
            #     f"Dtypes: {current_validation_prompt_embeds.dtype}, {self.validation_negative_prompt_embeds.dtype}"
            # )
            if (
                self.prompt_handler is not None
                and "deepfloyd" not in self.args.model_type
                and "pixart" not in self.args.model_type
            ):
                for text_encoder in self.prompt_handler.text_encoders:
                    if text_encoder:
                        text_encoder = text_encoder.to(
                            self.accelerator.device, dtype=self.weight_dtype
                        )
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
                        text_encoder = text_encoder.to(
                            self.accelerator.device, dtype=self.weight_dtype
                        )
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
        if validation_prompt == "" and "deepfloyd" not in self.args.model_type:
            prompt_embeds = {
                key: torch.zeros_like(value).to(
                    device=self.accelerator.device, dtype=self.weight_dtype
                )
                for key, value in prompt_embeds.items()
            }
        if StateTracker.get_model_type() == "pixart_sigma":
            prompt_embeds["prompt_mask"] = current_validation_prompt_mask
            prompt_embeds["negative_mask"] = self.validation_negative_prompt_mask

        if StateTracker.get_model_type() == "aura_diffusion":
            raise NotImplementedError(
                "Aura Diffusion text embed gathering is not yet fully implemented."
            )

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
            logger.debug("Starting validation process...")
            self.setup_pipeline(validation_type)
            if self.pipeline is None:
                logger.error(
                    "Not able to run validations, we did not obtain a valid pipeline."
                )
                self.validation_images = None
                return self
            self.setup_scheduler()
            self.process_prompts()
            self.finalize_validation(validation_type)
            logger.debug("Validation process completed.")
            self.clean_pipeline()

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
        if self.args.validation_noise_scheduler is None:
            return

        scheduler_args = {}
        if "variance_type" in self.pipeline.scheduler.config:
            variance_type = self.pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type
        if "deepfloyd" in self.args.model_type:
            self.args.validation_noise_scheduler = "ddpm"
        if self.args.sd3 and not self.args.sd3_uses_diffusion:
            # NO TOUCHIE FOR FLOW-MATCHING.
            # Touchie for diffusion though.
            return

        self.pipeline.scheduler = SCHEDULER_NAME_MAP[
            self.args.validation_noise_scheduler
        ].from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="scheduler",
            revision=self.args.revision,
            prediction_type=self.args.prediction_type,
            timestep_spacing=self.args.inference_scheduler_timestep_spacing,
            rescale_betas_zero_snr=self.args.rescale_betas_zero_snr,
            **scheduler_args,
        )

    def setup_pipeline(self, validation_type, enable_ema_model: bool = True):
        if validation_type == "intermediary" and self.args.use_ema:
            if enable_ema_model:
                if self.unet is not None:
                    self.ema_model.store(self.unet.parameters())
                    self.ema_model.copy_to(self.unet.parameters())
                if self.transformer is not None:
                    self.ema_model.store(self.transformer.parameters())
                    self.ema_model.copy_to(self.transformer.parameters())
                if self.args.ema_device != "accelerator":
                    logger.info("Moving EMA weights to GPU for inference.")
                    self.ema_model.to(self.accelerator.device)
            else:
                logger.debug(
                    f"Skipping EMA model setup for validation, as enable_ema_model=False."
                )

        if self.pipeline is None:
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
                if validation_type == "final":
                    if self.text_encoder_1 is not None:
                        extra_pipeline_kwargs["text_encoder_1"] = unwrap_model(
                            self.accelerator, self.text_encoder_1
                        )
                        extra_pipeline_kwargs["tokenizer_1"] = self.tokenizer_1
                        if self.text_encoder_2 is not None:
                            extra_pipeline_kwargs["text_encoder_2"] = unwrap_model(
                                self.accelerator, self.text_encoder_2
                            )
                            extra_pipeline_kwargs["tokenizer_2"] = self.tokenizer_2
                else:
                    extra_pipeline_kwargs["text_encoder_1"] = None
                    extra_pipeline_kwargs["tokenizer_1"] = None
                    extra_pipeline_kwargs["text_encoder_2"] = None
                    extra_pipeline_kwargs["tokenizer_2"] = None

            if self.args.controlnet:
                # ControlNet training has an additional adapter thingy.
                extra_pipeline_kwargs["controlnet"] = unwrap_model(
                    self.accelerator, self.controlnet
                )
            if self.unet is not None:
                extra_pipeline_kwargs["unet"] = unwrap_model(
                    self.accelerator, self.unet
                )

            if self.transformer is not None:
                extra_pipeline_kwargs["transformer"] = unwrap_model(
                    self.accelerator, self.transformer
                )

            if self.args.sd3 and self.args.train_text_encoder:
                if self.text_encoder_1 is not None:
                    extra_pipeline_kwargs["text_encoder"] = unwrap_model(
                        self.accelerator, self.text_encoder_1
                    )
                    extra_pipeline_kwargs["tokenizer"] = self.tokenizer_1
                if self.text_encoder_2 is not None:
                    extra_pipeline_kwargs["text_encoder_2"] = unwrap_model(
                        self.accelerator, self.text_encoder_2
                    )
                    extra_pipeline_kwargs["tokenizer_2"] = self.tokenizer_2
                if self.text_encoder_3 is not None:
                    extra_pipeline_kwargs["text_encoder_3"] = unwrap_model(
                        self.accelerator, self.text_encoder_3
                    )
                    extra_pipeline_kwargs["tokenizer_3"] = self.tokenizer_3

            if self.vae is None:
                extra_pipeline_kwargs["vae"] = self.init_vae()

            pipeline_kwargs = {
                "pretrained_model_name_or_path": self.args.pretrained_model_name_or_path,
                "revision": self.args.revision,
                "variant": self.args.variant,
                "torch_dtype": self.weight_dtype,
                **extra_pipeline_kwargs,
            }
            logger.debug(f"Initialising pipeline with kwargs: {pipeline_kwargs}")
            attempt = 0
            while attempt < 3:
                attempt += 1
                try:
                    self.pipeline = pipeline_cls.from_pretrained(**pipeline_kwargs)
                except Exception as e:
                    logger.error(e)
                    continue
                return None
            if self.args.validation_torch_compile:
                if self.unet is not None and not is_compiled_module(self.unet):
                    logger.warning(
                        f"Compiling the UNet for validation ({self.args.validation_torch_compile})"
                    )
                    self.pipeline.unet = torch.compile(
                        self.pipeline.unet,
                        mode=self.args.validation_torch_compile_mode,
                        fullgraph=False,
                    )
                if self.transformer is not None and not is_compiled_module(
                    self.transformer
                ):
                    logger.warning(
                        f"Compiling the transformer for validation ({self.args.validation_torch_compile})"
                    )
                    self.pipeline.transformer = torch.compile(
                        self.pipeline.transformer,
                        mode=self.args.validation_torch_compile_mode,
                        fullgraph=False,
                    )

        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

    def clean_pipeline(self):
        """Remove the pipeline."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

    def process_prompts(self):
        """Processes each validation prompt and logs the result."""
        validation_images = {}
        _content = zip(self.validation_shortnames, self.validation_prompts)
        total_samples = (
            len(self.validation_shortnames)
            if self.validation_shortnames is not None
            else 0
        )
        if self.validation_image_inputs:
            # Override the pipeline inputs to be entirely based upon the validation image inputs.
            _content = self.validation_image_inputs
            total_samples = len(_content) if _content is not None else 0
        for content in tqdm(
            _content if _content else [],
            desc="Processing validation prompts",
            total=total_samples,
            leave=False,
            position=1,
        ):
            validation_input_image = None
            logger.debug(f"content: {content}")
            if len(content) == 3:
                shortname, prompt, validation_input_image = content
            elif len(content) == 2:
                shortname, prompt = content
            else:
                raise ValueError(
                    f"Validation content is not in the correct format: {content}"
                )
            logger.debug(f"Processing validation for prompt: {prompt}")
            validation_images.update(
                self.validate_prompt(prompt, shortname, validation_input_image)
            )
            self._save_images(validation_images, shortname, prompt)
            self._log_validations_to_webhook(validation_images, shortname, prompt)
            logger.debug(f"Completed generating image: {prompt}")
        self.validation_images = validation_images
        self._log_validations_to_trackers(validation_images)

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

    def validate_prompt(
        self, prompt, validation_shortname, validation_input_image=None
    ):
        """Generate validation images for a single prompt."""
        # Placeholder for actual image generation and logging
        logger.debug(f"Validating prompt: {prompt}")
        validation_images = {}
        for resolution in self.validation_resolutions:
            extra_validation_kwargs = {}
            if not self.args.validation_randomize:
                extra_validation_kwargs["generator"] = self._get_generator()
                logger.debug(
                    f"Using a generator? {extra_validation_kwargs['generator']}"
                )
            if validation_input_image is not None:
                extra_validation_kwargs["image"] = validation_input_image
                if "deepfloyd-stage2" in self.args.model_type:
                    validation_resolution_width, validation_resolution_height = (
                        val * 4 for val in extra_validation_kwargs["image"].size
                    )
                elif self.args.controlnet or self.args.validation_using_datasets:
                    validation_resolution_width, validation_resolution_height = (
                        extra_validation_kwargs["image"].size
                    )
                else:
                    raise ValueError(
                        "Validation input images are not supported for this model type."
                    )
            else:
                validation_resolution_width, validation_resolution_height = resolution

            if "deepfloyd" not in self.args.model_type and not self.args.sd3:
                extra_validation_kwargs["guidance_rescale"] = (
                    self.args.validation_guidance_rescale
                )
            if StateTracker.get_args().validation_using_datasets:
                extra_validation_kwargs["strength"] = getattr(
                    self.args, "validation_strength", 0.2
                )
                logger.debug(
                    f"Set validation image denoise strength to {extra_validation_kwargs['strength']}"
                )

            logger.debug(
                f"Processing width/height: {validation_resolution_width}x{validation_resolution_height}"
            )
            if validation_shortname not in validation_images:
                validation_images[validation_shortname] = []
            try:
                extra_validation_kwargs.update(self._gather_prompt_embeds(prompt))
            except Exception as e:
                import traceback

                logger.error(
                    f"Error gathering text embed for validation prompt {prompt}: {e}, traceback: {traceback.format_exc()}"
                )
                continue
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
                logger.debug(
                    f"Image being generated with parameters: {pipeline_kwargs}"
                )
                # Print the device attr of any parameters that have one
                for key, value in pipeline_kwargs.items():
                    if hasattr(value, "device"):
                        logger.debug(f"Device for {key}: {value.device}")
                for key, value in self.pipeline.components.items():
                    if hasattr(value, "device"):
                        logger.debug(f"Device for {key}: {value.device}")
                if StateTracker.get_model_type() == "pixart_sigma":
                    if pipeline_kwargs.get("negative_prompt") is not None:
                        del pipeline_kwargs["negative_prompt"]
                    if pipeline_kwargs.get("prompt") is not None:
                        del pipeline_kwargs["prompt"]
                    pipeline_kwargs["prompt_attention_mask"] = pipeline_kwargs.pop(
                        "prompt_mask"
                    )[0].to(device=self.accelerator.device, dtype=self.weight_dtype)
                    pipeline_kwargs["negative_prompt_attention_mask"] = torch.unsqueeze(
                        pipeline_kwargs.pop("negative_mask")[0], dim=0
                    ).to(device=self.accelerator.device, dtype=self.weight_dtype)
                if StateTracker.get_model_type() == "aura_diffusion":
                    raise NotImplementedError(
                        "Aura Diffusion validation image generation is not yet implemented."
                    )

                validation_image_results = self.pipeline(**pipeline_kwargs).images
                if self.args.controlnet:
                    validation_image_results = self.stitch_conditioning_images(
                        validation_image_results, extra_validation_kwargs["image"]
                    )
                validation_images[validation_shortname].extend(validation_image_results)
            except Exception as e:
                import traceback

                logger.error(
                    f"Error generating validation image: {e}, {traceback.format_exc()}"
                )
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

    def finalize_validation(self, validation_type, enable_ema_model: bool = True):
        """Cleans up and restores original state if necessary."""
        if validation_type == "intermediary" and self.args.use_ema:
            if enable_ema_model:
                if self.unet is not None:
                    self.ema_model.restore(self.unet.parameters())
                if self.transformer is not None:
                    self.ema_model.restore(self.transformer.parameters())
                if self.args.ema_device != "accelerator":
                    self.ema_model.to(self.args.ema_device)
            else:
                logger.debug(
                    f"Skipping EMA model restoration for validation, as enable_ema_model=False."
                )
        if not self.args.keep_vae_loaded and self.args.vae_cache_preprocess:
            self.vae = None
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
