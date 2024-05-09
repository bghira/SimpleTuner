import os
import torch
from pathlib import Path
from tqdm import tqdm
from helpers.training.state_tracker import StateTracker
from diffusers.pipelines import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.training_utils import EMAModel
import logging

logger = logging.getLogger(__name__)


class Validation:
    def __init__(
        self,
        accelerator,
        prompt_handler,
        unet,
        args,
        validation_prompts,
        validation_shortnames,
        step,
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
        self.args = args
        self.unet = unet
        self.text_encoder_1 = text_encoder_1
        self.tokenizer = tokenizer
        self.vae_path = vae_path
        self.weight_dtype = weight_dtype
        self.embed_cache = embed_cache
        self.pipeline = pipeline
        self._update_state()

    def _update_state(self):
        """Updates internal state with the latest from StateTracker."""
        self.global_step = StateTracker.get_global_step()
        self.global_resume_step = StateTracker.get_global_resume_step() or 1

    def run_validations(
        self,
        validation_prompts,
        validation_shortnames,
        step,
        validation_type="training",
    ):
        self._update_state()
        should_validate = self.should_perform_validation(
            step, validation_prompts, validation_type
        )
        if not should_validate:
            return

        if self.accelerator.is_main_process:
            logger.info("Starting validation process...")
            self.setup_pipeline(validation_type)
            self.process_prompts(validation_prompts, validation_shortnames)
            self.finalize_validation(validation_type)
            logger.info("Validation process completed.")

    def should_perform_validation(self, step, validation_prompts, validation_type):
        should_do_intermediary_validation = (
            validation_prompts
            and self.global_step % self.args.validation_steps == 0
            and step % self.args.gradient_accumulation_steps == 0
            and self.global_step > self.global_resume_step
        )
        is_final_validation = validation_type == "finish"
        return (
            is_final_validation or should_do_intermediary_validation
        ) and self.accelerator.is_main_process

    def setup_pipeline(self, validation_type):
        if validation_type == "validation" and self.args.use_ema:
            self.ema_unet.store(self.unet.parameters())
            self.ema_unet.copy_to(self.unet.parameters())

        if not self.pipeline:
            self.initialize_pipeline()

    def initialize_pipeline(self):
        """Initializes the validation pipeline based on model type and other configurations."""
        model_type = StateTracker.get_model_type()
        if model_type == "sdxl":
            pipeline_cls = StableDiffusionXLPipeline
        elif model_type == "legacy":
            if "deepfloyd-stage2" in self.args.model_type:
                from diffusers.pipelines import IFSuperResolutionPipeline

                pipeline_cls = IFSuperResolutionPipeline
            else:
                pipeline_cls = DiffusionPipeline

        self.pipeline = pipeline_cls.from_pretrained(
            self.args.pretrained_model_name_or_path,
            unet=self.unet,
            text_encoder=self.text_encoder_1,
            tokenizer=self.tokenizer,
            vae=self.vae,
            revision=self.args.revision,
            safety_checker=None,
            torch_dtype=self.weight_dtype,
        )

    def process_prompts(self, validation_prompts, validation_shortnames):
        """Processes each validation prompt and logs the result."""
        for shortname, prompt in zip(validation_shortnames, validation_prompts):
            logger.debug(f"Processing validation for prompt: {prompt}")
            self.validate_prompt(prompt, shortname)

    def validate_prompt(self, prompt, shortname):
        """Generate validation images for a single prompt."""
        # Placeholder for actual image generation and logging
        logger.info(f"Validating prompt: {prompt}")

    def finalize_validation(self, validation_type):
        """Cleans up and restores original state if necessary."""
        if validation_type == "validation" and self.args.use_ema:
            self.ema_unet.restore(self.unet.parameters())
        if not self.args.keep_vae_loaded and self.args.vae_cache_preprocess:
            del self.vae
            self.vae = None
        del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
