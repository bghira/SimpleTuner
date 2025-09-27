import logging
import os

import torch
from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.sd1x.pipeline import (
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class StableDiffusion1(ImageModelFoundation):
    NAME = "Stable Diffusion 1.x"
    PREDICTION_TYPE = PredictionTypes.EPSILON
    MODEL_TYPE = ModelTypes.UNET
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 4
    DEFAULT_NOISE_SCHEDULER = "ddim"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default seems to help more with SD3.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = UNet2DConditionModel
    MODEL_SUBFOLDER = "unet"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: StableDiffusionPipeline,
        PipelineTypes.IMG2IMG: StableDiffusionImg2ImgPipeline,
        PipelineTypes.CONTROLNET: StableDiffusionControlNetPipeline,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "dreamshaper"
    HUGGINGFACE_PATHS = {
        "1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "1.4": "CompVis/stable-diffusion-v1-4",
        "dreamshaper": "Lykon/dreamshaper-8",
        "realvis": "SG161222/RealVisXL_V5.0",
    }
    MODEL_LICENSE = "openrail++"

    SUPPORTS_TEXT_ENCODER_TRAINING = True
    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "CLIP-L/14",
            "tokenizer": CLIPTokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": CLIPTextModel,
        },
    }

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        self.config:
            text_embedding (torch.Tensor): The embed to adjust.

        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        prompt_embeds = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor, prompt: str) -> dict:
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """
        prompt_embeds, _ = self.pipelines[PipelineTypes.TEXT2IMG].encode_prompt(
            prompt=prompts,
            device=self.accelerator.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

        return prompt_embeds

    def controlnet_init(self):
        logger.info("Creating the controlnet..")
        if self.config.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            self.controlnet = ControlNetModel.from_pretrained(self.config.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from base model")
            self.controlnet = ControlNetModel.from_unet(self.unwrap_model(self.model))
        self.controlnet.to(self.accelerator.device, self.config.weight_dtype)

    def controlnet_predict(self, prepared_batch: dict) -> dict:
        # ControlNet conditioning.
        controlnet_image = prepared_batch["conditioning_pixel_values"].to(
            device=self.accelerator.device, dtype=self.config.weight_dtype
        )
        logger.debug(f"Image shape: {controlnet_image.shape}")
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            prepared_batch["noisy_latents"].to(device=self.accelerator.device, dtype=self.config.base_weight_dtype),
            prepared_batch["timesteps"],
            encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(
                device=self.accelerator.device, dtype=self.config.base_weight_dtype
            ),
            # added_cond_kwargs=added_cond_kwargs,
            controlnet_cond=controlnet_image,
            return_dict=False,
        )

        return {
            "model_prediction": self.model(
                prepared_batch["noisy_latents"].to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                prepared_batch["timesteps"].to(self.accelerator.device),
                encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                # added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=[
                    sample.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
                    for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(
                    device=self.accelerator.device, dtype=self.config.weight_dtype
                ),
                return_dict=False,
            )[0]
        }

    def model_predict(self, prepared_batch):

        return {
            "model_prediction": self.model(
                prepared_batch["noisy_latents"].to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                prepared_batch["timesteps"],
                prepared_batch["encoder_hidden_states"].to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                return_dict=False,
            )[0]
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        if self.config.unet_attention_slice:
            if torch.backends.mps.is_available():
                logger.warning(
                    "Using attention slicing when training {self.NAME} on MPS can result in NaN errors on the first backward pass. If you run into issues, disable this option and reduce your batch size instead to reduce memory consumption."
                )
            if self.model.get_trained_component() is not None:
                self.model.get_trained_component().set_attention_slice("auto")

        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        if self.config.tokenizer_max_length is not None:
            logger.warning(f"-!- {self.NAME} supports a max length of 77 tokens, --tokenizer_max_length is ignored -!-")
        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                "{self.NAME} requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64

        if self.config.prediction_type is not None:
            logger.info(f"Setting {self.NAME} prediction type: {self.config.prediction_type}")
            self.PREDICTION_TYPE = PredictionTypes.from_str(self.config.prediction_type)
            if self.config.validation_noise_scheduler is None:
                self.config.validation_noise_scheduler = self.DEFAULT_NOISE_SCHEDULER

            if self.config.rescale_betas_zero_snr:
                self.config.training_scheduler_timestep_spacing = "trailing"

    def custom_model_card_schedule_info(self):
        output_args = []
        if self.config.snr_gamma:
            output_args.append(f"snr_gamma={self.config.snr_gamma}")
        if self.config.use_soft_min_snr:
            output_args.append(f"use_soft_min_snr")
            if self.config.soft_min_snr_sigma_data:
                output_args.append(f"soft_min_snr_sigma_data={self.config.soft_min_snr_sigma_data}")
        if self.config.rescale_betas_zero_snr:
            output_args.append(f"rescale_betas_zero_snr")
        if self.config.offset_noise:
            output_args.append(f"offset_noise")
            output_args.append(f"noise_offset={self.config.noise_offset}")
            output_args.append(f"noise_offset_probability={self.config.noise_offset_probability}")
        output_args.append(f"training_scheduler_timestep_spacing={self.config.training_scheduler_timestep_spacing}")
        output_args.append(f"inference_scheduler_timestep_spacing={self.config.inference_scheduler_timestep_spacing}")
        output_str = f" (extra parameters={output_args})" if output_args else " (no special parameters set)"

        return output_str


class StableDiffusion2(StableDiffusion1):
    NAME = "Stable Diffusion 2.x"
    PREDICTION_TYPE = PredictionTypes.V_PREDICTION
    DEFAULT_NOISE_SCHEDULER = "euler"

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "pseudoflex-v2"
    HUGGINGFACE_PATHS = {
        "digitaldiffusion": "Junglerally/Digital-Diffusion",
        "pseudoflex-v2": "bghira/pseudo-flex-v2",
        "pseudojourney": "bghira/pseudo-journey-v2",
        "2.1": "stabilityai/stable-diffusion-2-1",
        "2.0": "stabilityai/stable-diffusion-2-0",
    }

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        if self.config.unet_attention_slice:
            if torch.backends.mps.is_available():
                logger.warning(
                    "Using attention slicing when training {self.NAME} on MPS can result in NaN errors on the first backward pass. If you run into issues, disable this option and reduce your batch size instead to reduce memory consumption."
                )
            if self.model.get_trained_component() is not None:
                self.model.get_trained_component().set_attention_slice("auto")

        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        if self.config.tokenizer_max_length is not None:
            logger.warning(f"-!- {self.NAME} supports a max length of 77 tokens, --tokenizer_max_length is ignored -!-")
        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                "{self.NAME} requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64

        if not self.config.rescale_betas_zero_snr:
            logger.warning(
                f"Setting {self.NAME} rescale_betas_zero_snr to True. This is required for training with {self.NAME}."
            )
            self.config.rescale_betas_zero_snr = True
            self.config.training_scheduler_timestep_spacing = "trailing"

        if self.config.prediction_type is not None:
            logger.info(f"Setting {self.NAME} prediction type: {self.config.prediction_type}")
            self.PREDICTION_TYPE = PredictionTypes.from_str(self.config.prediction_type)
            if self.config.validation_noise_scheduler is None:
                self.config.validation_noise_scheduler = self.DEFAULT_NOISE_SCHEDULER


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("sd1x", StableDiffusion1)
ModelRegistry.register("sd2x", StableDiffusion2)
