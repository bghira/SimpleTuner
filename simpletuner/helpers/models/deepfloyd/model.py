import logging
import os

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines import IFPipeline, IFSuperResolutionPipeline
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from peft import set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from transformers import AutoTokenizer, T5EncoderModel

from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class DeepFloydIF(ImageModelFoundation):
    NAME = "DeepFloyd IF"
    MODEL_DESCRIPTION = "Pixel-space diffusion model with T5 text encoder"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.EPSILON
    MODEL_TYPE = ModelTypes.UNET
    # DeepFloyd-IF is a pixel space model.
    AUTOENCODER_CLASS = None
    LATENT_CHANNEL_COUNT = None
    DEFAULT_NOISE_SCHEDULER = "ddpm"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    SLIDER_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default seems to be the most stable..
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = UNet2DConditionModel
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: IFPipeline,
        PipelineTypes.IMG2IMG: IFSuperResolutionPipeline,
    }
    MODEL_SUBFOLDER = "unet"
    REQUIRES_FLAVOUR = True
    HUGGINGFACE_PATHS = {
        # stage one, text-to-image models
        "i-medium-400m": "DeepFloyd/IF-I-M-v1.0",
        "i-large-900m": "DeepFloyd/IF-I-L-v1.0",
        "i-xlarge-4.3b": "DeepFloyd/IF-I-XL-v1.0",
        # stage two, super-resolution models
        "ii-medium-450m": "DeepFloyd/IF-II-M-v1.0",
        "ii-large-1.2b": "DeepFloyd/IF-II-L-v1.0",
    }
    MODEL_LICENSE = "deepfloyd-if-license"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "T5 XXL v1.1",
            "tokenizer": AutoTokenizer,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": T5EncoderModel,
        },
    }

    def validation_image_input_edge_length(self):
        # If a model requires a specific input edge length (HiDream E1 -> 768px, DeepFloyd stage2 -> 64px)
        if self.config.model_flavour.startswith("ii-"):
            return 64
        return None

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        self.config:
            text_embedding (torch.Tensor): The embed to adjust.

        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        return {"prompt_embeds": text_embedding}

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]

        return {
            "prompt_embeds": prompt_embeds,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]

        return {
            "negative_prompt_embeds": prompt_embeds,
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt for a DeepFloyd model.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """

        positive_embed, negative_embed = self.pipelines[PipelineTypes.TEXT2IMG].encode_prompt(
            prompt=prompts,
            do_classifier_free_guidance=False,
            device=self.accelerator.device,
        )

        return positive_embed

    def model_predict(self, prepared_batch):
        logger.debug(
            "Input shapes:"
            f"\n{prepared_batch['noisy_latents'].shape}"
            f"\n{prepared_batch['timesteps'].shape}"
            f"\n{prepared_batch['encoder_hidden_states'].shape}"
        )
        prediction_kwargs = {
            "timestep": prepared_batch["timesteps"],
            "encoder_hidden_states": prepared_batch["encoder_hidden_states"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
        }
        if self.config.model_flavour.startswith("ii-"):
            # expand noisy latents by doubling dim
            logger.info(f"Pre-expansion shape: {prepared_batch['noisy_latents'].shape}")
            prepared_batch["noisy_latents"] = torch.cat(
                [prepared_batch["noisy_latents"], prepared_batch["noisy_latents"]],
                dim=1,
            )
            logger.info(f"Post--expansion shape: {prepared_batch['noisy_latents'].shape}")
            prediction_kwargs["class_labels"] = prepared_batch["timesteps"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            )
        model_pred = self.model(
            prepared_batch["noisy_latents"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            return_dict=False,
            **prediction_kwargs,
        )[0]
        # chunk prediction and discard learnt variance
        model_pred = model_pred.chunk(2, dim=1)[0]
        return {
            "model_prediction": model_pred,
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                "DeepFloyd does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        t5_max_length = 77
        if self.config.tokenizer_max_length is None or int(self.config.tokenizer_max_length) > t5_max_length:
            if not self.config.i_know_what_i_am_doing:
                logger.warning(f"Updating T5 XXL tokeniser max length to {t5_max_length} for DeepFloyd.")
                self.config.tokenizer_max_length = t5_max_length
            else:
                logger.warning(
                    f"-!- DeepFloyd supports a max length of {t5_max_length} tokens, but you have supplied `--i_know_what_i_am_doing`, so this limit will not be enforced. -!-"
                )
                logger.warning(
                    f"The model will begin to collapse after a short period of time, if the model you are continuing from has not been tuned beyond {t5_max_length} tokens."
                )
        # Disable custom VAEs for DeepFloyd.
        self.config.pretrained_vae_model_name_or_path = None
        self.config.vae_path = None

        if self.config.aspect_bucket_alignment != 32:
            logger.warning(
                "DeepFloyd requires an alignment value of 32px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 32

        # Stage II needs different default pipeline.
        if self.config.model_flavour.startswith("ii-"):
            self.DEFAULT_PIPELINE_TYPE = PipelineTypes.IMG2IMG

    def custom_model_card_schedule_info(self):
        output_args = []
        # TODO: Implement scheduler info for DeepFloyd.
        output_str = f" (extra parameters={output_args})" if output_args else " (no special parameters set)"

        return output_str


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("deepfloyd", DeepFloydIF)
