import logging
import os

import torch
from diffusers import AutoencoderDC
from diffusers.pipelines import SanaPipeline
from transformers import Gemma2Model, GemmaTokenizerFast

from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.sana.pipeline import SanaImg2ImgPipeline
from simpletuner.helpers.models.sana.transformer import SanaTransformer2DModel

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class Sana(ImageModelFoundation):
    NAME = "Sana"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderDC
    LATENT_CHANNEL_COUNT = 32
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = SanaTransformer2DModel
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: SanaPipeline,
        PipelineTypes.IMG2IMG: SanaImg2ImgPipeline,
    }
    MODEL_SUBFOLDER = "transformer"
    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "sana1.5-4.8b-1024"
    HUGGINGFACE_PATHS = {
        "sana1.5-4.8b-1024": "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        "sana1.5-1.6b-1024": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        "sana1.0-1.6b-2048": "Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers",
        "sana1.0-1.6b-1024": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        "sana1.0-600m-1024": "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        "sana1.0-600m-512": "Efficient-Large-Model/Sana_600M_512px_diffusers",
    }
    MODEL_LICENSE = "other"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Gemma2 2B-IT",
            "tokenizer": GemmaTokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": Gemma2Model,
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
        prompt_embeds, attention_mask = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "attention_mask": attention_mask.squeeze(0),
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['attention_mask'].shape}")
        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "prompt_attention_mask": text_embedding["attention_mask"].unsqueeze(0),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor, prompt: str) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['attention_mask'].shape}")
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "negative_prompt_attention_mask": text_embedding["attention_mask"].unsqueeze(0),
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt using the pipeline.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """
        prompt_embeds, prompt_attention_mask, _, _ = self.pipelines[PipelineTypes.TEXT2IMG].encode_prompt(
            prompt=prompts,
            do_classifier_free_guidance=False,
            device=self.accelerator.device,
            clean_caption=False,
            max_sequence_length=300,
            complex_human_instruction=(None if is_negative_prompt else self.config.sana_complex_human_instruction),
        )

        return prompt_embeds, prompt_attention_mask

    def model_predict(self, prepared_batch):
        return {
            "model_prediction": self.model(
                hidden_states=prepared_batch["noisy_latents"].to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                timestep=prepared_batch["timesteps"],
                encoder_attention_mask=prepared_batch["encoder_attention_mask"],
                encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(
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
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError("Sana does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead.")
        if self.config.tokenizer_max_length is not None:
            logger.warning("Tokenizer max length is ignored for Sana. It is fixed to 300 tokens.")
        # Disable custom VAEs for Sana.
        self.config.pretrained_vae_model_name_or_path = None
        if self.config.aspect_bucket_alignment != 64:
            logger.warning("MM-DiT requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment.")
            self.config.aspect_bucket_alignment = 64
        if "sageattention" in self.config.attention_mechanism:
            if self.config.model_family == "sana":
                logger.error(
                    f"{self.config.model_family} is not supported with SageAttention at this point. Disabling SageAttention."
                )
                self.config.attention_mechanism = "diffusers"

    def custom_model_card_schedule_info(self):
        output_args = []
        if self.config.flow_schedule_auto_shift:
            output_args.append("flow_schedule_auto_shift")
        if self.config.flow_schedule_shift is not None:
            output_args.append(f"shift={self.config.flow_schedule_shift}")
        if self.config.flow_use_beta_schedule:
            output_args.append(f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}")
            output_args.append(f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}")
        if self.config.flow_use_uniform_schedule:
            output_args.append(f"flow_use_uniform_schedule")
        output_str = f" (extra parameters={output_args})" if output_args else " (no special parameters set)"
        return logger.info(f"SANA loaded flow matching logit-normal distribution scheduler{output_str}")

    def tread_init(self):
        """
        Initialize the TREAD model training method.
        """
        from simpletuner.helpers.training.tread import TREADRouter

        if (
            getattr(self.config, "tread_config", None) is None
            or getattr(self.config, "tread_config", None) is {}
            or getattr(self.config, "tread_config", {}).get("routes", None) is None
        ):
            logger.error("TREAD training requires you to configure the routes in the TREAD config")
            import sys

            sys.exit(1)

        self.unwrap_model(model=self.model).set_router(
            TREADRouter(
                seed=getattr(self.config, "seed", None) or 42,
                device=self.accelerator.device,
            ),
            self.config.tread_config["routes"],
        )
