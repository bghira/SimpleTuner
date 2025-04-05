import torch, os, logging
import random
from helpers.models.common import (
    VideoModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)
from diffusers import AutoencoderKLWan
from helpers.models.wan.transformer import WanTransformer3DModel
from helpers.models.wan.pipeline import WanPipeline
from helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if _get_rank() == 0 else "ERROR"
)


class Wan(VideoModelFoundation):
    NAME = "Wan"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLWan
    LATENT_CHANNEL_COUNT = 16
    DEFAULT_NOISE_SCHEDULER = "unipc"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = WanTransformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: WanPipeline,
        # PipelineTypes.IMG2IMG: None,
        # PipelineTypes.CONTROLNET: None,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "t2v-480p-1.3b-2.1"
    HUGGINGFACE_PATHS = {
        "t2v-480p-1.3b-2.1": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "t2v-480p-14b-2.1": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        # "i2v-480p-14b-2.1": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        # "i2v-720p-14b-2.1": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "UMT5",
            "tokenizer": T5TokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": UMT5EncoderModel,
        },
    }

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        """
        When we're running the pipeline, we'll update the kwargs specifically for this model here.
        """
        # Wan video should max out around 81 frames for efficiency.
        pipeline_kwargs["num_frames"] = min(
            81, self.config.validation_num_video_frames or 81
        )
        pipeline_kwargs["output_type"] = "pil"
        # replace embeds with prompt

        return pipeline_kwargs

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        self.config:
            text_embedding (torch.Tensor): The embed to adjust.

        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        prompt_embeds, masks = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": masks,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            # "attention_mask": (
            #     text_embedding["attention_masks"].unsqueeze(0)
            #     if self.config.flux_attention_masked_training
            #     else None
            # ),
        }

    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            # "negative_mask": (
            #     text_embedding["attention_masks"].unsqueeze(0)
            #     if self.config.flux_attention_masked_training
            #     else None
            # ),
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """
        prompt_embeds, masks = (
            self.pipeline.encode_prompt(
                prompt=prompts,
                device=self.accelerator.device,
            )
        )
        if self.config.t5_padding == "zero":
            # we can zero the padding tokens if we're just going to mask them later anyway.
            prompt_embeds = prompt_embeds * masks.to(
                device=prompt_embeds.device
            ).unsqueeze(-1).expand(prompt_embeds.shape)

        return prompt_embeds, masks

    def model_predict(self, prepared_batch):
        model_pred = self.model(
            prepared_batch["noisy_latents"].to(self.config.weight_dtype),
            encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(
                self.config.weight_dtype
            ),
            # encoder_attention_mask=prepared_batch["encoder_attention_mask"],
            timestep=prepared_batch["timesteps"],
            return_dict=False,
        )[0]

        return {
            "model_prediction": model_pred,
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues.
        """
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        # Disable Compel.
        self.config.disable_compel = True
        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                "{self.NAME} requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64

        if self.config.prediction_type is not None:
            logger.warning(
                f"{self.NAME} does not support prediction type {self.config.prediction_type}."
            )

        if self.config.tokenizer_max_length is not None:
            logger.warning(
                f"-!- {self.NAME} supports a max length of 226 tokens, --tokenizer_max_length is ignored -!-"
            )
        self.config.tokenizer_max_length = 226
        if self.config.validation_num_inference_steps > 50:
            logger.warning(
                f"{self.NAME} {self.config.model_flavour} may be wasting compute with more than 50 steps. Consider reducing the value to save time."
            )
        if self.config.validation_num_inference_steps < 40:
            logger.warning(
                f"{self.NAME} {self.config.model_flavour} expects around 40 or more inference steps. Consider increasing --validation_num_inference_steps to 40."
            )
        if not self.config.validation_disable_unconditional:
            logger.info("Disabling unconditional validation to save on time.")
            self.config.validation_disable_unconditional = True

        if self.config.framerate is None:
            self.config.framerate = 15

        self.config.vae_enable_tiling = True
        self.config.vae_enable_slicing = True

    def custom_model_card_schedule_info(self):
        output_args = []
        if self.config.flux_fast_schedule:
            output_args.append("flux_fast_schedule")
        if self.config.flow_schedule_auto_shift:
            output_args.append("flow_schedule_auto_shift")
        if self.config.flow_schedule_shift is not None:
            output_args.append(f"shift={self.config.flow_schedule_shift}")
        output_args.append(f"flux_guidance_mode={self.config.flux_guidance_mode}")
        if self.config.flux_guidance_value:
            output_args.append(f"flux_guidance_value={self.config.flux_guidance_value}")
        if self.config.flux_guidance_min:
            output_args.append(f"flux_guidance_min={self.config.flux_guidance_min}")
        if self.config.flux_guidance_mode == "random-range":
            output_args.append(f"flux_guidance_max={self.config.flux_guidance_max}")
            output_args.append(f"flux_guidance_min={self.config.flux_guidance_min}")
        if self.config.flow_use_beta_schedule:
            output_args.append(
                f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}"
            )
            output_args.append(
                f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}"
            )
        if self.config.flux_attention_masked_training:
            output_args.append("flux_attention_masked_training")
        if self.config.t5_padding != "unmodified":
            output_args.append(f"t5_padding={self.config.t5_padding}")
        if (
            self.config.model_type == "lora"
            and self.config.lora_type == "standard"
            and self.config.flux_lora_target is not None
        ):
            output_args.append(f"flux_lora_target={self.config.flux_lora_target}")
        output_str = (
            f" (extra parameters={output_args})"
            if output_args
            else " (no special parameters set)"
        )

        return output_str
