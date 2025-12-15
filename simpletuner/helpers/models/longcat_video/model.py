import logging
import os
from typing import Dict, Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLWan
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from simpletuner.helpers.models.common import (
    ModelTypes,
    PipelineTypes,
    PredictionTypes,
    TextEmbedCacheKey,
    VideoModelFoundation,
)
from simpletuner.helpers.models.longcat_video.pipeline import LongCatVideoPipeline
from simpletuner.helpers.models.longcat_video.transformer import LongCatVideoTransformer3DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.musubi_block_swap import apply_musubi_pretrained_defaults
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class LongCatVideo(VideoModelFoundation):
    NAME = "LongCat-Video"
    MODEL_DESCRIPTION = "Flow-matching text-to-video and image-to-video transformer"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLWan
    LATENT_CHANNEL_COUNT = 16
    DEFAULT_NOISE_SCHEDULER = "flow_matching"
    DEFAULT_PIPELINE_TYPE = PipelineTypes.TEXT2IMG

    MODEL_CLASS = LongCatVideoTransformer3DModel
    MODEL_SUBFOLDER = "dit"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: LongCatVideoPipeline,
        PipelineTypes.IMG2IMG: LongCatVideoPipeline,
        PipelineTypes.IMG2VIDEO: LongCatVideoPipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "final"
    HUGGINGFACE_PATHS: Dict[str, str] = {
        "final": "meituan-longcat/LongCat-Video",
    }
    MODEL_LICENSE = "mit"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen-2.5 VL",
            "tokenizer": AutoTokenizer,
            "tokenizer_subfolder": None,
            "model": Qwen2_5_VLForConditionalGeneration,
            "subfolder": None,
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
        },
    }

    DEFAULT_LORA_TARGET = ["qkv", "proj", "q_linear", "kv_linear"]
    VALIDATION_USES_NEGATIVE_PROMPT = True

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        if getattr(self.config, "aspect_bucket_alignment", None) != 64:
            logger.warning(
                "LongCat-Video requires aspect_bucket_alignment=64. Overriding from %s.",
                getattr(self.config, "aspect_bucket_alignment", None),
            )
            self.config.aspect_bucket_alignment = 64
        if getattr(self.config, "flow_schedule_shift", None) is None:
            self.config.flow_schedule_shift = 12.0

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        return apply_musubi_pretrained_defaults(self.config, args)

    @classmethod
    def get_flavour_choices(cls):
        return tuple(cls.HUGGINGFACE_PATHS.keys())

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            pipeline.encode_prompt(
                prompt=prompts,
                device=self.accelerator.device,
                do_classifier_free_guidance=False,
            )
        )

        if is_negative_prompt:
            return prompt_embeds, prompt_attention_mask, None, None

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def _is_i2v_like_flavour(self) -> bool:
        """
        LongCat-Video uses a single flavour; I2V is triggered by datasets marked is_i2v
        (conditioning frames present). Expose this so collate/conditioners can route
        conditioning latents correctly.
        """
        return True

    def requires_conditioning_dataset(self) -> bool:
        # Conditioning is supported when provided, but not mandatory.
        return False

    def requires_conditioning_validation_inputs(self) -> bool:
        # Validation can run without conditioning inputs; use them if available.
        return False

    def supports_conditioning_dataset(self) -> bool:
        """
        Indicates conditioning datasets are supported (optional) so the UI can surface
        conditioning controls without forcing them.
        """
        return True

    def default_validation_pipeline_type(self):
        """
        LongCat-Video uses a single pipeline for both T2V and I2V. When validation
        is configured to use datasets (img2img mode), still route to the same pipeline.
        """
        return PipelineTypes.TEXT2IMG

    def _normalize_prompt_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.dim() == 4 and tensor.shape[1] == 1:
            return tensor.squeeze(1)
        return tensor

    def _normalize_attention_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.dim() == 4 and tensor.shape[1] == 1 and tensor.shape[2] == 1:
            tensor = tensor.squeeze(1).squeeze(1)
        if tensor.dim() == 3:
            if tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            if tensor.dim() == 3 and tensor.shape[2] == 1:
                tensor = tensor.squeeze(2)
        return tensor

    def _format_text_embedding(self, text_embedding: dict):
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = text_embedding
        prompt_embeds = self._normalize_prompt_tensor(prompt_embeds)
        prompt_attention_mask = self._normalize_attention_tensor(prompt_attention_mask)
        negative_prompt_embeds = self._normalize_prompt_tensor(negative_prompt_embeds)
        negative_prompt_attention_mask = self._normalize_attention_tensor(negative_prompt_attention_mask)
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict, pipeline_type=None) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding["prompt_attention_mask"]

        if prompt_embeds is not None:
            if prompt_embeds.dim() == 4 and prompt_embeds.shape[1] == 1:
                # already [B, 1, S, C]
                pass
            elif prompt_embeds.dim() == 3:
                # [B, S, C] -> [B, 1, S, C]
                prompt_embeds = prompt_embeds.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected prompt_embeds shape for LongCat: {prompt_embeds.shape}")

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attention_mask = attention_mask.squeeze(1).squeeze(1)
            elif attention_mask.dim() == 3 and attention_mask.shape[1] == 1:
                attention_mask = attention_mask.squeeze(1)
            elif attention_mask.dim() != 2:
                raise ValueError(f"Unexpected attention_mask shape for LongCat: {attention_mask.shape}")

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": attention_mask,
            "negative_prompt_embeds": text_embedding.get("negative_prompt_embeds"),
            "negative_prompt_attention_mask": text_embedding.get("negative_prompt_attention_mask"),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        negative_embeds = self._normalize_prompt_tensor(
            text_embedding.get("negative_prompt_embeds") or text_embedding.get("prompt_embeds")
        )
        negative_attention_mask = self._normalize_attention_tensor(
            text_embedding.get("negative_prompt_attention_mask") or text_embedding.get("prompt_attention_mask")
        )
        if negative_embeds is None:
            return {}
        if negative_embeds.dim() == 3:
            negative_embeds = negative_embeds.unsqueeze(1)
        if negative_attention_mask is not None:
            negative_attention_mask = self._normalize_attention_tensor(negative_attention_mask)
        return {
            "negative_prompt_embeds": negative_embeds,
            "negative_prompt_attention_mask": negative_attention_mask,
        }

    def pack_text_embeddings_for_cache(self, embeddings):
        """
        Strip extra singleton prompt/attention dimensions before writing to cache so collate sees 3D/2D tensors.
        """
        if isinstance(embeddings, tuple):
            embeddings = self._format_text_embedding(embeddings)

        prompt_embeds = self._normalize_prompt_tensor(embeddings.get("prompt_embeds"))
        prompt_attention_mask = self._normalize_attention_tensor(embeddings.get("prompt_attention_mask"))
        negative_prompt_embeds = self._normalize_prompt_tensor(embeddings.get("negative_prompt_embeds"))
        negative_prompt_attention_mask = self._normalize_attention_tensor(embeddings.get("negative_prompt_attention_mask"))

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
        }

    def unpack_text_embeddings_from_cache(self, embeddings):
        """
        Ensure cached embeddings are normalized before consumption.
        """
        if isinstance(embeddings, tuple):
            embeddings = self._format_text_embedding(embeddings)
        embeddings = dict(embeddings)
        embeddings["prompt_embeds"] = self._normalize_prompt_tensor(embeddings.get("prompt_embeds"))
        embeddings["prompt_attention_mask"] = self._normalize_attention_tensor(embeddings.get("prompt_attention_mask"))
        embeddings["negative_prompt_embeds"] = self._normalize_prompt_tensor(embeddings.get("negative_prompt_embeds"))
        embeddings["negative_prompt_attention_mask"] = self._normalize_attention_tensor(
            embeddings.get("negative_prompt_attention_mask")
        )
        return embeddings

    def collate_prompt_embeds(self, text_encoder_output: list[dict]) -> dict:
        return {}

    def text_embed_cache_key(self) -> TextEmbedCacheKey:
        return TextEmbedCacheKey.CAPTION

    def setup_training_noise_schedule(self):
        self.noise_schedule = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=self.config.flow_schedule_shift,
        )
        return self.config, self.noise_schedule

    def model_predict(self, prepared_batch):
        noisy_latents = prepared_batch["noisy_latents"].to(self.accelerator.device, dtype=self.config.weight_dtype)
        encoder_hidden_states = prepared_batch["encoder_hidden_states"].to(
            self.accelerator.device, dtype=self.config.weight_dtype
        )
        encoder_attention_mask = prepared_batch.get("encoder_attention_mask")
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device=self.accelerator.device)
        timesteps = prepared_batch["timesteps"]

        cond_count = int(prepared_batch.get("conditioning_latent_count", 0) or 0)

        model_pred = self.model(
            noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timesteps,
            timestep_sign=prepared_batch.get("twinflow_time_sign"),
            num_cond_latents=cond_count,
            return_dict=False,
        )[0]

        if cond_count > 0 and model_pred.dim() == 5 and model_pred.shape[2] > cond_count:
            model_pred = model_pred[:, :, cond_count:, :, :]

        return {
            "model_prediction": model_pred,
        }


ModelRegistry.register("longcat_video", LongCatVideo)
