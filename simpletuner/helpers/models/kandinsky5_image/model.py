# This was MIT-licensed by Kandinsky Lab; now AGPL-3.0-or-later, SimpleTuner (c) bghira
import logging
import os
from typing import Dict

import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from simpletuner.helpers.configuration.registry import ConfigRegistry, ConfigRule, RuleType, make_default_rule
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.kandinsky5_image.pipeline_kandinsky5_t2i import Kandinsky5T2IPipeline
from simpletuner.helpers.models.kandinsky5_image.transformer_kandinsky5 import Kandinsky5Transformer3DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class Kandinsky5Image(ImageModelFoundation):
    NAME = "Kandinsky5-Image"
    MODEL_DESCRIPTION = "Text-to-image diffusion transformer (flow matching)"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    AUTOENCODER_SCALING_FACTOR = 0.3611
    LATENT_CHANNEL_COUNT = 16
    DEFAULT_NOISE_SCHEDULER = "flow_matching"
    MODEL_CLASS = Kandinsky5Transformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: Kandinsky5T2IPipeline,
        PipelineTypes.IMG2IMG: None,  # filled below for I2I
    }
    DEFAULT_LORA_TARGET = ["to_key", "to_query", "to_value"]

    DEFAULT_MODEL_FLAVOUR = "t2i-lite-sft"
    HUGGINGFACE_PATHS: Dict[str, str] = {
        "t2i-lite-sft": "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers",
        "t2i-lite-pretrain": "kandinskylab/Kandinsky-5.0-T2I-Lite-pretrain-Diffusers",
        "i2i-lite-sft": "kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers",
        "i2i-lite-pretrain": "kandinskylab/Kandinsky-5.0-I2I-Lite-pretrain-Diffusers",
    }
    MODEL_LICENSE = "mit"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen2.5-VL",
            "tokenizer": Qwen2VLProcessor,
            "tokenizer_subfolder": "tokenizer",
            "model": Qwen2_5_VLForConditionalGeneration,
            "subfolder": "text_encoder",
        },
        "text_encoder_2": {
            "name": "CLIP-L/14",
            "tokenizer": CLIPTokenizer,
            "tokenizer_subfolder": "tokenizer_2",
            "model": CLIPTextModel,
            "subfolder": "text_encoder_2",
        },
    }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if self.pipelines.get(PipelineTypes.TEXT2IMG) is None:
            self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        pipeline = self.pipelines[PipelineTypes.TEXT2IMG]
        prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens = pipeline.encode_prompt(
            prompt=prompts,
            num_images_per_prompt=1,
            max_sequence_length=int(getattr(self.config, "tokenizer_max_length", 512)),
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        attention_mask = torch.ones(
            prompt_embeds_qwen.shape[:2],
            dtype=torch.int32,
            device=prompt_embeds_qwen.device,
        )
        return prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens, attention_mask

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        qwen_embeds, clip_embeds, cu_seqlens, attention_mask = text_embedding
        return {
            "prompt_embeds": qwen_embeds,
            "pooled_prompt_embeds": clip_embeds,
            "attention_masks": attention_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict, prompt: str) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        pooled_prompt_embeds = text_embedding["pooled_prompt_embeds"]
        attention_mask = text_embedding.get("attention_masks")
        prompt_cu_seqlens = None
        if attention_mask is not None:
            prompt_cu_seqlens = torch.cumsum(attention_mask.sum(1), dim=0)
            prompt_cu_seqlens = torch.cat([torch.zeros_like(prompt_cu_seqlens)[:1], prompt_cu_seqlens]).to(dtype=torch.int32)

        return {
            "prompt_embeds_qwen": prompt_embeds,
            "prompt_embeds_clip": pooled_prompt_embeds,
            "prompt_cu_seqlens": prompt_cu_seqlens,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict, prompt: str) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        pooled_prompt_embeds = text_embedding["pooled_prompt_embeds"]
        attention_mask = text_embedding.get("attention_masks")
        prompt_cu_seqlens = None
        if attention_mask is not None:
            prompt_cu_seqlens = torch.cumsum(attention_mask.sum(1), dim=0)
            prompt_cu_seqlens = torch.cat([torch.zeros_like(prompt_cu_seqlens)[:1], prompt_cu_seqlens]).to(dtype=torch.int32)

        return {
            "negative_prompt_embeds_qwen": prompt_embeds,
            "negative_prompt_embeds_clip": pooled_prompt_embeds,
            "negative_prompt_cu_seqlens": prompt_cu_seqlens,
        }

    def _is_i2i_flavour(self) -> bool:
        flavour = getattr(self.config, "model_flavour", None)
        return flavour is not None and flavour.startswith("i2i")

    def requires_conditioning_dataset(self) -> bool:
        return self._is_i2i_flavour() or super().requires_conditioning_dataset()

    def requires_conditioning_validation_inputs(self) -> bool:
        return self._is_i2i_flavour() or super().requires_conditioning_validation_inputs()

    def requires_validation_edit_captions(self) -> bool:
        return self._is_i2i_flavour() or super().requires_validation_edit_captions()

    def requires_conditioning_latents(self) -> bool:
        visual_cond = False
        if self.model is not None:
            try:
                visual_cond = getattr(self.unwrap_model(self.model).config, "visual_cond", False)
            except Exception:
                visual_cond = False
        if self._is_i2i_flavour() or visual_cond:
            return True
        return super().requires_conditioning_latents()

    def model_predict(self, prepared_batch: dict):
        """
        Forward through the transformer; image case uses single-frame latents.
        """
        latents = prepared_batch["noisy_latents"]  # shape (B, C, H, W)
        if latents.dim() != 4:
            raise ValueError(f"Expected 4D latents for image, got shape {latents.shape}")

        bsz, channels, height, width = latents.shape
        latents = latents.permute(0, 2, 3, 1).unsqueeze(1)  # (B, T=1, H, W, C)

        dtype = self.config.weight_dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        visual_rope_pos = [
            torch.arange(1, device=latents.device),
            torch.arange(height // 2, device=latents.device),
            torch.arange(width // 2, device=latents.device),
        ]
        text_rope_pos = torch.arange(
            prepared_batch["encoder_hidden_states"].shape[1],
            device=latents.device,
        )
        timesteps = prepared_batch["timesteps"].to(device=latents.device, dtype=dtype)

        if "added_cond_kwargs" not in prepared_batch:
            raise ValueError("Kandinsky5Image expects added_cond_kwargs containing pooled text embeddings.")
        added_kwargs = prepared_batch["added_cond_kwargs"]
        if "text_embeds" not in added_kwargs:
            raise ValueError("Kandinsky5Image requires added_cond_kwargs['text_embeds'].")
        pooled = added_kwargs["text_embeds"]

        if getattr(self.unwrap_model(self.model).config, "visual_cond", False):
            visual_cond = torch.zeros_like(latents)
            visual_cond_mask = torch.zeros(
                bsz,
                1,
                height,
                width,
                1,
                device=latents.device,
                dtype=latents.dtype,
            )
            cond_latents = prepared_batch.get("conditioning_latents")
            if cond_latents is not None:
                if cond_latents.dim() == 4:
                    cond_latents = cond_latents.unsqueeze(1)  # B, T=1, C, H, W
                cond_latents = cond_latents.permute(0, 1, 3, 4, 2)  # B, T, H, W, C
                visual_cond[:, : cond_latents.shape[1]] = cond_latents.to(device=latents.device, dtype=latents.dtype)
                visual_cond_mask[:, 0] = 1.0
            latents = torch.cat([latents, visual_cond, visual_cond_mask], dim=-1)

        model_pred = self.model(
            hidden_states=latents.to(dtype),
            encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(dtype),
            pooled_projections=pooled.to(dtype),
            timestep=timesteps,
            visual_rope_pos=visual_rope_pos,
            text_rope_pos=text_rope_pos,
            scale_factor=(1, 2, 2),
            sparse_params=None,
            return_dict=True,
        ).sample

        model_pred = model_pred.squeeze(1).permute(0, 3, 1, 2)  # back to (B, C, H, W)

        return {"model_prediction": model_pred}

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        # Wire IMG2IMG pipeline after class creation to avoid circular import
        from simpletuner.helpers.models.kandinsky5_image.pipeline_kandinsky5_i2i import Kandinsky5I2IPipeline

        self.PIPELINE_CLASSES[PipelineTypes.IMG2IMG] = Kandinsky5I2IPipeline

    def check_user_config(self):
        """
        Apply Kandinsky-specific defaults when user input is missing.
        """
        if getattr(self.config, "tokenizer_max_length", None) is None or int(self.config.tokenizer_max_length) > 256:
            logger.warning("Kandinsky5Image caps tokenizer_max_length at 256 tokens. Overriding provided value.")
            self.config.tokenizer_max_length = 256

    @classmethod
    def register_config_requirements(cls):
        rules = [
            make_default_rule(
                field_name="tokenizer_max_length",
                default_value=256,
                message="Kandinsky5-Image defaults to a 256 token context window.",
            ),
            ConfigRule(
                field_name="tokenizer_max_length",
                rule_type=RuleType.MAX,
                value=256,
                message="Kandinsky5-Image supports a maximum of 256 tokens.",
                error_level="warning",
            ),
        ]
        ConfigRegistry.register_rules("kandinsky5-image", rules)


Kandinsky5Image.register_config_requirements()
ModelRegistry.register("kandinsky5-image", Kandinsky5Image)
