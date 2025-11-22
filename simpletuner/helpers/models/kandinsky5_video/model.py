# This was MIT-licensed by Kandinsky Lab; now AGPL-3.0-or-later, SimpleTuner (c) bghira
import logging
import os
from typing import Dict, Optional

import torch
from diffusers import AutoencoderKLHunyuanVideo
from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from simpletuner.helpers.models.common import ModelTypes, PipelineTypes, PredictionTypes, VideoModelFoundation
from simpletuner.helpers.models.kandinsky5_video.pipeline_kandinsky5_i2v import Kandinsky5I2VPipeline
from simpletuner.helpers.models.kandinsky5_video.pipeline_kandinsky5_t2v import Kandinsky5T2VPipeline
from simpletuner.helpers.models.kandinsky5_video.transformer_kandinsky5 import Kandinsky5Transformer3DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class Kandinsky5Video(VideoModelFoundation):
    """
    Kandinsky 5.0 Video (Lite/Pro) text-to-video transformer with HunyuanVideo VAE and
    Qwen2.5-VL + CLIP dual text encoders.
    """

    NAME = "Kandinsky5-Video"
    MODEL_DESCRIPTION = "Text-to-video diffusion transformer (flow matching)"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLHunyuanVideo
    LATENT_CHANNEL_COUNT = 16
    DEFAULT_NOISE_SCHEDULER = "flow_matching"
    MODEL_CLASS = Kandinsky5Transformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: Kandinsky5T2VPipeline,
        PipelineTypes.IMG2VIDEO: Kandinsky5I2VPipeline,
    }

    # Default model flavor to use when none specified.
    DEFAULT_MODEL_FLAVOUR = "t2v-lite-sft-5s"
    HUGGINGFACE_PATHS: Dict[str, str] = {
        # Lite 5s/10s (keep SFT + pretrain; skip distilled/nocfg variants)
        "t2v-lite-sft-5s": "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers",
        "t2v-lite-pretrain-5s": "kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-5s-Diffusers",
        "t2v-lite-sft-10s": "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-10s-Diffusers",
        "t2v-lite-pretrain-10s": "kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-10s-Diffusers",
        # Pro (HD/SD) SFT + pretrain only
        "t2v-pro-sft-5s-hd": "kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers",
        "t2v-pro-sft-10s-hd": "kandinskylab/Kandinsky-5.0-T2V-Pro-sft-10s-Diffusers",
        "t2v-pro-pretrain-5s-hd": "kandinskylab/Kandinsky-5.0-T2V-Pro-pretrain-5s-Diffusers",
        "t2v-pro-pretrain-10s-hd": "kandinskylab/Kandinsky-5.0-T2V-Pro-pretrain-10s-Diffusers",
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
        """
        Encode prompts using the pipeline's dual encoders (Qwen2.5-VL + CLIP).
        """
        if self.pipelines.get(PipelineTypes.TEXT2IMG) is None:
            self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        pipeline = self.pipelines[PipelineTypes.TEXT2IMG]

        prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens = pipeline.encode_prompt(
            prompt=prompts,
            num_videos_per_prompt=1,
            max_sequence_length=int(getattr(self.config, "tokenizer_max_length", 512)),
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        # Build a simple attention mask to reconstruct cu_seqlens during training.
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

    def model_predict(self, prepared_batch: dict):
        """
        Forward pass through the transformer with proper rope positions and visual conditioning.
        """
        latents = prepared_batch["noisy_latents"]  # expected shape (B, C, T, H, W)
        if latents.dim() != 5:
            raise ValueError(f"Expected 5D latents for video, got shape {latents.shape}")

        bsz, channels, frames, height, width = latents.shape
        # Rearrange to (B, T, H, W, C)
        latents = latents.permute(0, 2, 3, 4, 1)

        # Append conditioning zeros + mask for visual_cond models.
        if getattr(self.unwrap_model(self.model).config, "visual_cond", False):
            visual_cond = torch.zeros_like(latents)
            visual_cond_mask = torch.zeros(
                bsz,
                frames,
                height,
                width,
                1,
                device=latents.device,
                dtype=latents.dtype,
            )

            cond_latents = prepared_batch.get("conditioning_latents")
            if cond_latents is not None:
                if cond_latents.dim() == 4:
                    # (B, C, H, W) -> add frame dim
                    cond_latents = cond_latents.unsqueeze(2)
                if cond_latents.shape[2] != 1:
                    # only first frame conditioning supported; squeeze extra frames
                    cond_latents = cond_latents[:, :, :1]
                # Move to HWC
                cond_latents = cond_latents.permute(0, 2, 3, 4, 1)  # B, T, H, W, C
                # Broadcast to match current latent size (latent space already scaled in cache)
                visual_cond[:, : cond_latents.shape[1]] = cond_latents.to(device=latents.device, dtype=latents.dtype)
                visual_cond_mask[:, 0] = 1.0

            latents = torch.cat([latents, visual_cond, visual_cond_mask], dim=-1)

        # RoPE positions
        visual_rope_pos = [
            torch.arange(frames, device=latents.device),
            torch.arange(height // 2, device=latents.device),
            torch.arange(width // 2, device=latents.device),
        ]
        text_rope_pos = torch.arange(
            prepared_batch["encoder_hidden_states"].shape[1],
            device=latents.device,
        )

        dtype = self.config.weight_dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        timesteps = prepared_batch["timesteps"].to(device=latents.device, dtype=dtype)
        pooled = prepared_batch.get("added_cond_kwargs", {}).get("text_embeds") or prepared_batch.get("add_text_embeds")
        if pooled is None:
            raise ValueError("Kandinsky5Video expects pooled text embeddings in added_cond_kwargs['text_embeds'].")
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

        # Restore to (B, C, T, H, W)
        model_pred = model_pred.permute(0, 4, 1, 2, 3)

        return {"model_prediction": model_pred}


ModelRegistry.register("kandinsky5-video", Kandinsky5Video)
