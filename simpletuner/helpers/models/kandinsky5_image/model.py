# This was MIT-licensed by Kandinsky Lab; now AGPL-3.0-or-later, SimpleTuner (c) bghira
import logging
import os
from typing import Dict, Optional

import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from simpletuner.helpers.acceleration import AccelerationBackend, AccelerationPreset, get_sdnq_presets
from simpletuner.helpers.configuration.registry import ConfigRegistry, ConfigRule, RuleType, make_default_rule
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.kandinsky5_image.pipeline_kandinsky5_t2i import Kandinsky5T2IPipeline
from simpletuner.helpers.models.kandinsky5_image.transformer_kandinsky5 import Kandinsky5Transformer3DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.models.tae.types import ImageTAESpec
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class Kandinsky5Image(ImageModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "Kandinsky5-Image"
    MODEL_DESCRIPTION = "Text-to-image diffusion transformer (flow matching)"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    AUTOENCODER_SCALING_FACTOR = 0.3611
    LATENT_CHANNEL_COUNT = 16
    VALIDATION_PREVIEW_SPEC = ImageTAESpec(repo_id="madebyollin/taef1")
    DEFAULT_NOISE_SCHEDULER = "flow_matching"
    MODEL_CLASS = Kandinsky5Transformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: Kandinsky5T2IPipeline,
        PipelineTypes.IMG2IMG: None,  # filled below for I2I
    }
    DEFAULT_LORA_TARGET = ["to_key", "to_query", "to_value"]
    SLIDER_LORA_TARGET = [
        "attn1.to_query",
        "attn1.to_key",
        "attn1.to_value",
        "conv_in",
        "conv_out",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
    ]

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

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        # Kandinsky5 Image has 2 text + 32 visual = 34 transformer blocks
        # Leave at least 1 block on GPU
        return 33

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        # Common settings for memory optimization presets
        _base_memory_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }

        return [
            # RamTorch presets - 3 levels (Light, Balanced, Aggressive)
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="light",
                name="RamTorch - Light",
                description="Streams 8 of 34 transformer blocks (~24%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~22%",
                tradeoff_speed="Increases training time by ~15%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "visual_transformer_blocks.0,visual_transformer_blocks.1,visual_transformer_blocks.2,visual_transformer_blocks.3,visual_transformer_blocks.4,visual_transformer_blocks.5,visual_transformer_blocks.6,visual_transformer_blocks.7",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="balanced",
                name="RamTorch - Balanced",
                description="Streams 17 of 34 transformer blocks (~50%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~45%",
                tradeoff_speed="Increases training time by ~35%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "visual_transformer_blocks.0,visual_transformer_blocks.1,visual_transformer_blocks.2,visual_transformer_blocks.3,visual_transformer_blocks.4,visual_transformer_blocks.5,visual_transformer_blocks.6,visual_transformer_blocks.7,visual_transformer_blocks.8,visual_transformer_blocks.9,visual_transformer_blocks.10,visual_transformer_blocks.11,visual_transformer_blocks.12,visual_transformer_blocks.13,visual_transformer_blocks.14,visual_transformer_blocks.15,visual_transformer_blocks.16",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="aggressive",
                name="RamTorch - Aggressive",
                description="Streams all transformer blocks (34 of 34).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~85%",
                tradeoff_speed="Increases training time by ~75%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "text_transformer_blocks.*,visual_transformer_blocks.*",
                },
            ),
            # Block Swap presets - 3 levels (Light, Balanced, Aggressive)
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="light",
                name="Block Swap - Light",
                description="Swaps 8 of 34 blocks (~24%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~22%",
                tradeoff_speed="Increases training time by ~15%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 8},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="balanced",
                name="Block Swap - Balanced",
                description="Swaps 17 of 34 blocks (~50%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~45%",
                tradeoff_speed="Increases training time by ~35%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 17},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="aggressive",
                name="Block Swap - Aggressive",
                description="Swaps 28 of 34 blocks (~82%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~75%",
                tradeoff_speed="Increases training time by ~65%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 28},
            ),
            # DeepSpeed presets (Advanced tab)
            AccelerationPreset(
                backend=AccelerationBackend.DEEPSPEED_ZERO_1,
                level="zero1",
                name="DeepSpeed ZeRO Stage 1",
                description="Shards optimizer states across GPUs.",
                tab="advanced",
                tradeoff_vram="Reduces optimizer memory by 75% per GPU",
                tradeoff_speed="Minimal overhead",
                tradeoff_notes="Requires multi-GPU setup.",
                config={**_base_memory_config, "deepspeed": "zero1"},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.DEEPSPEED_ZERO_2,
                level="zero2",
                name="DeepSpeed ZeRO Stage 2",
                description="Shards optimizer states and gradients across GPUs.",
                tab="advanced",
                tradeoff_vram="Reduces optimizer + gradient memory by 85% per GPU",
                tradeoff_speed="Moderate communication overhead",
                tradeoff_notes="Requires multi-GPU setup.",
                config={**_base_memory_config, "deepspeed": "zero2"},
            ),
            # SDNQ presets (works on AMD, Apple, NVIDIA)
            *get_sdnq_presets(_base_memory_config),
        ]

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        pipeline = self.pipelines.get(PipelineTypes.TEXT2IMG)
        if pipeline is None:
            pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
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

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
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

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
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

    def tread_init(self):
        """
        Initialize the TREAD model training method for Kandinsky5 image checkpoints.
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

        logger.info("TREAD training is enabled for Kandinsky5-Image")

    def _find_attention_mask(self, embeddings: dict):
        for key in ("attention_masks", "attention_mask", "prompt_attention_mask"):
            mask = embeddings.get(key)
            if mask is not None and torch.is_tensor(mask):
                return key, mask
        return None, None

    def pack_text_embeddings_for_cache(self, embeddings):
        """
        Trim trailing pad tokens (keeping one pad token) to shrink cache storage.
        """
        if not isinstance(embeddings, dict):
            return embeddings
        _, mask = self._find_attention_mask(embeddings)
        if mask is None or mask.dim() < 2:
            return embeddings

        pad_to = getattr(self.config, "tokenizer_max_length", None)
        if not pad_to or mask.shape[-1] != pad_to:
            return embeddings
        token_lens = mask.sum(dim=-1)
        max_tokens = int(token_lens.max().item())
        if max_tokens <= 0 or max_tokens >= pad_to:
            return embeddings

        trimmed_len = min(pad_to, max_tokens + 1)
        packed = dict(embeddings)
        pad_slices: Dict[str, torch.Tensor] = {}
        for key, val in embeddings.items():
            if not torch.is_tensor(val):
                continue
            if val.dim() >= 2 and val.shape[-2] == pad_to:
                pad_slices[key] = val[..., trimmed_len - 1 : trimmed_len, :]
                packed[key] = val[..., :trimmed_len, :]
            elif val.dim() >= 1 and val.shape[-1] == pad_to:
                packed[key] = val[..., :trimmed_len]
        if pad_slices:
            packed["_pad_slices"] = pad_slices
        return packed

    def unpack_text_embeddings_from_cache(self, embeddings):
        """
        Restore cached embeds to their original padded length using stored pad token.
        Protect pooled_prompt_embeds from accidental padding/reshaping.
        """
        if not isinstance(embeddings, dict):
            return embeddings
        pad_to = getattr(self.config, "tokenizer_max_length", None)
        if not pad_to:
            return embeddings

        pad_slices = embeddings.get("_pad_slices", {})
        unpacked = {k: v for k, v in embeddings.items() if k != "_pad_slices"}
        for key, val in list(unpacked.items()):
            if not torch.is_tensor(val):
                continue

            if key in ("attention_masks", "attention_mask", "prompt_attention_mask"):
                if val.dim() >= 1 and val.shape[-1] < pad_to:
                    pad_len = pad_to - val.shape[-1]
                    pad_shape = val.shape[:-1] + (pad_len,)
                    unpacked[key] = torch.cat([val, val.new_zeros(pad_shape)], dim=-1)
                continue

            if key == "pooled_prompt_embeds":
                if val.dim() == 3:
                    logger.warning(
                        "Cached pooled_prompt_embeds has an unexpected sequence dimension; using first token. shape=%s",
                        val.shape,
                    )
                    unpacked[key] = val[:, 0, :]
                elif val.dim() != 2:
                    raise ValueError(f"Unexpected pooled_prompt_embeds shape {val.shape}")
                continue

            if val.dim() >= 2 and val.shape[-2] < pad_to:
                pad_len = pad_to - val.shape[-2]
                pad_token = pad_slices.get(key, val[..., -1:, :])
                pad_repeat = pad_token.expand(*pad_token.shape[:-2], pad_len, pad_token.shape[-1])
                unpacked[key] = torch.cat([val, pad_repeat], dim=-2)
            elif val.dim() >= 1 and val.shape[-1] < pad_to:
                pad_len = pad_to - val.shape[-1]
                pad_shape = val.shape[:-1] + (pad_len,)
                unpacked[key] = torch.cat([val, val.new_zeros(pad_shape)], dim=-1)
        return unpacked

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
        timesteps = prepared_batch["timesteps"].to(device=latents.device, dtype=torch.float32)

        if "added_cond_kwargs" not in prepared_batch:
            raise ValueError("Kandinsky5Image expects added_cond_kwargs containing pooled text embeddings.")
        added_kwargs = prepared_batch["added_cond_kwargs"]
        if "text_embeds" not in added_kwargs:
            raise ValueError("Kandinsky5Image requires added_cond_kwargs['text_embeds'].")
        pooled = added_kwargs["text_embeds"]
        if pooled.dim() == 3 and pooled.shape[1] == 1:
            pooled = pooled.squeeze(1)
        elif pooled.dim() == 3 and pooled.shape[1] > 1:
            logger.warning(
                "Kandinsky5Image expected pooled text embeddings with a singleton prompt dimension; got shape %s. "
                "Using the first token to continue.",
                pooled.shape,
            )
            pooled = pooled[:, 0, :]
        elif pooled.dim() != 2:
            raise ValueError(f"Expected 2D pooled text embeddings, got shape {pooled.shape}")

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

        force_keep_mask = None
        raw_force_keep = prepared_batch.get("force_keep_mask")
        if raw_force_keep is not None and getattr(self.config, "tread_config", None):
            force_keep_mask = self._prepare_force_keep_mask(latents, raw_force_keep)

        model_pred = self.model(
            hidden_states=latents.to(dtype),
            encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(dtype),
            pooled_projections=pooled.to(dtype),
            timestep=timesteps,
            timestep_sign=(
                prepared_batch.get("twinflow_time_sign") if getattr(self.config, "twinflow_enabled", False) else None
            ),
            visual_rope_pos=visual_rope_pos,
            text_rope_pos=text_rope_pos,
            scale_factor=(1, 2, 2),
            sparse_params=None,
            return_dict=True,
            force_keep_mask=force_keep_mask,
        ).sample

        model_pred = model_pred.squeeze(1).permute(0, 3, 1, 2)  # back to (B, C, H, W)

        return {"model_prediction": model_pred}

    def _prepare_force_keep_mask(self, latents: torch.Tensor, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Validate and reshape the optional force_keep_mask to match token count for TREAD routing.
        """
        if mask is None:
            return None

        patch_size = getattr(self.unwrap_model(self.model).config, "patch_size", (1, 2, 2))
        if len(patch_size) != 3:
            raise ValueError(f"Unexpected patch_size format: {patch_size}")

        tokens_expected = (
            (latents.shape[1] // patch_size[0]) * (latents.shape[2] // patch_size[1]) * (latents.shape[3] // patch_size[2])
        )

        if mask.dim() > 2:
            mask = mask.view(mask.shape[0], -1)

        if mask.numel() == mask.shape[0] * tokens_expected and mask.shape[1] != tokens_expected:
            mask = mask.view(mask.shape[0], tokens_expected)

        if mask.shape[1] != tokens_expected:
            raise ValueError(
                f"force_keep_mask length {mask.shape[1]} does not match expected token count {tokens_expected} for patch_size {patch_size}."
            )

        return mask.to(device=latents.device, dtype=torch.bool)

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
