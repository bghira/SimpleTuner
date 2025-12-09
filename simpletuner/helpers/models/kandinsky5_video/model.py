# This was MIT-licensed by Kandinsky Lab; now AGPL-3.0-or-later, SimpleTuner (c) bghira
import logging
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLHunyuanVideo
from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from simpletuner.helpers.configuration.registry import ConfigRegistry, ConfigRule, RuleType, make_default_rule
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


def _patch_diffusers_hunyuanvideo_conv(memory_limit: int = 512 * 1024**2, force_temporal_roll: bool = False):
    """
    Monkeypatch diffusers' HunyuanVideo VAE Conv3d to split along time for lower peak VRAM.
    When force_temporal_roll is True, chunking is always applied to keep temporal memory footprints small.
    """
    from diffusers.models.autoencoders import autoencoder_kl_hunyuan_video as hv_mod

    already_patched = getattr(hv_mod, "_st_patch_conv_applied", False)
    already_rolled = getattr(hv_mod, "_st_patch_conv_roll", False)
    if already_patched and (force_temporal_roll is False or already_rolled):
        return hv_mod.HunyuanVideoCausalConv3d

    def find_split_indices(seq_len: int, part_num: int, stride: int):
        ideal_interval = seq_len / part_num
        possible_indices = list(range(0, seq_len, stride))
        selected_indices = []

        for i in range(1, part_num):
            closest = min(possible_indices, key=lambda x: abs(x - round(i * ideal_interval)))
            if closest not in selected_indices:
                selected_indices.append(closest)

        merged_indices = []
        prev_idx = 0
        for idx in selected_indices:
            if idx - prev_idx >= stride:
                merged_indices.append(idx)
                prev_idx = idx

        return merged_indices

    def patched_forward(self, hidden_states: torch.Tensor):
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        T = hidden_states.shape[2]
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        memory_count = torch.prod(torch.tensor(hidden_states.shape)).item() * 2 / memory_limit
        part_num = int(memory_count / 2) + 1

        use_chunking = force_temporal_roll or (T > kernel_size and memory_count > 0.6 and part_num >= 2)
        if use_chunking and T > 1:
            if force_temporal_roll:
                part_num = max(2, part_num, T // max(kernel_size, 1))

            max_parts = max(1, T // kernel_size)
            if part_num > max_parts:
                part_num = max_parts
            split_indices = find_split_indices(T, part_num, stride)
            if len(split_indices) == 0 or kernel_size == 1:
                input_chunks = torch.tensor_split(hidden_states, split_indices, dim=2) if split_indices else [hidden_states]
            else:
                boundaries = [0] + split_indices + [T]
                input_chunks = []
                for i in range(len(boundaries) - 1):
                    start = boundaries[i]
                    end = boundaries[i + 1]
                    overlap_start = max(start - kernel_size + 1, 0)
                    if i == 0:
                        input_chunks.append(hidden_states[:, :, start:end])
                    else:
                        input_chunks.append(hidden_states[:, :, overlap_start:end])
            output_chunks = [self.conv(chunk) for chunk in input_chunks]
            return torch.cat(output_chunks, dim=2)

        return self.conv(hidden_states)

    hv_mod.HunyuanVideoCausalConv3d.forward = patched_forward
    hv_mod._st_patch_conv_applied = True
    hv_mod._st_patch_conv_roll = already_rolled or force_temporal_roll
    return hv_mod.HunyuanVideoCausalConv3d


class Kandinsky5Video(VideoModelFoundation):
    SUPPORTS_MUON_CLIP = True
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
        # I2V first-frame variants
        "i2v-lite-5s": "kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers",
        "i2v-pro-sft-5s": "kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers",
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
        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)

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

    def load_vae(self, move_to_device: bool = True):
        enable_patch_conv = getattr(self.config, "vae_enable_patch_conv", False)
        enable_temporal_roll = getattr(self.config, "vae_enable_temporal_roll", False)
        if enable_patch_conv or enable_temporal_roll:
            logger.info(
                "Enabling HunyuanVideo VAE patch-based convolution%s for Kandinsky5 Video.",
                " with temporal rolling" if enable_temporal_roll else "",
            )
            _patch_diffusers_hunyuanvideo_conv(force_temporal_roll=enable_temporal_roll)
        return super().load_vae(move_to_device)

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        pooled_prompt_embeds = text_embedding["pooled_prompt_embeds"]
        attention_mask = text_embedding.get("attention_masks")

        prompt_cu_seqlens = None
        if attention_mask is not None:
            # Build cu_seqlens expected by the transformer from the attention mask.
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

            # Pooled CLIP output should stay 2D; never pad or reshape it.
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

            if val.dim() == 2 and val.shape[-1] < pad_to:
                pad_len = pad_to - val.shape[-1]
                pad_shape = val.shape[:-1] + (pad_len,)
                unpacked[key] = torch.cat([val, val.new_zeros(pad_shape)], dim=-1)
                continue

            if val.dim() >= 3 and val.shape[-2] < pad_to:
                pad_len = pad_to - val.shape[-2]
                pad_token = pad_slices.get(key, val[..., -1:, :])
                pad_repeat = pad_token.expand(*pad_token.shape[:-2], pad_len, pad_token.shape[-1])
                unpacked[key] = torch.cat([val, pad_repeat], dim=-2)
            elif val.dim() >= 1 and val.shape[-1] < pad_to:
                pad_len = pad_to - val.shape[-1]
                pad_shape = val.shape[:-1] + (pad_len,)
                unpacked[key] = torch.cat([val, val.new_zeros(pad_shape)], dim=-1)
        return unpacked

    def _is_i2v_flavour(self) -> bool:
        flavour = getattr(self.config, "model_flavour", None)
        return flavour is not None and str(flavour).lower().startswith("i2v")

    def requires_conditioning_dataset(self) -> bool:
        return self._is_i2v_flavour() or super().requires_conditioning_dataset()

    def requires_conditioning_validation_inputs(self) -> bool:
        return self._is_i2v_flavour() or super().requires_conditioning_validation_inputs()

    def requires_validation_i2v_samples(self) -> bool:
        return self._is_i2v_flavour() or super().requires_validation_i2v_samples()

    def requires_conditioning_latents(self) -> bool:
        visual_cond = False
        if self.model is not None:
            try:
                visual_cond = getattr(self.unwrap_model(self.model).config, "visual_cond", False)
            except Exception:
                visual_cond = False
        if self._is_i2v_flavour() or visual_cond:
            return True
        return super().requires_conditioning_latents()

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

        is_i2v_batch = bool(prepared_batch.get("is_i2v_data", False)) or self._is_i2v_flavour()
        visual_cond_enabled = getattr(self.unwrap_model(self.model).config, "visual_cond", False)

        # Append conditioning zeros + mask for visual_cond models.
        if visual_cond_enabled:
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
            if is_i2v_batch and cond_latents is None:
                raise ValueError(
                    "Kandinsky5 I2V training requires conditioning_latents. "
                    "Ensure the conditioning dataset is configured and VAE latents are cached."
                )
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
        elif is_i2v_batch:
            raise ValueError("I2V training batch detected but transformer.visual_cond is disabled for this checkpoint.")

        # RoPE positions
        visual_rope_pos = [
            torch.arange(frames, device=latents.device),
            torch.arange(height // 2, device=latents.device),
            torch.arange(width // 2, device=latents.device),
        ]

        dtype = self.config.weight_dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        encoder_hidden_states = prepared_batch["encoder_hidden_states"]
        if encoder_hidden_states.dim() == 4 and encoder_hidden_states.shape[1] == 1:
            # Cache stores prompt embeds with an extra singleton prompt dimension; drop it.
            encoder_hidden_states = encoder_hidden_states.squeeze(1)
        elif encoder_hidden_states.dim() != 3:
            raise ValueError(f"Expected 3D encoder_hidden_states, got shape {encoder_hidden_states.shape}")

        text_rope_pos = torch.arange(
            encoder_hidden_states.shape[1],
            device=latents.device,
        )

        timesteps = prepared_batch["timesteps"].to(device=latents.device, dtype=torch.float32)

        if "added_cond_kwargs" not in prepared_batch:
            raise ValueError("Kandinsky5Video expects added_cond_kwargs containing pooled text embeddings.")
        added_kwargs = prepared_batch["added_cond_kwargs"]
        if "text_embeds" not in added_kwargs:
            raise ValueError("Kandinsky5Video requires added_cond_kwargs['text_embeds'].")
        pooled = added_kwargs["text_embeds"]
        if pooled.dim() == 3 and pooled.shape[1] == 1:
            pooled = pooled.squeeze(1)
        elif pooled.dim() != 2:
            raise ValueError(f"Expected 2D pooled text embeddings, got shape {pooled.shape}")

        force_keep_mask = None
        raw_force_keep = prepared_batch.get("force_keep_mask")
        if raw_force_keep is not None and getattr(self.config, "tread_config", None):
            force_keep_mask = self._prepare_force_keep_mask(latents, raw_force_keep)

        model_pred = self.model(
            hidden_states=latents.to(dtype),
            encoder_hidden_states=encoder_hidden_states.to(dtype),
            pooled_projections=pooled.to(dtype),
            timestep=timesteps,
            visual_rope_pos=visual_rope_pos,
            text_rope_pos=text_rope_pos,
            scale_factor=(1, 2, 2),
            sparse_params=None,
            return_dict=True,
            force_keep_mask=force_keep_mask,
        ).sample

        # Restore to (B, C, T, H, W)
        model_pred = model_pred.permute(0, 4, 1, 2, 3)

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

    def check_user_config(self):
        """
        Apply Kandinsky-specific defaults when user input is missing.
        """
        if getattr(self.config, "framerate", None) is None:
            self.config.framerate = 24
        if getattr(self.config, "tokenizer_max_length", None) is None or int(self.config.tokenizer_max_length) > 256:
            logger.warning("Kandinsky5Video caps tokenizer_max_length at 256 tokens. Overriding provided value.")
            self.config.tokenizer_max_length = 256

    @classmethod
    def register_config_requirements(cls):
        rules = [
            make_default_rule(
                field_name="framerate",
                default_value=24,
                message="Kandinsky5-Video defaults to 24 fps.",
            ),
            make_default_rule(
                field_name="tokenizer_max_length",
                default_value=256,
                message="Kandinsky5-Video defaults to a 256 token context window.",
            ),
            ConfigRule(
                field_name="tokenizer_max_length",
                rule_type=RuleType.MAX,
                value=256,
                message="Kandinsky5-Video supports a maximum of 256 tokens.",
                error_level="warning",
            ),
        ]
        ConfigRegistry.register_rules("kandinsky5-video", rules)

    def tread_init(self):
        """
        Initialize the TREAD model training method for Kandinsky5 video checkpoints.
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

        logger.info("TREAD training is enabled for Kandinsky5-Video")


Kandinsky5Video.register_config_requirements()
ModelRegistry.register("kandinsky5-video", Kandinsky5Video)
