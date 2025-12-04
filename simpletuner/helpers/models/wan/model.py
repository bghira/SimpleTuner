import logging
import os
import random
import threading
from functools import partial
from typing import Dict, Optional

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPVisionModel, T5TokenizerFast, UMT5EncoderModel

from simpletuner.helpers.models.common import (
    ModelTypes,
    PipelineConditioningImageEmbedder,
    PipelineTypes,
    PredictionTypes,
    VideoModelFoundation,
    VideoToTensor,
)
from simpletuner.helpers.models.tae.types import VideoTAESpec
from simpletuner.helpers.models.wan.pipeline import WanPipeline
from simpletuner.helpers.models.wan.transformer import WanTransformer3DModel

logger = logging.getLogger(__name__)
from torch.nn import functional as F

from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.tread import TREADRouter
from simpletuner.helpers.training.wrappers import unwrap_model as accelerator_unwrap_model

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


def time_text_monkeypatch(
    self,
    timestep: torch.Tensor,
    encoder_hidden_states,
    encoder_hidden_states_image=None,
    timestep_seq_len=None,
):
    timestep = self.timesteps_proj(timestep)
    if timestep_seq_len is not None:
        timestep = timestep.unflatten(0, (encoder_hidden_states.shape[0], timestep_seq_len))

    time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
    if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
        timestep = timestep.to(time_embedder_dtype)
    temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
    timestep_proj = self.time_proj(self.act_fn(temb))

    encoder_hidden_states = self.text_embedder(encoder_hidden_states)
    if encoder_hidden_states_image is not None:
        encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

    return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


@torch.no_grad()
def add_first_frame_conditioning(
    latent_model_input: torch.Tensor,
    first_frame: torch.Tensor,
    vae: AutoencoderKLWan,
):
    """
    Adds first-frame conditioning for Wan 2.1-style I2V models by concatenating
    the encoded conditioning latents and mask alongside the noisy latents.
    """
    device = latent_model_input.device
    dtype = latent_model_input.dtype
    vae_scale_factor_temporal = 2 ** sum(getattr(vae, "temperal_downsample", []))

    _, _, num_latent_frames, latent_height, latent_width = latent_model_input.shape
    num_frames = (num_latent_frames - 1) * 4 + 1

    if first_frame.ndim == 3:
        first_frame = first_frame.unsqueeze(0)
    if first_frame.shape[0] != latent_model_input.shape[0]:
        first_frame = first_frame.expand(latent_model_input.shape[0], -1, -1, -1)

    vae_scale_factor = vae.config.scale_factor_spatial
    first_frame = torch.nn.functional.interpolate(
        first_frame,
        size=(
            latent_model_input.shape[3] * vae_scale_factor,
            latent_model_input.shape[4] * vae_scale_factor,
        ),
        mode="bilinear",
        align_corners=False,
    )
    first_frame = first_frame.unsqueeze(2)

    zero_frame = torch.zeros_like(first_frame)
    video_condition = torch.cat(
        [first_frame, *[zero_frame for _ in range(num_frames - 1)]],
        dim=2,
    )

    latent_condition = vae.encode(video_condition.to(device=device, dtype=dtype)).latent_dist.sample()
    latent_condition = latent_condition.to(device=device, dtype=dtype)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device=device, dtype=dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        device=device, dtype=dtype
    )
    latent_condition = (latent_condition - latents_mean) * latents_std

    mask_lat_size = torch.ones(
        latent_model_input.shape[0],
        1,
        num_frames,
        latent_height,
        latent_width,
        device=device,
        dtype=dtype,
    )
    mask_lat_size[:, :, list(range(1, num_frames))] = 0
    first_frame_mask = mask_lat_size[:, :, 0:1]
    first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=vae_scale_factor_temporal)
    mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
    mask_lat_size = mask_lat_size.view(
        latent_model_input.shape[0],
        -1,
        vae_scale_factor_temporal,
        latent_height,
        latent_width,
    )
    mask_lat_size = mask_lat_size.transpose(1, 2)

    first_frame_condition = torch.concat([mask_lat_size, latent_condition], dim=1)
    conditioned_latent = torch.cat([latent_model_input, first_frame_condition], dim=1)

    return conditioned_latent


@torch.no_grad()
def add_first_frame_conditioning_v22(
    latent_model_input: torch.Tensor,
    first_frame: torch.Tensor,
    vae: AutoencoderKLWan,
    last_frame: Optional[torch.Tensor] = None,
):
    """
    Adds first (and optional last) frame conditioning for Wan 2.2-style models that
    overwrite latent time steps rather than concatenating additional channels.
    """
    device = latent_model_input.device
    dtype = latent_model_input.dtype
    bs, _, T, H, W = latent_model_input.shape
    scale = vae.config.scale_factor_spatial
    target_h = H * scale
    target_w = W * scale

    if first_frame.ndim == 3:
        first_frame = first_frame.unsqueeze(0)
    if first_frame.shape[0] != bs:
        first_frame = first_frame.expand(bs, -1, -1, -1)

    first_frame_up = torch.nn.functional.interpolate(
        first_frame,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).unsqueeze(2)
    encoded = vae.encode(first_frame_up.to(device=device, dtype=dtype)).latent_dist.sample().to(dtype)

    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    encoded = (encoded - mean) * std

    latent = latent_model_input.clone()
    latent[:, :, : encoded.shape[2]] = encoded

    mask = torch.ones(bs, 1, T, H, W, device=device, dtype=dtype)
    mask[:, :, : encoded.shape[2]] = 0.0

    if last_frame is not None:
        if last_frame.ndim == 3:
            last_frame = last_frame.unsqueeze(0)
        if last_frame.shape[0] != bs:
            last_frame = last_frame.expand(bs, -1, -1, -1)
        last_frame_up = torch.nn.functional.interpolate(
            last_frame,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(2)
        last_encoded = vae.encode(last_frame_up.to(device=device, dtype=dtype)).latent_dist.sample().to(dtype)
        last_encoded = (last_encoded - mean) * std
        latent[:, :, -last_encoded.shape[2] :] = last_encoded
        mask[:, :, -last_encoded.shape[2] :] = 0.0

    return latent, mask


class Wan(VideoModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "Wan"
    MODEL_DESCRIPTION = "Video generation model (text-to-video)"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLWan
    LATENT_CHANNEL_COUNT = 16
    _TAE_SPEC_21 = VideoTAESpec(filename="taew2_1.pth", description="Wan 2.1 / 2.2 14B VAE")
    _TAE_SPEC_22 = VideoTAESpec(filename="taew2_2.pth", description="Wan 2.2 5B VAE", patch_size=2, latent_channels=48)
    DEFAULT_NOISE_SCHEDULER = "unipc"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    SLIDER_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = WanTransformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: WanPipeline,
        PipelineTypes.IMG2VIDEO: WanImageToVideoPipeline,
        # PipelineTypes.IMG2IMG: None,
        # PipelineTypes.CONTROLNET: None,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "t2v-480p-1.3b-2.1"
    HUGGINGFACE_PATHS = {
        "t2v-480p-1.3b-2.1": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "t2v-480p-14b-2.1": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "i2v-14b-2.1": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "i2v-14b-2.1-720p": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        "i2v-14b-2.2-high": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "i2v-14b-2.2-low": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "flf2v-14b-2.1": "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
        "vace-1.3b-2.1": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        "vace-14b-2.1": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "ti2v-5b-2.2": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        # "i2v-480p-14b-2.1": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        # "i2v-720p-14b-2.1": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    }
    MODEL_LICENSE = "apache-2.0"

    WAN_STAGE_OVERRIDES: Dict[str, Dict[str, object]] = {
        "i2v-14b-2.2-high": {
            "trained_stage": "high",
            "stage_subfolder": "transformer_2",
            "other_stage_subfolder": "transformer",
            "flow_shift": 5.0,
            "sample_steps": 40,
            "boundary_ratio": 0.90,
            "guidance": {"high": 3.5, "low": 3.5},
        },
        "i2v-14b-2.2-low": {
            "trained_stage": "low",
            "stage_subfolder": "transformer",
            "other_stage_subfolder": "transformer_2",
            "flow_shift": 5.0,
            "sample_steps": 40,
            "boundary_ratio": 0.90,
            "guidance": {"high": 3.5, "low": 3.5},
        },
        "flf2v-14b-2.2-high": {
            "trained_stage": "high",
            "stage_subfolder": "transformer_2",
            "other_stage_subfolder": "transformer",
            "flow_shift": 5.0,
            "sample_steps": 40,
            "boundary_ratio": 0.90,
            "guidance": {"high": 3.5, "low": 3.5},
        },
        "flf2v-14b-2.2-low": {
            "trained_stage": "low",
            "stage_subfolder": "transformer",
            "other_stage_subfolder": "transformer_2",
            "flow_shift": 5.0,
            "sample_steps": 40,
            "boundary_ratio": 0.90,
            "guidance": {"high": 3.5, "low": 3.5},
        },
    }

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "UMT5",
            "tokenizer": T5TokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": UMT5EncoderModel,
        },
    }

    I2V_FLAVOURS = frozenset(
        {
            "i2v-14b-2.1",
            "i2v-14b-2.1-720p",
            "i2v-14b-2.2-high",
            "i2v-14b-2.2-low",
        }
    )
    I2V_CLIP_CONDITIONED_FLAVOURS = frozenset(
        {
            "i2v-14b-2.1",
            "i2v-14b-2.1-720p",
        }
    )
    FLF2V_FLAVOURS = frozenset(
        {
            "flf2v-14b-2.1",
            "flf2v-14b-2.2-high",
            "flf2v-14b-2.2-low",
        }
    )
    TI2V_FLAVOURS = frozenset(
        {
            "ti2v-5b-2.2",
        }
    )
    EXPAND_TIMESTEP_FLAVOURS = frozenset(
        {
            "ti2v-5b-2.2",
        }
    )
    STRICT_I2V_FLAVOURS = tuple(sorted((I2V_FLAVOURS | FLF2V_FLAVOURS)))

    def get_validation_preview_spec(self):
        flavour = getattr(self.config, "model_flavour", self.DEFAULT_MODEL_FLAVOUR) or ""
        if "5b" in str(flavour).lower():
            return self._TAE_SPEC_22
        return self._TAE_SPEC_21

    @classmethod
    def supports_chunked_feed_forward(cls) -> bool:
        return True

    def enable_chunked_feed_forward(self, *, chunk_size: Optional[int] = None, chunk_dim: Optional[int] = None) -> None:
        transformer = self.unwrap_model(self.model)
        if transformer is None or not hasattr(transformer, "set_chunk_feed_forward"):
            raise RuntimeError("Wan transformer is not available for feed-forward chunking.")

        if chunk_size is None:
            transformer.set_chunk_feed_forward(None, chunk_dim)
            logger.info("Wan feed-forward chunking enabled (auto mode).")
            return

        chunk_value = int(chunk_size)
        if chunk_value <= 0:
            transformer.set_chunk_feed_forward(None, chunk_dim)
            logger.info("Wan feed-forward chunking enabled (auto mode).")
            return

        normalized_dim = chunk_dim if chunk_dim is not None else 0
        transformer.set_chunk_feed_forward(chunk_value, normalized_dim)
        logger.info("Wan feed-forward chunking enabled (chunk_size=%s, chunk_dim=%s).", chunk_value, normalized_dim)

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self._wan_cached_stage_modules: Dict[str, WanTransformer3DModel] = {}
        self._conditioning_image_embedder = None
        self._wan_logged_missing_img_encoder = False
        self._wan_vae_patch_lock = threading.Lock()
        self._wan_warned_missing_i2v_conditioning = False
        if not hasattr(self.config, "wan_force_2_1_time_embedding"):
            self.config.wan_force_2_1_time_embedding = False
        self._wan_expand_timesteps = False

    def requires_conditioning_image_embeds(self) -> bool:
        if not self._is_i2v_like_flavour():
            return False

        if not self._wan_transformers_require_image_conditioning():
            return False

        pipeline = self.pipelines.get(PipelineTypes.IMG2VIDEO)
        if pipeline is not None:
            if getattr(pipeline, "image_encoder", None) is None:
                if not self._wan_logged_missing_img_encoder:
                    logger.info(
                        "Wan flavour %s IMG2VIDEO pipeline missing image encoder; loading conditioning components separately.",
                        getattr(self.config, "model_flavour", "<unknown>"),
                    )
                    self._wan_logged_missing_img_encoder = True
            elif self._wan_logged_missing_img_encoder:
                self._wan_logged_missing_img_encoder = False

        return True

    def _current_flavour(self) -> str:
        flavour = getattr(self.config, "model_flavour", None)
        return str(flavour or "")

    def _flavour_in(self, collection) -> bool:
        return self._current_flavour() in collection

    def requires_conditioning_validation_inputs(self) -> bool:
        return self._flavour_in(self.I2V_FLAVOURS | self.FLF2V_FLAVOURS | self.TI2V_FLAVOURS)

    def requires_validation_i2v_samples(self) -> bool:
        return self._flavour_in(self.I2V_FLAVOURS)

    def prepare_batch_conditions(self, batch: dict, state: dict) -> dict:
        original_pixels = batch.get("conditioning_pixel_values")
        if isinstance(original_pixels, list) and len(original_pixels) > 0:
            batch["_wan_conditioning_pixel_values_list"] = original_pixels
        else:
            batch["_wan_conditioning_pixel_values_list"] = None

        batch = super().prepare_batch_conditions(batch, state)

        pixel_list = batch.pop("_wan_conditioning_pixel_values_list", None)
        if pixel_list:
            batch["conditioning_pixel_values_multi"] = [
                tensor.to(device=self.accelerator.device) if hasattr(tensor, "to") else tensor for tensor in pixel_list
            ]
        else:
            batch["conditioning_pixel_values_multi"] = None
        return batch

    def _is_i2v_like_flavour(self) -> bool:
        return self._flavour_in(self.I2V_FLAVOURS | self.FLF2V_FLAVOURS | self.TI2V_FLAVOURS)

    def _uses_last_frame_conditioning(self) -> bool:
        return self._flavour_in(self.FLF2V_FLAVOURS)

    def _module_requires_image_conditioning(self, module: Optional[torch.nn.Module]) -> bool:
        if module is None:
            return False
        config = getattr(module, "config", None)
        if config is None:
            try:
                unwrapped = accelerator_unwrap_model(self.accelerator, module)
            except Exception:  # pragma: no cover - defensive guard
                unwrapped = module
            config = getattr(unwrapped, "config", None)
        if config is None:
            return False
        image_dim = getattr(config, "image_dim", None)
        return image_dim is not None and image_dim != 0

    def _wan_transformers_require_image_conditioning(self) -> bool:
        if getattr(self.config, "wan_disable_conditioning_image_embeds", False):
            return False

        if getattr(self.config, "wan_force_conditioning_image_embeds", False):
            return True

        model = getattr(self, "model", None)
        if model is not None and self._module_requires_image_conditioning(model):
            return True

        for cached in self._wan_cached_stage_modules.values():
            if self._module_requires_image_conditioning(cached):
                return True

        pipeline = self.pipelines.get(PipelineTypes.IMG2VIDEO)
        if pipeline is not None:
            if self._module_requires_image_conditioning(getattr(pipeline, "transformer", None)):
                return True
            if self._module_requires_image_conditioning(getattr(pipeline, "transformer_2", None)):
                return True

        flavour = self._current_flavour()
        return flavour in self.I2V_CLIP_CONDITIONED_FLAVOURS

    def _extract_conditioning_frames(self, prepared_batch):
        multi = prepared_batch.get("conditioning_pixel_values_multi")
        first_frame = None
        last_frame = None
        if multi:
            first_frame = multi[0]
            if self._uses_last_frame_conditioning() and len(multi) > 1:
                last_frame = multi[-1]
        else:
            candidate = prepared_batch.get("conditioning_pixel_values")
            if torch.is_tensor(candidate):
                first_frame = candidate
        return first_frame, last_frame

    def _mask_to_force_keep(self, mask: torch.Tensor) -> Optional[torch.Tensor]:
        transformer = self.unwrap_model(self.model) if getattr(self, "model", None) is not None else None
        if transformer is None or not hasattr(transformer, "config"):
            return None
        patch_size = getattr(transformer.config, "patch_size", (1, 2, 2))
        t_step = max(int(patch_size[0]), 1)
        h_step = max(int(patch_size[1]), 1)
        w_step = max(int(patch_size[2]), 1)
        mask_tokens = mask[:, :, ::t_step, ::h_step, ::w_step]
        mask_tokens = mask_tokens.squeeze(1)
        force_keep = mask_tokens < 0.5
        return force_keep.flatten(1)

    def _build_expand_timesteps(self, base_timesteps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        transformer = self.unwrap_model(self.model) if getattr(self, "model", None) is not None else None
        if transformer is None or not hasattr(transformer, "config"):
            return base_timesteps
        patch_size = getattr(transformer.config, "patch_size", (1, 2, 2))
        t_step = max(int(patch_size[0]), 1)
        h_step = max(int(patch_size[1]), 1)
        w_step = max(int(patch_size[2]), 1)
        mask_tokens = mask[:, :, ::t_step, ::h_step, ::w_step]
        mask_tokens = mask_tokens.squeeze(1)
        base = base_timesteps.to(mask_tokens.device, dtype=mask_tokens.dtype).view(-1, 1, 1, 1)
        expanded = (mask_tokens * base).flatten(1)
        return expanded.to(device=base_timesteps.device, dtype=base_timesteps.dtype)

    def _wan_prepare_vae_encode_inputs(self, vae, samples: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if not torch.is_tensor(samples) or samples.ndim != 5:
            return samples, False

        vae_config = getattr(vae, "config", None)
        if vae_config is None:
            return samples, False

        patch_size = getattr(vae_config, "patch_size", None)
        if not isinstance(patch_size, int) or patch_size <= 1:
            return samples, False

        in_channels = getattr(vae_config, "in_channels", None)
        if not isinstance(in_channels, int):
            return samples, False

        batch, channels, frames, height, width = samples.shape
        if channels == in_channels:
            return samples, False

        expected_channels = channels * (patch_size**2)
        if expected_channels != in_channels:
            return samples, False

        if height % patch_size != 0 or width % patch_size != 0:
            logger.warning(
                "Unable to patchify VAE inputs: shape (%s, %s) not divisible by patch size %s.",
                height,
                width,
                patch_size,
            )
            return samples, False

        reshaped = samples.contiguous().view(
            batch,
            channels,
            frames,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        patched = reshaped.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
        patched = patched.view(batch, expected_channels, frames, height // patch_size, width // patch_size)
        return patched, True

    def _wan_encode_without_internal_patchify(self, vae, samples: torch.Tensor, original_patch_size):
        config = getattr(vae, "config", None)
        if config is None or original_patch_size is None:
            return vae.encode(samples)
        try:
            config.patch_size = None
            return vae.encode(samples)
        finally:
            config.patch_size = original_patch_size

    def encode_with_vae(self, vae, samples):
        patched_samples, disable_internal_patch = self._wan_prepare_vae_encode_inputs(vae, samples)
        if disable_internal_patch:
            original_patch_size = getattr(getattr(vae, "config", None), "patch_size", None)
            lock = getattr(self, "_wan_vae_patch_lock", None)
            if lock is not None:
                with lock:
                    return self._wan_encode_without_internal_patchify(vae, patched_samples, original_patch_size)
            return self._wan_encode_without_internal_patchify(vae, patched_samples, original_patch_size)
        return super().encode_with_vae(vae, patched_samples)

    def _apply_i2v_conditioning_to_kwargs(self, prepared_batch, transformer_kwargs):
        is_i2v_batch = bool(prepared_batch.get("is_i2v_data", False))
        if not (self._is_i2v_like_flavour() or is_i2v_batch):
            return
        first_frame, last_frame = self._extract_conditioning_frames(prepared_batch)
        if first_frame is None:
            if is_i2v_batch and not getattr(self, "_wan_warned_missing_i2v_conditioning", False) and should_log():
                logger.warning(
                    "Wan I2V conditioning data was requested but no conditioning frames were provided. "
                    "Ensure your dataset supplies conditioning images when training I2V flavours."
                )
                self._wan_warned_missing_i2v_conditioning = True
            return
        elif getattr(self, "_wan_warned_missing_i2v_conditioning", False):
            self._wan_warned_missing_i2v_conditioning = False

        latent_tensor = transformer_kwargs.get("hidden_states")
        if latent_tensor is None:
            return

        latent_device = latent_tensor.device
        latent_dtype = latent_tensor.dtype

        vae = self.get_vae()
        if vae is None:
            return
        try:
            vae_param = next(vae.parameters())
            if vae_param.device != latent_device:
                vae.to(latent_device)
        except StopIteration:
            pass

        def _prepare_frame(frame: torch.Tensor) -> torch.Tensor:
            if frame.device != latent_device or frame.dtype != vae.dtype:
                return frame.to(device=latent_device, dtype=vae.dtype)
            return frame

        first_frame_prepared = _prepare_frame(first_frame).detach()
        last_frame_prepared = None
        if self._uses_last_frame_conditioning() and last_frame is not None:
            last_frame_prepared = _prepare_frame(last_frame).detach()

        expand_timesteps = bool(self._wan_expand_timesteps)
        with torch.no_grad():
            if expand_timesteps:
                conditioned_latent, mask = add_first_frame_conditioning_v22(
                    latent_tensor,
                    first_frame_prepared,
                    vae,
                    last_frame=last_frame_prepared,
                )
                transformer_kwargs["hidden_states"] = conditioned_latent.to(dtype=latent_dtype)
                base_timesteps = prepared_batch["timesteps"]
                expanded_timesteps = self._build_expand_timesteps(base_timesteps, mask)
                transformer_kwargs["timestep"] = expanded_timesteps
                force_keep = self._mask_to_force_keep(mask)
                if force_keep is not None:
                    existing = transformer_kwargs.get("force_keep_mask")
                    transformer_kwargs["force_keep_mask"] = force_keep if existing is None else (existing | force_keep)
            else:
                conditioned_latent = add_first_frame_conditioning(
                    latent_tensor,
                    first_frame_prepared,
                    vae,
                )
                transformer_kwargs["hidden_states"] = conditioned_latent.to(dtype=latent_dtype)

    def _get_conditioning_image_embedder(self):
        pipeline = self.pipelines.get(PipelineTypes.IMG2VIDEO)
        if pipeline is None:
            try:
                pipeline = self.get_pipeline(PipelineTypes.IMG2VIDEO)
            except Exception:
                pipeline = None

        if pipeline is None:
            return None

        image_encoder = getattr(pipeline, "image_encoder", None)
        image_processor = getattr(pipeline, "image_processor", None)
        if image_encoder is None or image_processor is None:
            return None

        device = getattr(self.accelerator, "device", torch.device("cpu"))
        weight_dtype = getattr(self.config, "weight_dtype", None)
        return PipelineConditioningImageEmbedder(
            pipeline=pipeline,
            image_encoder=image_encoder,
            image_processor=image_processor,
            device=device,
            weight_dtype=weight_dtype,
        )

    def setup_model_flavour(self):
        super().setup_model_flavour()
        flavour = getattr(self.config, "model_flavour", None) or ""
        self._wan_expand_timesteps = flavour in self.EXPAND_TIMESTEP_FLAVOURS
        setattr(self.config, "wan_expand_timesteps", self._wan_expand_timesteps)
        stage_info = self._wan_stage_info()
        if stage_info is None:
            return

        if getattr(self.config, "pretrained_transformer_model_name_or_path", None) is None:
            self.config.pretrained_transformer_model_name_or_path = self.config.pretrained_model_name_or_path
        self.config.pretrained_transformer_subfolder = stage_info["stage_subfolder"]

        self.config.wan_trained_stage = stage_info["trained_stage"]
        self.config.wan_stage_main_subfolder = stage_info["stage_subfolder"]
        self.config.wan_stage_other_subfolder = stage_info["other_stage_subfolder"]
        self.config.wan_boundary_ratio = stage_info["boundary_ratio"]

        self.config.flow_schedule_shift = stage_info["flow_shift"]
        self.config.validation_num_inference_steps = stage_info["sample_steps"]
        self.config.validation_guidance = stage_info["guidance"][stage_info["trained_stage"]]

        if not hasattr(self.config, "wan_validation_load_other_stage"):
            self.config.wan_validation_load_other_stage = False

    def _wan_stage_info(self) -> Optional[Dict[str, object]]:
        flavour = getattr(self.config, "model_flavour", None)
        return self.WAN_STAGE_OVERRIDES.get(flavour)

    def _apply_time_embedding_override(self, transformer: Optional[WanTransformer3DModel]) -> None:
        if transformer is None:
            return
        target = self.unwrap_model(transformer)
        setter = getattr(target, "set_time_embedding_v2_1", None)
        if callable(setter):
            setter(bool(getattr(self.config, "wan_force_2_1_time_embedding", False)))

    def _patch_condition_embedder(self, transformer: Optional[WanTransformer3DModel]) -> None:
        if transformer is None:
            return
        target = self.unwrap_model(transformer)
        embedder = getattr(target, "condition_embedder", None)
        if embedder is None:
            return
        if getattr(embedder, "_simpletuner_time_text_patch", False):
            return
        embedder.forward = partial(time_text_monkeypatch, embedder)
        embedder._simpletuner_time_text_patch = True

    def post_model_load_setup(self):
        super().post_model_load_setup()
        self._apply_time_embedding_override(getattr(self, "model", None))
        self._patch_condition_embedder(getattr(self, "model", None))

    def _should_load_other_stage(self) -> bool:
        stage_info = self._wan_stage_info()
        if stage_info is None:
            return False
        return bool(getattr(self.config, "wan_validation_load_other_stage", False))

    def _get_or_load_wan_stage_module(self, subfolder: str) -> WanTransformer3DModel:
        if subfolder in self._wan_cached_stage_modules:
            return self._wan_cached_stage_modules[subfolder]

        logger.info("Loading Wan stage weights for validation from subfolder '%s'.", subfolder)
        stage = self.MODEL_CLASS.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=subfolder,
            torch_dtype=self.config.weight_dtype,
            use_safetensors=True,
        )
        stage.requires_grad_(False)
        stage.to(self.accelerator.device, dtype=self.config.weight_dtype)
        stage.eval()
        self._apply_time_embedding_override(stage)
        self._patch_condition_embedder(stage)
        self._wan_cached_stage_modules[subfolder] = stage
        return stage

    def unload_model(self):
        super().unload_model()
        self._wan_cached_stage_modules.clear()

    def set_prepared_model(self, model, base_model: bool = False):
        super().set_prepared_model(model, base_model)
        if not base_model:
            self._apply_time_embedding_override(self.model)
            self._patch_condition_embedder(self.model)

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        pipeline = super().get_pipeline(pipeline_type, load_base_model)
        if hasattr(pipeline, "config"):
            pipeline.config.expand_timesteps = bool(self._wan_expand_timesteps)
        stage_info = self._wan_stage_info()
        if stage_info is not None:
            load_other = self._should_load_other_stage()
            trained_stage = stage_info["trained_stage"]
            other_subfolder = stage_info["other_stage_subfolder"]

            if trained_stage == "low":
                if load_other:
                    pipeline.transformer_2 = pipeline.transformer
                    pipeline.transformer = self._get_or_load_wan_stage_module(other_subfolder)
                else:
                    pipeline.transformer_2 = None
            else:
                if load_other:
                    pipeline.transformer_2 = self._get_or_load_wan_stage_module(other_subfolder)
                else:
                    pipeline.transformer_2 = None

            if load_other:
                pipeline.config.boundary_ratio = stage_info["boundary_ratio"]
            else:
                pipeline.config.boundary_ratio = None

        transformer_primary = getattr(pipeline, "transformer", None)
        self._apply_time_embedding_override(transformer_primary)
        self._patch_condition_embedder(transformer_primary)
        if getattr(pipeline, "transformer_2", None) is not None:
            self._apply_time_embedding_override(pipeline.transformer_2)
            self._patch_condition_embedder(pipeline.transformer_2)

        if hasattr(pipeline, "config"):
            pipeline.config.expand_timesteps = bool(self._wan_expand_timesteps)

        return pipeline

    class _ConditioningImageEmbedder:
        def __init__(self, image_encoder: CLIPVisionModel, image_processor: CLIPImageProcessor, device, dtype):
            self.image_encoder = image_encoder
            self.image_processor = image_processor
            self.device = device
            self.dtype = dtype
            self.image_encoder.eval()
            self.image_encoder.to(device=self.device, dtype=self.dtype)
            for param in self.image_encoder.parameters():
                param.requires_grad_(False)

        @torch.no_grad()
        def encode(self, images):
            processed = self.image_processor(images=images, return_tensors="pt")
            pixel_values = processed["pixel_values"].to(device=self.device, dtype=self.dtype)
            outputs = self.image_encoder(pixel_values=pixel_values, output_hidden_states=True)
            hidden = outputs.hidden_states[-2]
            return [hidden[i] for i in range(hidden.shape[0])]

    def _load_conditioning_clip_components(self, pipeline):
        image_encoder = getattr(pipeline, "image_encoder", None)
        image_processor = getattr(pipeline, "image_processor", None)

        if image_encoder is not None and image_processor is not None:
            return image_encoder, image_processor

        repo_id = getattr(self.config, "image_encoder_pretrained_model_name_or_path", None)
        processor_repo_id = getattr(self.config, "image_processor_pretrained_model_name_or_path", None)

        if repo_id is None:
            repo_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
            if processor_repo_id is None:
                processor_repo_id = repo_id
        if processor_repo_id is None:
            processor_repo_id = repo_id

        def build_candidates(user_value, defaults):
            candidates = []
            if isinstance(user_value, (list, tuple, set)):
                candidates.extend([v for v in user_value if v])
            elif user_value:
                candidates.append(user_value)
            candidates.extend(defaults)
            seen = set()
            ordered = []
            for entry in candidates:
                if entry in seen:
                    continue
                seen.add(entry)
                ordered.append(entry)
            return ordered

        encoder_subfolders = build_candidates(
            getattr(self.config, "image_encoder_subfolder", None),
            ["image_encoder", "vision_encoder", None],
        )
        processor_subfolders = build_candidates(
            getattr(self.config, "image_processor_subfolder", None),
            ["image_processor", "feature_extractor", None],
        )

        encoder_errors = []
        for subfolder in encoder_subfolders:
            try:
                kwargs = {"use_safetensors": True}
                if subfolder is not None:
                    kwargs["subfolder"] = subfolder
                image_encoder = CLIPVisionModel.from_pretrained(repo_id, **kwargs)
                break
            except Exception as exc:  # pragma: no cover - defensive
                encoder_errors.append(f"{repo_id}/{subfolder or '.'}: {exc}")

        if image_encoder is None:
            raise ValueError(
                "Unable to load a CLIP vision encoder for conditioning image embeddings. "
                "Set `image_encoder_pretrained_model_name_or_path` (and optionally `image_encoder_subfolder`) to a "
                "compatible repository. Attempts failed with: " + "; ".join(encoder_errors)
            )

        processor_errors = []
        for subfolder in processor_subfolders:
            try:
                kwargs = {}
                if subfolder is not None:
                    kwargs["subfolder"] = subfolder
                image_processor = CLIPImageProcessor.from_pretrained(processor_repo_id, **kwargs)
                break
            except Exception as exc:  # pragma: no cover - defensive
                processor_errors.append(f"{processor_repo_id}/{subfolder or '.'}: {exc}")

        if image_processor is None:
            raise ValueError(
                "Unable to load a CLIP image processor for conditioning image embeddings. "
                "Set `image_processor_pretrained_model_name_or_path` (and optionally `image_processor_subfolder`). "
                "Attempts failed with: " + "; ".join(processor_errors)
            )

        if pipeline is not None:
            pipeline.image_encoder = image_encoder
            pipeline.image_processor = image_processor

        return image_encoder, image_processor

    def get_conditioning_image_embedder(self):
        if self._conditioning_image_embedder is not None:
            return self._conditioning_image_embedder

        pipeline = self.get_pipeline(PipelineTypes.IMG2VIDEO)
        image_encoder, image_processor = self._load_conditioning_clip_components(pipeline)

        device = getattr(self.accelerator, "device", torch.device("cpu"))
        dtype = getattr(self.config, "weight_dtype", torch.float32)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.float32)

        self._conditioning_image_embedder = self._ConditioningImageEmbedder(
            image_encoder=image_encoder,
            image_processor=image_processor,
            device=device,
            dtype=dtype,
        )
        return self._conditioning_image_embedder

    def tread_init(self):
        """
        Initialize the TREAD model training method for Wan.
        """

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

        logger.info("TREAD training is enabled for Wan")

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        """
        When we're running the pipeline, we'll update the kwargs specifically for this model here.
        """
        # Wan video should max out around 81 frames for efficiency.
        pipeline_kwargs["num_frames"] = min(81, self.config.validation_num_video_frames or 81)
        pipeline_kwargs["output_type"] = "pil"

        input_image = pipeline_kwargs.get("image")
        if isinstance(input_image, list):
            if len(input_image) > 0:
                pipeline_kwargs["image"] = input_image[0]
            if self._uses_last_frame_conditioning() and len(input_image) > 1:
                pipeline_kwargs["last_image"] = input_image[-1]
        elif self._uses_last_frame_conditioning() and input_image is not None and "last_image" not in pipeline_kwargs:
            pipeline_kwargs["last_image"] = input_image

        stage_info = self._wan_stage_info()
        if stage_info is not None:
            trained_stage = stage_info["trained_stage"]
            pipeline_kwargs["num_inference_steps"] = stage_info["sample_steps"]
            pipeline_kwargs["guidance_scale"] = stage_info["guidance"][trained_stage]
            if self._should_load_other_stage():
                other_stage = "low" if trained_stage == "high" else "high"
                pipeline_kwargs["guidance_scale_2"] = stage_info["guidance"][other_stage]
            else:
                pipeline_kwargs.pop("guidance_scale_2", None)
        else:
            pipeline_kwargs.pop("guidance_scale_2", None)

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
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]

        return {
            "prompt_embeds": prompt_embeds,
            # "attention_mask": (
            #     text_embedding["attention_masks"].unsqueeze(0)
            #     if self.config.flux_attention_masked_training
            #     else None
            # ),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]

        return {
            "negative_prompt_embeds": prompt_embeds,
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
        prompt_embeds, masks = self.pipelines[PipelineTypes.TEXT2IMG].encode_prompt(
            prompt=prompts,
            device=self.accelerator.device,
        )
        if self.config.t5_padding == "zero":
            # we can zero the padding tokens if we're just going to mask them later anyway.
            prompt_embeds = prompt_embeds * masks.to(device=prompt_embeds.device).unsqueeze(-1).expand(prompt_embeds.shape)

        return prompt_embeds, masks

    def model_predict(self, prepared_batch):
        """
        Modify the existing model_predict to support TREAD with masked training.
        """
        wan_transformer_kwargs = {
            "hidden_states": prepared_batch["noisy_latents"].to(self.config.weight_dtype),
            "encoder_hidden_states": prepared_batch["encoder_hidden_states"].to(self.config.weight_dtype),
            "timestep": prepared_batch["timesteps"],
            "return_dict": False,
        }

        if prepared_batch.get("conditioning_image_embeds") is not None:
            wan_transformer_kwargs["encoder_hidden_states_image"] = prepared_batch["conditioning_image_embeds"].to(
                self.config.weight_dtype
            )

        self._apply_i2v_conditioning_to_kwargs(prepared_batch, wan_transformer_kwargs)

        # For masking with TREAD, avoid dropping any tokens that are in the mask
        if (
            getattr(self.config, "tread_config", None) is not None
            and self.config.tread_config is not None
            and "conditioning_pixel_values" in prepared_batch
            and prepared_batch["conditioning_pixel_values"] is not None
            and prepared_batch.get("conditioning_type") in ("mask", "segmentation")
        ):
            with torch.no_grad():
                # For video: B, C, T, H, W
                b, c, t, h, w = prepared_batch["latents"].shape
                # Wan uses patch_size (1, 2, 2), so token dimensions are:
                t_tokens = t // 1  # temporal patches
                h_tokens = h // 2  # height patches
                w_tokens = w // 2  # width patches

                mask_vid = prepared_batch["conditioning_pixel_values"]  # (B,C,T,H,W) for video
                # fuse channels → single channel, map to [0,1]
                mask_vid = (mask_vid.mean(1, keepdim=True) + 1) / 2
                # downsample to match token dimensions
                mask_tok = F.interpolate(
                    mask_vid,
                    size=(t_tokens, h_tokens, w_tokens),
                    mode="trilinear",
                    align_corners=False,
                )  # (B,1,t_tok,h_tok,w_tok)
                # Flatten in the same order as patch_embedding
                # After conv3d: (B, D, T', H', W')
                # After flatten(2): (B, D, T'*H'*W') with order T->H->W
                # After transpose(1,2): (B, T'*H'*W', D)
                # So we flatten the mask with the same T->H->W order
                force_keep = mask_tok.squeeze(1).flatten(1) > 0.5  # (B, S_vid)
                existing_force_keep = wan_transformer_kwargs.get("force_keep_mask")
                wan_transformer_kwargs["force_keep_mask"] = (
                    force_keep if existing_force_keep is None else (existing_force_keep | force_keep)
                )

        model_pred = self.model(**wan_transformer_kwargs)[0]

        return {
            "model_prediction": model_pred,
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues.
        """
        stage_info = self._wan_stage_info()
        if stage_info is not None:
            trained_stage = stage_info["trained_stage"]
            self.config.validation_guidance = stage_info["guidance"][trained_stage]
            if hasattr(self.config, "validation_guidance_skip_layers"):
                self.config.validation_guidance_skip_layers = None
            if hasattr(self.config, "validation_guidance_skip_layers_start"):
                self.config.validation_guidance_skip_layers_start = None
            if hasattr(self.config, "validation_guidance_skip_layers_stop"):
                self.config.validation_guidance_skip_layers_stop = None
            if hasattr(self.config, "validation_guidance_skip_scale"):
                self.config.validation_guidance_skip_scale = None

        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        if self.config.aspect_bucket_alignment != 32:
            logger.warning(
                f"{self.NAME} requires an alignment value of 32px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 32

        if self.config.prediction_type is not None:
            logger.warning(f"{self.NAME} does not support prediction type {self.config.prediction_type}.")

        if self.config.tokenizer_max_length is not None:
            logger.warning(f"-!- {self.NAME} supports a max length of 512 tokens, --tokenizer_max_length is ignored -!-")
        self.config.tokenizer_max_length = 512
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
        if self.config.flow_schedule_auto_shift:
            output_args.append("flow_schedule_auto_shift")
        if self.config.flow_schedule_shift is not None:
            output_args.append(f"shift={self.config.flow_schedule_shift}")
        if self.config.flow_use_beta_schedule:
            output_args.append(f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}")
            output_args.append(f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}")
        if self.config.t5_padding != "unmodified":
            output_args.append(f"t5_padding={self.config.t5_padding}")
        output_str = f" (extra parameters={output_args})" if output_args else " (no special parameters set)"

        return output_str

    def get_transforms(self, dataset_type: str = "image"):
        return transforms.Compose(
            [
                VideoToTensor() if dataset_type == "video" else transforms.ToTensor(),
                # Normalize [0,1] input to [-1,1] range using (input - 0.5) / 0.5
                transforms.Normalize([0.5], [0.5]),
            ]
        )


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("wan", Wan)


# Monkeypatch to fix the device meta issue when the text encoder is moved to meta
def _patch_wan_pipeline_execution_device():
    """
    Monkeypatch the WanPipeline to fix the _execution_device property when text encoder is on meta device.
    This prevents the "device meta is invalid" error when using group offloading.
    """
    from diffusers import WanPipeline

    # Store the original property
    original_execution_device = WanPipeline._execution_device

    def _fixed_execution_device(self):
        """
        Fixed _execution_device property that returns the transformer device instead of meta.
        This fixes the issue when text encoder is moved to meta but transformer is on GPU.
        """
        # If we have a transformer and it's not on meta, use its device
        if hasattr(self, "transformer") and self.transformer is not None:
            transformer_device = next(self.transformer.parameters()).device
            if transformer_device.type != "meta":
                return transformer_device

        # Fall back to the original implementation
        return original_execution_device.fget(self)

    # Apply the monkeypatch
    WanPipeline._execution_device = property(_fixed_execution_device)


# Apply the monkeypatch when the module is imported
_patch_wan_pipeline_execution_device()
