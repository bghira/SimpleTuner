# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ftfy
import numpy as np
import regex as re
import torch
import torch.nn.functional as F
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2Model, Wav2Vec2Processor

from simpletuner.helpers.models.wan_s2v import WAV2VEC2_NUM_LAYERS
from simpletuner.helpers.models.wan_s2v.transformer import WanS2VTransformer3DModel
from simpletuner.helpers.training.lycoris import apply_tlora_inference_mask, clear_tlora_mask

logger = logging.get_logger(__name__)


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class WanS2VPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    """
    Pipeline for Speech-to-Video generation using Wan2.2-S2V.

    This pipeline generates video conditioned on audio (speech), text prompts,
    and reference images.
    """

    model_cpu_offload_seq = "text_encoder->audio_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        audio_encoder: Wav2Vec2Model,
        audio_processor: Wav2Vec2Processor,
        transformer: WanS2VTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            audio_encoder=audio_encoder,
            audio_processor=audio_processor,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

        self.vae_scale_factor_spatial = (
            2 ** sum(vae.config.temperal_downsample) if hasattr(vae.config, "temperal_downsample") else 8
        )
        self.vae_scale_factor_temporal = (
            2 ** sum(vae.config.temperal_downsample) if hasattr(vae.config, "temperal_downsample") else 4
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        prompt_embeds = self.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )[0]

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(p) for p in prompt]

        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        if do_classifier_free_guidance and negative_prompt is not None:
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt = [prompt_clean(p) for p in negative_prompt]
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            negative_prompt_embeds = None

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def encode_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Encode audio waveform using Wav2Vec2.

        Args:
            audio: Audio waveform tensor [B, samples] or [samples]
            sample_rate: Audio sample rate (should be 16000 for Wav2Vec2)
            device: Target device
            dtype: Target dtype

        Returns:
            Audio embeddings [B, num_layers, audio_dim, T]
        """
        device = device or self._execution_device
        dtype = dtype or torch.float32

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Process audio through Wav2Vec2
        audio = audio.to(device=device, dtype=dtype)

        with torch.no_grad():
            outputs = self.audio_encoder(
                audio,
                output_hidden_states=True,
                return_dict=True,
            )

        # Stack all hidden states [B, num_layers, T, audio_dim]
        hidden_states = torch.stack(outputs.hidden_states, dim=1)
        # Permute to [B, num_layers, audio_dim, T]
        hidden_states = hidden_states.permute(0, 1, 3, 2)

        return hidden_states

    def linear_interpolation(
        self,
        audio_embeds: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """Interpolate audio embeddings to match video frame count."""
        # audio_embeds: [B, num_layers, audio_dim, T_audio]
        # target_length: number of video frames
        audio_embeds = F.interpolate(
            audio_embeds.flatten(0, 1),  # [B*num_layers, audio_dim, T]
            size=target_length,
            mode="linear",
            align_corners=False,
        )
        audio_embeds = audio_embeds.unflatten(0, (-1, WAV2VEC2_NUM_LAYERS))
        return audio_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents

    def encode_image(
        self,
        image: Union[torch.Tensor, Image.Image],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode reference image to latent space."""
        if isinstance(image, Image.Image):
            image = self.video_processor.preprocess(image, height=None, width=None)

        image = image.to(device=device, dtype=dtype)

        if image.dim() == 4:
            image = image.unsqueeze(2)  # Add temporal dimension

        image_latents = self.vae.encode(image).latent_dist.sample()

        # Normalize latents
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
            image_latents = (image_latents - latents_mean) * latents_std

        return image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        audio: Optional[torch.Tensor] = None,
        image: Optional[Union[torch.Tensor, Image.Image]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.5,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Generate video from audio, text, and reference image.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt
            audio: Audio waveform tensor [samples] or [B, samples] at 16kHz
            image: Reference image for first frame
            height: Video height
            width: Video width
            num_frames: Number of frames to generate
            num_inference_steps: Denoising steps
            guidance_scale: Classifier-free guidance scale
            num_videos_per_prompt: Videos per prompt
            generator: Random generator
            latents: Pre-generated latents
            prompt_embeds: Pre-computed prompt embeddings
            negative_prompt_embeds: Pre-computed negative embeddings
            output_type: Output format ("pil", "np", "pt", "latent")
            return_dict: Return WanPipelineOutput or tuple
        """
        device = self._execution_device
        dtype = self.transformer.dtype

        # Validate inputs
        if audio is None:
            raise ValueError("Audio input is required for S2V generation")
        if image is None:
            raise ValueError("Reference image is required for S2V generation")

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode prompts
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # Encode audio
        audio_embeds = self.encode_audio(audio, device=device, dtype=dtype)
        # Interpolate to match video frames
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        audio_embeds = self.linear_interpolation(audio_embeds, num_latent_frames)

        if do_classifier_free_guidance:
            # Zero audio for unconditional
            audio_embeds = torch.cat([torch.zeros_like(audio_embeds), audio_embeds], dim=0)

        # Encode reference image
        image_latents = self.encode_image(image, device=device, dtype=dtype)
        if do_classifier_free_guidance:
            image_latents = torch.cat([image_latents, image_latents], dim=0)

        # Prepare latents
        num_channels_latents = self.vae.config.z_dim
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        if do_classifier_free_guidance:
            latents = torch.cat([latents, latents], dim=0)

        # Prepare pose latents (zeros for now - can be extended)
        pose_latents = torch.zeros_like(latents[:, :16])

        # Prepare empty motion latents (for first clip)
        motion_latents = torch.zeros(
            latents.shape[0], num_channels_latents, 0, latents.shape[3], latents.shape[4], device=device, dtype=dtype
        )

        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Denoising loop
        # Set begin index to avoid timestep lookup in scheduler.step() which can fail due to
        # floating-point precision issues. See: https://github.com/huggingface/diffusers/pull/11696
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            latent_model_input = latents

            # Prepare timestep
            timestep = t.expand(latent_model_input.shape[0])

            # Predict noise
            _tlora_cfg = getattr(self, "_tlora_config", None)
            if _tlora_cfg:
                apply_tlora_inference_mask(
                    timestep=int(t),
                    max_timestep=self.scheduler.config.num_train_timesteps,
                    max_rank=_tlora_cfg["max_rank"],
                    min_rank=_tlora_cfg["min_rank"],
                    alpha=_tlora_cfg["alpha"],
                )
            try:
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    motion_latents=motion_latents,
                    audio_embeds=audio_embeds,
                    image_latents=image_latents,
                    pose_latents=pose_latents,
                    motion_frames=[17, 5],
                    drop_motion_frames=True,
                    add_last_motion=0,
                    return_dict=False,
                )[0]
            finally:
                if _tlora_cfg:
                    clear_tlora_mask()

            # Classifier-free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = latents.chunk(2)[1]  # Keep only conditional

            # Step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if do_classifier_free_guidance:
                latents = torch.cat([latents, latents], dim=0)

            # Callback
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)

        # Final latents
        if do_classifier_free_guidance:
            latents = latents.chunk(2)[1]

        if output_type == "latent":
            return WanPipelineOutput(frames=latents)

        # Decode
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
            latents = latents / latents_std + latents_mean

        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
