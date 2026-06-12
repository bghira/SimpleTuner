from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import AutoTokenizer, T5GemmaModel

from simpletuner.helpers.models.zlab_i1.transformer import (
    FLUX2_LATENTS_MEAN,
    FLUX2_LATENTS_VAR,
    ZlabI1Transformer2DModel,
)


def _pixel_unshuffle_2x(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"i1 latents require even spatial dimensions, got {(height, width)}.")
    latents = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
    return latents.permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height // 2, width // 2)


def _pixel_shuffle_2x(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    if channels % 4 != 0:
        raise ValueError(f"i1 pixel-shuffle expects channel count divisible by 4, got {channels}.")
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    return latents.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, height * 2, width * 2)


def _time_grid(num_steps: int, shift: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    times = torch.linspace(0.0, 1.0, num_steps + 1, dtype=dtype, device=device)
    if shift != 0.0:
        shift_tensor = torch.tensor(shift, dtype=dtype, device=device)
        times = (shift_tensor * times) / (1.0 + (shift_tensor - 1.0) * times)
    return times


def optimized_scale(positive_flat: torch.Tensor, negative_flat: torch.Tensor):
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm


class ZlabI1Pipeline(DiffusionPipeline, FluxLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "attention_mask"]

    transformer_name = "transformer"
    text_encoder_name = "text_encoder"

    def __init__(
        self,
        transformer: ZlabI1Transformer2DModel,
        vae: AutoencoderKL,
        text_encoder: Optional[T5GemmaModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        scheduler=None,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        transformer: Optional[ZlabI1Transformer2DModel] = None,
        vae: Optional[AutoencoderKL] = None,
        text_encoder: Optional[T5GemmaModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        scheduler=None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        transformer_kwargs = {
            key: kwargs.pop(key)
            for key in ("musubi_blocks_to_swap", "musubi_block_swap_device")
            if key in kwargs
        }
        del kwargs
        if transformer is None:
            transformer = ZlabI1Transformer2DModel.from_pretrained(pretrained_model_name_or_path, **transformer_kwargs)
            if torch_dtype is not None:
                transformer = transformer.to(dtype=torch_dtype)
        if vae is None:
            vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae", torch_dtype=torch_dtype)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2b-2b-ul2-it", use_fast=False)
        if text_encoder is None:
            text_encoder = T5GemmaModel.from_pretrained("google/t5gemma-2b-2b-ul2-it", torch_dtype=torch_dtype)
        return cls(transformer=transformer, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler)

    @staticmethod
    def _prepare_text_mask(mask: Optional[torch.Tensor], prompt_embeds: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.ones(prompt_embeds.shape[:2], dtype=torch.bool, device=prompt_embeds.device)
        if mask.ndim == 3 and mask.shape[1] == 1:
            mask = mask[:, 0]
        if mask.ndim != 2:
            raise ValueError(f"attention_mask must have shape [batch, tokens], got {tuple(mask.shape)}.")
        return mask.to(device=prompt_embeds.device, dtype=torch.bool)

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        if prompt_embeds is None:
            if self.tokenizer is None or self.text_encoder is None:
                raise ValueError("`prompt_embeds` must be provided when tokenizer or text_encoder is not available.")
            self.text_encoder.to(device)
            prompts = [prompt] if isinstance(prompt, str) else prompt
            tokenized = self.tokenizer(
                prompts,
                max_length=max_sequence_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            inputs = {key: value.to(device) for key, value in tokenized.items()}
            encoder = getattr(self.text_encoder, "encoder", self.text_encoder)
            with torch.no_grad():
                outputs = encoder(**inputs)
            prompt_embeds = outputs.last_hidden_state.float()
            attention_mask = inputs.get("attention_mask")
        else:
            prompt_embeds = prompt_embeds.to(device=device)

        attention_mask = self._prepare_text_mask(attention_mask, prompt_embeds)
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_images_per_prompt, dim=0)
        return prompt_embeds, attention_mask

    def _prepare_cfg_conditioning(
        self,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, cond_len, _ = prompt_embeds.shape
        uncond = self.transformer.text_encoder_adapter.learnable_null_caption.to(
            device=prompt_embeds.device,
            dtype=dtype,
        )
        if uncond.shape[0] == 1 and batch > 1:
            uncond = uncond.repeat(batch, 1, 1)
        uncond_len = uncond.shape[1]
        if uncond_len < cond_len:
            uncond = torch.cat(
                [
                    uncond,
                    torch.zeros(batch, cond_len - uncond_len, uncond.shape[2], device=prompt_embeds.device, dtype=dtype),
                ],
                dim=1,
            )
            uncond_mask = attention_mask & (torch.arange(cond_len, device=prompt_embeds.device)[None] < uncond_len)
        else:
            uncond = uncond[:, :cond_len]
            uncond_mask = attention_mask
        return torch.cat([prompt_embeds.to(dtype=dtype), uncond], dim=0), torch.cat([attention_mask, uncond_mask], dim=0)

    def _predict_velocity(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        skip_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:
        forward_cache = self.transformer.prepare_forward_cache(
            prompt_embeds,
            attention_mask,
            self.transformer.hw * self.transformer.hw,
        )
        return self.transformer(
            latents,
            timestep,
            prompt_embeds,
            attention_mask,
            forward_cache,
            skip_layers=skip_layers,
        )

    @staticmethod
    def _unscale_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
        packed = _pixel_unshuffle_2x(latents)
        mean = torch.tensor(FLUX2_LATENTS_MEAN, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
        var = torch.tensor(FLUX2_LATENTS_VAR, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
        return _pixel_shuffle_2x(packed * torch.sqrt(var + 0.0001) + mean)

    def _decode_latents(self, latents: torch.Tensor, output_type: str) -> torch.Tensor | np.ndarray | list[Image.Image]:
        latents = self._unscale_flux2_latents(latents)
        image = self.vae.decode(latents.to(dtype=next(self.vae.parameters()).dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        if output_type == "pt":
            return image
        image = image.permute(0, 2, 3, 1).float().cpu().numpy()
        if output_type == "np":
            return image
        if output_type != "pil":
            raise ValueError("output_type must be one of 'pil', 'np', 'pt', or 'latent'.")
        image = (image * 255).round().astype("uint8")
        return [Image.fromarray(item) for item in image]

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[str | list[str]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 250,
        guidance_scale: float = 12.0,
        guidance_rescale: Optional[float] = 0.7,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator | list[torch.Generator]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str | list[str]] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        inference_timestep_shift: float = 0.0,
        skip_guidance_layers: Optional[list[int]] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        use_cfg_zero_star: bool = True,
        use_zero_init: bool = True,
        zero_steps: int = 0,
        no_cfg_until_timestep: int = 0,
        cfg_end_timestep: Optional[int] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: Optional[list[str]] = None,
        **kwargs,
    ):
        del negative_prompt, negative_prompt_embeds, negative_attention_mask, kwargs
        if height != width:
            raise ValueError("zlab i1 currently supports square validation images only.")
        expected_height = self.transformer.input_size * 8
        if height != expected_height or width != expected_height:
            raise ValueError(f"zlab i1 transformer expects {expected_height}x{expected_height}, got {height}x{width}.")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")

        device = self._execution_device
        self.transformer.to(device)
        self.vae.to(device)
        dtype = torch.bfloat16
        if prompt_embeds is not None:
            dtype = prompt_embeds.dtype if prompt_embeds.dtype in (torch.float16, torch.bfloat16, torch.float32) else dtype
        elif next(self.transformer.parameters()).dtype in (torch.float16, torch.bfloat16, torch.float32):
            dtype = next(self.transformer.parameters()).dtype

        prompt_embeds, attention_mask = self.encode_prompt(
            prompt=prompt or "",
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
        )
        batch_size = prompt_embeds.shape[0]

        if latents is None:
            shape = (batch_size, self.transformer.in_channels, self.transformer.input_size, self.transformer.input_size)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        times = _time_grid(num_inference_steps, inference_timestep_shift, device=device, dtype=dtype)

        callback_inputs = callback_on_step_end_tensor_inputs or self._callback_tensor_inputs
        for step_index in range(num_inference_steps):
            timestep = times[step_index].expand(batch_size)
            within_cfg_window = step_index >= int(no_cfg_until_timestep) and (
                cfg_end_timestep is None or step_index <= int(cfg_end_timestep)
            )
            apply_cfg = guidance_scale > 1.0 and within_cfg_window
            if apply_cfg:
                cfg_embeds, cfg_mask = self._prepare_cfg_conditioning(prompt_embeds, attention_mask, dtype=dtype)
                latent_input = torch.cat([latents, latents], dim=0)
                timestep_input = torch.cat([timestep, timestep], dim=0)
                velocity = self._predict_velocity(latent_input, timestep_input, cfg_embeds, cfg_mask)
                cond, uncond = velocity.chunk(2, dim=0)
                if use_cfg_zero_star:
                    positive_flat = cond.reshape(batch_size, -1)
                    negative_flat = uncond.reshape(batch_size, -1)
                    alpha = optimized_scale(positive_flat, negative_flat).view(batch_size, 1, 1, 1).to(dtype=cond.dtype)
                    if step_index <= int(zero_steps) and use_zero_init:
                        velocity = cond * 0.0
                    else:
                        velocity = uncond * alpha + guidance_scale * (cond - uncond * alpha)
                else:
                    velocity = cond + (guidance_scale - 1.0) * (cond - uncond)

                should_skip_layers = (
                    skip_guidance_layers is not None
                    and step_index > num_inference_steps * skip_layer_guidance_start
                    and step_index < num_inference_steps * skip_layer_guidance_stop
                )
                if should_skip_layers:
                    skip_velocity = self._predict_velocity(
                        latents,
                        timestep,
                        prompt_embeds.to(dtype=dtype),
                        attention_mask,
                        skip_layers=skip_guidance_layers,
                    )
                    velocity = velocity + (cond - skip_velocity) * skip_layer_guidance_scale

                if guidance_rescale is not None:
                    axes = tuple(range(1, velocity.ndim))
                    std_cond = torch.std(cond.float(), dim=axes, keepdim=True)
                    std_guided = torch.std(velocity.float(), dim=axes, keepdim=True)
                    factor = (std_cond / (std_guided + 1e-8)).to(dtype=velocity.dtype)
                    velocity = velocity * (1.0 - guidance_rescale + guidance_rescale * factor)
            else:
                velocity = self._predict_velocity(
                    latents,
                    timestep,
                    prompt_embeds.to(dtype=dtype),
                    attention_mask,
                )
            latents = latents + (times[step_index + 1] - times[step_index]) * velocity

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for key in callback_inputs:
                    if key == "latents":
                        callback_kwargs[key] = latents
                    elif key == "prompt_embeds":
                        callback_kwargs[key] = prompt_embeds
                    elif key == "attention_mask":
                        callback_kwargs[key] = attention_mask
                callback_result = callback_on_step_end(self, step_index, times[step_index], callback_kwargs)
                if isinstance(callback_result, dict) and "latents" in callback_result:
                    latents = callback_result["latents"]

        if output_type == "latent":
            images = latents
        else:
            images = self._decode_latents(latents, output_type=output_type)

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
