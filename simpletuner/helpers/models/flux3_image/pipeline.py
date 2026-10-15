"""Text-to-image and reference-image sampling for FLUX.3 Image Klein."""

import importlib
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Qwen3VLModel

from simpletuner.helpers.models.flux2.autoencoder import AutoencoderKLFlux2
from simpletuner.helpers.models.flux3_image import denormalize_latents, image_ids, normalize_latents, patchify, unpatchify
from simpletuner.helpers.models.flux3_image.text import MAX_TEXT_LENGTH, encode_prompts
from simpletuner.helpers.models.flux3_image.transformer import Flux3ImageTransformer2DModel

FLOW_SHIFT = 6.93


def guided_velocity(positive, negative, scale, zero_star=False):
    positive, negative = positive.float(), negative.float()
    if zero_star:
        p, n = positive.flatten(1), negative.flatten(1)
        alpha = (p * n).sum(1) / n.square().sum(1).clamp_min(1e-8)
        negative = negative * alpha.view(-1, *([1] * (negative.ndim - 1)))
    return negative + scale * (positive - negative)


class Flux3ImagePipeline(DiffusionPipeline, FluxLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["text_encoder", "tokenizer", "vae", "scheduler"]
    _callback_tensor_inputs = ["latents"]

    def __init__(self, transformer, vae=None, text_encoder=None, tokenizer=None, scheduler=None):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler or FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT),
        )
        self.vae_scale_factor = 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=16)
        self._interrupt = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        load_kwargs = {
            key: kwargs.pop(key) for key in ("revision", "cache_dir", "token", "local_files_only") if key in kwargs
        }
        root = Path(pretrained_model_name_or_path)
        if not root.is_dir():
            root = Path(
                snapshot_download(
                    pretrained_model_name_or_path, allow_patterns=["model_index.json", "scheduler/*"], **load_kwargs
                )
            )
        index_path = root / "model_index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        components = {}
        loaders = {
            "transformer": (Flux3ImageTransformer2DModel, "transformer", dtype),
            "vae": (AutoencoderKLFlux2, "vae", torch.float32),
            "text_encoder": (Qwen3VLModel, "text_encoder", dtype),
            "tokenizer": (AutoTokenizer, "text_encoder", None),
        }
        for name, (loader, subfolder, component_dtype) in loaders.items():
            if name in kwargs:
                components[name] = kwargs.pop(name)
            elif index.get(name) == [None, None]:
                components[name] = None
            else:
                options = dict(load_kwargs, subfolder=subfolder)
                if component_dtype is not None:
                    options["torch_dtype"] = component_dtype
                components[name] = loader.from_pretrained(pretrained_model_name_or_path, **options)
        if "scheduler" not in kwargs and "scheduler" in index:
            library, name = index["scheduler"]
            scheduler_cls = getattr(importlib.import_module(library), name)
            components["scheduler"] = scheduler_cls.from_pretrained(root, subfolder="scheduler")
        return cls(**components, **kwargs)

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def encode_prompt(self, prompt, device=None, max_sequence_length=MAX_TEXT_LENGTH):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Text encoding requires a tokenizer and Qwen3-VL text encoder, or cached prompt embeddings.")
        return encode_prompts(
            self.tokenizer,
            self.text_encoder,
            [prompt] if isinstance(prompt, str) else prompt,
            device or self._execution_device,
            max_sequence_length,
        )

    def _reference_tokens(self, images, batch_size, device, dtype, num_images_per_prompt=1):
        if images is None:
            return None, None
        images = images if isinstance(images, (list, tuple)) else [images]
        if len(images) > 10:
            raise ValueError("FLUX.3 Image Klein supports at most ten reference images.")
        if self.vae is None:
            raise ValueError("Encoding reference images requires the VAE.")
        tokens, positions = [], []
        for index, image in enumerate(images):
            pixels = self.image_processor.preprocess(image).to(device=device, dtype=self.vae.dtype)
            latent = self.vae.encode(pixels).latent_dist.mode()
            latent = normalize_latents(patchify(latent), self.vae).to(dtype)
            if latent.shape[0] == 1:
                latent = latent.expand(batch_size, -1, -1, -1)
            elif latent.shape[0] * num_images_per_prompt == batch_size:
                latent = latent.repeat_interleave(num_images_per_prompt, 0)
            if latent.shape[0] != batch_size:
                raise ValueError("Reference image batch size must match the prompt batch size.")
            tokens.append(latent.flatten(2).transpose(1, 2))
            positions.append(image_ids(*latent.shape[-2:], device, reference_index=index))
        return torch.cat(tokens, dim=1), torch.cat(positions, dim=0)

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        image=None,
        height=1024,
        width=1024,
        num_inference_steps=None,
        guidance_scale=None,
        guidance_rescale=0.0,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        output_type="pil",
        return_dict=True,
        max_sequence_length=MAX_TEXT_LENGTH,
        use_cfg_zero_star=False,
        use_zero_init=True,
        zero_steps=1,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=("latents",),
        attention_kwargs=None,
    ):
        if height % 16 or width % 16 or height <= 0 or width <= 0:
            raise ValueError("FLUX.3 image dimensions must be positive multiples of 16.")
        if num_images_per_prompt < 1:
            raise ValueError("num_images_per_prompt must be positive.")
        if any(key not in self._callback_tensor_inputs for key in callback_on_step_end_tensor_inputs):
            raise ValueError("FLUX.3 image callbacks expose only latents.")
        device = self._execution_device
        dtype = self.transformer.dtype
        distilled = self.transformer.config.flow_map
        steps = num_inference_steps if num_inference_steps is not None else (16 if distilled else 50)
        guidance_scale = guidance_scale if guidance_scale is not None else (1.0 if distilled else 3.5)
        if steps < 1:
            raise ValueError("num_inference_steps must be positive.")
        self._guidance_scale = guidance_scale
        self._interrupt = False
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self.encode_prompt(prompt, device, max_sequence_length)
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        batch = prompt_embeds.shape[0]
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype).repeat_interleave(num_images_per_prompt, dim=0)
        if prompt_attention_mask is not None:
            prompt_attention_mask = (
                prompt_attention_mask.to(device).reshape(batch, -1).repeat_interleave(num_images_per_prompt, 0)
            )
        if guidance_scale > 1:
            if negative_prompt_embeds is None:
                prompts = (
                    [negative_prompt or ""] * batch if isinstance(negative_prompt, (str, type(None))) else negative_prompt
                )
                negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                    prompts, device, max_sequence_length
                )
            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype).repeat_interleave(
                num_images_per_prompt, 0
            )
            if negative_prompt_attention_mask is not None:
                negative_prompt_attention_mask = (
                    negative_prompt_attention_mask.to(device).reshape(batch, -1).repeat_interleave(num_images_per_prompt, 0)
                )
        batch *= num_images_per_prompt
        shape = (batch, self.transformer.config.in_channels, height // 16, width // 16)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32).to(torch.bfloat16).float()
        else:
            if tuple(latents.shape) != shape:
                raise ValueError(f"Expected normalized packed image latents with shape {shape}.")
            latents = latents.to(device=device, dtype=torch.float32)
        reference, reference_ids = self._reference_tokens(image, batch, device, dtype, num_images_per_prompt)
        positions = image_ids(height // 16, width // 16, device)
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            self.scheduler.set_timesteps(sigmas=np.linspace(1, 0, steps + 1)[:-1], device=device)
        else:
            self.scheduler.set_timesteps(steps, device=device)
        self._num_timesteps = len(self.scheduler.timesteps)
        with self.progress_bar(total=steps) as progress:
            for index, time in enumerate(self.scheduler.timesteps):
                if self.interrupt:
                    break
                normalized_time = (time / 1000).expand(batch)
                target_time = self.scheduler.sigmas[index + 1].expand(batch) if distilled else None
                model_inputs = {
                    "hidden_states": latents.flatten(2).transpose(1, 2).to(dtype),
                    "timestep": normalized_time,
                    "target_timestep": target_time,
                    "img_ids": positions,
                    "conditioning_hidden_states": reference,
                    "conditioning_ids": reference_ids,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }
                positive = self.transformer(
                    encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_attention_mask, **model_inputs
                )[0].float()
                if guidance_scale > 1:
                    if use_cfg_zero_star and use_zero_init and index <= zero_steps:
                        positive = torch.zeros_like(positive)
                    else:
                        negative = self.transformer(
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_attention_mask=negative_prompt_attention_mask,
                            **model_inputs,
                        )[0].float()
                        guided = guided_velocity(positive, negative, guidance_scale, use_cfg_zero_star)
                        if guidance_rescale > 0:
                            dims = tuple(range(1, positive.ndim))
                            rescaled = guided * (
                                positive.std(dims, keepdim=True) / guided.std(dims, keepdim=True).clamp_min(1e-8)
                            )
                            guided = guidance_rescale * rescaled + (1 - guidance_rescale) * guided
                        positive = guided
                velocity = positive.transpose(1, 2).reshape(shape)
                latents = self.scheduler.step(velocity, time, latents, generator=generator, return_dict=False)[0].float()
                if callback_on_step_end is not None:
                    callback_kwargs = {key: latents for key in callback_on_step_end_tensor_inputs}
                    returned = callback_on_step_end(self, index, time, callback_kwargs)
                    latents = returned.pop("latents", latents)
                progress.update()
        if output_type == "latent":
            images = latents
        else:
            if self.vae is None:
                raise ValueError("Decoding generated images requires the VAE.")
            pixels = unpatchify(denormalize_latents(latents.to(self.vae.dtype), self.vae))
            pixels = self.vae.decode(pixels, return_dict=False)[0]
            images = self.image_processor.postprocess(pixels, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
