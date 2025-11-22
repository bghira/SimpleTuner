# This was MIT-licensed by Kandinsky Lab; now AGPL-3.0-or-later, SimpleTuner (c) bghira
from typing import Dict, List, Optional, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import logging
from PIL import Image
from torchvision import transforms

from .pipeline_kandinsky5_t2v import Kandinsky5T2VPipeline

logger = logging.get_logger(__name__)


class Kandinsky5I2VPipeline(Kandinsky5T2VPipeline):
    """
    Image-to-video pipeline variant that encodes a conditioning image into the visual conditioning
    channels expected by the Kandinsky5 transformer (first-frame conditioning + mask).
    """

    @staticmethod
    def _preprocess_image(image: Union[Image.Image, torch.Tensor], height: int, width: int, device, dtype):
        if isinstance(image, Image.Image):
            image = transforms.functional.resize(image, (height, width))
            image = transforms.functional.to_tensor(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                # (C, H, W)
                pass
            elif image.dim() == 4 and image.shape[0] == 1:
                image = image[0]
            else:
                raise ValueError(f"Unexpected image tensor shape {image.shape}")
            image = transforms.functional.resize(image, (height, width))
        else:
            raise ValueError("image must be PIL.Image or torch.Tensor")
        image = image.to(device=device, dtype=dtype)
        image = image * 2.0 - 1.0  # scale to [-1, 1]
        return image

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Union[Image.Image, torch.Tensor]] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds_qwen: Optional[torch.Tensor] = None,
        prompt_embeds_clip: Optional[torch.Tensor] = None,
        negative_prompt_embeds_qwen: Optional[torch.Tensor] = None,
        negative_prompt_embeds_clip: Optional[torch.Tensor] = None,
        prompt_cu_seqlens: Optional[torch.Tensor] = None,
        negative_prompt_cu_seqlens: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        **kwargs,
    ):
        if image is None:
            raise ValueError("Image-to-video pipeline requires an input image.")

        # Run the standard T2V path up to latent prep
        video_output = super().__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds_qwen=prompt_embeds_qwen,
            prompt_embeds_clip=prompt_embeds_clip,
            negative_prompt_embeds_qwen=negative_prompt_embeds_qwen,
            negative_prompt_embeds_clip=negative_prompt_embeds_clip,
            prompt_cu_seqlens=prompt_cu_seqlens,
            negative_prompt_cu_seqlens=negative_prompt_cu_seqlens,
            output_type=output_type,
            return_dict=return_dict,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            image=image,
            **kwargs,
        )
        return video_output

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        image: Optional[Union[Image.Image, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if image is None:
            raise ValueError("Image-to-video pipeline requires an input image.")

        latents = super().prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        if not self.transformer.visual_cond:
            return latents

        if self.vae is None:
            raise ValueError("VAE is not loaded; cannot encode conditioning image for I2V.")

        # Encode conditioning image into visual_cond channels (first latent frame) and set mask=1 for it.
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        cond_image = self._preprocess_image(image, height, width, device=device, dtype=self.vae.dtype)
        cond_image = cond_image.unsqueeze(0).unsqueeze(2)  # B=1, C, T=1, H, W
        with torch.no_grad():
            cond_latent = self.vae.encode(cond_image).latent_dist.sample()
        cond_latent = cond_latent * self.vae.config.scaling_factor
        cond_latent = cond_latent.to(dtype=dtype if dtype is not None else cond_latent.dtype)
        cond_latent = cond_latent.permute(0, 2, 3, 4, 1)  # B, T, H, W, C

        # latents layout: [noise (16), visual_cond (16), mask (1)]
        cond_start = num_channels_latents
        cond_end = cond_start + num_channels_latents
        mask_start = cond_end

        latents[:, 0, :, :, cond_start:cond_end] = cond_latent[:, 0]
        latents[:, 0, :, :, mask_start:] = 1.0
        # Leave other frames zeroed.

        return latents
