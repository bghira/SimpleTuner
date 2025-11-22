# This was MIT-licensed by Kandinsky Lab; now AGPL-3.0-or-later, SimpleTuner (c) bghira
from typing import List, Optional, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from PIL import Image
from torchvision import transforms

from .pipeline_kandinsky5_t2i import Kandinsky5T2IPipeline


class Kandinsky5I2IPipeline(Kandinsky5T2IPipeline):
    """
    Image-to-image pipeline variant that injects conditioning latents + mask for visual_cond=True checkpoints.
    """

    @staticmethod
    def _preprocess_image(image: Union[Image.Image, torch.Tensor], height: int, width: int, device, dtype):
        if isinstance(image, Image.Image):
            image = transforms.functional.resize(image, (height, width))
            image = transforms.functional.to_tensor(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
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

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Union[Image.Image, torch.Tensor]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_images_per_prompt: Optional[int] = 1,
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
            raise ValueError("Image-to-image pipeline requires an input image.")

        return super().__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
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

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 1024,
        width: int = 1024,
        num_frames: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        image: Optional[Union[Image.Image, torch.Tensor]] = None,
    ):
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

        if not getattr(self.transformer.config, "visual_cond", False):
            return latents
        if image is None:
            return latents
        if self.vae is None:
            raise ValueError("VAE is not loaded; cannot encode conditioning image for I2I.")

        cond_image = self._preprocess_image(
            image,
            height=height,
            width=width,
            device=device,
            dtype=self.vae.dtype if getattr(self, "vae", None) is not None else dtype,
        )
        cond_image = cond_image.unsqueeze(0).unsqueeze(2)  # B=1, C, T=1, H, W
        with torch.no_grad():
            cond_latent = self.vae.encode(cond_image).latent_dist.sample()
        cond_latent = cond_latent * getattr(self.vae.config, "scaling_factor", 1.0)
        cond_latent = cond_latent.permute(0, 2, 3, 4, 1)  # B, T, H, W, C

        cond_start = num_channels_latents
        cond_end = cond_start + num_channels_latents
        mask_start = cond_end

        latents[:, 0, :, :, cond_start:cond_end] = cond_latent[:, 0]
        latents[:, 0, :, :, mask_start:] = 1.0

        return latents
