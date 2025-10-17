from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from .controlnet import ChromaControlNetModel
from .pipeline import (
    ChromaLoraLoaderMixin,
    ChromaPipeline,
    ChromaPipelineOutput,
    XLA_AVAILABLE,
    calculate_shift,
)

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ChromaControlNetPipeline(ChromaPipeline):
    r"""
    Chroma text-to-image pipeline augmented with a single ControlNet branch.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->controlnet->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "controlnet_cond"]

    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        transformer,
        controlnet: ChromaControlNetModel,
        image_encoder=None,
        feature_extractor=None,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        if isinstance(controlnet, (list, tuple)):
            raise ValueError("ChromaControlNetPipeline only supports a single ControlNet instance.")
        self.register_modules(controlnet=controlnet)

    def _compute_control_schedule(
        self, num_steps: int, start: Union[float, List[float]], end: Union[float, List[float]]
    ) -> List[float]:
        if isinstance(start, list):
            start = start[0]
        if isinstance(end, list):
            end = end[0]
        start = float(start)
        end = float(end)
        keep = []
        for idx in range(num_steps):
            progress = idx / max(num_steps - 1, 1)
            keep.append(1.0 if start <= progress <= end else 0.0)
        return keep

    def _prepare_controlnet_cond(
        self,
        control_image: Optional[PipelineImageInput],
        controlnet_cond: Optional[torch.Tensor],
        batch_size: int,
        num_images_per_prompt: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        if controlnet_cond is not None:
            if isinstance(controlnet_cond, torch.Tensor):
                controlnet_cond = controlnet_cond.to(device=device, dtype=dtype)
                if controlnet_cond.ndim == 4:
                    b, c, h, w = controlnet_cond.shape
                    controlnet_cond = self._pack_latents(controlnet_cond, b, c, h, w)
                elif controlnet_cond.ndim != 3:
                    raise ValueError("`controlnet_cond` must be a packed tensor of shape (B, S, C) or (B, C, H, W).")
                return controlnet_cond
            raise ValueError("`controlnet_cond` must be a torch.Tensor when provided.")

        if control_image is None:
            raise ValueError("You must provide either `control_image` or `controlnet_cond`.")

        image = self.image_processor.preprocess(control_image, height=height, width=width)
        image = image.to(device=device, dtype=self.vae.dtype)
        desired_batch = batch_size * num_images_per_prompt
        if image.shape[0] == 1:
            image = image.repeat(desired_batch, 1, 1, 1)
        elif image.shape[0] == batch_size and num_images_per_prompt > 1:
            image = image.repeat_interleave(num_images_per_prompt, dim=0)
        elif image.shape[0] != desired_batch:
            raise ValueError(
                f"control_image batch dimension {image.shape[0]} does not match expected "
                f"{desired_batch} based on prompt batch size."
            )

        if isinstance(generator, list):
            control_generator = generator[0]
        else:
            control_generator = generator

        latents = self.vae.encode(image).latent_dist.sample(generator=control_generator)
        latents = latents * self.vae.config.scaling_factor
        latents = self._pack_latents(
            latents,
            latents.shape[0],
            latents.shape[1],
            latents.shape[2],
            latents.shape[3],
        )
        return latents.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 35,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        control_image: Optional[PipelineImageInput] = None,
        controlnet_cond: Optional[torch.Tensor] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        controlnet_blocks_repeat: bool = False,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        skip_guidance_layers: Optional[List[int]] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._current_timestep = None
        self._interrupt = False
        if skip_guidance_layers is not None and not isinstance(skip_guidance_layers, list):
            skip_guidance_layers = list(skip_guidance_layers)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        (
            prompt_embeds,
            text_ids,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_text_ids,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        original_prompt_embeds = prompt_embeds
        original_text_ids = text_ids

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        controlnet_latents = self._prepare_controlnet_cond(
            control_image=control_image,
            controlnet_cond=controlnet_cond,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            device=device,
            dtype=latents.dtype,
            generator=generator,
        )
        if controlnet_latents.shape != latents.shape:
            raise ValueError("Prepared ControlNet conditioning latents do not match the main latent shape.")

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

        attention_mask = self._prepare_attention_mask(
            batch_size=latents.shape[0],
            sequence_length=image_seq_len,
            dtype=latents.dtype,
            attention_mask=prompt_attention_mask,
        )
        negative_attention_mask = self._prepare_attention_mask(
            batch_size=latents.shape[0],
            sequence_length=image_seq_len,
            dtype=latents.dtype,
            attention_mask=negative_prompt_attention_mask,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        if isinstance(controlnet_conditioning_scale, list):
            control_scales = controlnet_conditioning_scale
        else:
            control_scales = [controlnet_conditioning_scale]

        control_keep = self._compute_control_schedule(num_inference_steps, control_guidance_start, control_guidance_end)

        if isinstance(self, ChromaLoraLoaderMixin) and USE_PEFT_BACKEND and lora_scale is not None:
            scale_lora_layers(self.controlnet, lora_scale)

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                cond_scale = control_scales[0] * control_keep[i]

                if cond_scale > 0:
                    controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
                        hidden_states=latents,
                        controlnet_cond=controlnet_latents,
                        conditioning_scale=cond_scale,
                        timestep=timestep / 1000,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        attention_mask=attention_mask,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )
                else:
                    controlnet_block_samples = None
                    controlnet_single_block_samples = None

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    attention_mask=attention_mask,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    return_dict=False,
                    controlnet_blocks_repeat=controlnet_blocks_repeat,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_text = noise_pred
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        attention_mask=negative_attention_mask,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]
                    if image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred_text - neg_noise_pred)

                    should_skip_layers = (
                        skip_guidance_layers is not None
                        and i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                    )
                    if should_skip_layers:
                        skip_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            encoder_hidden_states=original_prompt_embeds,
                            txt_ids=original_text_ids,
                            img_ids=latent_image_ids,
                            attention_mask=attention_mask,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            controlnet_block_samples=controlnet_block_samples,
                            controlnet_single_block_samples=controlnet_single_block_samples,
                            return_dict=False,
                            controlnet_blocks_repeat=controlnet_blocks_repeat,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = noise_pred + (noise_pred_text - skip_noise_pred) * self._skip_layer_guidance_scale

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    if skip_guidance_layers is not None:
                        original_prompt_embeds = prompt_embeds

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if isinstance(self, ChromaLoraLoaderMixin) and USE_PEFT_BACKEND and lora_scale is not None:
            unscale_lora_layers(self.controlnet, lora_scale)

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ChromaPipelineOutput(images=image)
