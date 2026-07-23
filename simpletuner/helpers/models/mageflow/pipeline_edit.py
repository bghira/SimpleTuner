from __future__ import annotations

from typing import Optional

import torch
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from einops import rearrange
from PIL import Image

from simpletuner.helpers.models.mageflow.pipeline import (
    MageFlowPipeline,
    _mageflow_velocity,
    _make_divisible_by_16,
    _pack_prompt_embeds,
    _tokens_to_image,
)
from simpletuner.helpers.models.mageflow.vendor.pipeline import (
    PROMPT_TEMPLATE,
    _build_pack_ctx,
    _lens_to_cu,
    _resize_long_edge,
)


def _load_image(image) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)
    return image.convert("RGB")


def _preprocess_ref_image(image: Image.Image, height: int, width: int, device, dtype) -> torch.Tensor:
    from torchvision.transforms import functional as TF

    image = image.convert("RGB")
    image = TF.resize(image, [height, width], interpolation=TF.InterpolationMode.BICUBIC)
    tensor = TF.to_tensor(image).mul_(2.0).sub_(1.0).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def _call_vae_encode(vae, pixels: torch.Tensor) -> torch.Tensor:
    encoded = vae.encode(pixels)
    if hasattr(encoded, "latent_dist"):
        return encoded.latent_dist.sample()
    if hasattr(encoded, "sample"):
        return encoded.sample
    return encoded


class MageFlowEditPipeline(MageFlowPipeline):
    @torch.no_grad()
    def encode_prompt(
        self,
        prompt,
        images=None,
        device=None,
        num_images_per_prompt: int = 1,
        prompt_template: str = "mage-flow-edit",
        max_sequence_length: int = 2048,
        vl_cond_long_edge: int = 384,
        **kwargs,
    ):
        del kwargs
        if images is None or self.processor is None:
            return super().encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                prompt_template=prompt_template,
                max_sequence_length=max_sequence_length,
            )
        if self.text_encoder is None:
            raise ValueError("MageFlowEditPipeline.encode_prompt requires a loaded text_encoder.")

        device = device or self._execution_device
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        images_per_prompt = images
        if not isinstance(images_per_prompt, (list, tuple)) or (
            images_per_prompt and not isinstance(images_per_prompt[0], (list, tuple))
        ):
            images_per_prompt = [images_per_prompt] * len(prompts)

        template = PROMPT_TEMPLATE[prompt_template]["template"]
        drop_idx = int(PROMPT_TEMPLATE[prompt_template].get("start_idx", 0))
        input_ids = []
        attention_masks = []
        pixel_values = []
        image_grid_thw = []
        for prompt_item, refs in zip(prompts, images_per_prompt, strict=True):
            ref_list = refs if isinstance(refs, (list, tuple)) else [refs]
            ref_list = [_resize_long_edge(_load_image(ref), vl_cond_long_edge) for ref in ref_list]
            placeholders = "\n".join(["<|vision_start|><|image_pad|><|vision_end|>"] * len(ref_list))
            body = f"{placeholders}\n{prompt_item}" if placeholders else str(prompt_item)
            processed = self.processor(
                text=[template.format(body)],
                images=ref_list,
                padding=True,
                truncation=True,
                max_length=max_sequence_length + drop_idx,
                return_tensors="pt",
            )
            input_ids.append(processed["input_ids"][0])
            attention_masks.append(processed["attention_mask"][0])
            if processed.get("pixel_values") is not None:
                pixel_values.append(processed["pixel_values"])
                image_grid_thw.append(processed["image_grid_thw"])

        padded_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=getattr(self.processor.tokenizer, "pad_token_id", 0) or 0,
        ).to(device)
        padded_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        ).to(device)
        model_kwargs = {
            "input_ids": padded_ids,
            "attention_mask": padded_masks,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if pixel_values:
            model_kwargs["pixel_values"] = torch.cat(pixel_values, dim=0).to(device)
            model_kwargs["image_grid_thw"] = torch.cat(image_grid_thw, dim=0).to(device)
        outputs = self.text_encoder(**model_kwargs)
        hidden = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) else outputs.last_hidden_state
        hidden = hidden[:, drop_idx:]
        masks = padded_masks[:, drop_idx:].to(dtype=torch.bool)
        if num_images_per_prompt > 1:
            hidden = hidden.repeat_interleave(num_images_per_prompt, dim=0)
            masks = masks.repeat_interleave(num_images_per_prompt, dim=0)
        return hidden, masks

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        image=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt=None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_size: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        static_shift: Optional[float] = None,
        skip_guidance_layers: Optional[list[int]] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        guidance_rescale: Optional[float] = None,
        use_cfg_zero_star: bool = True,
        return_dict: bool = True,
        **kwargs,
    ):
        del kwargs
        if image is None:
            return super().__call__(
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                negative_prompt=negative_prompt,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                height=height or 1024,
                width=width or 1024,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                static_shift=static_shift,
                skip_guidance_layers=skip_guidance_layers,
                skip_layer_guidance_scale=skip_layer_guidance_scale,
                skip_layer_guidance_stop=skip_layer_guidance_stop,
                skip_layer_guidance_start=skip_layer_guidance_start,
                guidance_rescale=guidance_rescale,
                use_cfg_zero_star=use_cfg_zero_star,
                return_dict=return_dict,
            )

        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype
        refs = image if isinstance(image, (list, tuple)) else [image]
        refs = [_load_image(ref) for ref in refs]
        primary = refs[0]
        if height is None or width is None:
            if max_size is not None:
                if primary.height >= primary.width:
                    height = max_size
                    width = round(primary.width * max_size / primary.height)
                else:
                    width = max_size
                    height = round(primary.height * max_size / primary.width)
            else:
                width, height = primary.size
        height = _make_divisible_by_16(int(height))
        width = _make_divisible_by_16(int(width))

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                prompt,
                images=[refs],
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
        elif num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            if prompt_embeds_mask is not None:
                prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)
        txt, txt_cu, txt_mask, vec = _pack_prompt_embeds(prompt_embeds, prompt_embeds_mask, device)

        if negative_prompt_embeds is None and negative_prompt is not None:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                negative_prompt,
                images=[refs],
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
        if negative_prompt_embeds is not None:
            neg_txt, neg_cu, neg_mask, neg_vec = _pack_prompt_embeds(
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
                device,
            )
        else:
            neg_txt = neg_cu = neg_mask = neg_vec = None

        batch_size = prompt_embeds.shape[0] if prompt_embeds.dim() == 3 else 1
        refs_per_sample = [refs] * batch_size
        latent_h = height // 16
        latent_w = width // 16
        target_tokens = []
        ref_tokens = []
        sample_lens = []
        target_lens = []
        shape_seq = []
        target_indices = []
        offset = 0
        for sample_refs in refs_per_sample:
            noise = torch.randn(
                1,
                self.transformer.config.in_channels,
                latent_h,
                latent_w,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            target = rearrange(noise, "b c h w -> b (h w) c")
            encoded_refs = []
            for ref in sample_refs:
                ref_pixels = _preprocess_ref_image(ref, height, width, device, dtype)
                encoded_refs.append(rearrange(_call_vae_encode(self.vae, ref_pixels), "b c h w -> b (h w) c"))
            refs_packed = torch.cat(encoded_refs, dim=1)
            target_tokens.append(target)
            ref_tokens.append(refs_packed)
            target_len = target.shape[1]
            total_len = target_len + refs_packed.shape[1]
            target_lens.append(target_len)
            sample_lens.append(total_len)
            shape_seq.append((1, latent_h, latent_w))
            shape_seq.extend((1, latent_h, latent_w) for _ in sample_refs)
            target_indices.append(torch.arange(offset, offset + target_len, device=device))
            offset += total_len

        ctx = _build_pack_ctx(
            None,
            _lens_to_cu(sample_lens, device),
            [shape_seq],
            sample_lens,
            txt.to(dtype=dtype),
            txt_cu,
            txt_mask,
            vec.to(dtype=dtype),
            neg_txt.to(dtype=dtype) if neg_txt is not None and guidance_scale > 1.0 else None,
            neg_cu if guidance_scale > 1.0 else None,
            neg_mask if guidance_scale > 1.0 else None,
            neg_vec.to(dtype=dtype) if neg_vec is not None and guidance_scale > 1.0 else None,
            guidance_scale,
            False,
            True,
            device,
        )
        target_idx = torch.cat(target_indices)
        self._set_scheduler_timesteps(num_inference_steps, device, static_shift)
        for step_index, timestep in enumerate(self.scheduler.timesteps):
            parts = []
            for target, ref in zip(target_tokens, ref_tokens, strict=True):
                parts.append(target)
                parts.append(ref)
            packed = torch.cat(parts, dim=1)
            velocity = _mageflow_velocity(
                self.transformer,
                packed,
                ctx,
                self.scheduler.sigmas[step_index].item(),
                step_index=step_index,
                num_inference_steps=num_inference_steps,
                skip_guidance_layers=skip_guidance_layers,
                skip_layer_guidance_scale=skip_layer_guidance_scale,
                skip_layer_guidance_stop=skip_layer_guidance_stop,
                skip_layer_guidance_start=skip_layer_guidance_start,
                guidance_rescale=guidance_rescale,
                use_cfg_zero_star=use_cfg_zero_star,
            )
            pred_target = velocity[:, target_idx]
            stepped = self.scheduler.step(
                pred_target,
                timestep,
                torch.cat(target_tokens, dim=1),
                return_dict=False,
            )[0]
            cursor = 0
            next_targets = []
            for target_len in target_lens:
                next_targets.append(stepped[:, cursor : cursor + target_len])
                cursor += target_len
            target_tokens = next_targets

        images = [_tokens_to_image(self.vae, target, height, width) for target in target_tokens]
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
