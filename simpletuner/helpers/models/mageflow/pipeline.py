from __future__ import annotations

import os
from typing import Optional

import torch
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from einops import rearrange
from huggingface_hub import snapshot_download

from simpletuner.helpers.models.mageflow.vendor.pipeline import (
    PROMPT_TEMPLATE,
    _build_pack_ctx,
    _lens_to_cu,
    _make_divisible_by_16,
    _velocity,
)


def _resolve_repo_dir(repo_id_or_path: str, *, revision: Optional[str] = None, local_files_only: bool = False) -> str:
    if os.path.isdir(repo_id_or_path):
        return os.path.abspath(repo_id_or_path)
    return snapshot_download(repo_id=repo_id_or_path, revision=revision, local_files_only=local_files_only)


def _call_vae_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    if hasattr(vae, "decode_to_tensor"):
        return vae.decode_to_tensor(latents)
    decoded = vae.decode(latents)
    if hasattr(decoded, "sample"):
        return decoded.sample
    return decoded


def _unpack_tokens(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(
        tokens,
        "b (h w) c -> b c h w",
        h=height // 16,
        w=width // 16,
    )


def _tokens_to_image(vae, tokens: torch.Tensor, height: int, width: int):
    latents = _unpack_tokens(tokens.float(), height, width)
    with torch.no_grad():
        decoded = _call_vae_decode(vae, latents)
    decoded = rearrange(decoded.clamp(-1, 1), "b c h w -> b h w c")
    decoded = (127.5 * (decoded + 1.0)).cpu().byte().numpy()
    from PIL import Image

    return Image.fromarray(decoded[0])


def _pack_prompt_embeds(prompt_embeds: torch.Tensor, prompt_embeds_mask: Optional[torch.Tensor], device):
    if prompt_embeds.dim() == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    if prompt_embeds_mask is None:
        prompt_embeds_mask = torch.ones(
            prompt_embeds.shape[:2],
            dtype=torch.bool,
            device=prompt_embeds.device,
        )
    if prompt_embeds_mask.dim() == 1:
        prompt_embeds_mask = prompt_embeds_mask.unsqueeze(0)

    chunks = []
    lengths = []
    for embeds, mask in zip(prompt_embeds, prompt_embeds_mask, strict=True):
        selected = embeds[mask.to(dtype=torch.bool)]
        chunks.append(selected)
        lengths.append(int(selected.shape[0]))
    txt = torch.cat(chunks, dim=0).unsqueeze(0).to(device)
    txt_cu = _lens_to_cu(lengths, device)
    vec = torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0).to(device)
    mask = torch.ones(1, txt.shape[1], dtype=torch.bool, device=device)
    return txt, txt_cu, mask, vec


class MageFlowPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["text_encoder", "tokenizer", "processor"]

    def __init__(self, transformer, vae, scheduler=None, text_encoder=None, tokenizer=None, processor=None):
        super().__init__()
        if scheduler is None:
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0, use_dynamic_shifting=False)
        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        revision = kwargs.pop("revision", None)
        local_files_only = bool(kwargs.pop("local_files_only", False))
        transformer = kwargs.pop("transformer")
        vae = kwargs.pop("vae")
        text_encoder = kwargs.pop("text_encoder", None)
        tokenizer = kwargs.pop("tokenizer", None)
        processor = kwargs.pop("processor", None)
        repo_dir = _resolve_repo_dir(
            pretrained_model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(repo_dir, "scheduler"))
        return cls(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
        )

    def _template_prompt(self, prompt, prompt_template: str = "mage-flow"):
        info = PROMPT_TEMPLATE[prompt_template]
        return info["template"].format(prompt), int(info.get("start_idx", 0))

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt: int = 1,
        prompt_template: str = "mage-flow",
        max_sequence_length: int = 2048,
        **kwargs,
    ):
        del kwargs
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("MageFlowPipeline.encode_prompt requires a loaded text_encoder and tokenizer.")
        device = device or self._execution_device
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        templated = []
        drop_idx = None
        for item in prompts:
            text, current_drop_idx = self._template_prompt(item, prompt_template)
            templated.append(text)
            drop_idx = current_drop_idx

        inputs = self.tokenizer(
            templated,
            padding=True,
            truncation=True,
            max_length=max_sequence_length + int(drop_idx or 0),
            return_tensors="pt",
        ).to(device)
        outputs = self.text_encoder(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) else outputs.last_hidden_state
        mask = inputs["attention_mask"].to(dtype=torch.bool)
        if drop_idx:
            hidden = hidden[:, drop_idx:]
            mask = mask[:, drop_idx:]
        if num_images_per_prompt > 1:
            hidden = hidden.repeat_interleave(num_images_per_prompt, dim=0)
            mask = mask.repeat_interleave(num_images_per_prompt, dim=0)
        return hidden, mask

    def _prepare_prompt_conditioning(
        self,
        prompt,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        num_images_per_prompt: int,
        device,
    ):
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                prompt,
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
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
        if negative_prompt_embeds is None:
            return txt, txt_cu, txt_mask, vec, None, None, None, None

        if num_images_per_prompt > 1 and negative_prompt_embeds.shape[0] != prompt_embeds.shape[0]:
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            if negative_prompt_embeds_mask is not None:
                negative_prompt_embeds_mask = negative_prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)
        neg_txt, neg_cu, neg_mask, neg_vec = _pack_prompt_embeds(negative_prompt_embeds, negative_prompt_embeds_mask, device)
        return txt, txt_cu, txt_mask, vec, neg_txt, neg_cu, neg_mask, neg_vec

    def _set_scheduler_timesteps(self, num_inference_steps: int, device, static_shift: Optional[float] = None):
        if static_shift is not None and hasattr(self.scheduler, "set_shift"):
            self.scheduler.set_shift(static_shift)
        sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps).tolist()
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt=None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        static_shift: Optional[float] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        del kwargs
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype
        height = _make_divisible_by_16(int(height))
        width = _make_divisible_by_16(int(width))

        (
            txt,
            txt_cu,
            txt_mask,
            vec,
            neg_txt,
            neg_cu,
            neg_mask,
            neg_vec,
        ) = self._prepare_prompt_conditioning(
            prompt,
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            num_images_per_prompt,
            device,
        )

        batch_size = len(txt_cu) - 1
        latent_h = height // 16
        latent_w = width // 16
        latents = torch.randn(
            batch_size,
            self.transformer.config.in_channels,
            latent_h,
            latent_w,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        tokens = rearrange(latents, "b c h w -> 1 (b h w) c")
        sample_lens = [latent_h * latent_w] * batch_size
        img_cu = _lens_to_cu(sample_lens, device)
        img_shapes = [[(1, latent_h, latent_w)] * batch_size]
        ctx = _build_pack_ctx(
            None,
            img_cu,
            img_shapes,
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
        self._set_scheduler_timesteps(num_inference_steps, device, static_shift)
        for step_index, timestep in enumerate(self.scheduler.timesteps):
            velocity = _velocity(self.transformer, tokens, ctx, self.scheduler.sigmas[step_index].item())
            tokens = self.scheduler.step(velocity, timestep, tokens, return_dict=False)[0]

        images = []
        offset = 0
        for _ in range(batch_size):
            length = latent_h * latent_w
            images.append(_tokens_to_image(self.vae, tokens[:, offset : offset + length], height, width))
            offset += length
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
