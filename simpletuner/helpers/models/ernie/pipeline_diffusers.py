# Copyright 2025 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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

"""
Ernie-Image Pipeline for HuggingFace Diffusers.
"""

import json
from typing import Callable, List, Optional, Union

import torch
from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from simpletuner.helpers.models.ernie.pipeline_output import ErnieImagePipelineOutput
from simpletuner.helpers.models.ernie.transformer import ErnieImageTransformer2DModel


class ErnieImagePipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using ErnieImageTransformer2DModel.

    This pipeline uses:
    - A custom DiT transformer model
    - A Flux2-style VAE for encoding/decoding latents
    - A text encoder (e.g., Qwen) for text conditioning
    - Flow Matching Euler Discrete Scheduler
    """

    model_cpu_offload_seq = "pe->text_encoder->transformer->vae"
    # For SGLang fallback ...
    _optional_components = ["pe", "pe_tokenizer"]
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: ErnieImageTransformer2DModel,
        vae: AutoencoderKLFlux2,
        text_encoder: AutoModel,
        tokenizer: AutoTokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler,
        pe: Optional[AutoModelForCausalLM] = None,
        pe_tokenizer: Optional[AutoTokenizer] = None,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            pe=pe,
            pe_tokenizer=pe_tokenizer,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels)) if getattr(self, "vae", None) else 16

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @torch.no_grad()
    def _enhance_prompt_with_pe(
        self,
        prompt: str,
        device: torch.device,
        width: int = 1024,
        height: int = 1024,
        system_prompt: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> str:
        """Use PE model to rewrite/enhance a short prompt via chat_template."""
        # Build user message as JSON carrying prompt text and target resolution
        user_content = json.dumps(
            {"prompt": prompt, "width": width, "height": height},
            ensure_ascii=False,
        )
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        # apply_chat_template picks up the chat_template.jinja loaded with pe_tokenizer
        input_text = self.pe_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # "Output:" is already in the user block
        )
        inputs = self.pe_tokenizer(input_text, return_tensors="pt").to(device)
        output_ids = self.pe.generate(
            **inputs,
            max_new_tokens=self.pe_tokenizer.model_max_length,
            do_sample=temperature != 1.0 or top_p != 1.0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.pe_tokenizer.pad_token_id,
            eos_token_id=self.pe_tokenizer.eos_token_id,
        )
        # Decode only newly generated tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.pe_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
    ) -> List[torch.Tensor]:
        """Encode text prompts to embeddings."""
        if isinstance(prompt, str):
            prompt = [prompt]

        text_hiddens = []

        for p in prompt:
            ids = self.tokenizer(
                p,
                add_special_tokens=True,
                truncation=True,
                padding=False,
            )["input_ids"]

            if len(ids) == 0:
                if self.tokenizer.bos_token_id is not None:
                    ids = [self.tokenizer.bos_token_id]
                else:
                    ids = [0]

            input_ids = torch.tensor([ids], device=device)
            with torch.no_grad():
                outputs = self.text_encoder(
                    input_ids=input_ids,
                    output_hidden_states=True,
                )
                # Use second to last hidden state (matches training)
                hidden = outputs.hidden_states[-2][0]  # [T, H]

            # Repeat for num_images_per_prompt
            for _ in range(num_images_per_prompt):
                text_hiddens.append(hidden)

        return text_hiddens

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """2x2 patchify: [B, 32, H, W] -> [B, 128, H/2, W/2]"""
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Reverse patchify: [B, 128, H/2, W/2] -> [B, 32, H, W]"""
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)

    @staticmethod
    def _pad_text(text_hiddens: List[torch.Tensor], device: torch.device, dtype: torch.dtype, text_in_dim: int):
        B = len(text_hiddens)
        if B == 0:
            return torch.zeros((0, 0, text_in_dim), device=device, dtype=dtype), torch.zeros(
                (0,), device=device, dtype=torch.long
            )
        normalized = [
            th.squeeze(1).to(device).to(dtype) if th.dim() == 3 else th.to(device).to(dtype) for th in text_hiddens
        ]
        lens = torch.tensor([t.shape[0] for t in normalized], device=device, dtype=torch.long)
        Tmax = int(lens.max().item())
        text_bth = torch.zeros((B, Tmax, text_in_dim), device=device, dtype=dtype)
        for i, t in enumerate(normalized):
            text_bth[i, : t.shape[0], :] = t
        return text_bth, lens

    @staticmethod
    def _repeat_text_hiddens(text_hiddens: List[torch.Tensor], num_images_per_prompt: int) -> List[torch.Tensor]:
        if num_images_per_prompt == 1:
            return list(text_hiddens)
        repeated = []
        for text_hidden in text_hiddens:
            repeated.extend([text_hidden] * num_images_per_prompt)
        return repeated

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: list[torch.FloatTensor] | None = None,
        negative_prompt_embeds: list[torch.FloatTensor] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        use_pe: bool = True,  # 默认使用PE进行改写
        skip_guidance_layers: Optional[List[int]] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        use_cfg_zero_star: bool = True,
        use_zero_init: bool = True,
        zero_steps: int = 0,
        no_cfg_until_timestep: int = 0,
    ):
        """
        Generate images from text prompts.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s) for CFG. Default is "".
            height: Image height in pixels (must be divisible by 16). Default: 1024.
            width: Image width in pixels (must be divisible by 16). Default: 1024.
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale (1.0 = no guidance). Default: 4.0.
            num_images_per_prompt: Number of images per prompt
            generator: Random generator for reproducibility
            latents: Pre-generated latents (optional)
            prompt_embeds: Pre-computed text embeddings for positive prompts (optional).
                If provided, `encode_prompt` is skipped for positive prompts.
            negative_prompt_embeds: Pre-computed text embeddings for negative prompts (optional).
                If provided, `encode_prompt` is skipped for negative prompts.
            output_type: "pil" or "latent"
            return_dict: Whether to return a dataclass
            callback_on_step_end: Optional callback invoked at the end of each denoising step.
                Called as `callback_on_step_end(pipeline, step, timestep, callback_kwargs)` where `callback_kwargs`
                contains the tensors listed in `callback_on_step_end_tensor_inputs`. The callback may return a dict to
                override those tensors for subsequent steps.
            callback_on_step_end_tensor_inputs: List of tensor names passed into the callback kwargs.
                Must be a subset of `_callback_tensor_inputs` (default: `["latents"]`).
            use_pe: Whether to use the PE model to enhance prompts before generation.
            skip_guidance_layers: Layer indices to skip when computing optional skipped-layer guidance.
            skip_layer_guidance_scale: Scale for skipped-layer guidance correction.
            skip_layer_guidance_stop: Fraction of inference after which skipped-layer guidance stops.
            skip_layer_guidance_start: Fraction of inference before which skipped-layer guidance is disabled.
            use_cfg_zero_star: Whether to use CFG-Zero* scaling when applying classifier-free guidance.
            use_zero_init: Whether CFG-Zero* should zero guided predictions for the first `zero_steps` steps.
            zero_steps: Number of initial denoising steps to zero when `use_zero_init` is enabled.
            no_cfg_until_timestep: Zero-indexed denoising step before which CFG is skipped.

        Returns:
            :class:`ErnieImagePipelineOutput` with `images` and `revised_prompts`.
        """
        device = self._execution_device
        dtype = self.transformer.dtype

        self._guidance_scale = guidance_scale

        # Validate prompt / prompt_embeds
        if prompt is None and prompt_embeds is None:
            raise ValueError("Must provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot provide both `prompt` and `prompt_embeds` at the same time.")
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = []
        elif not all(k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # Validate dimensions
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(f"Height and width must be divisible by {self.vae_scale_factor}")

        # Handle prompts
        if prompt is not None:
            if isinstance(prompt, str):
                prompt = [prompt]

        # [Phase 1] PE: enhance prompts
        revised_prompts: Optional[List[str]] = None
        if prompt is not None and use_pe and self.pe is not None and self.pe_tokenizer is not None:
            prompt = [self._enhance_prompt_with_pe(p, device, width=width, height=height) for p in prompt]
            revised_prompts = list(prompt)

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)
        total_batch_size = batch_size * num_images_per_prompt

        # Handle negative prompt
        if negative_prompt is None:
            negative_prompt = ""
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if len(negative_prompt) != batch_size:
            raise ValueError(f"negative_prompt must have same length as prompt ({batch_size})")

        # [Phase 2] Text encoding
        if prompt_embeds is not None:
            text_hiddens = self._repeat_text_hiddens(prompt_embeds, num_images_per_prompt)
        else:
            text_hiddens = self.encode_prompt(prompt, device, num_images_per_prompt)

        # CFG with negative prompt
        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is not None:
                if len(negative_prompt_embeds) != batch_size:
                    raise ValueError(f"negative_prompt_embeds must have same length as prompt ({batch_size})")
                uncond_text_hiddens = self._repeat_text_hiddens(negative_prompt_embeds, num_images_per_prompt)
            else:
                uncond_text_hiddens = self.encode_prompt(negative_prompt, device, num_images_per_prompt)

        # Latent dimensions
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels  # After patchify

        # Initialize latents
        if latents is None:
            latents = randn_tensor(
                (total_batch_size, latent_channels, latent_h, latent_w),
                generator=generator,
                device=device,
                dtype=dtype,
            )

        # Setup scheduler
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        self.scheduler.set_timesteps(sigmas=sigmas[:-1], device=device)

        # Denoising loop
        if self.do_classifier_free_guidance:
            cfg_text_hiddens = list(uncond_text_hiddens) + list(text_hiddens)
        else:
            cfg_text_hiddens = text_hiddens
        text_bth, text_lens = self._pad_text(
            text_hiddens=cfg_text_hiddens, device=device, dtype=dtype, text_in_dim=self.transformer.config.text_in_dim
        )
        positive_text_bth, positive_text_lens = self._pad_text(
            text_hiddens=text_hiddens, device=device, dtype=dtype, text_in_dim=self.transformer.config.text_in_dim
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    t_batch = torch.full((total_batch_size * 2,), t.item(), device=device, dtype=torch.float32)
                else:
                    latent_model_input = latents
                    t_batch = torch.full((total_batch_size,), t.item(), device=device, dtype=torch.float32)

                # Model prediction
                pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_batch,
                    text_bth=text_bth,
                    text_lens=text_lens,
                    return_dict=False,
                )[0]

                # Apply CFG
                if self.do_classifier_free_guidance:
                    pred_uncond, pred_cond = pred.chunk(2, dim=0)
                    if i >= no_cfg_until_timestep:
                        if use_cfg_zero_star:
                            pos_flat = pred_cond.float().reshape(pred_cond.shape[0], -1)
                            neg_flat = pred_uncond.float().reshape(pred_uncond.shape[0], -1)
                            alpha = (
                                pos_flat.norm(dim=1) / neg_flat.norm(dim=1).clamp_min(torch.finfo(neg_flat.dtype).eps)
                            ).view(pred_cond.shape[0], *([1] * (pred_cond.ndim - 1)))
                            pred = pred_uncond * alpha.to(dtype=pred_uncond.dtype) + guidance_scale * (
                                pred_cond - pred_uncond * alpha.to(dtype=pred_uncond.dtype)
                            )
                            if i <= zero_steps and use_zero_init:
                                pred = torch.zeros_like(pred)
                        else:
                            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                    else:
                        pred = pred_cond

                    should_skip_layers = (
                        skip_guidance_layers is not None
                        and len(skip_guidance_layers) > 0
                        and i >= no_cfg_until_timestep
                        and i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                    )
                    if should_skip_layers:
                        skip_pred = self.transformer(
                            hidden_states=latents,
                            timestep=torch.full((total_batch_size,), t.item(), device=device, dtype=torch.float32),
                            text_bth=positive_text_bth,
                            text_lens=positive_text_lens,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        pred = pred + (pred_cond - skip_pred) * skip_layer_guidance_scale

                # Scheduler step
                latents = self.scheduler.step(pred, t, latents).prev_sample

                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        if output_type == "latent":
            return latents

        # Decode latents to images
        # Unnormalize latents using VAE's BN stats
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device)
        bn_eps = getattr(self.vae.config, "batch_norm_eps", 1e-5)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + bn_eps).to(device)
        latents = latents * bn_std + bn_mean

        # Unpatchify
        latents = self._unpatchify_latents(latents)

        # Decode
        images = self.vae.decode(latents, return_dict=False)[0]

        # Post-process
        images = (images.clamp(-1, 1) + 1) / 2
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            images = [Image.fromarray((img * 255).astype("uint8")) for img in images]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)

        return ErnieImagePipelineOutput(images=images, revised_prompts=revised_prompts)
