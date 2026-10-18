# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenMOSS-Team/MOSS-Image-Base, revision 1ceede80badddd6a7812069619edaa5633587192.
"""MOSS-Image Euler sampling, native layout prompts, and streamed validation callbacks."""

import json
import math
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2, DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import Qwen2Tokenizer, Qwen3_5Model

from simpletuner.helpers.models.flux2.latents import denormalize_latents, normalize_latents, patchify, unpatchify
from simpletuner.helpers.models.guidance import guided_velocity
from simpletuner.helpers.models.moss_image.text import MAX_TEXT_LENGTH, encode_prompts, load_image_processor
from simpletuner.helpers.models.moss_image.transformer import MOSSImageTransformer, _validate_skip_layers
from simpletuner.helpers.training.adapter import _sidecar_destination


class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
    """Euler integration on a flow grid shifted once, including the terminal zero."""

    @register_to_config
    def __init__(self, num_train_timesteps=1000, shift=5.0):
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift)

    def set_timesteps(self, num_inference_steps=50, device=None):
        if num_inference_steps < 1 or self.config.shift <= 0:
            raise ValueError("Steps and shift must be positive.")
        values = torch.linspace(1, 0, num_inference_steps + 1, device=device)[:-1]
        shift = self.config.shift
        sigmas = shift * values / (1 + (shift - 1) * values)
        self._shift = 1.0
        super().set_timesteps(device=device, sigmas=sigmas.tolist())


class MOSSImagePipeline(DiffusionPipeline, FluxLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["text_encoder", "tokenizer", "vae", "scheduler"]
    _lora_loadable_modules = ["transformer"]
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self, transformer, scheduler=None, text_encoder=None, tokenizer=None, vae=None, reference_image_processor=None
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler or MOSSImageScheduler(),
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=16)
        self.reference_image_processor = reference_image_processor
        self.vae_scale_factor = 8
        self._interrupt = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        dtype = kwargs.pop("torch_dtype", kwargs.pop("dtype", torch.bfloat16))
        load_kwargs = {
            key: kwargs.pop(key) for key in ("revision", "cache_dir", "token", "local_files_only") if key in kwargs
        }
        root = Path(pretrained_model_name_or_path)
        index_path = (
            root / "model_index.json"
            if root.is_dir()
            else Path(hf_hub_download(pretrained_model_name_or_path, "model_index.json", **load_kwargs))
        )
        index = json.loads(index_path.read_text())
        components = {}
        loaders = {
            "transformer": (MOSSImageTransformer, "transformer", dtype),
            "text_encoder": (Qwen3_5Model, "text_encoder", dtype),
            "tokenizer": (Qwen2Tokenizer, "tokenizer", None),
            "vae": (AutoencoderKLFlux2, "vae", torch.float32),
            "scheduler": (MOSSImageScheduler, "scheduler", None),
        }
        for name, (loader, subfolder, component_dtype) in loaders.items():
            if name in kwargs:
                components[name] = kwargs.pop(name)
            elif name in cls._optional_components and index.get(name) == [None, None]:
                components[name] = None
            else:
                options = dict(load_kwargs, subfolder=subfolder)
                if component_dtype is not None:
                    options["torch_dtype"] = (
                        component_dtype.get(name, component_dtype.get("default", torch.bfloat16))
                        if isinstance(component_dtype, dict)
                        else component_dtype
                    )
                components[name] = loader.from_pretrained(pretrained_model_name_or_path, **options)
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

    def encode_prompt(self, prompt, max_sequence_length=MAX_TEXT_LENGTH, reference_images=None):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("MOSS-Image text encoding requires a tokenizer and Qwen3.5 text encoder.")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts or not all(isinstance(value, str) for value in prompts):
            raise ValueError("prompt must be a string or a nonempty list of strings.")
        if reference_images is not None and self.reference_image_processor is None:
            self.reference_image_processor = load_image_processor()
        return encode_prompts(
            self.tokenizer,
            self.text_encoder,
            prompts,
            self._execution_device,
            max_sequence_length,
            reference_images=reference_images,
            image_processor=self.reference_image_processor,
        )

    def _reference_images(self, image, batch):
        if image is None:
            return None
        if isinstance(image, Image.Image):
            image = [image]
        if not isinstance(image, (list, tuple)) or not image:
            raise ValueError("image must contain PIL reference images.")
        if isinstance(image[0], Image.Image):
            image = [image] * batch
        if len(image) != batch or any(not images for images in image):
            raise ValueError("Provide one nonempty reference image set per prompt, or a shared list of references.")
        if any(not isinstance(value, Image.Image) for images in image for value in images):
            raise ValueError("MOSS-Image references must be PIL images.")
        return [[value.convert("RGB") for value in images] for images in image]

    def _encode_reference_latents(self, references, height, width, device):
        if self.vae is None:
            raise ValueError("Reference image encoding requires a FLUX.2 VAE.")
        results = []
        for images in references:
            latents = []
            for image in images:
                scale = math.sqrt(height * width / (image.height * image.width))
                ref_height = max(16, round(image.height * scale / 16) * 16)
                ref_width = max(16, round(image.width * scale / 16) * 16)
                pixels = self.image_processor.preprocess(image, height=ref_height, width=ref_width)
                encoded = self.vae.encode(pixels.to(device=device, dtype=self.vae.dtype)).latent_dist.mode()
                latents.append(normalize_latents(patchify(encoded), self.vae)[0])
            results.append(latents)
        return results

    def load_lora_weights(self, source, adapter_name=None, hotswap=False, **kwargs):
        deltatime_type = kwargs.pop("flowmap_deltatime_type", None)
        gate_value = kwargs.pop("flowmap_gate_value", 0.25)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", True)
        state, alphas, metadata = self.lora_state_dict(source, return_alphas=True, return_lora_metadata=True, **kwargs)
        if not any(key.startswith("transformer.") and "lora_" in key for key in state):
            raise ValueError("No MOSS-Image transformer LoRA tensors were found.")
        sidecars = {}
        training_modules = ("crepa_projector.", "urepa_projector.", "self_transcendence_projector.")
        for key in list(state):
            name = key.removeprefix("transformer.")
            if name.startswith(("time_sign_embed.", "delta_timestep_embedder.")):
                sidecars[name] = state.pop(key)
            elif name.startswith(training_modules):
                state.pop(key)
        if any(name.startswith("delta_timestep_embedder.") for name in sidecars):
            if deltatime_type is not None:
                self.transformer.enable_flowmap_time_conditioning(gate_value, deltatime_type)
            elif self.transformer.delta_timestep_embedder is None:
                raise ValueError("Provide flowmap_deltatime_type and flowmap_gate_value from the AnyFlow training config.")
        if "time_sign_embed.weight" in sidecars and self.transformer.time_sign_embed is None:
            self.transformer.time_sign_embed = torch.nn.Embedding(
                2, self.transformer.config.dim, device=self.transformer.device, dtype=self.transformer.dtype
            )
            self.transformer.register_to_config(enable_time_sign_embed=True)
        if metadata is not None:
            metadata = dict(metadata)
            saved_modules = metadata.get("transformer.modules_to_save") or []
            metadata["transformer.modules_to_save"] = [
                name for name in saved_modules if not (name + ".").startswith(training_modules)
            ]
            for name, value in sidecars.items():
                if any(name.startswith(module + ".") for module in saved_modules):
                    state["transformer." + name] = value
        self.load_lora_into_transformer(
            state,
            network_alphas=alphas,
            transformer=self.transformer,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )
        for module_name in {name.split(".")[0] for name in sidecars}:
            getattr(self.transformer, module_name).to(device=self.transformer.device, dtype=self.transformer.dtype)
        model_state = self.transformer.state_dict()
        selected_adapter = adapter_name or self.transformer.active_adapters()[0]
        with torch.no_grad():
            for name, value in sidecars.items():
                parameter = _sidecar_destination(model_state, name, selected_adapter)
                if parameter is None:
                    raise ValueError(f"MOSS-Image adapter parameter is missing from the model: {name}.")
                if parameter.shape != value.shape:
                    raise ValueError(f"MOSS-Image adapter parameter shape mismatch for {name}.")
                parameter.copy_(value.to(parameter))

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        *,
        image=None,
        reference_latents=None,
        cache_reference_kv=True,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.0,
        guidance_rescale=0.0,
        negative_prompt="",
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        max_sequence_length=MAX_TEXT_LENGTH,
        output_type="pil",
        return_dict=True,
        uncond_skip_layers=(22,),
        slg_layers=None,
        slg_scale=0.0,
        slg_start=0.0,
        slg_stop=1.0,
        use_cfg_zero_star=False,
        use_zero_init=True,
        zero_steps=1,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=("latents",),
        attention_kwargs=None,
    ):
        uncond_skip_layers = _validate_skip_layers(uncond_skip_layers, len(self.transformer.layers), "uncond_skip_layers")
        slg_layers = _validate_skip_layers(slg_layers, len(self.transformer.layers), "slg_layers")
        if not math.isfinite(slg_scale) or slg_scale < 0:
            raise ValueError("slg_scale must be finite and nonnegative.")
        if not 0 <= slg_start <= slg_stop <= 1:
            raise ValueError("Require 0 <= slg_start <= slg_stop <= 1.")
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("height and width must be positive multiples of 16.")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be pil, np, pt or latent.")
        if num_images_per_prompt < 1 or not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("Invalid image count or guidance scale.")
        if not 0 <= guidance_rescale <= 1:
            raise ValueError("guidance_rescale must be between 0 and 1.")
        if not 1 <= max_sequence_length <= MAX_TEXT_LENGTH:
            raise ValueError("max_sequence_length must be between 1 and 4096.")
        if any(name not in self._callback_tensor_inputs for name in callback_on_step_end_tensor_inputs):
            raise ValueError("MOSS-Image callbacks expose only latents.")
        if (prompt is None) == (prompt_embeds is None):
            raise ValueError("Provide exactly one of prompt and prompt_embeds.")
        self._guidance_scale = guidance_scale
        self._interrupt = False
        device = self._execution_device
        batch = prompt_embeds.shape[0] if prompt_embeds is not None else (1 if isinstance(prompt, str) else len(prompt))
        references = self._reference_images(image, batch)
        if references is not None and reference_latents is not None:
            raise ValueError("Provide image or reference_latents, rather than both.")
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self.encode_prompt(prompt, max_sequence_length, references)
        batch = prompt_embeds.shape[0]
        if references is not None:
            reference_latents = self._encode_reference_latents(references, height, width, device)
        if reference_latents is not None:
            if len(reference_latents) != batch:
                raise ValueError("Reference latent sets must match the prompt batch size.")
            reference_latents = [sample for sample in reference_latents for _ in range(num_images_per_prompt)]

        def prepare(embeds, mask):
            if embeds.ndim != 3 or embeds.shape[0] != batch:
                raise ValueError("Prompt embeddings must be [B, S, C] with matching batch size.")
            if mask is None:
                mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device=embeds.device)
            return (
                embeds.to(device).repeat_interleave(num_images_per_prompt, 0),
                mask.to(device).repeat_interleave(num_images_per_prompt, 0),
            )

        positive, positive_mask = prepare(prompt_embeds, prompt_attention_mask)
        do_cfg = guidance_scale != 1.0
        if do_cfg:
            if negative_prompt_embeds is None:
                negative_prompt = "" if negative_prompt is None else negative_prompt
                negatives = [negative_prompt] * batch if isinstance(negative_prompt, str) else negative_prompt
                if len(negatives) != batch:
                    raise ValueError("negative_prompt must match prompt batch size.")
                negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                    negatives, max_sequence_length, references
                )
            negative, negative_mask = prepare(negative_prompt_embeds, negative_prompt_attention_mask)
        shape = (batch * num_images_per_prompt, self.transformer.config.latent_channels, height // 16, width // 16)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        elif tuple(latents.shape) != shape:
            raise ValueError(f"latents must have shape {shape}.")
        else:
            latents = latents.to(device=device, dtype=torch.float32)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        num_steps = self._num_timesteps = len(self.scheduler.timesteps)
        reference_caches = {
            name: [{} for _ in range(shape[0])] if cache_reference_kv and reference_latents is not None else None
            for name in ("positive", "negative", "skip")
        }
        for step_index, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if self.interrupt:
                break
            skip_active = slg_start <= step_index / num_steps < slg_stop
            model_kwargs = {"attention_kwargs": attention_kwargs, "reference_latents": reference_latents}
            if self.transformer.flowmap_deltatime_type is not None:
                model_kwargs["r_timestep"] = (
                    self.scheduler.sigmas[step_index + 1].to(device) * self.scheduler.config.num_train_timesteps
                )
            if self.transformer.time_sign_embed is not None:
                model_kwargs["timestep_sign"] = torch.ones(shape[0], device=device)
            conditional = self.transformer(
                latents,
                timestep,
                positive,
                positive_mask,
                _reference_cache=reference_caches["positive"],
                **model_kwargs,
            ).sample.float()
            prediction = conditional
            if do_cfg:
                if use_cfg_zero_star and use_zero_init and step_index <= zero_steps:
                    prediction = torch.zeros_like(conditional)
                else:
                    skip = uncond_skip_layers if uncond_skip_layers and skip_active else None
                    uncond = self.transformer(
                        latents,
                        timestep,
                        negative,
                        negative_mask,
                        skip_layers=skip,
                        _reference_cache=reference_caches.setdefault(
                            ("negative", tuple(sorted(skip or ()))),
                            [{} for _ in range(shape[0])] if cache_reference_kv and reference_latents is not None else None,
                        ),
                        **model_kwargs,
                    ).sample.float()
                    prediction = guided_velocity(conditional, uncond, guidance_scale, use_cfg_zero_star)
                    if guidance_rescale > 0:
                        dims = tuple(range(1, prediction.ndim))
                        rescaled = prediction * (
                            conditional.std(dims, keepdim=True) / prediction.std(dims, keepdim=True).clamp_min(1e-8)
                        )
                        prediction = guidance_rescale * rescaled + (1 - guidance_rescale) * prediction
            if slg_layers and slg_scale > 0 and skip_active:
                skipped = self.transformer(
                    latents,
                    timestep,
                    positive,
                    positive_mask,
                    skip_layers=slg_layers,
                    _reference_cache=reference_caches["skip"],
                    **model_kwargs,
                ).sample.float()
                if not do_cfg:
                    prediction = skipped
                prediction = prediction + slg_scale * (conditional - skipped)
            latents = self.scheduler.step(prediction, timestep, latents, return_dict=False)[0].float()
            if callback_on_step_end is not None:
                callback_kwargs = {name: latents for name in callback_on_step_end_tensor_inputs}
                returned = callback_on_step_end(self, step_index, timestep, callback_kwargs)
                latents = returned.pop("latents", latents)
        if output_type == "latent":
            images = latents
        else:
            if self.vae is None:
                raise ValueError("A FLUX.2 VAE is required for decoded output.")
            decoded = self.vae.decode(
                unpatchify(denormalize_latents(latents.to(self.vae.dtype), self.vae)), return_dict=False
            )[0].clamp(-1, 1)
            images = self.image_processor.postprocess(decoded, output_type=output_type)
        self.maybe_free_model_hooks()
        return ImagePipelineOutput(images=images) if return_dict else (images,)
