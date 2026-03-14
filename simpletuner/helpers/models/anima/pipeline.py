# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/pipelines/anima/pipeline_anima.py
# Adapted for SimpleTuner local imports.

"""Anima pipeline implementation with Diffusers-style loading conventions."""

import math
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLQwenImage, DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image
from transformers import PreTrainedModel

from .constants import FORGE_BETA_ALPHA, FORGE_BETA_BETA
from .generator_utils import _normalize_generator, _resolve_noise_runtime
from .image_processing import (
    align_tensor_batch_size,
    decode_latents,
    encode_image_to_latents,
    latent_hw,
    prepare_init_image_tensor,
    prepare_inpaint_mask_tensor,
)
from .loading import (
    _disable_vae_method,
    _enable_vae_method,
    build_anima_pipeline,
    coerce_anima_scheduler,
    load_prompt_tokenizer,
    loader_options_from_kwargs,
    normalize_loaded_component_buffers,
    resolve_patch_size,
    resolve_prompt_tokenizer_sources_for_local_dir,
    resolve_vae_scale_factor,
    runtime_options_from_kwargs,
    save_prompt_tokenizers_to_local_dir,
    scheduler_from_kwargs,
)
from .lora_pipeline import AnimaLoraLoaderMixin
from .options import AnimaComponents, AnimaLoaderOptions
from .pipeline_output import AnimaPipelineOutput
from .prompt_utils import _resolve_prompt_batches
from .sampling import GeneratorInput, randn_tensor, run_const_sigma_samplers, sample_flowmatch_euler
from .scheduler import AnimaFlowMatchEulerDiscreteScheduler, AnimaSamplingConfig
from .sigma_schedules import build_sampling_sigmas
from .strength_utils import _trim_flowmatch_timesteps_by_strength, _trim_sigmas_by_strength
from .text_encoding import AnimaPromptTokenizer, build_condition, prepare_condition_inputs
from .transformer import AnimaTransformerModel
from .validation import (
    _ANIMA_COMPONENT_OVERRIDE_KEYS,
    _DIFFUSERS_COMPAT_IGNORED_FROM_SINGLE_FILE_KEYS,
    ImageBatchInput,
    PromptInput,
    _looks_like_single_file_source,
    _partition_single_file_from_pretrained_kwargs,
    _pop_ignored_kwargs,
    _raise_if_removed_from_pretrained_runtime_feature_kwargs,
    _validate_callback_tensor_input_names,
    _validate_image_like_input,
    _validate_sampling_modes,
    _warn_ignored_sampling_arguments,
)


@contextmanager
def _module_execution_context(
    module: torch.nn.Module,
    *,
    execution_device: str,
    execution_dtype: torch.dtype,
    enable_offload: bool,
) -> Iterator[None]:
    if enable_offload and execution_device != "cpu":
        module.to(device=execution_device, dtype=execution_dtype)
        try:
            yield
        finally:
            module.to(device="cpu")
            if execution_device == "cuda":
                torch.cuda.empty_cache()
        return

    yield


# ---------------------------------------------------------------------------
# Internal image generation routine
# ---------------------------------------------------------------------------


def _prepare_prompt_embedding_inputs(
    pipe: "AnimaPipeline",
    *,
    prompt: list[str],
    negative_prompt: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with _module_execution_context(
        pipe.text_encoder,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.text_encoder_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        if pipe.prompt_tokenizer is None:
            raise RuntimeError(
                "AnimaPipeline requires a prompt_tokenizer. "
                "Load the pipeline via from_pretrained or from_single_file to ensure "
                "the tokenizer is initialised automatically."
            )
        pos_hidden, pos_t5_ids, pos_t5_weights = prepare_condition_inputs(
            pipe.prompt_tokenizer,
            pipe.text_encoder,
            prompt,
            execution_device=pipe.execution_device,
            model_dtype=pipe.model_dtype,
        )
        neg_hidden, neg_t5_ids, neg_t5_weights = prepare_condition_inputs(
            pipe.prompt_tokenizer,
            pipe.text_encoder,
            negative_prompt,
            execution_device=pipe.execution_device,
            model_dtype=pipe.model_dtype,
        )

    return (
        pos_hidden,
        pos_t5_ids,
        pos_t5_weights,
        neg_hidden,
        neg_t5_ids,
        neg_t5_weights,
    )


def _build_cfg_conditions_from_embeddings(
    pipe: "AnimaPipeline",
    *,
    pos_hidden: torch.Tensor,
    pos_t5_ids: torch.Tensor,
    pos_t5_weights: torch.Tensor,
    neg_hidden: torch.Tensor,
    neg_t5_ids: torch.Tensor,
    neg_t5_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_cond = build_condition(
        pipe.transformer,
        qwen_hidden=pos_hidden,
        t5_ids=pos_t5_ids,
        t5_weights=pos_t5_weights,
    )
    neg_cond = build_condition(
        pipe.transformer,
        qwen_hidden=neg_hidden,
        t5_ids=neg_t5_ids,
        t5_weights=neg_t5_weights,
    )
    return pos_cond, neg_cond


def _prepare_init_image_latents_and_inpaint_mask(
    pipe: "AnimaPipeline",
    *,
    image: ImageBatchInput | None,
    mask_image: ImageBatchInput | None,
    width: int,
    height: int,
    latent_h: int,
    latent_w: int,
    batch_size: int,
    init_generator: torch.Generator | list[torch.Generator] | None,
    sample_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if image is None:
        return None, None

    init_image_tensor = prepare_init_image_tensor(
        image,
        width=width,
        height=height,
    )
    init_image_tensor = align_tensor_batch_size(
        init_image_tensor,
        target_batch_size=batch_size,
        input_name="image",
    )
    with _module_execution_context(
        pipe.vae,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.model_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        init_image_latents = encode_image_to_latents(
            pipe.vae,
            image_tensor=init_image_tensor,
            execution_device=pipe.execution_device,
            model_dtype=pipe.model_dtype,
            generator=init_generator,
            sample_dtype=sample_dtype,
        )
    init_image_latents = init_image_latents.to(device=pipe.execution_device, dtype=sample_dtype)

    if tuple(init_image_latents.shape[-2:]) != (latent_h, latent_w):
        raise RuntimeError(
            "Encoded image latent shape does not match target resolution. "
            f"Expected {(latent_h, latent_w)}, got {tuple(init_image_latents.shape[-2:])}."
        )

    if mask_image is None:
        return init_image_latents, None

    mask_tensor = prepare_inpaint_mask_tensor(
        mask_image,
        width=width,
        height=height,
    )
    mask_latents = F.interpolate(
        mask_tensor,
        size=(latent_h, latent_w),
        mode="nearest",
    )
    inpaint_mask = mask_latents.to(device=pipe.execution_device, dtype=sample_dtype).unsqueeze(2)
    inpaint_mask = inpaint_mask.repeat(1, init_image_latents.shape[1], 1, 1, 1)
    inpaint_mask = align_tensor_batch_size(
        inpaint_mask,
        target_batch_size=batch_size,
        input_name="mask_image",
    )
    return init_image_latents, inpaint_mask


def _generate_image(
    pipe: "AnimaPipeline",
    *,
    prompt: PromptInput,
    negative_prompt: PromptInput | None = None,
    prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    image: ImageBatchInput | None = None,
    mask_image: ImageBatchInput | None = None,
    strength: float = 1.0,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 32,
    num_images_per_prompt: int = 1,
    guidance_scale: float = 4.0,
    generator: GeneratorInput | None = None,
    sampler: str = "euler_a_rf",
    sigma_schedule: str = "beta",
    beta_alpha: float = FORGE_BETA_ALPHA,
    beta_beta: float = FORGE_BETA_BETA,
    eta: float = 1.0,
    s_noise: float = 1.0,
    cfg_batch_mode: str = "split",
    output_type: str = "pil",
    callback_on_step_end: Callable[..., dict[str, Any] | None] | None = None,
    callback_on_step_end_tensor_inputs: list[str] | None = None,
) -> list[Image.Image] | torch.Tensor:
    """Internal end-to-end generation routine used by ``AnimaPipeline.__call__``."""
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    if prompt_embeds is not None:
        batch_size = prompt_embeds.shape[0]
        pos_hidden = pos_t5_ids = pos_t5_weights = None
        neg_hidden = neg_t5_ids = neg_t5_weights = None
    else:
        prompts, negative_prompts = _resolve_prompt_batches(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
        )
        batch_size = len(prompts)
        pos_hidden, pos_t5_ids, pos_t5_weights, neg_hidden, neg_t5_ids, neg_t5_weights = _prepare_prompt_embedding_inputs(
            pipe,
            prompt=prompts,
            negative_prompt=negative_prompts,
        )

    height, width, latent_h, latent_w = latent_hw(
        height=height,
        width=width,
        vae_scale_factor=pipe.vae_scale_factor,
        patch_size=pipe.patch_size,
    )
    sample_dtype = torch.float32

    resolved_callback_tensor_inputs = callback_on_step_end_tensor_inputs or ["latents"]
    init_generator, step_generator, noise_device, noise_dtype = _resolve_noise_runtime(
        execution_device=pipe.execution_device,
        generator=generator,
        batch_size=batch_size,
    )

    init_image_latents: torch.Tensor | None = None
    inpaint_mask: torch.Tensor | None = None
    init_noise: torch.Tensor | None = None

    init_image_latents, inpaint_mask = _prepare_init_image_latents_and_inpaint_mask(
        pipe,
        image=image,
        mask_image=mask_image,
        width=width,
        height=height,
        latent_h=latent_h,
        latent_w=latent_w,
        batch_size=batch_size,
        init_generator=init_generator,
        sample_dtype=sample_dtype,
    )

    flowmatch_timesteps: torch.Tensor | None = None
    sigmas: torch.Tensor | None = None
    input_is_noisy_latents = False

    if sampler == "flowmatch_euler":
        flowmatch_timesteps = _trim_flowmatch_timesteps_by_strength(
            pipe,
            num_inference_steps=num_inference_steps,
            strength=strength if init_image_latents is not None else 1.0,
        )
        if init_image_latents is None:
            latents = randn_tensor(
                (batch_size, 16, 1, latent_h, latent_w),
                device=noise_device,
                dtype=noise_dtype,
                generator=init_generator,
            )
            latents = latents.to(device=pipe.execution_device, dtype=sample_dtype)
        else:
            init_image_latents = align_tensor_batch_size(
                init_image_latents,
                target_batch_size=batch_size,
                input_name="image",
            )
            init_noise = randn_tensor(
                tuple(init_image_latents.shape),
                device=noise_device,
                dtype=noise_dtype,
                generator=init_generator,
            ).to(device=pipe.execution_device, dtype=sample_dtype)
            start_timestep = (
                flowmatch_timesteps[:1]
                .expand(init_image_latents.shape[0])
                .to(
                    device=pipe.execution_device,
                    dtype=torch.float32,
                )
            )
            latents = pipe.scheduler.scale_noise(init_image_latents, start_timestep, init_noise)
            latents = latents.to(device=pipe.execution_device, dtype=sample_dtype)
    else:
        sigmas = build_sampling_sigmas(
            pipe.scheduler,
            num_inference_steps=num_inference_steps,
            sigma_schedule=sigma_schedule,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            device=pipe.execution_device,
        )
        if init_image_latents is None:
            latents = randn_tensor(
                (batch_size, 16, 1, latent_h, latent_w),
                device=noise_device,
                dtype=noise_dtype,
                generator=init_generator,
            )
            latents = latents.to(device=pipe.execution_device, dtype=sample_dtype)
        else:
            init_image_latents = align_tensor_batch_size(
                init_image_latents,
                target_batch_size=batch_size,
                input_name="image",
            )
            sigmas = _trim_sigmas_by_strength(
                sigmas=sigmas,
                strength=strength,
            )
            init_noise = randn_tensor(
                tuple(init_image_latents.shape),
                device=noise_device,
                dtype=noise_dtype,
                generator=init_generator,
            ).to(device=pipe.execution_device, dtype=sample_dtype)
            sigma_start = sigmas[0].to(init_image_latents.dtype)
            latents = sigma_start * init_noise + (1.0 - sigma_start) * init_image_latents
            input_is_noisy_latents = True

    with _module_execution_context(
        pipe.transformer,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.model_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        if prompt_embeds is not None:
            pos_cond = prompt_embeds.to(device=pipe.execution_device, dtype=pipe.model_dtype)
            neg_cond = negative_prompt_embeds.to(  # type: ignore[union-attr]
                device=pipe.execution_device, dtype=pipe.model_dtype
            )
        else:
            pos_cond, neg_cond = _build_cfg_conditions_from_embeddings(
                pipe,
                pos_hidden=pos_hidden,
                pos_t5_ids=pos_t5_ids,
                pos_t5_weights=pos_t5_weights,
                neg_hidden=neg_hidden,
                neg_t5_ids=neg_t5_ids,
                neg_t5_weights=neg_t5_weights,
            )

        if sampler == "flowmatch_euler":
            if flowmatch_timesteps is None:
                raise RuntimeError("Internal error: flowmatch timesteps were not initialized.")
            latents = sample_flowmatch_euler(
                pipe.transformer,
                pipe.scheduler,
                pipe,
                latents,
                timesteps=flowmatch_timesteps,
                sigma_schedule=sigma_schedule,
                pos_cond=pos_cond,
                neg_cond=neg_cond,
                guidance_scale=guidance_scale,
                cfg_batch_mode=cfg_batch_mode,
                model_dtype=pipe.model_dtype,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=resolved_callback_tensor_inputs,
                inpaint_mask=inpaint_mask,
                init_image_latents=init_image_latents,
                init_noise=init_noise,
            )
        else:
            if sigmas is None:
                raise RuntimeError("Internal error: sigma schedule was not initialized.")
            latents = run_const_sigma_samplers(
                pipe.transformer,
                pipe,
                latents,
                sigmas=sigmas,
                sampler=sampler,
                pos_cond=pos_cond,
                neg_cond=neg_cond,
                guidance_scale=guidance_scale,
                eta=eta,
                s_noise=s_noise,
                generator=step_generator,
                cfg_batch_mode=cfg_batch_mode,
                model_dtype=pipe.model_dtype,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=resolved_callback_tensor_inputs,
                input_is_noisy_latents=input_is_noisy_latents,
                inpaint_mask=inpaint_mask,
                init_image_latents=init_image_latents,
                init_noise=init_noise,
            )

    if output_type == "latent":
        return latents

    with _module_execution_context(
        pipe.vae,
        execution_device=pipe.execution_device,
        execution_dtype=pipe.model_dtype,
        enable_offload=pipe.use_module_cpu_offload,
    ):
        return decode_latents(pipe.vae, latents, runtime_dtype=pipe.model_dtype)


# ---------------------------------------------------------------------------
# AnimaPipeline
# ---------------------------------------------------------------------------


class AnimaPipeline(DiffusionPipeline, AnimaLoraLoaderMixin):
    """Diffusers pipeline wrapper for Anima text-to-image, img2img, and inpaint.

    Anima defaults to ``AnimaFlowMatchEulerDiscreteScheduler``. You can override
    the scheduler component by passing ``scheduler=...`` to ``from_pretrained``.
    """

    transformer: AnimaTransformerModel
    vae: AutoencoderKLQwenImage
    scheduler: AnimaFlowMatchEulerDiscreteScheduler
    text_encoder: PreTrainedModel
    prompt_tokenizer: AnimaPromptTokenizer | None
    execution_device: str
    model_dtype: torch.dtype
    text_encoder_dtype: torch.dtype
    use_module_cpu_offload: bool
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    # prompt_tokenizer is intentionally NOT registered via register_modules because
    # AnimaPromptTokenizer is a custom class without Diffusers save_pretrained/from_pretrained
    # support. It is stored as a plain instance attribute and its constituent tokenizers are
    # saved/loaded via the overridden save_pretrained / from_pretrained methods.
    _optional_components: list[str] = []
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        *,
        transformer: AnimaTransformerModel,
        vae: AutoencoderKLQwenImage,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder: PreTrainedModel,
        prompt_tokenizer: AnimaPromptTokenizer | None = None,
        execution_device: str = "auto",
        model_dtype: torch.dtype = torch.float32,
        text_encoder_dtype: torch.dtype = torch.float32,
        use_module_cpu_offload: bool = False,
    ):
        super().__init__()
        resolved_scheduler = coerce_anima_scheduler(scheduler)
        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=resolved_scheduler,
            text_encoder=text_encoder,
        )

        # Diffusers passes [None, None] for model_index.json entries with null library/class.
        # Normalize to None so downstream checks work correctly.
        if isinstance(prompt_tokenizer, (list, tuple)):
            prompt_tokenizer = None
        self.prompt_tokenizer = prompt_tokenizer
        self.execution_device = execution_device
        self.model_dtype = model_dtype
        self.text_encoder_dtype = text_encoder_dtype
        self.use_module_cpu_offload = use_module_cpu_offload
        self.vae_scale_factor = resolve_vae_scale_factor(vae=self.vae)
        self.patch_size = resolve_patch_size(transformer=self.transformer)

    @property
    def execution_device(self) -> str:
        override = getattr(self, "_anima_execution_device", None)
        if isinstance(override, str) and override != "auto":
            return override
        # CPU-offload path: Diffusers sets _execution_device to the inference GPU.
        device = getattr(self, "_execution_device", None)
        if device is not None:
            return device.type
        # "auto" or unset: detect from the transformer's current device so that
        # pipe.to("cuda") is reflected without an explicit execution_device assignment.
        transformer = getattr(self, "transformer", None)
        if transformer is not None:
            try:
                return next(transformer.parameters()).device.type
            except StopIteration:
                pass
        return "cpu"

    @execution_device.setter
    def execution_device(self, value: str) -> None:
        self._anima_execution_device = str(value)

    @property
    def model_dtype(self) -> torch.dtype:
        override = getattr(self, "_anima_model_dtype", None)
        if isinstance(override, torch.dtype):
            return override
        dtype = getattr(self, "dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        return torch.float32

    @model_dtype.setter
    def model_dtype(self, value: torch.dtype) -> None:
        self._anima_model_dtype = value

    @property
    def spatial_step(self) -> int:
        """Return the required pixel step for width/height alignment."""
        return self.vae_scale_factor * self.patch_size

    def check_inputs(
        self,
        *,
        prompt: PromptInput,
        negative_prompt: PromptInput | None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image: ImageBatchInput | None,
        mask_image: ImageBatchInput | None,
        strength: float,
        width: int,
        height: int,
        num_inference_steps: int,
        num_images_per_prompt: int,
        generator: GeneratorInput | None,
        sampler: str,
        sigma_schedule: str,
        cfg_batch_mode: str,
        output_type: str,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> None:
        """Validate user-facing call arguments and runtime constraints."""
        if (prompt_embeds is None) != (negative_prompt_embeds is None):
            raise ValueError("`prompt_embeds` and `negative_prompt_embeds` must both be provided " "or both be None.")
        if prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            prompts, _ = _resolve_prompt_batches(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
            )
            batch_size = len(prompts)

        if strength <= 0.0 or strength > 1.0:
            raise ValueError("`strength` must be in (0.0, 1.0].")
        step = self.spatial_step
        if width < step or height < step:
            raise ValueError(f"`width` and `height` must be >= {step}.")
        if width % step != 0 or height % step != 0:
            suggested_height = height - (height % step)
            suggested_width = width - (width % step)
            raise ValueError(
                f"`width` and `height` must be divisible by {step} but are {width} and {height}. "
                f"Try width={suggested_width}, height={suggested_height}."
            )
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be >= 1.")
        if image is None and mask_image is not None:
            raise ValueError("`mask_image` requires `image`.")
        if image is None and not math.isclose(strength, 1.0):
            raise ValueError("`strength` can be changed only when `image` is provided.")

        _validate_image_like_input(image, input_name="image")
        _validate_image_like_input(mask_image, input_name="mask_image")
        _normalize_generator(generator, batch_size=batch_size)
        _validate_sampling_modes(
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            cfg_batch_mode=cfg_batch_mode,
            output_type=output_type,
        )
        _validate_callback_tensor_input_names(
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            allowed_inputs=self._callback_tensor_inputs,
        )

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: PromptInput,
        negative_prompt: PromptInput | None = None,
        num_images_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts to conditioning tensors.

        Pre-computing conditioning with this method allows reusing the same
        embeddings across multiple ``__call__`` invocations (e.g. sweeping seeds),
        avoiding redundant tokenisation and text-encoding on each call.

        Example::

            pos_cond, neg_cond = pipe.encode_prompt(prompt, negative_prompt)
            for seed in seeds:
                pipe(prompt, negative_prompt,
                     prompt_embeds=pos_cond, negative_prompt_embeds=neg_cond,
                     generator=torch.Generator().manual_seed(seed))

        Args:
            prompt: Text prompt(s) for generation.
            negative_prompt: Optional negative prompt(s). Defaults to empty string.
            num_images_per_prompt: Number of images to generate per prompt entry.

        Returns:
            A tuple ``(pos_cond, neg_cond)`` where each tensor has shape
            ``(batch_size, 512, text_embed_dim)``. Pass these as
            ``prompt_embeds`` / ``negative_prompt_embeds`` to ``__call__``.
        """
        prompts, negative_prompts = _resolve_prompt_batches(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
        )
        pos_hidden, pos_t5_ids, pos_t5_weights, neg_hidden, neg_t5_ids, neg_t5_weights = _prepare_prompt_embedding_inputs(
            self,
            prompt=prompts,
            negative_prompt=negative_prompts,
        )
        with _module_execution_context(
            self.transformer,
            execution_device=self.execution_device,
            execution_dtype=self.model_dtype,
            enable_offload=self.use_module_cpu_offload,
        ):
            pos_cond, neg_cond = _build_cfg_conditions_from_embeddings(
                self,
                pos_hidden=pos_hidden,
                pos_t5_ids=pos_t5_ids,
                pos_t5_weights=pos_t5_weights,
                neg_hidden=neg_hidden,
                neg_t5_ids=neg_t5_ids,
                neg_t5_weights=neg_t5_weights,
            )
        return pos_cond, neg_cond

    @torch.no_grad()
    def __call__(
        self,
        prompt: PromptInput,
        negative_prompt: PromptInput | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image: ImageBatchInput | None = None,
        mask_image: ImageBatchInput | None = None,
        strength: float = 1.0,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 32,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 4.0,
        generator: GeneratorInput | None = None,
        cfg_batch_mode: str = "split",
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[..., dict[str, Any] | None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> AnimaPipelineOutput | tuple[list[Image.Image] | np.ndarray | torch.Tensor]:
        """Generate images with Anima.

        Args:
            prompt: Text prompt(s) for generation.
            negative_prompt: Optional negative prompt(s).
            prompt_embeds: Pre-computed positive conditioning tensor from
                ``encode_prompt``. When provided, ``prompt`` is ignored for text
                encoding and this tensor is used directly. Requires
                ``negative_prompt_embeds`` to be provided as well.
            negative_prompt_embeds: Pre-computed negative conditioning tensor from
                ``encode_prompt``. Must be provided together with ``prompt_embeds``.
            image: Optional initial image for img2img or inpainting.
            mask_image: Optional inpaint mask (white = region to inpaint).
            strength: Noise strength for img2img (0.0–1.0].
            width: Output image width (must be divisible by ``spatial_step``).
            height: Output image height (must be divisible by ``spatial_step``).
            num_inference_steps: Number of denoising steps.
            num_images_per_prompt: Number of images to generate per prompt.
            guidance_scale: Classifier-free guidance scale.
            generator: Optional RNG seed(s).
            cfg_batch_mode: How to run classifier-free guidance. ``split`` runs
                positive and negative conditioning as two sequential forward passes
                (more numerically stable, lower peak VRAM per pass). ``concat``
                batches them into a single forward (higher throughput on large GPUs,
                but roughly doubles the batch size and may require more peak VRAM).
            output_type: ``pil``, ``np``, or ``latent``.
            return_dict: Return ``AnimaPipelineOutput`` when ``True``.
            callback_on_step_end: Optional callable invoked at each step end.
            callback_on_step_end_tensor_inputs: Tensor names passed to the callback.

        Notes:
            Sampling parameters (sampler, sigma_schedule, eta, s_noise, beta_alpha,
            beta_beta) are first-class scheduler config options. Set them with
            ``pipe.scheduler.set_sampling_config(...)`` before calling.

            - ``flowmatch_euler`` requires ``sigma_schedule='uniform'``.
            - ``eta`` and ``s_noise`` are ignored for ``flowmatch_euler`` and ``euler``.
            - ``beta_alpha`` and ``beta_beta`` are used only for ``sigma_schedule='beta'``.
        """
        sampling_config: AnimaSamplingConfig = self.scheduler.get_sampling_config()
        sampler = sampling_config.sampler
        sigma_schedule = sampling_config.sigma_schedule
        beta_alpha = sampling_config.beta_alpha
        beta_beta = sampling_config.beta_beta
        eta = sampling_config.eta
        s_noise = sampling_config.s_noise
        resolved_callback_tensor_inputs = callback_on_step_end_tensor_inputs
        if resolved_callback_tensor_inputs is None:
            resolved_callback_tensor_inputs = ["latents"]

        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=image,
            mask_image=mask_image,
            strength=strength,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            cfg_batch_mode=cfg_batch_mode,
            output_type=output_type,
            callback_on_step_end_tensor_inputs=resolved_callback_tensor_inputs,
        )
        _warn_ignored_sampling_arguments(
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            eta=eta,
            s_noise=s_noise,
        )

        try:
            images = _generate_image(
                self,
                prompt=prompt,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=image,
                mask_image=mask_image,
                strength=strength,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                generator=generator,
                sampler=sampler,
                sigma_schedule=sigma_schedule,
                beta_alpha=beta_alpha,
                beta_beta=beta_beta,
                eta=eta,
                s_noise=s_noise,
                cfg_batch_mode=cfg_batch_mode,
                output_type=output_type,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=resolved_callback_tensor_inputs,
            )
        finally:
            self.maybe_free_model_hooks()

        if output_type == "pil":
            output_images: list[Image.Image] | np.ndarray | torch.Tensor = images
        elif output_type == "np":
            output_images = np.stack([np.asarray(image, dtype=np.uint8) for image in images], axis=0)
        else:
            output_images = images

        if not return_dict:
            return (output_images,)
        return AnimaPipelineOutput(images=output_images)

    def enable_model_cpu_offload(
        self,
        gpu_id: int | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        """Move models between CPU and GPU around each inference stage.

        Replaces the Diffusers hook-based offload with Anima's
        ``_module_execution_context`` mechanism, which is aware of the
        manual text-encoder → transformer → VAE execution order used by this
        pipeline (including the adapter conditioning pass that runs before
        the main denoising loop).

        Args:
            gpu_id: Deprecated; ignored. Configure the execution device via
                the ``execution_device`` attribute instead.
            device: The accelerator device to use. Accepts a device string
                (``"cuda"``, ``"cuda:1"``, ``"mps"``) or ``torch.device``.
                Ignored when ``execution_device`` was set explicitly.
        """
        if getattr(self, "_anima_execution_device", "auto") == "auto":
            self._anima_execution_device = device.type if isinstance(device, torch.device) else str(device)
        self.use_module_cpu_offload = True

    def enable_vae_slicing(self) -> None:
        """Enable VAE slicing when the backend VAE implementation supports it."""
        _enable_vae_method(
            self.vae,
            enabled=True,
            method_name="enable_slicing",
            unsupported_feature_name="VAE slicing",
        )

    def disable_vae_slicing(self) -> None:
        """Disable VAE slicing when the backend VAE implementation supports it."""
        _disable_vae_method(
            self.vae,
            method_name="disable_slicing",
            unsupported_feature_name="VAE slicing",
        )

    def enable_vae_tiling(self) -> None:
        """Enable VAE tiling when the backend VAE implementation supports it."""
        _enable_vae_method(
            self.vae,
            enabled=True,
            method_name="enable_tiling",
            unsupported_feature_name="VAE tiling",
        )

    def disable_vae_tiling(self) -> None:
        """Disable VAE tiling when the backend VAE implementation supports it."""
        _disable_vae_method(
            self.vae,
            method_name="disable_tiling",
            unsupported_feature_name="VAE tiling",
        )

    def enable_vae_xformers_memory_efficient_attention(self) -> None:
        """Enable xformers memory-efficient attention for the VAE when supported."""
        method = getattr(self.vae, "set_use_memory_efficient_attention_xformers", None)
        if method is None:
            warnings.warn(
                "VAE xformers is not supported by the current VAE implementation.",
                UserWarning,
                stacklevel=2,
            )
            return
        try:
            method(True)
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Failed to enable VAE xformers attention: {exc}",
                stacklevel=2,
            )

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        """Save the pipeline and bundled prompt tokenizer artifacts."""
        super().save_pretrained(save_directory, **kwargs)
        if self.prompt_tokenizer is None:
            return
        save_prompt_tokenizers_to_local_dir(
            prompt_tokenizer=self.prompt_tokenizer,
            save_directory=Path(save_directory),
        )

    @classmethod
    def _from_pretrained_local_directory(
        cls,
        pretrained_model_name_or_path: str,
        *,
        pipeline_dir: Path,
        kwargs: dict[str, Any],
    ) -> "AnimaPipeline":
        load_options = loader_options_from_kwargs(kwargs, consume=False)
        loaded = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        normalize_loaded_component_buffers(loaded)
        loaded.scheduler = coerce_anima_scheduler(loaded.scheduler)

        if loaded.prompt_tokenizer is None:
            qwen_source, t5_source, uses_local_tokenizers = resolve_prompt_tokenizer_sources_for_local_dir(
                pipeline_dir=pipeline_dir,
            )
            tokenizer_options = load_options
            if uses_local_tokenizers:
                tokenizer_options = AnimaLoaderOptions(
                    local_files_only=True,
                    cache_dir=load_options.cache_dir,
                    force_download=False,
                    token=load_options.token,
                    revision=load_options.revision,
                    proxies=load_options.proxies,
                )
            loaded.prompt_tokenizer = load_prompt_tokenizer(
                qwen_tokenizer_source=qwen_source,
                t5_tokenizer_source=t5_source,
                options=tokenizer_options,
            )
        return loaded

    @classmethod
    def _from_pretrained_diffusers_repo(
        cls,
        pretrained_model_name_or_path: str,
        *,
        kwargs: dict[str, Any],
    ) -> "AnimaPipeline":
        load_options = loader_options_from_kwargs(kwargs, consume=False)
        loaded = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        normalize_loaded_component_buffers(loaded)
        loaded.scheduler = coerce_anima_scheduler(loaded.scheduler)

        if loaded.prompt_tokenizer is None:
            # Tokenizers are always bundled in the Diffusers-format repository under
            # fixed subdirectory names. If not present locally, loading.py falls back
            # to the fixed Anima HF repository sources.
            from .loading import _QWEN_TOKENIZER_SOURCE, _T5_TOKENIZER_SOURCE

            loaded.prompt_tokenizer = load_prompt_tokenizer(
                qwen_tokenizer_source=_QWEN_TOKENIZER_SOURCE,
                t5_tokenizer_source=_T5_TOKENIZER_SOURCE,
                options=load_options,
            )
        return loaded

    @classmethod
    def _from_single_file_source(
        cls,
        pretrained_model_name_or_path: str,
        *,
        kwargs: dict[str, Any],
    ) -> "AnimaPipeline":
        ignored, unknown = _partition_single_file_from_pretrained_kwargs(kwargs)
        for key in ignored:
            kwargs.pop(key, None)
        if ignored:
            warnings.warn(
                "Ignoring unsupported from_pretrained arguments for Anima single-file loading: " + ", ".join(ignored),
                stacklevel=2,
            )

        components = AnimaComponents(model_path=str(pretrained_model_name_or_path))
        scheduler = scheduler_from_kwargs(kwargs, consume=False)
        runtime_options = runtime_options_from_kwargs(kwargs, consume=False)
        load_options = loader_options_from_kwargs(kwargs, consume=False)
        if unknown:
            raise ValueError(f"Unsupported arguments for AnimaPipeline.from_single_file: {', '.join(unknown)}")

        runtime = build_anima_pipeline(
            components=components,
            device=runtime_options.device,
            dtype=runtime_options.dtype,
            text_encoder_dtype=runtime_options.text_encoder_dtype,
            local_files_only=load_options.local_files_only,
            cache_dir=load_options.cache_dir,
            force_download=load_options.force_download,
            token=load_options.token,
            revision=load_options.revision,
            proxies=load_options.proxies,
            scheduler=scheduler,
        )
        return runtime

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> "AnimaPipeline":
        """Load Anima from a Diffusers-format pipeline directory or Hub repository."""
        _raise_if_removed_from_pretrained_runtime_feature_kwargs(kwargs, api_name="from_pretrained")
        custom_single_file_only = sorted(key for key in kwargs if key in _ANIMA_COMPONENT_OVERRIDE_KEYS)
        custom_runtime_only = sorted(key for key in kwargs if key in {"device", "dtype", "text_encoder_dtype"})
        unsupported_custom = custom_single_file_only + custom_runtime_only
        if unsupported_custom:
            raise ValueError(
                "Unsupported `from_pretrained` arguments for Diffusers-format loading: "
                + ", ".join(unsupported_custom)
                + ". Use standard Diffusers `from_pretrained(...)` kwargs for converted repositories "
                + "or `from_single_file(...)` for raw checkpoints."
            )
        scheduler = kwargs.get("scheduler", None)
        if scheduler is not None:
            kwargs["scheduler"] = coerce_anima_scheduler(scheduler)

        source = str(pretrained_model_name_or_path)
        if _looks_like_single_file_source(source):
            raise ValueError(
                "`from_pretrained` only supports Diffusers-format repositories/directories. "
                "Use `AnimaPipeline.from_single_file(...)` for raw `.safetensors` checkpoints."
            )

        path = Path(source)
        if path.is_dir() and (path / "model_index.json").exists():
            return cls._from_pretrained_local_directory(
                pretrained_model_name_or_path,
                pipeline_dir=path,
                kwargs=kwargs,
            )
        return cls._from_pretrained_diffusers_repo(
            pretrained_model_name_or_path,
            kwargs=kwargs,
        )

    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path: str, **kwargs: Any) -> "AnimaPipeline":
        """Load Anima from a single-file checkpoint with Diffusers-like kwargs."""
        _raise_if_removed_from_pretrained_runtime_feature_kwargs(kwargs, api_name="from_single_file")
        _pop_ignored_kwargs(
            kwargs,
            ignored_keys=_DIFFUSERS_COMPAT_IGNORED_FROM_SINGLE_FILE_KEYS,
            api_name="from_single_file",
        )
        return cls._from_single_file_source(
            pretrained_model_link_or_path,
            kwargs=kwargs,
        )
