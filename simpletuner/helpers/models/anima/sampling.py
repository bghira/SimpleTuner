# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/pipelines/anima/sampling.py
# Adapted for SimpleTuner local imports.

"""Anima denoising sampling loops."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline, ModelMixin, SchedulerMixin

import torch

from .constants import ANIMA_SAMPLING_MULTIPLIER
from .image_processing import _ensure_finite

GeneratorInput = torch.Generator | list[torch.Generator] | tuple[torch.Generator, ...]


def randn_tensor(
    shape: tuple[int, ...],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | list[torch.Generator] | None,
) -> torch.Tensor:
    """Generate a random normal tensor, honouring a device-potentially-mismatched generator.

    When ``generator`` is on a different device than ``device``, noise is
    generated on the generator's device in float32 and moved afterwards.
    This matches Diffusers' ``randn_tensor`` utility behaviour.
    """
    target_device = torch.device(device)
    if generator is None:
        return torch.randn(shape, device=target_device, dtype=dtype)

    if isinstance(generator, list):
        if shape[0] != len(generator):
            raise ValueError(f"`generator` list length must match tensor batch size ({shape[0]}), got {len(generator)}.")
        samples: list[torch.Tensor] = []
        for item_generator in generator:
            sample = randn_tensor(
                (1, *shape[1:]),
                device=target_device,
                dtype=dtype,
                generator=item_generator,
            )
            samples.append(sample)
        return torch.cat(samples, dim=0)

    generator_device = generator.device.type if hasattr(generator, "device") else target_device.type
    if generator_device == target_device.type:
        return torch.randn(shape, device=target_device, dtype=dtype, generator=generator)

    noise = torch.randn(shape, device=generator.device, dtype=torch.float32, generator=generator)
    return noise.to(device=target_device, dtype=dtype)


def randn_like(sample: torch.Tensor, generator: torch.Generator | list[torch.Generator] | None) -> torch.Tensor:
    return randn_tensor(
        tuple(sample.shape),
        device=sample.device,
        dtype=sample.dtype,
        generator=generator,
    )


def _predict_noise_cfg(
    transformer: "ModelMixin",
    latents: torch.Tensor,
    *,
    model_timestep: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    cfg_batch_mode: str,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    model_input = latents.to(dtype=model_dtype)
    timestep = model_timestep.to(device=model_input.device, dtype=torch.float32)
    if timestep.ndim == 0:
        timestep = timestep.expand(model_input.shape[0])

    if cfg_batch_mode == "concat":
        model_input = torch.cat([model_input, model_input], dim=0)
        timestep = torch.cat([timestep, timestep], dim=0)
        encoder_hidden_states = torch.cat(
            [
                pos_cond.to(device=model_input.device, dtype=model_dtype),
                neg_cond.to(device=model_input.device, dtype=model_dtype),
            ],
            dim=0,
        )
        with torch.inference_mode():
            noise_pred = transformer(
                model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2, dim=0)
    elif cfg_batch_mode == "split":
        with torch.inference_mode():
            noise_pred_uncond = transformer(
                model_input,
                timestep,
                encoder_hidden_states=neg_cond.to(device=model_input.device, dtype=model_dtype),
                return_dict=False,
            )[0]
            noise_pred_text = transformer(
                model_input,
                timestep,
                encoder_hidden_states=pos_cond.to(device=model_input.device, dtype=model_dtype),
                return_dict=False,
            )[0]
    else:
        raise ValueError("cfg_batch_mode must be one of: split, concat.")

    noise = (noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)).float()
    if model_dtype == torch.float16:
        _ensure_finite(noise, name="noise prediction", runtime_dtype=model_dtype)
    return noise


def _predict_denoised_const(
    transformer: "ModelMixin",
    latents: torch.Tensor,
    *,
    sigma: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    cfg_batch_mode: str,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    model_timestep = (sigma * ANIMA_SAMPLING_MULTIPLIER).expand(latents.shape[0]).float()
    noise_pred = _predict_noise_cfg(
        transformer,
        latents,
        model_timestep=model_timestep,
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        guidance_scale=guidance_scale,
        cfg_batch_mode=cfg_batch_mode,
        model_dtype=model_dtype,
    )
    return latents - sigma.float() * noise_pred


def _run_step_callback(
    pipeline: "DiffusionPipeline",
    *,
    callback_on_step_end: Callable[..., dict[str, Any] | None] | None,
    callback_on_step_end_tensor_inputs: list[str],
    step_index: int,
    timestep: torch.Tensor,
    latents: torch.Tensor,
) -> torch.Tensor:
    if callback_on_step_end is None:
        return latents

    callback_kwargs: dict[str, Any] = {}
    if "latents" in callback_on_step_end_tensor_inputs:
        callback_kwargs["latents"] = latents

    callback_outputs = callback_on_step_end(pipeline, step_index, timestep, callback_kwargs)
    if callback_outputs is None:
        return latents
    if not isinstance(callback_outputs, dict):
        raise TypeError("callback_on_step_end must return dict[str, Any] or None.")

    return callback_outputs.pop("latents", latents)


def sample_euler(
    transformer: "ModelMixin",
    pipeline: "DiffusionPipeline",
    latents: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    cfg_batch_mode: str,
    model_dtype: torch.dtype,
    callback_on_step_end: Callable[..., dict[str, Any] | None] | None,
    callback_on_step_end_tensor_inputs: list[str],
    inpaint_mask: torch.Tensor | None = None,
    init_image_latents: torch.Tensor | None = None,
    init_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler sampler on a constant (non-flow-match) sigma trajectory."""
    if inpaint_mask is not None and (init_image_latents is None or init_noise is None):
        raise ValueError("inpaint sampling requires both `init_image_latents` and `init_noise`.")

    _iterable = pipeline.progress_bar(total=len(sigmas) - 1)
    for i in range(len(sigmas) - 1):
        _iterable.update(1)
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = _predict_denoised_const(
            transformer,
            latents,
            sigma=sigma,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            cfg_batch_mode=cfg_batch_mode,
            model_dtype=model_dtype,
        )
        if sigma_next.item() == 0.0:
            latents = denoised
            latents = _run_step_callback(
                pipeline,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                step_index=i,
                timestep=sigma,
                latents=latents,
            )
            continue

        d = (latents - denoised) / sigma.to(latents.dtype)
        dt = (sigma_next - sigma).to(latents.dtype)
        latents = latents + d * dt
        if inpaint_mask is not None and init_image_latents is not None and init_noise is not None:
            sigma_next_value = sigma_next.to(init_image_latents.dtype)
            source_latents = sigma_next_value * init_noise + (1.0 - sigma_next_value) * init_image_latents
            latents = (1.0 - inpaint_mask) * source_latents + inpaint_mask * latents
        latents = _run_step_callback(
            pipeline,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            step_index=i,
            timestep=sigma,
            latents=latents,
        )
    return latents


def sample_euler_ancestral_rf(
    transformer: "ModelMixin",
    pipeline: "DiffusionPipeline",
    latents: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    eta: float,
    s_noise: float,
    generator: torch.Generator | list[torch.Generator] | None,
    cfg_batch_mode: str,
    model_dtype: torch.dtype,
    callback_on_step_end: Callable[..., dict[str, Any] | None] | None,
    callback_on_step_end_tensor_inputs: list[str],
    inpaint_mask: torch.Tensor | None = None,
    init_image_latents: torch.Tensor | None = None,
    init_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ancestral RF Euler sampler on a constant (non-flow-match) sigma trajectory."""
    if inpaint_mask is not None and (init_image_latents is None or init_noise is None):
        raise ValueError("inpaint sampling requires both `init_image_latents` and `init_noise`.")

    _iterable = pipeline.progress_bar(total=len(sigmas) - 1)
    for i in range(len(sigmas) - 1):
        _iterable.update(1)
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = _predict_denoised_const(
            transformer,
            latents,
            sigma=sigma,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            cfg_batch_mode=cfg_batch_mode,
            model_dtype=model_dtype,
        )
        if sigma_next.item() == 0.0:
            latents = denoised
            latents = _run_step_callback(
                pipeline,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                step_index=i,
                timestep=sigma,
                latents=latents,
            )
            continue

        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
        sigma_down = sigma_next * downstep_ratio
        alpha_ip1 = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        renoise_sq = sigma_next**2 - sigma_down**2 * alpha_ip1**2 / (alpha_down**2)
        renoise_coeff = renoise_sq.clamp_min(0).sqrt()

        sigma_down_ratio = sigma_down / sigma
        latents = sigma_down_ratio.to(latents.dtype) * latents + (1.0 - sigma_down_ratio).to(latents.dtype) * denoised
        if eta > 0:
            noise = randn_like(latents, generator=generator)
            latents = (alpha_ip1 / alpha_down).to(latents.dtype) * latents + noise * s_noise * renoise_coeff.to(
                latents.dtype
            )
        if inpaint_mask is not None and init_image_latents is not None and init_noise is not None:
            sigma_next_value = sigma_next.to(init_image_latents.dtype)
            source_latents = sigma_next_value * init_noise + (1.0 - sigma_next_value) * init_image_latents
            latents = (1.0 - inpaint_mask) * source_latents + inpaint_mask * latents
        latents = _run_step_callback(
            pipeline,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            step_index=i,
            timestep=sigma,
            latents=latents,
        )
    return latents


def sample_flowmatch_euler(
    transformer: "ModelMixin",
    scheduler: "SchedulerMixin",
    pipeline: "DiffusionPipeline",
    latents: torch.Tensor,
    *,
    timesteps: torch.Tensor,
    sigma_schedule: str,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    cfg_batch_mode: str,
    model_dtype: torch.dtype,
    callback_on_step_end: Callable[..., dict[str, Any] | None] | None,
    callback_on_step_end_tensor_inputs: list[str],
    inpaint_mask: torch.Tensor | None = None,
    init_image_latents: torch.Tensor | None = None,
    init_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flow-match Euler sampler using the Diffusers scheduler step."""
    if sigma_schedule != "uniform":
        raise ValueError("flowmatch_euler sampler only supports sigma_schedule='uniform'.")
    if inpaint_mask is not None and (init_image_latents is None or init_noise is None):
        raise ValueError("inpaint sampling requires both `init_image_latents` and `init_noise`.")

    _iterable = pipeline.progress_bar(timesteps)
    for i, timestep in enumerate(_iterable):
        scheduler_timestep = timestep.expand(latents.shape[0]).float()
        model_timestep = (scheduler_timestep / float(scheduler.config.num_train_timesteps)) * ANIMA_SAMPLING_MULTIPLIER

        noise_pred = _predict_noise_cfg(
            transformer,
            latents,
            model_timestep=model_timestep,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            cfg_batch_mode=cfg_batch_mode,
            model_dtype=model_dtype,
        )

        latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        if inpaint_mask is not None and init_image_latents is not None and init_noise is not None:
            source_latents = init_image_latents
            if i < len(timesteps) - 1:
                next_timestep = (
                    timesteps[i + 1]
                    .expand(init_image_latents.shape[0])
                    .to(
                        device=init_image_latents.device,
                        dtype=torch.float32,
                    )
                )
                source_latents = scheduler.scale_noise(init_image_latents, next_timestep, init_noise)
            latents = (1.0 - inpaint_mask) * source_latents + inpaint_mask * latents
        latents = _run_step_callback(
            pipeline,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            step_index=i,
            timestep=timestep,
            latents=latents,
        )
    return latents


def run_const_sigma_samplers(
    transformer: "ModelMixin",
    pipeline: "DiffusionPipeline",
    latents: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    sampler: str,
    pos_cond: torch.Tensor,
    neg_cond: torch.Tensor,
    guidance_scale: float,
    eta: float,
    s_noise: float,
    generator: torch.Generator | list[torch.Generator] | None,
    cfg_batch_mode: str,
    model_dtype: torch.dtype,
    callback_on_step_end: Callable[..., dict[str, Any] | None] | None,
    callback_on_step_end_tensor_inputs: list[str],
    input_is_noisy_latents: bool = False,
    inpaint_mask: torch.Tensor | None = None,
    init_image_latents: torch.Tensor | None = None,
    init_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch to the appropriate non-flowmatch sampler."""
    if len(sigmas) < 2:
        raise ValueError("At least 1 denoising step is required.")
    if not input_is_noisy_latents:
        latents = latents * sigmas[0].to(latents.dtype)

    if sampler == "euler":
        return sample_euler(
            transformer,
            pipeline,
            latents,
            sigmas=sigmas,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            cfg_batch_mode=cfg_batch_mode,
            model_dtype=model_dtype,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            inpaint_mask=inpaint_mask,
            init_image_latents=init_image_latents,
            init_noise=init_noise,
        )

    if sampler in {"euler_a_rf", "euler_ancestral_rf"}:
        return sample_euler_ancestral_rf(
            transformer,
            pipeline,
            latents,
            sigmas=sigmas,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            guidance_scale=guidance_scale,
            eta=eta,
            s_noise=s_noise,
            generator=generator,
            cfg_batch_mode=cfg_batch_mode,
            model_dtype=model_dtype,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            inpaint_mask=inpaint_mask,
            init_image_latents=init_image_latents,
            init_noise=init_noise,
        )

    raise ValueError(
        f"Unsupported sampler '{sampler}'. Choose one of: " "flowmatch_euler, euler, euler_a_rf, euler_ancestral_rf."
    )
