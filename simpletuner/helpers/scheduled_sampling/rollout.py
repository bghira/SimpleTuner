from typing import Any

import torch
from skrample import common as sk_common

from simpletuner.helpers.models.common import PredictionTypes
from simpletuner.helpers.scheduled_sampling.skrample_adapter import make_sampler, make_sigma_schedule_from_ddpm


def _slice_batch_for_index(batch: dict[str, Any], idx: int, device: torch.device) -> dict[str, Any]:
    """
    Create a shallow copy of the batch with tensors sliced to a single item.
    Non-tensor entries are reused.
    """
    sliced = {}
    for key, value in batch.items():
        if key in ["scheduled_sampling_plan"]:
            continue
        if isinstance(value, torch.Tensor):
            if value.dim() > 0 and value.shape[0] > idx:
                sliced[key] = value[idx : idx + 1].to(device=device)
            else:
                sliced[key] = value.to(device=device)
        elif isinstance(value, dict):
            # Recursively slice dict values
            sliced[key] = _slice_batch_for_index(value, idx, device)
        else:
            sliced[key] = value
    return sliced


def _prediction_to_x(
    noisy_latents: torch.Tensor,
    model_pred: torch.Tensor,
    sigma: float,
    sigma_transform,
    prediction_type: PredictionTypes,
) -> torch.Tensor:
    """
    Convert model prediction (eps or v) to a denoised sample x0 compatible with Skrample samplers.
    """
    sigma_u, sigma_v = sigma_transform(float(sigma))
    sigma_u = torch.as_tensor(sigma_u, device=noisy_latents.device, dtype=noisy_latents.dtype)
    sigma_v = torch.as_tensor(sigma_v, device=noisy_latents.device, dtype=noisy_latents.dtype)

    if prediction_type is PredictionTypes.EPSILON:
        return (noisy_latents - sigma_u * model_pred) / sigma_v
    if prediction_type is PredictionTypes.V_PREDICTION:
        return sigma_v * noisy_latents - sigma_u * model_pred
    return model_pred


@torch.no_grad()
def apply_scheduled_sampling_rollout(model, prepared_batch: dict, noise_schedule, config) -> dict:
    """
    Apply scheduled sampling rollout to replace noisy_latents/timesteps for EPS/V models.
    Uses the model's own predictions with the current scheduler to step from source->target.
    """
    plan = prepared_batch.get("scheduled_sampling_plan")
    if plan is None:
        return prepared_batch

    if model.PREDICTION_TYPE not in [PredictionTypes.EPSILON, PredictionTypes.V_PREDICTION]:
        return prepared_batch

    rollout_steps = getattr(plan, "rollout_steps", None)
    source_ts = getattr(plan, "source_timesteps", None)
    target_ts = getattr(plan, "target_timesteps", None)
    if rollout_steps is None or source_ts is None or target_ts is None:
        return prepared_batch

    try:
        schedule, sigma_transform = make_sigma_schedule_from_ddpm(noise_schedule)
    except Exception:
        return prepared_batch

    sampler_name = getattr(config, "scheduled_sampling_sampler", "unipc")
    sampler_order = getattr(config, "scheduled_sampling_order", 2)
    sampler = make_sampler(sampler_name, sampler_order)

    expects_pair_schedule = getattr(sampler, "_expects_pair_schedule", True)
    sigma_schedule = schedule
    if not expects_pair_schedule and hasattr(schedule, "ndim") and getattr(schedule, "ndim", 1) > 1:
        sigma_schedule = schedule[:, 1]

    device = prepared_batch["noisy_latents"].device
    dtype = prepared_batch["noisy_latents"].dtype

    latents = prepared_batch["latents"]
    noise = prepared_batch.get("input_noise", prepared_batch.get("noise"))
    if noise is None:
        return prepared_batch

    # Buffers for updated latents/timesteps
    new_noisy = prepared_batch["noisy_latents"].clone()
    new_timesteps = prepared_batch["timesteps"].clone()

    bsz = latents.shape[0]
    schedule_len = len(sigma_schedule)
    # Track previous samples per batch for multistep samplers
    prev_cache: list[list] = [[] for _ in range(bsz)]
    for i in range(bsz):
        offset = int(rollout_steps[i].item())
        if offset <= 0:
            continue
        source_t = int(source_ts[i].item())
        target_t = int(target_ts[i].item())
        if source_t <= target_t or source_t >= schedule_len:
            continue

        # Recreate noisy latents at source timestep using the same latents/noise
        current = noise_schedule.add_noise(
            latents[i : i + 1].float(),
            noise[i : i + 1].float(),
            torch.tensor([source_t], device=device),
        ).to(device=device, dtype=dtype)

        prev_samples = prev_cache[i]

        # Step down from source_t -> target_t with the model using skrample sampler
        for t in range(source_t, target_t, -1):
            if t <= 0:
                break
            sigma = sigma_schedule[t][1] if expects_pair_schedule else sigma_schedule[t]
            # Build a sliced batch for this sample/timestep
            mini_batch = _slice_batch_for_index(prepared_batch, i, device)
            mini_batch["noisy_latents"] = current
            mini_batch["timesteps"] = torch.tensor([t], device=device, dtype=torch.long)
            mini_batch["input_noise"] = noise[i : i + 1].to(device=device, dtype=dtype)
            mini_batch.pop("scheduled_sampling_plan", None)

            if getattr(config, "controlnet", False):
                model_out = model.controlnet_predict(prepared_batch=mini_batch)
            else:
                model_out = model.model_predict(prepared_batch=mini_batch)

            if isinstance(model_out, dict):
                model_pred = model_out.get("model_prediction", None)
                if model_pred is None:
                    continue
            else:
                model_pred = model_out

            x_pred = _prediction_to_x(current, model_pred, sigma, sigma_transform, model.PREDICTION_TYPE)
            sk_result = sampler.sample(
                current,
                x_pred,
                step=t,
                schedule=schedule,
                sigma_transform=sigma_transform,
                noise=None,
                previous=tuple(prev_samples),
            )
            if sampler.require_previous > 0:
                prev_samples.append(sk_result)
                prev_samples = prev_samples[-sampler.require_previous :]
            current = sk_result.final.to(device=device, dtype=dtype)

        if sampler.require_previous > 0:
            prev_cache[i] = prev_samples[-sampler.require_previous :]
        else:
            prev_cache[i] = []
        new_noisy[i : i + 1] = current
        new_timesteps[i] = target_t

    prepared_batch["noisy_latents"] = new_noisy
    prepared_batch["timesteps"] = new_timesteps
    return prepared_batch
