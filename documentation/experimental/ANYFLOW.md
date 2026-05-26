# AnyFlow

AnyFlow is an experimental distillation mode for flow-matching models. It trains the model to condition on a pair of flow times, the normal training timestep `t` and a lower reference timestep `r`, so the network learns a flow map across an interval instead of only a single rectified-flow velocity.

SimpleTuner implements this through the existing FlowMap model hooks:

- `--distillation_method=anyflow` enables the `AnyFlowDistiller`.
- The distiller calls `enable_flowmap_time_conditioning()` on the trained component during startup.
- Each prepared batch receives `flowmap_r_timesteps`.
- The normal training target is replaced with an AnyFlow target before the model loss is computed.

AnyFlow is online in SimpleTuner. It does not require a precomputed ODE cache.

For a Wan continuation example using NVIDIA's released AnyFlow checkpoints, see [AnyFlow Continuation Quickstart](/documentation/quickstart/ANYFLOW.md).

## Quick Setup

```json
{
  "model_type": "lora",
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "target_mode": "online_teacher",
      "teacher_rollout_steps": 1,
      "r_timestep_sampler": "uniform",
      "min_interval_ratio": 0.02,
      "gate_value": 0.25,
      "deltatime_type": "r",
      "loss_weight": 1.0
    }
  }
}
```

Text encoder training is blocked for all SimpleTuner distillation methods, including AnyFlow.

## How It Works

For each flow-matching batch, SimpleTuner:

1. Uses the model's normal `prepare_batch()` path to sample `sigmas`, `timesteps`, `noisy_latents`, and the base flow target.
2. Samples `r < t` from the current sigma interval.
3. Writes `flowmap_r_timesteps` into the batch so model wrappers can pass it as `r_timestep`.
4. Builds the training target.
5. Lets the normal model loss compare the prediction to that target.

In `target_mode=online_teacher`, the target is an average velocity from the current noisy latent at `t` toward `r`. For LoRA and LyCORIS training, the distiller temporarily disables the adapter for the teacher rollout and re-enables it afterward.

In `target_mode=linear`, no teacher rollout is used. The target is the straight flow target `noise - latents`. This is useful for smoke tests and controlled ablations, but it is not the full AnyFlow teacher-map objective.

## Configuration

Common `distillation_config.anyflow` keys:

- `target_mode`: `online_teacher` or `linear`. Default: `online_teacher`.
- `teacher_rollout_steps`: number of online teacher Euler steps between `t` and `r`. Default: `1`.
- `r_timestep_sampler`: `uniform` or `zero`. Default: `uniform`.
- `min_interval_ratio`: minimum normalized interval left between `t` and `r`. Default: `0.02`.
- `gate_value`: blend weight for the FlowMap delta timestep embedding. Default: `0.25`.
- `deltatime_type`: `r` or `t-r`, matching the model FlowMap embedding mode. Default: `r`.
- `loss_weight`: multiplier applied to the already-computed training loss. Default: `1.0`.
- `timestep_scale`: override for models that use a custom timestep scale. Leave unset for normal operation.

`r_timestep_sampler=zero` always maps toward the clean endpoint. It is deterministic and useful for debugging. `uniform` samples inside the available interval.

## Supported Models

AnyFlow requires a flow-matching model whose trained component implements `enable_flowmap_time_conditioning()` and whose model wrapper forwards `flowmap_r_timesteps` to the model as `r_timestep`.

The current implementation covers the registered FlowMap-capable transformer families and the legacy Diffusers UNet families that use `FlowMapUNet2DConditionModel`.

## Limits

- Requires a flow-matching prediction type.
- Requires scalar per-sample timesteps. Tokenwise AnyFlow intervals are not wired yet.
- Requires `r_timestep < timestep`; timestep zero is rejected for AnyFlow training.
- The default online teacher mode is intended for LoRA/LyCORIS in the current trainer path. Full-rank online teacher training needs a separate student/teacher wiring pass.
- Standard validation can still run without `r_timestep`, but AnyFlow-style few-step sampling needs sampler or pipeline support that passes the interval endpoint as `r_timestep`. That generation-time integration is still a follow-up.

## Logs

AnyFlow adds:

- `anyflow_loss`
- `anyflow_timestep`
- `anyflow_r_timestep`
- `anyflow_interval`

These are emitted alongside the normal training loss metrics.
