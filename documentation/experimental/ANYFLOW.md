# AnyFlow

SimpleTuner implements NVIDIA AnyFlow as two explicit training stages for flow-matching models. Both stages train a
model that receives the current flow time `t` and an interval endpoint `r`.

- `stage=forward` implements NVIDIA's forward MeanFlow objective.
- `stage=onpolicy` implements Flow Map Backward Simulation and on-policy DMD while co-training the forward objective.

The removed `online_teacher` and `linear` target modes were SimpleTuner-specific objectives and are no longer
accepted.

For a Wan continuation example using NVIDIA's released checkpoints, see
[AnyFlow Continuation Quickstart](/documentation/quickstart/ANYFLOW.md).

## Forward Stage

```json
{
  "model_type": "lora",
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "stage": "forward",
      "diffusion_ratio": 0.5,
      "consistency_ratio": 0.25,
      "central_difference_epsilon": 0.005,
      "fuse_guidance_scale": 3.0,
      "meanflow_weight_type": "beta08",
      "meanflow_adaptive_weighting": true,
      "gate_value": 0.25,
      "deltatime_type": "r",
      "loss_weight": 1.0
    }
  }
}
```

For each global batch, the forward stage:

1. Samples two uniform flow times and sorts them into `t >= r`.
2. Assigns 50% of samples to diffusion intervals (`r=t`), 25% to endpoint intervals (`r=0`), and the remainder to
   arbitrary intervals.
3. Applies the model scheduler's flow shift to both endpoints.
4. Evaluates a central difference along the straight latent flow path.
5. Fuses the trainable conditional prediction with a detached unconditional pass at `fuse_guidance_scale`, while
   retaining the raw flow velocity as the MeanFlow target.
6. Builds the MeanFlow tangent target and applies NVIDIA's normalized `beta08` timestep weighting.
7. Balances each non-diffusion sample against the global diffusion-branch loss mean.

Guidance fusion requires cached unconditional text embeddings. SimpleTuner loads the caption-dropout embedding for
each sample and uses the same image context for models such as MiniMax-H3. Set `fuse_guidance_scale=1.0` only when the
student should retain external CFG instead of learning AnyFlow's guidance-distilled conditional field.

## On-Policy Stage

Start this stage from a forward-stage AnyFlow adapter by setting `init_lora` or resuming its checkpoint:

```json
{
  "model_type": "lora",
  "lora_type": "standard",
  "init_lora": "path-or-repo-to-forward-anyflow-adapter",
  "learning_rate": 0.000002,
  "optimizer_beta1": 0.0,
  "optimizer_beta2": 0.999,
  "optimizer_weight_decay": 0.0,
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "stage": "onpolicy",
      "cotrain_forward": true,
      "rollout_step_counts": [2, 4, 8, 16, 50],
      "dmd_weight": 1.0,
      "dmd_batch_size": 1,
      "real_score_guidance_scale": 0.0,
      "discriminator_lr": 0.000002,
      "discriminator_betas": [0.0, 0.999],
      "discriminator_weight_decay": 0.0,
      "discriminator_grad_clip": 1.0
    }
  }
}
```

The on-policy stage uses three score roles. Standard LoRA training shares one frozen base transformer between them:

- The loaded AnyFlow adapter is the generator.
- The base model with adapters disabled is the frozen real score.
- A separately optimized `anyflow_discriminator` adapter is the fake score.

Each generator update selects a rollout budget and one gradient grid index. It then performs at most three FlowMap
jumps: start to the selected index, one fine grid step, and the following index to the endpoint. The generated latent is
noised at a shifted uniform time before applying NVIDIA's normalized DMD gradient. Each discriminator
update performs a no-grad student rollout, samples a logit-normal shifted time, and trains the fake score on the normal
flow target. The discriminator adapter and optimizer are saved beside every SimpleTuner checkpoint as
`anyflow_discriminator.safetensors` and `anyflow_discriminator_optim.pt`.

MiniMax-H3 already contains CFG distillation, so its on-policy runs should normally keep
`real_score_guidance_scale=0`. Models that require an external real-score CFG pass must cache negative text embeddings
and can set the scale explicitly.

When `--seed` is set, AnyFlow samples MeanFlow intervals, rollout schedules, rollout latents, DMD noise, and DMD sigmas
from an isolated per-device Torch generator. This keeps AnyFlow samples stable when unrelated training code consumes the
global Torch RNG. It does not make CUDA attention backward bit-stable.

## Shared Configuration

- `stage`: `forward` or `onpolicy`. Default: `forward`.
- `diffusion_ratio`: global batch fraction using `r=t`. Default: `0.5`.
- `consistency_ratio`: global batch fraction using `r=0`. Default: `0.25`.
- `central_difference_epsilon`: normalized shifted-time offset. Default: `0.005`, matching NVIDIA's `5/1000`.
- `fuse_guidance_scale`: guidance scale distilled into the conditional student prediction. Default: `3.0`.
- `meanflow_weight_type`: `beta08` or `uniform`. Default: `beta08`.
- `meanflow_adaptive_weighting`: balance non-diffusion samples against the diffusion branch. Default: `true`.
- `gate_value`: FlowMap delta-timestep embedding blend. Default: `0.25`.
- `deltatime_type`: `r` or `t-r`. Default: `r`.
- `loss_weight`: forward MeanFlow loss multiplier. Default: `1.0`.

## Limits

- AnyFlow requires a flow-matching model with model-specific FlowMap interval conditioning.
- On-policy training currently requires standard PEFT LoRA. Sharing the base avoids allocating generator, real-score,
  and discriminator copies of a large transformer on every DDP rank.
- Joint MiniMax-H3 audio-video training is rejected. Video uses schedule shift 12 while audio uses shift 3; native
  dual-schedule MeanFlow targets and rollouts need to be implemented before AV training is valid.
- Text encoder training is disabled for all SimpleTuner distillation methods. Guidance fusion loads cached
  unconditional embeddings and does not run the text encoder in the training loop.
- Validation uses `AnyFlowValidationScheduler`, which supplies the next interval endpoint to registered FlowMap model
  components.

## Logs

Forward training adds `anyflow_forward_loss`, `anyflow_fuse_guidance_scale`, timestep and interval values, and global
branch fractions. On-policy
training also adds `anyflow_dmd_loss`, `anyflow_dmd_gradient_norm`, `anyflow_dmd_sigma`, and
`anyflow_rollout_steps` and `anyflow_rollout_grad_timestep`.
