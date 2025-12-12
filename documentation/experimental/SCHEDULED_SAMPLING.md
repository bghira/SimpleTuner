# Scheduled Sampling (Rollout)

## Background

Standard diffusion training relies on "Teacher Forcing". We take a clean image, add a precise amount of noise to it, and ask the model to predict that noise (or the velocity/original image). The input to the model is always "perfectly" noisy—it lies exactly on the theoretical noise schedule.

However, during inference (generation), the model feeds on its own outputs. If it makes a small error at step $t$, that error feeds into step $t-1$. These errors accumulate, causing the generation to drift off the manifold of valid images. This discrepancy between training (perfect inputs) and inference (imperfect inputs) is called **Exposure Bias**.

**Scheduled Sampling** (often called "Rollout" in this context) addresses this by training the model on its own generated outputs.

## How it works

Instead of simply adding noise to a clean image, the training loop occasionally performs a mini-inference session:

1.  Pick a **target timestep** $t$ (the step we want to train on).
2.  Pick a **source timestep** $t+k$ (a noisier step further back in the schedule).
3.  Use the model's *current* weights to actually generate (denoise) from $t+k$ down to $t$.
4.  Use this self-generated, slightly imperfect latent at step $t$ as the input for the training pass.

By doing this, the model sees inputs that contain the exact kind of artifacts and errors it currently produces. It learns to say, "Ah, I made this mistake, here is how I correct it," effectively pulling the generation back onto the valid path.

## Configuration

This feature is experimental and adds computational overhead, but it can significantly improve prompt adherence and structural stability, especially on small datasets (Dreambooth).

To enable it, you must configure a non-zero `max_step_offset`.

### Basic Setup

Add the following to your `config.json`:

```json
{
  "scheduled_sampling_max_step_offset": 10,
  "scheduled_sampling_probability": 1.0,
  "scheduled_sampling_sampler": "unipc"
}
```

### Options Reference

#### `scheduled_sampling_max_step_offset` (Integer)
**Default:** `0` (Disabled)
The maximum number of steps to roll out. If this is set to `10`, the trainer will pick a random rollout length between 0 and 10 for each sample.
> 🟢 **Recommendation:** Start small (e.g., `5` to `10`). Even short rollouts help the model learn error correction without drastically slowing down training.

#### `scheduled_sampling_probability` (Float)
**Default:** `0.0`
The chance (0.0 to 1.0) that any given batch item will undergo rollout.
*   `1.0`: Every sample is rolled out (heaviest compute).
*   `0.5`: 50% of samples are standard training, 50% are rollout.

#### `scheduled_sampling_ramp_steps` (Integer)
**Default:** `0`
If set, the probability will linearly ramp from `scheduled_sampling_prob_start` (default 0.0) to `scheduled_sampling_prob_end` (default 0.5) over this many global steps.
> 🟢 **Tip:** This acts as a "warmup". It lets the model learn basic denoising first before introducing the harder task of fixing its own errors.

#### `scheduled_sampling_sampler` (String)
**Default:** `unipc`
The solver used for the rollout generation steps.
*   **Choices:** `unipc` (recommended, fast & accurate), `euler`, `dpm`, `rk4`.
*   `unipc` is generally the best trade-off between speed and accuracy for these short sampling bursts.

### Flow Matching + ReflexFlow

For flow-matching models (`--prediction_type flow_matching`), scheduled sampling now supports ReflexFlow-style exposure bias mitigation:

*   `scheduled_sampling_reflexflow`: Enable ReflexFlow enhancements during rollout (auto-enabled for flow-matching models when scheduled sampling is active; pass `--scheduled_sampling_reflexflow=false` to opt out).
*   `scheduled_sampling_reflexflow_alpha`: Scale the exposure-bias-based loss weight (frequency compensation).
*   `scheduled_sampling_reflexflow_beta1`: Scale the directional anti-drift regularizer (default 10.0 to mirror the paper).
*   `scheduled_sampling_reflexflow_beta2`: Scale the frequency-compensated loss (default 1.0).

These reuse the rollout predictions/latents you already compute, avoiding an extra gradient pass, and help keep biased rollouts aligned with the clean trajectory while emphasizing missing low-frequency components early in denoising.

### Performance Impact

> ⚠️ **Warning:** Enabling rollout requires running the model in inference mode *inside* the training loop.
>
> If you set `max_step_offset=10`, the model might run up to 10 extra forward passes per training step. This will reduce your `it/s` (iterations per second). Adjust `scheduled_sampling_probability` to balance training speed vs. quality gains.
