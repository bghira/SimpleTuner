# Diff2Flow (Diffusion-to-Flow Bridge)

## Background

Historically, diffusion models have been categorized by their prediction targets:
*   **Epsilon ($\epsilon$):** Predict the noise added to the image (SD 1.5, SDXL).
*   **V-Prediction ($v$):** Predict a velocity combining noise and data (SD 2.0, SDXL Refiner).

Newer state-of-the-art models like **Flux**, **Stable Diffusion 3**, and **AuraFlow** utilize **Flow Matching** (specifically Rectified Flow). Flow Matching treats the generation process as an Ordinary Differential Equation (ODE) that moves particles from a noise distribution to a data distribution along straight paths.

This straight-line trajectory is generally easier for solvers to step through, allowing for fewer steps and more stable generation.

## The Bridge

**Diff2Flow** is a lightweight adapter that allows "Legacy" models (Epsilon or V-pred) to be trained using a Flow Matching objective without changing their underlying architecture.

It works by mathematically converting the model's native output (e.g., an epsilon prediction) into a flow vector field $u_t(x|1)$, and then computing the loss against the flow target ($x_1 - x_0$, or `noise - latents`).

> 🟡 **Experimental Status:** This feature effectively changes the loss landscape the model sees. While theoretically sound, it significantly alters training dynamics. It is primarily intended for research and experimentation.

## Configuration

To use Diff2Flow, you need to enable the bridge and optionally switch the loss function.

### Basic Setup

Add these keys to your `config.json`:

```json
{
  "diff2flow_enabled": true,
  "diff2flow_loss": true
}
```

### Options Reference

#### `--diff2flow_enabled` (Boolean)
**Default:** `false`
Initializes the mathematical bridge. This allocates a small buffer for timestep calculations but does not change training behavior on its own unless `diff2flow_loss` is also set.
*   **Required for:** `diff2flow_loss`.
*   **Supported Models:** Any model using `epsilon` or `v_prediction` (SD1.5, SD2.x, SDXL, DeepFloyd IF, PixArt Alpha).

#### `--diff2flow_loss` (Boolean)
**Default:** `false`
Switches the training objective.
*   **False:** The model minimizes the error between its prediction and the standard target (e.g., `MSE(pred_noise, real_noise)`).
*   **True:** The model minimizes the error between the *flow-converted* prediction and the flow target (`noise - latents`).

### Synergies

Diff2Flow pairs extremely well with **Scheduled Sampling**.

When you combine:
1.  **Diff2Flow** (Straightening the trajectories)
2.  **Scheduled Sampling** (Training on self-generated rollouts)

You effectively approximate the training recipe used for **Reflow** or **Rectified Flow** models, potentially imparting modern stability and quality traits onto older architectures like SDXL.
