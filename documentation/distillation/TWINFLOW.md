# TwinFlow (RCGM) Few-Step Training

TwinFlow is a lightweight, standalone few-step recipe built around **recursive consistency gradient matching (RCGM)**. It is **not part of the main `distillation_method` options**—you opt in directly via `twinflow_*` flags. The loader defaults `twinflow_enabled` to `false` on configs pulled from the hub so vanilla transformer configs stay untouched.

TwinFlow in SimpleTuner:
* Flow-matching only unless you explicitly bridge diffusion models with `diff2flow_enabled` + `twinflow_allow_diff2flow`.
* EMA teacher by default; RNG capture/restore is **always on** around teacher/CFG passes to mirror the reference TwinFlow run.
* Optional sign embeddings for negative-time semantics are wired on transformers but only used when `twinflow_enabled` is true; HF configs with no flag avoid any behavior change.
* Default losses use RCGM + real-velocity; optionally enable full self-adversarial training with `twinflow_adversarial_enabled: true` for L_adv and L_rectify losses. Expects 1–4 step generation at guidance `0.0`.
* W&B logging can emit an experimental TwinFlow trajectory scatter (theory noted as unverified) for debugging.

---

## Quick Config (flow-matching model)

Add the TwinFlow bits to your usual config (leave `distillation_method` unset/null):

```json
{
  "model_family": "sd3",
  "model_type": "lora",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
  "output_dir": "output/sd3-twinflow",

  "distillation_method": null,
  "use_ema": true,

  "twinflow_enabled": true,
  "twinflow_target_step_count": 2,
  "twinflow_estimate_order": 2,
  "twinflow_enhanced_ratio": 0.5,
  "twinflow_delta_t": 0.01,
  "twinflow_target_clamp": 1.0,

  "learning_rate": 1e-4,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "mixed_precision": "bf16",
  "validation_guidance": 0.0,
  "validation_num_inference_steps": 2
}
```

For diffusion models (epsilon/v prediction) opt in explicitly:

```json
{
  "prediction_type": "epsilon",
  "diff2flow_enabled": true,
  "twinflow_allow_diff2flow": true
}
```

> By default, TwinFlow uses RCGM + real-velocity losses. Enable `twinflow_adversarial_enabled: true` for full self-adversarial training with L_adv and L_rectify losses (no external discriminator needed).

---

## What to expect (paper data)

From arXiv:2512.05150 (PDF text):
* Inference benchmarks were measured on a **single A100 (BF16)** with throughput (batch=10) and latency (batch=1) at 1024×1024. Exact numbers were not in the text, just the hardware setting.
* A **GPU memory comparison** (1024×1024) for Qwen-Image-20B (LoRA) and SANA-1.6B shows TwinFlow fitting where DMD2 / SANA-Sprint can OOM.
* Training configs (Table 6) list **batch sizes 128/64/32/24** and **training steps 30k–60k (or 7k–10k shorter runs)**; constant LR, EMA decay often 0.99.
* The PDF does **not** report total GPU counts, node layouts, or wall-clock time.

Treat these as directional expectations, not guarantees. For exact hardware/runtime, you’d need author confirmation.

---

## Key Options

* `twinflow_enabled`: Turns on the RCGM auxiliary loss; keep `distillation_method` empty and scheduled sampling disabled. Defaults to `false` if missing from the config.
* `twinflow_target_step_count` (1–4 recommended): Guides training and is reused for validation/inference. Guidance is forced to `0.0` because CFG is baked in.
* `twinflow_estimate_order`: Integration order for the RCGM rollout (default 2). Higher values add teacher passes.
* `twinflow_enhanced_ratio`: Optional CFG-style target refinement from teacher cond/uncond predictions (0.5 default; set 0.0 to disable). Uses captured RNG so cond/uncond stay aligned.
* `twinflow_delta_t` / `twinflow_target_clamp`: Shape the recursive target; defaults mirror the paper’s stable settings.
* `use_ema` + `twinflow_require_ema` (default true): EMA weights are used as the teacher. Set `twinflow_allow_no_ema_teacher: true` only if you accept student-as-teacher quality.
* `twinflow_allow_diff2flow`: Enables bridging epsilon/v-prediction models when `diff2flow_enabled` is also true.
* RNG capture/restore: Always enabled to mirror the reference TwinFlow implementation for consistent teacher/CFG passes. There is no opt-out switch.
* Sign embeddings: When `twinflow_enabled` is true, models pass `twinflow_time_sign` into transformers that support `timestep_sign`; otherwise no extra embedding is used.

### Adversarial Branch (Full TwinFlow)

Enable the self-adversarial training from the original paper for improved quality:

* `twinflow_adversarial_enabled` (default false): Enable L_adv and L_rectify losses. These use negative time to train a "fake" trajectory, enabling distribution matching without external discriminators.
* `twinflow_adversarial_weight` (default 1.0): Weight multiplier for the adversarial loss (L_adv).
* `twinflow_rectify_weight` (default 1.0): Weight multiplier for the rectification loss (L_rectify).

When enabled, training generates fake samples via one-step generation, then trains both:
- **L_adv**: Fake velocity loss with negative time—teaches the model to map fake samples back to noise.
- **L_rectify**: Distribution matching loss—aligns real and fake trajectory predictions for straighter paths.

---

## Training & Validation Flow

1. Train as you would a normal flow-matching run (no distiller needed). EMA must exist unless you explicitly opt out; RNG alignment is automatic.
2. Validation automatically swaps in the **TwinFlow/UCGM scheduler** and uses `twinflow_target_step_count` steps with `guidance_scale=0.0`.
3. For exported pipelines, attach the scheduler manually:

```python
from simpletuner.helpers.training.custom_schedule import TwinFlowScheduler

pipe = ...  # your loaded diffusers pipeline
pipe.scheduler = TwinFlowScheduler(num_train_timesteps=1000, prediction_type="flow_matching", shift=1.0)
pipe.scheduler.set_timesteps(num_inference_steps=2, device=pipe.device)
result = pipe(prompt="A cinematic portrait, 35mm", guidance_scale=0.0, num_inference_steps=2).images
```

---

## Logging

* When `report_to=wandb` and `twinflow_enabled=true`, the trainer can log an experimental TwinFlow trajectory scatter (σ vs tt vs sign). The visual is for debugging only and is tagged in the UI as “experimental/theory unverified”.

---

## Troubleshooting

* **Error about flow-matching**: TwinFlow requires `prediction_type=flow_matching` unless you enable `diff2flow_enabled` + `twinflow_allow_diff2flow`.
* **EMA required**: Enable `use_ema` or set `twinflow_allow_no_ema_teacher: true` / `twinflow_require_ema: false` if you accept student-teacher fallback.
* **Quality flat at 1 step**: Try `twinflow_target_step_count: 2`–`4`, keep guidance at `0.0`, and reduce `twinflow_enhanced_ratio` if overfitting.
* **Teacher/Student drift**: RNG alignment is always enabled; drift should come from model mismatch, not stochastic differences. If your transformer lacks `timestep_sign`, leave `twinflow_enabled` off or update the model to consume it before enabling TwinFlow.
