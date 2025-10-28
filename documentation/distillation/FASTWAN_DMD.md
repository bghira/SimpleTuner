# DMD Distillation Quickstart (SimpleTuner)

In this example, we'll be training a **3-step student** using **DMD (Distribution Matching Distillation)** from a large flow-matching teacher model like [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).

DMD features:

* **Generator (Student)**: Learns to match teacher in fewer steps
* **Fake Score Transformer**: Discriminates between teacher and student outputs
* **Multi-step simulation**: Optional train-inference consistency mode

---

## ✅ Hardware Requirements


⚠️ DMD is memory-intensive due to the fake score transformer which requires a complete second copy of the base model be retained in-memory.

It's recommended to attempt LCM or DCM distillation methods for the 14B Wan model instead of DMD, if you do not have the needed VRAM.

An NVIDIA B200 may be required when distilling the 14B model without sparse attention support.

Using LoRA student training can reduce the requirements substantially, but still quite hefty.

---

## 📦 Installation

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.12 -m venv .venv && source .venv/bin/activate

# Install with automatic platform detection
pip install -e .
```

**Note:** The setup.py automatically detects your platform (CUDA/ROCm/Apple) and installs the appropriate dependencies.

---

## 📁 Configuration

Edit your `config/config.json`:

```json
{
    "aspect_bucket_rounding": 2,
    "attention_mechanism": "diffusers",
    "base_model_precision": "int8-quanto",
    "caption_dropout_probability": 0.1,
    "checkpoint_step_interval": 200,
    "checkpoints_total_limit": 3,
    "compress_disk_cache": true,
    "data_backend_config": "config/wan/multidatabackend.json",
    "delete_problematic_images": false,
    "disable_benchmark": false,
    "disable_bucket_pruning": true,
    "distillation_method": "dmd",
    "distillation_config": {
        "dmd_denoising_steps": "1000,757,522",
        "generator_update_interval": 1,
        "real_score_guidance_scale": 3.0,
        "fake_score_lr": 1e-5,
        "fake_score_weight_decay": 0.01,
        "fake_score_betas": [0.9, 0.999],
        "fake_score_eps": 1e-8,
        "fake_score_grad_clip": 1.0,
        "fake_score_guidance_scale": 0.0,
        "min_timestep_ratio": 0.02,
        "max_timestep_ratio": 0.98,
        "num_frame_per_block": 3,
        "independent_first_frame": false,
        "same_step_across_blocks": false,
        "last_step_only": false,
        "num_training_frames": 21,
        "context_noise": 0,
        "ts_schedule": true,
        "ts_schedule_max": false,
        "min_score_timestep": 0,
        "timestep_shift": 1.0
    },
    "ema_update_interval": 5,
    "ema_validation": "ema_only",
    "flow_schedule_shift": 5,
    "grad_clip_method": "value",
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": true,
    "hub_model_id": "wan-disney-DMD-3step",
    "ignore_final_epochs": true,
    "learning_rate": 2e-5,
    "lora_alpha": 128,
    "lora_rank": 128,
    "lora_type": "standard",
    "lr_scheduler": "cosine_with_min_lr",
    "lr_warmup_steps": 100,
    "max_grad_norm": 1.0,
    "max_train_steps": 4000,
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "wan",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/wan-dmd",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "quantize_via": "cpu",
    "report_to": "wandb",
    "resolution": 480,
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 1000,
    "text_encoder_1_precision": "int8-quanto",
    "tracker_project_name": "dmd-training",
    "tracker_run_name": "wan-DMD-3step",
    "train_batch_size": 1,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_guidance": 1.0,
    "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "validation_num_inference_steps": 3,
    "validation_num_video_frames": 121,
    "validation_prompt": "A black and white animated scene unfolds featuring a distressed upright cow with prominent horns and expressive eyes, suspended by its legs from a hook on a static background wall. A smaller Mickey Mouse-like character enters, standing near a wooden bench, initiating interaction between the two. The cow's posture changes as it leans, stretches, and falls, while the mouse watches with a concerned expression, its face a mixture of curiosity and worry, in a world devoid of color.",
    "validation_prompt_library": "config/wan/validation_prompts_dmd.json",
    "validation_resolution": "1280x704",
    "validation_seed": 42,
    "validation_step_interval": 200,
    "webhook_config": "config/wan/webhook.json"
}
```

### Key DMD Settings:

* **`dmd_denoising_steps`** – Target timesteps for the backward simulation (default: `1000,757,522` for a 3-step student).
* **`generator_update_interval`** – Run the expensive generator replay every _N_ trainer steps. Increase to trade sample quality for speed.
* **`fake_score_lr` / `fake_score_weight_decay` / `fake_score_betas`** – Optimiser hyperparameters for the fake score transformer.
* **`fake_score_guidance_scale`** – Optional classifier-free guidance on the fake score network (defaults to off).
* **`num_frame_per_block`, `same_step_across_blocks`, `last_step_only`** – Control how temporal blocks are scheduled during self-forcing rollout.
* **`num_training_frames`** – Maximum frames generated during the backward simulation (larger values improve fidelity at a memory cost).
* **`min_timestep_ratio`, `max_timestep_ratio`, `timestep_shift`** – Shape the KL sampling window. Match these with your teacher’s flow schedule if you deviate from defaults.

---

## 🎬 Dataset & Dataloader

For DMD to work well, you need **diverse, high-quality data**:

```json
{
  "dataset_type": "video",
  "cache_dir": "cache/wan-dmd",
  "resolution_type": "pixel_area",
  "crop": false,
  "num_frames": 121,
  "frame_interval": 1,
  "resolution": 480,
  "minimum_image_size": 0,
  "repeats": 0
}
```

> **Note**: The Disney dataset is **inadequate** for DMD. **DON'T use it!** It's provided merely for illustrative purposes.

You need:
> - High volume (10k+ videos minimum)
> - Diverse content (different styles, motions, subjects)
> - High quality (no compression artifacts)

These may be generated from the parent model.

---

## 🚀 Training Tips

1. **Keep generator interval small**: Start with `"generator_update_interval": 1`. Increase only if you need throughput and can tolerate noisier updates.
2. **Monitor both losses**: Watch `dmd_loss` and `fake_score_loss` in wandb
3. **Validation frequency**: DMD converges quickly, validate often
4. **Memory management**:
   - Use `gradient_checkpointing`
   - Lower `train_batch_size` to 1
   - Consider `base_model_precision: "int8-quanto"`

---

## 📌 DMD vs DCM

| Feature | DCM | DMD |
|---------|-----|-----|
| Memory usage | Lower | Higher (fake score model) |
| Training time | Longer | Shorter (4k steps typical) |
| Quality | Good | Excellent |
| Inference steps | 4-8+ | 3-8 |
| Stability | Stable | Requires tuning |

---

## 🧩 Troubleshooting

| Problem | Fix |
|---------|-----|
| **OOM errors** | Reduce `num_training_frames`, drop `generator_update_interval`, or lower batch size |
| **Fake score not learning** | Increase `fake_score_lr` or use different scheduler |
| **Generator overfitting** | Increase `generator_update_interval` to 10 |
| **Poor 3-step quality** | Try "1000,500" for 2-step first |
| **Training unstable** | Lower learning rates, check data quality |

---

## 🔬 Advanced Options

For brave souls wanting to experiment:

```json
"distillation_config": {
    "dmd_denoising_steps": "1000,666,333",
    "generator_update_interval": 4,
    "fake_score_guidance_scale": 1.2,
    "num_training_frames": 28,
    "same_step_across_blocks": true,
    "timestep_shift": 7.0
}
```

> ⚠️ It's recommended to use the original FastVideo implementation of DMD for resource-constrained projects as it supports sequence-parallel and video-sparse attention (VSA) for far more efficient runtime usage.
