# DCM Distillation Quickstart (SimpleTuner)

In this example, we'll be training a **4-step student** using **DCM distillation** from a large flow-matching teacher model like [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).

DCM supports:

* **Semantic** mode: standard flow-matching with CFG baked in.
* **Fine** mode: optional GAN-based adversarial supervision (experimental).

---

## ✅ Hardware Requirements

| Model     | Batch Size | Min VRAM | Notes                                  |
| --------- | ---------- | -------- | -------------------------------------- |
| Wan 1.3B  | 1          | 12 GB    | A5000 / 3090+ tier GPU                 |
| Wan 14B   | 1          | 24 GB    | Slower; use `--offload_during_startup` |
| Fine mode | 1          | +10%     | Discriminator runs per-GPU             |

> ⚠️ Mac and Apple silicon are slow and not recommended. You'll get 10 min/step runtimes even in semantic mode.

---

## 📦 Installation

Same steps as the Wan guide:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U poetry pip
poetry config virtualenvs.create false
poetry install
```

> If you're on ROCm or Apple: you must instead use `poetry install -C install/variant` where `variant` is `rocm` or `apple`.

---

## 📁 Configuration

Edit your `config/config.json`:

```json
{
    "aspect_bucket_rounding": 2,
    "attention_mechanism": "diffusers",
    "base_model_precision": "int8-quanto",
    "caption_dropout_probability": 0.1,
    "checkpointing_steps": 100,
    "checkpoints_total_limit": 5,
    "compress_disk_cache": true,
    "data_backend_config": "config/wan/multidatabackend.json",
    "delete_problematic_images": false,
    "disable_benchmark": false,
    "disable_bucket_pruning": true,
    "distillation_method": "dcm",
    "distillation_config": {
      "mode": "semantic",
      "euler_steps": 100
    },
    "ema_update_interval": 2,
    "ema_validation": "ema_only",
    "flow_schedule_shift": 17,
    "grad_clip_method": "value",
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": true,
    "hub_model_id": "wan-disney-DCM-distilled",
    "ignore_final_epochs": true,
    "learning_rate": 1e-4,
    "lora_alpha": 128,
    "lora_rank": 128,
    "lora_type": "standard",
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 400000,
    "lycoris_config": "config/wan/lycoris_config.json",
    "max_grad_norm": 0.01,
    "max_train_steps": 400000,
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "wan",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/wan",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "prodigy_steps": 100000,
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "quantize_via": "cpu",
    "report_to": "wandb",
    "resolution": 480,
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 42,
    "text_encoder_1_precision": "int8-quanto",
    "tracker_project_name": "lora-training",
    "tracker_run_name": "wan-AdamW-DCM",
    "train_batch_size": 2,
    "use_ema": false,
    "vae_batch_size": 1,
    "validation_guidance": 1.0,
    "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "validation_num_inference_steps": 8,
    "validation_num_video_frames": 16,
    "validation_prompt": "A black and white animated scene unfolds featuring a distressed upright cow with prominent horns and expressive eyes, suspended by its legs from a hook on a static background wall. A smaller Mickey Mouse-like character enters, standing near a wooden bench, initiating interaction between the two. The cow's posture changes as it leans, stretches, and falls, while the mouse watches with a concerned expression, its face a mixture of curiosity and worry, in a world devoid of color.",
    "validation_prompt_library": false,
    "validation_resolution": "832x480",
    "validation_seed": 42,
    "validation_steps": 4,
    "webhook_config": "config/wan/webhook.json"
}
```

### Optional:

* For **fine mode**, just change `"mode": "fine"`.
  - This mode is currently experimental in SimpleTuner and requires some extra steps to make use of, which are not yet outlined in this guide.

---

## 🎬 Dataset & Dataloader

Reuse the Disney dataset and `data_backend_config` JSON from the Wan quickstart.

> **Note**: This dataset is inadequate for distillation, **much** more diverse and higher volume of data is required to succeed.

Make sure:

* `num_frames`: 75–81
* `resolution`: 480
* `crop`: false (leave videos uncropped)
* `repeats`: 0 for now

---

## 📌 Notes

* **Semantic mode** is stable and recommended for most use cases.
* **Fine mode** adds realism, but needs more steps and tuning and SimpleTuner's current support level of this isn't great.

---

## 🧩 Troubleshooting

| Problem                      | Fix                                                                  |
| ---------------------------- | -------------------------------------------------------------------- |
| **Results are blurry**       | Use more euler_steps, or increase `multiphase`                       |
| **Validation is degrading**  | Use `validation_guidance: 1.0`                                       |
| **OOM in fine mode**         | Lower `train_batch_size`, reduce precision levels, or use larger GPU |
| **Fine mode not converging** | Don't use fine mode, it is not super well-tested in SimpleTuner      |