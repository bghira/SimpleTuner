# DCM 蒸馏快速入门（SimpleTuner）

在本示例中，我们将从大型 flow-matching 教师模型（如 [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)）使用 **DCM 蒸馏**训练 **4 步学生**。

DCM 支持：

* **Semantic** 模式：带 CFG 的标准 flow-matching。
* **Fine** 模式：可选的基于 GAN 的对抗监督（实验性）。

---

## ✅ 硬件要求

| 模型     | Batch Size | 最低显存 | 备注                                  |
| --------- | ---------- | -------- | -------------------------------------- |
| Wan 1.3B  | 1          | 12 GB    | A5000 / 3090+ 级别 GPU                 |
| Wan 14B   | 1          | 24 GB    | 更慢；使用 `--offload_during_startup` |
| Fine 模式 | 1          | +10%     | 判别器按 GPU 运行                      |

> ⚠️ Mac 和 Apple silicon 很慢，不推荐。即使在 semantic 模式下也可能达到 10 分钟/步。

---

## 📦 安装

与 Wan 指南相同步骤：

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv && source .venv/bin/activate

# Install with automatic platform detection
pip install -e .
```

**注记：** setup.py 会自动检测平台（CUDA/ROCm/Apple）并安装合适依赖。

---

## 📁 配置

编辑 `config/config.json`：

```json
{
    "aspect_bucket_rounding": 2,
    "attention_mechanism": "diffusers",
    "base_model_precision": "int8-quanto",
    "caption_dropout_probability": 0.1,
    "checkpoint_step_interval": 100,
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
    "validation_step_interval": 4,
    "webhook_config": "config/wan/webhook.json"
}
```

### 可选：

* 使用 **fine 模式** 时，将 `"mode": "fine"`。
  - 该模式在 SimpleTuner 中仍处于实验阶段，并需要一些额外步骤才能使用，本指南暂未覆盖。

---

## 🎬 数据集与数据加载器

复用 Wan 快速入门中的 Disney 数据集与 `data_backend_config` JSON。

> **注记**：该数据集不适合蒸馏，成功需要 **更大规模且多样** 的数据。

请确保：

* `num_frames`: 75–81
* `resolution`: 480
* `crop`: false（不要裁剪视频）
* `repeats`: 0（暂时）

---

## 📌 说明

* **Semantic 模式** 稳定，适用于多数场景。
* **Fine 模式** 增强真实感，但需要更多步数和调参，且 SimpleTuner 当前支持程度有限。

---

## 🧩 排障

| 问题                      | 解决办法                                                                  |
| ---------------------------- | -------------------------------------------------------------------- |
| **结果模糊**       | 增加 euler_steps，或提高 `multiphase`                       |
| **验证退化**  | 使用 `validation_guidance: 1.0`                                       |
| **fine 模式 OOM**         | 降低 `train_batch_size`，降低精度，或使用更大的 GPU |
| **fine 模式不收敛** | 不要使用 fine 模式，SimpleTuner 中尚未充分测试      |
