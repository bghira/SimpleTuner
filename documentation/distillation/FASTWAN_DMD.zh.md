# DMD 蒸馏快速入门（SimpleTuner）

在本示例中，我们将使用 **DMD（Distribution Matching Distillation）** 从大型 flow-matching 教师模型（例如 [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)）训练 **3 步学生**。

DMD 特性：

* **Generator（Student）**：学习以更少步数匹配教师
* **Fake Score Transformer**：区分教师与学生输出
* **多步模拟**：可选的训练-推理一致性模式

---

## ✅ 硬件要求


⚠️ DMD 由于 Fake Score Transformer 需要在内存中保留第二份完整基模，因此显存压力很大。

如果显存不足，建议对 14B 的 Wan 模型尝试 LCM 或 DCM 蒸馏方法，而不是 DMD。

在没有稀疏注意力支持的情况下蒸馏 14B 模型，可能需要 NVIDIA B200。

使用 LoRA 学生训练可显著降低需求，但仍然很重。

---

## 📦 安装

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.12 -m venv .venv && source .venv/bin/activate

# Install with automatic platform detection
pip install -e .
```

**注记：** setup.py 会自动检测你的平台（CUDA/ROCm/Apple）并安装相应依赖。

---

## 📁 配置

编辑 `config/config.json`：

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

### 关键 DMD 设置：

* **`dmd_denoising_steps`** – 反向模拟的目标时间步（3 步学生默认 `1000,757,522`）。
* **`generator_update_interval`** – 每 _N_ 个训练步进行一次昂贵的 generator 重放。增大可换速度但牺牲质量。
* **`fake_score_lr` / `fake_score_weight_decay` / `fake_score_betas`** – Fake Score Transformer 的优化器超参。
* **`fake_score_guidance_scale`** – Fake Score 网络的 classifier-free guidance（默认关闭）。
* **`num_frame_per_block`, `same_step_across_blocks`, `last_step_only`** – 控制 self-forcing rollout 时的时间块调度。
* **`num_training_frames`** – 反向模拟中的最大生成帧数（更大更逼真但更耗内存）。
* **`min_timestep_ratio`, `max_timestep_ratio`, `timestep_shift`** – 控制 KL 采样窗口。若偏离默认值，请与教师的 flow schedule 对齐。

---

## 🎬 数据集与数据加载器

要让 DMD 效果良好，需要 **多样且高质量的数据**：

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

> **注记**：Disney 数据集对 DMD **不够用**。**不要使用！** 这里只是示意。

你需要：
> - 高容量（至少 1 万条视频）
> - 多样内容（风格、动作、主体不同）
> - 高质量（无压缩伪影）

这些可以由母模型生成。

---

## 🚀 训练建议

1. **保持较小的 generator 间隔**：从 `"generator_update_interval": 1` 开始。只有在需要吞吐量且可接受更噪更新时再提高。
2. **监控两种损失**：在 wandb 中查看 `dmd_loss` 与 `fake_score_loss`
3. **验证频率**：DMD 收敛快，建议常验证
4. **内存管理**：
   - 使用 `gradient_checkpointing`
   - 将 `train_batch_size` 降到 1
   - 考虑 `base_model_precision: "int8-quanto"`

---

## 📌 DMD vs DCM

| 特性 | DCM | DMD |
|---------|-----|-----|
| 内存使用 | 更低 | 更高（fake score 模型） |
| 训练时间 | 更长 | 更短（通常 4k 步） |
| 质量 | 良好 | 优秀 |
| 推理步数 | 4-8+ | 3-8 |
| 稳定性 | 稳定 | 需要调参 |

---

## 🧩 排障

| 问题 | 解决办法 |
|---------|-----|
| **OOM 错误** | 减少 `num_training_frames`，降低 `generator_update_interval`，或减少 batch size |
| **Fake score 不学习** | 提高 `fake_score_lr` 或使用不同调度器 |
| **Generator 过拟合** | 将 `generator_update_interval` 增加到 10 |
| **3 步质量差** | 先尝试 2 步的 "1000,500" |
| **训练不稳定** | 降低学习率，检查数据质量 |

---

## 🔬 高级选项

给想尝试的人：

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

> ⚠️ 对资源受限项目，建议使用原始 FastVideo 的 DMD 实现，它支持序列并行与视频稀疏注意力（VSA），运行效率更高。
