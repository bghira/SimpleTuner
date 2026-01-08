# SDXL LCM 蒸馏快速入门（SimpleTuner）

在本示例中，我们将使用 **LCM（Latent Consistency Model）蒸馏** 从预训练 SDXL 教师模型训练 **4-8 步的 SDXL 学生**。

> **注记**：其他模型也可作为基础，这里仅以 SDXL 说明 LCM 的配置概念。

LCM 可以实现：
* 超快速推理（4-8 步 vs 25-50 步）
* 跨时间步一致性
* 用极少步数输出高质量结果

## 📦 安装

请按照标准的 SimpleTuner 安装[指南](../INSTALL.md)：

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.12 -m venv .venv && source .venv/bin/activate

# Install with automatic platform detection
pip install -e .
```

**注记：** setup.py 会自动检测平台（CUDA/ROCm/Apple）并安装相应依赖。

容器环境（Vast、RunPod 等）：
```bash
apt -y install nvidia-cuda-toolkit
```

---

## 📁 配置

为 SDXL LCM 创建 `config/config.json`：

```json
{
  "model_type": "lora",
  "model_family": "sdxl",
  "output_dir": "/home/user/output/sdxl-lcm",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",

  "distillation_method": "lcm",
  "distillation_config": {
    "lcm": {
      "num_ddim_timesteps": 50,
      "w_min": 1.0,
      "w_max": 12.0,
      "loss_type": "l2",
      "huber_c": 0.001,
      "timestep_scaling_factor": 10.0
    }
  },

  "resolution": 1024,
  "resolution_type": "pixel",
  "validation_resolution": "1024x1024,1280x768,768x1280",
  "aspect_bucket_rounding": 64,
  "minimum_image_size": 0.5,
  "maximum_image_size": 1.0,

  "learning_rate": 1e-4,
  "lr_scheduler": "constant_with_warmup",
  "lr_warmup_steps": 1000,
  "max_train_steps": 10000,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "gradient_checkpointing": true,
  "mixed_precision": "bf16",

  "lora_type": "standard",
  "lora_rank": 64,
  "lora_alpha": 64,
  "lora_dropout": 0.0,

  "validation_step_interval": 250,
  "validation_num_inference_steps": 4,
  "validation_guidance": 0.0,
  "validation_prompt": "A portrait of a woman with flowers in her hair, highly detailed, professional photography",
  "validation_negative_prompt": "blurry, low quality, distorted, amateur",

  "checkpoint_step_interval": 500,
  "checkpoints_total_limit": 5,
  "resume_from_checkpoint": "latest",

  "optimizer": "adamw_bf16",
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_weight_decay": 1e-2,
  "adam_epsilon": 1e-8,
  "max_grad_norm": 1.0,

  "seed": 42,
  "push_to_hub": true,
  "hub_model_id": "your-username/sdxl-lcm-distilled",
  "report_to": "wandb",
  "tracker_project_name": "sdxl-lcm-distillation",
  "tracker_run_name": "sdxl-lcm-4step"
}
```

### 关键 LCM 配置选项：

- **`num_ddim_timesteps`**：DDIM 求解器时间步数（通常 50-100）
- **`w_min/w_max`**：训练时 guidance scale 范围（SDXL 为 1.0-12.0）
- **`loss_type`**：使用 "l2" 或 "huber"（huber 对离群值更稳健）
- **`timestep_scaling_factor`**：边界条件缩放（默认 10.0）
- **`validation_num_inference_steps`**：用目标步数测试（4-8）
- **`validation_guidance`**：LCM 设为 0.0（推理时不使用 CFG）

### 量化训练（降低显存占用）：

添加以下选项以减少内存使用：
```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "optimizer": "optimi-lion"
}
```

---

## 🎬 数据集配置

在输出目录创建 `multidatabackend.json`：

```json
[
  {
    "id": "your-dataset-name",
    "type": "local",
    "crop": false,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1.0,
    "minimum_image_size": 0.5,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/sdxl/your-dataset",
    "instance_data_dir": "/path/to/your/dataset",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sdxl/your-dataset",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> **重要**：LCM 蒸馏需要多样且高质量的数据。建议至少 1 万张以上以获得良好效果。

---

## 🚀 训练

1. **登录服务**（使用 Hub 功能时）：
   ```bash
   huggingface-cli login
   wandb login
   ```

2. **开始训练**：
   ```bash
   bash train.sh
   ```

3. **监控进度**：
   - LCM loss 应持续下降
   - 4-8 步时验证图像应保持质量
   - 训练通常需要 5k-10k 步

---

## 📊 预期结果

| 指标 | 预期值 | 说明 |
| ------ | -------------- | ----- |
| LCM Loss | < 0.1 | 应稳定下降 |
| 验证质量 | 4 步表现良好 | 可能需要 guidance=0 |
| 训练时间 | 5-10 小时 | 单张 A100 |
| 最终推理 | 4-8 步 | 基础 SDXL 为 25-50 |

---

## 🧩 排障

| 问题 | 解决方案 |
| ------- | -------- |
| **OOM 错误** | 降低 batch size，启用梯度检查点，使用 int8 量化 |
| **输出模糊** | 增加 `num_ddim_timesteps`，检查数据质量，降低学习率 |
| **收敛慢** | 将学习率提高到 2e-4，确保数据集多样性 |
| **验证效果差** | 使用 `validation_guidance: 0.0`，检查是否使用正确调度器 |
| **低步数伪影** | <4 步常见，尝试训练更久或调整 `w_min/w_max` |

---

## 🔧 高级建议

1. **多分辨率训练**：SDXL 受益于多种长宽比训练：
   ```json
   "validation_resolution": "1024x1024,1280x768,768x1280,1152x896,896x1152"
   ```

2. **渐进式训练**：先用较多步数，再逐步减少：
   - Week 1：使用 `validation_num_inference_steps: 8`
   - Week 2：使用 `validation_num_inference_steps: 4` 微调

3. **推理调度器**：训练后使用 LCM 调度器：
   ```python
   from diffusers import LCMScheduler
   scheduler = LCMScheduler.from_pretrained(
       "stabilityai/stable-diffusion-xl-base-1.0",
       subfolder="scheduler"
   )
   ```

4. **与 ControlNet 结合**：LCM 在低步数下与 ControlNet 引导生成效果很好。

---

## 📚 额外资源

- [LCM 论文](https://arxiv.org/abs/2310.04378)
- [Diffusers LCM 文档](https://huggingface.co/docs/diffusers/using-diffusers/inference_with_lcm)
- [更多 SimpleTuner 文档](../quickstart/SDXL.md)

---

## 🎯 下一步

LCM 蒸馏成功后：
1. 用不同提示词测试 4-8 步的模型
2. 在不同底座模型上尝试 LCM-LoRA
3. 针对特定场景尝试更少步数（2-3）
4. 结合特定领域数据进行微调
