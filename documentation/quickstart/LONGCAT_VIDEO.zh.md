# LongCat‑Video 快速入门

LongCat‑Video 是一个 13.6B 双语（zh/en）文生视频/图生视频模型，使用流匹配、Qwen‑2.5‑VL 文本编码器和 Wan VAE。本指南带你完成 SimpleTuner 的设置、数据准备，以及首次训练/验证。

---

## 1) 硬件要求（参考）

- 13.6B Transformer + Wan VAE：比图像模型更吃 VRAM；建议从 `train_batch_size=1`、开启梯度检查点、较低 LoRA rank 开始。
- 系统内存：多帧剪辑建议 32GB 以上；数据集放在高速存储上。
- Apple MPS：支持预览；位置编码会自动降为 float32。

---

## 2) 前提条件

1. 确认 Python 3.12（SimpleTuner 默认会创建 `.venv`）：
   ```bash
   python --version
   ```
2. 按硬件选择 extras 安装 SimpleTuner：
   ```bash
   pip install "simpletuner[cuda]"   # NVIDIA
   pip install "simpletuner[mps]"    # Apple Silicon
   pip install "simpletuner[cpu]"    # CPU-only
   ```
3. 量化已内置（`int8-quanto`, `int4-quanto`, `fp8-torchao`），通常无需额外手动安装。

---

## 3) 环境设置

### Web UI
```bash
simpletuner server
```
打开 http://localhost:8001 并选择模型家族 `longcat_video`。

### CLI 基础配置（config/config.json）

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_video",
  "model_flavour": "final",
  "pretrained_model_name_or_path": null,      // auto-selected from flavour
  "base_model_precision": "bf16",             // int8-quanto/fp8-torchao also work for LoRA
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "learning_rate": 1e-4,
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0
}
```

**需保留的关键默认值**
- shift `12.0` 的流匹配调度器是自动的；无需额外噪声参数。
- 宽高比桶保持 64px 对齐；`aspect_bucket_alignment` 强制为 64。
- 最大 token 长度 512（Qwen‑2.5‑VL）；CFG 开启且未给负向提示词时，管线会自动补一个空负向。
- 帧数必须满足 `(num_frames - 1)` 可被 VAE 时间步长（默认 4）整除。默认 93 帧已满足该条件。

可选 VRAM 节省项：
- 降低 `lora_rank`（4–8）并使用 `int8-quanto` 基础精度。
- 启用 group offload：`--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`。
- 预览 OOM 时优先降低验证分辨率/帧数/步数。
- 注意力默认行为：CUDA 上 LongCat‑Video 会在可用时自动使用随附的 block‑sparse Triton kernel，缺失则回退标准调度器。无需手动切换。若想强制 xFormers，在配置/CLI 中设置 `attention_implementation: "xformers"`。

### 开始训练（CLI）
```bash
simpletuner train --config config/config.json
```
或打开 Web UI，用相同配置提交任务。

---

## 4) 数据加载器指导

- 使用带字幕的视频数据集；每个样本应包含帧（或短片段）与文本字幕。`dataset_type: video` 会通过 `VideoToTensor` 自动处理。
- 帧尺寸保持 64px 网格（如 480x832、720p 桶）。高/宽必须同时可被 Wan VAE 步长（内置设置为 16px）以及 64 整除。
- 图生视频训练需为每个样本提供条件图像；它会被放在首个 latent 帧，并在采样中保持固定。
- LongCat‑Video 设计为 30 fps。默认 93 帧约为 3.1 秒；如更改帧数，请保持 `(frames - 1) % 4 == 0`，同时注意时长会随 fps 变化。

### 视频分桶策略

在数据集的 `video` 段中可设置分桶方式：
- `bucket_strategy`：`aspect_ratio`（默认）按空间宽高比分组；`resolution_frames` 按 `WxH@F`（如 `480x832@93`）分组，适合混合分辨率/时长的数据集。
- `frame_interval`：使用 `resolution_frames` 时，将帧数按该间隔取整（例如设为 4 以匹配 VAE 时间步长）。

---

## 5) 验证与推理

- Guidance：3.5–5.0 效果较好；CFG 开启时会自动补空负向提示词。
- 步数：质量检查 35–45；快速预览可更低。
- 帧数：默认 93（对齐 VAE 时间步长 4）。
- 需要更低 VRAM 预览/训练时，可设置 `musubi_blocks_to_swap`（建议 4–8）并视需要设置 `musubi_block_swap_device`，将最后的 Transformer block 从 CPU 流式加载。吞吐会下降，但峰值 VRAM 会降低。

- 验证从配置中的 `validation_*` 字段或 WebUI 预览标签页触发。需要快速检查时优先用这些路径，而不是单独的 CLI 子命令。
- 数据集驱动验证（含 I2V）需设置 `validation_using_datasets: true`，并将 `eval_dataset_id` 指向验证分割。如果该分割标记 `is_i2v` 且链接了条件帧，管线会自动固定首帧。
- latent 预览在解码前会先 unpack，避免通道不匹配。

---

## 6) 故障排查

- **高/宽错误**：确保均可被 16 整除且在 64px 网格上。
- **MPS float64 警告**：已在内部处理；精度保持 bf16/float32。
- **OOM**：优先降低验证分辨率/帧数，降低 LoRA rank，启用 group offload，或切换到 `int8-quanto`/`fp8-torchao`。
- **CFG 空负向**：未提供负向提示词时，管线会自动插入空值。

---

## 7) 版本

- `final`：LongCat‑Video 主版本（单一检查点同时支持文生视频与图生视频）。
