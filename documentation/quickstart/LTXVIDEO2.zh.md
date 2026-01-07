# LTX Video 2 快速入门

本示例将使用 LTX-2 视频/音频 VAE 与 Gemma3 文本编码器训练 LTX Video 2 LoRA。

## 硬件要求

LTX Video 2 是重量级 **19B** 模型，由以下组件组成：
1.  **Gemma3**：文本编码器。
2.  **LTX-2 Video VAE**（音频条件时还会使用 Audio VAE）。
3.  **19B Video Transformer**：大型 DiT 主干。

该组合非常耗 VRAM，VAE 预缓存步骤可能会显著抬高内存峰值。

- **单 GPU 训练**：从 `train_batch_size: 1` 开始，并启用 group offload。
  - **注意**：初始 **VAE 预缓存步骤** 可能需要更多 VRAM。可能需要 CPU offload 或更大 GPU 仅用于缓存阶段。
  - **提示**：在 `config.json` 中设置 `"offload_during_startup": true`，确保 VAE 与文本编码器不会同时加载到 GPU，可显著降低预缓存压力。
- **多 GPU 训练**：若需要更大余量，推荐 **FSDP2** 或强力 **Group Offload**。
- **系统内存**：大规模训练建议 64GB+，更多内存有助于缓存。

### 内存卸载（关键）

多数单 GPU 训练 LTX Video 2 的场景都推荐启用分组卸载，以便为更大 batch/分辨率留出 VRAM 余量。

在 `config.json` 中添加：

<details>
<summary>查看示例配置</summary>

```json
{
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "group_offload_blocks_per_group": 1,
  "group_offload_use_stream": true
}
```
</details>

## 前提条件

确保已安装 Python 3.12。

```bash
python --version
```

## 安装

```bash
pip install simpletuner[cuda]
```

高级安装选项参见 [INSTALL.md](../INSTALL.md)。

## 设置环境

### Web 界面

```bash
simpletuner server
```

访问 http://localhost:8001。

### 手动配置

运行辅助脚本：

```bash
simpletuner configure
```

或复制示例并手动修改：

```bash
cp config/config.json.example config/config.json
```

#### 配置参数

LTX Video 2 的关键设置：

- `model_family`: `ltxvideo2`
- `model_flavour`: `2.0`（默认）
- `pretrained_model_name_or_path`: `Lightricks/LTX-2`（包含 combined checkpoint 的仓库）或本地 `.safetensors` 文件。
- `ltx2_checkpoint_filename`: 可选。当指向目录且 combined checkpoint 文件名不是 `ltx-2-19b-dev.safetensors` 时设置。
- `train_batch_size`: `1`。除非有 A100/H100，否则不要提高。
- `validation_resolution`:
  - `512x768` 是安全的测试默认值。
  - `720x1280`（720p）可行但较重。
- `validation_num_video_frames`: **必须与 VAE 压缩比例 (4x) 兼容。**
  - 5 秒（约 12-24fps）：使用 `61` 或 `49`。
  - 公式：`(frames - 1) % 4 == 0`。
- `validation_guidance`: `5.0`。
- `frame_rate`: 默认 25。

LTX-2 以单个约 43GB 的 `.safetensors` checkpoint 形式发布，包含 transformer、视频 VAE、音频 VAE 和 vocoder。
SimpleTuner 直接从该 combined 文件加载。

### 可选：VRAM 优化

如果需要更多 VRAM 余量：
- **Musubi 块交换**：设置 `musubi_blocks_to_swap`（建议 `4-8`），可选设置 `musubi_block_swap_device`（默认 `cpu`），将最后的 Transformer 块从 CPU 流式加载。吞吐下降但峰值 VRAM 降低。
- **VAE 补丁卷积**：设置 `--vae_enable_patch_conv=true` 启用 LTX-2 VAE 的时间分块；速度略降但峰值 VRAM 更低。
- **VAE temporal roll**：设置 `--vae_enable_temporal_roll=true` 进行更激进的时间分块（速度下降更明显）。
- **VAE 分块**：设置 `--vae_enable_tiling=true` 在大分辨率下对 VAE 编码/解码进行分块。

### 可选：CREPA 时间正则

为减少闪烁并保持主体跨帧稳定：
- 在 **Training → Loss functions** 中启用 **CREPA**。
- 推荐初始值：**Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**。
- 保持默认视觉编码器（`dinov2_vitg14`，尺寸 `518`），除非需要更小的 `dinov2_vits14` + `224`。
- 首次需要联网（或已缓存 torch hub）以获取 DINOv2 权重。
- 仅在完全使用缓存 latents 训练时启用 **Drop VAE Encoder**；否则保持关闭。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

#### 数据集注意事项

视频数据集需要仔细配置。创建 `config/multidatabackend.json`：

```json
[
  {
    "id": "my-video-dataset",
    "type": "local",
    "dataset_type": "video",
    "instance_data_dir": "datasets/videos",
    "caption_strategy": "textfile",
    "resolution": 512,
    "video": {
        "num_frames": 61,
        "min_frames": 61,
        "frame_rate": 25,
        "bucket_strategy": "aspect_ratio"
    },
    "audio": {
        "auto_split": true,
        "sample_rate": 16000,
        "channels": 1,
        "duration_interval": 3.0
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo2",
    "disabled": false
  }
]
```

在 `video` 子段中：
- `num_frames`: 训练目标帧数。
- `min_frames`: 最短视频长度（短于此会被丢弃）。
- `max_frames`: 最长视频长度过滤。
- `bucket_strategy`: 视频分桶方式：
  - `aspect_ratio`（默认）：只按空间宽高比分桶。
  - `resolution_frames`：按 `WxH@F` 格式（如 `1920x1080@61`）分桶，适合混合分辨率/时长数据。
- `frame_interval`: 使用 `resolution_frames` 时，将帧数舍入到该间隔。

如需音频条件，设置 `audio.auto_split: true`（如上）或提供单独音频数据集并通过 `s2v_datasets` 关联。
SimpleTuner 会缓存音频 latents，并与视频 latents 一并管理。

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

#### 目录设置

```bash
mkdir -p datasets/videos
</details>

# 将 .mp4 / .mov 文件放到这里
# 将对应 .txt 文件（同名）放到这里作为 caption
```

#### 登录

```bash
wandb login
huggingface-cli login
```

### 执行训练

```bash
simpletuner train
```

## 注意事项与排错提示

### Out of Memory (OOM)

视频训练非常耗资源，若 OOM：

1.  **降低分辨率**：尝试 480p（`480x854` 等）。
2.  **减少帧数**：将 `validation_num_video_frames` 与数据集 `num_frames` 降为 `33` 或 `49`。
3.  **检查卸载**：确保启用 `--enable_group_offload`。

### 验证视频质量

- **黑/噪声视频**：通常是 `validation_guidance` 过高（> 6.0）或过低（< 2.0）。建议保持在 `5.0`。
- **运动抖动**：检查数据集帧率是否与模型训练帧率一致（通常 25fps）。
- **静止视频**：模型可能训练不足，或提示词未描述运动。可使用 “camera pans right”“zoom in”“running”等。

### TREAD 训练

TREAD 也适用于视频，强烈推荐以节省算力。

在 `config.json` 中添加：

<details>
<summary>查看示例配置</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

根据比例可加速约 25-40%。

### 验证流程（T2V vs I2V）

- **T2V（文生视频）**：保持 `validation_using_datasets: false`，使用 `validation_prompt` 或 `validation_prompt_library`。
- **I2V（图生视频）**：设置 `validation_using_datasets: true`，并将 `eval_dataset_id` 指向提供参考图像的验证集。验证会切换到图生视频管线，并使用该图像作为条件输入。
- **S2V（音频条件）**：在 `validation_using_datasets: true` 下，确保 `eval_dataset_id` 指向带有 `s2v_datasets`（或 `audio.auto_split`）的数据集。验证会自动加载缓存的音频 latents。
