# Kandinsky 5.0 Video 快速入门

本示例将使用 HunyuanVideo VAE 与双文本编码器训练 Kandinsky 5.0 Video LoRA（Lite 或 Pro）。

## 硬件要求

Kandinsky 5.0 Video 是重量级模型，由以下组件组成：
1.  **Qwen2.5-VL (7B)**：超大型视觉语言文本编码器。
2.  **HunyuanVideo VAE**：高质量 3D VAE。
3.  **Video Transformer**：复杂的 DiT 架构。

该组合非常耗 VRAM，但 “Lite” 与 “Pro” 版本要求不同。

- **Lite 模型训练**：相当高效，可在 **~13GB VRAM** 上训练。
  - **注意**：初始 **VAE 预缓存步骤** 需要显著更多 VRAM，因为 HunyuanVideo VAE 非常大。可能需要 CPU offload 或更大 GPU 才能完成缓存阶段。
  - **提示**：在 `config.json` 中设置 `"offload_during_startup": true`，确保 VAE 与文本编码器不会同时加载到 GPU，可显著降低预缓存压力。
  - **若 VAE OOM**：设置 `--vae_enable_patch_conv=true` 以切分 HunyuanVideo VAE 3D 卷积；速度略慢但峰值 VRAM 更低。
- **Pro 模型训练**：需要 **FSDP2**（多 GPU）或配合 LoRA 的强力 **Group Offload** 才能在消费级硬件上运行。具体 VRAM/RAM 要求尚未确定，但越多越好。
- **系统内存**：Lite 模型在 **45GB** RAM 下较为舒适。建议 64GB+ 以更稳妥。

### 内存卸载（关键）

对于几乎所有单 GPU 训练 **Pro** 模型的场景，必须启用分组卸载。Lite 模型虽非强制，但推荐以节省 VRAM 给更大 batch/分辨率。

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

Kandinsky 5 Video 的关键设置：

- `model_family`: `kandinsky5-video`
- `model_flavour`:
  - `t2v-lite-sft-5s`: Lite 模型，约 5 秒输出（默认）
  - `t2v-lite-sft-10s`: Lite 模型，约 10 秒输出
  - `t2v-pro-sft-5s-hd`: Pro 模型，约 5 秒，高分辨率训练
  - `t2v-pro-sft-10s-hd`: Pro 模型，约 10 秒，高分辨率训练
  - `i2v-lite-5s`: 图生视频 Lite，5 秒输出（需条件图像）
  - `i2v-pro-sft-5s`: 图生视频 Pro SFT，5 秒输出（需条件图像）
  - *(以上均有 pretrain 版本)*
- `train_batch_size`: `1`。除非有 A100/H100，否则不要提高。
- `validation_resolution`:
  - `512x768` 是安全的测试默认值。
  - `720x1280`（720p）可行但较重。
- `validation_num_video_frames`: **必须与 VAE 压缩比例 (4x) 兼容。**
  - 5 秒（约 12-24fps）：使用 `61` 或 `49`。
  - 公式：`(frames - 1) % 4 == 0`。
- `validation_guidance`: `5.0`。
- `frame_rate`: 默认 24。

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
        "frame_rate": 24,
        "bucket_strategy": "aspect_ratio"
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/kandinsky5",
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
- **运动抖动**：检查数据集帧率是否与模型训练帧率一致（通常 24fps）。
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

### I2V（图生视频）训练

使用 `i2v` 版本时：
- SimpleTuner 会自动提取训练视频的首帧作为条件图像。
- 训练时会自动遮罩首帧。
- 验证需要提供输入图像；否则 SimpleTuner 会使用验证生成视频的首帧作为条件图像。
