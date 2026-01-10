# Hunyuan Video 1.5 快速入门

本指南将介绍如何使用 SimpleTuner 训练腾讯 8.3B 参数的 **Hunyuan Video 1.5**（`tencent/HunyuanVideo-1.5`）LoRA。

## 硬件要求

Hunyuan Video 1.5 是大型模型（8.3B 参数）。

- **最低**：在 480p、Rank-16 LoRA 且开启全量 gradient checkpointing 时，**24GB-32GB VRAM** 较为舒适。
- **推荐**：A6000 / A100（48GB-80GB）以支持 720p 训练或更大批大小。
- **系统内存**：建议 **64GB+** 以便顺利加载模型。

### 内存卸载（可选）

在 `config.json` 中添加以下内容：

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

- `--group_offload_use_stream`：仅在 CUDA 设备上生效。
- **不要**与 `--enable_model_cpu_offload` 同时使用。

## 前提条件

确保已安装 Python；SimpleTuner 在 3.10 到 3.12 版本上运行良好。

您可以运行以下命令检查：

```bash
python --version
```

如果您的 Ubuntu 系统未安装 Python 3.12，可以尝试以下命令：

```bash
apt -y install python3.12 python3.12-venv
```

### 容器镜像依赖

对于 Vast、RunPod 和 TensorDock（以及其他平台），在 CUDA 12.2-12.8 镜像上可以使用以下命令启用 CUDA 扩展编译：

```bash
apt -y install nvidia-cuda-toolkit
```

### AMD ROCm 后续步骤

要使 AMD MI300X 可用，必须执行以下操作：

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## 安装

通过 pip 安装 SimpleTuner：

```bash
pip install 'simpletuner[cuda]'
```

如需手动安装或开发环境设置，请参阅[安装文档](../INSTALL.md)。

### 必需的检查点

主仓库 `tencent/HunyuanVideo-1.5` 包含 transformer/vae/scheduler，但**文本编码器**（`text_encoder/llm`）与**视觉编码器**（`vision_encoder/siglip`）需要单独下载。启动前请将 SimpleTuner 指向本地路径：

```bash
export HUNYUANVIDEO_TEXT_ENCODER_PATH=/path/to/text_encoder_root
export HUNYUANVIDEO_VISION_ENCODER_PATH=/path/to/vision_encoder_root
```

如果未设置，SimpleTuner 会尝试从模型仓库拉取；多数镜像并未包含它们，因此请显式指定路径以避免启动错误。

## 设置环境

### Web 界面方式

SimpleTuner WebUI 可简化配置。启动服务器：

```bash
simpletuner server
```

默认会在 8001 端口创建 Web 服务器，可访问 http://localhost:8001。

### 手动 / 命令行方式

通过命令行运行 SimpleTuner 需要准备配置文件、数据集与模型目录，以及数据加载器配置文件。

#### 配置文件

实验脚本 `configure.py` 可能通过交互式步骤让你跳过本节。

**注意：**这不会配置数据加载器，你仍需手动配置。

运行：

```bash
simpletuner configure
```

如果更喜欢手动配置：

将 `config/config.json.example` 复制为 `config/config.json`：

```bash
cp config/config.json.example config/config.json
```

HunyuanVideo 的关键配置覆盖项：

<details>
<summary>查看示例配置</summary>

```json
{
  "model_type": "lora",
  "model_family": "hunyuanvideo",
  "pretrained_model_name_or_path": "tencent/HunyuanVideo-1.5",
  "model_flavour": "t2v-480p",
  "output_dir": "output/hunyuan-video",
  "validation_resolution": "854x480",
  "validation_num_video_frames": 61,
  "validation_guidance": 6.0,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "learning_rate": 1e-4,
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "lora_rank": 16,
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "dataset_backend_config": "config/multidatabackend.json"
}
```
</details>

- `model_flavour` 选项:
  - `t2v-480p`（默认）
  - `t2v-720p`
  - `i2v-480p`（图生视频）
  - `i2v-720p`（图生视频）
- `validation_num_video_frames`：必须满足 `(frames - 1) % 4 == 0`，例如 61、129。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

#### 数据集注意事项

创建 `--data_backend_config`（`config/multidatabackend.json`）文档如下：

```json
[
  {
    "id": "my-video-dataset",
    "type": "local",
    "dataset_type": "video",
    "instance_data_dir": "datasets/videos",
    "caption_strategy": "textfile",
    "resolution": 480,
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
    "cache_dir": "cache/text/hunyuan",
    "disabled": false
  }
]
```

在 `video` 子段中：
- `num_frames`：训练目标帧数，必须满足 `(frames - 1) % 4 == 0`。
- `min_frames`：最短视频长度（更短的视频会被丢弃）。
- `max_frames`：最长视频长度过滤。
- `bucket_strategy`：视频分桶方式：
  - `aspect_ratio`（默认）：仅按空间宽高比分桶。
  - `resolution_frames`：按 `WxH@F` 格式（如 `854x480@61`）分桶，适用于混合分辨率/时长。
- `frame_interval`：使用 `resolution_frames` 时，将帧数舍入到该间隔。

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

- **文本嵌入缓存**：强烈推荐。Hunyuan 使用大型 LLM 文本编码器，缓存可显著减少训练时 VRAM。

#### 登录 WandB 与 Huggingface Hub

```bash
wandb login
huggingface-cli login
```

</details>

### 执行训练

在 SimpleTuner 目录下运行：

```bash
simpletuner train
```

## 注意事项与排错提示

### VRAM 优化

- **Group Offload**：消费级 GPU 必备。确保 `enable_group_offload` 为 true。
- **分辨率**：显存有限时保持 480p（`854x480` 或类似）；720p（`1280x720`）会显著增加内存占用。
- **量化**：使用 `base_model_precision`（默认 `bf16`）；`int8-torchao` 可以进一步节省但速度更慢。
- **VAE patch convolution**：HunyuanVideo VAE OOM 时设置 `--vae_enable_patch_conv=true`（或在 UI 中开启）。这会切分 3D conv/attention 以降低峰值 VRAM，吞吐略降。

### 图生视频（I2V）

- 使用 `model_flavour="i2v-480p"`。
- SimpleTuner 会自动将视频样本的首帧作为条件图像。
- 确保验证设置包含条件输入，或依赖自动提取的首帧。

### 文本编码器

Hunyuan 使用双文本编码器（LLM + CLIP）。缓存阶段请确保系统内存足够加载它们。
