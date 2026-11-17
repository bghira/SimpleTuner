## Lumina2 快速入门

在本示例中，我们将训练一个 Lumina2 LoRA 或进行全模型微调。

### 硬件要求

Lumina2 是一个 2B 参数模型，比 Flux 或 SD3 等大型模型更易用。模型体积较小意味着：

当训练 rank-16 LoRA 时：
- LoRA 训练约需 12-14GB VRAM
- 全模型微调约需 16-20GB VRAM
- 启动时约需 20-30GB 系统内存

你需要：
- **最低**：单张 RTX 3060 12GB 或 RTX 4060 Ti 16GB
- **推荐**：RTX 3090、RTX 4090 或 A100 以更快训练
- **系统内存**：建议至少 32GB

### 前提条件

确保已安装 Python；SimpleTuner 在 3.10 到 3.12 版本上运行良好。

您可以运行以下命令检查：

```bash
python --version
```

如果您的 Ubuntu 系统未安装 Python 3.12，可以尝试以下命令：

```bash
apt -y install python3.13 python3.13-venv
```

#### 容器镜像依赖

对于 Vast、RunPod 和 TensorDock（以及其他平台），在 CUDA 12.2-12.8 镜像上可以使用以下命令：

```bash
apt -y install nvidia-cuda-toolkit
```

### 安装

通过 pip 安装 SimpleTuner：

```bash
pip install 'simpletuner[cuda]' --extra-index-url https://download.pytorch.org/whl/cu132

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu132
```

如需手动安装或开发环境设置，请参阅[安装文档](../INSTALL.md)。

### 设置环境

要运行 SimpleTuner，您需要设置配置文件、数据集和模型目录，以及数据加载器配置文件。

#### 配置文件

将 `config/config.json.example` 复制为 `config/config.json`：

```bash
cp config/config.json.example config/config.json
```

您需要修改以下变量：

- `model_type` - LoRA 训练设为 `lora`，全参微调设为 `full`。
- `model_family` - 设为 `lumina2`。
- `output_dir` - 设为存储检查点与验证图像的目录，建议使用完整路径。
- `train_batch_size` - 取 1-4，取决于显存与数据集大小。
- `validation_resolution` - Lumina2 支持多分辨率，常用值：`1024x1024`、`512x512`、`768x768`。
- `validation_guidance` - Lumina2 使用 CFG，引导值 3.5-7.0 效果较好。
- `validation_num_inference_steps` - 20-30 步适合 Lumina2。
- `gradient_accumulation_steps` - 用于模拟更大 batch，推荐 2-4。
- `optimizer` - 推荐 `adamw_bf16`，`lion` 和 `optimi-stableadamw` 也可。
- `mixed_precision` - 建议保持 `bf16`。
- `gradient_checkpointing` - 设为 `true` 以节省 VRAM。
- `learning_rate` - LoRA: `1e-4` 到 `5e-5`；全参微调: `1e-5` 到 `1e-6`。

#### Lumina2 示例配置

放入 `config.json`：

<details>
<summary>查看示例配置</summary>

```json
{
    "base_model_precision": "int8-torchao",
    "checkpoint_step_interval": 50,
    "data_backend_config": "config/lumina2/multidatabackend.json",
    "disable_bucket_pruning": true,
    "eval_steps_interval": 50,
    "evaluation_type": "clip",
    "flow_schedule_auto_shift": true,
    "gradient_checkpointing": true,
    "hub_model_id": "lumina2-lora",
    "learning_rate": 1e-4,
    "lora_alpha": 16,
    "lora_rank": 16,
    "lora_type": "standard",
    "lr_scheduler": "constant",
    "max_train_steps": 400000,
    "model_family": "lumina2",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/lumina2",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "quantize_via": "cpu",
    "report_to": "wandb",
    "seed": 42,
    "tracker_project_name": "lumina2-training",
    "tracker_run_name": "lumina2-lora",
    "train_batch_size": 4,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_disable_unconditional": true,
    "validation_guidance": 4.0,
    "validation_guidance_rescale": 0.0,
    "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
    "validation_num_inference_steps": 40,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_prompt_library": false,
    "validation_resolution": "1024x1024",
    "validation_seed": 42,
    "validation_step_interval": 50
}
```
</details>

若使用 Lycoris 训练，将 `lora_type` 改为 `lycoris`。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

#### 验证提示词

`config/config.json` 中包含“主验证提示词”。此外，创建一个提示词库文件：

```json
{
  "portrait": "a high-quality portrait photograph with natural lighting",
  "landscape": "a breathtaking landscape photograph with dramatic lighting",
  "artistic": "an artistic rendering with vibrant colors and creative composition",
  "detailed": "a highly detailed image with sharp focus and rich textures",
  "stylized": "a stylized illustration with unique artistic flair"
}
```

在配置中添加：
```json
{
  "--user_prompt_library": "config/user_prompt_library.json"
}
```

#### 数据集注意事项

Lumina2 受益于高质量训练数据。创建 `--data_backend_config`（`config/multidatabackend.json`）：

> 💡 **提示：**对于磁盘空间有限的大型数据集，可使用 `--vae_cache_disable` 进行在线 VAE 编码，避免写入磁盘缓存。

```json
[
  {
    "id": "lumina2-training",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 2048,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/lumina2/training",
    "instance_data_dir": "/datasets/training",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/lumina2",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

创建数据集目录。请将路径替换为实际位置。

```bash
mkdir -p /datasets/training
</details>

# 将图片与 caption 文件放到 /datasets/training/
```

caption 文件与图片同名，后缀为 `.txt`。

#### 登录 WandB

SimpleTuner 提供**可选**跟踪支持，主要面向 Weights & Biases。可使用 `report_to=none` 关闭。

如需启用 wandb，运行：

```bash
wandb login
```

#### 登录 Huggingface Hub

若要推送检查点到 Huggingface Hub，请确保
```bash
huggingface-cli login
```

### 执行训练

从 SimpleTuner 目录可选择以下方式启动训练：

**选项 1（推荐 - pip 安装）：**
```bash
pip install 'simpletuner[cuda]' --extra-index-url https://download.pytorch.org/whl/cu132

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu132
simpletuner train
```

**选项 2（Git clone 方式）：**
```bash
simpletuner train
```

**选项 3（Legacy 方式 - 仍可用）：**
```bash
./train.sh
```

这将开始将文本嵌入与 VAE 输出缓存到磁盘。

## Lumina2 训练建议

### 学习率

#### LoRA 训练
- 从 `1e-4` 开始，根据结果调整
- Lumina2 收敛快，需密切关注早期迭代
- Rank 8-32 适合大多数用途，64-128 需更谨慎，256-512 可用于注入全新任务

#### 全参微调
- 使用更低学习率：`1e-5` 到 `5e-6`
- 考虑使用 EMA（指数滑动平均）稳定训练
- 建议 gradient clipping（`max_grad_norm`）为 1.0

### 分辨率考虑

Lumina2 支持灵活分辨率：
- 1024x1024 质量最佳
- 混合分辨率训练（512px、768px、1024px）对质量的影响尚未测试
- 宽高比分桶适配良好

### 训练时长

由于 Lumina2 仅 2B 参数：
- LoRA 通常 500-2000 步就能收敛
- 全参微调可能需要 2000-5000 步
- 训练速度快，请频繁观察验证图像

### 常见问题与解决方案

1. **模型收敛过快**：降低学习率，从 Lion 改为 AdamW
2. **生成图像出现伪影**：确保数据质量高，并考虑降低学习率
3. **显存不足**：启用 gradient checkpointing 并减小 batch size
4. **容易过拟合**：使用正则化数据集

## 推理建议

### 使用训练后的模型

Lumina2 模型可用于：
- 直接使用 Diffusers
- ComfyUI（需对应节点）
- 其他支持 Gemma2 架构的推理框架

### 最佳推理设置

- Guidance scale: 4.0-6.0
- 推理步数: 20-50
- 使用与训练相同的分辨率效果最佳

## 备注

### Lumina2 优势

- 2B 参数，训练快
- 质量/规模比佳
- 支持多种训练模式（LoRA、LyCORIS、全参）
- 内存占用较低

### 当前限制

- 暂无 ControlNet 支持
- 仅支持文本到图像
- 需要较高质量的 caption 才能取得最佳效果

### 内存优化

与大模型不同，Lumina2 通常不需要：
- 模型量化
- 极端内存优化技巧
- 复杂混合精度策略
