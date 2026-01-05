# Kandinsky 5.0 Image 快速入门

本示例将训练 Kandinsky 5.0 Image LoRA。

## 硬件要求

Kandinsky 5.0 除了标准 CLIP 编码器与 Flux VAE，还使用**巨大的 7B 参数 Qwen2.5-VL 文本编码器**，对 VRAM 和系统内存要求都很高。

仅加载 Qwen 编码器就需要约 **14GB** 内存。若使用完整的 gradient checkpointing 训练 rank-16 LoRA：

- **24GB VRAM** 是舒适的最低配置（RTX 3090/4090）。
- **16GB VRAM** 也可行，但需要激进的卸载，并且很可能要对基础模型做 `int8` 量化。

你需要：

- **系统内存**：至少 32GB，最好 64GB，确保初始加载不崩溃。
- **GPU**：NVIDIA RTX 3090 / 4090 或专业卡（A6000、A100 等）。

### 内存卸载（推荐）

考虑到文本编码器体积巨大，在消费级硬件上几乎必须使用分组卸载。该功能会在不计算时将 transformer block 卸载到 CPU 内存。

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

- `--group_offload_use_stream`：仅在 CUDA 设备上生效。
- **不要**与 `--enable_model_cpu_offload` 同时使用。

另外，在 `config.json` 中设置 `"offload_during_startup": true`，以减少初始化和缓存阶段的 VRAM 占用。这会避免文本编码器与 VAE 同时加载。

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

## 安装

通过 pip 安装 SimpleTuner：

```bash
pip install simpletuner[cuda]
```

如需手动安装或开发环境设置，请参阅[安装文档](../INSTALL.md)。

## 设置环境

### Web 界面方式

SimpleTuner WebUI 可简化配置。启动服务器：

```bash
simpletuner server
```

访问 http://localhost:8001。

### 手动 / 命令行方式

通过命令行运行 SimpleTuner 需要准备配置文件、数据集与模型目录，以及数据加载器配置文件。

#### 配置文件

实验脚本 `configure.py` 可能通过交互式步骤让你跳过本节：

```bash
simpletuner configure
```

如果您更喜欢手动配置：

将 `config/config.json.example` 复制为 `config/config.json`：

```bash
cp config/config.json.example config/config.json
```

你需要修改以下变量：

- `model_type`: `lora`
- `model_family`: `kandinsky5-image`
- `model_flavour`:
  - `t2i-lite-sft`:（默认）标准 SFT 检查点，适合微调风格/角色。
  - `t2i-lite-pretrain`: 预训练检查点，更适合从零学习新概念。
  - `i2i-lite-sft` / `i2i-lite-pretrain`: 图像到图像训练，需要数据集中有条件图像。
- `output_dir`: 检查点保存目录。
- `train_batch_size`: 从 `1` 开始。
- `gradient_accumulation_steps`: 用 `1` 或更大来模拟更大 batch。
- `validation_resolution`: 该模型标准为 `1024x1024`。
- `validation_guidance`: `5.0` 是 Kandinsky 5 推荐默认值。
- `flow_schedule_shift`: 默认 `1.0`。调整会改变模型在细节与构图之间的偏好（见下文）。

#### 验证提示词

`config/config.json` 中包含“主验证提示词”。你也可以在 `config/user_prompt_library.json` 中创建提示词库：

<details>
<summary>查看示例配置</summary>

```json
{
  "portrait": "A high quality portrait of a woman, cinematic lighting, 8k",
  "landscape": "A beautiful mountain landscape at sunset, oil painting style"
}
```
</details>

在 `config.json` 中添加如下内容启用：

<details>
<summary>查看示例配置</summary>

```json
{
  "user_prompt_library": "config/user_prompt_library.json"
}
```
</details>

#### Flow schedule shifting

Kandinsky 5 是流匹配模型。`shift` 参数控制训练与推理时的噪声分布。

- **Shift 1.0（默认）**：训练平衡。
- **更低 Shift (< 1.0)**：更关注高频细节（纹理、噪点）。
- **更高 Shift (> 1.0)**：更关注低频细节（构图、色彩、结构）。

如果模型学会了风格但构图不行，尝试提高 shift；若构图有了但质感不足，尝试降低 shift。

#### 量化模型训练

可将 transformer 量化为 8-bit 以显著降低 VRAM。

在 `config.json` 中：

<details>
<summary>查看示例配置</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "base_model_default_dtype": "bf16"
```
</details>

> **注意**：不推荐量化文本编码器（`no_change`），因为 Qwen2.5-VL 对量化敏感且已是管线中最重的部分。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

#### 数据集注意事项

你需要一个数据集配置文件，例如 `config/multidatabackend.json`。

```json
[
  {
    "id": "my-image-dataset",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/my_images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "crop": true,
    "crop_aspect": "square",
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

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

然后创建数据集目录：

```bash
mkdir -p datasets/my_images
</details>

# 在此放置图片与 .txt caption 文件
```

#### 登录 WandB 与 Huggingface Hub

```bash
wandb login
huggingface-cli login
```

### 执行训练

**选项 1（推荐）：**

```bash
simpletuner train
```

**选项 2（Legacy）：**

```bash
./train.sh
```

## 注意事项与排错提示

### 最低 VRAM 配置

在 16GB 或受限 24GB 环境下运行：

1.  **启用 Group Offload**：`--enable_group_offload`。
2.  **量化基础模型**：设为 `"base_model_precision": "int8-quanto"`。
3.  **批大小**：保持为 `1`。

### 伪影与“烧焦”图像

若验证图像过饱和或噪点严重（“烧焦”）：

- **检查 Guidance**：确保 `validation_guidance` 在 5.0 左右。过高（如 7.0+）会导致图像过曝或失真。
- **检查 Flow Shift**：极端 `flow_schedule_shift` 会导致不稳定。建议从 1.0 开始。
- **学习率**：LoRA 标准为 1e-4，若出现伪影，可降至 5e-5。

### TREAD 训练

Kandinsky 5 支持 [TREAD](../TREAD.md) 以通过丢弃 token 加速训练。

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

这会在中间层丢弃 50% token，从而加速 transformer 前向。
