## PixArt Sigma 快速入门

本示例将使用 SimpleTuner 工具包训练 PixArt Sigma 模型，并使用 `full` 模型类型，因为该模型较小，应该能放进 VRAM。

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

对于 Vast、RunPod 和 TensorDock（以及其他平台），在 CUDA 12.2-12.8 镜像上可以使用以下命令启用 CUDA 扩展编译：

```bash
apt -y install nvidia-cuda-toolkit
```

### 安装

通过 pip 安装 SimpleTuner：

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

如需手动安装或开发环境设置，请参阅[安装文档](../INSTALL.md)。

#### AMD ROCm 后续步骤

要使 AMD MI300X 可用，必须执行以下操作：

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

### 设置环境

要运行 SimpleTuner，您需要设置配置文件、数据集和模型目录，以及数据加载器配置文件。

#### 配置文件

一个实验性脚本 `configure.py` 可能通过交互式的逐步配置让您完全跳过本节。它包含一些安全功能，有助于避免常见陷阱。

**注意：**这不会配置您的数据加载器。您稍后仍需手动配置。

运行方式：

```bash
simpletuner configure
```
> ⚠️ 对于位于 Hugging Face Hub 访问受限国家的用户，您应该根据系统使用的 `$SHELL` 将 `HF_ENDPOINT=https://hf-mirror.com` 添加到 `~/.bashrc` 或 `~/.zshrc` 中。

如果您更喜欢手动配置：

将 `config/config.json.example` 复制为 `config/config.json`：

```bash
cp config/config.json.example config/config.json
```

您需要修改以下变量：

<details>
<summary>查看示例配置</summary>

```json
{
  "model_type": "full",
  "use_bitfit": false,
  "pretrained_model_name_or_path": "pixart-alpha/pixart-sigma-xl-2-1024-ms",
  "model_family": "pixart_sigma",
  "output_dir": "/home/user/output/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.5
}
```
</details>

- `pretrained_model_name_or_path` - 设置为 `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`。
- `MODEL_TYPE` - 设置为 `full`。
- `USE_BITFIT` - 设置为 `false`。
- `MODEL_FAMILY` - 设置为 `pixart_sigma`。
- `OUTPUT_DIR` - 设置为您想要存储检查点和验证图像的目录。建议使用完整路径。
- `VALIDATION_RESOLUTION` - PixArt Sigma 有 1024px 或 2048px 模型，应在此示例中设置为 `1024x1024`。
  - PixArt 也在多宽高比桶上微调，可用逗号分隔指定其他分辨率：`1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - PixArt 适合较低值。设置为 `3.6` 到 `4.4` 之间。
- `pixart_validation_pipeline_mode` - 常规验证保持 `trained-stage`。验证 v0.7 split pipeline（包括 900M MoE-style stage split）时使用 `full-pipeline`：stage 1 以 latents 运行到 `1 - refiner_training_strength`，然后 stage 2 从同一边界继续。
  - 如果只训练一个 stage，需要覆盖验证时使用的固定 peer-stage checkpoint，可设置 `pixart_validation_stage1_model` 或 `pixart_validation_stage2_model`。

如果使用 Mac M 系列机器，还有一些额外设置：

- `mixed_precision` 应设置为 `no`。

> 💡 **提示：**对于磁盘空间有限的大型数据集，可使用 `--vae_cache_disable` 进行在线 VAE 编码，避免写入磁盘缓存。

#### 数据集注意事项

模型训练需要足够大的数据集。数据集规模存在限制，你必须确保数据集足够大才能有效训练模型。最小数据集规模为 `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`，过小则训练器无法发现数据集。

根据你拥有的数据集，需要以不同方式设置数据集目录和数据加载器配置文件。本示例使用 [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) 数据集。

在 `/home/user/simpletuner/config` 目录中创建 multidatabackend.json：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "pseudo-camera-10k-pixart",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 1.0,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/pixart/pseudo-camera-10k",
    "instance_data_dir": "/home/user/simpletuner/datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/pixart/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

然后创建 `datasets` 目录：

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
popd
```

这将把约 10k 张照片样本下载到 `datasets/pseudo-camera-10k` 目录，并自动创建。

#### 登录 WandB 与 Huggingface Hub

在训练开始前登录 WandB 与 HF Hub，尤其当你使用 `push_to_hub: true` 和 `--report_to=wandb` 时。

如果手动推送到 Git LFS 仓库，还应运行 `git config --global credential.helper store`。

运行以下命令：

```bash
wandb login
```

以及

```bash
huggingface-cli login
```

按提示完成登录。

### 执行训练

在 SimpleTuner 目录中直接运行：

```bash
bash train.sh
```

这将开始将文本嵌入与 VAE 输出缓存到磁盘。

更多信息请参阅 [dataloader](../DATALOADER.md) 和 [tutorial](../TUTORIAL.md) 文档。

### CLIP 分数跟踪

如需启用评估以评分模型性能，请参阅[此文档](../evaluation/CLIP_SCORES.md)。

# 稳定评估损失

如果您希望使用稳定的 MSE 损失来评估模型性能，请参阅[此文档](../evaluation/EVAL_LOSS.md)。

#### 验证预览

SimpleTuner 支持使用 Tiny AutoEncoder 在生成过程中流式输出中间验证预览。这样可以通过 webhook 回调实时查看逐步生成的验证图像。

启用方式：
<details>
<summary>查看示例配置</summary>

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```
</details>

**要求：**
- Webhook 配置
- 验证已启用

将 `validation_preview_steps` 提高（例如 3 或 5）可降低 Tiny AutoEncoder 开销。若 `validation_num_inference_steps=20` 且 `validation_preview_steps=5`，你会在第 5、10、15、20 步收到预览图。

### SageAttention

使用 `--attention_mechanism=sageattention` 时，推理验证速度可能更快。

**注意**：并不适用于所有模型配置，但值得尝试。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。
*   **[Diff2Flow](../experimental/DIFF2FLOW.md)：**允许以流匹配目标训练，可能提高生成的直线性与质量。

> ⚠️ 这些功能会增加训练的计算开销。
</details>
