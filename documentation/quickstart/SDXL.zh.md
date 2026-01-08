## Stable Diffusion XL 快速入门

在本示例中，我们将使用 SimpleTuner 工具包训练一个 Stable Diffusion XL 模型，并使用 `lora` 模型类型。

与现代更大的模型相比，SDXL 的规模相当适中，因此可能可以使用 `full` 训练，但这将比 LoRA 训练需要更多的 VRAM，并需要其他超参数调整。

### 前提条件

确保您已安装 Python；SimpleTuner 在 3.10 到 3.12 版本上运行良好（AMD ROCm 机器需要 3.12）。

您可以运行以下命令检查：

```bash
python --version
```

如果您的 Ubuntu 系统未安装 Python 3.12，可以尝试以下命令：

```bash
apt -y install python3.12 python3.12-venv
```

#### 容器镜像依赖

对于 Vast、RunPod 和 TensorDock（以及其他平台），在 CUDA 12.2-12.8 镜像上可以使用以下命令启用 CUDA 扩展编译：

```bash
apt -y install nvidia-cuda-toolkit
```

### 安装

通过 pip 安装 SimpleTuner：

```bash
pip install simpletuner[cuda]
```

如需手动安装或开发环境设置，请参阅[安装文档](../INSTALL.md)。

### 设置环境

要运行 SimpleTuner，您需要设置配置文件、数据集和模型目录，以及数据加载器配置文件。

#### 配置文件

一个实验性脚本 `configure.py` 可以通过交互式的逐步配置完全跳过本节。它包含一些安全功能，有助于避免常见陷阱。

**注意：**这不会**完全**配置您的数据加载器。您稍后仍需手动配置。

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

#### AMD ROCm 后续步骤

要使 AMD MI300X 可用，必须执行以下操作：

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

您需要修改以下变量：

<details>
<summary>查看示例配置</summary>

```json
{
  "model_type": "lora",
  "model_family": "sdxl",
  "model_flavour": "base-1.0",
  "output_dir": "/home/user/output/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.4,
  "use_gradient_checkpointing": true,
  "learning_rate": 1e-4
}
```
</details>

- `model_family` - 设置为 `sdxl`。
- `model_flavour` - 设置为 `base-1.0`，或使用 `pretrained_model_name_or_path` 指向其他模型。
- `model_type` - 设置为 `lora`。
- `use_dora` - 如果您希望训练 DoRA，请设置为 `true`。
- `output_dir` - 设置为您想要存储检查点和验证图像的目录。建议使用完整路径。
- `validation_resolution` - 本示例设置为 `1024x1024`。
  - 此外，Stable Diffusion XL 在多宽高比桶上进行了微调，可以使用逗号分隔指定其他分辨率：`1024x1024,1280x768`
- `validation_guidance` - 使用您在推理时习惯使用的值。设置在 `4.2` 到 `6.4` 之间。
- `use_gradient_checkpointing` - 除非您有大量 VRAM 并想牺牲一些来加快速度，否则这应该是 `true`。
- `learning_rate` - `1e-4` 对于低秩网络来说相当常见，但如果您注意到任何"烧焦"或早期过度训练，`1e-5` 可能是更保守的选择。

如果使用 Mac M 系列机器，还有一些额外设置：

- `mixed_precision` 应设置为 `no`。
  - 这在 pytorch 2.4 时是正确的，但从 2.6+ 开始可能可以使用 bf16
- `attention_mechanism` 可以设置为 `xformers` 来使用它，但它已经有些过时了。

#### 量化模型训练

在 Apple 和 NVIDIA 系统上经过测试，Hugging Face Optimum-Quanto 可用于降低 Unet 的精度和 VRAM 要求，但它在 Diffusion Transformer 模型（如 SD3/Flux）上的效果不如其他模型好，因此不推荐使用。

但是，如果您的资源受限，仍然可以使用它。

对于 `config.json`：
<details>
<summary>查看示例配置</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "optimizer": "optimi-lion"
}
```
</details>

#### 高级实验性功能

<details>
<summary>显示高级实验性详情</summary>


SimpleTuner 包含可以显著改善训练稳定性和性能的实验性功能，特别适用于较小的数据集或像 SDXL 这样的旧架构。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：** 通过让模型在训练期间生成自己的输入，减少暴露偏差并提高输出质量。
*   **[Diff2Flow](../experimental/DIFF2FLOW.md)：** 允许使用 Flow Matching 目标训练 SDXL，可能改善生成的直接性和质量。

> ⚠️ 这些功能会增加训练的计算开销。

</details>

#### 数据集注意事项

拥有足够大的数据集来训练模型至关重要。数据集大小有限制，您需要确保数据集足够大以有效训练模型。请注意，最小数据集大小为 `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`。如果数据集太小，训练器将无法发现它。

> 💡 **提示：** 对于磁盘空间紧张的大型数据集，您可以使用 `--vae_cache_disable` 来执行在线 VAE 编码而不将结果缓存到磁盘。如果您使用 `--vae_cache_ondemand`，这将隐式启用，但添加 `--vae_cache_disable` 可确保不会写入磁盘。

根据您拥有的数据集，您需要以不同方式设置数据集目录和数据加载器配置文件。在此示例中，我们将使用 [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) 作为数据集。

在您的 `OUTPUT_DIR` 目录中，创建一个 multidatabackend.json：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sdxl",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 1.0,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/sdxl/pseudo-camera-10k",
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
    "cache_dir": "cache/text/sdxl/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> 请参阅 [DATALOADER.md](../DATALOADER.md#caption_strategy) 中的 caption_strategy 选项和要求。

然后，创建一个 `datasets` 目录：

```bash
mkdir -p datasets
huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=datasets/pseudo-camera-10k
```

这将下载约 10k 张照片样本到您的 `datasets/pseudo-camera-10k` 目录，该目录将自动为您创建。

#### 登录 WandB 和 Huggingface Hub

在开始训练之前，您需要登录 WandB 和 HF Hub，特别是如果您使用 `push_to_hub: true` 和 `--report_to=wandb`。

如果您要手动将项目推送到 Git LFS 仓库，您还应该运行 `git config --global credential.helper store`

运行以下命令：

```bash
wandb login
```

和

```bash
huggingface-cli login
```

按照说明登录这两个服务。

### 执行训练运行

从 SimpleTuner 目录，只需运行：

```bash
bash train.sh
```

这将开始将文本嵌入和 VAE 输出缓存到磁盘。

有关更多信息，请参阅[数据加载器](../DATALOADER.md)和[教程](../TUTORIAL.md)文档。

### CLIP 分数跟踪

如果您希望启用评估来评分模型性能，请参阅[此文档](../evaluation/CLIP_SCORES.md)了解如何配置和解释 CLIP 分数。

# 稳定评估损失

如果您希望使用稳定 MSE 损失来评分模型性能，请参阅[此文档](../evaluation/EVAL_LOSS.md)了解如何配置和解释评估损失。

#### 验证预览

SimpleTuner 支持使用 Tiny AutoEncoder 模型在生成过程中流式传输中间验证预览。这允许您通过 webhook 回调实时逐步查看正在生成的验证图像。

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
- 启用验证

将 `validation_preview_steps` 设置为更高的值（例如 3 或 5）以减少 Tiny AutoEncoder 开销。使用 `validation_num_inference_steps=20` 和 `validation_preview_steps=5`，您将在步骤 5、10、15 和 20 收到预览图像。
