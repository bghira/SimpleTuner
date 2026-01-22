# Flux[dev] / Flux[schnell] 快速入门

![image](https://github.com/user-attachments/assets/6409d790-3bb4-457c-a4b4-a51a45fc91d1)

在本示例中，我们将训练一个 Flux.1 Krea LoRA。

## 硬件要求

Flux 除了需要 GPU 显存外，还需要大量的**系统内存**。仅在启动时量化模型就需要约 50GB 的系统内存。如果耗时过长，您可能需要评估硬件能力并确定是否需要进行调整。

当您训练 rank-16 LoRA 的所有组件（MLP、投影层、多模态块）时，显存使用情况如下：

- 不量化基础模型时需要略多于 30G VRAM
- 量化为 int8 + bf16 基础/LoRA 权重时需要略多于 18G VRAM
- 量化为 int4 + bf16 基础/LoRA 权重时需要略多于 13G VRAM
- 量化为 NF4 + bf16 基础/LoRA 权重时需要略多于 9G VRAM
- 量化为 int2 + bf16 基础/LoRA 权重时需要略多于 9G VRAM

您需要：

- **绝对最低配置**是单张 **3080 10G**
- **实际最低配置**是单张 3090 或 V100 GPU
- **理想配置**是多张 4090、A6000、L40S 或更高端显卡

幸运的是，这些显卡可以通过 [LambdaLabs](https://lambdalabs.com) 等供应商获取，该供应商提供最低价格，并为多节点训练提供本地化集群。

**与其他模型不同，Apple GPU 目前不支持训练 Flux。**


## 前提条件

确保您已安装 Python；SimpleTuner 在 3.10 到 3.12 版本上运行良好。

您可以运行以下命令检查：

```bash
python --version
```

如果您的 Ubuntu 系统未安装 Python 3.12，可以尝试以下命令：

```bash
apt -y install python3.13 python3.13-venv
```

### 容器镜像依赖

对于 Vast、RunPod 和 TensorDock（以及其他平台），在 CUDA 12.2-12.8 镜像上可以使用以下命令启用 CUDA 扩展编译：

```bash
apt -y install nvidia-cuda-toolkit
```

## 安装

通过 pip 安装 SimpleTuner：

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

如需手动安装或开发环境设置，请参阅[安装文档](../INSTALL.md)。

### AMD ROCm 后续步骤

要使 AMD MI300X 可用，必须执行以下操作：

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## 设置环境

### 网页界面方式

SimpleTuner WebUI 使设置变得相当简单。要运行服务器：

```bash
simpletuner server
```

这将默认在 8001 端口创建一个网页服务器，您可以通过访问 http://localhost:8001 来使用它。

### 手动/命令行方式

要通过命令行工具运行 SimpleTuner，您需要设置配置文件、数据集和模型目录，以及数据加载器配置文件。

#### 配置文件

一个实验性脚本 `configure.py` 可以通过交互式的逐步配置完全跳过本节。它包含一些安全功能，有助于避免常见陷阱。

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

您可能需要修改以下变量：

- `model_type` - 设置为 `lora`。
- `model_family` - 设置为 `flux`。
- `model_flavour` - 默认为 `krea`，但也可设置为 `dev` 以训练原始 FLUX.1-Dev 版本。
  - `krea` - 默认的 FLUX.1-Krea [dev] 模型，是 Krea 1 的开放权重变体，这是 BFL 和 Krea.ai 合作的专有模型
  - `dev` - Dev 模型变体，之前的默认选项
  - `schnell` - Schnell 模型变体；快速入门会自动设置快速噪声调度和助手 LoRA 堆栈
  - `kontext` - Kontext 训练（请参阅[此指南](../quickstart/FLUX_KONTEXT.md)获取具体指导）
  - `fluxbooru` - 基于 FLUX.1-Dev 的去蒸馏（需要 CFG）模型，名为 [FluxBooru](https://hf.co/terminusresearch/fluxbooru-v0.3)，由 terminus research group 创建
  - `libreflux` - 基于 FLUX.1-Schnell 的去蒸馏模型，需要对 T5 文本编码器输入进行注意力掩码
- `offload_during_startup` - 如果在 VAE 编码期间内存不足，请设置为 `true`。
- `pretrained_model_name_or_path` - 设置为 `black-forest-labs/FLUX.1-dev`。
- `pretrained_vae_model_name_or_path` - 设置为 `black-forest-labs/FLUX.1-dev`。
  - 请注意，您需要登录 Huggingface 并获得下载此模型的权限。我们将在本教程后面介绍如何登录 Huggingface。
- `output_dir` - 设置为您想要存储检查点和验证图像的目录。建议使用完整路径。
- `train_batch_size` - 应保持为 1，特别是如果您的数据集非常小。
- `validation_resolution` - 由于 Flux 是 1024px 模型，您可以将其设置为 `1024x1024`。
  - 此外，Flux 在多宽高比桶上进行了微调，可以使用逗号分隔指定其他分辨率：`1024x1024,1280x768,2048x2048`
- `validation_guidance` - 使用您在 Flux 推理时习惯选择的值。
- `validation_guidance_real` - 使用 >1.0 来启用 Flux 推理的 CFG。这会减慢验证速度，但产生更好的结果。配合空的 `VALIDATION_NEGATIVE_PROMPT` 效果最佳。
- `validation_num_inference_steps` - 使用约 20 步可以在节省时间的同时看到不错的质量。Flux 变化不大，更多步骤可能只是浪费时间。
- `--lora_rank=4` 如果您希望大幅减小正在训练的 LoRA 的大小。这可以帮助节省 VRAM。
- Schnell LoRA 运行通过快速入门默认值自动使用快速调度；无需额外标志。

- `gradient_accumulation_steps` - 之前的指导是避免在 bf16 训练中使用这些，因为它们会降低模型质量。进一步测试表明，对于 Flux 来说这不一定是问题。
  - 此选项会导致更新步骤在多个步骤中累积。这将线性增加训练运行时间，例如值为 2 将使您的训练运行速度减半，耗时加倍。
- `optimizer` - 建议初学者使用 adamw_bf16，尽管 optimi-lion 和 optimi-stableadamw 也是不错的选择。
- `mixed_precision` - 初学者应保持为 `bf16`
- `gradient_checkpointing` - 几乎在所有情况下和所有设备上都应设置为 true
- `gradient_checkpointing_interval` - 在较大的 GPU 上可以设置为 2 或更高的值，以仅对每 _n_ 个块进行检查点。值为 2 将检查点一半的块，值为 3 将检查点三分之一的块。

### 高级实验性功能

<details>
<summary>显示高级实验性详情</summary>


SimpleTuner 包含可以显著改善训练稳定性和性能的实验性功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：** 通过让模型在训练期间生成自己的输入，减少暴露偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

</details>

### 内存卸载（可选）

Flux 通过 diffusers v0.33+ 支持分组模块卸载。当您受到 transformer 权重的瓶颈时，这可以显著减少 VRAM 压力。您可以通过在 `TRAINER_EXTRA_ARGS`（或 WebUI 硬件页面）中添加以下标志来启用它：

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# 可选：将卸载的权重溢出到磁盘而不是 RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- `--group_offload_use_stream` 仅在 CUDA 设备上有效；SimpleTuner 在 ROCm、MPS 和 CPU 后端上自动禁用流。
- **不要**将此与 `--enable_model_cpu_offload` 结合使用——这两种策略是互斥的。
- 使用 `--group_offload_to_disk_path` 时，优先选择快速的本地 SSD/NVMe 目标。

#### 验证提示词

`config/config.json` 中包含"主验证提示词"，通常是您针对单个主题或风格训练的主 instance_prompt。此外，可以创建一个 JSON 文件，包含验证期间运行的额外提示词。

示例配置文件 `config/user_prompt_library.json.example` 包含以下格式：

<details>
<summary>查看示例配置</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

昵称是验证的文件名，因此请保持简短并与您的文件系统兼容。

要将训练器指向此提示词库，请在 `config.json` 末尾添加新行将其添加到 TRAINER_EXTRA_ARGS：

<details>
<summary>查看示例配置</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

一组多样化的提示词将有助于确定模型在训练过程中是否正在崩溃。在此示例中，单词 `<token>` 应替换为您的主题名称（instance_prompt）。

<details>
<summary>查看示例配置</summary>

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing her essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing her unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing her dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```
</details>

> ℹ️ Flux 是一个流匹配模型，较短且高度相似的提示词将导致模型生成几乎相同的图像。请确保使用更长、更具描述性的提示词。

#### CLIP 分数跟踪

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

#### Flux 时间调度偏移

Flux 和 SD3 等流匹配模型具有一个名为"shift"的属性，允许我们使用简单的十进制值来偏移训练的时间步调度部分。

##### 默认值

默认情况下，不对 flux 应用调度偏移，这会导致时间步采样分布呈 sigmoid 钟形。这对于 Flux 来说可能不是理想的方法，但它比自动偏移在更短的时间内产生更多的学习。

##### 自动偏移

一个常见推荐的方法是遵循多项近期研究并启用分辨率依赖的时间步偏移，`--flow_schedule_auto_shift`，它对较大的图像使用较高的偏移值，对较小的图像使用较低的偏移值。这会产生稳定但可能平庸的训练结果。

##### 手动指定

（_感谢 Discord 上的 General Awareness 提供以下示例_）

当使用 `--flow_schedule_shift` 值为 0.1（非常低的值）时，只有图像的精细细节受到影响：
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

当使用 `--flow_schedule_shift` 值为 4.0（非常高的值）时，模型的大型构图特征和可能的色彩空间会受到影响：
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)

#### 量化模型训练

在 Apple 和 NVIDIA 系统上经过测试，Hugging Face Optimum-Quanto 可用于降低精度和 VRAM 要求，仅需 16GB 即可训练 Flux。

对于 `config.json` 用户：

<details>
<summary>查看示例配置</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

##### LoRA 特定设置（非 LyCORIS）

```bash
# 当训练 'mmdit' 时，我们发现训练非常稳定，但模型学习时间更长。
# 当训练 'all' 时，我们可以轻松改变模型分布，但更容易遗忘，并受益于高质量数据。
# 当训练 'all+ffs' 时，除了前馈层外，所有注意力层都会被训练，这有助于为 LoRA 调整模型目标。
# - 据报告此模式缺乏可移植性，ComfyUI 等平台可能无法加载该 LoRA。
# 也提供了仅训练 'context' 块的选项，但其影响未知，作为实验性选择提供。
# - 此模式的扩展 'context+ffs' 也可用，适用于在通过 `--init_lora` 继续微调之前将新标记预训练到 LoRA 中。
# 其他选项包括 'tiny' 和 'nano'，它们仅训练 1 或 2 层。
"--flux_lora_target": "all",

# 如果您想使用 LoftQ 初始化，则不能使用 Quanto 来量化基础模型。
# 这可能提供更好/更快的收敛，但仅适用于 NVIDIA 设备，需要 Bits n Bytes，并且与 Quanto 不兼容。
# 其他选项有 'default'、'gaussian'（困难），以及未测试的选项：'olora' 和 'pissa'。
"--lora_init_type": "loftq",
```

#### 数据集注意事项

> ⚠️ 对于 Flux 来说，训练图像质量比大多数其他模型更重要，因为它会_首先_吸收图像中的伪影，然后才学习概念/主题。

拥有足够大的数据集来训练模型至关重要。数据集大小有限制，您需要确保数据集足够大以有效训练模型。请注意，最小数据集大小为 `train_batch_size * gradient_accumulation_steps`，并且需要大于 `vae_batch_size`。如果数据集太小将无法使用。

> ℹ️ 如果图像数量太少，您可能会看到消息 **no images detected in dataset** - 增加 `repeats` 值可以克服此限制。

根据您拥有的数据集，您需要以不同方式设置数据集目录和数据加载器配置文件。在此示例中，我们将使用 [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) 作为数据集。

创建一个 `--data_backend_config`（`config/multidatabackend.json`）文档，包含以下内容：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "pseudo-camera-10k-flux",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "dreambooth-subject-512",
    "type": "local",
    "crop": false,
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/dreambooth-subject-512",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> 请参阅 [DATALOADER.md](../DATALOADER.md#caption_strategy) 中的 caption_strategy 选项和要求。

> ℹ️ 支持同时运行 512px 和 1024px 数据集，这可能会为 Flux 带来更好的收敛效果。

然后，创建一个 `datasets` 目录：

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

这将下载约 10k 张照片样本到您的 `datasets/pseudo-camera-10k` 目录，该目录将自动为您创建。

您的 Dreambooth 图像应放入 `datasets/dreambooth-subject` 目录。

#### 登录 WandB 和 Huggingface Hub

在开始训练之前，您需要登录 WandB 和 HF Hub，特别是如果您使用 `--push_to_hub` 和 `--report_to=wandb`。

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

从 SimpleTuner 目录，您有多个选项来开始训练：

**选项 1（推荐 - pip 安装）：**

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
simpletuner train
```

**选项 2（Git clone 方式）：**

```bash
simpletuner train
```

**选项 3（传统方式 - 仍然有效）：**

```bash
./train.sh
```

这将开始将文本嵌入和 VAE 输出缓存到磁盘。

有关更多信息，请参阅[数据加载器](../DATALOADER.md)和[教程](../TUTORIAL.md)文档。

**注意：**目前不清楚 Flux 的多宽高比桶训练是否正常工作。建议使用 `crop_style=random` 和 `crop_aspect=square`。

## 多 GPU 配置

SimpleTuner 通过 WebUI 包含**自动 GPU 检测**。在引导过程中，您将配置：

- **自动模式**：自动使用所有检测到的 GPU，并采用最优设置
- **手动模式**：选择特定 GPU 或设置自定义进程数
- **禁用模式**：单 GPU 训练

WebUI 检测您的硬件并自动配置 `--num_processes` 和 `CUDA_VISIBLE_DEVICES`。

有关手动配置或高级设置，请参阅安装指南中的[多 GPU 训练部分](../INSTALL.md#multiple-gpu-training)。

## 推理技巧

### CFG 训练的 LoRA（flux_guidance_value > 1）

在 ComfyUI 中，您需要将 Flux 通过另一个名为 AdaptiveGuider 的节点。我们社区的一位成员在此处提供了一个修改后的节点：

（**外部链接**）[IdiotSandwichTheThird/ComfyUI-Adaptive-Guidan...](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps) 以及他们的示例工作流 [在这里](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps/blob/master/ExampleWorkflow.json)

### CFG 蒸馏的 LoRA（flux_guidance_scale == 1）

推理 CFG 蒸馏的 LoRA 非常简单，只需使用与训练时相近的较低 guidance_scale 值。

## 注意事项和故障排除提示

### 最低 VRAM 配置

目前，最低 VRAM 使用量（9090M）可以通过以下配置实现：

- 操作系统：Ubuntu Linux 24
- GPU：单张 NVIDIA CUDA 设备（10G、12G）
- 系统内存：大约 50G 系统内存
- 基础模型精度：`nf4-bnb`
- 优化器：Lion 8Bit Paged，`bnb-lion8bit-paged`
- 分辨率：512px
  - 1024px 需要 >= 12G VRAM
- 批次大小：1，零梯度累积步骤
- DeepSpeed：禁用/未配置
- PyTorch：2.6 Nightly（9月29日构建）
- 使用 `--quantize_via=cpu` 避免在 <=16G 显卡上启动时出现 outOfMemory 错误。
- 使用 `--attention_mechanism=sageattention` 进一步减少 0.1GB VRAM 并提高训练验证图像生成速度。
- 确保启用 `--gradient_checkpointing`，否则无论您做什么都会 OOM

**注意**：VAE 嵌入和文本编码器输出的预缓存可能使用更多内存并仍然 OOM。如果是这样，可以通过 `--vae_enable_tiling=true` 启用文本编码器量化和 VAE 分块。可以通过 `--offload_during_startup=true` 在启动时进一步节省内存。

在 4090 上速度约为每秒 1.4 次迭代。

### SageAttention

使用 `--attention_mechanism=sageattention` 时，可以在验证时加速推理。

**注意**：这与_每个_模型配置都不兼容，但值得尝试。

### NF4 量化训练

简单来说，NF4 是模型的 4 位_左右_表示，这意味着训练有严重的稳定性问题需要解决。

在早期测试中，以下结论成立：

- Lion 优化器导致模型崩溃但使用最少的 VRAM；AdamW 变体有助于保持稳定；bnb-adamw8bit、adamw_bf16 是很好的选择
  - AdEMAMix 效果不佳，但未探索设置
- `--max_grad_norm=0.01` 进一步有助于减少模型损坏，防止在太短时间内对模型进行巨大更改
- NF4、AdamW8bit 和更大的批次大小都有助于克服稳定性问题，代价是更多的训练时间或 VRAM 使用
- 将分辨率从 512px 提升到 1024px 会使训练从每步 1.4 秒减慢到每步 3.5 秒（批次大小为 1，4090）
- 在 int8 或 bf16 上难以训练的内容在 NF4 上会更困难
- 它与 SageAttention 等选项的兼容性较差

NF4 不适用于 torch.compile，所以您获得的速度就是您得到的。

如果 VRAM 不是问题（例如 48G 或更大），那么带有 torch.compile 的 int8 是您最好、最快的选择。

### 掩码损失

如果您正在训练主题或风格并想对其中之一进行掩码，请参阅 Dreambooth 指南中的[掩码损失训练](../DREAMBOOTH.md#masked-loss)部分。

### TREAD 训练 {#tread-training}

> ⚠️ **实验性**：TREAD 是一个新实现的功能。虽然功能正常，但最优配置仍在探索中。

[TREAD](../TREAD.md)（论文）代表 **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion。这是一种通过智能地将令牌路由通过 transformer 层来加速 Flux 训练的方法。加速与您丢弃的令牌数量成正比。

#### 快速设置

将此添加到您的 `config.json`：

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

此配置将：

- 在第 2 层到倒数第二层期间仅保留 50% 的图像令牌
- 文本令牌永远不会被丢弃
- 训练加速约 25%，对质量影响最小

#### 要点

- **有限的架构支持** - TREAD 仅为 Flux 和 Wan 模型实现
- **在高分辨率下效果最佳** - 由于注意力的 O(n²) 复杂度，在 1024x1024+ 时获得最大加速
- **与掩码损失兼容** - 掩码区域会自动保留（但这会减少加速）
- **适用于量化** - 可以与 int8/int4/NF4 训练结合使用
- **预期初始损失峰值** - 开始 LoRA/LoKr 训练时，损失最初会较高，但会快速纠正

#### 调优提示

- **保守（注重质量）**：使用 `selection_ratio` 0.3-0.5
- **激进（注重速度）**：使用 `selection_ratio` 0.6-0.8
- **避免早期/后期层**：不要在 0-1 层或最后一层进行路由
- **对于 LoRA 训练**：可能会看到轻微减速 - 尝试不同的配置
- **分辨率越高 = 加速越好**：在 1024px 及以上最有益

#### 已知行为

- 丢弃的令牌越多（更高的 `selection_ratio`），训练越快但初始损失越高
- LoRA/LoKr 训练显示初始损失峰值，随着网络适应会快速纠正
- 一些 LoRA 配置可能训练略慢 - 最优配置仍在探索中
- RoPE（旋转位置嵌入）实现可用但可能不是 100% 正确

有关详细配置选项和故障排除，请参阅[完整 TREAD 文档](../TREAD.md)。

### 无分类器引导

#### 问题

Dev 模型开箱即用时是引导蒸馏的，这意味着它以非常直接的轨迹到达教师模型输出。这是通过在训练和推理时馈入模型的引导向量完成的 - 此向量的值极大地影响您最终获得的 LoRA 类型：

#### 解决方案

- 值为 1.0（**默认值**）将保留对 Dev 模型进行的初始蒸馏
  - 这是最兼容的模式
  - 推理速度与原始模型一样快
  - 流匹配蒸馏降低了模型的创造力和输出变化性，与原始 Flux Dev 模型一样（一切都保持相同的构图/外观）
- 更高的值（测试约为 3.5-4.5）将重新引入 CFG 目标到模型中
  - 这需要推理管道支持 CFG
  - 推理慢 50% 且 VRAM 增加 0% **或者**由于批处理 CFG 推理慢约 20% 且 VRAM 增加 20%
  - 然而，这种训练风格提高了创造力和模型输出变化性，这可能是某些训练任务所需的

我们可以通过使用向量值 1.0 继续调优您的模型，部分地将蒸馏重新引入去蒸馏的模型。它永远不会完全恢复，但至少会更可用。

#### 注意事项

- 这最终的影响是**以下之一**：
  - 当我们按顺序计算无条件输出时，推理延迟增加 2 倍，例如通过两次单独的前向传递
  - VRAM 消耗增加相当于使用 `num_images_per_prompt=2` 并在推理时接收两张图像，伴随着相同百分比的减速。
    - 这通常比顺序计算的减速更温和，但 VRAM 使用可能对大多数消费级训练硬件来说太高。
    - 此方法目前_尚未_集成到 SimpleTuner 中，但工作正在进行中。
- ComfyUI 或其他应用程序（如 AUTOMATIC1111）的推理工作流需要修改以也启用"真正的" CFG，这可能目前无法开箱即用。

### 量化

- 16G 显卡训练此模型需要最低 8 位量化
  - 在 bfloat16/float16 中，rank-1 LoRA 的内存使用略高于 30GB
- 将模型量化到 8 位不会损害训练
  - 它允许您推送更大的批次大小并可能获得更好的结果
  - 行为与全精度训练相同 - fp32 不会使您的模型比 bf16+int8 更好。
- **int8** 在较新的 NVIDIA 硬件（3090 或更高）上具有硬件加速和 `torch.compile()` 支持
- **nf4-bnb** 将 VRAM 要求降至 9GB，可以装入 10G 显卡（需要 bfloat16 支持）
- 稍后在 ComfyUI 中加载 LoRA 时，您**必须**使用与训练 LoRA 时相同的基础模型精度。
- **int4** 依赖于自定义 bf16 内核，如果您的显卡不支持 bfloat16 将无法工作

### 崩溃

- 如果在文本编码器卸载后收到 SIGKILL，这意味着您没有足够的系统内存来量化 Flux。
  - 尝试加载 `--base_model_precision=bf16`，但如果不起作用，您可能只需要更多内存..
  - 尝试 `--quantize_via=accelerator` 来使用 GPU 代替

### Schnell

- 如果您在 Dev 上训练 LyCORIS LoKr，它**通常**在 Schnell 上仅需 4 步就能很好地工作。
  - 直接 Schnell 训练确实需要更多时间来完善 - 目前，结果看起来不太好

> ℹ️ 以任何方式将 Schnell 与 Dev 合并时，Dev 的许可证将接管，并变为非商业用途。这对大多数用户来说应该不重要，但值得注意。

### 学习率

#### LoRA（--lora_type=standard）

- LoRA 对于较大数据集的整体性能比 LoKr 差
- 据报告，Flux LoRA 训练类似于 SD 1.5 LoRA
- 然而，像 12B 这样大的模型在经验上使用**较低的学习率**表现更好。
  - 1e-3 的 LoRA 可能会完全烧掉它。1e-5 的 LoRA 几乎什么都不做。
- 由于随着基础模型大小增加而扩大的一般困难，像 64 到 128 这样大的 rank 在 12B 模型上可能不太理想。
  - 首先尝试较小的网络（rank-1、rank-4）然后逐步增加 - 它们训练更快，可能能满足您的所有需求。
  - 如果您发现将概念训练到模型中极其困难，您可能需要更高的 rank 和更多的正则化数据。
- 其他扩散 transformer 模型如 PixArt 和 SD3 主要受益于 `--max_grad_norm`，SimpleTuner 在 Flux 上默认保持相当高的值。
  - 较低的值会防止模型过早崩溃，但也会使学习远离基础模型数据分布的新概念变得非常困难。模型可能会卡住且永不改进。

#### LoKr（--lora_type=lycoris）

- 更高的学习率对 LoKr 更好（AdamW 使用 `1e-3`，Lion 使用 `2e-4`）
- 其他算法需要更多探索。
- 在此类数据集上设置 `is_regularisation_data` 可能有助于保留/防止泄漏并提高最终结果模型的质量。
  - 这与"先验损失保留"的行为不同，后者以加倍训练批次大小且不太改善结果而闻名
  - SimpleTuner 的正则化数据实现提供了一种保留基础模型的有效方式

### 图像伪影

Flux 会立即吸收不良图像伪影。这就是它的特点 - 可能需要仅在高质量数据上进行最终训练运行来在最后修复它。

当您执行以下操作（以及其他操作）时，样本中**可能**开始出现一些方形网格伪影：

- 使用低质量数据过度训练
- 使用过高的学习率
- 过度训练（一般情况），低容量网络配合过多图像
- 训练不足（同样），高容量网络配合过少图像
- 使用非常规的宽高比或训练数据大小

### 宽高比分桶

- 在方形裁剪上训练太久可能不会太损害这个模型。放心使用，它非常好且可靠。
- 另一方面，使用数据集的自然宽高比桶可能会在推理时过度偏向这些形状。
  - 这可能是一个理想的质量，因为它可以防止电影风格等宽高比依赖的风格过多渗入其他分辨率。
  - 然而，如果您想在多个宽高比桶上同等改善结果，您可能需要尝试 `crop_aspect=random`，这有其自身的缺点。
- 通过多次定义您的图像目录数据集来混合数据集配置已产生了非常好的结果和良好泛化的模型。

### 训练自定义微调的 Flux 模型

Hugging Face Hub 上的一些微调 Flux 模型（如 Dev2Pro）缺少完整的目录结构，需要设置这些特定选项。

如果有相关信息，请确保根据创建者的方式设置这些选项 `flux_guidance_value`、`validation_guidance_real` 和 `flux_attention_masked_training`。

<details>
<summary>查看示例配置</summary>

```json
{
    "model_family": "flux",
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_transformer_model_name_or_path": "ashen0209/Flux-Dev2Pro",
    "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_transformer_subfolder": "none",
}
```
</details>
