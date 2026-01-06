## NVLabs Sana 快速入门

在本示例中，我们将对 NVLabs Sana 模型进行全量训练。

### 硬件要求

Sana 非常轻量，在 24G 卡上甚至可能不需要启用完整的梯度检查点，因此训练速度非常快！

- **绝对最低** 约 12G VRAM，但本指南可能无法完全覆盖这种配置
- **现实的最低** 是单卡 3090 或 V100
- **理想** 是多张 4090、A6000、L40S 或更高

Sana 的架构与其他 SimpleTuner 可训练模型相比有些特殊：

- 最初 Sana 只能使用 fp16 训练，bf16 会崩溃
  - NVIDIA 的模型作者随后提供了 bf16 可微调权重
- 由于 bf16/fp16 问题，量化在该模型家族上可能更敏感
- SageAttention 目前不能用于 Sana（head_dim 形状暂不支持）
- Sana 训练时损失值非常高，可能需要明显更低的学习率（例如 `1e-5` 左右）
- 训练可能出现 NaN，原因尚不明确

梯度检查点可以释放 VRAM，但会降低训练速度。下图是 4090 + 5800X3D 的测试结果：

![image](https://github.com/user-attachments/assets/310bf099-a077-4378-acf4-f60b4b82fdc4)

SimpleTuner 的 Sana 建模代码允许使用 `--gradient_checkpointing_interval`，每 _n_ 个块做一次检查点，达到上图中展示的效果。

### 前提条件

确保您已安装 Python；SimpleTuner 在 3.10 到 3.12 版本上运行良好。

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

接下来可能需要修改以下变量：

- `model_type` - 设置为 `full`。
- `model_family` - 设置为 `sana`。
- `pretrained_model_name_or_path` - 设置为 `terminusresearch/sana-1.6b-1024px`
- `output_dir` - 设置为您想要存储检查点和验证图像的目录。建议使用完整路径。
- `train_batch_size` - 对于 24G 卡并启用完整梯度检查点时，可高达 6。
- `validation_resolution` - 该 Sana 检查点是 1024px 模型，设置为 `1024x1024` 或 Sana 支持的其他分辨率。
  - 可用逗号分隔指定其他分辨率：`1024x1024,1280x768,2048x2048`
- `validation_guidance` - 使用您在推理时习惯的值。
- `validation_num_inference_steps` - 为获得最佳质量可使用 50 左右；若对结果满意可适当减少。
- `use_ema` - 设为 `true` 有助于在主检查点之外获得更平滑的结果。

- `optimizer` - 可以使用您熟悉的优化器，但本示例使用 `optimi-adamw`。
- `mixed_precision` - 建议设置为 `bf16` 以获得最有效的训练；也可设为 `no`（更耗内存、更慢）。
  - `fp16` 在此不推荐，但某些 Sana 微调可能需要它（会引入额外问题）。
- `gradient_checkpointing` - 禁用最快三，但会限制 batch size。要最低 VRAM 必须启用。
- `gradient_checkpointing_interval` - 若完整梯度检查点过于保守，可设置为 2 或更高，仅每 _n_ 个块做一次检查点。2 表示一半块，3 表示三分之一。

多 GPU 用户可参考[此文档](../OPTIONS.md#environment-configuration-variables)了解 GPU 数量的配置方式。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

#### 验证提示词

在 `config/config.json` 中有“主验证提示词”，通常是你为单个主题或风格训练的主 instance_prompt。此外，可以创建一个 JSON 文件，其中包含在验证期间运行的额外提示词。

示例配置文件 `config/user_prompt_library.json.example` 包含以下格式：

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

昵称是验证的文件名，因此请保持简短并与你的文件系统兼容。

要将训练器指向此提示词库，请通过在 `config.json` 末尾添加新行将其添加到 TRAINER_EXTRA_ARGS：
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

多样化的提示词集合可帮助判断模型是否在训练中崩塌。在此示例中，`<token>` 应替换为你的主体名称（instance_prompt）。

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

> ℹ️ Sana 使用较为特殊的文本编码器配置，这意味着较短的提示词可能会表现很差。

#### CLIP 分数跟踪

如需启用评估以评分模型性能，请参阅[此文档](../evaluation/CLIP_SCORES.md)了解 CLIP 分数的配置与解读。

</details>

# 稳定评估损失

如需使用稳定的 MSE 损失来评分模型性能，请参阅[此文档](../evaluation/EVAL_LOSS.md)了解评估损失的配置与解读。

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

#### Sana 时间调度偏移

Sana、Flux、SD3 等流匹配模型有一个名为 “shift” 的属性，允许我们使用简单的小数值来偏移时间步调度的训练部分。

##### 自动偏移
一种常推荐的方法是遵循最近的几项工作，启用分辨率相关的时间步偏移 `--flow_schedule_auto_shift`，对较大的图像使用较高偏移值，对较小的图像使用较低偏移值。这会产生稳定但可能平庸的训练结果。

##### 手动指定
_感谢 Discord 上的 General Awareness 提供以下示例_

当使用 0.1（非常低的值）的 `--flow_schedule_shift` 值时，只有图像的细节受到影响：
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

当使用 4.0（非常高的值）的 `--flow_schedule_shift` 值时，模型的大型构图特征和潜在的色彩空间会受到影响：
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)

#### 数据集注意事项

> ⚠️ 对 Sana 来说，训练图像质量比多数模型更重要，因为它会先吸收图像中的瑕疵，再学习概念/主体。

拥有足够大的数据集来训练模型至关重要。数据集大小有限制，您需要确保数据集足够大以有效训练模型。请注意，最小数据集大小为 `train_batch_size * gradient_accumulation_steps`，并且要大于 `vae_batch_size`。如果数据集太小将无法使用。

> ℹ️ 当图像数量很少时，可能会看到 **no images detected in dataset** 的消息 — 增加 `repeats` 值可以克服这个限制。

根据您拥有的数据集，您需要以不同方式设置数据集目录和数据加载器配置文件。在此示例中，我们将使用 [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) 作为数据集。

创建一个 `--data_backend_config`（`config/multidatabackend.json`）文档，内容如下：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sana",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/sana/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/sana/dreambooth-subject",
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
    "cache_dir_vae": "cache/vae/sana/dreambooth-subject-512",
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
    "cache_dir": "cache/text/sana",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> 请参阅 [DATALOADER.md](../DATALOADER.md#caption_strategy) 中的 caption_strategy 选项和要求。

> ℹ️ 可以同时运行 512px 和 1024px 数据集，这可能有助于 Sana 的收敛。

然后创建一个 `datasets` 目录：

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

这将下载约 10k 张照片样本到您的 `datasets/pseudo-camera-10k` 目录，该目录将自动为您创建。

Dreambooth 图片应放入 `datasets/dreambooth-subject` 目录。

#### 登录 WandB 和 Huggingface Hub

在开始训练之前，您需要登录 WandB 和 HF Hub，特别是如果您使用 `--push_to_hub` 和 `--report_to=wandb`。

如果您要手动将项目推送到 Git LFS 仓库，您还应该运行 `git config --global credential.helper store`

运行以下命令：

```bash
wandb login
```

以及

```bash
huggingface-cli login
```

按照说明登录这两个服务。

### 执行训练运行

从 SimpleTuner 目录，您可以使用以下方式启动训练：

**选项 1（推荐 - pip 安装）：**
```bash
pip install simpletuner[cuda]
simpletuner train
```

**选项 2（Git clone 方法）：**
```bash
simpletuner train
```

**选项 3（传统方法 - 仍然有效）：**
```bash
./train.sh
```

这将开始将文本嵌入和 VAE 输出缓存到磁盘。

有关更多信息，请参阅[数据加载器](../DATALOADER.md)和[教程](../TUTORIAL.md)文档。

## 注意事项和故障排查提示

### 最低 VRAM 配置

目前最低的 VRAM 使用配置如下：

- OS: Ubuntu Linux 24
- GPU: 单个 NVIDIA CUDA 设备（10G、12G）
- 系统内存: 约 50G
- 基础模型精度: `nf4-bnb`
- 优化器: Lion 8Bit Paged，`bnb-lion8bit-paged`
- 分辨率: 1024px
- 批次大小: 1，零梯度累积
- DeepSpeed: 禁用 / 未配置
- PyTorch: 2.5.1
- 使用 `--quantize_via=cpu` 以避免 <=16G 卡启动时 outOfMemory
- 启用 `--gradient_checkpointing`

**注意**：VAE 嵌入和文本编码器输出的预缓存可能会使用更多内存并仍然 OOM。若如此，可启用文本编码器量化。当前 Sana 的 VAE 平铺可能不可用。对于磁盘空间紧张的大型数据集，可使用 `--vae_cache_disable` 进行在线编码而不缓存到磁盘。

在 4090 上速度约为 1.4 次迭代/秒。

### Masked loss

如果您要训练主体或风格并希望遮罩其一，请参阅 Dreambooth 指南的[遮罩损失训练](../DREAMBOOTH.md#masked-loss)部分。

### 量化

尚未充分测试。

### 学习率

#### LoRA (--lora_type=standard)

*不支持。*

#### LoKr (--lora_type=lycoris)
- LoKr 适合较温和的学习率（AdamW 为 `1e-4`，Lion 为 `2e-5`）
- 其他算法仍需进一步探索
- `is_regularisation_data` 对 Sana 的影响未知（未测试）

### 图像伪影

Sana 对图像伪影的反应尚不清楚。

目前尚不清楚是否会出现常见训练伪影，或其原因。

如出现图像质量问题，请在 GitHub 上提交 issue。

### 宽高比分桶

该模型对宽高比分桶数据的反应未知。建议进行实验。
