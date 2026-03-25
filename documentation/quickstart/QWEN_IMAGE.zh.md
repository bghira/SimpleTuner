## Qwen Image 快速入门

> 🆕 想要编辑检查点？请参阅 [Qwen Image Edit 快速入门](./QWEN_EDIT.md) 获取成对参考训练说明。

本示例将训练 Qwen Image 的 LoRA。Qwen Image 是一个 20B 参数的视觉语言模型。由于体积很大，需要采用激进的内存优化。

24GB GPU 是最低配置，即便如此也需要大量量化和谨慎配置。建议 40GB+ 以获得更顺畅体验。

在 24G 上训练时，验证可能会 OOM，除非降低分辨率或使用比 int8 更激进的量化。

### 硬件要求

Qwen Image 是一个 20B 参数模型，仅文本编码器在量化前就消耗 ~16GB VRAM。模型使用自定义 16 通道 VAE。

**重要限制：**
- **不支持 AMD ROCm 或 MacOS**（缺乏高效的 Flash Attention）
- 批大小 > 1 目前无法正确运行；请使用梯度累积
- TREAD（Text-Representation Enhanced Adversarial Diffusion）尚不支持

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

您可能需要修改以下变量：

- `model_type` - 设置为 `lora`。
- `lora_type` - PEFT LoRA 设为 `standard`，LoKr 设为 `lycoris`。
- `model_family` - 设置为 `qwen_image`。
- `model_flavour` - 设置为 `v1.0`。
- `output_dir` - 设置为您想要存储检查点和验证图像的目录。建议使用完整路径。
- `train_batch_size` - 按可用 VRAM 设置。当前 SimpleTuner 的 Qwen override 已支持大于 1 的批大小。
- `gradient_accumulation_steps` - 如果想在不增加单步 VRAM 的情况下提升有效 batch，可设为 2-8。
- `validation_resolution` - 建议 `1024x1024` 或更低，以适应内存限制。
  - 24G 无法处理 1024x1024 验证，需要降低尺寸
  - 其他分辨率可用逗号分隔：`1024x1024,768x768,512x512`
- `validation_guidance` - 使用 3.0-4.0 左右。
- `validation_num_inference_steps` - 约 30。
- `use_ema` - 设为 `true` 可获得更平滑的结果，但会占用更多内存。

- `optimizer` - 推荐 `optimi-lion`，如有余量可用 `adamw-bf16`。
- `mixed_precision` - Qwen Image 必须设为 `bf16`。
- `gradient_checkpointing` - **必须**启用（`true`）以获得合理内存占用。
- `base_model_precision` - **强烈推荐**设为 `int8-quanto` 或 `nf4-bnb`（24GB 显卡）。
- `quantize_via` - 设为 `cpu`，避免小显卡量化时 OOM。
- `quantize_activations` - 保持 `false` 以维持训练质量。

24GB GPU 的内存优化建议：
- `lora_rank` - 使用 8 或更低。
- `lora_alpha` - 与 lora_rank 相同。
- `flow_schedule_shift` - 设为 1.73（或在 1.0-3.0 间探索）。

最小配置示例：

<details>
<summary>查看示例配置</summary>

```json
{
    "model_type": "lora",
    "model_family": "qwen_image",
    "model_flavour": "v1.0",
    "lora_type": "standard",
    "lora_rank": 8,
    "lora_alpha": 8,
    "output_dir": "output/models-qwen_image",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "validation_resolution": "1024x1024",
    "validation_guidance": 4.0,
    "validation_num_inference_steps": 30,
    "validation_seed": 42,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_step_interval": 100,
    "vae_batch_size": 1,
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 1000,
    "max_grad_norm": 0.01,
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "learning_rate": "1e-4",
    "gradient_checkpointing": "true",
    "base_model_precision": "int2-quanto",
    "quantize_via": "cpu",
    "quantize_activations": false,
    "flow_schedule_shift": 1.73,
    "disable_benchmark": false,
    "data_backend_config": "config/qwen_image/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ 多 GPU 用户可参考[此文档](../OPTIONS.md#environment-configuration-variables)了解 GPU 数量配置。

> ⚠️ **24GB GPU 关键点**：仅文本编码器就需 ~16GB VRAM。`int2-quanto` 或 `nf4-bnb` 可大幅降低。

快速验证可用以下已知配置：

**选项 1（推荐 - pip 安装）：**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train example=qwen_image.peft-lora
```

**选项 2（Git clone 方式）：**
```bash
simpletuner train env=examples/qwen_image.peft-lora
```

**选项 3（Legacy 方式 - 仍可用）：**
```bash
ENV=examples/qwen_image.peft-lora ./train.sh
```

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

#### 验证提示词

`config/config.json` 中包含“主验证提示词”，通常为你正在训练的主体或风格的 instance_prompt。此外，可创建一个 JSON 文件包含额外验证提示词。

示例配置文件 `config/user_prompt_library.json.example` 格式如下：

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

昵称将作为验证文件名，请保持简短并与文件系统兼容。

要让训练器使用该提示词库，请在 config.json 中添加：
```json
  "validation_prompt_library": "config/user_prompt_library.json",
```

多样化提示词有助于判断模型是否在正常学习：

```json
{
    "anime_style": "a breathtaking anime-style portrait with vibrant colors and expressive features",
    "chef_cooking": "a high-quality, detailed photograph of a sous-chef immersed in culinary creation",
    "portrait": "a lifelike and intimate portrait showcasing unique personality and charm",
    "cinematic": "a cinematic, visually stunning photo with dramatic and captivating presence",
    "elegant": "an elegant and timeless portrait exuding grace and sophistication",
    "adventurous": "a dynamic and adventurous photo captured in an exciting moment",
    "mysterious": "a mysterious and enigmatic portrait shrouded in shadows and intrigue",
    "vintage": "a vintage-style portrait evoking the charm and nostalgia of a bygone era",
    "artistic": "an artistic and abstract representation blending creativity with visual storytelling",
    "futuristic": "a futuristic and cutting-edge portrayal set against advanced technology"
}
```

#### CLIP 分数跟踪

如需启用评估以评分模型性能，请参阅[此文档](../evaluation/CLIP_SCORES.md)。

#### 稳定评估损失

如需使用稳定的 MSE 损失评估模型性能，请参阅[此文档](../evaluation/EVAL_LOSS.md)。

#### 验证预览

SimpleTuner 支持使用 Tiny AutoEncoder 在生成过程中流式输出中间验证预览。这样可以通过 webhook 回调实时查看逐步生成的验证图像。

启用方式：
```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

**要求：**
- Webhook 配置
- 验证已启用

将 `validation_preview_steps` 提高（例如 3 或 5）可降低 Tiny AutoEncoder 开销。若 `validation_num_inference_steps=20` 且 `validation_preview_steps=5`，你会在第 5、10、15、20 步收到预览图。

#### Flow schedule shifting

Qwen Image 是流匹配模型，支持通过时间表偏移来控制训练覆盖的生成过程部分。

`flow_schedule_shift` 参数控制：
- 较低值（0.1-1.0）：关注细节
- 中等值（1.0-3.0）：平衡训练（推荐）
- 较高值（3.0-6.0）：关注大构图特征

##### 自动偏移

可启用分辨率相关时间步偏移 `--flow_schedule_auto_shift`。它对大图使用更高 shift 值，对小图使用更低值，结果更稳定但可能较为中庸。

##### 手动指定

`--flow_schedule_shift` 的起始推荐值为 1.73，但需根据数据集和目标自行调整。

#### 数据集注意事项

模型训练需要足够大的数据集。数据集规模存在限制，你必须确保数据集足够大才能有效训练模型。

> ℹ️ 若图像过少，可能出现 **no images detected in dataset** 提示——增加 `repeats` 值可解决。

> ⚠️ **重要**：由于当前限制，请保持 `train_batch_size` 为 1，用 `gradient_accumulation_steps` 模拟更大 batch。

创建 `--data_backend_config`（`config/multidatabackend.json`）文档如下：

```json
[
  {
    "id": "pseudo-camera-10k-qwen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/pseudo-camera-10k",
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
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/dreambooth-subject",
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
    "cache_dir": "cache/text/qwen_image",
    "disabled": false,
    "write_batch_size": 16
  }
]
```

> ℹ️ 如果你有包含 caption 的 `.txt` 文件，请使用 `caption_strategy=textfile`。
> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).
> ℹ️ 注意 text embeds 使用较小的 `write_batch_size` 以避免 OOM。

然后创建 `datasets` 目录：

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

这将把约 10k 张照片样本下载到 `datasets/pseudo-camera-10k` 目录，并自动创建。

Dreambooth 图片应放到 `datasets/dreambooth-subject`。

#### 登录 WandB 与 Huggingface Hub

在训练开始前登录 WandB 与 HF Hub，尤其当你使用 `--push_to_hub` 和 `--report_to=wandb` 时。

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

</details>

### 执行训练

在 SimpleTuner 目录中，直接运行：

```bash
./train.sh
```

这将开始将文本嵌入与 VAE 输出缓存到磁盘。

更多信息请参阅 [dataloader](../DATALOADER.md) 和 [tutorial](../TUTORIAL.md) 文档。

### 内存优化建议

#### 最低 VRAM 配置（24GB 最低）

Qwen Image 的最低 VRAM 配置约为 24GB：

- OS: Ubuntu Linux 24
- GPU: 单张 NVIDIA CUDA（至少 24GB）
- 系统内存: 建议 64GB+
- 基础模型精度:
  - NVIDIA 系统：`int2-quanto` 或 `nf4-bnb`（24GB 必需）
  - `int4-quanto` 可用但质量可能更低
- 优化器：`optimi-lion` 或 `bnb-lion8bit-paged` 更省内存
- 分辨率：先用 512px 或 768px，内存允许再升到 1024px
- 批大小：1（当前限制）
- 梯度累积：2-8 模拟更大 batch
- 启用 `--gradient_checkpointing`（必需）
- 使用 `--quantize_via=cpu` 避免启动 OOM
- 使用较小 LoRA rank（1-8）
- 设置环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 有助于减少 VRAM

**注意**：预缓存 VAE 嵌入与文本编码器输出会占用大量内存。若 OOM，可启用 `offload_during_startup=true`。

### 训练后的 LoRA 推理

由于 Qwen Image 是新模型，以下为可用推理示例：

<details>
<summary>Show Python inference example</summary>

```python
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

model_id = 'Qwen/Qwen-Image'
adapter_id = 'your-username/your-lora-name'

# Load the pipeline
pipeline = QwenImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
pipeline.load_lora_weights(adapter_id)

# Optional: quantize the model to save VRAM
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

# Move to device
pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

# Generate an image
prompt = "Your test prompt here"
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=4.0,
    generator=torch.Generator(device='cuda').manual_seed(42),
    width=1024,
    height=1024,
).images[0]

image.save("output.png", format="PNG")
```
</details>

### 注意事项与排错提示

#### 批大小限制

较旧的 diffusers Qwen 实现曾因文本嵌入 padding 和 attention mask 处理问题而无法稳定使用批大小 > 1。当前 SimpleTuner 的 Qwen override 已同时修复这两条路径，因此在 VRAM 允许时可以使用更大的 batch。
- 只有在确认显存余量足够后再提高 `train_batch_size`。
- 如果旧环境里仍然出现伪影，请更新并重新生成旧的 text embeds。

#### 量化

- `int2-quanto` 内存节省最多，但可能影响质量
- `nf4-bnb` 在内存与质量之间更平衡
- `int4-quanto` 为折中方案
- 除非有 40GB+ VRAM，否则避免 `int8`

#### 学习率

LoRA 训练：
- 小 LoRA（rank 1-8）：学习率约 1e-4
- 大 LoRA（rank 16-32）：学习率约 5e-5
- 使用 Prodigy 优化器时：从 1.0 开始自适应

#### 图像伪影

若出现伪影：
- 降低学习率
- 提高梯度累积
- 确保图像质量高且预处理正确
- 初期使用较低分辨率

#### 多分辨率训练

先用低分辨率（512px 或 768px）训练，再在 1024px 上微调。不同分辨率训练时建议启用 `--flow_schedule_auto_shift`。

### 平台限制

**不支持：**
- AMD ROCm（缺乏高效 Flash Attention 实现）
- Apple Silicon/MacOS（内存与注意力限制）
- VRAM < 24GB 的消费级 GPU

### 当前已知问题

1. 批大小 > 1 无法正常工作（请使用梯度累积）
2. 尚不支持 TREAD
3. 文本编码器内存占用高（量化前约 16GB）
4. 序列长度处理问题（[上游问题](https://github.com/huggingface/diffusers/issues/12075)）

如需更多帮助与排查，请参阅 [SimpleTuner 文档](/documentation) 或加入社区 Discord。
