## Auraflow 快速入门

在本示例中，我们将训练一个 Auraflow 的 Lycoris LoKr。

由于 6B 参数量，该模型的全量微调需要大量 VRAM，您需要使用 [DeepSpeed](../DEEPSPEED.md) 才能完成。

### 硬件要求

Auraflow v0.3 是一个 6B 参数的 MMDiT，使用 Pile T5 进行编码文本表示，使用 4 通道 SDXL VAE 进行潜在图像表示。

该模型推理速度较慢，但训练速度尚可。

### 内存卸载（可选）

Auraflow 可以从新的分组卸载路径中获得显著收益。如果您仅有单张 24G（或更小）GPU，请在训练参数中添加以下内容：

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# 可选：将卸载的权重溢出到磁盘而不是 RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- 在非 CUDA 后端上，流会自动禁用，因此该命令在 ROCm 和 MPS 上可以安全重用。
- 不要将此与 `--enable_model_cpu_offload` 结合使用。
- 磁盘卸载以吞吐量换取较低的主机 RAM 压力；请将其放在本地 SSD 上以获得最佳效果。

### 前提条件

确保您已安装 Python；SimpleTuner 在 3.10 到 3.12 版本上运行良好。

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
- `lora_type` - 设置为 `lycoris`。
- `model_family` - 设置为 `auraflow`。
- `model_flavour` - 设置为 `pony`，或留空以使用默认模型。
- `output_dir` - 设置为您想要存储检查点和验证图像的目录。建议使用完整路径。
- `train_batch_size` - 对于 24G 显卡，1 到 4 应该可以工作。
- `validation_resolution` - 您应该设置为 `1024x1024` 或 Auraflow 支持的其他分辨率之一。
  - 可以使用逗号分隔指定其他分辨率：`1024x1024,1280x768,1536x1536`
  - 请注意，Auraflow 的位置嵌入有些特殊，使用多尺度图像（多个基础分辨率）进行训练的结果尚不确定。
- `validation_guidance` - 使用您在 Auraflow 推理时习惯选择的值；较低的值（约 3.5-4.0）会产生更逼真的结果
- `validation_num_inference_steps` - 使用约 30-50 步
- `use_ema` - 将此设置为 `true` 将大大有助于在主训练检查点旁边获得更平滑的结果。

- `optimizer` - 您可以使用任何您熟悉的优化器，但我们将在此示例中使用 `optimi-lion`。
  - Pony Flow 的作者建议使用 `adamw_bf16` 以获得最少的问题和最稳定可靠的训练结果
  - 我们在此演示中使用 Lion 以帮助您看到模型更快地训练，但对于长期运行，`adamw_bf16` 将是一个安全的选择。
- `learning_rate` - 对于使用 Lycoris LoKr 的 Lion 优化器，`4e-5` 是一个很好的起点。
  - 如果您选择 `adamw_bf16`，您需要将学习率设置为约 10 倍，即 `2.5e-4`
  - 较小的 Lycoris/LoRA rank 需要**更高的学习率**，较大的 Lycoris/LoRA 需要**较低的学习率**
- `mixed_precision` - 建议设置为 `bf16` 以获得最高效的训练配置，或设置为 `no` 以获得更好的结果（但会消耗更多内存且速度更慢）。
- `gradient_checkpointing` - 禁用此选项会最快，但限制您的批次大小。必须启用此选项才能获得最低的 VRAM 使用量。


这些选项的影响目前尚不清楚。

您的 config.json 最终会类似于这样：

<details>
<summary>查看示例配置</summary>

```json
{
    "validation_torch_compile": "false",
    "validation_step_interval": 200,
    "validation_seed": 42,
    "validation_resolution": "1024x1024",
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_num_inference_steps": "20",
    "validation_guidance": 2.0,
    "validation_guidance_rescale": "0.0",
    "vae_cache_ondemand": true,
    "vae_batch_size": 1,
    "train_batch_size": 1,
    "tracker_run_name": "eval_loss_test1",
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "output_dir": "output/models-auraflow",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "auraflow",
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 10000,
    "max_grad_norm": 0.01,
    "lycoris_config": "config/lycoris_config.json",
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant",
    "lora_type": "lycoris",
    "learning_rate": "4e-5",
    "gradient_checkpointing": "true",
    "grad_clip_method": "value",
    "eval_steps_interval": 100,
    "disable_benchmark": false,
    "data_backend_config": "config/auraflow/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "int8-quanto",
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ 多 GPU 用户可以参考[此文档](../OPTIONS.md#environment-configuration-variables)了解如何配置使用的 GPU 数量。

以及一个简单的 `config/lycoris_config.json` 文件：

<details>
<summary>查看示例配置</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 16,
    "apply_preset": {
        "target_module": [
            "Attention",
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 8
            },
        }
    }
}
```
</details>

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样 (Rollout)](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

#### 验证提示词

`config/config.json` 中包含"主验证提示词"，通常是您针对单个主题或风格训练的主 instance_prompt。此外，可以创建一个 JSON 文件，包含验证期间运行的额外提示词。

示例配置文件 `config/user_prompt_library.json.example` 包含以下格式：

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

昵称是验证的文件名，因此请保持简短并与您的文件系统兼容。

要将训练器指向此提示词库，请在 `config.json` 末尾添加新行将其添加到 TRAINER_EXTRA_ARGS：
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

一组多样化的提示词将有助于确定模型在训练过程中是否正在崩溃。在此示例中，单词 `<token>` 应替换为您的主题名称（instance_prompt）。

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

> ℹ️ Auraflow 默认使用 128 个 token，然后截断。

#### CLIP 分数跟踪

如果您希望启用评估来评分模型性能，请参阅[此文档](../evaluation/CLIP_SCORES.md)了解如何配置和解释 CLIP 分数。

</details>

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

#### 流调度偏移

Flux、Sana、SD3 和 OmniGen 等流匹配模型具有一个名为"shift"的属性，允许我们使用简单的十进制值来偏移训练的时间步调度部分。

##### 自动偏移
一个常见推荐的方法是遵循多项近期研究并启用分辨率依赖的时间步偏移 `--flow_schedule_auto_shift`，它对较大的图像使用较高的偏移值，对较小的图像使用较低的偏移值。这会产生稳定但可能平庸的训练结果。

##### 手动指定
_感谢 Discord 上的 General Awareness 提供以下示例_

当使用 `--flow_schedule_shift` 值为 0.1（非常低的值）时，只有图像的精细细节受到影响：
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

当使用 `--flow_schedule_shift` 值为 4.0（非常高的值）时，模型的大型构图特征和可能的色彩空间会受到影响：
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### 数据集注意事项

拥有足够大的数据集来训练模型至关重要。数据集大小有限制，您需要确保数据集足够大以有效训练模型。请注意，最小数据集大小为 `train_batch_size * gradient_accumulation_steps`，并且需要大于 `vae_batch_size`。如果数据集太小将无法使用。

> ℹ️ 如果图像数量太少，您可能会看到消息 **no images detected in dataset** - 增加 `repeats` 值可以克服此限制。

根据您拥有的数据集，您需要以不同方式设置数据集目录和数据加载器配置文件。在此示例中，我们将使用 [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) 作为数据集。

创建一个 `--data_backend_config`（`config/multidatabackend.json`）文档，包含以下内容：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "pseudo-camera-10k-auraflow",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/auraflow/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/auraflow/dreambooth-subject",
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
    "cache_dir": "cache/text/auraflow",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> ℹ️ 如果您有包含标注的 `.txt` 文件，请使用 `caption_strategy=textfile`。
> 请参阅 [DATALOADER.md](../DATALOADER.md#caption_strategy) 中的 caption_strategy 选项和要求。

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
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
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

### 训练后运行 LoKr 推理

由于这是一个新模型，示例需要进行一些调整才能工作。以下是一个可用的示例：

<details>
<summary>显示 Python 推理示例</summary>

```py
import torch
from helpers.models.auraflow.pipeline import AuraFlowPipeline
from helpers.models.auraflow.transformer import AuraFlowTransformer2DModel
from lycoris import create_lycoris_from_weights
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

model_id = 'terminusresearch/auraflow-v0.3'
adapter_repo_id = 'bghira/auraflow-photo-1mp-Prodigy'
adapter_filename = 'pytorch_lora_weights.safetensors'

def download_adapter(repo_id: str):
    import os
    from huggingface_hub import hf_hub_download
    adapter_filename = "pytorch_lora_weights.safetensors"
    cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
    cleaned_adapter_path = repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
    path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
    os.makedirs(path_to_adapter, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
    )

    return path_to_adapter_file

adapter_file_path = download_adapter(repo_id=adapter_repo_id)
transformer = AuraFlowTransformer2DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, subfolder="transformer")
pipeline = AuraFlowPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    transformer=transformer,
)
lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

prompt = "Place your test prompt here."
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

## Optional: quantise the model to save on vram.
## Note: The model was quantised during training, and so it is recommended to do the same during inference time.
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # the pipeline is already in its target precision level
t5_embeds, negative_t5_embeds, attention_mask, negative_attention_mask = pipeline.encode_prompt(
    prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt, num_images_per_prompt=1
)
# We'll unload the text encoders to save memory.
pipeline.text_encoder.to("meta")
pipeline.text_encoder_2.to("meta")
pipeline.text_encoder_3.to("meta")
model_output = pipeline(
    prompt_embeds=t5_embeds,
    prompt_attention_mask=attention_mask,
    negative_prompt_embeds=negative_t5_embeds,
    negative_prompt_attention_mask=negative_attention_mask,
    num_inference_steps=30,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=3.2,
).images[0]

model_output.save("output.png", format="PNG")

```
</details>

## 注意事项和故障排除提示

### 最低 VRAM 配置

最低 VRAM Auraflow 配置约为 20-22G：

- 操作系统：Ubuntu Linux 24
- GPU：单张 NVIDIA CUDA 设备（10G、12G）
- 系统内存：大约 50G 系统内存（可能更多或更少）
- 基础模型精度：
  - 对于 Apple 和 AMD 系统，使用 `int8-quanto`（或 `fp8-torchao`、`int8-torchao`，它们具有相似的内存使用特征）
    - `int4-quanto` 也可以工作，但您可能会有较低的准确性/较差的结果
  - 对于 NVIDIA 系统，据报告 `nf4-bnb` 工作良好，但会比 `int8-quanto` 慢
- 优化器：Lion 8Bit Paged，`bnb-lion8bit-paged`
- 分辨率：1024px
- 批次大小：1，零梯度累积步骤
- DeepSpeed：禁用/未配置
- PyTorch：2.7+
- 使用 `--quantize_via=cpu` 避免在 <=16G 显卡上启动时出现 outOfMemory 错误。
- 启用 `--gradient_checkpointing`
- 使用小型 LoRA 或 Lycoris 配置（例如 LoRA rank 1 或 Lokr factor 25）

**注意**：VAE 嵌入和文本编码器输出的预缓存可能使用更多内存并仍然 OOM。VAE 分块和切片默认启用。如果您看到 OOM，可能需要启用 `offload_during_startup=true`；否则，您可能只是运气不佳。

在 NVIDIA 4090 上使用 Pytorch 2.7 和 CUDA 12.8，速度约为每秒 3 次迭代

### 遮罩损失

如果您正在训练主题或风格并想对其中之一进行遮罩，请参阅 Dreambooth 指南中的[遮罩损失训练](../DREAMBOOTH.md#masked-loss)部分。

### 量化

Auraflow 对低至 `int4` 精度级别的量化响应良好，如果您负担不起 `bf16`，`int8` 将是质量和稳定性的最佳平衡点。

### 学习率

#### LoRA (--lora_type=standard)

*不支持。*

#### LoKr (--lora_type=lycoris)
- 温和的学习率对 LoKr 更好（AdamW 使用 `1e-4`，Lion 使用 `2e-5`）
- 其他算法需要更多探索。
- 设置 `is_regularisation_data` 对 Auraflow 的影响/效果未知（未测试，但应该没问题）

### 图像伪影

Auraflow 对图像伪影的响应未知，尽管它使用 Flux VAE，并且具有类似的精细细节限制。

如果出现任何图像质量问题，请在 Github 上提交 issue。

### 宽高比分桶

模型的 patch embed 实现的一些限制意味着某些分辨率会导致错误。

实验会很有帮助，以及详尽的错误报告。

### 全秩微调

DeepSpeed 在 Auraflow 上会使用大量系统内存，全量微调可能不会像您希望的那样在学习概念或避免模型崩溃方面表现良好。

建议使用 Lycoris LoKr 代替全秩微调，因为它更稳定且内存占用更低。
