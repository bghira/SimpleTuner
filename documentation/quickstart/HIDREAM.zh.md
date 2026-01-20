## HiDream 快速入门

本示例将训练 HiDream 的 Lycoris LoKr，对内存需求较高。

如果不进行大量分块卸载和融合反向传播，24G GPU 可能是最低配置。Lycoris LoKr 同样适用。

### 硬件要求

HiDream 总参数 17B，任一时刻约 8B 激活，通过学习的 MoE 门控分配计算。它使用**四个**文本编码器以及 Flux VAE。

总体而言，该模型架构复杂，似乎是 Flux Dev 的衍生版本，无论是直接蒸馏还是继续微调，都能从部分验证样例看出相似性。

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
pip install 'simpletuner[cuda13]'
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
- `lora_type` - 设置为 `lycoris`。
- `model_family` - 设置为 `hidream`。
- `model_flavour` - 设置为 `full`，因为 `dev` 经过蒸馏，不适合直接训练，除非你愿意打破蒸馏体系。
  - 实际上 `full` 也不容易训练，但它是唯一没有蒸馏的版本。
- `output_dir` - 设置为您想要存储检查点和验证图像的目录。建议使用完整路径。
- `train_batch_size` - 1，也许？
- `validation_resolution` - 设为 `1024x1024` 或 HiDream 支持的其他分辨率。
  - 可用逗号分隔指定其他分辨率：`1024x1024,1280x768,2048x2048`
- `validation_guidance` - 使用你推理时习惯的值；2.5-3.0 的较低值会更写实。
- `validation_num_inference_steps` - 约 30。
- `use_ema` - 设为 `true` 可显著帮助得到更平滑的结果。

- `optimizer` - 可使用熟悉的优化器，本示例使用 `optimi-lion`。
- `mixed_precision` - 推荐 `bf16` 以获得最高效的训练；`no` 也可用但更慢且占用更大内存。
- `gradient_checkpointing` - 关闭会更快但限制批大小；要最小化 VRAM 必须启用。

一些高级 HiDream 选项可在训练中加入 MoE 辅助损失。启用后损失值会明显增大。

- `hidream_use_load_balancing_loss` - 设为 `true` 启用负载均衡损失。
- `hidream_load_balancing_loss_weight` - 辅助损失强度。默认 `0.01`，可设置为 `0.1` 或 `0.2` 更激进。

这些选项影响尚不明确。

最终配置示例：

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
    "validation_guidance": 3.0,
    "validation_guidance_rescale": "0.0",
    "vae_batch_size": 1,
    "train_batch_size": 1,
    "tracker_run_name": "eval_loss_test1",
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "output_dir": "output/models-hidream",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "hidream",
    "offload_during_startup": true,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 10000,
    "max_grad_norm": 0.01,
    "lycoris_config": "config/lycoris_config.json",
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "lora_type": "lycoris",
    "learning_rate": "4e-5",
    "gradient_checkpointing": "true",
    "grad_clip_method": "value",
    "eval_steps_interval": 100,
    "disable_benchmark": false,
    "data_backend_config": "config/hidream/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "int8-quanto",
    "text_encoder_3_precision": "int8-quanto",
    "text_encoder_4_precision": "int8-quanto",
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ 多 GPU 用户可参考[此文档](../OPTIONS.md#environment-configuration-variables)了解 GPU 数量配置。

> ℹ️ 本配置将 T5 (#3) 和 Llama (#4) 文本编码器精度设为 int8，以节省 24G 显卡内存。如有更多内存可将其删除或设为 `no_change`。

并提供一个简单的 `config/lycoris_config.json` 文件 — 为增强训练稳定性，可移除 `FeedForward`。

<details>
<summary>查看示例配置</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 16384,
    "linear_alpha": 1,
    "full_matrix": true,
    "use_scalar": true,
    "factor": 16,
    "apply_preset": {
        "target_module": [
            "Attention",
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 16
            },
        }
    }
}
```
</details>

在 `config/lycoris_config.json` 中设置 `"use_scalar": true` 或在 `config/config.json` 中设置 `"init_lokr_norm": 1e-4` 都会显著加速训练。两者同时启用会稍微降低速度。注意 `init_lokr_norm` 会轻微改变第 0 步的验证图像。

向 `config/lycoris_config.json` 添加 `FeedForward` 模块会训练更多参数，包括所有专家层。但训练专家层似乎很困难。

更容易的方案是只训练专家外的前馈参数，使用如下 `config/lycoris_config.json`：

<details>
<summary>查看示例配置</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 16384,
    "linear_alpha": 1,
    "full_matrix": true,
    "use_scalar": true,
    "factor": 16,
    "apply_preset": {
        "name_algo_map": {
            "double_stream_blocks.*.block.attn*": {
                "factor": 16
            },
            "double_stream_blocks.*.block.ff_t*": {
                "factor": 16
            },
            "double_stream_blocks.*.block.ff_i.shared_experts*": {
                "factor": 16
            },
            "single_stream_blocks.*.block.attn*": {
                "factor": 16
            },
            "single_stream_blocks.*.block.ff_i.shared_experts*": {
                "factor": 16
            }
        },
        "use_fnmatch": true
    }
}
```
</details>

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

#### 验证提示词

`config/config.json` 中包含“主验证提示词”，通常是你正在训练的单一主体或风格的主 instance_prompt。此外，还可以创建 JSON 文件包含额外验证提示词。

示例配置文件 `config/user_prompt_library.json.example` 格式如下：

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

昵称将作为验证文件名，请保持简短并与文件系统兼容。

要让训练器使用该提示词库，请在 `config.json` 末尾向 TRAINER_EXTRA_ARGS 添加新的一行：
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

多样化提示词有助于判断模型是否崩溃。在本示例中，将 `<token>` 替换为你的主体名称（instance_prompt）。

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

> ℹ️ HiDream 默认 128 token，之后会截断。

#### CLIP 分数跟踪

如需启用评估以评分模型性能，请参阅[此文档](../evaluation/CLIP_SCORES.md)。

</details>

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

#### Flow schedule shifting

OmniGen、Sana、Flux 和 SD3 等流匹配模型具有名为 “shift” 的属性，可用一个小数值移动时间步分布中参与训练的部分。

`full` 模型使用 `3.0`，`dev` 使用 `6.0`。

实践中使用如此高的 shift 值往往会破坏模型。`1.0` 是一个不错的起点，但可能偏小，`3.0` 可能过高。

##### 自动偏移

常见的推荐做法是启用分辨率相关的时间步偏移 `--flow_schedule_auto_shift`。它对大图使用更高 shift 值，对小图使用更低值，结果更稳定但可能较为中庸。

##### 手动指定
_感谢 Discord 的 General Awareness 提供以下示例_

当使用 `--flow_schedule_shift` 值 0.1（很低）时，只会影响图像的细节：
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

当使用 `--flow_schedule_shift` 值 4.0（很高）时，大的构图特征甚至色彩空间会受到影响：
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### 数据集注意事项

模型训练需要足够大的数据集。数据集规模存在限制，你必须确保数据集足够大才能有效训练模型。最小数据集规模为 `train_batch_size * gradient_accumulation_steps` 且必须大于 `vae_batch_size`，数据集过小将无法使用。

> ℹ️ 若图像过少，可能出现 **no images detected in dataset** 提示——增加 `repeats` 值可解决。

根据你拥有的数据集，需要以不同方式设置数据集目录和数据加载器配置文件。本示例使用 [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) 数据集。

创建 `--data_backend_config`（`config/multidatabackend.json`）文件如下：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "pseudo-camera-10k-hidream",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/hidream/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/hidream/dreambooth-subject",
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
    "cache_dir": "cache/text/hidream",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> ℹ️ 如有 `.txt` caption 文件，请使用 `caption_strategy=textfile`。
> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

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

### 执行训练

从 SimpleTuner 目录可选择以下方式启动训练：

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

**选项 3（Legacy 方式 - 仍可用）：**
```bash
./train.sh
```

这将开始将文本嵌入与 VAE 输出缓存到磁盘。

更多信息请参阅 [dataloader](../DATALOADER.md) 和 [tutorial](../TUTORIAL.md)。

### 训练后的 LoKr 推理

作为新模型，示例需略作调整。以下为可运行示例：

<details>
<summary>Show Python inference example</summary>

```py
import torch
from helpers.models.hidream.pipeline import HiDreamImagePipeline
from helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
from lycoris import create_lycoris_from_weights
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

llama_repo = "unsloth/Meta-Llama-3.1-8B-Instruct"
model_id = 'HiDream-ai/HiDream-I1-Dev'
adapter_repo_id = 'bghira/hidream5m-photo-1mp-Prodigy'
adapter_filename = 'pytorch_lora_weights.safetensors'

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
    llama_repo,
)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    llama_repo,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)

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
transformer = HiDreamImageTransformer2DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, subfolder="transformer")
pipeline = HiDreamImagePipeline.from_pretrained(
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
t5_embeds, llama_embeds, negative_t5_embeds, negative_llama_embeds, pooled_embeds, negative_pooled_embeds = pipeline.encode_prompt(
    prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt, num_images_per_prompt=1
)
# We'll unload the text encoders to save memory.
pipeline.text_encoder.to("meta")
pipeline.text_encoder_2.to("meta")
pipeline.text_encoder_3.to("meta")
pipeline.text_encoder_4.to("meta")
model_output = pipeline(
    t5_prompt_embeds=t5_embeds,
    llama_prompt_embeds=llama_embeds,
    pooled_prompt_embeds=pooled_embeds,
    negative_t5_prompt_embeds=negative_t5_embeds,
    negative_llama_prompt_embeds=negative_llama_embeds,
    negative_pooled_prompt_embeds=negative_pooled_embeds,
    num_inference_steps=30,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=3.2,
).images[0]

model_output.save("output.png", format="PNG")

```
</details>

## 注意事项与排错提示

### 最低 VRAM 配置

最低 VRAM 配置约为 20-22G：

- OS: Ubuntu Linux 24
- GPU: 单张 NVIDIA CUDA (10G, 12G)
- 系统内存: 约 50G（可能更多或更少）
- 基础模型精度:
  - Apple 与 AMD 系统使用 `int8-quanto`（或 `fp8-torchao`, `int8-torchao` 具有相似内存表现）
    - `int4-quanto` 也可，但精度与效果可能更差
  - NVIDIA 系统 `nf4-bnb` 被报告效果不错，但比 `int8-quanto` 更慢
- 优化器: Lion 8Bit Paged, `bnb-lion8bit-paged`
- 分辨率: 1024px
- 批大小: 1，零梯度累积
- DeepSpeed: 禁用 / 未配置
- PyTorch: 2.7+
- 启动时使用 `--quantize_via=cpu` 以避免 <=16G 卡 OOM
- 启用 `--gradient_checkpointing`
- 使用小型 LoRA 或 Lycoris 配置（如 LoRA rank 1 或 Lokr factor 25）
- 设置环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 有助于多宽高比训练时减少 VRAM

**注意**：预缓存 VAE 嵌入与文本编码器输出可能占用更多内存并导致 OOM。默认已启用 VAE 切片/分块。若仍 OOM，尝试 `offload_during_startup=true`；否则可能无解。

在 NVIDIA 4090 + Pytorch 2.7 + CUDA 12.8 下速度约 3 it/s。

### Masked loss

如需对主体或风格进行遮罩训练，请参阅 Dreambooth 指南中的 [masked loss training](../DREAMBOOTH.md#masked-loss)。

### 量化

虽然 `int8` 在速度/质量/内存平衡上最佳，但也可以使用 `nf4` 和 `int4`。HiDream 不推荐 `int4`，可能导致更差结果，但足够长的训练仍能得到可用模型。

### 学习率

#### LoRA (--lora_type=standard)

- 小 LoRA（rank 1-8）建议较高学习率约 4e-4
- 大 LoRA（rank 64-256）建议较低学习率约 6e-5
- 由于 Diffusers 限制，不支持 `lora_alpha` 与 `lora_rank` 不一致，除非你清楚推理侧如何处理。
  - 推理细节不在本文范围内，但将 `lora_alpha` 设为 1.0 可让不同 rank 使用相同学习率。

#### LoKr (--lora_type=lycoris)

- LoKr 更适合温和学习率（AdamW 1e-4，Lion 2e-5）
- 其他算法仍需探索。
- Prodigy 对 LoRA/LoKr 可能不错，但会高估学习率并使皮肤变得过于平滑。

### 图像伪影

HiDream 对伪影的反应未知，但使用 Flux VAE，细节限制相似。

最常见的问题是学习率过高和/或批大小过小，会导致皮肤过平滑、模糊和像素化等伪影。

### 宽高比分桶

最初模型对宽高比分桶响应较差，但社区已改进该实现。

### 多分辨率训练

可先用较低分辨率（如 512px）进行初始训练以加快速度，但不确定高分辨率泛化效果。先 512px 后 1024px 的顺序训练可能是最佳方案。

在 1024px 之外训练时建议启用 `--flow_schedule_auto_shift`，较低分辨率 VRAM 更少，可用更大批大小。

### 全秩调优

HiDream 的 DeepSpeed 会占用大量系统内存，但在超大系统上可正常全参训练。

推荐使用 Lycoris LoKr 代替全参训练，更稳定且内存占用更低。

PEFT LoRA 适合简单风格，但较难保留细节。
