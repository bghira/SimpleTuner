## Cosmos2 Predict（Image）快速入门

本示例将训练 Cosmos2 Predict（Image）的 Lycoris LoKr，这是 NVIDIA 的流匹配模型。

### 硬件要求

Cosmos2 Predict（Image）是基于视觉 Transformer 的流匹配模型。

**注意**：由于其架构原因，训练时不应量化，因此需要足够 VRAM 来容纳完整 bf16 精度。

建议至少 24GB GPU 作为舒适训练的最低配置。

### 内存卸载（可选）

若需将 Cosmos2 挤入更小 GPU，可启用分组卸载：

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# 可选：将卸载权重写入磁盘而非 RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams 仅在 CUDA 生效；其他设备会自动回退。
- 不要与 `--enable_model_cpu_offload` 同用。
- 磁盘暂存可选，当系统内存是瓶颈时有帮助。

### 前提条件

确保已安装 Python；SimpleTuner 在 3.10 到 3.12 版本上运行良好。

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
- `model_family` - 设置为 `cosmos2image`。
- `base_model_precision` - **重要**：设为 `no_change`，Cosmos2 不应量化。
- `output_dir` - 设置为您想要存储检查点和验证图像的目录。建议使用完整路径。
- `train_batch_size` - 从 1 开始，VRAM 足够时可增加。
- `validation_resolution` - Cosmos2 默认 `1024x1024`。
  - 其他分辨率可用逗号分隔：`1024x1024,768x768`
- `validation_guidance` - Cosmos2 建议使用 4.0 左右。
- `validation_num_inference_steps` - 约 20 步。
- `use_ema` - 设为 `true` 可获得更平滑结果并保留主检查点。
- `optimizer` - 示例使用 `adamw_bf16`。
- `mixed_precision` - 建议设为 `bf16` 以获得最佳效率。
- `gradient_checkpointing` - 开启以降低 VRAM，代价是训练更慢。

示例 config.json：

<details>
<summary>查看示例配置</summary>

```json
{
    "base_model_precision": "no_change",
    "checkpoint_step_interval": 500,
    "data_backend_config": "config/cosmos2image/multidatabackend.json",
    "disable_bucket_pruning": true,
    "flow_schedule_shift": 0.0,
    "flow_schedule_auto_shift": true,
    "gradient_checkpointing": true,
    "hub_model_id": "cosmos2image-lora",
    "learning_rate": 6e-5,
    "lora_type": "lycoris",
    "lycoris_config": "config/cosmos2image/lycoris_config.json",
    "lr_scheduler": "constant",
    "lr_warmup_steps": 100,
    "max_train_steps": 10000,
    "model_family": "cosmos2image",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/cosmos2image",
    "push_checkpoints_to_hub": false,
    "push_to_hub": false,
    "quantize_via": "cpu",
    "report_to": "tensorboard",
    "seed": 42,
    "tracker_project_name": "cosmos2image-training",
    "tracker_run_name": "cosmos2image-lora",
    "train_batch_size": 1,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_disable_unconditional": true,
    "validation_guidance": 4.0,
    "validation_guidance_rescale": 0.0,
    "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
    "validation_num_inference_steps": 20,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_prompt_library": false,
    "validation_resolution": "512x512",
    "validation_seed": 42,
    "validation_step_interval": 500
}
```
</details>

> ℹ️ 多 GPU 用户可参考[此文档](../OPTIONS.md#environment-configuration-variables)了解 GPU 数量配置。

`config/cosmos2image/lycoris_config.json` 示例：

<details>
<summary>查看示例配置</summary>

```json
{
    "bypass_mode": true,
    "algo": "lokr",
    "multiplier": 1.0,
    "full_matrix": true,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 4,
    "apply_preset": {
        "target_module": [
            "Attention"
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 4
            }
        }
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

`config/config.json` 中包含“主验证提示词”，通常是你正在训练的单一主体或风格的主 instance_prompt。此外，可创建 JSON 文件包含额外验证提示词。

示例配置文件 `config/user_prompt_library.json.example` 格式如下：

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

昵称将作为验证文件名，请保持简短并与文件系统兼容。

要让训练器使用该提示词库，请在配置中设置：
```json
"validation_prompt_library": "config/user_prompt_library.json"
```

多样化提示词有助于判断模型是否崩溃。本示例中将 `<token>` 替换为你的主体名称（instance_prompt）。

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```

#### CLIP 分数跟踪

如需启用评估以评分模型性能，请参阅[此文档](../evaluation/CLIP_SCORES.md)。

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

Cosmos2 为流匹配模型，具有“shift”属性，可用小数值移动时间步分布中参与训练的部分。

配置默认启用 `flow_schedule_auto_shift`，基于分辨率调整 shift：大图更高、小图更低。

#### 数据集注意事项

模型训练需要足够大的数据集。最小数据集规模为 `train_batch_size * gradient_accumulation_steps` 且必须大于 `vae_batch_size`，过小将无法使用。

> ℹ️ 若图像过少，可能出现 **no images detected in dataset** 提示——增加 `repeats` 值可解决。

根据你拥有的数据集，需要以不同方式设置数据集目录和数据加载器配置文件。本示例使用 [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) 数据集。

创建 `--data_backend_config`（`config/cosmos2image/multidatabackend.json`）文档如下：

```json
[
  {
    "id": "pseudo-camera-10k-cosmos2",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/cosmos2/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/cosmos2/dreambooth-subject",
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
    "cache_dir": "cache/text/cosmos2",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> ℹ️ 如果你有包含 caption 的 `.txt` 文件，请使用 `caption_strategy=textfile`。
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

</details>

### 执行训练

从 SimpleTuner 目录可选择以下方式启动训练：

**选项 1（推荐 - pip 安装）：**
```bash
pip install simpletuner[cuda]
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

### LoKr 推理

由于 Cosmos2 仍较新，推理示例可能需调整。基本示例如下：

<details>
<summary>Show Python inference example</summary>

```py
import torch
from lycoris import create_lycoris_from_weights

# Model and adapter paths
model_id = 'nvidia/Cosmos-1.0-Predict-Image-Text2World-12B'
adapter_repo_id = 'your-username/your-cosmos2-lora'
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

# Load the model and adapter

import torch
from diffusers import Cosmos2TextToImagePipeline

# Available checkpoints: nvidia/Cosmos-Predict2-2B-Text2Image, nvidia/Cosmos-Predict2-14B-Text2Image
model_id = "nvidia/Cosmos-Predict2-2B-Text2Image"
adapter_repo_id = "youruser/your-repo-name"

adapter_file_path = download_adapter(repo_id=adapter_repo_id)
pipe = Cosmos2TextToImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

pipe.to("cuda")

prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

output = pipe(
    prompt=prompt, negative_prompt=negative_prompt, generator=torch.Generator().manual_seed(1)
).images[0]
output.save("output.png")

```

</details>

## 注意事项与排错提示

### 内存方面

Cosmos2 训练无法量化，内存占用高于量化模型。降低 VRAM 关键设置：

- 启用 `gradient_checkpointing`
- 批大小使用 1
- 若内存紧张，可尝试 `adamw_8bit` 优化器
- 设置环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 有助于多宽高比训练时减少 VRAM
- 使用 `--vae_cache_disable` 进行在线 VAE 编码避免磁盘缓存，但会增加训练时间/内存压力

### 训练注意事项

Cosmos2 是新模型，最佳训练参数仍在探索中：

- 示例使用 AdamW 学习率 `6e-5`
- 启用 Flow schedule auto-shift 以处理多分辨率
- 使用 CLIP 评估监控训练进度

### 宽高比分桶

配置中 `disable_bucket_pruning` 为 true，可根据数据集情况调整。

### 多分辨率训练

模型可先在 512px 训练，再在更高分辨率训练。`flow_schedule_auto_shift` 有助于多分辨率训练。

### Masked loss

如需对主体或风格进行遮罩训练，请参阅 Dreambooth 指南中的 [masked loss training](../DREAMBOOTH.md#masked-loss)。

### 已知限制

- 尚未实现系统提示词处理
- 可训练性特征仍在探索
- 不支持量化，应避免使用
