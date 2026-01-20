## OmniGen 快速入门

本示例将训练 OmniGen 的 Lycoris LoKr，目标是提升通用 T2I 性能（目前不做 edit/instruct 训练）。

### 硬件要求

OmniGen 约 3.8B 参数，体量不大；它使用 SDXL VAE，但不使用文本编码器。相反，OmniGen 以原生 token ID 作为输入，表现为多模态模型。

训练内存占用尚不明确，但预计在 batch size 2 或 3 时可以轻松适配 24G 显卡。模型可量化以进一步节省 VRAM。

与 SimpleTuner 支持的其他模型相比，OmniGen 架构较为特殊：

- 目前仅支持 t2i（文本到图像）训练，模型输出与训练提示词和输入图像对齐。
- 图像到图像训练尚未支持，但未来可能会加入。
  - 该模式可提供第二张图像作为输入，并将其作为输出的条件/参考数据。
- OmniGen 训练时损失值非常高，原因未知。


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
- `model_family` - 设置为 `omnigen`。
- `model_flavour` - 设置为 `v1`。
- `output_dir` - 设置为您想要存储检查点和验证图像的目录。建议使用完整路径。
- `train_batch_size` - 对 24G 卡，开启全量 gradient checkpointing 时可设到 6。
- `validation_resolution` - OmniGen 该检查点为 1024px 模型，请设为 `1024x1024` 或 OmniGen 其他支持分辨率。
  - 其他分辨率可用逗号分隔：`1024x1024,1280x768,2048x2048`
- `validation_guidance` - 使用你在 OmniGen 推理时习惯的值；2.5-3.0 的较低值更写实
- `validation_num_inference_steps` - 约 30
- `use_ema` - 设为 `true` 可显著获得更平滑的结果（同时保留主检查点）

- `optimizer` - 可使用熟悉的优化器，本示例使用 `adamw_bf16`。
- `mixed_precision` - 建议设为 `bf16` 获得最高效训练；或设为 `no`（更慢且占用更大内存）。
- `gradient_checkpointing` - 关闭会最快但限制 batch；要最小化 VRAM 必须启用。

多 GPU 用户可参考[此文档](../OPTIONS.md#environment-configuration-variables)了解 GPU 数量配置。

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
    "output_dir": "output/models-omnigen",
    "optimizer": "adamw_bf16",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "omnigen",
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
    "data_backend_config": "config/omnigen/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "no_change",
    "aspect_bucket_rounding": 2
}
```
</details>

以及一个简单的 `config/lycoris_config.json` 文件——为提高训练稳定性可移除 `FeedForward`。

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
            "FeedForward"
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 16
            },
            "FeedForward": {
                "factor": 8
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

`config/config.json` 中包含“主验证提示词”（`--validation_prompt`），通常是你在单主体或风格训练时使用的主 instance_prompt。此外，可创建 JSON 文件包含额外验证提示词。

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

多样化提示词有助于判断模型是否崩溃。本示例中将 `<token>` 替换为你的主体名称（instance_prompt）。

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

> ℹ️ OmniGen 似乎最多理解约 122 个 token，超过此长度是否能理解尚不确定。

#### CLIP 分数跟踪

如需启用评估以评分模型性能，请参阅[此文档](../evaluation/CLIP_SCORES.md)。

</details>

# 稳定评估损失

如需使用稳定的 MSE 损失评估模型性能，请参阅[此文档](../evaluation/EVAL_LOSS.md)。

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

当前 OmniGen 硬编码使用其特殊流匹配公式，时间表偏移对其不生效。

<!--
Flow-matching models such as OmniGen, Sana, Flux, and SD3 have a property called "shift" that allows us to shift the trained portion of the timestep schedule using a simple decimal value.

##### Auto-shift
A commonly-recommended approach is to follow several recent works and enable resolution-dependent timestep shift, `--flow_schedule_auto_shift` which uses higher shift values for larger images, and lower shift values for smaller images. This results in stable but potentially mediocre training results.

##### Manual specification
_Thanks to General Awareness from Discord for the following examples_

When using a `--flow_schedule_shift` value of 0.1 (a very low value), only the finer details of the image are affected:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

When using a `--flow_schedule_shift` value of 4.0 (a very high value), the large compositional features and potentially colour space of the model becomes impacted:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)
-->

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
    "id": "pseudo-camera-10k-omnigen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/omnigen/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/omnigen/dreambooth-subject",
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
    "cache_dir_vae": "cache/vae/omnigen/dreambooth-subject-512",
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
    "cache_dir": "cache/text/omnigen",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

> ℹ️ 同时运行 512px 与 1024px 数据集是支持的，可能会改善收敛。

> ℹ️ OmniGen 不生成文本编码器嵌入，但目前仍需要定义一个（暂时如此）。

我的 OmniGen 配置非常基础，使用稳定评估损失训练集时如下：

<details>
<summary>查看示例配置</summary>

```json
[
    {
        "id": "something-special-to-remember-by",
        "type": "local",
        "instance_data_dir": "/datasets/pseudo-camera-10k/train",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": "filename",
        "cache_dir_vae": "cache/vae/omnigen",
        "vae_cache_clear_each_epoch": false,
        "crop": true,
        "crop_aspect": "square"
    },
    {
        "id": "omnigen-eval",
        "type": "local",
        "dataset_type": "eval",
        "crop": true,
        "crop_aspect": "square",
        "instance_data_dir": "/datasets/test_datasets/squares",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/omnigen-eval",
        "caption_strategy": "filename"
    },
    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/omnigen"
    }
]
```
</details>


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

在 SimpleTuner 目录中可选择以下方式启动训练：

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

## 注意事项与排错提示

### 最低 VRAM 配置

OmniGen 的最低 VRAM 配置尚未知，但预计类似如下：

- OS: Ubuntu Linux 24
- GPU: 单张 NVIDIA CUDA（10G, 12G）
- 系统内存: 约 50G
- 基础模型精度: `int8-quanto`（或 `fp8-torchao`, `int8-torchao` 具有相似内存表现）
- 优化器: Lion 8Bit Paged, `bnb-lion8bit-paged`
- 分辨率: 1024px
- 批大小: 1，零梯度累积
- DeepSpeed: 禁用 / 未配置
- PyTorch: 2.7+
- 使用 `--quantize_via=cpu` 避免 <=16G 卡启动 OOM
- 启用 `--gradient_checkpointing`
- 使用小型 LoRA/Lycoris 配置（如 LoRA rank 1 或 Lokr factor 25）

**注意**：预缓存 VAE 嵌入与文本编码器输出可能使用更多内存并导致 OOM。如发生，可启用 VAE 切片。文本编码器可通过 `offload_during_startup=true` 在 VAE 缓存期间卸载到 CPU。若数据集较大且不希望使用磁盘缓存，可使用 `--vae_cache_disable`。

在 AMD 7900XTX + Pytorch 2.7 + ROCm 6.3 上速度约 3.4 it/s。

### Masked loss

如需对主体或风格进行遮罩训练，请参阅 Dreambooth 指南中的 [masked loss training](../DREAMBOOTH.md#masked-loss)。

### 量化

尚未充分测试。

### 学习率

#### LoRA (--lora_type=standard)

*不支持。*

#### LoKr (--lora_type=lycoris)
- LoKr 更适合温和学习率（AdamW 1e-4，Lion 2e-5）
- 其他算法仍需探索
- 设置 `is_regularisation_data` 对 OmniGen 的影响未知（未测试）

### 图像伪影

OmniGen 对伪影的反应尚未知，但使用 SDXL VAE，细节限制一致。

若出现画质问题，请在 GitHub 提交 issue。

### 宽高比分桶

该模型对宽高比数据的反应未知，建议进行实验。

### 高损失值

OmniGen 的损失值非常高，原因未知。建议忽略损失值，关注生成图像的视觉质量。
