## Sana Video 快速入门

在本示例中，我们将训练 Sana Video 2B 480p 模型。

### 硬件要求

Sana Video 使用 Wan 自动编码器，默认处理 480p 的 81 帧序列。内存开销与其他视频模型相当；建议尽早启用梯度检查点，并仅在确认 VRAM 余量后再提升 `train_batch_size`。

### 内存卸载（可选）

如果接近 VRAM 上限，请在配置中启用分组卸载：

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- CUDA 用户可从 `--group_offload_use_stream` 中获益；其他后端会自动忽略。
- 除非系统内存受限，否则不要使用 `--group_offload_to_disk_path` — 磁盘分级更慢但更稳定。
- 使用分组卸载时禁用 `--enable_model_cpu_offload`。

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

接下来可能需要修改以下变量：

- `model_type` - 设置为 `full`。
- `model_family` - 设置为 `sanavideo`。
- `pretrained_model_name_or_path` - 设置为 `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`。
- `pretrained_vae_model_name_or_path` - 设置为 `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`。
- `output_dir` - 设置为您想要存储检查点和验证视频的目录。建议使用完整路径。
- `train_batch_size` - 先从较小值开始，仅在确认 VRAM 使用情况后再提升。
- `validation_resolution` - Sana Video 是 480p 模型；使用 `832x480` 或您要验证的宽高比桶。
- `validation_num_video_frames` - 设置为 `81` 以匹配默认采样长度。
- `validation_guidance` - 使用您在推理时习惯的值。
- `validation_num_inference_steps` - 使用 50 左右以获得稳定质量。
- `framerate` - 如未设置，Sana Video 默认 16 fps；请设置为与数据集一致的值。

- `optimizer` - 可以使用您熟悉的优化器，但本示例使用 `optimi-adamw`。
- `mixed_precision` - 建议设置为 `bf16` 以获得最有效的训练；也可设为 `no`（更耗内存、更慢）。
- `gradient_checkpointing` - 启用以控制 VRAM 使用。
- `use_ema` - 设为 `true` 有助于在主检查点之外获得更平滑的结果。

多 GPU 用户可参考[此文档](../OPTIONS.md#environment-configuration-variables)了解 GPU 数量的配置方式。

最终配置应类似于：

<details>
<summary>查看示例配置</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "data_backend_config": "config/sanavideo/multidatabackend.json",
  "seed": 42,
  "output_dir": "output/sanavideo",
  "max_train_steps": 400000,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "tracker_project_name": "video-training",
  "tracker_run_name": "sanavideo-2b-480p",
  "report_to": "wandb",
  "model_type": "full",
  "pretrained_model_name_or_path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
  "pretrained_vae_model_name_or_path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
  "model_family": "sanavideo",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 200,
  "validation_resolution": "832x480",
  "validation_negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "validation_guidance": 6.0,
  "validation_num_inference_steps": 50,
  "validation_num_video_frames": 81,
  "validation_prompt": "A short video of a small, fluffy animal exploring a sunny room with soft window light and gentle camera motion.",
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.0,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "bf16",
  "vae_batch_size": 1,
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "framerate": 16,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

### 可选：CREPA 时间正则化

如果视频出现闪烁或主体漂移，可启用 CREPA：
- 在 **Training → Loss functions** 中开启 **CREPA**。
- 建议默认值：**Block Index = 10**、**Weight = 0.5**、**Adjacent Distance = 1**、**Temporal Decay = 1.0**。
- 除非需要更小选项以节省 VRAM，否则保持默认编码器（`dinov2_vitg14`，尺寸 `518`）。
- 首次运行会通过 torch hub 下载 DINOv2；离线训练请提前缓存/预取。
- 仅在完全基于缓存 latent 训练时才开启 **Drop VAE Encoder**；否则保持关闭以继续编码像素。

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
    "anime_<token>": "a breathtaking anime-style video featuring <token>, capturing her essence with vibrant colors, dynamic motion, and expressive storytelling",
    "chef_<token>": "a high-quality, detailed video of <token> as a sous-chef, immersed in the art of culinary creation with captivating close-ups and engaging sequences",
    "just_<token>": "a lifelike and intimate video portrait of <token>, showcasing her unique personality and charm through nuanced movement and expression",
    "cinematic_<token>": "a cinematic, visually stunning video of <token>, emphasizing her dramatic and captivating presence through fluid camera movements and atmospheric effects",
    "elegant_<token>": "an elegant and timeless video portrait of <token>, exuding grace and sophistication with smooth transitions and refined visuals",
    "adventurous_<token>": "a dynamic and adventurous video featuring <token>, captured in an exciting, action-filled sequence that highlights her energy and spirit",
    "mysterious_<token>": "a mysterious and enigmatic video portrait of <token>, shrouded in shadows and intrigue with a narrative that unfolds in subtle, cinematic layers",
    "vintage_<token>": "a vintage-style video of <token>, evoking the charm and nostalgia of a bygone era through sepia tones and period-inspired visual storytelling",
    "artistic_<token>": "an artistic and abstract video representation of <token>, blending creativity with visual storytelling through experimental techniques and fluid visuals",
    "futuristic_<token>": "a futuristic and cutting-edge video portrayal of <token>, set against a backdrop of advanced technology with sleek, high-tech visuals",
    "woman": "a beautifully crafted video portrait of a woman, highlighting her natural beauty and unique features through elegant motion and storytelling",
    "man": "a powerful and striking video portrait of a man, capturing his strength and character with dynamic sequences and compelling visuals",
    "boy": "a playful and spirited video portrait of a boy, capturing youthful energy and innocence through lively scenes and engaging motion",
    "girl": "a charming and vibrant video portrait of a girl, emphasizing her bright personality and joy with colorful visuals and fluid movement",
    "family": "a heartwarming and cohesive family video, showcasing the bonds and connections between loved ones through intimate moments and shared experiences"
}
```

> ℹ️ Sana Video 是流匹配模型；较短的提示词可能信息不足，尽量使用更具描述性的提示词。

#### CLIP 分数跟踪

目前不应在视频模型训练中启用此功能。

</details>

# 稳定评估损失

如需使用稳定的 MSE 损失来评分模型性能，请参阅[此文档](../evaluation/EVAL_LOSS.md)了解评估损失的配置与解读。

#### 验证预览

SimpleTuner 支持使用 Tiny AutoEncoder 模型在生成过程中流式传输中间验证预览。这允许您通过 webhook 回调实时逐步查看正在生成的验证视频。

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

将 `validation_preview_steps` 设置为更高的值（例如 3 或 5）以减少 Tiny AutoEncoder 开销。使用 `validation_num_inference_steps=20` 和 `validation_preview_steps=5`，您将在步骤 5、10、15 和 20 收到预览帧。

#### 流匹配调度

Sana Video 使用检查点中提供的标准流匹配调度。用户提供的 shift 覆盖会被忽略；请保持 `flow_schedule_shift` 和 `flow_schedule_auto_shift` 未设置。

#### 量化模型训练

配置中支持 bf16、int8、fp8 等精度选项；根据硬件选择，若出现不稳定请回退到更高精度。

#### 数据集注意事项

除了处理和训练所需的计算与时间外，数据集大小限制较少。

您必须确保数据集足够大以有效训练模型，但又不能大到超出可用算力。

最小数据集大小为 `train_batch_size * gradient_accumulation_steps`，并且要大于 `vae_batch_size`。如果数据集太小将无法使用。

> ℹ️ 当样本数量很少时，可能会看到 **no samples detected in dataset** 的消息 — 增加 `repeats` 值可以克服这个限制。

根据您拥有的数据集，您需要以不同方式设置数据集目录和数据加载器配置文件。

在此示例中，我们将使用 [video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized) 作为数据集。

创建一个 `--data_backend_config`（`config/multidatabackend.json`）文档，内容如下：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/sanavideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 81,
        "min_frames": 81,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sanavideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> 请参阅 [DATALOADER.md](../DATALOADER.md#caption_strategy) 中的 caption_strategy 选项和要求。

- 在 `video` 子部分中，我们可以设置以下键：
  - `num_frames`（可选，int）是我们将训练的帧数。
  - `min_frames`（可选，int）确定将被考虑用于训练的视频的最小长度。
  - `max_frames`（可选，int）确定将被考虑用于训练的视频的最大长度。
  - `is_i2v`（可选，bool）确定是否在该数据集上进行 i2v 训练。
  - `bucket_strategy`（可选，string）确定视频如何分组到桶中：
    - `aspect_ratio`（默认）：仅按空间宽高比分组（例如 `1.78`、`0.75`）。
    - `resolution_frames`：按分辨率和帧数以 `WxH@F` 格式分组（例如 `832x480@81`）。适用于混合分辨率/时长的数据集。
  - `frame_interval`（可选，int）在使用 `resolution_frames` 时，将帧数舍入到该间隔。

然后创建一个 `datasets` 目录：

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset sayakpaul/video-dataset-disney-organized --local-dir=disney-black-and-white
popd
```

这将下载所有 Disney 视频样本到 `datasets/disney-black-and-white` 目录，该目录将自动为您创建。

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
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
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

### 验证默认值

- 未提供验证设置时，Sana Video 默认 81 帧、16 fps。
- Wan 自动编码器路径应与基础模型路径一致，保持一致以避免加载错误。

### Masked loss

如果您要训练主体或风格并希望遮罩其一，请参阅 Dreambooth 指南的[遮罩损失训练](../DREAMBOOTH.md#masked-loss)部分。
