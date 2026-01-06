## Wan 2.2 S2V 快速入门

在本示例中，我们将训练一个 Wan 2.2 S2V (Speech-to-Video) LoRA。S2V 模型根据音频输入生成视频，实现音频驱动的视频生成。

### 硬件要求

Wan 2.2 S2V **14B** 是一个对硬件要求较高的模型，需要大量 GPU 显存。

#### Speech to Video

14B - https://huggingface.co/tolgacangoz/Wan2.2-S2V-14B-Diffusers
- 分辨率：832x480
- 可以在 24G 显存下运行，但需要调整一些设置。

你需要：
- **实际最低要求** 是 24GB，即单张 4090 或 A6000 GPU
- **理想情况** 多张 4090、A6000、L40S 或更好的 GPU

Apple 芯片系统目前与 Wan 2.2 的兼容性不太好，单个训练步骤可能需要约 10 分钟。

### 前提条件

确保你已安装 Python；SimpleTuner 适用于 3.10 到 3.12 版本。

你可以通过运行以下命令检查：

```bash
python --version
```

如果你的 Ubuntu 系统没有安装 Python 3.12，可以尝试以下命令：

```bash
apt -y install python3.12 python3.12-venv
```

#### 容器镜像依赖

对于 Vast、RunPod 和 TensorDock（以及其他平台），在 CUDA 12.2-12.8 镜像上使用以下命令来启用 CUDA 扩展编译：

```bash
apt -y install nvidia-cuda-toolkit
```

### 安装

通过 pip 安装 SimpleTuner：

```bash
pip install simpletuner[cuda]
```

如需手动安装或开发设置，请参阅[安装文档](/documentation/INSTALL.md)。
#### SageAttention 2

如果你想使用 SageAttention 2，需要按照以下步骤操作。

> 注意：SageAttention 提供的加速效果有限，效果不是特别明显；原因不明。在 4090 上测试过。

在 Python venv 环境中运行以下命令：
```bash
git clone https://github.com/thu-ml/SageAttention
pushd SageAttention
  pip install . --no-build-isolation
popd
```

#### AMD ROCm 后续步骤

要使用 AMD MI300X，必须执行以下命令：

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

### 环境设置

要运行 SimpleTuner，你需要设置配置文件、数据集和模型目录，以及数据加载器配置文件。

#### 配置文件

一个实验性脚本 `configure.py` 可能允许你通过交互式分步配置完全跳过本节。它包含一些安全功能，有助于避免常见问题。

**注意：** 这不会配置你的数据加载器。你仍然需要稍后手动完成。

运行方式：

```bash
simpletuner configure
```

> 对于位于无法方便访问 Hugging Face Hub 的国家的用户，你应该根据系统使用的 `$SHELL` 在 `~/.bashrc` 或 `~/.zshrc` 中添加 `HF_ENDPOINT=https://hf-mirror.com`。

### 内存卸载（可选）

Wan 是 SimpleTuner 支持的最重的模型之一。如果你接近显存上限，请启用分组卸载：

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# 可选：将卸载的权重溢出到磁盘而不是内存
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- 只有 CUDA 设备支持 `--group_offload_use_stream`；ROCm/MPS 会自动回退。
- 除非 CPU 内存是瓶颈，否则保持磁盘暂存选项注释掉。
- `--enable_model_cpu_offload` 与分组卸载互斥。

### 前馈分块（可选）

如果 14B 检查点在梯度检查点期间仍然 OOM，请对 Wan 前馈层进行分块：

```bash
--enable_chunked_feed_forward \
--feed_forward_chunk_size 2 \
```

这与配置向导中的新开关匹配（`Training -> Memory Optimisation`）。较小的分块大小可以节省更多内存，但会减慢每一步的速度。你也可以在环境中设置 `WAN_FEED_FORWARD_CHUNK_SIZE=2` 进行快速实验。


如果你更喜欢手动配置：

将 `config/config.json.example` 复制到 `config/config.json`：

```bash
cp config/config.json.example config/config.json
```

多 GPU 用户可以参考[此文档](/documentation/OPTIONS.md#environment-configuration-variables)了解如何配置使用的 GPU 数量。

你最终的配置将类似于我的：

<details>
<summary>查看示例配置</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "attention_mechanism": "sageattention",
  "data_backend_config": "config/wan_s2v/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "offload_during_startup": true,
  "disable_benchmark": false,
  "output_dir": "output/wan_s2v",
  "lora_type": "standard",
  "lycoris_config": "config/wan_s2v/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "wan-s2v-lora",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "wan-s2v-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
  "pretrained_t5_model_name_or_path": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
  "model_family": "wan_s2v",
  "model_flavour": "s2v-14b-2.2",
  "train_batch_size": 2,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "832x480",
  "validation_prompt": "A person speaking with natural gestures",
  "validation_negative_prompt": "blurry, low quality, distorted",
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 40,
  "validation_num_video_frames": 81,
  "mixed_precision": "bf16",
  "optimizer": "optimi-lion",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.01,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "no_change",
  "vae_batch_size": 1,
  "webhook_config": "config/wan_s2v/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "lora_rank": 16,
  "flow_schedule_shift": 3,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

此配置中特别重要的是验证设置。没有这些设置，输出效果可能不太理想。

### 可选：CREPA 时序正则化器

为了在 Wan S2V 上获得更平滑的运动和更少的身份漂移：
- 在 **Training -> Loss functions** 中，启用 **CREPA**。
- 初始设置为 **Block Index = 8**、**Weight = 0.5**、**Adjacent Distance = 1**、**Temporal Decay = 1.0**。
- 默认编码器（`dinov2_vitg14`，大小 `518`）效果良好；仅在需要减少显存时切换到 `dinov2_vits14` + `224`。
- 首次运行会通过 torch hub 下载 DINOv2；如果离线训练，请预先缓存或获取。
- 只有在完全从缓存的 latents 进行训练时才启用 **Drop VAE Encoder**；否则保持关闭以便像素编码仍然有效。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可以显著提高训练稳定性和性能的实验功能。

*   **[计划采样 (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md)：** 通过让模型在训练期间生成自己的输入来减少暴露偏差并提高输出质量。

> 这些功能会增加训练的计算开销。

</details>

### TREAD 训练

> **实验性功能**：TREAD 是一个新实现的功能。虽然可以正常工作，但最佳配置仍在探索中。

[TREAD](/documentation/TREAD.md)（论文）代表 **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion。这是一种通过智能路由 token 通过 transformer 层来加速 Wan S2V 训练的方法。加速效果与丢弃的 token 数量成正比。

#### 快速设置

将以下内容添加到你的 `config.json` 以获得简单且保守的方法：

<details>
<summary>查看示例配置</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.1,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

此配置将：
- 在第 2 层到倒数第二层之间只保留 50% 的图像 token
- 文本 token 永远不会被丢弃
- 训练加速约 25%，对质量影响最小
- 可能改善训练质量和收敛性

#### 关键点

- **有限的架构支持** - TREAD 仅为 Flux 和 Wan 模型（包括 S2V）实现
- **高分辨率效果最佳** - 由于注意力机制的 O(n^2) 复杂度，在 1024x1024+ 分辨率下加速最大
- **与 masked loss 兼容** - 遮罩区域会自动保留（但这会降低加速效果）
- **与量化配合使用** - 可以与 int8/int4/NF4 训练结合使用
- **预期初始损失峰值** - 开始 LoRA/LoKr 训练时，损失会更高，但会很快纠正

#### 调优技巧

- **保守（质量优先）**：使用 0.1-0.3 的 `selection_ratio`
- **激进（速度优先）**：使用 0.3-0.5 的 `selection_ratio` 并接受质量影响
- **避免早期/晚期层**：不要在 0-1 层或最后一层进行路由
- **LoRA 训练**：可能会有轻微的减速 - 尝试不同的配置
- **分辨率越高 = 加速越好**：在 1024px 及以上效果最佳

有关详细配置选项和故障排除，请参阅[完整 TREAD 文档](/documentation/TREAD.md)。


#### 验证提示词

在 `config/config.json` 中有"主验证提示词"，通常是你为单个主题或风格训练的主 instance_prompt。此外，可以创建一个 JSON 文件，其中包含在验证期间运行的额外提示词。

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

昵称是验证的文件名，因此请保持简短并与你的文件系统兼容。

要将训练器指向此提示词库，请通过在 `config.json` 末尾添加新行将其添加到 TRAINER_EXTRA_ARGS：
<details>
<summary>查看示例配置</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

> S2V 使用 UMT5 文本编码器，其嵌入中有大量局部信息，这意味着较短的提示词可能没有足够的信息让模型做好工作。请务必使用更长、更具描述性的提示词。

#### CLIP 分数跟踪

目前不应该为视频模型训练启用此功能。

# 稳定评估损失

如果你希望使用稳定的 MSE 损失来评分模型性能，请参阅[此文档](/documentation/evaluation/EVAL_LOSS.md)了解配置和解释评估损失的信息。

#### 验证预览

SimpleTuner 支持在生成过程中使用 Tiny AutoEncoder 模型流式传输中间验证预览。这允许你通过 webhook 回调实时查看正在生成的验证图像。

要启用：
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
- 已启用验证

将 `validation_preview_steps` 设置为较高的值（例如 3 或 5）可以减少 Tiny AutoEncoder 的开销。使用 `validation_num_inference_steps=20` 和 `validation_preview_steps=5`，你将在第 5、10、15 和 20 步收到预览图像。

#### 流匹配调度偏移

Flux、Sana、SD3、LTX Video 和 Wan S2V 等流匹配模型有一个名为 `shift` 的属性，允许我们使用简单的十进制值来偏移时间步调度的训练部分。

##### 默认值
默认情况下，不应用调度偏移，这会导致时间步采样分布呈 sigmoid 钟形曲线，也称为 `logit_norm`。

##### 自动偏移
一种常推荐的方法是遵循最近的几项工作，启用分辨率相关的时间步偏移，`--flow_schedule_auto_shift` 对较大的图像使用较高的偏移值，对较小的图像使用较低的偏移值。这会产生稳定但可能平庸的训练结果。

##### 手动指定
_感谢 Discord 上的 General Awareness 提供以下示例_

> 这些示例展示了该值如何使用 Flux Dev 工作，尽管 Wan S2V 应该非常相似。

当使用 0.1（非常低的值）的 `--flow_schedule_shift` 值时，只有图像的细节受到影响：
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

当使用 4.0（非常高的值）的 `--flow_schedule_shift` 值时，模型的大型构图特征和潜在的色彩空间会受到影响：
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### 量化模型训练

在 Apple 和 NVIDIA 系统上测试过，可以使用 Hugging Face Optimum-Quanto 来降低精度和显存要求，仅需 16GB 即可训练。



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

#### 验证设置

在初期探索中，Wan S2V 可能会产生较差的输出质量，原因可能有以下几点：

- 推理步数不够
  - 除非你使用 UniPC，否则可能需要至少 40 步。UniPC 可以稍微减少步数，但你需要实验。
- 调度器配置不正确
  - 它使用的是普通的 Euler 流匹配调度，但 Betas 分布似乎效果最好
  - 如果你没有改动过这个设置，现在应该没问题了
- 分辨率不正确
  - Wan S2V 只有在其训练的分辨率上才能正常工作，偶尔能成功是运气，但结果不好是很常见的
- 错误的 CFG 值
  - 4.0-5.0 左右的值似乎比较安全
- 不当的提示词
  - 当然，视频模型似乎需要一队神秘主义者花几个月时间在山上进行禅修，才能学会提示词的神圣艺术，因为他们的数据集和标注风格像圣杯一样被严密守护。
  - 简而言之：尝试不同的提示词。
- 缺少或不匹配的音频
  - S2V 验证需要音频输入 - 确保你的验证样本有对应的音频文件

尽管如此，除非你的批次大小太小和/或学习率太高，否则模型将在你喜欢的推理工具中正确运行（假设你已经有一个能获得良好结果的工具）。

#### 数据集注意事项

S2V 训练需要配对的视频和音频数据。你需要一个视频数据集和一个通过 `s2v_datasets` 链接的相应音频数据集。

除了需要多少计算资源和时间来处理和训练之外，对数据集大小几乎没有限制。

你必须确保数据集足够大以有效训练你的模型，但又不能大到超出你可用的计算资源。

请注意，数据集的最小大小是 `train_batch_size * gradient_accumulation_steps`，并且要大于 `vae_batch_size`。如果数据集太小将无法使用。

> 如果样本太少，你可能会看到 **no samples detected in dataset** 的消息 - 增加 `repeats` 值可以克服这个限制。

#### 音频数据集设置

##### 从视频自动提取音频（推荐）

如果你的视频已经包含音轨，SimpleTuner 可以自动提取和处理音频，无需单独的音频数据集。这是最简单的方法：

```json
[
  {
    "id": "s2v-videos",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan_s2v/videos",
    "instance_data_dir": "datasets/s2v-videos",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    },
    "audio": {
        "auto_split": true,
        "sample_rate": 16000,
        "channels": 1
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan_s2v",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

使用 `audio.auto_split: true`，SimpleTuner 将：
1. 自动生成音频数据集配置（`s2v-videos_audio`）
2. 在元数据发现期间从每个视频中提取音频
3. 在专用目录中缓存音频 VAE latents
4. 通过 `s2v_datasets` 自动链接音频数据集

**音频配置选项：**
- `audio.auto_split`（布尔值）：启用从视频自动提取音频
- `audio.sample_rate`（整数）：目标采样率，单位 Hz（默认：16000，适用于 Wav2Vec2）
- `audio.channels`（整数）：音频通道数（默认：1，单声道）
- `audio.allow_zero_audio`（布尔值）：为没有音轨的视频生成零填充音频（默认：false）
- `audio.max_duration_seconds`（浮点数）：最大音频时长；超过的文件将被跳过
- `audio.duration_interval`（浮点数）：用于分桶的持续时间间隔，单位秒（默认：3.0）
- `audio.truncation_mode`（字符串）：如何截断过长的音频："beginning"、"end"、"random"（默认："beginning"）

**注意**：没有音轨的视频会自动跳过 S2V 训练，除非设置了 `audio.allow_zero_audio: true`。

##### 手动音频数据集（替代方案）

如果你更喜欢使用单独的音频文件或需要自定义音频处理，S2V 模型也可以使用与视频文件按文件名匹配的预提取音频文件。例如：
- `video_001.mp4` 应该有对应的 `video_001.wav`（或 `.mp3`、`.flac`、`.ogg`、`.m4a`）

音频文件应该在一个单独的目录中，你将把它配置为 `s2v_datasets` 后端。

##### 从视频提取音频（手动）

如果你的视频已经包含音频，使用提供的脚本来提取：

```bash
# 仅提取音频（保持原始视频不变）
python scripts/generate_s2v_audio.py \
    --input-dir datasets/s2v-videos \
    --output-dir datasets/s2v-audio

# 提取音频并从源视频中移除（推荐以避免冗余数据）
python scripts/generate_s2v_audio.py \
    --input-dir datasets/s2v-videos \
    --output-dir datasets/s2v-audio \
    --strip-audio
```

该脚本：
- 以 16kHz 单声道 WAV 提取音频（Wav2Vec2 的原生采样率）
- 自动匹配文件名（例如，`video.mp4` -> `video.wav`）
- 跳过没有音轨的视频
- 需要安装 `ffmpeg`

##### 数据集配置（手动）

创建一个 `--data_backend_config`（`config/multidatabackend.json`）文档，包含以下内容：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "s2v-videos",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan_s2v/videos",
    "instance_data_dir": "datasets/s2v-videos",
    "s2v_datasets": ["s2v-audio"],
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "s2v-audio",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/s2v-audio",
    "disabled": false
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan_s2v",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> 请参阅 [DATALOADER.md](../DATALOADER.md#caption_strategy) 中的 caption_strategy 选项和要求。

S2V 数据集配置的关键点：
- 视频数据集上的 `s2v_datasets` 字段指向音频后端
- 音频文件按文件名主干匹配（例如，`video_001.mp4` 匹配 `video_001.wav`）
- 音频使用 Wav2Vec2 即时编码（约 600MB 显存），无需缓存
- 音频数据集类型是 `audio`

- 在 `video` 子部分中，我们可以设置以下键：
  - `num_frames`（可选，整数）是我们将训练的帧数。
    - 在 15 fps 下，75 帧是 5 秒的视频，即标准输出。这应该是你的目标。
  - `min_frames`（可选，整数）确定将被考虑用于训练的视频的最小长度。
    - 这应该至少等于 `num_frames`。不设置则确保它们相等。
  - `max_frames`（可选，整数）确定将被考虑用于训练的视频的最大长度。
  - `bucket_strategy`（可选，字符串）确定视频如何分组到桶中：
    - `aspect_ratio`（默认）：仅按空间宽高比分组（例如，`1.78`、`0.75`）。
    - `resolution_frames`：按分辨率和帧数以 `WxH@F` 格式分组（例如，`832x480@75`）。适用于混合分辨率/时长的数据集。
  - `frame_interval`（可选，整数）使用 `resolution_frames` 时，将帧数舍入到此间隔。

然后，创建一个包含视频和音频文件的 `datasets` 目录：

```bash
mkdir -p datasets/s2v-videos datasets/s2v-audio
# 将你的视频文件放在 datasets/s2v-videos/
# 将你的音频文件放在 datasets/s2v-audio/
```

确保每个视频都有按文件名主干匹配的音频文件。

#### 登录 WandB 和 Huggingface Hub

在开始训练之前，你需要登录 WandB 和 HF Hub，特别是如果你使用 `--push_to_hub` 和 `--report_to=wandb`。

如果你要手动将项目推送到 Git LFS 仓库，你还应该运行 `git config --global credential.helper store`

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

在 SimpleTuner 目录中，你有几个选项来启动训练：

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

有关更多信息，请参阅[数据加载器](/documentation/DATALOADER.md)和[教程](/documentation/TUTORIAL.md)文档。

## 注意事项和故障排除技巧

### 最低显存配置

Wan S2V 对量化敏感，目前不能使用 NF4 或 INT4。

- 操作系统：Ubuntu Linux 24
- GPU：单个 NVIDIA CUDA 设备（推荐 24G）
- 系统内存：大约 16G 系统内存
- 基础模型精度：`int8-quanto`
- 优化器：Lion 8Bit Paged，`bnb-lion8bit-paged`
- 分辨率：480px
- 批次大小：1，零梯度累积步骤
- DeepSpeed：禁用/未配置
- PyTorch：2.6
- 务必启用 `--gradient_checkpointing`，否则无论你做什么都会 OOM
- 只在图像上训练，或将视频数据集的 `num_frames` 设置为 1

**注意**：VAE 嵌入和文本编码器输出的预缓存可能会使用更多内存并仍然 OOM。因此，`--offload_during_startup=true` 基本上是必需的。如果是这样，可以启用文本编码器量化和 VAE 分块。（Wan 目前不支持 VAE 分块/切片）

### SageAttention

使用 `--attention_mechanism=sageattention` 时，可以在验证时加速推理。

**注意**：这与最终的 VAE 解码步骤不兼容，不会加速该部分。

### Masked loss

不要在 Wan S2V 中使用此功能。

### 量化
- 根据批次大小，可能需要量化才能在 24G 中训练此模型

### 图像伪影
Wan 需要使用 Euler Betas 流匹配调度或（默认情况下）UniPC 多步求解器，这是一个会做出更强预测的高阶调度器。

与其他 DiT 模型一样，如果你做了这些事情（以及其他），样本中**可能**会开始出现一些方格伪影：
- 使用低质量数据过度训练
- 使用过高的学习率
- 过度训练（一般而言），低容量网络配合过多图像
- 训练不足（同样），高容量网络配合过少图像
- 使用奇怪的宽高比或训练数据大小

### 宽高比分桶
- 视频像图像一样分桶。
- 在方形裁剪上训练太长时间可能不会对这个模型造成太大损害。放心使用，它很棒也很可靠。
- 另一方面，使用数据集的自然宽高比桶可能会在推理时过度偏向这些形状。
  - 这可能是一个理想的特性，因为它可以防止像电影风格这样的宽高比相关的风格过多地渗透到其他分辨率中。
  - 然而，如果你想在多个宽高比桶中同样改善结果，你可能需要尝试 `crop_aspect=random`，这有其自身的缺点。
- 通过多次定义图像目录数据集来混合数据集配置已经产生了非常好的结果和一个很好的泛化模型。

### 音频同步

为了在 S2V 中获得最佳效果：
- 确保音频时长与视频时长匹配
- 音频在内部重采样到 16kHz
- Wav2Vec2 编码器即时处理音频（约 600MB 显存开销）
- 音频特征被插值以匹配视频帧数
