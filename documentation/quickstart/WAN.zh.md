## Wan 2.1 快速入门

在本示例中，我们将使用 Sayak Paul 的[公共领域迪士尼数据集](https://hf.co/datasets/sayakpaul/video-dataset-disney-organized)来训练一个 Wan 2.1 LoRA。



https://github.com/user-attachments/assets/51e6cbfd-5c46-407c-9398-5932fa5fa561


### 硬件要求

Wan 2.1 **1.3B** 对系统内存**和** GPU 内存的要求都不高。同样支持的 **14B** 模型则有更高的要求。

目前，Wan 不支持图生视频训练，但文生视频的 LoRA 和 Lycoris 可以在图生视频模型上运行。

#### 文生视频

1.3B - https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- 分辨率：832x480
- Rank-16 LoRA 使用略超过 12G 显存（批次大小为 4）

14B - https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers
- 分辨率：832x480
- 可以在 24G 显存内运行，但需要调整一些设置。

<!--
#### 图生视频
14B (720p) - https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
- 分辨率：1280x720
-->

#### 图生视频 (Wan 2.2)

最新的 Wan 2.2 图生视频检查点使用相同的训练流程：

- High stage: https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-Diffusers/tree/main/high_noise_model
- Low stage: https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-Diffusers/tree/main/low_noise_model

您可以使用本指南后续介绍的 `model_flavour` 和 `wan_validation_load_other_stage` 设置来指定目标阶段。

您需要：
- **实际最低配置**是 16GB，或单张 3090 或 V100 GPU
- **理想配置**是多张 4090、A6000、L40S 或更高端显卡

如果在运行 Wan 2.2 检查点时遇到时间嵌入层的形状不匹配问题，请启用新的 `wan_force_2_1_time_embedding` 标志。这会强制 transformer 回退到 Wan 2.1 风格的时间嵌入，从而解决兼容性问题。

#### 阶段预设与验证

- `model_flavour=i2v-14b-2.2-high` 目标是 Wan 2.2 高噪声阶段。
- `model_flavour=i2v-14b-2.2-low` 目标是低噪声阶段（相同的检查点，不同的子文件夹）。
- 切换 `wan_validation_load_other_stage=true` 可在验证渲染时加载与训练阶段相反的阶段。
- 保持 flavour 未设置（或使用 `t2v-480p-1.3b-2.1`）以进行标准的 Wan 2.1 文生视频运行。

Apple 芯片系统目前对 Wan 2.1 的支持不太理想，单个训练步骤可能需要约 10 分钟。

### 前置条件

请确保已安装 Python；SimpleTuner 在 3.10 到 3.12 版本上运行良好。

您可以运行以下命令检查：

```bash
python --version
```

如果您在 Ubuntu 上没有安装 Python 3.12，可以尝试以下命令：

```bash
apt -y install python3.12 python3.12-venv
```

#### 容器镜像依赖

对于 Vast、RunPod 和 TensorDock（以及其他平台），以下命令可在 CUDA 12.2-12.8 镜像上启用 CUDA 扩展编译：

```bash
apt -y install nvidia-cuda-toolkit
```

### 安装

通过 pip 安装 SimpleTuner：

```bash
pip install simpletuner[cuda]
```

有关手动安装或开发环境设置，请参阅[安装文档](../INSTALL.md)。

#### SageAttention 2

如果您想使用 SageAttention 2，需要执行以下步骤。

> 注意：SageAttention 提供的加速效果有限，不是特别有效；原因不明。已在 4090 上测试。

在 Python 虚拟环境中运行以下命令：
```bash
git clone https://github.com/thu-ml/SageAttention
pushd SageAttention
  pip install . --no-build-isolation
popd
```

#### AMD ROCm 后续步骤

以下命令必须执行才能使用 AMD MI300X：

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

### 设置环境

要运行 SimpleTuner，您需要设置配置文件、数据集和模型目录，以及数据加载器配置文件。

#### 配置文件

一个实验性脚本 `configure.py` 可能允许您通过交互式的逐步配置完全跳过本节。它包含一些安全功能，有助于避免常见陷阱。

**注意：**这不会配置您的数据加载器。您稍后仍需要手动完成该操作。

运行方式：

```bash
simpletuner configure
```

> ⚠️ 对于位于 Hugging Face Hub 访问受限国家的用户，您应该将 `HF_ENDPOINT=https://hf-mirror.com` 添加到 `~/.bashrc` 或 `~/.zshrc`，具体取决于您系统使用的 `$SHELL`。

### 内存卸载（可选）

Wan 是 SimpleTuner 支持的最重型模型之一。如果您接近 VRAM 上限，请启用分组卸载：

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# 可选：将卸载的权重溢出到磁盘而不是 RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- 只有 CUDA 设备支持 `--group_offload_use_stream`；ROCm/MPS 会自动回退。
- 除非 CPU 内存是瓶颈，否则请保持磁盘暂存为注释状态。
- `--enable_model_cpu_offload` 与分组卸载互斥。

### 前馈层分块（可选）

如果 14B 检查点在梯度检查点期间仍然 OOM，请对 Wan 前馈层进行分块：

```bash
--enable_chunked_feed_forward \
--feed_forward_chunk_size 2 \
```

这与配置向导中的新开关（`Training → Memory Optimisation`）匹配。较小的分块大小可以节省更多内存，但会减慢每一步的速度。您也可以在环境中设置 `WAN_FEED_FORWARD_CHUNK_SIZE=2` 进行快速实验。


如果您更喜欢手动配置：

将 `config/config.json.example` 复制为 `config/config.json`：

```bash
cp config/config.json.example config/config.json
```

多 GPU 用户可以参考[此文档](../OPTIONS.md#environment-configuration-variables)了解如何配置使用的 GPU 数量。

您的最终配置将类似于以下示例：

<details>
<summary>查看示例配置</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "attention_mechanism": "sageattention",
  "data_backend_config": "config/wan/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "offload_during_startup": true,
  "disable_benchmark": false,
  "output_dir": "output/wan",
  "lora_type": "standard",
  "lycoris_config": "config/wan/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "wan-disney",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "wan-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "model_family": "wan",
  "train_batch_size": 2,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "832x480",
  "validation_prompt": "两只拟人化的猫咪身穿舒适的拳击装备，戴着鲜艳的手套，在聚光灯照射的舞台上激烈对战",
  "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
  "validation_guidance": 5.2,
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
  "webhook_config": "config/wan/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "validation_guidance_skip_layers": [9],
  "validation_guidance_skip_layers_start": 0.0,
  "validation_guidance_skip_layers_stop": 1.0,
  "lora_rank": 16,
  "flow_schedule_shift": 3,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

此配置中特别重要的是验证设置。没有这些设置，输出效果可能不太理想。

### 可选：CREPA 时间正则化器

为了在 Wan 上获得更平滑的运动和更少的身份漂移：
- 在 **Training → Loss functions** 中，启用 **CREPA**。
- 从 **Block Index = 8**、**Weight = 0.5**、**Adjacent Distance = 1**、**Temporal Decay = 1.0** 开始。
- 默认编码器（`dinov2_vitg14`，大小 `518`）效果良好；仅在需要减少 VRAM 时切换到 `dinov2_vits14` + `224`。
- 首次运行会通过 torch hub 下载 DINOv2；如果离线训练，请提前缓存或预取。
- 仅在完全从缓存的潜在表示进行训练时启用 **Drop VAE Encoder**；否则保持关闭以便像素编码仍能正常工作。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样 (Rollout)](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

</details>

### TREAD 训练

> ⚠️ **实验性**：TREAD 是一个新实现的功能。虽然可用，但最佳配置仍在探索中。

[TREAD](../TREAD.md)（论文）代表 **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion。这是一种通过智能路由 token 通过 transformer 层来加速 Flux 训练的方法。加速效果与丢弃的 token 数量成正比。

#### 快速设置

将以下内容添加到您的 `config.json` 以获得简单保守的方法，在 bs=2 和 480p 下达到约 5 秒/步（原始速度为 10 秒/步）：

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
- 在第 2 层到倒数第二层期间仅保留 50% 的图像 token
- 文本 token 永远不会被丢弃
- 训练加速约 25%，对质量影响最小
- 可能提高训练质量和收敛性

对于 Wan 1.3B，我们可以使用覆盖所有 29 层的渐进路由设置来增强此方法，在 bs=2 和 480p 下达到约 7.7 秒/步：

<details>
<summary>查看示例配置</summary>

```json
{
  "tread_config": {
      "routes": [
          { "selection_ratio": 0.1, "start_layer_idx": 2, "end_layer_idx": 8 },
          { "selection_ratio": 0.25, "start_layer_idx": 9, "end_layer_idx": 11 },
          { "selection_ratio": 0.35, "start_layer_idx": 12, "end_layer_idx": 15 },
          { "selection_ratio": 0.25, "start_layer_idx": 16, "end_layer_idx": 23 },
          { "selection_ratio": 0.1, "start_layer_idx": 24, "end_layer_idx": -2 }
      ]
  }
}
```
</details>

此配置将尝试在模型内层使用更激进的 token 丢弃，因为这些层的语义知识不那么重要。

对于某些数据集，更激进的丢弃可能是可以接受的，但对于 Wan 2.1 来说，0.5 的值相当高。

#### 要点

- **有限的架构支持** - TREAD 仅为 Flux 和 Wan 模型实现
- **高分辨率效果最佳** - 由于注意力的 O(n²) 复杂度，在 1024x1024 及以上分辨率获得最大加速
- **与遮罩损失兼容** - 遮罩区域会自动保留（但这会减少加速效果）
- **支持量化** - 可与 int8/int4/NF4 训练结合使用
- **预期初始损失峰值** - 开始 LoRA/LoKr 训练时，损失会暂时较高但会快速修正

#### 调优技巧

- **保守（注重质量）**：使用 0.1-0.3 的 `selection_ratio`
- **激进（注重速度）**：使用 0.3-0.5 的 `selection_ratio` 并接受质量影响
- **避免早期/晚期层**：不要在第 0-1 层或最后一层进行路由
- **LoRA 训练**：可能会略微变慢 - 尝试不同的配置
- **分辨率越高 = 加速越好**：在 1024px 及以上分辨率效果最佳

#### 已知行为

- 丢弃的 token 越多（`selection_ratio` 越高），训练越快但初始损失越高
- LoRA/LoKr 训练会出现初始损失峰值，随着网络适应会快速修正
  - 使用较不激进的训练配置或在内层使用更高级别的多路由可以缓解此问题
- 某些 LoRA 配置可能训练略慢 - 最佳配置仍在探索中
- RoPE（旋转位置嵌入）实现是可用的，但可能不是 100% 正确

有关详细的配置选项和故障排除，请参阅[完整的 TREAD 文档](../TREAD.md)。


#### 验证提示词

在 `config/config.json` 中有"主验证提示词"，通常是您为单一主题或风格训练的主要 instance_prompt。此外，可以创建一个 JSON 文件，包含验证期间运行的额外提示词。

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

昵称是验证的文件名，所以请保持简短且与您的文件系统兼容。

要将训练器指向此提示词库，请在 `config.json` 末尾添加新行来添加到 TRAINER_EXTRA_ARGS：
<details>
<summary>查看示例配置</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

一组多样化的提示词将有助于确定模型在训练时是否正在崩溃。在此示例中，`<token>` 应替换为您的主题名称（instance_prompt）。

<details>
<summary>查看示例配置</summary>

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
</details>

> ℹ️ Wan 2.1 仅使用 UMT5 文本编码器，其嵌入中包含大量局部信息，这意味着较短的提示词可能没有足够的信息让模型做好工作。请务必使用更长、更具描述性的提示词。

#### CLIP 分数跟踪

目前不应为视频模型训练启用此功能。

# 稳定评估损失

如果您想使用稳定 MSE 损失来评分模型的性能，请参阅[此文档](../evaluation/EVAL_LOSS.md)了解配置和解释评估损失的信息。

#### 验证预览

SimpleTuner 支持使用 Tiny AutoEncoder 模型在生成过程中流式传输中间验证预览。这允许您通过 webhook 回调实时查看正在逐步生成的验证图像。

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
- 已启用验证

将 `validation_preview_steps` 设置为更高的值（例如 3 或 5）以减少 Tiny AutoEncoder 开销。使用 `validation_num_inference_steps=20` 和 `validation_preview_steps=5`，您将在第 5、10、15 和 20 步收到预览图像。

#### 流匹配调度偏移

流匹配模型如 Flux、Sana、SD3、LTX Video 和 Wan 2.1 有一个名为 `shift` 的属性，允许我们使用简单的十进制值来偏移时间步调度的训练部分。

##### 默认值
默认情况下，不应用调度偏移，这会导致时间步采样分布呈 sigmoid 钟形，也称为 `logit_norm`。

##### 自动偏移
一种常被推荐的方法是遵循最近的几项工作，启用分辨率相关的时间步偏移 `--flow_schedule_auto_shift`，它对较大的图像使用较高的偏移值，对较小的图像使用较低的偏移值。这会产生稳定但可能中等的训练结果。

##### 手动指定
_感谢 Discord 上的 General Awareness 提供以下示例_

> ℹ️ 这些示例展示了使用 Flux Dev 时该值如何工作，尽管 Wan 2.1 应该非常相似。

当使用 `--flow_schedule_shift` 值为 0.1（非常低的值）时，只有图像的精细细节会受到影响：
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

当使用 `--flow_schedule_shift` 值为 4.0（非常高的值）时，模型的大型构图特征和潜在的颜色空间会受到影响：
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### 量化模型训练

在 Apple 和 NVIDIA 系统上测试过，Hugging Face Optimum-Quanto 可用于降低精度和 VRAM 要求，仅需 16GB 即可训练。



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

在最初探索将 Wan 2.1 添加到 SimpleTuner 期间，Wan 2.1 产生了可怕的噩梦般的输出，这归结于几个原因：

- 推理步数不足
  - 除非您使用 UniPC，否则可能需要至少 40 步。UniPC 可以稍微减少步数，但您需要自己实验。
- 调度器配置不正确
  - 它使用的是普通的 Euler 流匹配调度，但 Betas 分布似乎效果最好
  - 如果您没有修改过此设置，现在应该没问题
- 分辨率不正确
  - Wan 2.1 只有在其训练的分辨率上才能正确工作，能成功是运气，但通常会得到糟糕的结果
- CFG 值不当
  - Wan 2.1 1.3B 特别对 CFG 值敏感，但 4.0-5.0 左右的值似乎是安全的
- 提示词不当
  - 当然，视频模型似乎需要一个神秘主义者团队在山中进行数月的禅修来学习神圣的提示艺术，因为他们的数据集和标注风格像圣杯一样被守护着。
  - 简而言之，尝试不同的提示词。

尽管如此，除非您的批次大小太低和/或学习率太高，否则模型将在您喜欢的推理工具中正确运行（假设您已经有一个能获得良好结果的工具）。

#### 数据集注意事项

除了处理和训练所需的计算量和时间外，数据集大小几乎没有限制。

您必须确保数据集足够大以有效训练您的模型，但又不能大到超出您可用的计算资源。

请注意，最小数据集大小是 `train_batch_size * gradient_accumulation_steps` 以及大于 `vae_batch_size`。如果数据集太小，将无法使用。

> ℹ️ 如果样本太少，您可能会看到消息 **no samples detected in dataset** - 增加 `repeats` 值可以克服此限制。

根据您拥有的数据集，您需要以不同方式设置数据集目录和数据加载器配置文件。

在此示例中，我们将使用 [video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized) 作为数据集。

创建包含以下内容的 `--data_backend_config`（`config/multidatabackend.json`）文档：

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
    "cache_dir_vae": "cache/vae/wan/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
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
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> 请参阅 [DATALOADER.md](../DATALOADER.md#caption_strategy) 中的 caption_strategy 选项和要求。

- Wan 2.2 图生视频运行会创建 CLIP 条件缓存。在 **video** 数据集条目中，指向专用后端并（可选）覆盖缓存路径：

<details>
<summary>查看示例配置</summary>

```json
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "conditioning_image_embeds": "disney-conditioning",
    "cache_dir_conditioning_image_embeds": "cache/conditioning_image_embeds/disney-black-and-white"
  }
```
</details>

- 定义一次条件后端，可在多个数据集中重用（此处显示完整对象以便清晰理解）：

<details>
<summary>查看示例配置</summary>

```json
  {
    "id": "disney-conditioning",
    "type": "local",
    "dataset_type": "conditioning_image_embeds",
    "cache_dir": "cache/conditioning_image_embeds/disney-conditioning",
    "disabled": false
  }
```
</details>

- 在 `video` 子部分中，我们可以设置以下键：
  - `num_frames`（可选，int）是我们将训练的帧数。
    - 在 15 fps 下，75 帧是 5 秒的视频，是标准输出。这应该是您的目标。
  - `min_frames`（可选，int）确定将被考虑用于训练的视频的最小长度。
    - 这应该至少等于 `num_frames`。不设置它可确保它们相等。
  - `max_frames`（可选，int）确定将被考虑用于训练的视频的最大长度。
  - `bucket_strategy`（可选，string）确定视频如何分组到桶中：
    - `aspect_ratio`（默认）：仅按空间宽高比分组（例如 `1.78`、`0.75`）。
    - `resolution_frames`：按 `WxH@F` 格式的分辨率和帧数分组（例如 `832x480@75`）。适用于混合分辨率/时长的数据集。
  - `frame_interval`（可选，int）使用 `resolution_frames` 时，将帧数四舍五入到此间隔。
<!--  - `is_i2v`（可选，bool）确定是否在数据集上进行图生视频训练。
    - Wan 2.1 默认设置为 True。但是，您可以禁用它。
-->

然后，创建 `datasets` 目录：

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset sayakpaul/video-dataset-disney-organized --local-dir=disney-black-and-white
popd
```

这将把所有迪士尼视频样本下载到您的 `datasets/disney-black-and-white` 目录，该目录将自动为您创建。

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

按照说明登录两个服务。

### 执行训练运行

从 SimpleTuner 目录，您有几个选项来开始训练：

**选项 1（推荐 - pip 安装）：**
```bash
pip install simpletuner[cuda]
simpletuner train
```

**选项 2（Git 克隆方法）：**
```bash
simpletuner train
```

> ℹ️ 当您训练 Wan 2.2 时，在 `TRAINER_EXTRA_ARGS` 或您的 CLI 调用中添加 `--model_flavour i2v-14b-2.2-high`（或 `low`），如果需要，还可以添加 `--wan_validation_load_other_stage`。仅当检查点报告时间嵌入形状不匹配时才添加 `--wan_force_2_1_time_embedding`。

**选项 3（传统方法 - 仍然有效）：**
```bash
./train.sh
```

这将开始将文本嵌入和 VAE 输出缓存到磁盘。

有关更多信息，请参阅[数据加载器](../DATALOADER.md)和[教程](../TUTORIAL.md)文档。

## 注意事项和故障排除技巧

### 最低 VRAM 配置

Wan 2.1 对量化敏感，目前无法与 NF4 或 INT4 一起使用。

- 操作系统：Ubuntu Linux 24
- GPU：单个 NVIDIA CUDA 设备（10G、12G）
- 系统内存：大约 12G 系统内存
- 基础模型精度：`int8-quanto`
- 优化器：Lion 8Bit Paged，`bnb-lion8bit-paged`
- 分辨率：480px
- 批次大小：1，零梯度累积步骤
- DeepSpeed：禁用/未配置
- PyTorch：2.6
- 确保启用 `--gradient_checkpointing`，否则无论您做什么都无法阻止 OOM
- 仅在图像上训练，或为您的视频数据集将 `num_frames` 设置为 1

**注意**：VAE 嵌入和文本编码器输出的预缓存可能会使用更多内存并仍然 OOM。因此，`--offload_during_startup=true` 基本上是必需的。如果是这样，可以启用文本编码器量化和 VAE 平铺。（Wan 目前不支持 VAE 平铺/切片）

速度：
- M3 Max Macbook Pro 上 665.8 秒/迭代
- NVIDIA 4090 上批次大小为 1 时 2 秒/迭代
- NVIDIA 4090 上批次大小为 4 时 11 秒/迭代

### SageAttention

使用 `--attention_mechanism=sageattention` 时，可以在验证时加速推理。

**注意**：这与最终的 VAE 解码步骤不兼容，不会加速该部分。

### 遮罩损失

不要在 Wan 2.1 中使用此功能。

### 量化
- 在 24G 显存中训练此模型不需要量化

### 图像伪影
Wan 需要使用 Euler Betas 流匹配调度或（默认情况下）UniPC 多步求解器，这是一个高阶调度器，可以做出更强的预测。

与其他 DiT 模型一样，如果您执行以下操作（等等），样本中**可能**会开始出现一些方格伪影：
- 使用低质量数据过度训练
- 使用过高的学习率
- 过度训练（一般情况），低容量网络使用太多图像
- 训练不足（同样），高容量网络使用太少图像
- 使用奇怪的宽高比或训练数据大小

### 宽高比分桶
- 视频像图像一样分桶。
- 在正方形裁剪上训练太长时间可能不会对此模型造成太大损害。尽管尝试，它非常棒且可靠。
- 另一方面，使用数据集的自然宽高比桶可能会在推理时过度偏向这些形状。
  - 这可能是一个理想的特性，因为它可以防止电影风格等依赖宽高比的风格过多渗透到其他分辨率。
  - 但是，如果您希望在多个宽高比桶上同样改善结果，您可能需要尝试 `crop_aspect=random`，它有其自身的缺点。
- 通过多次定义图像目录数据集来混合数据集配置产生了非常好的结果和一个良好泛化的模型。

### 训练自定义微调的 Wan 2.1 模型

Hugging Face Hub 上的一些微调模型缺少完整的目录结构，需要设置特定选项。

<details>
<summary>查看示例配置</summary>

```json
{
    "model_family": "wan",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_transformer_model_name_or_path": "path/to-the-other-model",
    "pretrained_vae_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_transformer_subfolder": "none",
}
```
</details>

> 注意：您可以为 `pretrained_transformer_name_or_path` 提供单文件 `.safetensors` 的路径
