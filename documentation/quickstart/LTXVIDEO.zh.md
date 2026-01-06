## LTX Video 快速入门

在本示例中，我们将使用 Sayak Paul 的 [公共领域 Disney 数据集](https://hf.co/datasets/sayakpaul/video-dataset-disney-organized) 训练一个 LTX-Video LoRA。

### 硬件要求

LTX 对系统内存和 GPU 显存要求都不高。

当你训练 rank-16 LoRA 的所有组件（MLP、投影层、多模态块）时，在 M3 Mac（批大小 4）上大约只会用到 12G 多一些。

你需要:
- **现实最低配置**：16GB 或单张 3090 或 V100 GPU
- **理想配置**：多张 4090、A6000、L40S 或更好的 GPU

Apple Silicon 目前对 LTX 运行良好，但由于 PyTorch MPS 后端限制，分辨率会偏低。

### 内存卸载（可选）

如果接近 VRAM 上限，可在配置中启用组卸载:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# 可选：将卸载权重写入磁盘而非 RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- CUDA 用户可受益于 `--group_offload_use_stream`；其他后端会自动忽略。
- 除非系统内存 <64GB，否则不建议使用 `--group_offload_to_disk_path` — 磁盘暂存更慢但更稳定。
- 使用组卸载时请关闭 `--enable_model_cpu_offload`。

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

#### AMD ROCm 后续步骤

要使 AMD MI300X 可用，必须执行以下操作：

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
- `model_family` - 设置为 `ltxvideo`。
- `pretrained_model_name_or_path` - 设置为 `Lightricks/LTX-Video-0.9.5`。
- `pretrained_vae_model_name_or_path` - 设置为 `Lightricks/LTX-Video-0.9.5`。
- `output_dir` - 设置为您想要存储检查点和验证图像的目录。建议使用完整路径。
- `train_batch_size` - 可以增大以获得更稳定的训练，起步使用 4 即可。
- `validation_resolution` - 设置为你使用 LTX 生成视频时的常用分辨率（`768x512`）。
  - 可用逗号分隔指定多个分辨率：`1280x768,768x512`
- `validation_guidance` - 使用你在 LTX 推理时习惯使用的值。
- `validation_num_inference_steps` - 约 25 步即可节省时间并保持较好质量。
- 如果想显著减少 LoRA 大小，可设置 `--lora_rank=4`，这有助于降低 VRAM 占用，但会降低学习容量。

- `gradient_accumulation_steps` - 将更新累积多个步骤。
  - 该值会线性增加训练时长，例如设为 2 会使训练速度减半、总时间翻倍。
- `optimizer` - 初学者建议使用 adamw_bf16，optimi-lion 与 optimi-stableadamw 也不错。
- `mixed_precision` - 初学者建议保持 `bf16`。
- `gradient_checkpointing` - 几乎所有设备和场景都应设为 true。
- `gradient_checkpointing_interval` - LTX Video 暂不支持，应从配置中移除。

多 GPU 用户可参考[此文档](../OPTIONS.md#environment-configuration-variables)了解如何配置 GPU 数量。

最终配置大致如下：

<details>
<summary>查看示例配置</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "data_backend_config": "config/ltxvideo/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "disable_benchmark": false,
  "offload_during_startup": true,
  "output_dir": "output/ltxvideo",
  "lora_type": "lycoris",
  "lycoris_config": "config/ltxvideo/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "ltxvideo-disney",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "ltxvideo-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5",
  "model_family": "ltxvideo",
  "train_batch_size": 8,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 800,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "768x512",
  "validation_negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "validation_guidance": 3.0,
  "validation_num_inference_steps": 40,
  "validation_prompt": "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day. And then the camera switch to a inding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region.",
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.0,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "fp8-torchao",
  "vae_batch_size": 1,
  "webhook_config": "config/ltxvideo/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "lora_rank": 128,
  "flow_schedule_shift": 1,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

### 可选: CREPA 时间正则

若 LTX 训练出现闪烁或身份漂移，可尝试 CREPA（跨帧对齐）：
- 在 WebUI 中进入 **Training → Loss functions** 并启用 **CREPA**。
- 从 **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0** 开始。
- 保持默认视觉编码器（`dinov2_vitg14`，尺寸 `518`）。若需要更低 VRAM 才切换到 `dinov2_vits14` + `224`。
- 首次需要联网（或已缓存 torch hub）以获取 DINOv2 权重。
- 可选：若完全使用缓存 latents 训练，可启用 **Drop VAE Encoder** 省内存；若需编码新视频则保持关闭。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

#### 验证提示词

`config/config.json` 中包含“主验证提示词”，通常是你在单主体或风格训练中的主 instance_prompt。此外，你还可以创建一个 JSON 文件，包含验证时要跑的额外提示词。

示例配置文件 `config/user_prompt_library.json.example` 的格式如下：

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

昵称会作为验证文件名，请保持简短并与文件系统兼容。

要让训练器使用该提示词库，请在 `config.json` 末尾向 TRAINER_EXTRA_ARGS 添加一行：
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

多样化的提示词有助于判断模型是否发生崩溃。在此示例中，将 `<token>` 替换为你的主体名称（instance_prompt）。

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

> ℹ️ LTX Video 是基于 T5 XXL 的流匹配模型；较短的提示词可能信息不足。请使用更长、更详细的提示词。

#### CLIP 分数跟踪

当前不建议用于视频模型训练。

</details>

# 稳定评估损失

如果您希望使用稳定的 MSE 损失来评估模型性能，请参阅[此文档](../evaluation/EVAL_LOSS.md)了解配置和解释方法。

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

#### 流匹配时间表偏移

Flux、Sana、SD3 和 LTX Video 等流匹配模型拥有 `shift` 属性，可用一个小数值移动时间步分布中参与训练的部分。

##### 默认
默认不应用 schedule shift，会形成 S 形钟形分布（又称 `logit_norm`）。

##### 自动偏移
常见的推荐做法是启用分辨率相关的时间步偏移 `--flow_schedule_auto_shift`。它对大图使用更高 shift 值，对小图使用更低值。结果更稳定，但可能较为中庸。

##### 手动指定
_感谢 Discord 的 General Awareness 提供以下示例_

> ℹ️ 这些示例展示了 Flux Dev 上的效果，LTX Video 应该非常相似。

当使用 `--flow_schedule_shift` 值 0.1（很低）时，只会影响图像的细节：
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

当使用 `--flow_schedule_shift` 值 4.0（很高）时，大的构图特征甚至色彩空间会受到影响：
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### 量化模型训练

在 Apple 和 NVIDIA 系统上测试过，Hugging Face Optimum-Quanto 可用于降低精度与 VRAM 要求，可在 16GB 上训练。



对于 `config.json` 用户:
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

#### 数据集注意事项

除处理和训练所需的计算与时间外，数据集规模限制较少。

请确保数据集足够大以有效训练模型，但不要超过你的算力可承受范围。

请注意最小数据集规模为 `train_batch_size * gradient_accumulation_steps` 且必须大于 `vae_batch_size`。数据集过小将无法使用。

> ℹ️ 如果样本过少，可能出现 **no samples detected in dataset** 提示 — 提高 `repeats` 值可解决此问题。

根据你拥有的数据集，需要以不同方式设置数据集目录和数据加载器配置文件。

本示例使用 [video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized) 数据集。

创建 `--data_backend_config`（`config/multidatabackend.json`）文件，内容如下：

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
    "cache_dir_vae": "cache/vae/ltxvideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 125,
        "min_frames": 125,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

- 在 `video` 子段中可设置以下键:
  - `num_frames`（可选, int）为训练使用的帧数。
    - 25 fps 下，125 帧约为 5 秒视频，属于标准输出。建议作为目标。
  - `min_frames`（可选, int）为训练最短视频长度。
    - 应至少等于 `num_frames`。不设置时默认为 `num_frames`。
  - `max_frames`（可选, int）为训练时允许的最长视频长度。
  - `is_i2v`（可选, bool）表示是否进行 i2v 训练。
    - LTX 默认启用，可关闭。
  - `bucket_strategy`（可选, string）指定分桶方式:
    - `aspect_ratio`（默认）：只按空间宽高比分桶（如 `1.78`、`0.75`）。
    - `resolution_frames`：按 `WxH@F` 格式分桶（如 `768x512@125`）。适用于混合分辨率/时长数据集。
  - `frame_interval`（可选, int）在使用 `resolution_frames` 时用于将帧数取整。设置为模型要求的帧数因子。

然后创建 `datasets` 目录：

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset sayakpaul/video-dataset-disney-organized --local-dir=disney-black-and-white
popd
```

这将把所有 Disney 视频样本下载到 `datasets/disney-black-and-white` 目录，并自动创建目录。

#### 登录 WandB 与 Huggingface Hub

在训练开始前登录 WandB 和 HF Hub，尤其当你使用 `--push_to_hub` 或 `--report_to=wandb` 时。

如果要手动推送到 Git LFS 仓库，还应运行 `git config --global credential.helper store`。

运行以下命令：

```bash
wandb login
```

以及

```bash
huggingface-cli login
```

按照提示完成登录。

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

## 注意事项与排错提示

### 最低 VRAM 配置

与其他模型一样，最低 VRAM 配置可能如下：

- OS: Ubuntu Linux 24
- GPU: 单张 NVIDIA CUDA (10G, 12G)
- 系统内存: 约 11G
- 基础模型精度: `nf4-bnb`
- 优化器: Lion 8Bit Paged, `bnb-lion8bit-paged`
- 分辨率: 480px
- 批大小: 1，零梯度累积
- DeepSpeed: 禁用 / 未配置
- PyTorch: 2.6
- **务必**启用 `--gradient_checkpointing`，否则一定会 OOM

**注意**：预缓存 VAE 嵌入与文本编码器输出可能会占用更多内存并导致 OOM。可启用文本编码器量化与 VAE 切片，进一步可设置 `--offload_during_startup=true` 避免 VAE 与文本编码器同时占用内存。

在 M3 Max Macbook Pro 上使用 Pytorch 2.7 的速度约为 0.8 次迭代/秒。

### SageAttention

使用 `--attention_mechanism=sageattention` 时，推理验证速度可能更快。

**注意**：并不适用于所有模型配置，但值得尝试。

### NF4 量化训练

简单来说，NF4 是一种 4bit-ish 表示，会带来严重的稳定性问题。

早期测试表明：
- Lion 优化器会导致模型崩溃但 VRAM 最少；AdamW 系列更稳定；bnb-adamw8bit、adamw_bf16 是更好的选择
  - AdEMAMix 表现不佳，但设置未深入探索
- `--max_grad_norm=0.01` 有助于防止模型短时间内剧烈变化
- NF4、AdamW8bit 和更高批大小有助于稳定，但会增加训练时间或 VRAM 使用
- 提高分辨率会显著减慢训练并可能伤害模型
- 增加视频长度也会消耗大量内存；降低 `num_frames` 可以缓解
- int8 或 bf16 难训练的内容在 NF4 下更难
- 与 SageAttention 之类的选项兼容性更差

NF4 不支持 torch.compile，因此速度提升有限。

若 VRAM 不是问题，使用 int8 + torch.compile 速度更快。

### Masked loss

不要在 LTX Video 上使用此项。


### 量化
- 训练该模型不需要量化

### 图像伪影
与其他 DiT 模型一样，以下情况可能出现方格伪影:
- 用低质量数据过度训练
- 学习率过高
- 过度训练（低容量网络 + 过多图像）
- 训练不足（高容量网络 + 图像过少）
- 使用非常规的宽高比或训练尺寸

### 宽高比分桶
- 视频的分桶方式与图像相同。
- 长时间在方形裁剪上训练通常不会伤害模型，效果稳定可靠。
- 但使用数据集的原生宽高比分桶可能在推理时偏向这些形状。
  - 这可能是有益的，因为能减少特定风格（如电影感）向其他分辨率的迁移。
  - 若希望在多种宽高比上均衡提升，可能要试试 `crop_aspect=random`，但也有缺点。
- 通过多次定义同一图像目录来混合数据集配置，往往能得到更通用的模型。

### 训练自定义微调的 LTX 模型

Hugging Face Hub 上部分微调模型缺少完整目录结构，需要设置特定选项。

<details>
<summary>查看示例配置</summary>

```json
{
    "model_family": "ltxvideo",
    "pretrained_model_name_or_path": "Lightricks/LTX-Video",
    "pretrained_transformer_model_name_or_path": "path/to-the-other-model",
    "pretrained_vae_model_name_or_path": "Lightricks/LTX-Video",
    "pretrained_transformer_subfolder": "none",
}
```
</details>

## 致谢

[finetrainers](https://github.com/a-r-r-o-w/finetrainers) 项目与 Diffusers 团队。
- 最初借鉴了部分 SimpleTuner 的设计理念
- 目前为更易用的视频训练贡献了洞见和代码
