## Qwen Image 2.1

Qwen Image 2.1 是默认版本（`model_flavour: "v2.1"`），使用 `Qwen/Qwen-Image-2.1`。它包含 32 层 Transformer、Qwen3-VL 文本编码器和具有 16 倍空间压缩率的 64 通道 VAE。

`qwen_image.peft-lora` 示例使用 `RareConcepts/Domokun` 数据集，以 512px 分辨率训练，触发词为 `🟫`。首先使用 BF16（`base_model_precision: "no_change"`），显存不足时启用梯度检查点。示例为 2.1 使用独立的潜变量和文本缓存；请勿复用旧版本缓存。

```bash
simpletuner train example=qwen_image.peft-lora
```

验证时使用 `validation_guidance: 1.0`、`validation_guidance_real: 1.0` 和 `validation_num_inference_steps: 40`。在验证提示词中保留触发词，以检查模型是否学会了该主体。

Qwen Image 2.1 解码单张图像时不再保留未使用的时序特征缓存。在 H200 上以 BF16 单独解码一张 2048×2048 图像时，峰值已分配显存从 26.87 GiB 降至 15.28 GiB，输出完全一致。分块解码可进一步节省显存，但限制空间上下文可能产生彩色接缝；移除未使用的缓存并不能修复这些接缝。

旧版本仍然可用：`v1.0` 对应 Qwen-Image，`v2.0` 对应 Qwen-Image-2512，`edit-*` 保留原有检查点。这些版本的适配器和潜变量缓存不能与 2.1 互换。

250 步 Domokun 配置用于吞吐量测试，并非可靠的收敛配方。较早的 checkpoint 在重新加载后生成了可辨认的 Domokun，但新启动的 250 步训练未能复现。保留 padding mask、关闭编译以及恢复较早 RoPE 表达式的对照实验也失败了。缓存 latent 解码后确实是正确主体。训练质量下降的原因尚未确定；计时表不能证明不同注意力后端的图像质量相当。

### 显存预设

这些示例在 512px 下使用 BF16、rank-32 LoRA、Optimi Lion 和区域编译，不启用梯度检查点。首次运行需要编译时间；比较速度时应使用预热后的训练步。24 GB 和 32 GB 显存预算在 L40S 上验证，并非在对应容量的独立显卡上测试。

| 显存预算 | 示例 | 数据集批大小 | 峰值显存 (GiB) | 预热后单步 (秒) |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 1 | 20.6 | 0.238 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 2 | 26.5 | 0.390 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 2 | 26.5 | 0.390 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 10 | 71.8 | 0.639 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 20 | 128.4 | 1.223 |

测量使用 L40S（24/32/48 GB 预设）、H100（80 GB）和 H200（144 GB），共运行 20 步，计时排除前五步。峰值显存包括初始化。这些是 512px 下对应批大小的结果，不能保证更大图像或更长提示词使用相同资源。

48 GB 预设也使用批大小 2：在 L40S 上，其每张图像的吞吐量优于批大小 3、4、5。批大小 5 占用 43.3 GiB、每步 0.991 秒；批大小 2 每步为 0.390 秒。

```bash
simpletuner train example=qwen_image-2.1-48g.peft-lora
```

请使用各示例自带的数据集文件，其中明确设置了批大小。更改批大小或数据集设置后应开始新训练，不要复用不兼容的训练状态检查点。

若要降低显存占用，请启用 `gradient_checkpointing: true` 和 `gradient_checkpointing_interval: 2`。现在这会对连续的两个 block 进行分组 checkpoint。实际取舍请参阅 [Qwen Image 2.1 checkpoint 与注意力测量](../experimental/SEGMENTED_CHECKPOINTING.zh.md#qwen-image-21)；此前每隔一个 block 进行 checkpoint 的结果已被替代。这些预设使用 BF16 即可，无需 int8 checkpoint。

文生图路径避免依赖张量数据的序列组装，实现无图中断捕获。实数 RoPE 允许 Inductor 融合归一化与旋转；调制、残差和 MLP 的后处理也参与编译。这些训练示例不使用现有的 Hopper CuTe ConvRot GEMM 或仅支持推理的 LTX RoPE 内核。


### 实验性 AnyFlow 试验

三个 `qwen_image-2.1-anyflow-stage*.peft-lora` 示例测试区间蒸馏能否在引入 `🟫` 的同时保留基础模型的行为。这些配置仍属实验性质；Qwen 模型卡并未证实之前的退化由引导蒸馏导致。

所有阶段使用 1024px、BF16、AdamW、rank 32 和 batch 4。阶段 1 在 H200 上关闭梯度检查点；阶段 2 和 3 对连续两个块进行梯度检查点计算。此负载与上面的 512px 吞吐量预设不同。运行前请安装 `webshart`。

这些 AnyFlow 示例明确使用 `grad_clip_method: "value"` 和 `max_grad_norm: 0.01`，将每个梯度元素限制在 ±0.01 范围内，而不是限制全局梯度范数。若要测试阈值为 1.0 的范数裁剪，须同时设置 `grad_clip_method: "norm"` 和 `max_grad_norm: 1.0`。

1. **阶段 1：** 以 `1e-5` 进行 10,000 次 forward AnyFlow 更新。CC12M 使用 `webshart/cc12m-structured-captions` 和 `caption_key: "long_caption"`；e621 使用 `webshart/e621-2024-webp-4Mpixel-webshart-indices`。两个先验数据集各限制为 4,096 张合格图片，采样权重各为 0.49；`RareConcepts/Domokun` 使用触发词 `🟫`、权重 0.02 和 `repeats: 0`。
2. **阶段 2：** 以 `2e-6` 进行 2,000 次 on-policy DMD 更新，同时在相同数据混合上训练 forward 目标。这是 AnyFlow DMD，并非偏好对 DPO。两个阶段均包含角色样本；仅靠冻结教师无法教授新角色。
3. **阶段 3：** 以 `5e-7` 更新 100 次。Domokun 权重为 0.5，两个正则化数据集权重各为 0.25、各限制为 64 张图片。所有区间使用 `r=t` 和原始 flow 目标，保留区间嵌入器并进行简短监督微调。正则化批次使用禁用 adapter 的基础模型预测。

先检查第 2,000、5,000 和 10,000 次更新的检查点，再决定是否将阶段 1 延长至 20,000 次。阶段 2 的预算为 2,000 次更新；若相同条件下的验证图片退化，应提前停止。更新次数本身并不能证明模型已完成训练。

配置启用了动态编译以处理可变描述长度。`schedule_shift: 2.000802574061872` 对应已发布调度器在 1024px（4,096 个潜变量 token）下的设置；更改分辨率时需重新计算。

```bash
simpletuner train example=qwen_image-2.1-anyflow-stage1.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage2.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage3.peft-lora
```

阶段 2、3 通过 `init_lora` 加载上一阶段的最终 adapter，重新初始化优化器和数据加载器状态。采样权重控制尚未耗尽的数据集之间的交替，并不保证最终样本比例。此受限试验不覆盖完整先验语料库。字符串字段 `long_caption` 可使用现有 Webshart caption selector，无需原生 JSON 对象 caption 支持。

阶段 1、2 使用 `diffusion_target: "base_prediction"` 和 `fuse_guidance_scale: 1.0`：diffusion 分支保留冻结的条件预测场，其余分支从数据学习。这是保留能力的假设，并不保证学会角色。请在 4、16、40 个推理步下比较所附角色和先验提示词，并以基础模型的 40 步结果为参考；自动验证使用 4 步。目标函数和检查点要求见 [AnyFlow](../experimental/ANYFLOW.zh.md)。

已完成的试验：阶段 1 达到 10,000 次更新，阶段 2 达到 2,000 次。重新加载适配器后的 40 步狐狸图像和角色提示词图像仍然连贯，但角色提示词生成的是人而非 Domokun。4 步图像仍然模糊或充满噪声。另一次 L40S 对比对两个检查点均使用 1024px、种子 42、CFG 1/2/4/6 和空负面提示词。前向调用断言确认 CFG 大于 1 时每步有两次前向。已检查的狐狸和海滩样本未得到修复：更高 CFG 增加了饱和度和伪影。这些结果尚不能确定原因；阶段 3 尚未验证。

从同一检查点继续阶段 1 的两个分支各增加了 1,000 次更新，比较逐元素裁剪阈值 0.01 和 1.0。重新加载后的 4 步狐狸图像在两组中仍有明显噪声；40 步海滩图像仍然呈现人物。阈值 0.01 在 65/1,000 次更新中触发裁剪，阈值 1.0 为 0/1,000 次。两组均使用未修正的优化器，且在裁剪首次触发前梯度已存在差异，因此不能把细微变化完全归因于阈值。

<a id="qwen21-optimizer-correction"></a>

此处的阶段 1 和辅助 LoRA 试验均在修复 AdamW BF16 的随机舍入加法辅助函数之前运行：该函数计算的是 `other + alpha * input`，而非 `input + alpha * other`。当 β₁ = 0.9 时，一阶矩因此按 `m = 0.09 * m + g` 更新，而非 `m = 0.9 * m + 0.1 * g`。精确算术回归测试已覆盖 CPU、MPS 和 CUDA。图像观察结果对这些检查点仍然有效，但不能据此独立判断架构或蒸馏目标是否有效；必须使用修正后的优化器重新验证训练。

使用修正后的优化器、相同起始检查点和逐元素裁剪阈值 1.0 再继续更新 1,000 次，仍未修复已检查的 4 步狐狸和海滩图像。40 步狐狸图像保持连贯，而海滩提示词仍生成人物。此试验检验的是现有检查点的恢复能力，而不是使用修正后的优化器从头训练；整体失败原因仍未确定。

### 旧版 Qwen Image 配置（v1.0 / v2.0）

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

<a id="assistant-lora"></a>

### 使用字幕训练辅助 LoRA

`qwen_image-2.1-assistant-lora.peft-lora` 是面向 L40S 的实验起点：BF16、批量 1、间隔 2 的梯度检查点、rank 32，以及学习率 `1e-4` 的 AdamW BF16。预算为 1,000 次更新，每 50 次验证和保存。正式训练前，请用多样化字幕替换十二条冒烟测试字幕；此示例尚未验证收敛效果。

`grad_clip_method: "norm"`, `max_grad_norm: 1.0`.

设置 `distillation_method: assistant_lora`。字幕后端预计算文本嵌入；每批禁用适配器，以 40 个原生推理步、CFG 1 生成新的基础模型潜变量。独立生成管线复用 transformer，不加载 VAE、处理器或文本编码器。随后恢复适配器并执行普通去噪训练，不缓存终态潜变量。目前仅支持 Qwen Image 2.1 文生图。

`distillation_config.assistant_lora` 接受 `num_inference_steps`（默认 40）、`resolutions`（非空 `[宽, 高]` 列表，默认 `[[1024, 1024]]`）和 `seed`（默认 42）。尺寸必须为 32 的倍数。分辨率按批轮换，噪声种子按样本递增，包括不足一批的样本；检查点保存计数器。恢复时保持数据集、批量、梯度累积和分布式拓扑不变。不支持按需文本缓存。

流程测试可用 8 次更新、2 个教师步；评估图像前恢复 40 步。在 100、250、500 和 1,000 次更新时评估，并以相同提示词和种子比较适配器与基础模型。辅助适配器的实际价值仍需另一次概念训练验证。

[Ostris 描述了以低学习率训练模型自身生成图像的方法](https://huggingface.co/ostris/zimage_turbo_training_adapter)。此处训练正向适配器。后续概念训练将 `assistant_lora_path` 指向输出并启用辅助加载；SimpleTuner 在训练时冻结它，在采样时移除它。不会自动下载 Qwen 2.1 辅助适配器，其效果仍属实验。

请将此方法作为唯一蒸馏方法；不支持与其他蒸馏器组合。

字幕数据集要求 `dataloader_prefetch: false`，确保检查点游标对应已消费的字幕。恢复时若字幕标识或内容、批量、重复次数、打乱设置、种子、梯度累积或分布式布局发生变化，将报错。 辅助 LoRA 检查点同样拒绝更改生成种子、分辨率列表或教师推理步数。

<a id="assistant-lora-multires"></a>

#### 四种基础分辨率与宽高比分桶的辅助 LoRA 实验

`qwen_image-2.1-assistant-lora-multires.peft-lora` 循环使用基础分辨率 512、1024、1536 和 2048 对应的共 12 个宽高比桶。每种基础分辨率都有正方形、4:7 竖图和 7:4 横图三个桶；每个批次使用一个桶，在完整循环中各基础分辨率及宽高比的训练次数相同。它保留基础示例的教师 40 步、BF16 批量 1、每 2 个块进行检查点重计算、秩 32、AdamW BF16 和 1,000 次更新，并共用其描述文本后端及验证提示词。请将测试文本替换为基准实验中使用的同一组多样化描述。在新的输出目录中从全新的适配器和优化器开始；`resume_from_checkpoint: ""` 禁用恢复。保留第一次运行的权重以便比较。

此实验检验多种图像尺寸是否能改善辅助适配器，并不证明原生分辨率要求或质量收益。L40S 的 1,000 次更新已完成全部 12 个桶，并在 1024×1024 和 2048×2048 下验证。最终狐狸和肖像图像仍然连贯，但存在分块 VAE 解码已知的颜色接缝。教师直接生成潜变量目标，不经过 VAE 解码。对后续概念训练的收益仍未验证。

Assistant LoRA 与 AnyFlow 一样，仅在运行期间将 Dynamo 缓存下限设为 32 项，以容纳教师、学生和验证的不同变体。保留用户设置的更高上限，退出时恢复原值。增加分辨率或更长描述时，请监测重编译。

后续 Domokun 对照试验以 2048px、batch 1、学习率 `1e-5` 和逐元素裁剪阈值 1.0 分别训练两个全新适配器，各更新 250 次。对照组的训练辅助强度为 0，辅助组为 1；两组推理时均禁用辅助适配器。初始适配器权重及验证图像完全一致。以 1024px、40 个推理步检查最终结果时，两组的角色提示词仍生成人物，狐狸和肖像先验仍然连贯，尚未证明辅助适配器有益。两组训练及辅助适配器的准备过程均使用了[上文](#qwen21-optimizer-correction)所述的未修正优化器。

<a id="assistant-lora-offline"></a>

#### 使用可复用生成图像训练辅助 LoRA

`qwen_image-2.1-assistant-lora-offline.peft-lora` 通过 Webshart 使用 [10,000 张生成图像](https://huggingface.co/datasets/webshart/qwen-image-2.1-generated-images)训练。数据集使用 CC12M 的 `long_caption` 提示词、40 步原生教师推理、CFG 1 和完整图像 VAE 解码。12 个图像后端覆盖 512、1024、1536、2048 基础分辨率的正方形、竖幅和横幅桶，采样权重相等，不设置重复。

此配方从头训练，使用已修正的 `adamw_bf16`、学习率 `1e-4`、`grad_clip_method: "norm"`、`max_grad_norm: 1.0`、BF16、批量 1、秩 32 和每两块一组的梯度检查点。计划训练 1,000 次更新，每 50 次保存，每 100 次以 1024 分辨率验证。发布前另行生成 2048px 预览。关闭 VAE 分块，编码批量为 1。`vae_cache_ondemand: true` 在采样时编码并缓存图像，避免在 1,000 次更新前先编码全部 10,000 张图像。普通图像训练取代在线教师生成；创建辅助适配器时省略 `distillation_method`，并保留 `disable_assistant_lora: true`。

数据集可跨实验复用。PNG 需要再次经过 VAE 编码，因此目标不完全等同于教师的最终潜变量。发布前检查验证图像，并通过独立的概念训练评估辅助效果。从纯字幕配方切换时应重新开始，不要恢复原优化器或数据集状态。

[替换后的 v1 助手](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v1) 已在 L40S 上完成此方案的 1,000 次更新，并检查了最终的 1024 和 2048 图像。随后进行的 Domokun 对照实验采用 2048 分辨率、LR `1e-4`、全局范数裁剪 1.0，各训练 1,000 次更新；助手在训练时冻结，在验证时禁用。对照组和助手组在两个角色提示词下仍生成人物。对照组将明显的 Domokun 特征带入了无关的狐狸提示词；助手组保留了可辨认的狐狸。两组的人像均保持连贯。这仅是概念外溢减少的有限证据，不代表成功学会角色或普遍提升质量；评估仅使用一个训练种子和四个提示词。

```bash
simpletuner train example=qwen_image-2.1-assistant-lora-offline.peft-lora
```
