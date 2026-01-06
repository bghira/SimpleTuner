# Z-Image [base / turbo] 快速入门

本示例将训练 Z-Image Turbo LoRA。Z-Image 是一个 6B 的流匹配 transformer（约为 Flux 的一半）并提供 base 与 turbo 两种版本。Turbo 需要一个 assistant adapter；SimpleTuner 可自动加载。

## 硬件要求

Z-Image 比 Flux 省显存，但仍受益于高性能 GPU。训练 rank-16 LoRA 所有组件（MLP、投影层、transformer 块）时，典型显存如下：

- 不量化基础模型时约 ~32-40G VRAM
- 量化为 int8 + bf16 基础/LoRA 权重时约 ~16-24G VRAM
- 量化为 NF4 + bf16 基础/LoRA 权重时约 ~10–12G VRAM

此外，Ramtorch 与组卸载可进一步降低显存。多 GPU 用户可用 FSDP2 以多卡分摊。

你需要：

- **绝对最低**为单张 **3080 10G**（需激进量化/卸载）
- **现实最低**为单张 3090/4090 或 V100/A6000
- **理想**为多张 4090、A6000、L40S 或更强

不建议使用 Apple GPU 训练。

### 内存卸载（可选）

组模块卸载能显著降低 transformer 权重带来的 VRAM 压力。可在 `TRAINER_EXTRA_ARGS`（或 WebUI 硬件页面）中添加：

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# 可选：将卸载权重写入磁盘而非 RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams 仅在 CUDA 生效；SimpleTuner 会在 ROCm、MPS 和 CPU 后端自动禁用。
- **不要**与其他 CPU offload 策略同时使用。
- 组卸载与 Quanto 量化不兼容。
- 若需写入磁盘，请优先使用本地高速 SSD/NVMe。

## 前提条件

确保已安装 Python；SimpleTuner 在 3.10 到 3.12 版本上运行良好。

您可以运行以下命令检查：

```bash
python --version
```

如果您的 Ubuntu 系统未安装 Python 3.12，可以尝试以下命令：

```bash
apt -y install python3.12 python3.12-venv
```

### 容器镜像依赖

对于 Vast、RunPod 和 TensorDock（以及其他平台），在 CUDA 12.x 镜像上可以使用以下命令启用 CUDA 扩展编译：

```bash
apt -y install nvidia-cuda-toolkit-12-8
```

## 安装

通过 pip 安装 SimpleTuner：

```bash
pip install simpletuner[cuda]
```

如需手动安装或开发环境设置，请参阅[安装文档](../INSTALL.md)。

### AMD ROCm 后续步骤

要使 AMD MI300X 可用，必须执行以下操作：

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## 设置环境

### Web 界面方式

SimpleTuner WebUI 可简化配置。启动服务器：

```bash
simpletuner server
```

默认会在 8001 端口创建 Web 服务器，可访问 http://localhost:8001。

### 手动 / 命令行方式

通过命令行运行 SimpleTuner 需要准备配置文件、数据集与模型目录，以及数据加载器配置文件。

#### 配置文件

实验脚本 `configure.py` 可能通过交互式步骤让你跳过本节。

**注意：**这不会配置数据加载器，你仍需手动配置。

运行：

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
- `model_family` - 设置为 `z-image`。
- `model_flavour` - 设为 `turbo`（或 `turbo-ostris-v2` 用于 v2 assistant adapter）；base 版本指向当前不可用的检查点。
- `pretrained_model_name_or_path` - 设置为 `TONGYI-MAI/Z-Image-Turbo`。
- `output_dir` - 设置为您想要存储检查点和验证图像的目录。建议使用完整路径。
- `train_batch_size` - 保持为 1，尤其是数据集很小时。
- `validation_resolution` - Z-Image 为 1024px；使用 `1024x1024` 或多宽高比桶：`1024x1024,1280x768,2048x2048`。
- `validation_guidance` - Turbo 通常使用 0–1 的低 guidance；Base 需要 4-6。
- `validation_num_inference_steps` - Turbo 仅需 8，Base 可用 50-100。
- 若想显著减少 LoRA 大小，可设置 `--lora_rank=4`，有助于降低 VRAM。
- Turbo 需要 assistant adapter（见下文），或显式禁用。

- `gradient_accumulation_steps` - 线性增加训练时长；当需降低 VRAM 时使用。
- `optimizer` - 初学者建议使用 adamw_bf16，其他 adamw/lion 也可。
- `mixed_precision` - 现代 GPU 用 `bf16`，否则 `fp16`。
- `gradient_checkpointing` - 几乎所有设备与场景都应设为 true。
- `gradient_checkpointing_interval` - 可在大 GPU 上设为 2+，每隔 n 个块 checkpoint。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

</details>

### Assistant LoRA（Turbo）

Turbo 需要 assistant adapter：

- `assistant_lora_path`: `ostris/zimage_turbo_training_adapter`
- `assistant_lora_weight_name`:
  - `turbo`: `zimage_turbo_training_adapter_v1.safetensors`
  - `turbo-ostris-v2`: `zimage_turbo_training_adapter_v2.safetensors`

SimpleTuner 会为 turbo 版本自动填充这些参数，除非你覆盖它们。若接受质量下降，可用 `--disable_assistant_lora` 关闭。

### 验证提示词

`config/config.json` 中包含“主验证提示词”，通常是你正在训练的单一主体或风格的主 instance_prompt。此外，还可创建 JSON 文件包含额外验证提示词。

示例配置文件 `config/user_prompt_library.json.example` 格式如下：

<details>
<summary>查看示例配置</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

昵称将作为验证文件名，请保持简短并与文件系统兼容。

要让训练器使用该提示词库，请在 `config.json` 末尾向 TRAINER_EXTRA_ARGS 添加一行：

<details>
<summary>查看示例配置</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

多样化提示词有助于判断模型是否崩溃。本示例中将 `<token>` 替换为你的主体名称（instance_prompt）。

<details>
<summary>查看示例配置</summary>

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
</details>

> ℹ️ Z-Image 是流匹配模型，短提示词若高度相似会几乎生成相同图像。请使用更长、更具体的提示词。

### CLIP 分数跟踪

如需启用评估以评分模型性能，请参阅[此文档](../evaluation/CLIP_SCORES.md)。

### 稳定评估损失

如需使用稳定的 MSE 损失评估模型性能，请参阅[此文档](../evaluation/EVAL_LOSS.md)。

### 验证预览

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

将 `validation_preview_steps` 提高（例如 3 或 5）可降低 Tiny AutoEncoder 开销。

### Flow schedule shifting（流匹配）

Z-Image 等流匹配模型拥有 “shift” 参数用于移动时间步分布中参与训练的部分。基于分辨率的自动偏移是安全默认值。手动提高 shift 会让模型更关注粗特征；降低则偏向细节。对 turbo 模型，修改这些值可能会伤害模型。

### 量化模型训练

TorchAO 或其他量化可降低精度与 VRAM 要求——Optimum Quanto 虽然已不再维护，但仍可使用。

对 `config.json` 用户：

<details>
<summary>查看示例配置</summary>

```json
  "base_model_precision": "int8-torchao",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

### 数据集注意事项

> ⚠️ 训练图像质量至关重要；Z-Image 会快速吸收伪影，可能需要用高质量数据进行最终清洗。

保持数据集足够大（至少 `train_batch_size * gradient_accumulation_steps` 且大于 `vae_batch_size`）。若出现 **no images detected in dataset**，增加 `repeats`。

示例 multi-backend 配置（`config/multidatabackend.json`）：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "pseudo-camera-10k-zimage",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zimage/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/zimage/dreambooth-subject",
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
    "cache_dir_vae": "cache/vae/zimage/dreambooth-subject-512",
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
    "cache_dir": "cache/text/zimage",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

支持同时使用 512px 与 1024px 数据集，可提升收敛表现。

创建 datasets 目录：

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

### 登录 WandB 与 Huggingface Hub

训练前请先登录，尤其当你使用 `--push_to_hub` 和 `--report_to=wandb` 时：

```bash
wandb login
huggingface-cli login
```

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

更多信息请参阅 [dataloader](../DATALOADER.md) 和 [tutorial](../TUTORIAL.md) 文档。

## 多 GPU 配置

SimpleTuner 在 WebUI 中提供**自动 GPU 检测**。在引导流程中，你将配置：

- **自动模式**：自动使用所有检测到的 GPU 并设置最优参数
- **手动模式**：选择特定 GPU 或自定义进程数
- **禁用模式**：单 GPU 训练

WebUI 会自动配置 `--num_processes` 与 `CUDA_VISIBLE_DEVICES`。

手动配置或高级设置请参阅安装指南中的 [多 GPU 训练](../INSTALL.md#multiple-gpu-training)。

## 推理建议

### Guidance 设置

Z-Image 为流匹配模型；较低 guidance（约 0–1）通常能保持质量与多样性。如果训练时使用更高 guidance，请确保推理管线支持 CFG，并预计更慢生成或更高 VRAM（批量 CFG）。

## 注意事项与排错提示

### 最低 VRAM 配置

- GPU：单张 NVIDIA CUDA (10–12G) + 激进量化/卸载
- 系统内存：~32–48G
- 基础模型精度：`nf4-bnb` 或 `int8`
- 优化器：Lion 8Bit Paged（`bnb-lion8bit-paged`）或 adamw 系列
- 分辨率：512px（1024px 需要更多 VRAM）
- 批大小：1，零梯度累积
- DeepSpeed：禁用 / 未配置
- 启动 OOM 时使用 `--quantize_via=cpu`
- 启用 `--gradient_checkpointing`
- 启用 Ramtorch 或组卸载

预缓存阶段可能 OOM；可通过 `--text_encoder_precision=int8-torchao` 与 `--vae_enable_tiling=true` 降低内存。进一步可设置 `--offload_during_startup=true`，避免文本编码器与 VAE 同时占用内存。

### 量化

- 对 16G 卡训练该模型通常至少需要 8bit 量化。
- 量化到 8bit 通常不会伤害训练，并允许更大 batch。
- **int8** 受硬件加速；**nf4-bnb** 更省内存但更敏感。
- 加载 LoRA 推理时，**最好**使用与训练相同的基础模型精度。

### 宽高比分桶

- 仅用方形裁剪训练通常可行，但多宽高比桶可提升泛化。
- 使用原生宽高比桶可能产生形状偏置；若需要更广泛覆盖，可尝试随机裁剪。
- 通过多次定义图像目录来混合数据集配置，通常可获得更好的泛化。

### 学习率

#### LoRA (--lora_type=standard)

- 大型 transformer 往往更适合较低学习率。
- 先使用较低 rank（4–16），再尝试更高 rank。
- 若模型不稳定，降低 `max_grad_norm`；若学习停滞则提高。

#### LoKr (--lora_type=lycoris)

- 较高学习率可能有效（例如 AdamW 1e-3、Lion 2e-4），需自行调参。
- 将正则化数据集标记为 `is_regularisation_data` 有助于保持基础模型。

### 图像伪影

Z-Image 会很快吸收图像伪影。可能需要用高质量数据做最终清理。若学习率过高、数据质量低或宽高比处理不当，容易出现网格伪影。

### 训练自定义微调的 Z-Image 模型

部分微调检查点缺少完整目录结构，必要时设置以下字段：

<details>
<summary>查看示例配置</summary>

```json
{
    "model_family": "z-image",
    "pretrained_model_name_or_path": "TONGYI-MAI/Z-Image-Base",
    "pretrained_transformer_model_name_or_path": "your-custom-transformer",
    "pretrained_vae_model_name_or_path": "TONGYI-MAI/Z-Image-Base",
    "pretrained_transformer_subfolder": "none"
}
```
</details>

## 故障排除

- 启动 OOM：启用组卸载（不与 Quanto 同用）、降低 LoRA rank 或量化（`--base_model_precision int8`/`nf4`）。
- 输出模糊：提高 `validation_num_inference_steps`（如 24–28）或将 guidance 提高到接近 1.0。
- 伪影/过拟合：降低 rank 或学习率，增加多样化提示词，或缩短训练。
- Assistant adapter 问题：Turbo 需要 adapter 路径/权重；只有在接受质量下降时才禁用。
- 验证过慢：降低验证分辨率或步数；流匹配收敛较快。
