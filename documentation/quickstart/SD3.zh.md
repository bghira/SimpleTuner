## Stable Diffusion 3

在本示例中，我们将使用 SimpleTuner 工具包训练一个 Stable Diffusion 3 模型，并使用 `lora` 模型类型。

### 前提条件

确保您已安装 Python；SimpleTuner 在 3.10 到 3.12 版本上运行良好。

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

接下来需要修改以下变量：

<details>
<summary>查看示例配置</summary>

```json
{
  "model_type": "lora",
  "model_family": "sd3",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
  "output_dir": "/home/user/outputs/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.0,
  "validation_prompt": "your main test prompt here",
  "user_prompt_library": "config/user_prompt_library.json"
}
```
</details>


- `pretrained_model_name_or_path` - 设置为 `stabilityai/stable-diffusion-3.5-large`。注意需要登录 Huggingface 并获得访问权限才能下载该模型。稍后我们会说明登录 Huggingface 的步骤。
  - 如果您更想训练较旧的 SD3.0 Medium（2B），请改用 `stabilityai/stable-diffusion-3-medium-diffusers`。
- `MODEL_TYPE` - 设置为 `lora`。
- `MODEL_FAMILY` - 设置为 `sd3`。
- `OUTPUT_DIR` - 设置为您想要存储检查点和验证图像的目录。建议使用完整路径。
- `VALIDATION_RESOLUTION` - 由于 SD3 是 1024px 模型，可设置为 `1024x1024`。
  - SD3 还在多宽高比桶上进行了微调，可用逗号分隔指定其他分辨率：`1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - SD3 适合非常低的值。设置为 `3.0`。

如果使用 Mac M 系列机器，还有一些额外设置：

- `mixed_precision` 应设置为 `no`。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

#### 量化模型训练

在 Apple 和 NVIDIA 系统上测试过，Hugging Face Optimum-Quanto 可将精度和 VRAM 要求降低到远低于基础 SDXL 训练的水平。



> ⚠️ 如果使用 JSON 配置文件，请在 `config.json` 中使用以下格式，而不是 `config.env`：

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "text_encoder_3_precision": "no_change",
  "optimizer": "adamw_bf16"
}
```

对于 `config.env` 用户（已弃用）：

```bash
</details>

# choices: int8-quanto, int4-quanto, int2-quanto, fp8-quanto
# int8-quanto was tested with a single subject dreambooth LoRA.
# fp8-quanto does not work on Apple systems. you must use int levels.
# int2-quanto is pretty extreme and gets the whole rank-1 LoRA down to about 13.9GB VRAM.
# may the gods have mercy on your soul, should you push things Too Far.
export TRAINER_EXTRA_ARGS="--base_model_precision=int8-quanto"

# Maybe you want the text encoders to remain full precision so your text embeds are cake.
# We unload the text encoders before training, so, that's not an issue during training time - only during pre-caching.
# Alternatively, you can go ham on quantisation here and run them in int4 or int8 mode, because no one can stop you.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --text_encoder_1_precision=no_change --text_encoder_2_precision=no_change"

# When you're quantising the model, --base_model_default_dtype is set to bf16 by default. This setup requires adamw_bf16, but saves the most memory.
# adamw_bf16 only supports bf16 training, but any other optimiser will support both bf16 or fp32 training precision.
export OPTIMIZER="adamw_bf16"
```

#### 数据集注意事项

拥有足够大的数据集来训练模型至关重要。数据集大小有限制，您需要确保数据集足够大以有效训练模型。请注意，最小数据集大小为 `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`，并且要大于 `VAE_BATCH_SIZE`。如果数据集太小将无法使用。

根据您拥有的数据集，您需要以不同方式设置数据集目录和数据加载器配置文件。在此示例中，我们将使用 [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) 作为数据集。

在 `/home/user/simpletuner/config` 目录中创建 multidatabackend.json：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sd3",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 0,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/home/user/simpletuner/output/cache/vae/sd3/pseudo-camera-10k",
    "instance_data_dir": "/home/user/simpletuner/datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sd3/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> 请参阅 [DATALOADER.md](../DATALOADER.md#caption_strategy) 中的 caption_strategy 选项和要求。

然后创建一个 `datasets` 目录：

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
popd
```

这将下载约 10k 张照片样本到您的 `datasets/pseudo-camera-10k` 目录，该目录将自动为您创建。

#### 登录 WandB 和 Huggingface Hub

在开始训练之前，您需要登录 WandB 和 HF Hub，特别是如果您使用 `push_to_hub: true` 和 `--report_to=wandb`。

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

从 SimpleTuner 目录，只需运行：

```bash
bash train.sh
```

这将开始将文本嵌入和 VAE 输出缓存到磁盘。

有关更多信息，请参阅[数据加载器](../DATALOADER.md)和[教程](../TUTORIAL.md)文档。

## 注意事项和故障排查提示

### Skip-layer guidance（SD3.5 Medium）

StabilityAI 建议在 SD 3.5 Medium 推理时启用 SLG（Skip-layer guidance）。这不会影响训练结果，只影响验证样本质量。

`config.json` 推荐值如下：

<details>
<summary>查看示例配置</summary>

```json
{
  "--validation_guidance_skip_layers": [7, 8, 9],
  "--validation_guidance_skip_layers_start": 0.01,
  "--validation_guidance_skip_layers_stop": 0.2,
  "--validation_guidance_skip_scale": 2.8,
  "--validation_guidance": 4.0,
  "--flow_use_uniform_schedule": true,
  "--flow_schedule_auto_shift": true
}
```
</details>

- `..skip_scale` 决定在 skip-layer guidance 中正向提示预测的缩放量。默认值 2.8 适用于 `7, 8, 9` 的 skip 值；如跳过更多层，需要每增加一层就将该值翻倍。
- `..skip_layers` 指定在负向提示预测时跳过哪些层。
- `..skip_layers_start` 指定在推理流程中从哪一阶段开始应用 SLG。
- `..skip_layers_stop` 指定在总推理步数中何时停止 SLG。

SLG 可以在更少的步数内应用，以减弱效果或降低推理速度损失。

LoRA 或 LyCORIS 的大量训练似乎需要调整这些值，但具体如何变化尚不明确。

**推理时必须使用更低的 CFG。**

### 模型不稳定性

SD 3.5 Large 8B 模型在训练期间可能存在不稳定性：

- 较高的 `--max_grad_norm` 会允许模型探索潜在危险的权重更新
- 学习率非常敏感；`1e-5` 在 StableAdamW 下有效，但 `4e-5` 可能会失稳
- 更高的批次大小 **非常有帮助**
- 关闭量化或使用纯 fp32 训练并不会改善稳定性

SD3.5 官方训练代码未随模型发布，开发者只能基于 [SD3.5 仓库内容](https://github.com/stabilityai/sd3.5) 推测训练实现。

SimpleTuner 针对 SD3.5 的支持做了如下调整：
- 排除更多层不参与量化
- 不再默认将 T5 padding 区域置零（`--t5_padding`）
- 提供 `--sd3_clip_uncond_behaviour` 与 `--sd3_t5_uncond_behaviour` 用于选择无条件预测使用空编码字幕（`empty_string`，**默认**）或全零（`zero`，不建议调整）
- SD3.5 的训练损失函数更新为与 StabilityAI/SD3.5 上游仓库一致
- 默认 `--flow_schedule_shift` 更新为 3，与 SD3 的静态 1024px 值一致
  - StabilityAI 后续文档建议 `--flow_schedule_shift=1` 并配合 `--flow_use_uniform_schedule`
  - 社区反馈在多宽高比或多分辨率训练时 `--flow_schedule_auto_shift` 更好
- 将 tokenizer 序列长度硬编码上限更新为 **154**，并提供选项将其恢复到 **77**，以节省磁盘或计算，但会牺牲输出质量


#### 稳定配置值

以下选项被认为能尽量保持 SD3.5 稳定：
- optimizer=adamw_bf16
- flow_schedule_shift=1
- learning_rate=1e-4
- batch_size=4 * 3 GPUs
- max_grad_norm=0.1
- base_model_precision=int8-quanto
- 不使用 loss masking 或正则化数据集（对不稳定性的影响未知）
- `validation_guidance_skip_layers=[7,8,9]`

### 最低 VRAM 配置

- OS: Ubuntu Linux 24
- GPU: 单个 NVIDIA CUDA 设备（10G、12G）
- 系统内存: 约 50G
- 基础模型精度: `nf4-bnb`
- 优化器: Lion 8Bit Paged，`bnb-lion8bit-paged`
- 分辨率: 512px
- 批次大小: 1，零梯度累积
- DeepSpeed: 禁用/未配置
- PyTorch: 2.5

### SageAttention

使用 `--attention_mechanism=sageattention` 时，可在验证时加速推理。

**注意**：这与 _所有_ 模型配置不兼容，但值得尝试。

### Masked loss

如果您要训练主体或风格并希望遮罩其一，请参阅 Dreambooth 指南的[遮罩损失训练](../DREAMBOOTH.md#masked-loss)部分。

### 正则化数据

有关正则化数据集，请参阅 Dreambooth 指南中的[该部分](../DREAMBOOTH.md#prior-preservation-loss)与[该部分](../DREAMBOOTH.md#regularisation-dataset-considerations)。

### 量化训练

关于 SD3 和其他模型的量化配置，请参阅 Dreambooth 指南中的[此部分](../DREAMBOOTH.md#quantised-model-training-loralycoris-only)。

### CLIP 分数跟踪

如需启用评估以评分模型性能，请参阅[此文档](../evaluation/CLIP_SCORES.md)了解 CLIP 分数的配置与解读。

# 稳定评估损失

如需使用稳定的 MSE 损失来评分模型性能，请参阅[此文档](../evaluation/EVAL_LOSS.md)了解评估损失的配置与解读。

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
