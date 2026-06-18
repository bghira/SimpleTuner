# Boogu-Image 0.1 快速开始

本指南介绍如何在 SimpleTuner 中训练 Boogu-Image 0.1 的 LoRA 和 LyCORIS LoKr。Boogu-Image 是一个 flow matching 图像模型，包含 text-to-image、turbo 和 edit flavours。SimpleTuner 集成使用本地 pipeline 和 transformer 代码，导出的 pipeline checkpoint 托管在 Hugging Face 的 `SimpleTuner` 命名空间下。

内置起始配置:

```bash
simpletuner/examples/boogu-image-v0.1.peft-lora/config.json
simpletuner/examples/boogu-image-v0.1.lycoris-lokr/config.json
```

## 硬件需求

请把 Boogu-Image 当作大型 transformer 图像模型来规划资源。首次运行建议使用 1024px、batch size 1、bf16 mixed precision，并启用 gradient checkpointing。

推荐起点:

- **默认:** `v0.1-base`，bf16 LoRA 权重，rank 16。
- **更低 VRAM:** 使用 FP8 flavour，例如 `v0.1-base-fp8`、`v0.1-turbo-fp8` 或 `v0.1-edit-fp8`。
- **快速验证/推理:** 使用 turbo flavour，但请注意下面的 assistant LoRA 状态。
- **编辑:** 使用 `v0.1-edit` 或 `v0.1-edit-fp8`，并提供成对 conditioning 数据。

内存使用会随 rank、优化器、验证分辨率、offload、compile 设置以及是否使用 FP8 权重而变化。单张 H100 可以在 1024px 下运行内置 PEFT LoRA 示例 1000 steps，并启用 benchmark 与 validation samples。

较小显卡建议从 FP8 weights、rank 8-16、`train_batch_size=1`、gradient checkpointing 和 model/group offload 开始。

### 内存 offload

当 transformer 权重成为 VRAM 瓶颈时，可以使用 group offload:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

可选磁盘 offload:

```bash
--group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- stream 只在 CUDA 上有效；SimpleTuner 会在 ROCm、MPS 和 CPU 上禁用。
- 不要把 group offload 与其他 CPU offload 策略混用。
- 磁盘 offload 推荐使用高速本地 NVMe。

### Torch compile 与 attention

在 NVIDIA GPU 上，可用时使用 Hugging Face Hub kernel attention alias:

```json
{
  "attention_mechanism": "flash-attn-3-hub",
  "dynamo_backend": "inductor",
  "dynamo_use_regional_compilation": true
}
```

如果某个 GPU/driver 组合在 compiled validation 中生成黑图，先禁用 torch compile 并重新测试，再调整训练配方。

## 安装

通过 pip 安装 SimpleTuner:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell 用户
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

手动安装或开发环境设置请参考[安装文档](../INSTALL.md)。

## 环境设置

### Web 界面

SimpleTuner WebUI 可以创建 Boogu-Image 训练配置:

```bash
simpletuner server
```

打开 http://localhost:8001，并选择 `boogu_image` 作为 model family。

### 手动 / 命令行

复制 `config/config.json.example`:

```bash
cp config/config.json.example config/config.json
```

检查这些值:

- `model_type` - `lora`。
- `lora_type` - PEFT LoRA 使用 `standard`，LyCORIS LoKr 使用 `lycoris`。
- `model_family` - `boogu_image`。
- `model_flavour` - `v0.1-base`、`v0.1-base-fp8`、`v0.1-turbo`、`v0.1-turbo-fp8`、`v0.1-edit` 或 `v0.1-edit-fp8`。
- `pretrained_model_name_or_path` - 通常留空，让 flavour 自动选择 `SimpleTuner/Boogu-Image-0.1-*` pipeline。
- `output_dir` - checkpoint 与验证图像输出目录。
- `train_batch_size` - 从 `1` 开始。
- `resolution` - 从 `1024` 开始。
- `resolution_type` - 多宽高比 bucket 使用 `pixel_area`。
- `validation_resolution` - 使用 `1024x1024`；多个尺寸可用逗号分隔。
- `validation_guidance` - base/edit 建议从 `4.0` 左右开始。
- `validation_num_inference_steps` - base/edit 建议从 `30` 左右开始；turbo 可使用更少 steps。
- `mixed_precision` - 现代 NVIDIA GPU 使用 `bf16`。
- `gradient_checkpointing` - 保持启用。
- `flow_schedule_shift` - 示例使用 `3`。

最小 PEFT LoRA 配置:

```json
{
  "model_type": "lora",
  "model_family": "boogu_image",
  "model_flavour": "v0.1-base",
  "lora_type": "standard",
  "lora_rank": 16,
  "lora_alpha": 16,
  "output_dir": "output/models-boogu-image-v0.1",
  "train_batch_size": 1,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.0,
  "validation_num_inference_steps": 30,
  "validation_prompt": "a polished product photo of a ceramic mug on a walnut desk",
  "validation_steps": 50,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "flow_schedule_shift": 3,
  "optimizer": "adamw_bf16",
  "learning_rate": 1e-4,
  "lr_scheduler": "constant_with_warmup",
  "lr_warmup_steps": 10,
  "max_train_steps": 1000,
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "data_backend_config": "config/examples/multidatabackend-small-dreambooth-1024px.json"
}
```

## 运行示例

```bash
simpletuner train example=boogu-image-v0.1.peft-lora
simpletuner train example=boogu-image-v0.1.lycoris-lokr
```

开发 checkout 形式:

```bash
simpletuner train env=examples/boogu-image-v0.1.peft-lora
simpletuner train env=examples/boogu-image-v0.1.lycoris-lokr
```

## FP8 flavours

需要加载导出的 FP8 pipeline 权重时，使用 `-fp8` flavour:

```json
{
  "model_family": "boogu_image",
  "model_flavour": "v0.1-base-fp8"
}
```

`v0.1-turbo-fp8` 和 `v0.1-edit-fp8` 也使用相同模式。不需要把 SimpleTuner 指向 Boogu 的 `.bin` 文件。

## Turbo assistant LoRA

SimpleTuner 为 `v0.1-turbo` 和 `v0.1-turbo-fp8` 启用了 assistant LoRA 代码路径。目前 adapter path 是 `None` placeholder，因为此集成还没有单独发布的 assistant adapter。

在 adapter 可用之前，请把 turbo 当作导出的 pipeline target，并直接通过 validation 检查质量。最可预测的训练 baseline 是 `v0.1-base`。

## Edit 训练

Boogu edit flavour 需要成对 conditioning 数据。请使用 [Qwen Image Edit quickstart](./QWEN_EDIT.md) 中描述的 paired-reference dataset 结构。

普通 text-to-image LoRA 请使用 base 或 turbo flavour。

## 验证 prompts

`validation_prompt` 是主要验证 prompt。要扩大覆盖面，可以添加 prompt library:

```json
{
  "product": "a polished product photo of <token> on a walnut desk",
  "studio": "a clean studio portrait of <token> with softbox lighting",
  "cinematic": "a cinematic scene featuring <token>, detailed lighting, shallow depth of field"
}
```

在配置中指向它:

```json
{
  "validation_prompt_library": "config/user_prompt_library.json"
}
```

使用差异足够大的 prompts，以便发现过拟合、prompt collapse 和风格漂移。

## 推理

训练后，用训练时相同的 Boogu-Image pipeline flavour 加载保存的 adapter。主要文件通常是:

```bash
output/models-boogu-image-v0.1/pytorch_lora_weights.safetensors
```
