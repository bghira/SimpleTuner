# Ideogram 4 快速开始

本指南介绍在 SimpleTuner 中训练 Ideogram 4 LoRA。Ideogram 4 是约 9B 参数的 flow-matching 图像模型，擅长文字、排版和复杂提示。公开权重以 FP8 发布；SimpleTuner 默认使用 FP8 版本。

示例配置：

```bash
simpletuner/examples/ideogram-fp8.peft-lora/config.json
```

## 硬件和精度

推荐起点：

- **默认选择：** FP8 基础权重，bf16 LoRA 可训练权重，rank 16-32。
- **低显存：** 基础模型使用 NF4。
- **高显存：** 如果显存充足，可以使用 bf16-upcast 权重以避免量化加载开销。

80G NVIDIA GPU 上，1024px、batch size 1 的 FP8 或 bf16-upcast LoRA 训练通常可以正常运行。小显存机器建议使用 FP8 或 NF4、rank 8-16、梯度检查点和 offload。Apple GPU 不推荐用于 Ideogram 4 训练。

## 配置

复制示例配置和 dataloader：

```bash
mkdir -p config/examples
cp simpletuner/examples/ideogram-fp8.peft-lora/config.json config/config.json
cp simpletuner/examples/multidatabackend-ideogram-dreambooth-1024px.json config/examples/multidatabackend-ideogram-dreambooth-1024px.json
```

关键字段：

```json
{
  "model_type": "lora",
  "model_family": "ideogram",
  "model_flavour": "fp8",
  "base_model_precision": "no_change",
  "quantize_via": "cpu",
  "mixed_precision": "bf16",
  "train_batch_size": 1,
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "gradient_checkpointing": true,
  "ideogram_auto_json": true,
  "ideogram_validation": true,
  "ideogram_schedule_mu": 0.0,
  "ideogram_schedule_std": 1.5
}
```

FP8 是默认推荐：

```json
{
  "model_flavour": "fp8",
  "base_model_precision": "no_change",
  "quantize_via": "cpu"
}
```

低显存时可改用 NF4：

```json
{
  "base_model_precision": "nf4-bnb",
  "base_model_default_dtype": "bf16",
  "quantize_via": "cpu"
}
```

## 验证

Ideogram 验证默认关闭。要启用：

```json
{
  "ideogram_validation": true
}
```

这是一个临时开关。上游 Ideogram CFG 推理路径预期有单独的 unconditional transformer；SimpleTuner 当前默认只训练 conditional transformer。启用后，验证会用 conditional transformer 做 negative/unconditional pass，因此仍可检查提示词和 negative prompt 的效果。

## Caption 格式

Ideogram 4 更适合结构化 JSON caption。推荐字段包括：

- `high_level_description`
- `style_description`
- `style_description.color_palette`，颜色使用 hex code
- `compositional_deconstruction.background`
- `compositional_deconstruction.elements`
- 可选 `bbox`，格式为 `[x1, y1, x2, y2]`

如果数据集中混合了普通文本和 JSON caption，保留：

```json
{
  "ideogram_auto_json": true
}
```

普通文本会被包装为 Ideogram JSON schema，已有 JSON 会被规范化并保留。手写 JSON caption 仍然更好，特别是包含构图、背景、元素和颜色信息时。

## Prompt upsampling

可选启用 prompt upsampling：

```json
{
  "ideogram_prompt_upsample": true,
  "ideogram_prompt_enhancer_head_id": "diffusers/qwen3-vl-8b-instruct-lm-head"
}
```

它会在 JSON 转换前通过 Ideogram prompt upsampler 改写提示词。此功能更慢，建议先确认基础训练路径正常后再启用。

## LoRA 和 LyCORIS

默认 PEFT LoRA 目标是 attention projection：

```json
{
  "lora_type": "standard",
  "lora_rank": 32
}
```

LyCORIS/LoKr 可使用 Ideogram 暴露的 `Attention` 和 `FeedForward` 模块类。full-matrix LoKr 可能非常大；快速迭代时优先使用标准 LoRA。

## Loss 预期

Ideogram 的 loss 可能比其他模型看起来高。接近或高于 `1.0` 不一定表示模型损坏，也不一定会破坏验证图像的一致性。

测试中，即使 step loss 在大约 `0.3-1.3` 之间波动并偶尔出现更高 spike，Ideogram 仍能生成连贯的验证图像。判断训练时优先看验证图像、提示遵循能力，以及 loss 是否持续爆炸。

## 训练

```bash
simpletuner train
```

开发环境：

```bash
CONFIG_BACKEND=json CONFIG_PATH=config/config.json .venv/bin/python simpletuner/train.py
```
