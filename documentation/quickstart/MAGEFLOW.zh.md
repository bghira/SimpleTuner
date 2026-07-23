# Mage-Flow 快速开始

本指南介绍在 SimpleTuner 中训练 Mage-Flow LoRA。Mage-Flow 是 Microsoft 发布的 4B rectified-flow 图像生成和编辑系列，使用原生分辨率 MMDiT、Qwen3-VL 文本条件，以及 128 通道、16x 下采样的 Mage-VAE。

## 硬件

Mage-Flow 比 Flux.1 和 Qwen-Image 小，但仍然是大型 transformer，并且需要冻结的 Qwen3-VL 文本编码器。

建议起点：

- `bf16`、512px、batch 1 用于快速检查
- `bf16`、1024px、batch 1 用于常规 LoRA
- 显存不足时使用 `int8-torchao` 或 NF4
- Turbo flavours 使用 4 步验证

24GB 可用于低分辨率或量化实验，48GB 更适合 1024px，80GB 更适合编辑训练和更大 batch。

## 配置

安装 SimpleTuner：

```bash
pip install 'simpletuner[cuda]'
```

文本到图像起始配置：

```json
{
  "model_family": "mageflow",
  "model_flavour": "base",
  "model_type": "lora",
  "pretrained_model_name_or_path": "microsoft/Mage-Flow-Base",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "optimizer": "optimi-lion",
  "learning_rate": 1e-4,
  "lora_rank": 32,
  "train_batch_size": 1,
  "resolution": 1024,
  "validation_resolution": "1024x1024",
  "validation_num_inference_steps": 30,
  "validation_guidance": 5.0
}
```

可用 flavours：

- `base` - `microsoft/Mage-Flow-Base`
- `default` - `microsoft/Mage-Flow`
- `turbo` - `microsoft/Mage-Flow-Turbo`
- `edit-base` - `microsoft/Mage-Flow-Edit-Base`
- `edit` - `microsoft/Mage-Flow-Edit`
- `edit-turbo` - `microsoft/Mage-Flow-Edit-Turbo`

编辑训练：

```json
{
  "model_family": "mageflow",
  "model_flavour": "edit-turbo",
  "pretrained_model_name_or_path": "microsoft/Mage-Flow-Edit-Turbo",
  "validation_num_inference_steps": 4
}
```

编辑 flavours 需要 conditioning image 数据集。SimpleTuner 会在 `check_user_config` 中自动切换到编辑 pipeline，方式与 Flux Kontext 类似。

## Dataloader

普通 subject/style LoRA 使用标准图像 dataloader：

```json
[
  {
    "id": "dreambooth-1024",
    "type": "local",
    "instance_data_dir": "/path/to/images",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "resolution": 1024,
    "resolution_type": "pixel",
    "metadata_backend": "discovery",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "cache_dir_vae": "cache/vae/mageflow/dreambooth-1024"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/mageflow"
  }
]
```

编辑训练使用 source/target 配对数据。caption 应该是编辑指令，而不只是目标图像描述。

## 内存预设

Mage-Flow 在内存优化菜单中提供 RAMTorch 和 Musubi block swap 预设。需要将 Transformer 权重常驻 CPU RAM 时使用 RAMTorch；只想在 forward/backward 期间流式加载最后几个 Transformer block 时使用 Musubi block swap。它们在配置器中互斥。

## 验证和量化

`default` 建议约 20 步，`base` 约 30 步，`turbo` / `edit-turbo` 使用 4 步。

```json
{
  "base_model_precision": "int8-torchao",
  "quantize_via": "cpu"
}
```

SimpleTuner vendored 了 MIT 许可的 Mage-Flow 代码，并用原生 Diffusers pipeline 包装以便验证和保存流程一致。
