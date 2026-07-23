# Mage-Flow 快速开始

本指南介绍在 SimpleTuner 中训练 Mage-Flow LoRA。Mage-Flow 是 Microsoft 发布的 4B rectified-flow 图像生成和编辑系列，使用原生分辨率 MMDiT、Qwen3-VL 文本条件，以及 128 通道、16x 下采样的 Mage-VAE。

## 硬件

Mage-Flow 比 Flux.1 和 Qwen-Image 小，但仍然是大型 transformer，并且需要冻结的 Qwen3-VL 文本编码器。

建议起点：

- `bf16`、512px、batch 1 用于快速检查
- `bf16`、1024px、batch 1 用于常规 LoRA
- 在 Ada/Hopper 或更新的 NVIDIA GPU 上，显存不足时优先使用 `fp8wo-torchao`
- Turbo flavours 使用 4 步验证

24GB 可用于低分辨率或量化实验，48GB 更适合 1024px，80GB 更适合编辑训练和更大 batch。

## 配置

安装 SimpleTuner：

```bash
pip install 'simpletuner[cuda]'
```

Mage-Flow 使用 packed variable-length attention。若想使用 FlashAttention 2 且不在本地编译 `flash-attn` 包，请设置 `"attention_mechanism": "flash-attn-varlen-hub"`，让 SimpleTuner 从 Hugging Face Hub 加载 kernel。使用 PyTorch SDPA 时保留默认的 `diffusers` 即可。

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

## Mage Flow (Edit) Considerations

Mage-Flow edit checkpoint 不需要 conditioning 或 reference 数据集。Microsoft 在生成和编辑任务上联合训练了 edit 模型，因此生成先验仍然保留。在 SimpleTuner 中，即使 `model_flavour` 是 `edit-base`、`edit` 或 `edit-turbo`，也可以继续使用普通图像数据集进行 subject、style 或 concept LoRA 微调。

只有在明确想训练编辑行为时，才需要使用 source/target 配对数据。SimpleTuner 会自动使用支持编辑的 pipeline；没有提供 conditioning image 时，验证和 prompt encoding 会走 text-to-image 路径。

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

如需训练可选的编辑行为，请使用 source/target 配对数据。caption 应该是编辑指令，而不只是目标图像描述。

## 内存预设

Mage-Flow 在内存优化菜单中提供 RAMTorch 和 Musubi block swap 预设。需要将 Transformer 权重常驻 CPU RAM 时使用 RAMTorch；只想在 forward/backward 期间流式加载最后几个 Transformer block 时使用 Musubi block swap。它们在配置器中互斥。

## 验证和量化

`default` 建议约 20 步，`base` 约 30 步，`turbo` / `edit-turbo` 使用 4 步。

```json
{
  "base_model_precision": "fp8wo-torchao",
  "quantize_via": "cpu"
}
```

在 Mage-Flow LoRA 快速测试中，int8 量化相比 FP8 weight-only TorchAO 出现了可疑的 loss 峰值。除非你已经在自己的数据集上验证 loss 曲线，否则请避免使用 Mage-Flow 的 int8 预设。NF4 和其他量化预设仍可能有用。

SimpleTuner vendored 了 MIT 许可的 Mage-Flow 代码，并用原生 Diffusers pipeline 包装以便验证和保存流程一致。
