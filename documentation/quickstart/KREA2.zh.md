# Krea2 快速开始

本指南介绍在 SimpleTuner 中训练 Krea2 LoRA。Krea2 是大型 flow-matching 图像 transformer，使用 Qwen 风格文本条件和 Qwen Image VAE。它更适合高显存 NVIDIA GPU。

起始示例位于：

```bash
simpletuner/examples/krea2.peft-lora/config.json
```

## 推荐起点

第一次运行时，建议从示例配置开始，并保持保守设置：

```json
{
  "model_family": "krea2",
  "model_flavour": "raw",
  "model_type": "lora",
  "pretrained_model_name_or_path": "krea/Krea-2-Raw",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "fuse_qkv_projections": true,
  "train_batch_size": 1,
  "base_model_precision": "no_change"
}
```

Krea2 是 1024px 原生图像模型，但 512px 和 768px 适合快速迭代和检查数据集。运行稳定后再切换到 1024px dataloader。

## 硬件说明

在我们的测试中，Krea2 可以在 80GB H100 上以 bf16、1024px、batch 1 训练。未启用 compile 时更大的 batch 也能放下，但 compile 会增加 graph/cudagraph 内存，很多大 batch 设置会 OOM。

TorchAO int8 weight-only 可以显著降低 VRAM，但在测试的 SimpleTuner 训练路径中并不比 bf16 更快。内存容量比速度更重要时再使用。

推荐：

- 能放下时使用 `bf16`。
- 需要显存余量时使用 `int8-torchao`。
- 保持 `gradient_checkpointing=true`。
- 保持 `fuse_qkv_projections=true`。
- 只有在确认 batch/resolution 能放下后，才启用 `dynamo_backend=inductor`、`dynamo_mode=reduce-overhead` 和 `dynamo_use_regional_compilation=true`。

## 关键配置值

```json
{
  "model_family": "krea2",
  "model_flavour": "raw",
  "model_type": "lora",
  "pretrained_model_name_or_path": "krea/Krea-2-Raw",
  "base_model_precision": "no_change",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "fuse_qkv_projections": true,
  "optimizer": "optimi-lion",
  "learning_rate": 1e-4,
  "lora_rank": 64,
  "train_batch_size": 1,
  "resolution": 1024,
  "validation_resolution": "1024x1024"
}
```

TorchAO int8：

```json
{
  "base_model_precision": "int8-torchao",
  "quantize_via": "cpu"
}
```

reduce-overhead compile：

```json
{
  "dynamo_backend": "inductor",
  "dynamo_mode": "reduce-overhead",
  "dynamo_use_regional_compilation": true
}
```

## 参考图像训练

Krea2 支持面向编辑数据集的可选参考 latent 条件。若 dataloader 提供成对参考图像或缓存的参考 latents，可启用：

```json
{
  "krea2_reference_latents": true
}
```

参考 latents 必须与目标 latents 的 shape 匹配。

## Dataloader 配置

Krea2 使用与其他图像 transformer 类似的 dataloader 结构。真实训练分辨率由 dataloader JSON 决定，而不只是主配置里的 `resolution`。如果要训练 1024px，请确保 dataloader 中的 `resolution`、`maximum_image_size` 和 `target_downsample_size` 也是 1024。

512px 数据集适合快速测试、检查 caption 和发现裁剪问题。最终质量判断通常需要 1024px run。

本地数据集使用 `type: local`，设置 `instance_data_dir`，并选择 caption strategy。小型 subject LoRA 可以从 `caption_strategy=instanceprompt` 开始；风格 LoRA 通常更适合 filenames 或完整 captions。

## 验证

Krea2 验证成本较高，调参时先使用少量 prompts。单个 prompt 可能掩盖 overfit 或记忆问题；运行稳定后再加入小型 prompt library。

```json
{
  "validation_prompt": "a studio portrait of <token>, soft directional light, detailed fabric texture",
  "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
  "validation_num_inference_steps": 28,
  "validation_guidance": 4.5,
  "validation_resolution": "1024x1024"
}
```

## 量化说明

`int8-torchao` 将 transformer 基础权重以 int8 存储，并在其上训练 bf16 LoRA 权重。在 H100 上它显著降低了 VRAM，但在该训练路径中比 bf16 慢。它主要是容量选项，不是吞吐保证。

## Benchmark 结果

以下数据来自单张 NVIDIA H100 80GB，使用 SimpleTuner 真实 trainer、Krea2 LoRA、QKV fusion、gradient checkpointing 和小型 Domokun 数据集。VRAM 通过 `nvidia-smi` 外部采样。请只将这些数值作为比较参考；不同 PyTorch、CUDA、驱动、数据集、LoRA rank、优化器、attention backend 和 GPU 都可能改变结果。

### QKV Fusion + Checkpointing，关闭 Compile

| 精度 | 分辨率 | Batch | 稳定 s/step | 峰值 VRAM |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 512 | 1 | 0.353 | 31.10 GiB |
| bf16 | 512 | 4 | 1.230 | 39.31 GiB |
| bf16 | 512 | 8 | 2.430 | 50.32 GiB |
| bf16 | 1024 | 1 | 0.990 | 33.28 GiB |
| bf16 | 1024 | 4 | 3.850 | 48.35 GiB |
| bf16 | 1024 | 8 | 7.690 | 67.88 GiB |
| int8-torchao | 512 | 1 | 0.535 | 18.10 GiB |
| int8-torchao | 512 | 4 | 1.690 | 27.46 GiB |
| int8-torchao | 512 | 8 | 3.220 | 40.52 GiB |
| int8-torchao | 1024 | 1 | 1.330 | 20.35 GiB |
| int8-torchao | 1024 | 4 | 4.850 | 36.99 GiB |
| int8-torchao | 1024 | 8 | 9.520 | 58.84 GiB |

### QKV Fusion + Checkpointing + Reduce-Overhead Compile

| 精度 | 分辨率 | Batch | 状态 | 稳定 s/step | 峰值 VRAM |
| --- | ---: | ---: | --- | ---: | ---: |
| bf16 | 512 | 1 | ok | 0.260 | 41.20 GiB |
| bf16 | 512 | 4 | OOM | - | 79.07 GiB |
| bf16 | 512 | 8 | OOM | - | 79.10 GiB |
| bf16 | 1024 | 1 | ok | 0.704 | 63.71 GiB |
| bf16 | 1024 | 4 | OOM | - | 79.11 GiB |
| bf16 | 1024 | 8 | OOM | - | 78.40 GiB |
| int8-torchao | 512 | 1 | ok | 0.410 | 30.93 GiB |
| int8-torchao | 512 | 4 | ok | 1.300 | 78.60 GiB |
| int8-torchao | 512 | 8 | OOM | - | 79.12 GiB |
| int8-torchao | 1024 | 1 | ok | 0.990 | 58.68 GiB |
| int8-torchao | 1024 | 4 | OOM | - | 78.92 GiB |
| int8-torchao | 1024 | 8 | OOM | - | 78.09 GiB |

## 实用建议

- 在单张 H100 上最快迭代：bf16、QKV fusion、checkpointing、开启 compile、batch 1。
- 若需要更大的有效 batch，优先使用未 compile 的 bf16，并逐步提高 `train_batch_size` 直到 VRAM 成为限制。
- 内存受限时使用 `int8-torchao`；VRAM 更低，但 step 更慢。
- compile 对 batch 1 有用，但可能消耗大量 VRAM，导致更大 batch 失败。

## 常见问题

- 如果期望 1024px 但日志显示 512px，请检查 dataloader JSON。
- 如果 compile OOM 但非 compile 能运行，请降低 batch size 或关闭 compile。
- 如果 int8 显存更低但更慢，这与我们的 H100 测试一致。
- 如果参考图像没有影响验证结果，请确认 `krea2_reference_latents=true` 且验证 dataset 使用成对参考数据。
- 如果很快 overfit，请降低 learning rate、减少 steps 或增加数据集多样性。
