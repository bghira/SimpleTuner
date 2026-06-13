# zlab i1 快速开始

本指南介绍如何训练 [zlab-princeton i1](https://huggingface.co/zlab-princeton/i1-3B) 的 LoRA。i1 是一个 3B flow-matching transformer，官方发布了 JAX/TPU 训练配方和 PyTorch 推理权重。SimpleTuner 使用原生 PyTorch 集成来训练它，并默认使用 [`bghira/zlab-i1-diffusers`](https://huggingface.co/bghira/zlab-i1-diffusers) 的 Diffusers safetensors 转换。

i1 不是 Flux 的简单变体。它使用 FLUX.2 VAE、T5Gemma 文本编码器、32 通道 latent，以及用于 CFG 的可学习空 caption。

## 硬件需求

1024px LoRA 训练建议：

- 24G 现代 GPU 搭配 int8 量化，可用于小型 LoRA
- 40G+ GPU 会更从容
- 更高 rank、更大数据集或更少量化时建议多 GPU

示例配置使用 `int8-quanto`、`bf16`、`gradient_checkpointing=true` 和 `train_batch_size=1`。推荐 CUDA；不建议使用 Apple GPU 训练 i1。

## 内置示例

```bash
simpletuner train example=zlab-i1.peft-lora
simpletuner train example=zlab-i1.lycoris-lokr
```

建议先跑 PEFT LoRA。只有在需要 LoKr 分解时再使用 LyCORIS LoKr 示例。

## 关键配置

```json
{
  "model_type": "lora",
  "model_family": "zlab_i1",
  "model_flavour": "3b",
  "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.2-dev",
  "validation_resolution": "1024x1024",
  "validation_guidance": 12.0,
  "validation_guidance_rescale": 0.7,
  "validation_num_inference_steps": 250
}
```

`3b` flavour 会解析到 `bghira/zlab-i1-diffusers`，其中 transformer 位于标准 Diffusers `transformer/` 子目录并使用 safetensors。只有测试自定义转换时才需要设置 `pretrained_transformer_model_name_or_path`。

## 验证

验证会使用 i1 的原生 pipeline。快速 smoke test 可以临时降低步数：

```bash
simpletuner train example=zlab-i1.peft-lora validation_num_inference_steps=4 num_eval_images=1
```

4 步只适合确认 pipeline 能生成和保存图片。判断质量前请使用默认的 250 步。

## 高级功能

i1 接入了 SimpleTuner 通用的 transformer 功能路径：

- TwinFlow 可在原生 flow-matching 模式下工作。i1 的 timestep 输入与上游一致会被忽略，因此 TwinFlow 改变的是 noisy latent 轨迹和目标构造，而不是新增时间嵌入。
- CREPA Self-Flow 和 LayerSync 使用 i1 的图像 token hidden-state buffer。CREPA block index 应按 i1 的 29 个 transformer 层设置。
- TREAD 只路由图像 token。文本 token 保持完整，这样 T5Gemma conditioning mask 的语义不会改变。
- 验证支持 CFG Zero*、通过 `validation_no_cfg_until_timestep` 跳过早期 CFG，以及通过 `validation_guidance_skip_layers` 使用 skip-layer guidance。
- 支持 RamTorch、Musubi block swap 和 VAE tiling。RamTorch 与 Musubi 应保持互斥。

## 数据集

i1 使用 FLUX.2 VAE，并期望 32 通道 latent。不要复用 SDXL、Flux.1、PixArt 或其他模型族的 VAE cache。

```json
[
  {
    "id": "my-i1-dataset",
    "type": "local",
    "instance_data_dir": "/datasets/my-subject",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zlab_i1/my-i1-dataset"
  }
]
```

先不修改 PEFT 示例，确认 base benchmark、有限 loss、验证图片和 `pytorch_lora_weights.safetensors` 都正常后，再替换数据集和提示词。
