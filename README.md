# SimpleTuner 💹

> ℹ️ No data is sent to any third parties except through opt-in flag `report_to`, `push_to_hub`, or webhooks which must be manually configured.

**SimpleTuner** is geared towards simplicity, with a focus on making the code easily understood. This codebase serves as a shared academic exercise, and contributions are welcome.

If you'd like to join our community, we can be found [on Discord](https://discord.gg/JGkSwEbjRb) via Terminus Research Group.
If you have any questions, please feel free to reach out to us there.

<img width="1944" height="1657" alt="image" src="https://github.com/user-attachments/assets/af3a24ec-7347-4ddf-8edf-99818a246de1" />


## Table of Contents

- [Design Philosophy](#design-philosophy)
- [Tutorial](#tutorial)
- [Features](#features)
  - [Core Training Features](#core-training-features)
  - [Model Architecture Support](#model-architecture-support)
  - [Advanced Training Techniques](#advanced-training-techniques)
  - [Model-Specific Features](#model-specific-features)
  - [Quickstart Guides](#quickstart-guides)
- [Hardware Requirements](#hardware-requirements)
- [Toolkit](#toolkit)
- [Setup](#setup)
- [Troubleshooting](#troubleshooting)

## Design Philosophy

- **Simplicity**: Aiming to have good default settings for most use cases, so less tinkering is required.
- **Versatility**: Designed to handle a wide range of image quantities - from small datasets to extensive collections.
- **Cutting-Edge Features**: Only incorporates features that have proven efficacy, avoiding the addition of untested options.

## Tutorial

Please fully explore this README before embarking on the [new web UI tutorial](/documentation/webui/TUTORIAL.md) or [the class command-line tutorial](/documentation/TUTORIAL.md), as this document contains vital information that you might need to know first.

For a manually configured quick start without reading the full documentation or using any web interfaces, you can use the [Quick Start](/documentation/QUICKSTART.md) guide.

For memory-constrained systems, see the [DeepSpeed document](/documentation/DEEPSPEED.md) which explains how to use 🤗Accelerate to configure Microsoft's DeepSpeed for optimiser state offload. For DTensor-based sharding and context parallelism, read the [FSDP2 guide](/documentation/FSDP2.md) which covers the new FullyShardedDataParallel v2 workflow inside SimpleTuner.

For multi-node distributed training, [this guide](/documentation/DISTRIBUTED.md) will help tweak the configurations from the INSTALL and Quickstart guides to be suitable for multi-node training, and optimising for image datasets numbering in the billions of samples.

---

## Features

SimpleTuner provides comprehensive training support across multiple diffusion model architectures with consistent feature availability:

### Core Training Features

- **User-friendly web UI** - Manage your entire training lifecycle through a sleek dashboard
- **Multi-modal training** - Unified pipeline for **Image, Video, and Audio** generative models
- **Multi-GPU training** - Distributed training across multiple GPUs with automatic optimization
- **Advanced caching** - Image, video, audio, and caption embeddings cached to disk for faster training
- **Aspect bucketing** - Support for varied image/video sizes and aspect ratios
- **Concept sliders** - Slider-friendly targeting for LoRA/LyCORIS/full (via LyCORIS `full`) with positive/negative/neutral sampling and per-prompt strength; see [Slider LoRA guide](/documentation/SLIDER_LORA.md)
- **Memory optimization** - Most models trainable on 24G GPU, many on 16G with optimizations
- **DeepSpeed & FSDP2 integration** - Train large models on smaller GPUs with optim/grad/parameter sharding, context parallel attention, gradient checkpointing, and optimizer state offload
- **S3 training** - Train directly from cloud storage (Cloudflare R2, Wasabi S3)
- **EMA support** - Exponential moving average weights for improved stability and quality
- **Custom experiment trackers** - Drop an `accelerate.GeneralTracker` into `simpletuner/custom-trackers` and use `--report_to=custom-tracker --custom_tracker=<name>`
- **Custom experiment trackers** - Drop an `accelerate.GeneralTracker` into `simpletuner/custom-trackers` and use `--report_to=custom-tracker --custom_tracker=<name>`

### Model Architecture Support

| Model | Parameters | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders |
|-------|------------|-----------|---------|-----------|------------|--------------|---------------|---------------|
| **Stable Diffusion XL** | 3.5B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L/G |
| **Stable Diffusion 3** | 2B-8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L/G + T5-XXL |
| **Flux.1** | 12B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL |
| **Flux.2** | 32B | ✓ | ✓ | ✓* | ✗ | int8/fp8/nf4 | ✓ | Mistral-3 Small |
| **ACE-Step** | 3.5B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **Chroma 1** | 8.9B | ✓ | ✓ | ✓* | ✗ | int8/fp8/nf4 | ✓ | T5-XXL |
| **Auraflow** | 6.8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | UMT5-XXL |
| **PixArt Sigma** | 0.6B-0.9B | ✗ | ✓ | ✓ | ✓ | int8 | ✗ | T5-XXL |
| **Sana** | 0.6B-4.8B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2-2B |
| **Lumina2** | 2B | ✓ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2 |
| **Kwai Kolors** | 5B | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ChatGLM-6B |
| **LTX Video** | 5B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **Wan Video** | 1.3B-14B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **HiDream** | 17B (8.5B MoE) | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL + Llama |
| **Cosmos2** | 2B-14B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | T5-XXL |
| **OmniGen** | 3.8B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **Qwen Image** | 20B | ✓ | ✓ | ✓* | ✗ | int8/nf4 (req.) | ✓ | T5-XXL |
| **SD 1.x/2.x (Legacy)** | 0.9B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L |

*✓ = Supported, ✗ = Not supported, * = Requires DeepSpeed for full-rank training*

### Advanced Training Techniques

- **TREAD** - Token-wise dropout for transformer models, including Kontext training
- **Masked loss training** - Superior convergence with segmentation/depth guidance
- **Prior regularization** - Enhanced training stability for character consistency
- **Gradient checkpointing** - Configurable intervals for memory/speed optimization
- **Loss functions** - L2, Huber, Smooth L1 with scheduling support
- **SNR weighting** - Min-SNR gamma weighting for improved training dynamics
- **Group offloading** - Diffusers v0.33+ module-group CPU/disk staging with optional CUDA streams
- **Validation adapter sweeps** - Temporarily attach LoRA adapters (single or JSON presets) during validation to measure adapter-only or comparison renders without touching the training loop
- **External validation hooks** - Swap the built-in validation pipeline or post-upload steps for your own scripts, so you can run checks on another GPU or forward artifacts to any cloud provider of your choice ([details](/documentation/OPTIONS.md#validation_method))
- **LoRA I/O formats** - Load/save PEFT LoRAs in standard Diffusers layout or ComfyUI-style `diffusion_model.*` keys (Flux/Flux2/Lumina2/Z-Image auto-detect ComfyUI inputs)

### Model-Specific Features

- **Flux Kontext** - Edit conditioning and image-to-image training for Flux models
- **PixArt two-stage** - eDiff training pipeline support for PixArt Sigma
- **Flow matching models** - Advanced scheduling with beta/uniform distributions
- **HiDream MoE** - Mixture of Experts gate loss augmentation
- **T5 masked training** - Enhanced fine details for Flux and compatible models
- **QKV fusion** - Memory and speed optimizations (Flux, Lumina2)
- **TREAD integration** - Selective token routing for most models
- **Wan 2.x I2V** - High/low stage presets plus a 2.1 time-embedding fallback (see Wan quickstart)
- **Classifier-free guidance** - Optional CFG reintroduction for distilled models

### Quickstart Guides

Detailed quickstart guides are available for all supported models:

- **[Flux.1 Guide](/documentation/quickstart/FLUX.md)** - Includes Kontext editing support and QKV fusion
- **[Flux.2 Guide](/documentation/quickstart/FLUX2.md)** - **NEW!** Latest enormous Flux model with Mistral-3 text encoder
- **[Z-Image Guide](/documentation/quickstart/ZIMAGE.md)** - Base/Turbo LoRA with assistant adapter + TREAD acceleration
- **[ACE-Step Guide](/documentation/quickstart/ACE_STEP.md)** - **NEW!** Audio generation model training (text-to-music)
- **[Chroma Guide](/documentation/quickstart/CHROMA.md)** - Lodestone's flow-matching transformer with Chroma-specific schedules
- **[Stable Diffusion 3 Guide](/documentation/quickstart/SD3.md)** - Full and LoRA training with ControlNet
- **[Stable Diffusion XL Guide](/documentation/quickstart/SDXL.md)** - Complete SDXL training pipeline
- **[Auraflow Guide](/documentation/quickstart/AURAFLOW.md)** - Flow-matching model training
- **[PixArt Sigma Guide](/documentation/quickstart/SIGMA.md)** - DiT model with two-stage support
- **[Sana Guide](/documentation/quickstart/SANA.md)** - Lightweight flow-matching model
- **[Lumina2 Guide](/documentation/quickstart/LUMINA2.md)** - 2B parameter flow-matching model
- **[Kwai Kolors Guide](/documentation/quickstart/KOLORS.md)** - SDXL-based with ChatGLM encoder
- **[LTX Video Guide](/documentation/quickstart/LTXVIDEO.md)** - Video diffusion training
- **[Hunyuan Video 1.5 Guide](/documentation/quickstart/HUNYUANVIDEO.md)** - 8.3B flow-matching T2V/I2V with SR stages
- **[Wan Video Guide](/documentation/quickstart/WAN.md)** - Video flow-matching with TREAD support
- **[HiDream Guide](/documentation/quickstart/HIDREAM.md)** - MoE model with advanced features
- **[Cosmos2 Guide](/documentation/quickstart/COSMOS2IMAGE.md)** - Multi-modal image generation
- **[OmniGen Guide](/documentation/quickstart/OMNIGEN.md)** - Unified image generation model
- **[Qwen Image Guide](/documentation/quickstart/QWEN_IMAGE.md)** - 20B parameter large-scale training
- **[Stable Cascade Stage C Guide](/quickstart/STABLE_CASCADE_C.md)** - Prior LoRAs with combined prior+decoder validation
- **[Kandinsky 5.0 Image Guide](/documentation/quickstart/KANDINSKY5_IMAGE.md)** - Image generation with Qwen2.5-VL + Flux VAE
- **[Kandinsky 5.0 Video Guide](/documentation/quickstart/KANDINSKY5_VIDEO.md)** - Video generation with HunyuanVideo VAE

---

## Hardware Requirements

### General Requirements

- **NVIDIA**: RTX 3080+ recommended (tested up to H200)
- **AMD**: 7900 XTX 24GB and MI300X verified (higher memory usage vs NVIDIA)
- **Apple**: M3 Max+ with 24GB+ unified memory for LoRA training

### Memory Guidelines by Model Size

- **Large models (12B+)**: A100-80G for full-rank, 24G+ for LoRA/Lycoris
- **Medium models (2B-8B)**: 16G+ for LoRA, 40G+ for full-rank training
- **Small models (<2B)**: 12G+ sufficient for most training types

**Note**: Quantization (int8/fp8/nf4) significantly reduces memory requirements. See individual [quickstart guides](#quickstart-guides) for model-specific requirements.

## Setup

SimpleTuner can be installed via pip for most users:

```bash
# Base installation (CPU-only PyTorch)
pip install simpletuner

# CUDA users (NVIDIA GPUs)
pip install simpletuner[cuda]

# ROCm users (AMD GPUs)
pip install simpletuner[rocm]

# Apple Silicon users (M1/M2/M3/M4 Macs)
pip install simpletuner[apple]
```

For manual installation or development setup, see the [installation documentation](/documentation/INSTALL.md).

## Troubleshooting

Enable debug logs for a more detailed insight by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file.

For performance analysis of the training loop, setting `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` will have timestamps that highlight any issues in your configuration.

For a comprehensive list of options available, consult [this documentation](/documentation/OPTIONS.md).
