# SimpleTuner 💹

> ⚠️ **Warning**: The scripts in this repository have the potential to damage your training data. Always maintain backups before proceeding.

**SimpleTuner** is geared towards simplicity, with a focus on making the code easily understood. This codebase serves as a shared academic exercise, and contributions are welcome.

If you'd like to join our community, we can be found [on Discord](https://discord.gg/uRZPwbPEGG) via Terminus Research Group.
If you have any questions, please feel free to reach out to us there.

## Table of Contents

- [Design Philosophy](#design-philosophy)
- [Tutorial](#tutorial)
- [Features](#features)
  - [Flux](#flux1)
  - [Wan 2.1 Video](#wan-video)
  - [LTX Video](#ltx-video)
  - [PixArt Sigma](#pixart-sigma)
  - [NVLabs Sana](#nvlabs-sana)
  - [Stable Diffusion 2.0/2.1](#stable-diffusion-20--21)
  - [Stable Diffusion 3.0](#stable-diffusion-3)
  - [Kwai Kolors](#kwai-kolors)
- [Hardware Requirements](#hardware-requirements)
  - [Flux](#flux1-dev-schnell)
  - [SDXL](#sdxl-1024px)
  - [Stable Diffusion (Legacy)](#stable-diffusion-2x-768px)
- [Scripts](#scripts)
- [Toolkit](#toolkit)
- [Setup](#setup)
- [Troubleshooting](#troubleshooting)

## Design Philosophy

- **Simplicity**: Aiming to have good default settings for most use cases, so less tinkering is required.
- **Versatility**: Designed to handle a wide range of image quantities - from small datasets to extensive collections.
- **Cutting-Edge Features**: Only incorporates features that have proven efficacy, avoiding the addition of untested options.

## Tutorial

Please fully explore this README before embarking on [the tutorial](/TUTORIAL.md), as it contains vital information that you might need to know first.

For a quick start without reading the full documentation, you can use the [Quick Start](/documentation/QUICKSTART.md) guide.

For memory-constrained systems, see the [DeepSpeed document](/documentation/DEEPSPEED.md) which explains how to use 🤗Accelerate to configure Microsoft's DeepSpeed for optimiser state offload.

For multi-node distributed training, [this guide](/documentation/DISTRIBUTED.md) will help tweak the configurations from the INSTALL and Quickstart guides to be suitable for multi-node training, and optimising for image datasets numbering in the billions of samples.

---

## Features

- Multi-GPU training
- Image, video, and caption features (embeds) are cached to the hard drive in advance, so that training runs faster and with less memory consumption
- Aspect bucketing: support for a variety of image/video sizes and aspect ratios, enabling widescreen and portrait training.
- Refiner LoRA or full u-net training for SDXL
- Most models are trainable on a 24G GPU, or even down to 16G at lower base resolutions.
  - LoRA/LyCORIS training for PixArt, SDXL, SD3, and SD 2.x that uses less than 16G VRAM
- DeepSpeed integration allowing for [training SDXL's full u-net on 12G of VRAM](/documentation/DEEPSPEED.md), albeit very slowly.
- Quantised NF4/INT8/FP8 LoRA training, using low-precision base model to reduce VRAM consumption.
- Optional EMA (Exponential moving average) weight network to counteract model overfitting and improve training stability.
- Train directly from an S3-compatible storage provider, eliminating the requirement for expensive local storage. (Tested with Cloudflare R2 and Wasabi S3)
- For only SDXL and SD 1.x/2.x, full [ControlNet model training](/documentation/CONTROLNET.md) (not ControlLoRA or ControlLite)
- Training [Mixture of Experts](/documentation/MIXTURE_OF_EXPERTS.md) for lightweight, high-quality diffusion models
- [Masked loss training](/documentation/DREAMBOOTH.md#masked-loss) for superior convergence and reduced overfitting on any model
- Strong [prior regularisation](/documentation/DATALOADER.md#is_regularisation_data) training support for LyCORIS models
- Webhook support for updating eg. Discord channels with your training progress, validations, and errors
- Integration with the [Hugging Face Hub](https://huggingface.co) for seamless model upload and nice automatically-generated model cards.

### HiDream

Full training support for HiDream is included:

- Memory-efficient training for NVIDIA GPUs (AMD support is planned)
- Dev and Full both functioning and trainable. Fast is untested.
- Optional MoEGate loss augmentation
- Lycoris or full tuning via DeepSpeed ZeRO on a single GPU
- Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-quanto` for major memory savings
- Quantise Llama LLM using `--text_encoder_4_precision` set to `int4-quanto` or `int8-quanto` to run on 24G cards.

See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1

Full training support for Flux.1 is included:

- Classifier-free guidance training
  - Leave it disabled and preserve the dev model's distillation qualities
  - Or, reintroduce CFG to the model and improve its creativity at the cost of inference speed and training time.
- (optional) T5 attention masked training for superior fine details and generalisation capabilities
- LoRA or full tuning via DeepSpeed ZeRO on a single GPU
- Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-quanto` for major memory savings

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video

SimpleTuner has preliminary training integration for Wan 2.1 which has a 14B and 1.3B type, both of which work.

- Text to Video training is supported.
- Image to Video training is not yet supported.
- Text encoder training is not supported.
- VAE training is not supported.
- LyCORIS, PEFT, and full tuning all work as expected
- ControlNet training is not yet supported

See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide to start training.

### LTX Video

SimpleTuner has preliminary training integration for LTX Video, efficiently training on less than 16G.

- Text encoder training is not supported.
- VAE training is not supported.
- LyCORIS, PEFT, and full tuning all work as expected
- ControlNet training is not yet supported

See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide to start training.

### PixArt Sigma

SimpleTuner has extensive training integration with PixArt Sigma - both the 600M & 900M models load without modification.

- Text encoder training is not supported.
- LyCORIS and full tuning both work as expected
- ControlNet training is not yet supported
- [Two-stage PixArt](https://huggingface.co/ptx0/pixart-900m-1024-ft-v0.7-stage1) training support (see: [MIXTURE_OF_EXPERTS](/documentation/MIXTURE_OF_EXPERTS.md))

See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide to start training.

### NVLabs Sana

SimpleTuner has extensive training integration with NVLabs Sana.

This is a lightweight, fun, and fast model that makes getting into model training highly accessible to a wider audience.

- LyCORIS and full tuning both work as expected.
- Text encoder training is not supported.
- PEFT Standard LoRA is not supported.
- ControlNet training is not yet supported

See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide to start training.

### Stable Diffusion 3

- LoRA and full finetuning are supported as usual.
- ControlNet is not yet implemented.
- Certain features such as segmented timestep selection and Compel long prompt weighting are not yet supported.
- Parameters have been optimised to get the best results, validated through from-scratch training of SD3 models

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md) to get going.

### Kwai Kolors

An SDXL-based model with ChatGLM (General Language Model) 6B as its text encoder, **doubling** the hidden dimension size and substantially increasing the level of local detail included in the prompt embeds.

Kolors support is almost as deep as SDXL, minus ControlNet training support.

### Legacy Stable Diffusion models

RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

---

## Hardware Requirements

### NVIDIA

Pretty much anything 3080 and up is a safe bet. YMMV.

### AMD

LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.

Lacking `xformers`, it will use more memory than Nvidia equivalent hardware.

### Apple

LoRA and full-rank tuning are tested to work on an M3 Max with 128G memory, taking about **12G** of "Wired" memory and **4G** of system memory for SDXL.
  - You likely need a 24G or greater machine for machine learning with M-series hardware due to the lack of memory-efficient attention.
  - Subscribing to Pytorch issues for MPS is probably a good idea, as random bugs will make training stop working.

### HiDream [dev, full]

- A100-80G (Full tune with DeepSpeed)
- A100-40G (LoRA, LoKr)
- 3090 24G (LoRA, LoKr)

HiDream has not been tested on 16G cards, but with aggressive quantisation and pre-caching of embeds, you might make it work.


### Flux.1 [dev, schnell]

- A100-80G (Full tune with DeepSpeed)
- A100-40G (LoRA, LoKr)
- 3090 24G (LoRA, LoKr)
- 4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
- 4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

Flux prefers being trained with multiple large GPUs but a single 16G card should be able to do it with quantisation of the transformer and text encoders.

### Auraflow

- A100-80G (Full tune with DeepSpeed)
- A100-40G (LoRA, LoKr)
- 3090 24G (LoRA, LoKr)
- 4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
- 4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### SDXL, 1024px

- A100-80G (EMA, large batches, LoRA @ insane batch sizes)
- A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
- A100-40G (EMA@1024px, EMA@768px, EMA@512px, LoRA @ high batch sizes)
- 4090-24G (EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
- 4080-12G (LoRA @ low-medium batch sizes)

### Stable Diffusion 2.x, 768px

- 16G or better


## Toolkit

For more information about the associated toolkit distributed with SimpleTuner, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup information is available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs for a more detailed insight by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file.

For performance analysis of the training loop, setting `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` will have timestamps that highlight any issues in your configuration.

For a comprehensive list of options available, consult [this documentation](/OPTIONS.md).
