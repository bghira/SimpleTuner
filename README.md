# SimpleTuner 💹

> ⚠️ **Warning**: The scripts in this repository have the potential to damage your training data. Always maintain backups before proceeding.

**SimpleTuner** is a repository dedicated to a set of experimental scripts designed for training optimization. The project is geared towards simplicity, with a focus on making the code easy to read and understand. This codebase serves as a shared academic exercise, and contributions are welcome.

## Table of Contents

- [Design Philosophy](#design-philosophy)
- [Tutorial](#tutorial)
- [Features](#features)
  - [Flux](#flux1)
  - [PixArt Sigma](#pixart-sigma)
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

---

## Features

- Multi-GPU training
- Image and caption features (embeds) are cached to the hard drive in advance, so that training runs faster and with less memory consumption
- Aspect bucketing: support for a variety of image sizes and aspect ratios, enabling widescreen and portrait training.
- Refiner LoRA or full u-net training for SDXL
- Most models are trainable on a 24G GPU, or even down to 16G at lower base resolutions.
  - LoRA/LyCORIS training for PixArt, SDXL, SD3, and SD 2.x that uses less than 16G VRAM
- DeepSpeed integration allowing for [training SDXL's full u-net on 12G of VRAM](/documentation/DEEPSPEED.md), albeit very slowly.
- Quantised LoRA training, using low-precision base model or text encoder weights to reduce VRAM consumption while still allowing DreamBooth.
- Optional EMA (Exponential moving average) weight network to counteract model overfitting and improve training stability. **Note:** This does not apply to LoRA.
- Train directly from an S3-compatible storage provider, eliminating the requirement for expensive local storage. (Tested with Cloudflare R2 and Wasabi S3)
- For only SDXL and SD 1.x/2.x, full [ControlNet model training](/documentation/CONTROLNET.md) (not ControlLoRA or ControlLite)
- Training [Mixture of Experts](/documentation/MIXTURE_OF_EXPERTS.md) for lightweight, high-quality diffusion models
- Webhook support for updating eg. Discord channels with your training progress, validations, and errors
- Integration with the [Hugging Face Hub](https://huggingface.co) for seamless model upload and nice automatically-generated model cards.

### Flux.1

Preliminary training support for Flux.1 is included:

- Low loss training using optimised approach
  - Preserve the dev model's distillation qualities
  - Or, reintroduce CFG to the model and improve its creativity at the cost of inference speed.
- LoRA or full tuning via DeepSpeed ZeRO
- ControlNet training is not yet supported
- Train either Schnell or Dev models
- Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-quanto` for major memory savings

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### PixArt Sigma

SimpleTuner has extensive training integration with PixArt Sigma - both the 600M & 900M models load without any fuss.

- Text encoder training is not supported, as T5 is enormous.
- LoRA and full tuning both work as expected
- ControlNet training is not yet supported
- [Two-stage PixArt](https://huggingface.co/ptx0/pixart-900m-1024-ft-v0.7-stage1) training support (see: [MIXTURE_OF_EXPERTS](/documentation/MIXTURE_OF_EXPERTS.md))

See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide to start training.

### Stable Diffusion 2.0 & 2.1

Stable Diffusion 2.1 is known for difficulty during fine-tuning, but this doesn't have to be the case. Related features in SimpleTuner include:

- Training only the text encoder's later layers
- Enforced zero SNR on the terminal timestep instead of offset noise for clearer images.
- The use of EMA (exponential moving average) during training to ensure we do not "fry" the model.
- The ability to train on multiple datasets with different base resolutions in each, eg. 512px and 768px images simultaneously

### Stable Diffusion 3

- LoRA and full finetuning are supported as usual.
- ControlNet is not yet implemented.
- Certain features such as segmented timestep selection and Compel long prompt weighting are not yet supported.
- Parameters have been optimised to get the best results, validated through from-scratch training of SD3 models

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md) to get going.

### Kwai Kolors

An SDXL-based model with ChatGLM (General Language Model) 6B as its text encoder, **doubling** the hidden dimension size and substantially increasing the level of local detail included in the prompt embeds.

Kolors support is almost as deep as SDXL, minus ControlNet training support.

---

## Hardware Requirements

EMA (exponential moving average) weights are a memory-heavy affair, but provide fantastic results at the end of training. Options like `--ema_cpu_only` can improve this situation by loading EMA weights onto the CPU and then keeping them there.

Without EMA, more care must be taken not to drastically change the model leading to "catastrophic forgetting" through the use of regularisation data.

### GPU vendors

- NVIDIA - pretty much anything 3090 and up is a safe bet. YMMV.
- AMD - SDXL LoRA and UNet are verified working on a 7900 XTX 24GB. Lacking `xformers`, it will likely use more memory than Nvidia equivalents
- Apple - LoRA and full u-net tuning are tested to work on an M3 Max with 128G memory, taking about **12G** of "Wired" memory and **4G** of system memory for SDXL.
  - You likely need a 24G or greater machine for machine learning with M-series hardware due to the lack of memory-efficient attention.

### Flux.1 [dev, schnell]

- A100-40G (LoRA, rank-128 or lower)
- A100-80G (LoRA, up to rank-256, Full tune with DeepSpeed)

Flux prefers being trained with multiple large GPUs but a single 16G card should be able to do it with quantisation.

### SDXL, 1024px

- A100-80G (EMA, large batches, LoRA @ insane batch sizes)
- A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
- A100-40G (no EMA@1024px, no EMA@768px, EMA@512px, LoRA @ high batch sizes)
- 4090-24G (no EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
- 4080-12G (LoRA @ low-medium batch sizes)

### Stable Diffusion 2.x, 768px

- A100-40, A40, A6000 or better (EMA, 1024px training)
- NVIDIA RTX 4090 or better (24G, no EMA)
- NVIDIA RTX 4080 or better (LoRA only)

## Scripts

- `ubuntu.sh` - This is a basic "installer" that makes it quick to deploy on a Vast.ai instance. It might not work for every single container image.
- `train.sh` - The main training script for SDXL.
- `config/config.env.example` - These are training parameters, you should copy to `config/config.env`

## Toolkit

For more information about the associated toolkit distributed with SimpleTuner, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup information is available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs for a more detailed insight by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment file.

For performance analysis of the training loop, setting `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` will have timestamps that hilight any issues in your configuration.

For a comprehensive list of options available, consult [this documentation](/OPTIONS.md).

## Discord

For more help or to discuss training with like-minded folks, join [our Discord server](https://discord.gg/cSmvcU9Me9)
