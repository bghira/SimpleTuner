# SimpleTuner 💹

> ⚠️ **Warning**: The scripts in this repository have the potential to damage your training data. Always maintain backups before proceeding.

**SimpleTuner** is a repository dedicated to a set of experimental scripts designed for training optimization. The project is geared towards simplicity, with a focus on making the code easy to read and understand. This codebase serves as a shared academic exercise, and contributions to its improvement are welcome.

* Multi-GPU training
* Aspect bucketing "just works"; fill a folder of images and let it rip
* Multiple datasets can be used in a single training session, each with a different base resolution.
* VRAM-saving techniques, such as pre-computing VAE and text encoder outputs
* Full featured fine-tuning support
  * Bias training (BitFit)
* LoRA training support

## Table of Contents

- [Design Philosophy](#design-philosophy)
- [Tutorial](#tutorial)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
  - [SDXL](#sdxl)
  - [Stable Diffusion 2.0/2.1](#stable-diffusion-2x)
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

For memory-constrained systems, see the [DeepSpeed document](/documentation/DEEPSPEED.md) which explains how to use 🤗Accelerate to configure Microsoft's DeepSpeed for optimiser state offload.

## Features

- Precomputed VAE (latents) outputs saved to storage, eliminating the need to invoke the VAE during training.
- Precomputed captions are run through the text encoder(s) and saved to storage to save on VRAM.
- Trainable on a 24G GPU, or even down to 16G at lower base resolutions.
  - LoRA training for SDXL and SD 2.x that uses less than 16G VRAM.
- DeepSpeed integration allowing for [training SDXL's full u-net on 12G of VRAM](/documentation/DEEPSPEED.md), albeit very slowly.
- Optional EMA (Exponential moving average) weight network to counteract model overfitting and improve training stability. **Note:** This does not apply to LoRA.
- Support for a variety of image sizes and aspect ratios, enabling widescreen and portrait training on SDXL and SD 2.x.
- Train directly from an S3-compatible storage provider, eliminating the requirement for expensive local storage. (Tested with Cloudflare R2 and Wasabi S3)
- DeepFloyd stage I and II full u-net or parameter-efficient fine-tuning via LoRA using 22G VRAM
- Webhook support for updating eg. Discord channels with your training progress, validations, and errors

### Stable Diffusion 2.0/2.1

Stable Diffusion 2.1 is known for difficulty during fine-tuning, but this doesn't have to be the case. Related features in SimpleTuner include:

- Training only the text encoder's later layers
- Enforced zero SNR on the terminal timestep instead of offset noise for clearer images.
- The use of EMA (exponential moving average) during training to ensure we do not "fry" the model.
- The ability to train on multiple datasets with different base resolutions in each, eg. 512px and 768px images simultaneously

## Hardware Requirements

EMA (exponential moving average) weights are a memory-heavy affair, but provide fantastic results at the end of training. Without it, training can still be done, but more care must be taken not to drastically change the model leading to "catastrophic forgetting".

### GPU vendors

* NVIDIA - pretty much anything 3090 and up is a safe bet. YMMV.
* AMD - SDXL LoRA and UNet are verified working on a 7900 XTX 24GB. Lacking `xformers`, it will likely use more memory than Nvidia equivalents
* Apple - LoRA and full u-net tuning are tested to work on an M3 Max with 128G memory, taking about **12G** of "Wired" memory and **4G** of system memory for SDXL.
  * You likely need a 24G or greater machine for machine learning with M-series hardware due to the lack of memory-efficient attention.

### SDXL, 1024px

* A100-80G (EMA, large batches, LoRA @ insane batch sizes)
* A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
* A100-40G (no EMA@1024px, no EMA@768px, EMA@512px, LoRA @ high batch sizes)
* 4090-24G (no EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
* 4080-12G (LoRA @ low-medium batch sizes)

### Stable Diffusion 2.x, 768px

* A100-40, A40, A6000 or better (EMA, 1024px training)
* NVIDIA RTX 4090 or better (24G, no EMA)
* NVIDIA RTX 4080 or better (LoRA only)

## Scripts

* `ubuntu.sh` - This is a basic "installer" that makes it quick to deploy on a Vast.ai instance. It might not work for every single container image.
* `train_sdxl.sh` - The main training script for SDXL.
* `train_sd2x.sh` - This is the Stable Diffusion 1.x / 2.x trainer.
* `sdxl-env.sh.example` - These are the SDXL training parameters, you should copy to `sdxl-env.sh`
* `sd21-env.sh.example` - These are the training parameters, copy to `env.sh`

## Toolkit

For more information about the associated toolkit distributed with SimpleTuner, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup information is available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs for a more detailed insight by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment file.

For a comprehensive list of options available for the SDXL trainer, consult [this documentation](/OPTIONS.md).
