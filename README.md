# SimpleTuner 💹

> ⚠️ **Warning**: The scripts in this repository have the potential to damage your training data. Always maintain backups before proceeding.

**SimpleTuner** is a repository dedicated to a set of experimental scripts designed for training optimization. The project is geared towards simplicity, with a focus on making the code easy to read and understand. This codebase serves as a shared academic exercise, and contributions to its improvement are welcome.

The features implemented will eventually be shared between SD 2.1 and SDXL as much as possible.

* Multi-GPU training is supported, and encouraged
* Aspect bucketing is a "just works" thing; fill a folder of images and let it rip
* SDXL trainer caches the VAE latents and text embeddings to save on VRAM during training
* Full featured fine-tuning support for SDXL and SD 2.x

## Table of Contents

- [Design Philosophy](#design-philosophy)
- [Features](#features)
  - [SDXL Training Features](#sdxl-training-features)
  - [Stable Diffusion 2.0/2.1](#stable-diffusion-20-21)
- [Hardware Requirements](#hardware-requirements)
- [Scripts](#scripts)
- [Toolkit](#toolkit)
- [Setup](#setup)
- [Troubleshooting](#troubleshooting)

## Design Philosophy

- **Simplicity**: Just add captioned images to a directory, and the script handles the rest.
- **Versatility**: Designed to handle a wide range of image quantities - from small datasets to extensive collections.
- **Cutting-Edge Features**: Only incorporates features that have proven efficacy, avoiding the addition of untested options.

## Tutorial

Please fully explore this README before embarking on [the tutorial](/TUTORIAL.md), as it contains vital information that you might need to know first.

## Features

- Precomputed VAE (latents) outputs saved to storage, eliminating the need to invoke the VAE during the forward pass.
- Precomputed captions are run through the text encoder(s) and saved to storage to save on VRAM.
- Trainable on a 40G GPU at lower base resolutions. **Note: SDXL's full U-net is incompatible with 24G GPUs.**
- Optional EMA (Exponential moving average) weight network to counteract model overfitting and improve training stability.
- Support for a variety of image sizes, not limited to 768x768 squares, for improved generalization across aspect ratios.
- Train directly from an S3-compatible storage provider, eliminating the requirement for expensive local storage. (Tested with Cloudflare R2 and Wasabi S3)

### Stable Diffusion 2.0/2.1

Stable Diffusion 2.1 is known for difficulty during fine-tuning, but this doesn't have to be the case. Related features in StableTuner include:

- Training only the text encoder's later layers
- Enforced zero SNR on the terminal timestep instead of offset noise for clearer images.
- The use of EMA (exponential moving average) during training to ensure we do not "fry" the model.

Some of these features exist in other trainers, but EMA seems to be unique here.
## Hardware Requirements

EMA (exponential moving average) weights are a memory-heavy affair, but provide fantastic results at the end of training. Without it, training can still be done, but more care must be taken not to drastically change the model leading to "catastrophic forgetting".

### SDXL

* A100-80G (EMA, large batches)
* A6000 48G (EMA@768px, no EMA@1024px)
* A100 40G (no EMA@1024px, no EMA@768px, EMA@512px)

### Stable Diffusion 2.x

* NVIDIA RTX 3090 or better (24G, no EMA)
* A100-40, A40, or A6000 (EMA)

More optimisation work can be done to bring the memory requirements of SD 2.1 down to about 16G.

## Scripts

* `ubuntu.sh` - This is a basic "installer" that makes it quick to deploy on a Vast.ai instance.
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