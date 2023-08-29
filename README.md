# SimpleTuner 💹

This repository contains a set of experimental scripts that could damage your training data. Keep backups!

This project's code intended to be simple and easy to read. Parts of it are difficult to follow, though it's hoped they'll improve over time.

This code is a shared academic exercise. Please feel free to contribute improvements, or open issue reports.

The features implemented will eventually be shared between SD 2.1 and SDXL as much as possible.

* Multi-GPU support is in a minimal implementation, and more help is wanted there.
* Aspect bucketing is shared
* Legacy trainer does not implement precomputed embeds/latents
* Currently, the legacy trainer is somewhat neglected. The last release pre-SDXL support should be used for SD 2.1.

## Tutorial

Please fully explore this README before embarking on [the tutorial](/TUTORIAL.md), as it contains vital information that you might need to know first.

## General design philosophy

* Just throw captioned images into a dir and the script does the rest.
* The more images, the merrier - though small datasets are supported, too.
* Not supporting cutting edge features just for the sake of it - they must be proven first.

## SDXL Training Features

* VAE (latents) outputs are precomputed before training and saved to storage, so that we do not need to invoke the VAE during the forward pass.
* Since SDXL has two text encoders, we precompute all of the captions into embeds and then store those as well.
* **Train on a 40G GPU** when using lower base resolutions. Sorry, but it's just not doable to train SDXL's full U-net on 24G, even with Adafactor.
* EMA (Exponential moving average) weight network as an optional way to reduce model over-cooking.

## Stable Diffusion 2.0 / 2.1

Stable Diffusion 2.1 is notoriously difficult to fine-tune. Many of the default scripts are not making the smartest choices, and result in poor-quality outputs:

* Training OpenCLIP concurrently to the U-net. They must be trained in sequence, with the text encoder being tuned first.
* Not using enforced zero SNR on the terminal timestep, using offset noise instead. This results in a more noisy image.
* Training on only square, 768x768 images, that will result in the model losing the ability to (or at the very least, simply not improving) generalise across aspect ratios.

## Hardware Requirements

All testing of this script has been done using:

* A100-80G
* A6000 48G
* 4090 24G

Despite optimisations, SDXL training **will not work on a 24G GPU**, though SD 2.1 training works fantastically well there.

### SDXL 1.0

At 1024x1024 batch size 10, we can nearly saturate a single 80G A100's entire VRAM pool!

At 1024x1024 batch size 4, we can begin to make use of a 48G A6000 GPU, which substantially reduces the cost of multi-GPU training!

With a resolution reduction down to 768 pixels, you can shift requirements down to an A100-40G.

For further reductions, when training at a resolution of `256x256` the model can still generalise training data quite well, in addition to supporting a much higher batch size around 15 if the VRAM is present.

### Stable Diffusion 2.x

Generally, a batch size of 4-8 for aspect bucketed data at 768px base was achievable within 24G of VRAM.

On an A100-80G, a batch size of 15 could be reached with nearly all of the VRAM in use

For 1024px training, the VRAM requirement goes up substantially, but it is still doable in roughly an equivalent footprint to an _optimised_ SDXL setup.

Optimizations from the SDXL trainer could be ported to the legacy trainer (text embed cache, precomputed latents) to bring this down, substantially, and make 1024px training more viable on consumer kit.

## Scripts

* `ubuntu.sh` - This is a basic "installer" that makes it quick to deploy on a Vast.ai instance.
* `train_sdxl.sh` - This is where the magic happens.
* `training.sh` - This is the legacy Stable Diffusion 1.x / 2.x trainer. The last stable version was before SDXL support was introduced. 😞
* `sdxl-env.sh.example` - These are the SDXL training parameters, you should copy to `sdxl-env.sh`
* `sd21-env.sh.example` - These are the training parameters, copy to `env.sh`

## Toolkit

For information on the associated toolkit distributed with SimpleTuner, see [this page](/toolkit/README.md).

## Setup

For setup information, see the [install documentation](/INSTALL.md).

## Troubleshooting

* To enable debug logs (caution, this is extremely noisy) add `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your env file.

For a web version of the options available for the SDXL trainer, see [this document](/OPTIONS.md)