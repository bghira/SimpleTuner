# SimpleTuner 💹

This repository contains a set of experimental scripts that will not be well-supported, should you choose to make use of them.

This code is public merely as a learning resource and so that hopefully I can also learn from anyone else who comes along.

## Why?

Stable Diffusion 2.1 is notoriously difficult to fine-tune. Many of the default scripts are not making the smartest choices, and result in poor-quality outputs.

Some of the problems I've encountered:

* High learning rates damage the OpenCLIP text encoder. Stable Diffusion 1.5's CLIP encoder seems to tolerate much more abuse, possibly because of the way its layers were frozen, and what they were learning for each.

SD 1.5's CLIP has 12 layers while SD 2.1's OpenCLIP has 24. We don't know the details of CLIP's training, but LAION has released details about OpenCLIP, documenting several issues they'd faced with instability while scaling up their training and how they'd resolved it. It's possible that these problems were addressed in a different manner by OpenAI during CLIP's training, or, that they followed an altogether different training schedule.

* Using the same learning rate for the unet and the text encoder. They learn at different rates naturally, and this also impacts the convergence.

* Not using enforced zero SNR on the terminal timestep. This results in a more noisy image. It's possible to train past this with traditional fine-tuning, but not without harming the text encoder's foundational concepts in the process.

To resolve them, the parameters that can be tweaked when training are minimised or eliminated if they were discovered to have an adverse impact. SNR Gamma and Offset noise are not supported by SimpleTuner, because they are not good techniques.

## Scripts

* `training.sh` - some variables are here, but if they are, they're not meant to be tuned.
* `env.sh.example` - These are the training parameters you will want to set up.

* `interrogate.py` - This is useful for labelling datasets using BLIP. Not very accurate, but good enough for a LARGE dataset that's being used for fine-tuning.

* `helpers/broken_images.py` - Scan and remove any images that will not load properly.

Another note here: You might want to make sure it knows your most important concepts. If it doesn't, you can try to fine-tune BLIP using a subset of your data with manually created captions. This generally has a lot of success.

* `inference.py` - Generate validation results from the prompts catalogue (`prompts.py`) using DDIMScheduler.
* `inference_ddpm.py` - Use DDPMScheduler to assemble a checkpoint from a base model configuration and run through validation prompts.
* `inference_karras.py` - Use the Karras sigmas with DPM 2M Karras. Useful for testing what might happen in Automatic1111.
* `tile_shortnames.py` - Tile the outputs from the above scripts into strips.

* `inference_snr_test.py` - Generate a large number of CFG range images, and catalogue the results for tiling.
* `tile_images.py` - Generate large image tiles to compare CFG results for zero SNR training / inference tuning.

## Setup

I'm sorry it's not easier to set this repo up locally. I do that to add a barrier of entry to those unprepared for dealing with Python issues or incompatibilities.

1. Clone the repository and install the dependencies. That's your first barrier.

2. Copy env.sh.example to env.sh. Be sure to fill out the details. Try to change as little as possible.

3. Run the training.sh script, probably by redirecting the output to a log file:

```bash
bash training.sh > /path/to/training-$(date +%s).log 2>&1
```

From here, that's really up to you.