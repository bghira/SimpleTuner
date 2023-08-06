# SimpleTuner 💹

This repository contains a set of experimental scripts that could damage your training data. Keep backups!

This project's code is simple in its implementations. If anything is overly complicated, it is likely a bug.

This code is a shared academic exercise. Please feel free to contribute improvements, or open issue reports.

## Why? (SDXL)

The popular trainers available have complicated code that seems to intentionally make things as difficult to understand.

Alternatively, I'm simply just one who needs things written a bit simpler (and in English)!

The functionality of this script is shared between SD 2.1 and SDXL as much as possible, with room for improvement;

* Aspect bucketing is shared
* Latent caching is currently only done for SDXL
* Prompt embed caching is also only done for SDXL
* Multi-GPU support has been enhanced and fixed

With this script, at 1024x1024 batch size 10, we can nearly saturate a single 80G A100!

At 1024x1024 batch size 4, we can use a 48G A6000 GPU, which reduces the cost of multi-GPU training!

## Why? (Stable Diffusion 2.1)

Stable Diffusion 2.1 is notoriously difficult to fine-tune. Many of the default scripts are not making the smartest choices, and result in poor-quality outputs.

Some of the problems I've encountered in other tools:

* Training OpenCLIP concurrently to the U-net. They must be trained in sequence, with the text encoder being tuned first.

* Not using enforced zero SNR on the terminal timestep, using offset noise instead. This results in a more noisy image.

* Training on only square, 768x768 images, that will result in the model losing the ability to (or at the very least, simply not improving) super-resolution its output into other aspect ratios.

* Overfitting the unet on textures, results in "burning". So far, I've not worked around this much other than mix-matching text encoder and unet checkpoints.

Additionally, if something does not provide value to the training process by default, it is simply not included.

## Scripts

* `training.sh` - some variables are here, but if they are, they're not meant to be tuned.
* `sdxl-env.sh.example` - These are the SDXL training parameters, you should copy to `sdxl-env.sh`
* `sd21-env.sh.example` - These are the training parameters, copy to `env.sh`

* `interrogate.py` - This is useful for labelling datasets using BLIP. Not very accurate, but good enough for a LARGE dataset that's being used for fine-tuning.

* `analyze_laion_data.py` - After downloading a lot of LAION's data, you can use this to throw a lot of it away.
* `analyze_aspect_ratios_json.py` - Use the output from `analyze_laion_data.py` to nuke images that do not fit our aspect goals.
* `helpers/broken_images.py` - Scan and remove any images that will not load properly.

Another note here: You might want to make sure it knows your most important concepts. If it doesn't, you can try to fine-tune BLIP using a subset of your data with manually created captions. This generally has a lot of success.

* `inference.py` - Generate validation results from the prompts catalogue (`prompts.py`) using DDIMScheduler.
* `inference_ddpm.py` - Use DDPMScheduler to assemble a checkpoint from a base model configuration and run through validation prompts.
* `inference_karras.py` - Use the Karras sigmas with DPM 2M Karras. Useful for testing what might happen in Automatic1111.
* `tile_shortnames.py` - Tile the outputs from the above scripts into strips.

* `inference_snr_test.py` - Generate a large number of CFG range images, and catalogue the results for tiling.
* `tile_images.py` - Generate large image tiles to compare CFG results for zero SNR training / inference tuning.

## Setup

1. Clone the repository and install the dependencies:

```bash
git clone https://github.com/bghira/SimpleTuner --branch release
python -m venv .venv
pip3 install -U poetry pip
poetry install
```

2. For SD2.1, copy `sd21-env.sh.example` to `env.sh` - be sure to fill out the details. Try to change as little as possible.

For SDXL, copy `sdxl-env.sh.example` to `sdxl-env.sh` and then fill in the details.

For both training scripts, any missing values from your user config will fallback to the defaults.

3. If you are using `--report_to='wandb'` (the default), the following will help you report your statistics:

```bash
wandb login
```

Follow the instructions that are printed, to locate your API key and configure it.

Once that is done, any of your training sessions and validation data will be available on Weights & Biases.

4. For SD2.1, run the `training.sh` script, probably by redirecting the output to a log file:

```bash
bash training.sh > /path/to/training-$(date +%s).log 2>&1
```

For SDXL, run the `train_sdxl.sh` script, redirecting outputs to the log file:

```bash
bash train_sdxl.sh > /path/to/training-$(date +%s).log 2>&1
```

From here, that's really up to you.


## Known issues

* For very poorly distributed aspect buckets, some problems with uneven training are being worked on.
* Some hardcoded values need to be adjusted/removed - images under 860x860 are discarded.
* SDXL latent caching is currently non-deterministic, and will be adjusted for a better hashing method soon.