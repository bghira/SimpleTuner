# This tutorial is a work-in-progress.

## Introduction

You'll need to set up a Python environment and create an "env" file for SimpleTuner before it can be run.

This document aims to get you set up and running with a basic training environment, including example data to use if you do not currently have any.

## Installation

**SimpleTuner requires Linux.**

These steps can be followed to the best of your abilities here. If you face any difficulties, please [start a discussion](https://github.com/bghira/SimpleTuner/discussions/new/choose) on the forum here on GitHub.

1. Clone the SimpleTuner repository:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner
```

2. Install the required packages as per [INSTALL.md](/INSTALL.md).
3. Follow the below section, [Training data](#training-data) to produce a set of valid training data, or to obtain example data.
4. Modify the `sdxl-env.sh` file in the `SimpleTuner/` project root directory. These contain all of the settings that the trainer will use to process your data.
  - Use the instructions in the below section [Example Environment File Explained](#example-environment-file-explained) to modify these values.
5. Run the [train_sdxl.py](/train_sdxl.py) script.

## Hardware Requirements

Ensure your hardware meets the requirements for the resolution and batch size you plan to use. High-end GPUs with more than 24G VRAM are generally recommended.

Although SimpleTuner has an option to `--fully_unload_text_encoder` and by default will unload the VAE during training, the base SDXL u-net consumes 12.5GB at idle. When the first forward pass runs, a 24G GPU will hit an Out of Memory condition, *even* with 128x128 training data.

This occurs with Adafactor, AdamW8Bit, Prodigy, and D-adaptation.

40G GPUs can meaningfully train SDXL.

## Dependencies

Install SimpleTuner as detailed in [INSTALL.md](/INSTALL.md)

## Training data

A publicly-available dataset is available [on huggingface hub](https://huggingface.co/datasets/ptx0/mj51-data).

Approximately 162GB of images are available in the `split_train` directory, although this format is not required by SimpleTuner.

You can simply create a single folder full of jumbled-up images, or they can be neatly organised into subdirectories.

**Here are some important guidelines:**

### Training batch size

Your maximum batch size is a function of your available VRAM and image resolution:

```
vram use = batch size * resolution + base_requirements
```

To reduce VRAM use, you can reduce batch size or resolution, but the base requirements will always bite us in the ass. SDXL is a **huge** model.

To summarise:

- You want as high of a batch size as you can tolerate.
- The larger you set `RESOLUTION`, the more VRAM is used, and the lower your batch size can be.
- A larger batch size requires more training data in each bucket, since each one **must** contain a minimum of that many images.
- If you can't get a single iteration done with batch size of 1 and resolution of 128x128 on Adafactor or AdamW8Bit, your hardware just won't work.

Which brings up the next point: **you should use as much high quality training data as you can acquire.**

### Selecting images

- JPEG artifacts and blurry images are a no-go. The model **will** pick these up.
- Same goes for watermarks and "badges", artist signatures. That will all be picked up effortlessly.
- If you're trying to extract frames from a movie to train from, you're going to have a bad time. Compression ruins most films - only the large 40+ GB releases are really going to be useful for improving image clarity.
- Image resolutions optimally should be divisible by 64.
  - This isn't **required**, but is beneficial to follow.
- Square images are not required, though they will work.
  - The model might fail to generalise across aspect ratios if they are not seen during training. This means if you train on only square images, you might not get a very good widescreen effect when you are done.
- The trainer will resize images so that the smaller side is equal to the value of `RESOLUTION`, while maintaining the aspect ratio.
  - If your images all hover around a certain resolution, eg. `512x768`, `1280x720` and `640x480`, you might then set `RESOLUTION=640`, which would result in upscaling a minimal number of images during training time.
  - If your images are all above a given base resolution, the trainer will downsample them to your base `RESOLUTION`
- Your dataset should be **as varied as possible** to get the highest quality.
- Synthetic data works great. This means AI-generated images, from either GAN upscaling or a different model entirely. Using outputs from a different model is called **transfer learning** and can be highly effective.

### Captioning

SimpleTuner provides a [captioning](/toolkit/captioning/README.md) script that can be used to mass-rename files in a format that is acceptable to SimpleTuner.

Currently, it uses T5 Flan and BLIP2 to produce high quality captions, though it can be very slow and resource hungry.

Other tools are available from third-party sources, such as Captionr.

For a caption to be useful by SimpleTuner:

- It could be the image's filename (the default behaviour)
- It could be the contents of a .txt file with the same name as the image (if `--caption_strategy=textfile` is provided)

Longer captions aren't necessarily better for training. Simpler, concise captions work best!

#### Caption Dropout Parameter: CAPTION_DROPOUT_PROBABILITY

Foundational models like Stable Diffusion are built using 10% caption drop-out, meaning the model is shown an "empty" caption instead of the real one, about 10% of the time. This ends up substantially improving the quality of generations, especially for prompts that involve subject matter that do not exist in your training data.

In other words, caption drop-out will allow you to introduce a style or concept more broadly across the model. You might not want to use this at all if you really want to restrict your changes to just the captions you show the model during training.

### Advanced Configuration

For users who are more familiar with model training and wish to tweak settings eg. `MIXED_PRECISION`, enabling offset noise, or setting up zero terminal SNR - detailed explanations can be found in [OPTIONS.md](/OPTIONS.md).

## Monitoring and Logging

If `--report_to=wandb` is passed to the trainer (the default), it will ask on startup whether you wish to register on Weights & Biases to monitor your training run there. While you can always select option **3** or remove `--report_to=...` and disable reporting, it's encouraged to give it a try and watch your loss value drop as your training runs!

### Post-Training Steps

You might not want to train all the way to the end once you realise your progress has been "good enough". At this point, it would be best to reduce `NUM_EPOCHS` to `1` and start another training run. This will in fact, not do any more training, but will simply export the model into the pipeline directory - assuming a single epoch has been hit yet. **This may not be the case for very large datasets**. You can switch to a small folder of files to force it to export.

Once the training is complete, you can evaluate the model using [the provided evaluation script](/inference.py) or [other options in the inference toolkit](/toolkit/inference/inference_ddpm.py).

If you require a single 13GiB safetensors file for eg. AUTOMATIC1111's Stable Diffusion WebUI or for uploading to CivitAI, you should make use of the [SDXL checkpoint conversion script](/convert_sdxl_checkpoint.py):

```bash
python3 convert_sdxl_checkpoint.py --model_path="/path/to/SimpleTuner/simpletuner-results/pipeline" --checkpoint_path=/path/to/your/output.safetensors --half --use_safetensors
```

Thank you to watusi on Discord for providing these instructions and requesting this addition.

## Model integration / usage

For using the model in your own projects, refer to the [Diffusers project](https://github.com/huggingface/diffusers).

## Debugging

For extra information when running SimpleTuner you can add the following to your env file:

```bash
export SIMPLETUNER_LOG_LEVEL=INFO
```

This can be placed anywhere in the file on its own line. It will bump the verbosity from the default `WARNING` value up to `INFO`. For even more information (God help us) set the log level to `DEBUG`.

At this point, you may wish to create a log file of your training run:

```bash
bash train_sdxl.sh > train.log 2>&1
```

This command will capture the output of your training run into `train.log`, located in the **SimpleTuner** project directory.

### Seen images, current epoch, etc

In each model checkpoint directory is a `tracker_state.json` file which contains the current epoch that training was on or the images it has seen so far.


### Example Environment File Explained

Here's a breakdown of what each environment variable does:

#### General Settings

- `DATALOADER_CONFIG`: This file is mandatory, and an example copy can be found in `multidatabackend.json.example` which contains an example for a multi-dataset configuration split between S3 and local data storage.
  - One or more datasets can be configured, but it's not necessary to use multiple.
  - Some config options that have an equivalent commandline option name can be omitted, in favour of the global option
  - Some config options are mandatory, but errors will emit for those on startup. Feel free to experiment.
  - Each dataset can have its own crop and resolution config.
- `TRAINING_SEED`: You may set a numeric value here and it will make your training reproducible to that seed across all other given settings.
  - You may wish to set this to -1 so that your training is absolutely random, which prevents overfitting to a given seed.
- `RESUME_CHECKPOINT`: Specifies which checkpoint to resume from. "latest" will pick the most recent one.
  - Do not set this value to a full pipeline. It will not work. To resume training a pipeline, use `MODEL_NAME` and provide an `/absolute/path`
- `CHECKPOINTING_STEPS`: Frequency of checkpointing during training.
  - Too many checkpoints created can slow down training. However, it might be necessary on providers that could unexpectedly shut down or restart your environment.
- `CHECKPOINTING_LIMIT`: Maximum number of checkpoints to keep.
  - Using a higher value here will make it safer to leave training running attended for longer, at the cost of higher disk consumption - MUCH higher, in the case of SDXL.
- `LEARNING_RATE`: The initial learning rate for the model.
  - A value of `4e-7` may be considered the lowest effective learning rate when using EMA. A value of `1e-5` is much too high.
  - Somewhere in the range of `4e-7` to `4e-6` most likely lies your sweet spot.
  - You want the model to explore new territory (higher learning rate), but not so boldly that it explodes in catastrophic forgetting or worse.
  - If your learning rate is too low, it's possible to have some improvements in the beginning that then plateau. However, it can help prevent overfitting. Your mileage may vary.

#### Model and Data Settings

- `MODEL_NAME`: Specifies the pretrained model to use. Can be a HuggingFace Hub model or a local path. Either method requires a full Diffusers-style layout be available.
  - You can find some [here](https://huggingface.co/stabilityai) from Stability AI.
- `TRACKER_PROJECT_NAME` and `TRACKER_RUN_NAME`: Names for the tracking project on Weights and Biases. Currently, run names are non-functional.
- `INSTANCE_PROMPT`: Optional prompt to append to each caption. This can be useful if you want to add a **trigger keyword** for your model's style to associate with.
  - Make sure the instance prompt you use is similar to your data, or you could actually end up doing harm to the model.
- `VALIDATION_PROMPT`: The prompt used for validation.
  - Optionally, a user prompt library or the built-in prompt library may be used to generate more than 84 images on each checkpoint across a large number of concepts.

#### Data Locations

- `BASE_DIR`, `INSTANCE_DIR`, `OUTPUT_DIR`: Directories for the training data, instance data, and output models.
  - `BASE_DIR` - Used for populating other variables, mostly.
  - `INSTANCE_DIR` - Where your actual training data is. This can be anywhere, it does not need to be underneath `BASE_DIR`.
  - `OUTPUT_DIR` - Where the model pipeline results are stored during training, and after it completes.
- `STATE_PATH`, `SEEN_STATE_PATH`: Paths for the training state and seen images.
  - These can effectively be ignored, unless you want to make use of this data for integrations in eg. a Discord bot and need it placed in a particular location.

#### Training Parameters

- `MAX_NUM_STEPS`, `NUM_EPOCHS`: Max number of steps or epochs for training.
  - If you use `MAX_NUM_STEPS`, it's recommended to set `NUM_EPOCHS` to `0`.
  - Similarly, if you use `NUM_EPOCHS`, it is recommended to set `MAX_NUM_STEPS` to `0`.
  - This simply signals to the trainer that you explicitly wish to use one or the other.
  - If you supply `NUM_EPOCHS` and `MAX_NUM_STEPS` together, the training will stop running at whichever happens first.
- `LR_SCHEDULE`, `LR_WARMUP_STEPS`: Learning rate schedule and warmup steps.
  - `LR_SCHEDULE` - stick to `constant`, as it is most likely to be stable and less chaotic. However, `polynomial` and `constant_with_warmup` have potential of moving the model's local minima before settling in and reducing the loss. Experimentation can pay off here.
- `TRAIN_BATCH_SIZE`: Batch size for training. You want this **as high as you can get it** without running out of VRAM or making your training unnecessarily **slow** (eg. 300-400% increase in training runtime - yikes! 💸)

## Additional Notes

For more details, consult the [INSTALL](/INSTALL.md) and [OPTIONS](/OPTIONS.md) documents.