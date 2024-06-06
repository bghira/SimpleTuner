# This tutorial is a work-in-progress.

## Introduction

You'll need to set up a Python environment and create an "env" file for SimpleTuner before it can be run.

This document aims to get you set up and running with a basic training environment, including example data to use if you do not currently have any.

## Installation

**SimpleTuner requires Linux or MacOS (Apple Silicon).**

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

## Advanced users: Kohya config conversion

If you are accustomed to training on Kohya, you can convert your Kohya config to a command-line string for SimpleTuner by using `kohya_config.py --config_path=/path/to/kohya/config.json`.

This isn't as ideal as going through the parameters and setting them manually, but it can be a good starting point if you just want to dive right in.

The script prints out many warnings and errors to help you get a better understanding of what you need to change.

## Hardware Requirements

Ensure your hardware meets the requirements for the resolution and batch size you plan to use. High-end GPUs with more than 24G VRAM are generally recommended. For LoRA, 24G is more than enough - you can get by with a 12G or 16G GPU. More is better, but there's a threshold of diminishing returns around 24G for LoRA.

**For full u-net tuning:** Although SimpleTuner has an option to `--fully_unload_text_encoder` and by default will unload the VAE during training, the base SDXL u-net consumes 12.5GB at idle. When the first forward pass runs, a 24G GPU will hit an Out of Memory condition, *even* with 128x128 training data.

This occurs with Adafactor, AdamW8Bit, Prodigy, and D-adaptation due to a bug in PyTorch. Ensure you are using the **latest** 2.1.x release of PyTorch, which allows **full u-net tuning in ~22G of VRAM without DeepSpeed**.

24G GPUs can meaningfully train SDXL, though 40G is the sweet spot for full fine-tune - an 80G GPU is pure heaven.

## Dependencies

Install SimpleTuner as detailed in [INSTALL.md](/INSTALL.md)

## Training data

A publicly-available dataset is available [on huggingface hub](https://huggingface.co/datasets/ptx0/pseudo-camera-10k).

Approximately 10k images are available in this repository with their caption as their filename, ready to be imported for use in SimpleTuner.

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
- A larger batch size requires more training data in each bucket, since each one **must** contain a minimum of that many images - a batch size of 8 means each bucket must have at least 8 images.
- If you can't get a single iteration done with batch size of 1 and resolution of 128x128 on Adafactor or AdamW8Bit, your hardware just won't work.

Which brings up the next point: **you should use as much high quality training data as you can acquire.**

### Selecting images

- JPEG artifacts and blurry images are a no-go. The model **will** pick these up.
- High-res images introduce their own problems - they must be downsampled to fit their aspect bucket, and this inherently damages their quality.
- For high quality photographs, some grainy CMOS sensors in the camera itself end up producing a lot of noise. Too much of this will result in nearly every image your model produces, containing the same sensor noise.
- Same goes for watermarks and "badges", artist signatures. That will all be picked up effortlessly.
- If you're trying to extract frames from a movie to train from, you're going to have a bad time. Compression ruins most films - only the large 40+ GB releases are really going to be useful for improving image clarity.
  - Using 1080p Bluray extractions really helps - 4k isn't absolutely required, but you're going to need to reduce expectations as to what kind of content will actually WORK.
  - Anime content will generally work very well if it's minimally compressed, but live action stuff tends to look blurry.
- Image resolutions optimally should be divisible by 64.
  - This isn't **required**, but is beneficial to follow.
- Square images are not required, though they will work.
  - If you train on ONLY square images or ONLY non-square images, you might not get a very good balance of capabilities in the resulting model.
- Synthetic data works great. This means AI-generated images, from either GAN upscaling or a different model entirely. Using outputs from a different model is called **transfer learning** and can be highly effective.
  - Using ONLY synthetic data can harm the model's ability to generate more realistic details. A decent balance of regularisation images (eg. concepts that aren't your target) will help to maintain broad capabilities.
- Your dataset should be **as varied as possible** to get the highest quality. It should be balanced across different concepts, unless heavily biasing the model is desired.

### Captioning

SimpleTuner provides multiple [captioning](/toolkit/captioning/README.md) scripts that can be used to mass-rename files in a format that is acceptable to SimpleTuner.

Options:

- BLIP3 is currently the best option, as it follows instruction prompts very well and produces prompts comparable to CogVLM with fewer hallucinations.
- T5 Flan and BLIP2 produce mediocre captions; it can be very slow and resource hungry.
- LLaVA produces acceptable captions but misses subtle details.
  - It is better than BLIP, can sometimes read text but invents details and speculates.
  - Follows instruction templates better than CogVLM and BLIP.
- CogVLM produces sterile but accurate captions and requires the most time/resources.
  - It still speculates, especially when given long instruct queries.
  - It does not follow instruct queries very well.


Other tools are available from third-party sources, such as Captionr.

For a caption to be useful by SimpleTuner:

- It could be the image's filename (the default behaviour)
- It could be the contents of a .txt file with the same name as the image (if `--caption_strategy=textfile` is provided)
- (Advanced users) You may compile your dataset metadata into a parquet, json, or jsonl file and [provide it directly to SimpleTuner](/documentation/DATALOADER.md#advanced-techniques)

Longer captions aren't necessarily better for training. Simpler, concise captions work well, but a hybrid dataset mixing short and long captions will cover all bases.

#### Caption Dropout Parameter: CAPTION_DROPOUT_PROBABILITY

Foundational models like Stable Diffusion are built using 10% caption drop-out, meaning the model is shown an "empty" caption instead of the real one, about 10% of the time. This ends up substantially improving the quality of generations when using no negative prompt, especially for prompts that involve subject matter that do not exist in your training data.

Disabling caption dropout can damage the model's ability to generalise to unseen prompts. Conversely, using too much caption dropout will damage the model's ability to adhere to prompts.

A value of 25% seems to provide some additional benefits such as reducing the number of required steps during inference on v-prediction models, but the resulting model will be prone to forgetting.

### Advanced Configuration

For users who are more familiar with model training and wish to tweak settings eg. `MIXED_PRECISION`, enabling offset noise, or setting up zero terminal SNR - detailed explanations can be found in [OPTIONS.md](/OPTIONS.md).

## Publishing checkpoints to Hugging Face Hub

Setting two values inside `sdxl-env.sh` or `sd2x-env.sh` will cause the trainer to automatically push your model up to the Hugging Face Hub upon training completion:

```bash
export PUSH_TO_HUB="true"
export HUB_MODEL_NAME="what-you-will-call-this"
```

Be sure to login before you begin training by executing:

```bash
huggingface-cli login
```

A model card will be automatically generated containing a majority of the relevant training session parameters.

By default, every checkpoint will be uploaded to the Hub. However, if you wish to disable this behaviour to conserve bandwidth or for privacy reasons, you can set the following value in your `sdxl-env.sh`:

```bash
export PUSH_CHECKPOINTS="false"
```

## Monitoring and Logging

If `--report_to=wandb` is passed to the trainer (the default), it will ask on startup whether you wish to register on Weights & Biases to monitor your training run there. While you can always select option **3** or remove `--report_to=...` and disable reporting, it's encouraged to give it a try and watch your loss value drop as your training runs!

### Discord webhook monitoring

SimpleTuner can submit messages to a Discord webhook:

- Startup/training summary
- Periodic status lines indicating the current epoch, step, loss, and EMA decay
- Validations images, as they generate, grouped by prompt (ten at a time)
- Most fatal errors

To configure a Discord webhook, add `--webhook_config=webhook.json` to your env file:

```bash
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --webhook_config=webhook.json"
```

In the SimpleTuner root directory, create the file `webhook.json`:

```json
{
  "webhook_url": "https://path/to/discord/webhook",
  "webhook_type": "discord",
  "message_prefix": "system-name-example",
  "log_level": "critical"
}
```

- `webhook_url`
  - The value obtained from your Discord "Integrations" server settings.
- `webhook_type`
  - Currently, only discord is supported.
- `message_prefix`
  - This will be appended to the front of every message. If unset, it will default to the tracker project and run name.
- `log_level`
  - Values (decreasing level of spamminess, left-to-right): `debug`, `info`, `warning`, `error`, `critical`
  - `debug` is the most information, and `critical` will be limited to important updates.

### Post-Training Steps

#### How do I end training early?

You might not want to train all the way to the end.

At this point, reduce `--max_train_steps` value to one smaller than your current training step to force a pipeline export into your `output_dir`.

#### How do I test the model without wandb (Weights & Biases)?

You can evaluate the model using [the provided evaluation script](/inference.py) or [other options in the inference toolkit](/toolkit/inference/inference_ddpm.py).

If you used `--push_to_hub`, the Huggingface Diffusers SDXL example scripts will be useable with the same model name.

If you require a single 13GiB safetensors file for eg. AUTOMATIC1111's Stable Diffusion WebUI or for uploading to CivitAI, you should make use of the [SDXL checkpoint conversion script](/convert_sdxl_checkpoint.py):

> **Note**: If you're planning to export the resulting pipeline to eg. CivitAI, use the `--save_text_encoder` option to ensure it's copied to the output directory. It's okay if you forget or don't set this option, but it will require manually copying the text encoder.

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
export SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=INFO
```

This can be placed anywhere in the file on its own line. It will bump the verbosity from the default `WARNING` value up to `INFO`. For even more information (God help us) set the log level to `DEBUG`.

A log file named `debug.log` will be written to the SimpleTuner project root directory, containing all log entries from `ERROR` to `DEBUG`.

### Seen images, current epoch, etc

In each model checkpoint directory is a `tracker_state.json` file which contains the current epoch that training was on or the images it has seen so far.

Each dataset will have its own tracking state documents in this directory as well. This contains the step count, number of images seen, and other metadata required to resume completely.


### Example Environment File Explained

Here's a breakdown of what each environment variable does:

#### General Settings

- `DATALOADER_CONFIG`: This file is mandatory, and an example copy can be found in `multidatabackend.json.example` which contains an example for a multi-dataset configuration split between S3 and local data storage.
  - See [this document](/documentation/DATALOADER.md) for more information on configuring the data loader.
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
  - Each dataset entry in `multidatabackend.json` can have its own `instance_prompt` set in lieu of using this main variable.
- `VALIDATION_PROMPT`: The prompt used for validation.
  - Optionally, a user prompt library or the built-in prompt library may be used to generate more than 84 images on each checkpoint across a large number of concepts.
  - See `--user_prompt_library` for more information.

  For DeepFloyd, a page is maintained with specific options to set. Visit [this document](/documentation/DEEPFLOYD.md) for a head start.

#### Data Locations

- `BASE_DIR`, `OUTPUT_DIR`: Directories for the training data, instance data, and output models.
  - `BASE_DIR` - Used for populating other variables, mostly.
  - `OUTPUT_DIR` - Where the model pipeline results are stored during training, and after it completes.

#### Training Parameters

- `MAX_NUM_STEPS`, `NUM_EPOCHS`: Max number of steps or epochs for training.
  - If you use `MAX_NUM_STEPS`, it's recommended to set `NUM_EPOCHS` to `0`.
  - Similarly, if you use `NUM_EPOCHS`, it is recommended to set `MAX_NUM_STEPS` to `0`.
  - This simply signals to the trainer that you explicitly wish to use one or the other.
  - Don't supply `NUM_EPOCHS` and `MAX_NUM_STEPS` values together, it won't let you begin training, to ensure there is no ambiguity about which you expect to take priority.
- `LR_SCHEDULE`, `LR_WARMUP_STEPS`: Learning rate schedule and warmup steps.
  - `LR_SCHEDULE` - stick to `constant`, as it is most likely to be stable and less chaotic. However, `polynomial` and `constant_with_warmup` have potential of moving the model's local minima before settling in and reducing the loss. Experimentation can pay off here, especially using the `cosine` and `sine` schedulers, which offer a unique approach to learning rate scheduling.
- `TRAIN_BATCH_SIZE`: Batch size for training. You want this **as high as you can get it** without running out of VRAM or making your training unnecessarily **slow** (eg. 300-400% increase in training runtime - yikes! 💸)

## Additional Notes

For more details, consult the [INSTALL](/INSTALL.md) and [OPTIONS](/OPTIONS.md) documents or the [DATALOADER](/documentation/DATALOADER.md) information page for specific details on the dataset config file.

### Single-subject fine-tuning (Dreambooth)

See [DREAMBOOTH](/documentation/DREAMBOOTH.md) for a breakdown on how Dreambooth training can be configured in SimpleTuner.

### Mixture-of-Experts split-schedule model training

See [MIXTURE-OF-EXPERTS](/documentation/MIXTURE_OF_EXPERTS.md) for information on how to split training over two models, such that one is responsible for composition and large details, and the other is responsible for finalising and filling in the fine details.