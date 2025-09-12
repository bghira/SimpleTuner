# This tutorial is a work-in-progress.

## Introduction

For a more quick and to-the-point setup guide, see the [QUICKSTART](/documentation/QUICKSTART.md) document.

You'll need to set up a Python environment and create an "env" file for SimpleTuner before it can be run.

This document aims to get you set up and running with a basic training environment, including example data to use if you do not currently have any.

## Installation

**SimpleTuner requires Linux or MacOS (Apple Silicon).**

These steps can be followed to the best of your abilities here. If you face any difficulties, please [start a discussion](https://github.com/bghira/SimpleTuner/discussions/new/choose) on the forum here on GitHub.

1. Install the required packages as per [INSTALL.md](/documentation/INSTALL.md).
2. Follow the below section, [Training data](#training-data) to produce a set of valid training data, or to obtain example data.
3. Copy the `config/config.json.example` file in the `SimpleTuner/` project root directory to `config/config.json` and fill it with your configuration options - use [DATALOADER](/documentation//DATALOADER.md) as a guide for this.
  - Use `configure.py` instead if you would prefer an interactive configurator.
4. Run the [train.sh](/train.sh) script.

> ⚠️ For users located in countries where Hugging Face Hub is not readily accessible, you should add `HF_ENDPOINT=https://hf-mirror.com` to your `~/.bashrc` or `~/.zshrc` depending on which `$SHELL` your system uses.


## Hardware Requirements

Ensure your hardware meets the requirements for the resolution and batch size you plan to use. High-end GPUs with more than 24G VRAM are generally recommended. For LoRA, 24G is more than enough - you can get by with a 12G or 16G GPU. More is better, but there's a threshold of diminishing returns around 24G for LoRAs on smaller models (eg. not Flux)

See the main [README](/README.md) for more up-to-date hardware requirement information.

## Dependencies

Install SimpleTuner as detailed in [INSTALL.md](/documentation/INSTALL.md)

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
- A larger batch size should have more training data in each bucket, since a batch size of 8 means a bucket of 1 image will be seen 8 times in one shot - no good!
- If you can't get a single iteration done with batch size of 1 and resolution of 128x128 on the Lion optimiser, your hardware just won't work.

Which brings up the next point: **you should use as much high quality training data as you can acquire.**

### Selecting images

- JPEG artifacts and blurry images are a no-go. The model **will** pick these up, especially newer ones like Flux or PixArt.
- For high quality photographs, some grainy CMOS sensors in the camera itself end up producing a lot of noise.
  - Too much of this will result in nearly every image your model produces, containing the same sensor noise.
- Same goes for watermarks and "badges", artist signatures. That will all be picked up effortlessly.
- If you're trying to extract frames from a movie to train from, you're going to have a bad time. Compression ruins most films - only the large 40+ GB releases are really going to be useful for improving image clarity.
  - Using 1080p Bluray extractions really helps - 4k isn't absolutely required, but you're going to need to reduce expectations as to what kind of content will actually WORK.
  - Anime content will generally work very well if it's minimally compressed, but live action stuff tends to look blurry.
  - Try and locate frame stills from the production company instead, eg. on iMDB.
- Image resolutions optimally should be divisible by 64.
  - This isn't **required**, but is beneficial to follow, as it will allow the trainer to reuse your original images without resizing or cropping.
- Square images are not required, though they will work.
  - If you train on ONLY square images or ONLY non-square images, you might not get a very good balance of capabilities in the resulting model.
  - If you train on ONLY aspect bucketing, your resulting model will heavily bias these buckets for each type of content.
- Synthetic data works great. This means AI-generated images or captions. Using outputs from a different model is called **transfer learning** and can be highly effective.
  - Using ONLY synthetic data can harm the model's ability to generate more realistic details. A decent balance of regularisation images (eg. concepts that aren't your target) will help to maintain broad capabilities.
- Your dataset should be **as varied as possible** to get the highest quality. It should be balanced across different concepts, unless heavily biasing the model is desired.

### Captioning

SimpleTuner provides multiple [captioning](/scripts/toolkit/README.md) scripts that can be used to mass-rename files in a format that is acceptable to SimpleTuner.

Options:

- InternVL2 is the best option - it is very large however, and will be slow. This is best for smaller sets.
- Florence2 is likely the fastest and lightest weight, but some people really take a disliking to its outputs.
- BLIP3 is currently the best lightweight model that follows instruction prompts very well and produces prompts comparable to CogVLM with fewer hallucinations.
- T5 Flan and BLIP2 produce mediocre captions; it can be very slow and resource hungry.
- LLaVA produces acceptable captions but misses subtle details.
  - It is better than BLIP, can sometimes read text but invents details and speculates.
  - Follows instruction templates better than CogVLM and BLIP.
- CogVLM produces sterile but accurate captions and required the most time/resources until InternVL2 was integrated.
  - It still speculates, especially when given long instruct queries.
  - It does not follow instruct queries very well.

Other tools are available from third-party sources, such as Captionr.

For a caption to be useful by SimpleTuner:

- It could be the image's filename (the default behaviour)
- It could be the contents of a .txt file with the same name as the image (if `--caption_strategy=textfile` is provided)
- It could be directly in a jsonl table
- You could have a CSV file of URLs with a caption column
- (Advanced users) You may compile your dataset metadata into a parquet, json, or jsonl file and [provide it directly to SimpleTuner](/documentation/DATALOADER.md#advanced-techniques)

Longer captions aren't necessarily better for training. Simpler, concise captions work well, but a hybrid dataset mixing short and long captions will cover all bases.

#### Caption Dropout Parameter: CAPTION_DROPOUT_PROBABILITY

Foundational models like Stable Diffusion are built using 10% caption drop-out, meaning the model is shown an "empty" caption instead of the real one, about 10% of the time. This ends up substantially improving the quality of generations when using no negative prompt, especially for prompts that involve subject matter that do not exist in your training data.

Disabling caption dropout can damage the model's general quality. Conversely, using too much caption dropout will damage the model's ability to adhere to prompts.

A value of 25% seems to provide some additional benefits such as reducing the number of required steps during inference on v-prediction models, but the resulting model will be prone to forgetting.

Flux has [its own series of considerations](/documentation//quickstart/FLUX.md) and should be investigated before beginning training.

### Advanced Configuration

For users who are more familiar with model training and wish to tweak settings eg. `MIXED_PRECISION`, enabling offset noise, or setting up zero terminal SNR - detailed explanations can be found in [OPTIONS.md](/documentation/OPTIONS.md).

## Publishing checkpoints to Hugging Face Hub

Setting two values inside `config/config.json` will cause the trainer to automatically push your model up to the Hugging Face Hub upon training completion:

```bash
"push_to_hub": true,
"hub_model_name": "what-you-will-call-this",
```

Be sure to login before you begin training by executing:

```bash
huggingface-cli login
```

A model card will be automatically generated containing a majority of the relevant training session parameters.

By default, every checkpoint will be uploaded to the Hub. However, if you wish to disable this behaviour to conserve bandwidth or for privacy reasons, you can set the following value in `config/config.json`:

```bash
"push_checkpoints_to_hub": false,
```

## Monitoring and Logging

If `--report_to=wandb` is passed to the trainer (the default), it will ask on startup whether you wish to register on Weights & Biases to monitor your training run there. While you can always select option **3** or remove `--report_to=...` and disable reporting, it's encouraged to give it a try and watch your loss value drop as your training runs!

### Discord webhook monitoring

SimpleTuner can submit messages to a Discord webhook:

- Startup/training summary
- Periodic status lines indicating the current epoch, step, loss, and EMA decay
- Validations images, as they generate, grouped by prompt (ten at a time)
- Most fatal errors

To configure a Discord webhook, add `--webhook_config=webhook.json` to your config file:

```bash
"webhook_config": "webhook.json",
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

You can evaluate the model using [the provided evaluation script](/simpletuner/inference.py) or [other options in the inference toolkit](/scripts/toolkit/inference/inference_ddpm.py).

If you used `--push_to_hub`, the Huggingface Diffusers SDXL example scripts will be useable with the same model name.

If you require a single 13GiB safetensors file for eg. AUTOMATIC1111's Stable Diffusion WebUI or for uploading to CivitAI, you should make use of the [SDXL checkpoint conversion script](/scripts/convert_sdxl_checkpoint.py):

> **Note**: If you're planning to export the resulting pipeline to eg. CivitAI, use the `--save_text_encoder` option to ensure it's copied to the output directory. It's okay if you forget or don't set this option, but it will require manually copying the text encoder.

```bash
python3 scripts/convert_sdxl_checkpoint.py --model_path="/path/to/SimpleTuner/simpletuner-results/pipeline" --checkpoint_path=/path/to/your/output.safetensors --half --use_safetensors
```

Thank you to watusi on Discord for providing these instructions and requesting this addition.

## Model integration / usage

For using the model in your own projects, refer to the [Diffusers project](https://github.com/huggingface/diffusers).

## Debugging

For extra information when running SimpleTuner you can add the following to `config/config.env`:

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

- `model_family`: Set this to the model arch you are training; kolors, sdxl, pixart_sigma, flux, sd3, legacy
- `data_backend_config`: This file is mandatory, and an example copy can be found in `multidatabackend.json.example` which contains an example for a multi-dataset configuration split between S3 and local data storage.
  - See [this document](/documentation/DATALOADER.md) for more information on configuring the data loader.
  - One or more datasets can be configured, but it's not necessary to use multiple.
  - Some config options that have an equivalent commandline option name can be omitted, in favour of the global option
  - Some config options are mandatory, but errors will emit for those on startup. Feel free to experiment.
  - Each dataset can have its own crop and resolution config.
- `seed`: You may set a numeric value here and it will make your training reproducible to that seed across all other given settings.
  - You may wish to set this to -1 so that your training is absolutely random, which prevents overfitting to a given seed.
- `resume_from_checkpoint`: Specifies which checkpoint to resume from. "latest" will pick the most recent one.
  - Do not set this value to a full pipeline. It will not work. To resume training a pipeline, use `pretrained_model_name_or_path` and provide an `/absolute/path`
- `checkpointing_steps`: Frequency of checkpointing during training.
  - Too many checkpoints created can slow down training. However, it might be necessary on providers that could unexpectedly shut down or restart your environment.
- `checkpoints_total_limit`: Maximum number of checkpoints to keep.
  - Using a higher value here will make it safer to leave training running attended for longer, at the cost of higher disk consumption - MUCH higher, in the case of SDXL.
- `learning_rate`: The initial learning rate for the model.
  - A value of `4e-7` may be considered the lowest effective learning rate when using EMA. A value of `1e-5` is much too high.
  - Somewhere in the range of `4e-7` to `4e-6` most likely lies your sweet spot.
  - You want the model to explore new territory (higher learning rate), but not so boldly that it explodes in catastrophic forgetting or worse.
  - If your learning rate is too low, it's possible to have some improvements in the beginning that then plateau. However, it can help prevent overfitting. Your mileage may vary.

#### Model and Data Settings

- `pretrained_model_name_or_path`: Specifies the pretrained model to use. Can be a HuggingFace Hub model or a local path. Either method requires a full Diffusers-style layout be available.
  - You can find some [here](https://huggingface.co/stabilityai) from Stability AI.
- `tracker_project_name` and `tracker_run_name`: Names for the tracking project on Weights and Biases. Currently, run names are non-functional.
- `instance_prompt`: Optional prompt to append to each caption. This can be useful if you want to add a **trigger keyword** for your model's style to associate with.
  - Make sure the instance prompt you use is similar to your data, or you could actually end up doing harm to the model.
  - Each dataset entry in `multidatabackend.json` can have its own `instance_prompt` set in lieu of using this main variable.
- `validation_prompt`: The prompt used for validation.

  - Optionally, a user prompt library or the built-in prompt library may be used to generate more than 84 images on each checkpoint across a large number of concepts.
  - See `--user_prompt_library` for more information.

  For DeepFloyd, a page is maintained with specific options to set. Visit [this document](/documentation/DEEPFLOYD.md) for a head start.

#### Data Locations

- `output_dir` - Where the model pipeline results are stored during training, and after it completes.

#### Training Parameters

- `max_train_steps`, `num_train_epochs`: Max number of steps or epochs for training.
  - If you use `max_train_steps`, it's recommended to set `num_train_epochs` to `0`.
  - Similarly, if you use `num_train_epochs`, it is recommended to set `max_train_steps` to `0`.
  - This simply signals to the trainer that you explicitly wish to use one or the other.
  - Don't supply `num_train_epochs` and `max_train_steps` values together, it won't let you begin training, to ensure there is no ambiguity about which you expect to take priority.
- `lr_scheduler`, `lr_warmup_steps`: Learning rate schedule and warmup steps.
  - `lr_scheduler` - stick to `constant`, as it is most likely to be stable and less chaotic. However, `polynomial` and `constant_with_warmup` have potential of moving the model's local minima before settling in and reducing the loss. Experimentation can pay off here, especially using the `cosine` and `sine` schedulers, which offer a unique approach to learning rate scheduling.
- `train_batch_size`: Batch size for training. You want this **as high as you can get it** without running out of VRAM or making your training unnecessarily **slow** (eg. 300-400% increase in training runtime - yikes! 💸)

## Additional Notes

For more details, consult the [INSTALL](/documentation/INSTALL.md) and [OPTIONS](/documentation/OPTIONS.md) documents or the [DATALOADER](/documentation/DATALOADER.md) information page for specific details on the dataset config file.

### Single-subject fine-tuning (Dreambooth)

See [DREAMBOOTH](/documentation/DREAMBOOTH.md) for a breakdown on how Dreambooth training can be configured in SimpleTuner.

### Mixture-of-Experts split-schedule model training

See [MIXTURE-OF-EXPERTS](/documentation/MIXTURE_OF_EXPERTS.md) for information on how to split training over two models, such that one is responsible for composition and large details, and the other is responsible for finalising and filling in the fine details.

### Quantised model training

Tested on Apple and NVIDIA systems, Hugging Face Optimum-Quanto can be used to reduce the precision and VRAM requirements, training even Flux.1 on just 20GB or less.

Inside your SimpleTuner venv:

```bash
pip install optimum-quanto
```

```bash
# Basically, any optimiser should work here.
"optimizer": "optimi-stableadamw",

# choices: int8-quanto, int4-quanto, int2-quanto, fp8-quanto
# int8-quanto was tested with a single subject dreambooth LoRA.
# fp8-quanto does not work on Apple systems. you must use int levels.
# int2-quanto is pretty extreme and gets the whole rank-1 LoRA down to about 13.9GB VRAM.
# may the gods have mercy on your soul, should you push things Too Far.
"base_model_precision": "int8-quanto",

# Maybe you want the text encoders to remain full precision so your text embeds are cake.
# We unload the text encoders before training, so, that's not an issue during training time - only during pre-caching.
# Alternatively, you can go ham on quantisation here and run them in int4 or int8 mode, because no one can stop you.
"--text_encoder_1_precision": "no_change",
"--text_encoder_2_precision": "no_change",
"--text_encoder_3_precision": "no_change",
```
