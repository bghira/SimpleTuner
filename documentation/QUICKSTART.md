# Quickstart Guide

**Note**: This tutorial is very basic, step-by-step guide to get a basic training run going without any real explanations for what you are configuring, or what any of the options will do.

For extensive information on the configuration process, see the [tutorial](/TUTORIAL.md), [dataloader configuration guide](/documentation/DATALOADER.md), and the [options breakdown](/OPTIONS.md) pages.

## Stable Diffusion 3

### Configuration files

#### Environment file

Place the following in `SimpleTuner/sdxl-env.sh`

```bash
#!/bin/bash

# If you're adventurous, you can set this to 'full', but it is VERY VRAM-hungry.
export MODEL_TYPE="lora"

export STABLE_DIFFUSION_3=true
export CONTROLNET=false
export USE_DORA=false
export USE_BITFIT=false

# How often to checkpoint. Depending on your learning rate, you may wish to change this.
# For the default settings with 10 gradient accumulations, more frequent checkpoints might be preferable at first.
export CHECKPOINTING_STEPS=200
export LEARNING_RATE=4e-5 #@param {type:"number"}
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"

# Make DEBUG_EXTRA_ARGS empty to disable wandb.
# You'll want to use 'wandb login' on commandline to set this up.
export DEBUG_EXTRA_ARGS="--report_to=wandb"
export TRACKER_PROJECT_NAME="simpletuner-project-name"
export CURRENT_TIME
CURRENT_TIME=$(date +%s)
export TRACKER_RUN_NAME="${CURRENT_TIME}"
export VALIDATION_PROMPT="a studio portrait photograph of a teddy bear holding a sign that reads, 'Hello World'"
export VALIDATION_GUIDANCE=5.5
export VALIDATION_STEPS=100


# Location of training data.
export BASE_DIR="/path/to/outputs"
export DATALOADER_CONFIG="multidatabackend.json"
export INSTANCE_DIR="${BASE_DIR}/datasets/training_data"
export OUTPUT_DIR="${BASE_DIR}/models"

export PUSH_TO_HUB="false"
export PUSH_CHECKPOINTS="false"
# This is how many checkpoints we will keep. Two is safe, but three is safer.
export CHECKPOINTING_LIMIT=10

# By default, images will be resized so their SMALLER EDGE is 1024 pixels, maintaining aspect ratio.
# Setting this value to 768px might result in more reasonable training data sizes for SDXL.
export RESOLUTION=1
export MINIMUM_RESOLUTION=$RESOLUTION
export VALIDATION_RESOLUTION=1280x768
export VALIDATION_NUM_INFERENCE_STEPS=50
export RESOLUTION_TYPE="area"
export ASPECT_BUCKET_ROUNDING=2

# Adjust this for your GPU memory size. This, and resolution, are the biggest VRAM killers.
export TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1
export MAX_NUM_STEPS=1000
export NUM_EPOCHS=0

# Set this to 'true' if you are using a patched Diffusers install.
# Ignore this if you don't know what it does. Having it at 'false' uses more VRAM.
export USE_GRADIENT_CHECKPOINTING=false

export LR_SCHEDULE="constant"
export LR_WARMUP_STEPS=0
export CAPTION_DROPOUT_PROBABILITY=0.1
export OPTIMIZER="adamw_bf16"


# Reproducible training. Set to -1 to disable.
export TRAINING_SEED=42
export VALIDATION_SEED=2
export MIXED_PRECISION="bf16"

# This has to be changed if you're training with multiple GPUs.
export TRAINING_NUM_PROCESSES=1

# Leave this alone.
export TRAINER_EXTRA_ARGS=""
export ACCELERATE_EXTRA_ARGS=""
export USE_XFORMERS=false
```

#### Dataloader

Place the following in `SimpleTuner/multidatabackend.json`

```json
[
    {
        "id": "pseudo-camera-10k-sd3",
        "type": "local",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 0.5,
        "minimum_image_size": 0.25,
        "maximum_image_size": 1.0,
        "target_downsample_size": 1.0,
        "resolution_type": "area",
        "cache_dir_vae": "cache/vae/sd3/pseudo-camera-10k",
        "instance_data_dir": "datasets/pseudo-camera-10k",
        "disabled": false,
        "skip_file_discovery": "",
        "caption_strategy": "filename",
        "metadata_backend": "json",
    },
    {
        "id": "text-embeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "cache/text/sd3/pseudo-camera-10k",
        "disabled": false,
        "write_batch_size": 128
    }
]
```

#### Download dataset

Execute the following commands:

```bash
apt -y install git-lfs
mkdir -p datasets
pushd datasets
    git clone https://huggingface.co/datasets/ptx0/pseudo-camera-10k
popd
```

This will download about 10k photograph samples to your `datasets/pseudo-camera-10k` directory, which will be automatically created for you.

#### Login to WandB and Huggingface Hub

You'll want to login to WandB and HF Hub before beginning training, especially if you're using `PUSH_TO_HUB=true` and `--report_to=wandb`.

If you're going to be pushing items to a Git LFS repository manually, you can run this command:

Otherwise, skip this command and run the next ones.

```bash
git config --global credential.helper store
```

Run the following commands:

```bash
wandb login
```

If you didn't run the `git config` command earlier, you'll want to say `no` when it asks if you want to add the credentials to your Git credentials store:

```bash
huggingface-cli login
```

### Executing the training run

From the SimpleTuner directory, one simply has to run:

```bash
bash train_sdxl.sh
```

This will begin the text embed and VAE output caching to disk.

For more information, see the [dataloader](/documentation/DATALOADER.md) and [tutorial](/TUTORIAL.md) documents.