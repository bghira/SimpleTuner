# DeepFloyd IF

> ⚠️ Support for tuning DeepFloyd-IF is in its initial support stages. Not all SimpleTuner features are currently available for this model.

> 🤷🏽‍♂️ Training DeepFloyd requires at least 24G VRAM for a LoRA. This guide focuses on the 400M parameter base model, though the 4.3B XL flavour can be trained using the same guidelines.

## Background

In spring of 2023, StabilityAI released a cascaded pixel diffusion model called DeepFloyd.
![](https://tripleback.net/public/deepfloyd.png)

Comparing briefly to Stable Diffusion XL:
- Text encoder
  - SDXL uses two CLIP encoders, "OpenCLIP G/14" and "OpenAI CLIP-L/14"
  - DeepFloyd uses a single self-supervised transformer model, Google's T5 XXL
- Parameter count
  - DeepFloyd comes in multiple flavours of density: 400M, 900M, and 4.3B parameters. Each larger unit is successively more expensive to train.
  - SDXL has just one, ~3B parameters.
  - DeepFloyd's text encoder has 11B parameters in it alone, making the fattest configuration roughly 15.3B parameters.
- Model count
  - DeepFloyd runs in **three** stages: 64px -> 256px -> 1024px
    - Each stage fully completes its denoising objective
  - SDXL runs in **two** stages, including its refiner, from 1024px -> 1024px
    - Each stage only partly completes its denoising objective
- Design
  - DeepFloyd's three models increase resolution and fine details
  - SDXL's two models manage fine details and composition

For both models, the first stage defines most of the image's composition (where large items / shadows appear).

## Model assessment

Here's what you can expect when using DeepFloyd for training or inference.

### Aesthetics

When compared to SDXL or Stable Diffusion 1.x/2.x, DeepFloyd's aesthetics lie somewhere between Stable Diffusion 2.x and SDXL.


### Disadvantages

This is not a popular model, for various reasons:

- Inference-time compute VRAM requirement is heavier than other models
- Training-time compute VRAM requirements dwarf other models
  - A full u-net tune needing more than 48G VRAM
  - LoRA at rank-32, batch-4 needs ~24G VRAM
  - The text embed cache objects are ENORMOUS (multiple Megabytes each, vs hundreds of Kilobytes for SDXL's dual CLIP embeds)
  - The text embed cache objects are SLOW TO CREATE, about 9-10 per second currently on an A6000 non-Ada.
- The default aesthetic is worse than other models (like trying to train vanilla SD 1.5)
- There's **three** models to finetune or load onto your system during inference (four if you count the text encoder)
- The promises from StabilityAI did not meet the reality of what it felt like to use the model (over-hyped)
- The DeepFloyd-IF license is restrictive against commercial use.
  - This didn't impact the NovelAI weights, which were in fact leaked illicitly. The commercial license nature seems like a convenient excuse, considering the other, bigger issues.

### Advantages

However, DeepFloyd really has its upsides that often go overlooked:

- At inference time, the T5 text encoder demonstrates a strong understanding of the world
- Can be natively trained on very-long captions
- The first stage is ~64x64 pixel area, and can be trained on multi-aspect resolutions
  - The low-resolution nature of the training data means DeepFloyd was _the only model_ capable of training on _ALL_ of LAION-A (few images are under 64x64 in LAION)
- Each stage can be tuned independently, focusing on different objectives
  - The first stage can be tuned focusing on compositional qualities, and the later stages are tuned for better upscaled details
- It trains very quickly despite its larger training memory footprint
  - Trains quicker in terms of throughput - a high samples per hour rate is observed on stage 1 tuning
  - Learns more quickly than a CLIP equivalent model, perhaps to the detriment of people used to training CLIP models
    - In other words, you will have to adjust your expectations of learning rates and training schedules
- There is no VAE, the training samples are directly downscaled into their target size and the pixels are consumed by the U-net
- It supports ControlNet LoRAs and many other tricks that work on typical linear CLIP u-nets.

## Fine-tuning a LoRA

> ⚠️ Due to the compute requirements of full u-net backpropagation in even DeepFloyd's smallest 400M model, it has not been tested. LoRA will be used for this document, though full u-net tuning should also work.

Training DeepFloyd makes use of the "legacy" SD 1.x/2.x trainer in SimpleTuner to reduce code duplication by keeping similar models together.

As such, we'll be making use of the `sd2x-env.sh` configuration file for tuning DeepFloyd:

### sd2x-env.sh

```bash
# Possible values:
# - deepfloyd-full
# - deepfloyd-lora
# - deepfloyd-stage2
# - deepfloyd-stage2-lora
export MODEL_TYPE="deepfloyd-lora"

# DoRA isn't tested a whole lot yet. It's still new and experimental.
export USE_DORA=false
# Bitfit hasn't been tested for efficacy on DeepFloyd.
# It will probably work, but no idea what the outcome is.
export USE_BITFIT=false

# Highest learning rate to use.
export LEARNING_RATE=4e-5 #@param {type:"number"}
# For schedules that decay or oscillate, this will be the end LR or the bottom of the valley.
export LEARNING_RATE_END=4e-6 #@param {type:"number"}

## Using a Huggingface Hub model for Stage 1 tuning:
#export MODEL_NAME="DeepFloyd/IF-I-M-v1.0"
## Using a Huggingface Hub model for Stage 2 tuning:
#export MODEL_NAME="DeepFloyd/IF-II-M-v1.0"
# Using a local path to a huggingface hub model or saved checkpoint:
#export MODEL_NAME="/notebooks/datasets/models/pipeline"

# Where to store your results.
export BASE_DIR="/training"
export INSTANCE_DIR="${BASE_DIR}/data"
export OUTPUT_DIR="${BASE_DIR}/models/deepfloyd"
export DATALOADER_CONFIG="multidatabackend_deepfloyd.json"

# Max number of steps OR epochs can be used. But we default to Epochs.
export MAX_NUM_STEPS=50000
export NUM_EPOCHS=0

# Adjust this for your GPU memory size.
export TRAIN_BATCH_SIZE=1

# "pixel" is using pixel edge length on the smaller or square side of the image.
# this is how DeepFloyd was originally trained.
export RESOLUTION_TYPE="pixel"
export RESOLUTION=64          # 1.0 Megapixel training sizes

# Validation is when the model is used during training to make test outputs.
export VALIDATION_RESOLUTION=96x64                                                  # The resolution of the validation images. Default: 64x64
export VALIDATION_STEPS=250                                                         # How long between each validation run. Default: 250
export VALIDATION_NUM_INFERENCE_STEPS=25                                            # How many inference steps to do. Default: 25
export VALIDATION_PROMPT="an ethnographic photograph of a teddy bear at a picnic"   # What to make for the first/only test image.
export VALIDATION_NEGATIVE_PROMPT="blurry, ugly, cropped, amputated"                # What to avoid in the first/only test image.

# These can be left alone.
export VALIDATION_GUIDANCE=7.5
export VALIDATION_GUIDANCE_RESCALE=0.0
export VALIDATION_SEED=42

export GRADIENT_ACCUMULATION_STEPS=1         # Accumulate over many steps. Default: 1
export MIXED_PRECISION="bf16"                # SimpleTuner requires bf16.
export PURE_BF16=true                        # Will not use mixed precision, but rather pure bf16 (bf16 requires pytorch 2.3 on MPS.)
export OPTIMIZER="adamw_bf16"
export USE_XFORMERS=true
```

A keen eye will have observed the following:

- The `MODEL_TYPE` is specified as deepfloyd-compatible
- The `MODEL_NAME` is pointing to Stage I or II
- `RESOLUTION` is now `64` and `RESOLUTION_TYPE` is `pixel`
- `USE_XFORMERS` is set to `true`, but AMD and Apple users won't be able to set this, requiring more VRAM.
  - **Note** Apple MPS currently has a bug preventing DeepFloyd tuning from working at all.

For more thorough validations, the value for `VALIDATION_RESOLUTION` can be set as:

- `VALIDATION_RESOLUTION=64` will result in a 64x64 square image.
- `VALIDATION_RESOLUTION=96x64` will result in a 3:2 widescreen image.
- `VALIDATION_RESOLUTION=64,96,64x96,96x64` will result in four images being generated for each validation:
  - 64x64
  - 96x96
  - 64x96
  - 96x64

### multidatabackend_deepfloyd.json

Now let's move onto configuring the dataloader for DeepFloyd training. This will be nearly identical to configuration of SDXL or legacy model datasets, with a focus on resolution parameters.

```json
[
    {
        "id": "primary-dataset",
        "type": "local",
        "instance_data_dir": "/training/data/primary-dataset",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "random",
        "resolution": 64,
        "resolution_type": "pixel",
        "minimum_image_size": 64,
        "maximum_image_size": 256,
        "target_downsample_size": 128,
        "prepend_instance_prompt": false,
        "instance_prompt": "Your Subject Trigger Phrase or Word",
        "caption_strategy": "instanceprompt",
        "repeats": 1
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "disable": false,
        "type": "local",
        "cache_dir": "/training/cache/deepfloyd/text/dreambooth"
    }
]
```

Provided above is a basic Dreambooth configuration for DeepFloyd:

- The values for `resolution` and `resolution_type` are set to `64` and `pixel`, respectively
- The value for `minimum_image_size` is reduced to 64 pixels to ensure we don't accidentally upsample any smaller images
- The value for `maximum_image_size` is set to 256 pixels to ensure that any large images do not become cropped at a ratio of more than 4:1, which may result in catastrophic scene context loss
- The value for `target_downsample_size` is set to 128 pixels so that any images larger than `maximum_image_size` of 256 pixels are first resized to 128 pixels before cropping

Note: images are downsampled 25% at a time so to avoid extreme leaps in image size causing an incorrect averaging of the scene's details.

## Running inference

Currently, DeepFloyd does not have any dedicated inference scripts in the SimpleTuner toolkit.

Other than the built-in validations process, you may want to reference [this document from Hugging Face](https://huggingface.co/docs/diffusers/v0.23.1/en/training/dreambooth#if) which contains a small example for running inference afterward:

```py
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", use_safetensors=True)
pipe.load_lora_weights("<lora weights path>")
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
```

> ⚠️ Note that the first value for `DiffusionPipeline.from_pretrained(...)` is set to `IF-I-M-v1.0`, but you must update this to use the base model path that you trained your LoRA on.

> ⚠️ Note that not all of the recommendations from Hugging Face apply to SimpleTuner. For example, we can tune DeepFloyd stage I LoRA in just 22G of VRAM vs 28G for Diffusers' example dreambooth scripts thanks to efficient pre-caching and pure-bf16 optimiser states. 8Bit AdamW isn't currently supported by SimpleTuner.

## Fine-tuning the super-resolution stage II model

Currently, stage II training has a bug in SimpleTuner resulting in a crash on the first step.

This is being identified and resolved, but until then, only stage I can be tuned.