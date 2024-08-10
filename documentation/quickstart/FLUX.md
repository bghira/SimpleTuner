## Flux[dev] / Flux[schnell] Quickstart

![image](https://github.com/user-attachments/assets/6409d790-3bb4-457c-a4b4-a51a45fc91d1)

In this example, we'll be training a Flux.1 LoRA model using the SimpleTuner toolkit.

### Hardware requirements

Flux requires a lot of **system RAM** in addition to GPU memory. Simply quantising the model at startup requires about 50GB of system memory. If it takes an excessively long time, you may need to assess your hardware's capabilities and whether any changes are needed.

When you're training every component of a rank-16 LoRA (MLP, projections, multimodal blocks), it ends up using:
- a bit more than 32G VRAM when not quantising the base model
- a bit more than 20G VRAM when quantising to int8 + bf16 base/LoRA weights
- a bit more than 13G VRAM when quantising to int2 + bf16 base/LoRA weights

To have reliable results, you'll need: 
- **at minimum** a single 3090 or V100 GPU
- **ideally** multiple A6000s

Luckily, these are readily available through providers such as TensorDock for extremely low rates (<$2/hr for A6000s, <$1/hr for 3090s>).

**Unlike other models, AMD and Apple GPUs do not work for training Flux.**

### Prerequisites

Make sure that you have python installed; SimpleTuner does well with 3.10 or 3.11. **Python 3.12 should not be used**.

You can check this by running:

```bash
python --version
```

If you don't have python 3.11 installed on Ubuntu, you can try the following:

```bash
apt -y install python3.11
```

### Installation

Clone the SimpleTuner repository and set up the python venv:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

# if python --version shows 3.11 you can just also use the 'python' command here.
python3.11 -m venv .venv

source .venv/bin/activate

pip install -U poetry pip
```

**Note:** We're currently installing the `release` branch here; the `main` branch may contain experimental features that might have better results or lower memory use.

Depending on your system, you will run one of 3 commands:

```bash
# MacOS
poetry install --no-root -C install/apple

# Linux
poetry install --no-root

# Linux with ROCM
poetry install --no-root -C install/rocm
```

#### Custom Diffusers build

For LoRA support in Diffusers, the latest release does not yet have Flux LoRA support, so we must install directly from the main branch.

To obtain the correct build, run the following commands:

```bash
pip uninstall diffusers
pip install git+https://github.com/huggingface/diffusers
```

### Setting up the environment

To run SimpleTuner, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

#### Configuration file

Copy `config/config.env.example` to `config/config.env`:

```bash
cp config/config.env.example config/config.env
```

There, you will need to modify the following variables:

- `MODEL_TYPE` - Set this to `lora`.
- `FLUX` - Set this to `true`.
- `MODEL_NAME` - Set this to `black-forest-labs/FLUX.1-dev`.
  - Note that you will *probably* need to log in to Huggingface and be granted access to download this model. We will go over logging in to Huggingface later in this tutorial.
- `OUTPUT_DIR` - Set this to the directory where you want to store your outputs and datasets. It's recommended to use a full path here.
- `VALIDATION_RESOLUTION` - As Flux is a 1024px model, you can set this to `1024x1024`.
  - Additionally, Flux was fine-tuned on multi-aspect buckets, and other resolutions may be specified using commas to separate them: `1024x1024,1280x768,2048x2048`
- `VALIDATION_GUIDANCE` - Use whatever you are used to selecting at inference time for Flux.
- `VALIDATION_GUIDANCE_REAL` - Use >1.0 to use CFG for flux inference. Slows validations down, but produces better results. Does best with an empty `VALIDATION_NEGATIVE_PROMPT`.
- `TRAINER_EXTRA_ARGS` - Here, you can place `--lora_rank=4` if you wish to substantially reduce the size of the LoRA being trained. This can help with VRAM use.
  - If training a Schnell LoRA, you'll have to supply `--flux_fast_schedule` manually here as well.

#### Validation prompts

Inside `config.env` is the "primary validation prompt", which is typically the main instance_prompt you are training on for your single subject or style. Additionally, a JSON file may be created that contains extra prompts to run through during validations.

The example config file `config/user_prompt_library.json.example` contains the following format:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

The nicknames are the filename for the validation, so keep them short and compatible with your filesystem.

To point the trainer to this prompt library, add it to TRAINER_EXTRA_ARGS by adding a new line at the end of `config.env`:
```bash
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --user_prompt_library=config/user_prompt_library.json"
```

A set of diverse prompt will help determine whether the model is collapsing as it trains. In this example, the word `<token>` should be replaced with your subject name (instance_prompt).

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing her essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing her unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing her dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```

> ℹ️ Flux is a flow-matching model and shorter prompts that have strong similarities will result in practically the same image being produced by the model. Be sure to use longer, more descriptive prompts.

#### Quantised model training

Tested on Apple and NVIDIA systems, Hugging Face Optimum-Quanto can be used to reduce the precision and VRAM requirements, training Flux on just 20GB.

Inside your SimpleTuner venv:

```bash
pip install optimum-quanto
```

```bash
# choices: int8-quanto, int4-quanto, int2-quanto, fp8-quanto
# int8-quanto was tested with a single subject dreambooth LoRA.
# fp8-quanto does not work on Apple systems. you must use int levels.
# int2-quanto is pretty extreme and gets the whole rank-1 LoRA down to about 13.9GB VRAM.
# may the gods have mercy on your soul, should you push things Too Far.
export TRAINER_EXTRA_ARGS="--base_model_precision=int8-quanto"

# Maybe you want the text encoders to remain full precision so your text embeds are cake.
# We unload the text encoders before training, so, that's not an issue during training time - only during pre-caching.
# Alternatively, you can go ham on quantisation here and run them in int4 or int8 mode, because no one can stop you.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --text_encoder_1_precision=no_change --text_encoder_2_precision=no_change"

# We'll enable some more Flux-specific options here to try and get better results.

# LoRA sizing you can adjust.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --lora_rank=16"
# Limiting gradient norms might preserve the model for longer, and fp32 gradients allow the use of accumulation steps.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --max_grad_norm=1.0 --gradient_precision=fp32"
# These options are the defaults, but they're restated here for clarity.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --base_model_default_dtype=bf16 --lora_init_type=loftq --flux_lora_target=mmdit"


# When you're quantising the model, --base_model_default_dtype is set to bf16 by default. This setup requires adamw_bf16, but saves the most memory.
# If you'd like to use another optimizer, you can override this with --base_model_default_dtype=fp32.
# option one - BF16 training:
export OPTIMIZER="adamw_bf16"
# option two - FP32 training:
#export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --base_model_default_dtype=fp32"
#export OPTIMIZER="adafactor" # or maybe prodigy
```


#### Dataset considerations

> ⚠️ Image quality for training is more important for Flux than for most other models, as it will absorb the artifacts in your images *first*, and then learn the concept/subject.

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively. Note that the bare minimum dataset size is `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS` as well as more than `VAE_BATCH_SIZE`. The dataset will not be useable if it is too small.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/ptx0/pseudo-camera-10k) as the dataset.

In your `OUTPUT_DIR` directory, create a multidatabackend.json:

```json
[
  {
    "id": "pseudo-camera-10k-flux",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel",
    "cache_dir_vae": "cache/vae/flux/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "json"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> ⚠️ 512-pixel training is recommended for Flux; it is more reliable than high-resolution training, which tends to diverge.

> ℹ️ Running 512px and 1024px datasets concurrently is supported, and could result in better convergence for Flux.

Then, navigate to the `OUTPUT_DIR` directory and create a `datasets` directory:

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

If you're going to be pushing items to a Git LFS repository manually, you should also run `git config --global credential.helper store`

Run the following commands:

```bash
wandb login
```

and

```bash
huggingface-cli login
```

Follow the instructions to log in to both services.

### Executing the training run

From the SimpleTuner directory, one simply has to run:

```bash
bash train.sh
```

This will begin the text embed and VAE output caching to disk.

For more information, see the [dataloader](/documentation/DATALOADER.md) and [tutorial](/TUTORIAL.md) documents.

**Note:** It's unclear whether training on multi-aspect buckets works correctly for Flux at the moment. It's recommended to use `crop_style=random` and `crop_aspect=square`.

## Notes & troubleshooting tips

### Quantisation
- Minimum 8bit quantisation is required for a 24G card to train this model - but 32G (V100) cards suffer a more tragic fate.
  - Without quantising the model, a rank-1 LoRA sits at just over 32GB of mem use, in a way that prevents a 32G V100 from actually working
  - Adafactor works, reducing VRAM to ~24G or further with sub-1024x1024 training
- Quantising the model isn't a bad thing
  - It allows you to push higher batch sizes and possibly obtain a better result
  - It unlocks the non-bf16 optimisers for use, such as Prodigy, Adafactor, Dadaptation, AdamW, and AdamW8Bit
  - Full model tuning has been compared to quantised and it behaves nearly the same - any issues you will encounter with quanto will happen without.
- As usual, **fp8 quantisation runs more slowly** than **int8** and might have a worse result due to the use of `e4m3fn` in Quanto
  - fp16 training similarly is bad for Flux; this model wants the range of bf16
  - `e5m2` level precision is better at fp8 but haven't looked into how to enable it yet. Sorry, H100 owners. We weep for you.

### Crashing
- If you get SIGKILL after the text encoders are unloaded, this means you do not have enough system memory to quantise Flux.
  - Try loading the `--base_model_precision=bf16` but if that does not work, you might just need more memory..

### Schnell
- Direct Schnell training really needs a bit more time in the oven - currently, the results do not look good
- Training a LoRA on Dev will then run just fine on Schnell
- Dev+Schnell merge 50/50 just fine, and the LoRAs can possibly be trained from that, which will then run on Schnell **or** Dev

### Learning rates
- A model as large as 12B has empirically performed better with **lower learning rates.**
  - LoRA at 1e-4 might totally roast the thing. LoRA at 1e-7 does nearly nothing.
- Ranks as large as 64 through 128 might be undesirable on a 12B model due to general difficulties that scale up with the size of the base model.
  - Try a smaller network first (rank-1, rank-4) and work your way up - they'll train faster, and might do everything you need.
- We're overriding `--max_grad_norm` on all DiT models currently - providing the flag `--i_know_what_im_doing` will allow you to bypass this limit and experiment with higher gradient norm scales
  - The low value keeps the model from falling apart too soon, but can also make it very difficult to learn new concepts that venture far from the base model data distribution

### Image artifacts
When you do these things (among others), some square grid artifacts **may** begin appearing in the samples:
- Overtrain with low quality data
- Use too high of a learning rate
- Select a bad optimiser
- Overtraining (in general), a low-capacity network with too many images
- Undertraining (also), a high-capacity network with too few images
- Using weird aspect ratios or training data sizes

### Aspect bucketing
- Training for too long on square crops probably won't damage this model. Go nuts, it's great and reliable.

### X-Flux LoRA trainer settings

To "match" the behaviour of the X-Flux trainer:

- Retrieve the two datasets of ~1M Midjourney/Nijijourney images that the X-flux Realism LoRA was trained on. **This will use more than 2tb of local disk space**, but this is how many images X-flux used.
  - https://huggingface.co/datasets/terminusresearch/midjourney-v6-520k-raw
    - SimpleTuner dataset preset [here](/documentation/data_presets/preset_midjourney.md)
  - https://huggingface.co/datasets/terminusresearch/nijijourney-v6-520k-raw
    - SimpleTuner dataset preset [here](/documentation/data_presets/preset_nijijourney.md)
- Configure DeepSpeed ZeRO 2 using `accelerate config` - see [DEEPSPEED.md](/documentation/DEEPSPEED.md) for more information on this
- Use the following settings inside `config.env`:

```bash
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=1e-5
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=10
# this is kinda crazy, but at 512px it trains rather quickly anyway.
CHECKPOINTING_STEPS=2500
# because of DeepSpeed, you can use the below flags to enable mixed-precision bf16 training:
OPTIMIZER="adamw" # unfortunately this is your only option with DeepSpeed, but x-flux does the same.
MIXED_PRECISION="bf16"
PURE_BF16=false
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --i_know_what_i_am_doing"
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --lora_rank=16"
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --max_grad_norm=1.0 --gradient_precision=fp32"
# x-flux only trains the mmdit blocks but you can change lora_target to all or context to experiment.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --base_model_default_dtype=bf16 --lora_init_type=default --flux_lora_target=mmdit"

```