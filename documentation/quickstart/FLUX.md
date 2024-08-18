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

Luckily, these are readily available through providers such as [LambdaLabs](https://lambdalabs.com) which provides the lowest available rates, and localised clusters for multi-node training.

**Unlike other models, AMD and Apple GPUs do not work for training Flux.**

### Prerequisites

Make sure that you have python installed; SimpleTuner does well with 3.10 or 3.11. **Python 3.12 should not be used**.

You can check this by running:

```bash
python --version
```

If you don't have python 3.11 installed on Ubuntu, you can try the following:

```bash
apt -y install python3.11 python3.11-venv
```

#### Container image dependencies

For Vast, RunPod, and TensorDock (among others), the following will work on a CUDA 12.2-12.4 image:

```bash
apt -y install nvidia-cuda-toolkit libgl1-mesa-glx
```

If `libgl1-mesa-glx` is not found, you might need to use `libgl1-mesa-dri` instead. Your mileage may vary.

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

#### Removing DeepSpeed & Bits n Bytes

These two dependencies cause numerous issues for container hosts such as RunPod and Vast.

To remove them after `poetry` has installed them, run the following command in the same terminal:

```bash
pip uninstall -y deepspeed bitsandbytes
```

#### Custom Diffusers build

We currently rely on Git upstream Diffusers builds for the most recent fixes in the Flux ecosystem.

To obtain the correct build, run the following commands:

```bash
pip uninstall diffusers
pip install git+https://github.com/huggingface/diffusers
```

### Setting up the environment

To run SimpleTuner, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

#### Configuration file

An experimental script, `configure.py`, may allow you to entirely skip this section through an interactive step-by-step configuration. It contains some safety features that help avoid common pitfalls.

**Note:** This doesn't configure your dataloader. You will still have to do that manually, later.

To run it:

```bash
python configure.py
```

If you prefer to manually configure:

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
- `TRAIN_BATCH_SIZE` - this should be kept at 1, especially if you have a very small dataset.
- `VALIDATION_RESOLUTION` - As Flux is a 1024px model, you can set this to `1024x1024`.
  - Additionally, Flux was fine-tuned on multi-aspect buckets, and other resolutions may be specified using commas to separate them: `1024x1024,1280x768,2048x2048`
- `VALIDATION_GUIDANCE` - Use whatever you are used to selecting at inference time for Flux.
- `VALIDATION_GUIDANCE_REAL` - Use >1.0 to use CFG for flux inference. Slows validations down, but produces better results. Does best with an empty `VALIDATION_NEGATIVE_PROMPT`.
- `VALIDATION_NUM_INFERENCE_STEPS` - Use somewhere around 20 to save time while still seeing decent quality. Flux isn't very diverse, and more steps might just waste time.
- `VALIDATION_NO_CFG_UNTIL_TIMESTEP` - When using `VALIDATION_GUIDANCE_REAL` with Flux, skip doing CFG until this timestep. Default 2.
- `TRAINER_EXTRA_ARGS` - Here, you can place `--lora_rank=4` if you wish to substantially reduce the size of the LoRA being trained. This can help with VRAM use.
  - If training a Schnell LoRA, you'll have to supply `--flux_fast_schedule` manually here as well.
- `GRADIENT_ACCUMULATION_STEPS` - Keep this low. 1 will disable it, which is recommended to maintain higher quality and reduce training runtime.
- `OPTIMIZER` - Beginners are recommended to stick with adamw_bf16, though Lion and StableAdamW are also good choices.
- `MIXED_PRECISION` - Beginners should keep this in `bf16` with `PURE_BF16=true` along with the adamw_bf16 optimiser.

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
#  - validations on int2 look pretty awful but the LoRA generally works on int8 / fp8 models at inference time.
# may the gods have mercy on your soul, should you push things Too Far.
export TRAINER_EXTRA_ARGS="--base_model_precision=int8-quanto"

# Maybe you want the text encoders to remain full precision so your text embeds are cake.
# We unload the text encoders before training, so, that's not an issue during training time - only during pre-caching.
# Alternatively, you can go ham on quantisation here and run them in int4 or int8 mode, because no one can stop you.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --text_encoder_1_precision=no_change --text_encoder_2_precision=no_change"

# LoRA sizing you can adjust.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --lora_rank=16"

# Limiting gradient norms might preserve the model for longer
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --max_grad_norm=1.0"
# Keeping the base in bf16 still allows you to quantise the model, but it saves a lot of memory.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --base_model_default_dtype=bf16"

# When training 'mmdit', we find very stable training that makes the model take longer to learn.
# When training 'all', we can easily shift the model distribution, but it is more prone to forgetting and benefits from high quality data.
# When training 'all+ffs', all attention layers are trained in addition to the feed-forward which can help with adapting the model objective for the LoRA.
# - This mode has been reported to lack portability, and platforms such as ComfyUI might not be able to load the LoRA.
# The option to train only the 'context' blocks is offered as well, but its impact is unknown, and is offered as an experimental choice.
# - An extension to this mode, 'context+ffs' is also available, which is useful for pretraining new tokens into a LoRA before continuing finetuning it via `--init_lora`.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --flux_lora_target=all"

# If you want to use LoftQ initialisation, you can't use Quanto to quantise the base model.
# This possibly offers better/faster convergence, but only works on NVIDIA devices and requires Bits n Bytes and is incompatible with Quanto.
# Other options are 'default', 'gaussian' (difficult), and untested options: 'olora' and 'pissa'.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --lora_init_type=loftq"

# When you're quantising the model, --base_model_default_dtype is set to bf16 by default. This setup requires adamw_bf16, but saves the most memory.
# Quantising the model has been found to result in negligible-to-quality loss for training.
# option one (recommended) - adamw_bf16; this optimiser setup is fairly forgiving
export OPTIMIZER="adamw_bf16"
# option two - FP32 training supports any optimiser BUT adamw_bf16
#export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --base_model_default_dtype=fp32"
#export OPTIMIZER="optimi-ranger" # or maybe optimi-lion
```


#### Dataset considerations

> ⚠️ Image quality for training is more important for Flux than for most other models, as it will absorb the artifacts in your images *first*, and then learn the concept/subject.

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively. Note that the bare minimum dataset size is `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS` as well as more than `VAE_BATCH_SIZE`. The dataset will not be useable if it is too small.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/ptx0/pseudo-camera-10k) as the dataset.

create a `DATALOADER_CONFIG` (config/multidatabackend.json) with this:

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
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "ignore_epochs": true,
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "json"
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
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

Then, create a `datasets` directory:

```bash
apt -y install git-lfs
mkdir -p datasets
pushd datasets
    git clone https://huggingface.co/datasets/ptx0/pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

This will download about 10k photograph samples to your `datasets/pseudo-camera-10k` directory, which will be automatically created for you.

Your Dreambooth images should go into the `datasets/dreambooth-subject` directory.

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

## Inference tips

### CFG-trained LoRAs (flux_guidance_value > 1)

In ComfyUI, you'll need to put Flux through another node called AdaptiveGuider. One of the members from our community has provided a modified node here:

(**external links**) [IdiotSandwichTheThird/ComfyUI-Adaptive-Guidan...](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps) and their example workflow [here](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps/blob/master/ExampleWorkflow.json)

### CFG-distilled LoRA (flux_guidance_scale == 1)

Inferencing the CFG-distilled LoRA is as easy as using a lower guidance_scale around the value trained with.


## Notes & troubleshooting tips

### Classifier-free guidance

#### Problem
The Dev model arrives guidance-distilled out of the box, which means it does a very straight shot trajectory to the teacher model outputs. This is done through a guidance vector that is fed into the model at training and inference time - the value of this vector greatly impacts what type of resulting LoRA you end up with:
- A value of 1.0 will preserve the initial distillation done to the Dev model
  - This is the most compatible mode
  - Inference is just as fast as the original model
  - Flow-matching distillation reduces the creativity and output variability of the model, as with the original Flux Dev model (everything keeps the same composition/look)
- A higher value (tested around 3.5-4.5) will reintroduce the CFG objective into the model
  - This requires the inference pipeline to have support for CFG
  - Inference is 50% slower and 0% VRAM increase **or** about 20% slower and 20% VRAM increase due to batched CFG inference
  - However, this style of training improves creativity and model output variability, which might be required for certain training tasks

It's not clear if we can reintroduce CFG to a de-distilled model by continuing tuning using a vector value of 1.0.

#### Solution
The solution for this is already enabled in the main branch; it is necessary to enable true CFG sampling at inference time when using LoRAs on Dev.

#### Caveats
- This has the end impact of **either**:
  - Increasing inference latency by 2x when we sequentially calculate the unconditional output, eg. with two separate forward pass
  - Increasing the VRAM consumption equivalently to using `num_images_per_prompt=2` and receiving two images at inference time, accompanied by the same percent slowdown.
    - This is often less extreme slowdown than sequential computation, but the VRAM use might be too much for most consumer training hardware.
    - This method is not *currently* integrated into SimpleTuner, but work is ongoing.
- Inference workflows for ComfyUI or other applications (eg. AUTOMATIC1111) will need to be modified to also enable "true" CFG, which might not be currently possible out of the box.

### Quantisation
- Minimum 8bit quantisation is required for a 24G card to train this model - but 32G (V100) cards suffer a more tragic fate.
  - Without quantising the model, a rank-1 LoRA sits at just over 32GB of mem use, in a way that prevents a 32G V100 from actually working
  - Using the optimi-lion optimiser may reduce training just enough to make the V100 work.
- Quantising the model doesn't harm training
  - It allows you to push higher batch sizes and possibly obtain a better result
  - Behaves the same as full-precision training - fp32 won't make your model any better than bf16+int8.
- As usual, **fp8 quantisation runs more slowly** than **int8** and might have a worse result due to the use of `e4m3fn` in Quanto
  - fp16 training similarly is bad for Flux; this model wants the range of bf16
  - `e5m2` level precision is better at fp8 but haven't looked into how to enable it yet. Sorry, H100 owners. We weep for you.
- When loading the LoRA in ComfyUI later, you **must** use the same base model precision as you trained your LoRA on.

### Crashing
- If you get SIGKILL after the text encoders are unloaded, this means you do not have enough system memory to quantise Flux.
  - Try loading the `--base_model_precision=bf16` but if that does not work, you might just need more memory..

### Schnell
- Direct Schnell training really needs a bit more time in the oven - currently, the results do not look good
  - If you absolutely must train Schnell, try the x-flux trainer from X-Labs
  - Ostris' ai-toolkit uses a low-rank adapter probably pulled from OpenFLUX.1 as a source of CFG that can be inverted from the final result - this will probably be implemented here eventually after results are more widely available and tests have completed
- Training a LoRA on Dev will however, run just fine on Schnell
- Dev+Schnell merge 50/50 just fine, and the LoRAs can possibly be trained from that, which will then run on Schnell **or** Dev

> ℹ️ When merging Schnell with Dev in any way, the license of Dev takes over and it becomes non-commercial. This shouldn't really matter for most users, but it's worth noting.

### Learning rates
- It's been reported that Flux trains similarly to SD 1.5 LoRAs
- However, a model as large as 12B has empirically performed better with **lower learning rates.**
  - LoRA at 1e-3 might totally roast the thing. LoRA at 1e-5 does nearly nothing.
- Ranks as large as 64 through 128 might be undesirable on a 12B model due to general difficulties that scale up with the size of the base model.
  - Try a smaller network first (rank-1, rank-4) and work your way up - they'll train faster, and might do everything you need.
  - If you're finding that it's excessively difficult to train your concept into the model, you might need a higher rank and more regularisation data.
- Other diffusion transformer models like PixArt and SD3 majorly benefit from `--max_grad_norm` and SimpleTuner keeps a pretty high value for this by default on Flux.
  - A lower value would keep the model from falling apart too soon, but can also make it very difficult to learn new concepts that venture far from the base model data distribution. The model might get stuck and never improve.

### Image artifacts
Flux will immediately absorb bad image artifacts. It's just how it is - a final training run on just high quality data may be required to fix it at the end.

When you do these things (among others), some square grid artifacts **may** begin appearing in the samples:
- Overtrain with low quality data
- Use too high of a learning rate
- Select a bad optimiser
- Overtraining (in general), a low-capacity network with too many images
- Undertraining (also), a high-capacity network with too few images
- Using weird aspect ratios or training data sizes
- Using gradient accumulation steps with pure bf16 training and `--gradient_precision=unmodified`

### Gradient accumulation steps

They really slow training down and might not be worth it unless you have several datasets configured in your dataloader backend.

The AdamWBF16 optimiser requires fp32 gradients for precise accumulation, but the Optimi selections such as Lion and StableAdamW claim to handle this more reliably. YMMV.

It's usually recommended to just avoid these.

### Aspect bucketing
- Training for too long on square crops probably won't damage this model. Go nuts, it's great and reliable.
- On the other hand, using the natural aspect buckets of your dataset might overly bias these shapes during inference time.
  - This could be a desirable quality, as it keeps aspect-dependent styles like cinematic stuff from bleeding into other resolutions too much.
  - However, if you're looking to improve results equally across many aspect buckets, you might have to experiment with `crop_aspect=random` which comes with its own downsides.

### Reproducing the results of X-Flux trainer / realism LoRA

To "match" the behaviour of the X-Flux trainer and reproduce their Realism LoRA:

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
OPTIMIZER="optimi-adamw" # unfortunately this is your only option with DeepSpeed, but x-flux does the same.
MIXED_PRECISION="bf16"
PURE_BF16=false
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --i_know_what_i_am_doing"
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --lora_rank=16"
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --max_grad_norm=1.0 --gradient_precision=fp32"
# x-flux only trains the mmdit blocks but you can change lora_target to all or context to experiment.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --base_model_default_dtype=bf16 --lora_init_type=default --flux_lora_target=mmdit"
```

## Credits

The users of [Terminus Research](https://huggingface.co/terminusresearch) who worked on this probably more than their day jobs to figure it out

[Lambda Labs](https://lambdalabs.com) for generous compute allocations that were used for tests and verifications for large scale training runs

Especially [@JimmyCarter](https://huggingface.co/jimmycarter) and [@kaibioinfo](https://github.com/kaibioinfo) for coming up with some of the best ideas and putting them into action, offering pull requests and running exhaustive tests for analysis - even daring to use _their own faces_ for DreamBooth experimentation.
