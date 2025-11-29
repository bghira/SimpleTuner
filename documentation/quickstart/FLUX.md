# Flux[dev] / Flux[schnell] Quickstart

![image](https://github.com/user-attachments/assets/6409d790-3bb4-457c-a4b4-a51a45fc91d1)

In this example, we'll be training a Flux.1 Krea LoRA.

## Hardware requirements

Flux requires a lot of **system RAM** in addition to GPU memory. Simply quantising the model at startup requires about 50GB of system memory. If it takes an excessively long time, you may need to assess your hardware's capabilities and whether any changes are needed.

When you're training every component of a rank-16 LoRA (MLP, projections, multimodal blocks), it ends up using:

- a bit more than 30G VRAM when not quantising the base model
- a bit more than 18G VRAM when quantising to int8 + bf16 base/LoRA weights
- a bit more than 13G VRAM when quantising to int4 + bf16 base/LoRA weights
- a bit more than 9G VRAM when quantising to NF4 + bf16 base/LoRA weights
- a bit more than 9G VRAM when quantising to int2 + bf16 base/LoRA weights

You'll need:

- **the absolute minimum** is a single **3080 10G**
- **a realistic minimum** is a single 3090 or V100 GPU
- **ideally** multiple 4090, A6000, L40S, or better

Luckily, these are readily available through providers such as [LambdaLabs](https://lambdalabs.com) which provides the lowest available rates, and localised clusters for multi-node training.

**Unlike other models, Apple GPUs do not currently work for training Flux.**


## Prerequisites

Make sure that you have python installed; SimpleTuner does well with 3.10 through 3.12.

You can check this by running:

```bash
python --version
```

If you don't have python 3.12 installed on Ubuntu, you can try the following:

```bash
apt -y install python3.12 python3.12-venv
```

### Container image dependencies

For Vast, RunPod, and TensorDock (among others), the following will work on a CUDA 12.2-12.8 image to enable compiling of CUDA extensions:

```bash
apt -y install nvidia-cuda-toolkit
```

## Installation

Install SimpleTuner via pip:

```bash
pip install simpletuner[cuda]
```

For manual installation or development setup, see the [installation documentation](/documentation/INSTALL.md).

### AMD ROCm follow-up steps

The following must be executed for an AMD MI300X to be useable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## Setting up the environment

### Web interface method

The SimpleTuner WebUI makes setup fairly straightforward. To run the server:

```bash
simpletuner server
```

This will create a webserver on port 8001 by default, which you can access by visiting http://localhost:8001.

### Manual / command-line method

To run SimpleTuner via command-line tools, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

#### Configuration file

An experimental script, `configure.py`, may allow you to entirely skip this section through an interactive step-by-step configuration. It contains some safety features that help avoid common pitfalls.

**Note:** This doesn't configure your dataloader. You will still have to do that manually, later.

To run it:

```bash
simpletuner configure
```

> ⚠️ For users located in countries where Hugging Face Hub is not readily accessible, you should add `HF_ENDPOINT=https://hf-mirror.com` to your `~/.bashrc` or `~/.zshrc` depending on which `$SHELL` your system uses.

If you prefer to manually configure:

Copy `config/config.json.example` to `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

There, you will possibly need to modify the following variables:

- `model_type` - Set this to `lora`.
- `model_family` - Set this to `flux`.
- `model_flavour` - this is `krea` by default, but may be set to `dev` to train the original FLUX.1-Dev release.
  - `krea` - The default FLUX.1-Krea [dev] model, an open-weights variant of Krea 1, a proprietary model collaboration between BFL and Krea.ai
  - `dev` - Dev model flavour, the previous default
  - `schnell` - Schnell model flavour, and set any appropriate options incl. fast training schedule
  - `kontext` - Kontext training (see [this guide](/documentation/quickstart/FLUX_KONTEXT.md) for specific guidance)
  - `fluxbooru` - A de-distilled (requires CFG) model based on FLUX.1-Dev called [FluxBooru](https://hf.co/terminusresearch/fluxbooru-v0.3), created by terminus research group
  - `libreflux` - A de-distilled model based on FLUX.1-Schnell that requires attention masking on the T5 text encoder inputs
- `offload_during_startup` - Set this to `true` if you run out of memory during VAE encodes.
- `pretrained_model_name_or_path` - Set this to `black-forest-labs/FLUX.1-dev`.
- `pretrained_vae_model_name_or_path` - Set this to `black-forest-labs/FLUX.1-dev`.
  - Note that you will need to log in to Huggingface and be granted access to download this model. We will go over logging in to Huggingface later in this tutorial.
- `output_dir` - Set this to the directory where you want to store your checkpoints and validation images. It's recommended to use a full path here.
- `train_batch_size` - this should be kept at 1, especially if you have a very small dataset.
- `validation_resolution` - As Flux is a 1024px model, you can set this to `1024x1024`.
  - Additionally, Flux was fine-tuned on multi-aspect buckets, and other resolutions may be specified using commas to separate them: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - Use whatever you are used to selecting at inference time for Flux.
- `validation_guidance_real` - Use >1.0 to use CFG for flux inference. Slows validations down, but produces better results. Does best with an empty `VALIDATION_NEGATIVE_PROMPT`.
- `validation_num_inference_steps` - Use somewhere around 20 to save time while still seeing decent quality. Flux isn't very diverse, and more steps might just waste time.
- `--lora_rank=4` if you wish to substantially reduce the size of the LoRA being trained. This can help with VRAM use.
- If training a Schnell LoRA, you'll have to supply `--flux_fast_schedule=true` manually here as well.

- `gradient_accumulation_steps` - Previous guidance was to avoid these with bf16 training since they would degrade the model. Further testing showed this is not necessarily the case for Flux.
  - This option causes update steps to be accumulated over several steps. This will increase the training runtime linearly, such that a value of 2 will make your training run half as quickly, and take twice as long.
- `optimizer` - Beginners are recommended to stick with adamw_bf16, though optimi-lion and optimi-stableadamw are also good choices.
- `mixed_precision` - Beginners should keep this in `bf16`
- `gradient_checkpointing` - set this to true in practically every situation on every device
- `gradient_checkpointing_interval` - this could be set to a value of 2 or higher on larger GPUs to only checkpoint every _n_ blocks. A value of 2 would checkpoint half of the blocks, and 3 would be one-third.

### Memory offloading (optional)

Flux supports grouped module offloading via diffusers v0.33+. This dramatically reduces VRAM pressure when you are bottlenecked by the transformer weights. You can enable it by adding the following flags to `TRAINER_EXTRA_ARGS` (or the WebUI Hardware page):

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- `--group_offload_use_stream` is only effective on CUDA devices; SimpleTuner automatically disables streams on ROCm, MPS and CPU backends.
- Do **not** combine this with `--enable_model_cpu_offload` — the two strategies are mutually exclusive.
- When using `--group_offload_to_disk_path`, prefer a fast local SSD/NVMe target.

#### Validation prompts

Inside `config/config.json` is the "primary validation prompt", which is typically the main instance_prompt you are training on for your single subject or style. Additionally, a JSON file may be created that contains extra prompts to run through during validations.

The example config file `config/user_prompt_library.json.example` contains the following format:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

The nicknames are the filename for the validation, so keep them short and compatible with your filesystem.

To point the trainer to this prompt library, add it to TRAINER_EXTRA_ARGS by adding a new line at the end of `config.json`:

```json
  "--user_prompt_library": "config/user_prompt_library.json",
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

#### CLIP score tracking

If you wish to enable evaluations to score the model's performance, see [this document](/documentation/evaluation/CLIP_SCORES.md) for information on configuring and interpreting CLIP scores.

# Stable evaluation loss

If you wish to use stable MSE loss to score the model's performance, see [this document](/documentation/evaluation/EVAL_LOSS.md) for information on configuring and interpreting evaluation loss.

#### Validation previews

SimpleTuner supports streaming intermediate validation previews during generation using Tiny AutoEncoder models. This allows you to see validation images being generated step-by-step in real-time via webhook callbacks.

To enable:
```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

**Requirements:**
- Webhook configuration
- Validation enabled

Set `validation_preview_steps` to a higher value (e.g., 3 or 5) to reduce Tiny AutoEncoder overhead. With `validation_num_inference_steps=20` and `validation_preview_steps=5`, you'll receive preview images at steps 5, 10, 15, and 20.

#### Flux time schedule shifting

Flow-matching models such as Flux and SD3 have a property called "shift" that allows us to shift the trained portion of the timestep schedule using a simple decimal value.

##### Defaults

By default, no schedule shift is applied to flux, which results in a sigmoid bell-shape to the timestep sampling distribution. This is unlikely to be the ideal approach for Flux, but it results in a greater amount of learning in a shorter period of time than auto-shift.

##### Auto-shift

A commonly-recommended approach is to follow several recent works and enable resolution-dependent timestep shift, `--flow_schedule_auto_shift` which uses higher shift values for larger images, and lower shift values for smaller images. This results in stable but potentially mediocre training results.

##### Manual specification

(_Thanks to General Awareness from Discord for the following examples_)

When using a `--flow_schedule_shift` value of 0.1 (a very low value), only the finer details of the image are affected:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

When using a `--flow_schedule_shift` value of 4.0 (a very high value), the large compositional features and potentially colour space of the model becomes impacted:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)

#### Quantised model training

Tested on Apple and NVIDIA systems, Hugging Face Optimum-Quanto can be used to reduce the precision and VRAM requirements, training Flux on just 16GB.

For `config.json` users:

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```

##### LoRA-specific settings (not LyCORIS)

```bash
# When training 'mmdit', we find very stable training that makes the model take longer to learn.
# When training 'all', we can easily shift the model distribution, but it is more prone to forgetting and benefits from high quality data.
# When training 'all+ffs', all attention layers are trained in addition to the feed-forward which can help with adapting the model objective for the LoRA.
# - This mode has been reported to lack portability, and platforms such as ComfyUI might not be able to load the LoRA.
# The option to train only the 'context' blocks is offered as well, but its impact is unknown, and is offered as an experimental choice.
# - An extension to this mode, 'context+ffs' is also available, which is useful for pretraining new tokens into a LoRA before continuing finetuning it via `--init_lora`.
# Other options include 'tiny' and 'nano' which train just 1 or 2 layers.
"--flux_lora_target": "all",

# If you want to use LoftQ initialisation, you can't use Quanto to quantise the base model.
# This possibly offers better/faster convergence, but only works on NVIDIA devices and requires Bits n Bytes and is incompatible with Quanto.
# Other options are 'default', 'gaussian' (difficult), and untested options: 'olora' and 'pissa'.
"--lora_init_type": "loftq",
```

#### Dataset considerations

> ⚠️ Image quality for training is more important for Flux than for most other models, as it will absorb the artifacts in your images _first_, and then learn the concept/subject.

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively. Note that the bare minimum dataset size is `train_batch_size * gradient_accumulation_steps` as well as more than `vae_batch_size`. The dataset will not be useable if it is too small.

> ℹ️ With few enough images, you might see a message **no images detected in dataset** - increasing the `repeats` value will overcome this limitation.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) as the dataset.

Create a `--data_backend_config` (`config/multidatabackend.json`) document containing this:

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
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
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
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "dreambooth-subject-512",
    "type": "local",
    "crop": false,
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/dreambooth-subject-512",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> ℹ️ Running 512px and 1024px datasets concurrently is supported, and could result in better convergence for Flux.

Then, create a `datasets` directory:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

This will download about 10k photograph samples to your `datasets/pseudo-camera-10k` directory, which will be automatically created for you.

Your Dreambooth images should go into the `datasets/dreambooth-subject` directory.

#### Login to WandB and Huggingface Hub

You'll want to login to WandB and HF Hub before beginning training, especially if you're using `--push_to_hub` and `--report_to=wandb`.

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

From the SimpleTuner directory, you have several options to start training:

**Option 1 (Recommended - pip install):**

```bash
pip install simpletuner[cuda]
simpletuner train
```

**Option 2 (Git clone method):**

```bash
simpletuner train
```

**Option 3 (Legacy method - still works):**

```bash
./train.sh
```

This will begin the text embed and VAE output caching to disk.

For more information, see the [dataloader](/documentation/DATALOADER.md) and [tutorial](/documentation/TUTORIAL.md) documents.

**Note:** It's unclear whether training on multi-aspect buckets works correctly for Flux at the moment. It's recommended to use `crop_style=random` and `crop_aspect=square`.

## Multi-GPU Configuration

SimpleTuner includes **automatic GPU detection** through the WebUI. During onboarding, you'll configure:

- **Auto Mode**: Automatically uses all detected GPUs with optimal settings
- **Manual Mode**: Select specific GPUs or set custom process count
- **Disabled Mode**: Single GPU training

The WebUI detects your hardware and configures `--num_processes` and `CUDA_VISIBLE_DEVICES` automatically.

For manual configuration or advanced setups, see the [Multi-GPU Training section](/documentation/INSTALL.md#multiple-gpu-training) in the installation guide.

## Inference tips

### CFG-trained LoRAs (flux_guidance_value > 1)

In ComfyUI, you'll need to put Flux through another node called AdaptiveGuider. One of the members from our community has provided a modified node here:

(**external links**) [IdiotSandwichTheThird/ComfyUI-Adaptive-Guidan...](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps) and their example workflow [here](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps/blob/master/ExampleWorkflow.json)

### CFG-distilled LoRA (flux_guidance_scale == 1)

Inferencing the CFG-distilled LoRA is as easy as using a lower guidance_scale around the value trained with.

## Notes & troubleshooting tips

### Lowest VRAM config

Currently, the lowest VRAM utilisation (9090M) can be attained with:

- OS: Ubuntu Linux 24
- GPU: A single NVIDIA CUDA device (10G, 12G)
- System memory: 50G of system memory approximately
- Base model precision: `nf4-bnb`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 512px
  - 1024px requires >= 12G VRAM
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: disabled / unconfigured
- PyTorch: 2.6 Nightly (Sept 29th build)
- Using `--quantize_via=cpu` to avoid outOfMemory error during startup on <=16G cards.
- With `--attention_mechanism=sageattention` to further reduce VRAM by 0.1GB and improve training validation image generation speed.
- Be sure to enable `--gradient_checkpointing` or nothing you do will stop it from OOMing

**NOTE**: Pre-caching of VAE embeds and text encoder outputs may use more memory and still OOM. If so, text encoder quantisation and VAE tiling can be enabled via `--vae_enable_tiling=true`. Further memory can be saved on startup with `--offload_during_startup=true`.

Speed was approximately 1.4 iterations per second on a 4090.

### SageAttention

When using `--attention_mechanism=sageattention`, inference can be sped-up at validation time.

**Note**: This isn't compatible with _every_ model configuration, but it's worth trying.

### NF4-quantised training

In simplest terms, NF4 is a 4bit-_ish_ representation of the model, which means training has serious stability concerns to address.

In early tests, the following holds true:

- Lion optimiser causes model collapse but uses least VRAM; AdamW variants help to hold it together; bnb-adamw8bit, adamw_bf16 are great choices
  - AdEMAMix didn't fare well, but settings were not explored
- `--max_grad_norm=0.01` further helps reduce model breakage by preventing huge changes to the model in too short a time
- NF4, AdamW8bit, and a higher batch size all help to overcome the stability issues, at the cost of more time spent training or VRAM used
- Upping the resolution from 512px to 1024px slows training down from, for example, 1.4 seconds per step to 3.5 seconds per step (batch size of 1, 4090)
- Anything that's difficult to train on int8 or bf16 becomes harder in NF4
- It's less compatible with options like SageAttention

NF4 does not work with torch.compile, so whatever you get for speed is what you get.

If VRAM is not a concern (eg. 48G or greater) then int8 with torch.compile is your best, fastest option.

### Masked loss

If you are training a subject or style and would like to mask one or the other, see the [masked loss training](/documentation/DREAMBOOTH.md#masked-loss) section of the Dreambooth guide.

### TREAD training

> ⚠️ **Experimental**: TREAD is a newly implemented feature. While functional, optimal configurations are still being explored.

[TREAD](/documentation/TREAD.md) (paper) stands for **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion. It is a method that can accelerate Flux training by intelligently routing tokens through transformer layers. The speedup is proportional to how many tokens you drop.

#### Quick setup

Add this to your `config.json`:

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```

This configuration will:

- Keep only 50% of image tokens during layers 2 through second-to-last
- Text tokens are never dropped
- Training speedup of ~25% with minimal quality impact

#### Key points

- **Limited architecture support** - TREAD is only implemented for Flux and Wan models
- **Best at high resolutions** - Biggest speedups at 1024x1024+ due to attention's O(n²) complexity
- **Compatible with masked loss** - Masked regions are automatically preserved (but this reduces speedup)
- **Works with quantization** - Can be combined with int8/int4/NF4 training
- **Expect initial loss spike** - When starting LoRA/LoKr training, loss will be higher initially but corrects quickly

#### Tuning tips

- **Conservative (quality-focused)**: Use `selection_ratio` of 0.3-0.5
- **Aggressive (speed-focused)**: Use `selection_ratio` of 0.6-0.8
- **Avoid early/late layers**: Don't route in layers 0-1 or the final layer
- **For LoRA training**: May see slight slowdowns - experiment with different configs
- **Higher resolution = better speedup**: Most beneficial at 1024px and above

#### Known behavior

- The more tokens dropped (higher `selection_ratio`), the faster training but higher initial loss
- LoRA/LoKr training shows an initial loss spike that rapidly corrects as the network adapts
- Some LoRA configurations may train slightly slower - optimal configs still being explored
- The RoPE (rotary position embedding) implementation is functional but may not be 100% correct

For detailed configuration options and troubleshooting, see the [full TREAD documentation](/documentation/TREAD.md).

### Classifier-free guidance

#### Problem

The Dev model arrives guidance-distilled out of the box, which means it does a very straight shot trajectory to the teacher model outputs. This is done through a guidance vector that is fed into the model at training and inference time - the value of this vector greatly impacts what type of resulting LoRA you end up with:

#### Solution

- A value of 1.0 (**the default**) will preserve the initial distillation done to the Dev model
  - This is the most compatible mode
  - Inference is just as fast as the original model
  - Flow-matching distillation reduces the creativity and output variability of the model, as with the original Flux Dev model (everything keeps the same composition/look)
- A higher value (tested around 3.5-4.5) will reintroduce the CFG objective into the model
  - This requires the inference pipeline to have support for CFG
  - Inference is 50% slower and 0% VRAM increase **or** about 20% slower and 20% VRAM increase due to batched CFG inference
  - However, this style of training improves creativity and model output variability, which might be required for certain training tasks

We can partially reintroduce distillation to a de-distilled model by continuing tuning your model using a vector value of 1.0. It will never fully recover, but it'll at least be more useable.

#### Caveats

- This has the end impact of **either**:
  - Increasing inference latency by 2x when we sequentially calculate the unconditional output, eg. with two separate forward pass
  - Increasing the VRAM consumption equivalently to using `num_images_per_prompt=2` and receiving two images at inference time, accompanied by the same percent slowdown.
    - This is often less extreme slowdown than sequential computation, but the VRAM use might be too much for most consumer training hardware.
    - This method is not _currently_ integrated into SimpleTuner, but work is ongoing.
- Inference workflows for ComfyUI or other applications (eg. AUTOMATIC1111) will need to be modified to also enable "true" CFG, which might not be currently possible out of the box.

### Quantisation

- Minimum 8bit quantisation is required for a 16G card to train this model
  - In bfloat16/float16, a rank-1 LoRA sits at just over 30GB of mem use
- Quantising the model to 8bit doesn't harm training
  - It allows you to push higher batch sizes and possibly obtain a better result
  - Behaves the same as full-precision training - fp32 won't make your model any better than bf16+int8.
- **int8** has hardware acceleration and `torch.compile()` support on newer NVIDIA hardware (3090 or better)
- **nf4-bnb** brings VRAM requirements down to 9GB, fitting on a 10G card (with bfloat16 support)
- When loading the LoRA in ComfyUI later, you **must** use the same base model precision as you trained your LoRA on.
- **int4** is relies on custom bf16 kernels, and will not work if your card does not support bfloat16

### Crashing

- If you get SIGKILL after the text encoders are unloaded, this means you do not have enough system memory to quantise Flux.
  - Try loading the `--base_model_precision=bf16` but if that does not work, you might just need more memory..
  - Try `--quantize_via=accelerator` to use the GPU instead

### Schnell

- If you train a LyCORIS LoKr on Dev, it **generally** works very well on Schnell at just 4 steps later.
  - Direct Schnell training really needs a bit more time in the oven - currently, the results do not look good

> ℹ️ When merging Schnell with Dev in any way, the license of Dev takes over and it becomes non-commercial. This shouldn't really matter for most users, but it's worth noting.

### Learning rates

#### LoRA (--lora_type=standard)

- LoRA has overall worse performance than LoKr for larger datasets
- It's been reported that Flux LoRA trains similarly to SD 1.5 LoRAs
- However, a model as large as 12B has empirically performed better with **lower learning rates.**
  - LoRA at 1e-3 might totally roast the thing. LoRA at 1e-5 does nearly nothing.
- Ranks as large as 64 through 128 might be undesirable on a 12B model due to general difficulties that scale up with the size of the base model.
  - Try a smaller network first (rank-1, rank-4) and work your way up - they'll train faster, and might do everything you need.
  - If you're finding that it's excessively difficult to train your concept into the model, you might need a higher rank and more regularisation data.
- Other diffusion transformer models like PixArt and SD3 majorly benefit from `--max_grad_norm` and SimpleTuner keeps a pretty high value for this by default on Flux.
  - A lower value would keep the model from falling apart too soon, but can also make it very difficult to learn new concepts that venture far from the base model data distribution. The model might get stuck and never improve.

#### LoKr (--lora_type=lycoris)

- Higher learning rates are better for LoKr (`1e-3` with AdamW, `2e-4` with Lion)
- Other algo need more exploration.
- Setting `is_regularisation_data` on such datasets may help preserve / prevent bleed and improve the final resulting model's quality.
  - This behaves differently from "prior loss preservation" which is known for doubling training batch sizes and not improving the result much
  - SimpleTuner's regularisation data implementation provides an efficient manner of preserving the base model

### Image artifacts

Flux will immediately absorb bad image artifacts. It's just how it is - a final training run on just high quality data may be required to fix it at the end.

When you do these things (among others), some square grid artifacts **may** begin appearing in the samples:

- Overtrain with low quality data
- Use too high of a learning rate
- Overtraining (in general), a low-capacity network with too many images
- Undertraining (also), a high-capacity network with too few images
- Using weird aspect ratios or training data sizes

### Aspect bucketing

- Training for too long on square crops probably won't damage this model too much. Go nuts, it's great and reliable.
- On the other hand, using the natural aspect buckets of your dataset might overly bias these shapes during inference time.
  - This could be a desirable quality, as it keeps aspect-dependent styles like cinematic stuff from bleeding into other resolutions too much.
  - However, if you're looking to improve results equally across many aspect buckets, you might have to experiment with `crop_aspect=random` which comes with its own downsides.
- Mixing dataset configurations by defining your image directory dataset multiple times has produced really good results and a nicely generalised model.

### Training custom fine-tuned Flux models

Some fine-tuned Flux models on Hugging Face Hub (such as Dev2Pro) lack the full directory structure, requiring these specific options be set.

Make sure to set these options `flux_guidance_value`,  `validation_guidance_real` and `flux_attention_masked_training` according to the way the creator did as well if that information is available.

```json
{
    "model_family": "flux",
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_transformer_model_name_or_path": "ashen0209/Flux-Dev2Pro",
    "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_transformer_subfolder": "none",
}
```
