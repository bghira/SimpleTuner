# Z-Image [base / turbo] Quickstart

In this example, we'll be training a Z-Image Turbo LoRA. Z-Image is a 6B flow-matching transformer (about half the size of Flux) with base and turbo flavours. Turbo expects an assistant adapter; SimpleTuner can load it automatically.

## Hardware requirements

Z-Image needs less memory than Flux but still benefits from strong GPUs. When you're training every component of a rank-16 LoRA (MLP, projections, transformer blocks), it typically uses:

- ~32-40G VRAM when not quantising the base model
- ~16-24G VRAM when quantising to int8 + bf16 base/LoRA weights
- ~10–12G VRAM when quantising to NF4 + bf16 base/LoRA weights

Additionally, Ramtorch and group offload can be used in attempts to lower VRAM use further. For Multi-GPU users, FSDP2 will allow you to run across many smaller GPUs as well.

You'll need:

- **the absolute minimum** is a single **3080 10G** (with aggressive quantisation/offload)
- **a realistic minimum** is a single 3090/4090 or V100/A6000
- **ideally** multiple 4090, A6000, L40S, or better

Apple GPUs are not recommended for training.

### Memory offloading (optional)

Grouped module offloading dramatically reduces VRAM pressure when you are bottlenecked by the transformer weights. You can enable it by adding the following flags to `TRAINER_EXTRA_ARGS` (or the WebUI Hardware page):

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams are only effective on CUDA; SimpleTuner automatically disables them on ROCm, MPS and CPU backends.
- Do **not** combine this with other CPU offload strategies.
- Group offload is not compatible with Quanto quantisation.
- Prefer a fast local SSD/NVMe target when offloading to disk.

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

For Vast, RunPod, and TensorDock (among others), the following will work on a CUDA 12.x image to enable compiling of CUDA extensions:

```bash
apt -y install nvidia-cuda-toolkit-12-8
```

## Installation

Install SimpleTuner via pip:

```bash
pip install simpletuner[cuda]
```

For manual installation or development setup, see the [installation documentation](/documentation/INSTALL.md).

### AMD ROCm follow-up steps

The following must be executed for an AMD MI300X to be usable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## Setting up the environment

### Web interface method

The SimpleTuner WebUI makes setup straightforward. To run the server:

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
- `model_family` - Set this to `z-image`.
- `model_flavour` - set to `turbo` (or `turbo-ostris-v2` for the v2 assistant adapter); the base flavour points to a currently-unavailable checkpoint.
- `pretrained_model_name_or_path` - Set this to `TONGYI-MAI/Z-Image-Turbo`.
- `output_dir` - Set this to the directory where you want to store your checkpoints and validation images. It's recommended to use a full path here.
- `train_batch_size` - keep at 1, especially if you have a very small dataset.
- `validation_resolution` - Z-Image is 1024px; use `1024x1024` or multi-aspect buckets: `1024x1024,1280x768,2048x2048`.
- `validation_guidance` - Low guidance (0–1) is typical for Z-Image Turbo, but the base flavour requires a range between 4-6.
- `validation_num_inference_steps` - Turbo requires just 8, but Base can get by with around 50-100.
- `--lora_rank=4` if you wish to substantially reduce the size of the LoRA being trained. This can help with VRAM use.
- For turbo, supply the assistant adapter (see below) or disable it explicitly.

- `gradient_accumulation_steps` - increases runtime linearly; use if you need VRAM relief.
- `optimizer` - Beginners are recommended to stick with adamw_bf16, though other adamw/lion variants are also good choices.
- `mixed_precision` - `bf16` on modern GPUs; `fp16` otherwise.
- `gradient_checkpointing` - set this to true in practically every situation on every device.
- `gradient_checkpointing_interval` - can be set to 2+ on larger GPUs to checkpoint every _n_ blocks.

### Advanced Experimental Features

SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> ⚠️ These features increase the computational overhead of training.

### Assistant LoRA (Turbo)

Turbo expects an assistant adapter:

- `assistant_lora_path`: `ostris/zimage_turbo_training_adapter`
- `assistant_lora_weight_name`:
  - `turbo`: `zimage_turbo_training_adapter_v1.safetensors`
  - `turbo-ostris-v2`: `zimage_turbo_training_adapter_v2.safetensors`

SimpleTuner auto-fills these for turbo flavours unless you override them. Disable with `--disable_assistant_lora` if you accept the quality hit.

### Validation prompts

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

A set of diverse prompts will help determine whether the model is collapsing as it trains. In this example, the word `<token>` should be replaced with your subject name (instance_prompt).

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

> ℹ️ Z-Image is a flow-matching model and shorter prompts that have strong similarities will result in practically the same image being produced. Use longer, more descriptive prompts.

### CLIP score tracking

If you wish to enable evaluations to score the model's performance, see [this document](/documentation/evaluation/CLIP_SCORES.md) for information on configuring and interpreting CLIP scores.

### Stable evaluation loss

If you wish to use stable MSE loss to score the model's performance, see [this document](/documentation/evaluation/EVAL_LOSS.md) for information on configuring and interpreting evaluation loss.

### Validation previews

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

Set `validation_preview_steps` to a higher value (e.g., 3 or 5) to reduce Tiny AutoEncoder overhead.

### Flow schedule shifting (flow matching)

Flow-matching models such as Z-Image have a "shift" parameter to move the trained portion of the timestep schedule. Auto-shift based on resolution is a safe default. Manually increasing shift moves learning toward coarse features; reducing it biases fine details. For the turbo model, it's possible that modifying these values may harm the model.

### Quantised model training

TorchAO or other quantisation can reduce precision and VRAM requirements - Optimum Quanto is now on life support, but is also available.

For `config.json` users:

```json
  "base_model_precision": "int8-torchao",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```

### Dataset considerations

> ⚠️ Image quality for training is critical; Z-Image will absorb artifacts early. A final pass on high-quality data may be required.

Keep your dataset large enough (at least `train_batch_size * gradient_accumulation_steps`, and more than `vae_batch_size`). Increase `repeats` if you see **no images detected in dataset**.

Example multi-backend config (`config/multidatabackend.json`):

```json
[
  {
    "id": "pseudo-camera-10k-zimage",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zimage/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/zimage/dreambooth-subject",
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
    "cache_dir_vae": "cache/vae/zimage/dreambooth-subject-512",
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
    "cache_dir": "cache/text/zimage",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

Running 512px and 1024px datasets concurrently is supported and can improve convergence.

Create the datasets directory:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

### Login to WandB and Huggingface Hub

Log in before training, especially if you're using `--push_to_hub` and `--report_to=wandb`:

```bash
wandb login
huggingface-cli login
```

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

## Multi-GPU Configuration

SimpleTuner includes **automatic GPU detection** through the WebUI. During onboarding, you'll configure:

- **Auto Mode**: Automatically uses all detected GPUs with optimal settings
- **Manual Mode**: Select specific GPUs or set custom process count
- **Disabled Mode**: Single GPU training

The WebUI detects your hardware and configures `--num_processes` and `CUDA_VISIBLE_DEVICES` automatically.

For manual configuration or advanced setups, see the [Multi-GPU Training section](/documentation/INSTALL.md#multiple-gpu-training) in the installation guide.

## Inference tips

### Guidance settings

Z-Image is flow-matching; lower guidance values (around 0–1) tend to preserve quality and diversity. If you train with higher guidance vectors, ensure your inference pipeline supports CFG and expect slower generation or higher VRAM use with batched CFG.

## Notes & troubleshooting tips

### Lowest VRAM config

- GPU: a single NVIDIA CUDA device (10–12G) with aggressive quantisation/offload
- System memory: ~32–48G
- Base model precision: `nf4-bnb` or `int8`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged` or adamw variants
- Resolution: 512px (1024px requires more VRAM)
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: disabled / unconfigured
- Use `--quantize_via=cpu` if startup OOMs on <=16G cards
- Enable `--gradient_checkpointing`
- Enable Ramtorch or group offload

The pre-caching stage can run out of memory; Text encoder quantisation and VAE tiling can be enabled via `--text_encoder_precision=int8-torchao` and `--vae_enable_tiling=true`. Further memory can be saved on startup with `--offload_during_startup=true`, which will keep only the text encoder or VAE loaded, and not both.

### Quantisation

- Minimum 8bit quantisation is often required for a 16G card to train this model.
- Quantising the model to 8bit generally doesn't harm training and allows higher batch sizes.
- **int8** benefits from hardware acceleration; **nf4-bnb** reduces VRAM further but is more sensitive.
- When loading the LoRA later, you **should ideally** use the same base model precision as you trained with.

### Aspect bucketing

- Training only on square crops generally works, but multi-aspect buckets can improve generalisation.
- Using natural aspect buckets can bias shapes; random cropping can help if you need broader coverage.
- Mixing dataset configurations by defining your image directory multiple times has produced good generalisation.

### Learning rates

#### LoRA (--lora_type=standard)

- Lower learning rates often behave better on large transformers.
- Start with modest ranks (4–16) before trying very high ranks.
- Reduce `max_grad_norm` if the model destabilises; increase if learning stalls.

#### LoKr (--lora_type=lycoris)

- Higher learning rates (e.g., `1e-3` with AdamW, `2e-4` with Lion) can work well; tune to taste.
- Mark regularisation datasets with `is_regularisation_data` to help preserve the base model.

### Image artifacts

Z-Image will absorb bad image artifacts early. A final pass on high-quality data may be required to clean up. Watch for grid artifacts if learning rate is too high, data is low quality, or aspect handling is off.

### Training custom fine-tuned Z-Image models

Some fine-tuned checkpoints may lack full directory structure. Set these fields appropriately if needed:

```json
{
    "model_family": "z-image",
    "pretrained_model_name_or_path": "TONGYI-MAI/Z-Image-Base",
    "pretrained_transformer_model_name_or_path": "your-custom-transformer",
    "pretrained_vae_model_name_or_path": "TONGYI-MAI/Z-Image-Base",
    "pretrained_transformer_subfolder": "none"
}
```

## Troubleshooting

- OOM at startup: enable group offload (not with Quanto), lower LoRA rank, or quantise (`--base_model_precision int8`/`nf4`).
- Blurry outputs: increase `validation_num_inference_steps` (e.g., 24–28) or raise guidance toward 1.0.
- Artifacts/overfitting: reduce rank or learning rate, add more diverse prompts, or shorten training.
- Assistant adapter issues: turbo expects the adapter path/weight; only disable if you accept quality loss.
- Slow validations: trim validation resolutions or steps; flow matching converges quickly.
