## Stable Diffusion 3

In this example, we'll be training a Stable Diffusion 3 model using the SimpleTuner toolkit and will be using the `lora` model type.

### Prerequisites

Make sure that you have python installed; SimpleTuner does well with 3.10 through 3.12.

You can check this by running:

```bash
python --version
```

If you don't have python 3.12 installed on Ubuntu, you can try the following:

```bash
apt -y install python3.12 python3.12-venv
```

#### Container image dependencies

For Vast, RunPod, and TensorDock (among others), the following will work on a CUDA 12.2-12.8 image to enable compiling of CUDA extensions:

```bash
apt -y install nvidia-cuda-toolkit
```

### Installation

Install SimpleTuner via pip:

```bash
pip install simpletuner[cuda]
```

For manual installation or development setup, see the [installation documentation](/documentation/INSTALL.md).

#### AMD ROCm follow-up steps

The following must be executed for an AMD MI300X to be useable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

### Setting up the environment

To run SimpleTuner, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

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

There, you will need to modify the following variables:

```json
{
  "model_type": "lora",
  "model_family": "sd3",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
  "output_dir": "/home/user/outputs/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.0,
  "validation_prompt": "your main test prompt here",
  "user_prompt_library": "config/user_prompt_library.json"
}
```


- `pretrained_model_name_or_path` - Set this to `stabilityai/stable-diffusion-3.5-large`. Note that you will need to log in to Huggingface and be granted access to download this model. We will go over logging in to Huggingface later in this tutorial.
  - If you prefer to train the older SD3.0 Medium (2B), use `stabilityai/stable-diffusion-3-medium-diffusers` instead.
- `MODEL_TYPE` - Set this to `lora`.
- `MODEL_FAMILY` - Set this to `sd3`.
- `OUTPUT_DIR` - Set this to the directory where you want to store your checkpoints and validation images. It's recommended to use a full path here.
- `VALIDATION_RESOLUTION` - As SD3 is a 1024px model, you can set this to `1024x1024`.
  - Additionally, SD3 was fine-tuned on multi-aspect buckets, and other resolutions may be specified using commas to separate them: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - SD3 benefits from a very-low value. Set this to `3.0`.

There are a few more if using a Mac M-series machine:

- `mixed_precision` should be set to `no`.

### Advanced Experimental Features

SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> ⚠️ These features increase the computational overhead of training.

#### Quantised model training

Tested on Apple and NVIDIA systems, Hugging Face Optimum-Quanto can be used to reduce the precision and VRAM requirements well below the requirements of base SDXL training.



> ⚠️ If using a JSON config file, be sure to use this format in `config.json` instead of `config.env`:

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "text_encoder_3_precision": "no_change",
  "optimizer": "adamw_bf16"
}
```

For `config.env` users (deprecated):

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

# When you're quantising the model, --base_model_default_dtype is set to bf16 by default. This setup requires adamw_bf16, but saves the most memory.
# adamw_bf16 only supports bf16 training, but any other optimiser will support both bf16 or fp32 training precision.
export OPTIMIZER="adamw_bf16"
```

#### Dataset considerations

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively. Note that the bare minimum dataset size is `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS` as well as more than `VAE_BATCH_SIZE`. The dataset will not be useable if it is too small.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) as the dataset.

In your `/home/user/simpletuner/config` directory, create a multidatabackend.json:

```json
[
  {
    "id": "pseudo-camera-10k-sd3",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 0,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/home/user/simpletuner/output/cache/vae/sd3/pseudo-camera-10k",
    "instance_data_dir": "/home/user/simpletuner/datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery"
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

Then, create a `datasets` directory:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
popd
```

This will download about 10k photograph samples to your `datasets/pseudo-camera-10k` directory, which will be automatically created for you.

#### Login to WandB and Huggingface Hub

You'll want to login to WandB and HF Hub before beginning training, especially if you're using `push_to_hub: true` and `--report_to=wandb`.

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

For more information, see the [dataloader](/documentation/DATALOADER.md) and [tutorial](/documentation/TUTORIAL.md) documents.

## Notes & troubleshooting tips

### Skip-layer guidance (SD3.5 Medium)

StabilityAI recommends enabling SLG (Skip-layer guidance) on SD 3.5 Medium inference. This doesn't impact training results, only the validation sample quality.

The following values are recommended for `config.json`:

```json
{
  "--validation_guidance_skip_layers": [7, 8, 9],
  "--validation_guidance_skip_layers_start": 0.01,
  "--validation_guidance_skip_layers_stop": 0.2,
  "--validation_guidance_skip_scale": 2.8,
  "--validation_guidance": 4.0,
  "--flow_use_uniform_schedule": true,
  "--flow_schedule_auto_shift": true
}
```

- `..skip_scale` determines how much to scale the positive prompt prediction during skip-layer guidance. The default value of 2.8 is safe for the base model's skip value of `7, 8, 9` but will need to be increased if more layers are skipped, doubling it for each additional layer.
- `..skip_layers` tells which layers to skip during the negative prompt prediction.
- `..skip_layers_start` determine the fraction of the inference pipeline during which skip-layer guidance should begin to be applied.
- `..skip_layers_stop` will set the fraction of the total number of inference steps after which SLG will no longer be applied.

SLG can be applied for fewer steps for a weaker effect or less reduction of inference speed.

It seems that extensive training of a LoRA or LyCORIS model will require modification to these values, though it's not clear how exactly it changes.

**Lower CFG must be used during inference.**

### Model instability

The SD 3.5 Large 8B model has potential instabilities during training:

- High `--max_grad_norm` values will allow the model to explore potentially dangerous weight updates
- Learning rates can be extremely sensitive; `1e-5` works with StableAdamW but `4e-5` could explode
- Higher batch sizes help **a lot**
- The stability is not impacted by disabling quantisation or training in pure fp32

Official training code was not released alongside SD3.5, leaving developers to guess how to implement the training loop based on the [SD3.5 repository contents](https://github.com/stabilityai/sd3.5).

Some changes were made to SimpleTuner's SD3.5 support:
- Excluding more layers from quantisation
- No longer zeroing T5 padding space by default (`--t5_padding`)
- Offering a switch (`--sd3_clip_uncond_behaviour` and `--sd3_t5_uncond_behaviour`) to use empty encoded blank captions for unconditional predictions (`empty_string`, **default**) or zeros (`zero`), not a recommended setting to tweak.
- SD3.5 training loss function was updated to match that found in the upstream StabilityAI/SD3.5 repository
- Updated default `--flow_schedule_shift` value to 3 to match the static 1024px value for SD3
  - StabilityAI followed-up with documentation to use `--flow_schedule_shift=1` with `--flow_use_uniform_schedule`
  - Community members have reported that `--flow_schedule_auto_shift` works better when using mult-aspect or multi-resolution training
- Updated the hard-coded tokeniser sequence length limit to **154** with the option to revert it to **77** tokens to save disk space or compute at the cost of output quality degradation


#### Stable configuration values

These options have been known to keep SD3.5 in-tact for as long as possible:
- optimizer=adamw_bf16
- flow_schedule_shift=1
- learning_rate=1e-4
- batch_size=4 * 3 GPUs
- max_grad_norm=0.1
- base_model_precision=int8-quanto
- No loss masking or dataset regularisation, as their contribution to this instability is unknown
- `validation_guidance_skip_layers=[7,8,9]`

### Lowest VRAM config

- OS: Ubuntu Linux 24
- GPU: A single NVIDIA CUDA device (10G, 12G)
- System memory: 50G of system memory approximately
- Base model precision: `nf4-bnb`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 512px
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: disabled / unconfigured
- PyTorch: 2.5

### SageAttention

When using `--attention_mechanism=sageattention`, inference can be sped-up at validation time.

**Note**: This isn't compatible with _every_ model configuration, but it's worth trying.

### Masked loss

If you are training a subject or style and would like to mask one or the other, see the [masked loss training](/documentation/DREAMBOOTH.md#masked-loss) section of the Dreambooth guide.

### Regularisation data

For more information on regularisation datasets, see [this section](/documentation/DREAMBOOTH.md#prior-preservation-loss) and [this section](/documentation/DREAMBOOTH.md#regularisation-dataset-considerations) of the Dreambooth guide.

### Quantised training

See [this section](/documentation/DREAMBOOTH.md#quantised-model-training-loralycoris-only) of the Dreambooth guide for information on configuring quantisation for SD3 and other models.

### CLIP score tracking

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
