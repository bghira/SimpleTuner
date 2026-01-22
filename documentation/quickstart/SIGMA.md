## PixArt Sigma Quickstart

In this example, we'll be training a PixArt Sigma model using the SimpleTuner toolkit and will be using the `full` model type, as it being a smaller model will likely fit in VRAM.

### Prerequisites

Make sure that you have python installed; SimpleTuner does well with 3.10 through 3.13.

You can check this by running:

```bash
python --version
```

If you don't have python 3.13 installed on Ubuntu, you can try the following:

```bash
apt -y install python3.13 python3.13-venv
```

#### Container image dependencies

For Vast, RunPod, and TensorDock (among others), the following will work on a CUDA 12.2-12.8 image to enable compiling of CUDA extensions:

```bash
apt -y install nvidia-cuda-toolkit
```

### Installation

Install SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

For manual installation or development setup, see the [installation documentation](../INSTALL.md).

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

<details>
<summary>View example config</summary>

```json
{
  "model_type": "full",
  "use_bitfit": false,
  "pretrained_model_name_or_path": "pixart-alpha/pixart-sigma-xl-2-1024-ms",
  "model_family": "pixart_sigma",
  "output_dir": "/home/user/output/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.5
}
```
</details>

- `pretrained_model_name_or_path` - Set this to `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`.
- `MODEL_TYPE` - Set this to `full`.
- `USE_BITFIT` - Set this to `false`.
- `MODEL_FAMILY` - Set this to `pixart_sigma`.
- `OUTPUT_DIR` - Set this to the directory where you want to store your checkpoints and validation images. It's recommended to use a full path here.
- `VALIDATION_RESOLUTION` - As PixArt Sigma comes in a 1024px or 2048xp model format, you should carefully set this to `1024x1024` for this example.
  - Additionally, PixArt was fine-tuned on multi-aspect buckets, and other resolutions may be specified using commas to separate them: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - PixArt benefits from a very-low value. Set this between `3.6` to `4.4`.

There are a few more if using a Mac M-series machine:

- `mixed_precision` should be set to `no`.

> 💡 **Tip:** For large datasets where disk space is a concern, you can use `--vae_cache_disable` to perform online VAE encoding without caching the results to disk.

#### Dataset considerations

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively. Note that the bare minimum dataset size is `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`. The dataset will not be discoverable by the trainer if it is too small.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) as the dataset.

In your `/home/user/simpletuner/config` directory, create a multidatabackend.json:

<details>
<summary>View example config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-pixart",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 1.0,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/pixart/pseudo-camera-10k",
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
    "cache_dir": "cache/text/pixart/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

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

For more information, see the [dataloader](../DATALOADER.md) and [tutorial](../TUTORIAL.md) documents.

### CLIP score tracking

If you wish to enable evaluations to score the model's performance, see [this document](../evaluation/CLIP_SCORES.md) for information on configuring and interpreting CLIP scores.

# Stable evaluation loss

If you wish to use stable MSE loss to score the model's performance, see [this document](../evaluation/EVAL_LOSS.md) for information on configuring and interpreting evaluation loss.

#### Validation previews

SimpleTuner supports streaming intermediate validation previews during generation using Tiny AutoEncoder models. This allows you to see validation images being generated step-by-step in real-time via webhook callbacks.

To enable:
<details>
<summary>View example config</summary>

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```
</details>

**Requirements:**
- Webhook configuration
- Validation enabled

Set `validation_preview_steps` to a higher value (e.g., 3 or 5) to reduce Tiny AutoEncoder overhead. With `validation_num_inference_steps=20` and `validation_preview_steps=5`, you'll receive preview images at steps 5, 10, 15, and 20.

### SageAttention

When using `--attention_mechanism=sageattention`, inference can be sped-up at validation time.

**Note**: This isn't compatible with _every_ model configuration, but it's worth trying.

### Advanced Experimental Features

<details>
<summary>Show advanced experimental details</summary>


SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.
*   **[Diff2Flow](../experimental/DIFF2FLOW.md):** allows training with a Flow Matching objective, potentially improving generation straightness and quality.

> ⚠️ These features increase the computational overhead of training.
</details>
