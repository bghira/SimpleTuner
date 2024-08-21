## PixArt Sigma Quickstart

In this example, we'll be training a PixArt Sigma model using the SimpleTuner toolkit and will be using the `full` model type, as it being a smaller model will likely fit in VRAM.

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

python -m venv .venv

source .venv/bin/activate

pip install -U poetry pip
```

Depending on your system, you will run one of 3 commands:

```bash
# MacOS
poetry install --no-root -C install/apple

# Linux
poetry install --no-root

# Linux with ROCM
poetry install --no-root -C install/rocm
```

#### AMD ROCm follow-up steps

The following must be executed for an AMD MI300X to be useable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

#### Removing DeepSpeed & Bits n Bytes

These two dependencies cause numerous issues for container hosts such as RunPod and Vast.

To remove them after `poetry` has installed them, run the following command in the same terminal:

```bash
pip uninstall -y deepspeed bitsandbytes
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

- `MODEL_TYPE` - Set this to `full`.
- `USE_BITFIT` - Set this to `false`.
- `PIXART_SIGMA` - Set this to `true`.
- `MODEL_NAME` - Set this to `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`.
- `OUTPUT_DIR` - Set this to the directory where you want to store your outputs and datasets. It's recommended to use a full path here.
- `VALIDATION_RESOLUTION` - As PixArt Sigma comes in a 1024px or 2048xp model format, you should carefully set this to `1024x1024` for this example.
  - Additionally, PixArt was fine-tuned on multi-aspect buckets, and other resolutions may be specified using commas to separate them: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - PixArt benefits from a very-low value. Set this between `3.6` to `4.4`.

There are a few more if using a Mac M-series machine:

- `MIXED_PRECISION` should be set to `no`.
- `USE_XFORMERS` should be set to `false`.

#### Dataset considerations

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively. Note that the bare minimum dataset size is `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`. The dataset will not be discoverable by the trainer if it is too small.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/ptx0/pseudo-camera-10k) as the dataset.

In your `OUTPUT_DIR` directory, create a multidatabackend.json:

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
    "cache_dir": "cache/text/pixart/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

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
