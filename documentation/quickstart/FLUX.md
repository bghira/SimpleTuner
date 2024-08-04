## Flux[dev] / Flux[schnell] Quickstart

In this example, we'll be training a Flux.1 LoRA model using the SimpleTuner toolkit.

### Hardware requirements

When you're training every component of the model, a rank-16 LoRA ends up using a bit more than 40GB of VRAM for training.

You'll need at minimum, a single A40 GPU, or, ideally, multiple A6000s. Luckily, these are readily available through providers such as TensorDock for extremely low rates (<$2/hr).

**Unlike other models, AMD and Apple GPUs do not work for training Flux.**

### Prerequisites

Make sure that you have python installed. You can check this by running:

```bash
python --version
```

### Installation

Clone the SimpleTuner repository and set up the python venv:

```bash
git clone --branch=main https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

python -m venv .venv

source .venv/bin/activate

pip install -U poetry pip
```

**Note:** We're currently relying on the `main` branch here, but after the next release, we'll use the `release` branch instead.

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

For LoRA support in Diffusers, the current main branch does not yet have this merged in.

To obtain the correct build, run the following commands:

```bash
pip uninstall diffusers
pip install git+https://github.com/huggingface/diffusers@lora-support-flux
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
- `TRAINER_EXTRA_ARGS` - Here, you can place `--lora_rank=4` if you wish to substantially reduce the size of the LoRA being trained. This can help with VRAM use.


#### Dataset considerations

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
    "resolution": 1.0,
    "minimum_image_size": 0.5,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
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