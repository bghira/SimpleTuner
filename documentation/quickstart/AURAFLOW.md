## AuraFlow v0.1

In this example, we'll be running a **full fine-tune** on an AuraFlow model using the SimpleTuner toolkit.

> ⚠️ LoRA is not currently supported for AuraFlow, this will require more resources.

### Prerequisites

Make sure that you have python installed. You can check this by running:

```bash
python --version
```

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

Additionally, because of AuraFlow's preliminary support in the Diffusers project, you will have to manually install a fork that contains the AuraFlow patches already-integrated.

This command should be executed while you are still inside your venv from earlier:

```bash
pip install git+https://github.com/bghira/diffusers@feature/lavender-flow-complete
```

For your own security, you may audit the changes between this branch and upstream Diffusers repository [here](https://github.com/bghira/diffusers/tree/feature/lavender-flow-complete).

This page will be updated after full support lands in the Diffusers project, negating the requirement for this part to be done manually. That progress can be viewed [here](https://github.com/huggingface/diffusers/pull/8796). If that issue has been closed and this page has not yet been updated, please take the time to open an issue report on GitHub [here](https://github.com/bghira/SimpleTuner/isssues).

### Setting up the environment

To run SimpleTuner, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

#### Configuration file

Copy `sdxl-env.sh.example` to `sdxl-env.sh`:

```bash
cp sdxl-env.sh.example sdxl-env.sh
```

There, you will need to modify the following variables:

- `MODEL_TYPE` - This should remain set as `full`. **LoRA will not work.**
- `AURA_FLOW` - Set this to `true`.
- `MODEL_NAME` - Set this to `AuraDiffusion/auradiffusion-v0.1a0`. Note that you will need to log in to Huggingface and be granted access to download this model. We will go over logging in to Huggingface later in this tutorial.
- `BASE_DIR` - Set this to the directory where you want to store your outputs and datasets. It's recommended to use a full path here.

There are a few more if using a Mac M-series machine:

- `MIXED_PRECISION` should be set to `no`.
- `USE_XFORMERS` should be set to `false`.

#### Dataset considerations

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively. Note that the bare minimum dataset size is `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`. The dataset will not be useable if it is too small.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/ptx0/pseudo-camera-10k) as the dataset.

In your `BASE_DIR` directory, create a multidatabackend.json:

```json
[
  {
    "id": "pseudo-camera-10k-aura",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 0.5,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/aura/pseudo-camera-10k",
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
    "cache_dir": "cache/text/aura/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

Then, navigate to the `BASE_DIR` directory and create a `datasets` directory:

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
bash train_sdxl.sh
```

This will begin the text embed and VAE output caching to disk.

For more information, see the [dataloader](/documentation/DATALOADER.md) and [tutorial](/TUTORIAL.md) documents.
