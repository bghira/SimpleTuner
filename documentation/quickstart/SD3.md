## Stable Diffusion 3

In this example, we'll be training a Stable Diffusion 3 model using the SimpleTuner toolkit and will be using the `lora` model type.

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

# Necessary on some systems to prevent it from deciding it knows better than us.
poetry config virtualenvs.create false
```

Depending on your system, you will run one of 3 commands:

```bash
# MacOS
poetry install -C install/apple

# Linux
poetry install

# Linux with ROCM
poetry install -C install/rocm
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

#### Quantised model training

Tested on Apple and NVIDIA systems, Hugging Face Optimum-Quanto can be used to reduce the precision and VRAM requirements well below the requirements of base SDXL training.

Inside your SimpleTuner venv:

```bash
pip install optimum-quanto
```

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

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/ptx0/pseudo-camera-10k) as the dataset.

In your `/home/user/simpletuner/config` directory, create a multidatabackend.json:

```json
[
  {
    "id": "pseudo-camera-10k-sd3",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1.0,
    "minimum_image_size": 0,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/sd3/pseudo-camera-10k",
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

For more information, see the [dataloader](/documentation/DATALOADER.md) and [tutorial](/TUTORIAL.md) documents.

## Notes & troubleshooting tips

### Model instability

Unlike what StabilityAI employees reported on Twitter, the SD 3.5 Large 8B model has noted instabilities during training:

- The issue is compounded by high `--max_grad_norm` values, and hardly prevented by low values
- Learning rates are extremely sensitive; `1e-5` works with StableAdamW but `4e-5` led to exploding gradients in no time
- Higher batch sizes help **a lot**
- The stability is not impacted by switching to BF16 / disabling quantisation
- LyCORIS seems to bring this on a lot more readily than standard LoRA
  - Reducing the model target to just Attention seemed to help, but not solve the stability issues
- Standard LoRA did not learn the likeness of characters or styles
  - And PEFT LoRA does not function with multiGPU when quantising

Official training code was not released alongside SD3.5, leaving developers to guess how to implement the training loop based on the [SD3.5 repository contents](https://github.com/stabilityai/sd3.5) which leaves us with possibly subpar results.

Some things were attempted to resolve the issue;
- Excluding more layers from quantisation
- Zeroing T5 embed padding space vs not zeroing it
- Zeroing embeds for dropout vs using encoded space
- Augmenting the training loss target to mimic the SD3.5 repo method
- Investigating schedule shift values, updating the default
- Downgrading pytorch from 2.5 to 2.4.1
- Upgrading all dependencies

Unfortunately, not much has helped. The resulting most stable configuration (which might end up not learning much):

- optimizer=optimi-stableadamw
- learning_rate=1e-5
- batch_size=4 * 3 GPUs
- max_grad_norm=0.01
- base_model_precision=int8-quanto
- No loss masking or dataset regularisation, as their contribution to this instability is unknown

### Lowest VRAM config

- OS: Ubuntu Linux 24
- GPU: A single NVIDIA CUDA device (10G, 12G)
- System memory: 50G of system memory approximately
- Base model precision: `bnb-nf4`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 512px
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: disabled / unconfigured
- PyTorch: 2.5

### Masked loss

If you are training a subject or style and would like to mask one or the other, see the [masked loss training](/documentation/DREAMBOOTH.md#masked-loss) section of the Dreambooth guide.

### Regularisation data

For more information on regularisation datasets, see [this section](/documentation/DREAMBOOTH.md#prior-preservation-loss) and [this section](/documentation/DREAMBOOTH.md#regularisation-dataset-considerations) of the Dreambooth guide.

### Quantised training

See [this section](/documentation/DREAMBOOTH.md#quantised-model-training-loralycoris-only) of the Dreambooth guide for information on configuring quantisation for SD3 and other models.