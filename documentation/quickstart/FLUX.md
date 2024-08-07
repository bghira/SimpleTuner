## Flux[dev] / Flux[schnell] Quickstart

![image](https://github.com/user-attachments/assets/6409d790-3bb4-457c-a4b4-a51a45fc91d1)

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
- `TRAINER_EXTRA_ARGS` - Here, you can place `--lora_rank=4` if you wish to substantially reduce the size of the LoRA being trained. This can help with VRAM use.

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

# When you're quantising the model, --base_model_default_dtype is set to bf16 by default. This setup requires adamw_bf16, but saves the most memory.
# If you'd like to use another optimizer, you can override this with --base_model_default_dtype=fp32.
# option one:
export OPTIMIZER="adamw_bf16" # or maybe prodigy
# option two:
#export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --base_model_default_dtype=fp32"
#export OPTIMIZER="adafactor" # or maybe prodigy

```


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

## Notes & troubleshooting tips

- Schnell training really needs a bit more time in the oven - currently, the results do not look good
- Dev LoRAs run just fine on Schnell
- Dev+Schnell merge 50/50 just fine, and the LoRAs can possibly be trained from that, which will then run on Schnell **or** Dev
- A model as large as 12B has empirically performed better with lower learning rates.
  - LoRA at 1e-4 might totally roast the thing. LoRA at 1e-7 does nearly nothing.
- Minimum 8bit quantisation is required for a 24G card to train this model - but 32G (V100) cards suffer a more tragic fate.
  - Without quantising the model, a rank-1 LoRA sits at just over 32GB of mem use, in a way that prevents a 32G V100 from actually working
  - Adafactor works, reducing VRAM to ~24G or further with sub-1024x1024 training
- Quantising the model isn't a bad thing
  - It allows you to push higher batch sizes and possibly obtain a better result
  - It unlocks the non-bf16 optimisers for use, such as Prodigy, Adafactor, Dadaptation, AdamW, and AdamW8Bit
- As usual, **fp8 quantisation runs more slowly** than **int8** and might have a worse result due to the use of `e4m3fn` in Quanto
  - fp16 training similarly is bad for Flux; this model wants the range of bf16
  - `e5m2` level precision is better at fp8 but haven't looked into how to enable it yet. Sorry, H100 owners. We weep for you.
- Larger rank models might be undesirable on a 12B model due to the general training dynamics of large models.
  - Try a smaller network first (rank-1, rank-4) and work your way up - they'll train faster, and might do everything you need.
- When you do these things (among others), some square grid artifacts **may** begin appearing in the samples:
  - Overtrain with low quality data
  - Use too high of a learning rate
  - Select a bad optimiser
  - Overtraining (in general), a low-capacity network with too many images
  - Undertraining (also), a high-capacity network with too few images
  - Using weird aspect ratios or training data sizes
- Training for too long on square crops probably won't damage this model. Go nuts, it's great and reliable.
- We're overriding `--max_grad_norm` on all DiT models currently - providing the flag `--i_know_what_im_doing` will allow you to bypass this limit and experiment with higher gradient norm scales
  - The low value keeps the model from falling apart too soon, but can also make it very difficult to learn new concepts that venture far from the base model data distribution
