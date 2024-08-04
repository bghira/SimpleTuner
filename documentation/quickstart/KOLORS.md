## Kwai Kolors Quickstart

In this example, we'll be training a Kwai Kolors model using the SimpleTuner toolkit and will be using the `lora` model type.

Kolors is roughly the same size as SDXL, so you can try `full` training, but the changes for that are not described in this quickstart guide.

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

### Setting up the environment

To run SimpleTuner, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

#### Configuration file

Copy `config/config.env.example` to `config/config.env`:

```bash
cp config/config.env.example config/config.env
```

There, you will need to modify the following variables:

- `MODEL_TYPE` - Set this to `lora`.
- `USE_DORA` - Set this to `true` if you wish to train DoRA.
- `KOLORS` - Set this to `true`.
- `MODEL_NAME` - Set this to `Kwai-Kolors/Kolors-diffusers`.
- `OUTPUT_DIR` - Set this to the directory where you want to store your outputs and datasets. It's recommended to use a full path here.
- `VALIDATION_RESOLUTION` - Set this to `1024x1024` for this example.
  - Additionally, Kolors was fine-tuned on multi-aspect buckets, and other resolutions may be specified using commas to separate them: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - Use whatever value you are comfortable with for testing at inference time. Set this between `4.2` to `6.4`.
- `USE_GRADIENT_CHECKPOINTING` - This should probably be `true` unless you have a LOT of VRAM and want to sacrifice some to make it go faster.
- `LEARNING_RATE` - `1e-4` is fairly common for low-rank networks, though `1e-5` might be a more conservative choice if you notice any "burning" or early overtraining.

There are a few more if using a Mac M-series machine:

- `MIXED_PRECISION` should be set to `no`.
- `USE_XFORMERS` should be set to `false`.

#### Quantised model training

Tested on Apple and NVIDIA systems, Hugging Face Optimum-Quanto can be used to reduce the precision and VRAM requirements of especially ChatGLM 6B (the text encoder).

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
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --text_encoder_1_precision=no_change"

# When you're quantising the model, we're not in pure bf16 anymore.
# Since adamw_bf16 will never work with this setup, select another optimiser.
# I know the spelling is different than everywhere else, but we're in too deep to fix it now.
export OPTIMIZER="adafactor" # or maybe prodigy
```

#### Dataset considerations

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively. Note that the bare minimum dataset size is `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`. The dataset will not be discoverable by the trainer if it is too small.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/ptx0/pseudo-camera-10k) as the dataset.

In your `OUTPUT_DIR` directory, create a multidatabackend.json:

```json
[
  {
    "id": "pseudo-camera-10k-kolors",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 1.0,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/kolors/pseudo-camera-10k",
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
    "cache_dir": "cache/text/kolors/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

Then, navigate to the `OUTPUT_DIR` directory and create a `datasets` directory:

```bash
mkdir -p datasets
huggingface-cli download --repo-type=dataset ptx0/pseudo-camera-10k --local-dir=datasets/pseudo-camera-10k
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
