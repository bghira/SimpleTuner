## OmniGen Quickstart

In this example, we'll be training a Lycoris LoKr for OmniGen, focused on general T2I performance improvements (not edit/instruct training at this point).

### Hardware requirements

OmniGen is a pretty modestly sized model around 3.8B parameters; it makes use of the SDXL VAE but does not use a text encoder. Instead, OmniGen uses native token IDs as inputs, and behaves as a multi-modal model.

The memory use during training is not yet known, but can be expected to easily fit on a 24G card with a batch size of 2 or 3. The model can be quantised, saving more VRAM.

OmniGen is a strange architecture relative to other models that are trainable by SimpleTuner;

- Currently, only t2i (text-to-image) training is supported, where the model output is aligned with training prompts and input images.
- An image-to-image training mode is not yet supported, but may be in the future.
  - This mode allows providing a 2nd image as input, and the model will use this as conditioning/reference data for the output.
- The loss value when training OmniGen is very high, and it's not known why this is the case.


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

For Vast, RunPod, and TensorDock (among others), the following will work on a CUDA 12.2-12.8 image:

```bash
apt -y install nvidia-cuda-toolkit libgl1-mesa-glx
```

If `libgl1-mesa-glx` is not found, you might need to use `libgl1-mesa-dri` instead. Your mileage may vary.

### Installation

Clone the SimpleTuner repository and set up the python venv:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

# if python --version shows 3.12 you can just also use the 'python' command here.
python3.12 -m venv .venv

source .venv/bin/activate

pip install -U poetry pip

# Necessary on some systems to prevent it from deciding it knows better than us.
poetry config virtualenvs.create false
```

**Note:** We're currently installing the `release` branch here; the `main` branch may contain experimental features that might have better results or lower memory use.

Depending on your system, you will run one of 3 commands:

```bash
# Linux
poetry install
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

There, you will possibly need to modify the following variables:

- `model_type` - Set this to `full`.
- `model_family` - Set this to `omnigen`.
- `model_flavour` - Set this to `v1`
- `output_dir` - Set this to the directory where you want to store your checkpoints and validation images. It's recommended to use a full path here.
- `train_batch_size` - for a 24G card with full gradient checkpointing, this can be as high as 6.
- `validation_resolution` - This checkpoint for OmniGen is a 1024px model, you should set this to `1024x1024` or one of OmniGen's other supported resolutions.
  - Other resolutions may be specified using commas to separate them: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - Use whatever you are used to selecting at inference time for OmniGen; a lower value around 2.5-3.0 makes more realistic results
- `validation_num_inference_steps` - Use somewhere around 30
- `use_ema` - setting this to `true` will greatly help obtain a more smoothed result alongside your main trained checkpoint.

- `optimizer` - You can use any optimiser you are comfortable and familiar with, but we will use `adamw_bf16` for this example.
- `mixed_precision` - It's recommended to set this to `bf16` for the most efficient training configuration, or `no` (but will consume more memory and be slower).
- `gradient_checkpointing` - Disabling this will go the fastest, but limits your batch sizes. It is required to enable this to get the lowest VRAM usage.

Multi-GPU users can reference [this document](/OPTIONS.md#environment-configuration-variables) for information on configuring the number of GPUs to use.

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

> ℹ️ OmniGen seems to cap out around 122 tokens of understanding. It's not known if it can understand more than this.

#### CLIP score tracking

If you wish to enable evaluations to score the model's performance, see [this document](/documentation/evaluation/CLIP_SCORES.md) for information on configuring and interpreting CLIP scores.

# Stable evaluation loss

If you wish to use stable MSE loss to score the model's performance, see [this document](/documentation/evaluation/EVAL_LOSS.md) for information on configuring and interpreting evaluation loss.

#### Flow schedule shifting

Currently, OmniGen is hard-coded to use its own special flow-matching formulation, and schedule shift will not apply to it.

<!--
Flow-matching models such as OmniGen, Sana, Flux, and SD3 have a property called "shift" that allows us to shift the trained portion of the timestep schedule using a simple decimal value.

##### Auto-shift
A commonly-recommended approach is to follow several recent works and enable resolution-dependent timestep shift, `--flow_schedule_auto_shift` which uses higher shift values for larger images, and lower shift values for smaller images. This results in stable but potentially mediocre training results.

##### Manual specification
_Thanks to General Awareness from Discord for the following examples_

When using a `--flow_schedule_shift` value of 0.1 (a very low value), only the finer details of the image are affected:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

When using a `--flow_schedule_shift` value of 4.0 (a very high value), the large compositional features and potentially colour space of the model becomes impacted:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)
-->

#### Dataset considerations

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively. Note that the bare minimum dataset size is `train_batch_size * gradient_accumulation_steps` as well as more than `vae_batch_size`. The dataset will not be useable if it is too small.

> ℹ️ With few enough images, you might see a message **no images detected in dataset** - increasing the `repeats` value will overcome this limitation.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/ptx0/pseudo-camera-10k) as the dataset.

Create a `--data_backend_config` (`config/multidatabackend.json`) document containing this:

```json
[
  {
    "id": "pseudo-camera-10k-omnigen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/omnigen/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/omnigen/dreambooth-subject",
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
    "cache_dir_vae": "cache/vae/omnigen/dreambooth-subject-512",
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
    "cache_dir": "cache/text/omnigen",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> ℹ️ Running 512px and 1024px datasets concurrently is supported, and could result in better convergence.

> ℹ️ Text encoder embeds are not generated by OmniGen, but one is still required to be defined (for now).

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

From the SimpleTuner directory, one simply has to run:

```bash
./train.sh
```

This will begin the text embed and VAE output caching to disk.

For more information, see the [dataloader](/documentation/DATALOADER.md) and [tutorial](/TUTORIAL.md) documents.

## Notes & troubleshooting tips

### Lowest VRAM config

The lowest VRAM OmniGen configuration is not yet known, but it is expected to be similar to the following:

- OS: Ubuntu Linux 24
- GPU: A single NVIDIA CUDA device (10G, 12G)
- System memory: 50G of system memory approximately
- Base model precision: `int8-quanto` (or `fp8-torchao`, `int8-torchao` all follow similar memory use profiles)
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 1024px
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: disabled / unconfigured
- PyTorch: 2.7+
- Using `--quantize_via=cpu` to avoid outOfMemory error during startup on <=16G cards.
- Enable `--gradient_checkpointing`
- Use a tiny LoRA or Lycoris configuration (eg. LoRA rank 1 or Lokr factor 25)

**NOTE**: Pre-caching of VAE embeds and text encoder outputs may use more memory and still OOM. If so, VAE tiling and slicing can be optionally enabled.

Speed was approximately 3.4 iterations per second on an AMD 7900XTX using Pytorch 2.7 and ROCm 6.3.

### Masked loss

If you are training a subject or style and would like to mask one or the other, see the [masked loss training](/documentation/DREAMBOOTH.md#masked-loss) section of the Dreambooth guide.

### Quantisation

Not tested thoroughly (yet).

### Learning rates

#### LoRA (--lora_type=standard)

*Not supported.*

#### LoKr (--lora_type=lycoris)
- Mild learning rates are better for LoKr (`1e-4` with AdamW, `2e-5` with Lion)
- Other algo need more exploration.
- Setting `is_regularisation_data` has unknown impact/effect with OmniGen (not tested)

### Image artifacts

OmniGen has an unknown response to image artifacts, though it uses the SDXL VAE, and has identical fine-details limitations.

If any image quality issues arise, please open an issue on Github.

### Aspect bucketing

This model has an unknown response to aspect bucketed data. Experimentation will be helpful.

### High loss values

OmniGen has a very high loss value, and it is not known why this is the case. It is recommended to ignore the loss value and focus on the visual quality of the generated images.