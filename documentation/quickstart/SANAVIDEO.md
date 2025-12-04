## Sana Video Quickstart

In this example, we'll be training the Sana Video 2B 480p model.

### Hardware requirements

Sana Video uses the Wan autoencoder and processes 81-frame sequences at 480p by default. Expect memory use on par with other video models; enable gradient checkpointing early and scale `train_batch_size` up only after verifying VRAM headroom.

### Memory offloading (optional)

If you are close to the VRAM limit, enable grouped offloading in your config:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- CUDA users benefit from `--group_offload_use_stream`; other backends ignore it automatically.
- Skip `--group_offload_to_disk_path` unless system RAM is limited — disk staging is slower but keeps runs stable.
- Disable `--enable_model_cpu_offload` when using group offloading.

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

There, you will possibly need to modify the following variables:

- `model_type` - Set this to `full`.
- `model_family` - Set this to `sanavideo`.
- `pretrained_model_name_or_path` - Set this to `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`.
- `pretrained_vae_model_name_or_path` - Set this to `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`.
- `output_dir` - Set this to the directory where you want to store your checkpoints and validation videos. It's recommended to use a full path here.
- `train_batch_size` - start low for video training and increase only after confirming VRAM usage.
- `validation_resolution` - Sana Video ships as a 480p model; use `832x480` or the aspect buckets you intend to validate.
- `validation_num_video_frames` - Set this to `81` to match the default sampler length.
- `validation_guidance` - Use whatever you are used to selecting at inference time for Sana Video.
- `validation_num_inference_steps` - Use somewhere around 50 for steady quality.
- `framerate` - If omitted, Sana Video defaults to 16 fps; set this to match your dataset.

- `optimizer` - You can use any optimiser you are comfortable and familiar with, but we will use `optimi-adamw` for this example.
- `mixed_precision` - It's recommended to set this to `bf16` for the most efficient training configuration, or `no` (but will consume more memory and be slower).
- `gradient_checkpointing` - Enable this to control VRAM usage.
- `use_ema` - setting this to `true` will greatly help obtain a more smoothed result alongside your main trained checkpoint.

Multi-GPU users can reference [this document](/documentation/OPTIONS.md#environment-configuration-variables) for information on configuring the number of GPUs to use.

At the end, your config should resemble:

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "data_backend_config": "config/sanavideo/multidatabackend.json",
  "seed": 42,
  "output_dir": "output/sanavideo",
  "max_train_steps": 400000,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "tracker_project_name": "video-training",
  "tracker_run_name": "sanavideo-2b-480p",
  "report_to": "wandb",
  "model_type": "full",
  "pretrained_model_name_or_path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
  "pretrained_vae_model_name_or_path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
  "model_family": "sanavideo",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 200,
  "validation_resolution": "832x480",
  "validation_negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "validation_guidance": 6.0,
  "validation_num_inference_steps": 50,
  "validation_num_video_frames": 81,
  "validation_prompt": "A short video of a small, fluffy animal exploring a sunny room with soft window light and gentle camera motion.",
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.0,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "bf16",
  "vae_batch_size": 1,
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "framerate": 16,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```

### Advanced Experimental Features

SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> ⚠️ These features increase the computational overhead of training.

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

A set of diverse prompts will help determine whether the model is collapsing as it trains. In this example, the word `<token>` should be replaced with your subject name (instance_prompt).

```json
{
    "anime_<token>": "a breathtaking anime-style video featuring <token>, capturing her essence with vibrant colors, dynamic motion, and expressive storytelling",
    "chef_<token>": "a high-quality, detailed video of <token> as a sous-chef, immersed in the art of culinary creation with captivating close-ups and engaging sequences",
    "just_<token>": "a lifelike and intimate video portrait of <token>, showcasing her unique personality and charm through nuanced movement and expression",
    "cinematic_<token>": "a cinematic, visually stunning video of <token>, emphasizing her dramatic and captivating presence through fluid camera movements and atmospheric effects",
    "elegant_<token>": "an elegant and timeless video portrait of <token>, exuding grace and sophistication with smooth transitions and refined visuals",
    "adventurous_<token>": "a dynamic and adventurous video featuring <token>, captured in an exciting, action-filled sequence that highlights her energy and spirit",
    "mysterious_<token>": "a mysterious and enigmatic video portrait of <token>, shrouded in shadows and intrigue with a narrative that unfolds in subtle, cinematic layers",
    "vintage_<token>": "a vintage-style video of <token>, evoking the charm and nostalgia of a bygone era through sepia tones and period-inspired visual storytelling",
    "artistic_<token>": "an artistic and abstract video representation of <token>, blending creativity with visual storytelling through experimental techniques and fluid visuals",
    "futuristic_<token>": "a futuristic and cutting-edge video portrayal of <token>, set against a backdrop of advanced technology with sleek, high-tech visuals",
    "woman": "a beautifully crafted video portrait of a woman, highlighting her natural beauty and unique features through elegant motion and storytelling",
    "man": "a powerful and striking video portrait of a man, capturing his strength and character with dynamic sequences and compelling visuals",
    "boy": "a playful and spirited video portrait of a boy, capturing youthful energy and innocence through lively scenes and engaging motion",
    "girl": "a charming and vibrant video portrait of a girl, emphasizing her bright personality and joy with colorful visuals and fluid movement",
    "family": "a heartwarming and cohesive family video, showcasing the bonds and connections between loved ones through intimate moments and shared experiences"
}
```

> ℹ️ Sana Video is a flow-matching model; shorter prompts may not have enough information for the model to do a good job. Use descriptive prompts whenever possible.

#### CLIP score tracking

This should not be enabled for video model training, at the present time.

# Stable evaluation loss

If you wish to use stable MSE loss to score the model's performance, see [this document](/documentation/evaluation/EVAL_LOSS.md) for information on configuring and interpreting evaluation loss.

#### Validation previews

SimpleTuner supports streaming intermediate validation previews during generation using Tiny AutoEncoder models. This allows you to see validation videos being generated step-by-step in real-time via webhook callbacks.

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

Set `validation_preview_steps` to a higher value (e.g., 3 or 5) to reduce Tiny AutoEncoder overhead. With `validation_num_inference_steps=20` and `validation_preview_steps=5`, you'll receive preview frames at steps 5, 10, 15, and 20.

#### Flow-matching schedule

Sana Video uses the canonical flow-matching schedule from the checkpoint. User-provided shift overrides are ignored; keep `flow_schedule_shift` and `flow_schedule_auto_shift` unset for this model.

#### Quantised model training

Precision options (bf16, int8, fp8) are available in the config; match them to your hardware and fall back to higher precision if you encounter instabilities.

#### Dataset considerations

There are few limitations on the dataset size other than how much compute and time it will take to process and train.

You must ensure that the dataset is large enough to train your model effectively, but not too large for the amount of compute you have available.

Note that the bare minimum dataset size is `train_batch_size * gradient_accumulation_steps` as well as more than `vae_batch_size`. The dataset will not be useable if it is too small.

> ℹ️ With few enough samples, you might see a message **no samples detected in dataset** - increasing the `repeats` value will overcome this limitation.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently.

In this example, we will be using [video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized) as the dataset.

Create a `--data_backend_config` (`config/multidatabackend.json`) document containing this:

```json
[
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/sanavideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 81,
        "min_frames": 81
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sanavideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

- In the `video` subsection, we have the following keys we can set:
  - `num_frames` (optional, int) is how many frames of data we'll train on.
  - `min_frames` (optional, int) determines the minimum length of a video that will be considered for training.
  - `max_frames` (optional, int) determines the maximum length of a video that will be considered for training.
  - `is_i2v` (optional, bool) determines whether i2v training will be done on a dataset.

Then, create a `datasets` directory:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset sayakpaul/video-dataset-disney-organized --local-dir=disney-black-and-white
popd
```

This will download all of the Disney video samples to your `datasets/disney-black-and-white` directory, which will be automatically created for you.

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

## Notes & troubleshooting tips

### Validation defaults

- Sana Video defaults to 81 frames and 16 fps when validation settings are not provided.
- The Wan autoencoder path should match the base model path; keep them aligned to avoid load-time errors.

### Masked loss

If you are training a subject or style and would like to mask one or the other, see the [masked loss training](/documentation/DREAMBOOTH.md#masked-loss) section of the Dreambooth guide.
