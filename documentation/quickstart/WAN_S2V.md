## Wan 2.2 S2V Quickstart

In this example, we'll be training a Wan 2.2 S2V (Speech-to-Video) LoRA. S2V models generate video conditioned on audio input, enabling audio-driven video generation.

### Hardware requirements

Wan 2.2 S2V **14B** is a demanding model that requires significant GPU memory.

#### Speech to Video

14B - https://huggingface.co/tolgacangoz/Wan2.2-S2V-14B-Diffusers
- Resolution: 832x480
- It'll fit in 24G, but you'll have to fiddle with the settings a bit.

You'll need:
- **a realistic minimum** is 24GB or, a single 4090 or A6000 GPU
- **ideally** multiple 4090, A6000, L40S, or better

Apple silicon systems do not work super well with Wan 2.2 so far, something like 10 minutes for a single training step can be expected.

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

For manual installation or development setup, see the [installation documentation](/documentation/INSTALL.md).
#### SageAttention 2

If you wish to use SageAttention 2, some steps should be followed.

> Note: SageAttention provides minimal speed-up, not super effective; not sure why. Tested on 4090.

Run the following while still inside your python venv:
```bash
git clone https://github.com/thu-ml/SageAttention
pushd SageAttention
  pip install . --no-build-isolation
popd
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

### Setting up the environment

To run SimpleTuner, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

#### Configuration file

An experimental script, `configure.py`, may allow you to entirely skip this section through an interactive step-by-step configuration. It contains some safety features that help avoid common pitfalls.

**Note:** This doesn't configure your dataloader. You will still have to do that manually, later.

To run it:

```bash
simpletuner configure
```

> For users located in countries where Hugging Face Hub is not readily accessible, you should add `HF_ENDPOINT=https://hf-mirror.com` to your `~/.bashrc` or `~/.zshrc` depending on which `$SHELL` your system uses.

### Memory offloading (optional)

Wan is one of the heaviest models SimpleTuner supports. Enable grouped offloading if you are close to the VRAM ceiling:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Only CUDA devices honour `--group_offload_use_stream`; ROCm/MPS fall back automatically.
- Leave disk staging commented out unless CPU memory is the bottleneck.
- `--enable_model_cpu_offload` is mutually exclusive with group offload.

### Feed-forward chunking (optional)

If the 14B checkpoints still OOM during gradient checkpointing, chunk the Wan feed-forward layers:

```bash
--enable_chunked_feed_forward \
--feed_forward_chunk_size 2 \
```

This matches the new toggle in the configuration wizard (`Training -> Memory Optimisation`). Smaller chunk sizes save more
memory but slow each step. You can also set `WAN_FEED_FORWARD_CHUNK_SIZE=2` in your environment for quick experiments.


If you prefer to manually configure:

Copy `config/config.json.example` to `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Multi-GPU users can reference [this document](/documentation/OPTIONS.md#environment-configuration-variables) for information on configuring the number of GPUs to use.

Your config at the end will look like mine:

<details>
<summary>View example config</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "attention_mechanism": "sageattention",
  "data_backend_config": "config/wan_s2v/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "offload_during_startup": true,
  "disable_benchmark": false,
  "output_dir": "output/wan_s2v",
  "lora_type": "standard",
  "lycoris_config": "config/wan_s2v/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "wan-s2v-lora",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "wan-s2v-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
  "pretrained_t5_model_name_or_path": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
  "model_family": "wan_s2v",
  "model_flavour": "s2v-14b-2.2",
  "train_batch_size": 2,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "832x480",
  "validation_prompt": "A person speaking with natural gestures",
  "validation_negative_prompt": "blurry, low quality, distorted",
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 40,
  "validation_num_video_frames": 81,
  "mixed_precision": "bf16",
  "optimizer": "optimi-lion",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.01,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "no_change",
  "vae_batch_size": 1,
  "webhook_config": "config/wan_s2v/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "lora_rank": 16,
  "flow_schedule_shift": 3,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

Of particular importance in this configuration are the validation settings. Without these, the outputs do not look super great.

### Optional: CREPA temporal regularizer

For smoother motion and less identity drift on Wan S2V:
- In **Training -> Loss functions**, enable **CREPA**.
- Start with **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- Default encoder (`dinov2_vitg14`, size `518`) works well; swap to `dinov2_vits14` + `224` only if you need to trim VRAM.
- First run downloads DINOv2 via torch hub; cache or prefetch if you train offline.
- Only enable **Drop VAE Encoder** when training entirely from cached latents; otherwise keep it off so pixel encodes still work.

### Advanced Experimental Features

<details>
<summary>Show advanced experimental details</summary>


SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> These features increase the computational overhead of training.

</details>

### TREAD training

> **Experimental**: TREAD is a newly implemented feature. While functional, optimal configurations are still being explored.

[TREAD](/documentation/TREAD.md) (paper) stands for **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion. It is a method that can accelerate Wan S2V training by intelligently routing tokens through transformer layers. The speedup is proportional to how many tokens you drop.

#### Quick setup

Add this to your `config.json` for a simple and conservative approach:

<details>
<summary>View example config</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.1,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

This configuration will:
- Keep only 50% of image tokens during layers 2 through second-to-last
- Text tokens are never dropped
- Training speedup of ~25% with minimal quality impact
- Potentially improves training quality and convergence

#### Key points

- **Limited architecture support** - TREAD is only implemented for Flux and Wan models (including S2V)
- **Best at high resolutions** - Biggest speedups at 1024x1024+ due to attention's O(n^2) complexity
- **Compatible with masked loss** - Masked regions are automatically preserved (but this reduces speedup)
- **Works with quantization** - Can be combined with int8/int4/NF4 training
- **Expect initial loss spike** - When starting LoRA/LoKr training, loss will be higher initially but corrects quickly

#### Tuning tips

- **Conservative (quality-focused)**: Use `selection_ratio` of 0.1-0.3
- **Aggressive (speed-focused)**: Use `selection_ratio` of 0.3-0.5 and accept the quality impact
- **Avoid early/late layers**: Don't route in layers 0-1 or the final layer
- **For LoRA training**: May see slight slowdowns - experiment with different configs
- **Higher resolution = better speedup**: Most beneficial at 1024px and above

For detailed configuration options and troubleshooting, see the [full TREAD documentation](/documentation/TREAD.md).


#### Validation prompts

Inside `config/config.json` is the "primary validation prompt", which is typically the main instance_prompt you are training on for your single subject or style. Additionally, a JSON file may be created that contains extra prompts to run through during validations.

The example config file `config/user_prompt_library.json.example` contains the following format:

<details>
<summary>View example config</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

The nicknames are the filename for the validation, so keep them short and compatible with your filesystem.

To point the trainer to this prompt library, add it to TRAINER_EXTRA_ARGS by adding a new line at the end of `config.json`:
<details>
<summary>View example config</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

> S2V uses the UMT5 text encoder, which has a lot of local information in its embeddings which means that shorter prompts may not have enough information for the model to do a good job. Be sure to use longer, more descriptive prompts.

#### CLIP score tracking

This should not be enabled for video model training, at the present time.

# Stable evaluation loss

If you wish to use stable MSE loss to score the model's performance, see [this document](/documentation/evaluation/EVAL_LOSS.md) for information on configuring and interpreting evaluation loss.

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

#### Flow-matching schedule shift

Flow-matching models such as Flux, Sana, SD3, LTX Video and Wan S2V have a property called `shift` that allows us to shift the trained portion of the timestep schedule using a simple decimal value.

##### Defaults
By default, no schedule shift is applied, which results in a sigmoid bell-shape to the timestep sampling distribution, otherwise known as `logit_norm`.

##### Auto-shift
A commonly-recommended approach is to follow several recent works and enable resolution-dependent timestep shift, `--flow_schedule_auto_shift` which uses higher shift values for larger images, and lower shift values for smaller images. This results in stable but potentially mediocre training results.

##### Manual specification
_Thanks to General Awareness from Discord for the following examples_

> These examples show how the value works using Flux Dev, though Wan S2V should be very similar.

When using a `--flow_schedule_shift` value of 0.1 (a very low value), only the finer details of the image are affected:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

When using a `--flow_schedule_shift` value of 4.0 (a very high value), the large compositional features and potentially colour space of the model becomes impacted:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### Quantised model training

Tested on Apple and NVIDIA systems, Hugging Face Optimum-Quanto can be used to reduce the precision and VRAM requirements, training on just 16GB.



For `config.json` users:
<details>
<summary>View example config</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

#### Validation settings

During initial exploration, poor output quality may come from Wan S2V, and this boils down to a couple reasons:

- Not enough steps for inference
  - Unless you're using UniPC, you probably need at least 40 steps. UniPC can bring the number down a little, but you'll have to experiment.
- Incorrect scheduler configuration
  - It was using normal Euler flow matching schedule, but the Betas distribution seems to work best
  - If you haven't touched this setting, it should be fine now
- Incorrect resolution
  - Wan S2V only really works correctly on the resolutions it was trained on, you get lucky if it works, but it's common for it to be bad results
- Bad CFG value
  - A value around 4.0-5.0 seems safe
- Bad prompting
  - Of course, video models seem to require a team of mystics to spend months in the mountains on a zen retreat to learn the sacred art of prompting, because their datasets and caption style are guarded like the Holy Grail.
  - tl;dr try different prompts.
- Missing or mismatched audio
  - S2V requires audio input for validation - ensure your validation samples have corresponding audio files

Despite all of this, unless your batch size is too low and / or your learning rate is too high, the model will run correctly in your favourite inference tool (assuming you already have one that you get good results from).

#### Dataset considerations

S2V training requires paired video and audio data. By default, SimpleTuner auto-splits audio from video datasets, so you
do not need to define a separate audio dataset unless you want custom processing. Set `audio.auto_split: false` to opt
out and provide `s2v_datasets` manually.

There are few limitations on the dataset size other than how much compute and time it will take to process and train.

You must ensure that the dataset is large enough to train your model effectively, but not too large for the amount of compute you have available.

Note that the bare minimum dataset size is `train_batch_size * gradient_accumulation_steps` as well as more than `vae_batch_size`. The dataset will not be useable if it is too small.

> With few enough samples, you might see a message **no samples detected in dataset** - increasing the `repeats` value will overcome this limitation.

#### Audio dataset setup

##### Automatic audio extraction from videos (Recommended)

If your videos already contain audio tracks, SimpleTuner can automatically extract and process audio without requiring a separate audio dataset. This is the default and simplest approach:

```json
[
  {
    "id": "s2v-videos",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan_s2v/videos",
    "instance_data_dir": "datasets/s2v-videos",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    },
    "audio": {
        "auto_split": true,
        "sample_rate": 16000,
        "channels": 1
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan_s2v",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

With audio auto-split enabled (default), SimpleTuner will:
1. Auto-generate an audio dataset configuration (`s2v-videos_audio`)
2. Extract audio from each video during metadata discovery
3. Cache audio VAE latents in a dedicated directory
4. Automatically link the audio dataset via `s2v_datasets`

**Audio configuration options:**
- `audio.auto_split` (bool): Enable automatic audio extraction from videos (default: true)
- `audio.sample_rate` (int): Target sample rate in Hz (default: 16000 for Wav2Vec2)
- `audio.channels` (int): Number of audio channels (default: 1 for mono)
- `audio.allow_zero_audio` (bool): Generate zero-filled audio for videos without audio streams (default: false)
- `audio.max_duration_seconds` (float): Maximum audio duration; longer files are skipped
- `audio.duration_interval` (float): Duration interval for bucket grouping in seconds (default: 3.0)
- `audio.truncation_mode` (string): How to truncate long audio: "beginning", "end", "random" (default: "beginning")

**Note**: Videos without audio tracks are automatically skipped for S2V training unless `audio.allow_zero_audio: true` is set.

##### Manual audio dataset (Alternative)

If you prefer separate audio files, need custom audio processing, or disable auto-split, S2V models can also use
pre-extracted audio files that match your video files by filename. For example:
- `video_001.mp4` should have a corresponding `video_001.wav` (or `.mp3`, `.flac`, `.ogg`, `.m4a`)

The audio files should be in a separate directory that you'll configure as an `s2v_datasets` backend.

##### Extracting audio from videos (Manual)

If your videos already contain audio, use the provided script to extract it:

```bash
# Extract audio only (keeps original videos unchanged)
python scripts/generate_s2v_audio.py \
    --input-dir datasets/s2v-videos \
    --output-dir datasets/s2v-audio

# Extract audio and remove it from source videos (recommended to avoid redundant data)
python scripts/generate_s2v_audio.py \
    --input-dir datasets/s2v-videos \
    --output-dir datasets/s2v-audio \
    --strip-audio
```

The script:
- Extracts audio at 16kHz mono WAV (native Wav2Vec2 sample rate)
- Matches filenames automatically (e.g., `video.mp4` -> `video.wav`)
- Skips videos without audio streams
- Requires `ffmpeg` to be installed

##### Dataset configuration (Manual)

Create a `--data_backend_config` (`config/multidatabackend.json`) document containing this:

<details>
<summary>View example config</summary>

```json
[
  {
    "id": "s2v-videos",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan_s2v/videos",
    "instance_data_dir": "datasets/s2v-videos",
    "s2v_datasets": ["s2v-audio"],
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "s2v-audio",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/s2v-audio",
    "disabled": false
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan_s2v",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

Key points for S2V dataset configuration:
- The `s2v_datasets` field on your video dataset points to the audio backend(s)
- Audio files are matched by filename stem (e.g., `video_001.mp4` matches `video_001.wav`)
- Audio is encoded on-the-fly using Wav2Vec2 (~600MB VRAM), no caching required
- The audio dataset type is `audio`

- In the `video` subsection, we have the following keys we can set:
  - `num_frames` (optional, int) is how many frames of data we'll train on.
    - At 15 fps, 75 frames is 5 seconds of video, standard output. This should be your target.
  - `min_frames` (optional, int) determines the minimum length of a video that will be considered for training.
    - This should be at least equal to `num_frames`. Not setting it ensures it'll be equal.
  - `max_frames` (optional, int) determines the maximum length of a video that will be considered for training.
  - `bucket_strategy` (optional, string) determines how videos are grouped into buckets:
    - `aspect_ratio` (default): Group by spatial aspect ratio only (e.g., `1.78`, `0.75`).
    - `resolution_frames`: Group by resolution and frame count in `WxH@F` format (e.g., `832x480@75`). Useful for mixed-resolution/duration datasets.
  - `frame_interval` (optional, int) when using `resolution_frames`, round frame counts to this interval.

Then, create a `datasets` directory with your video and audio files:

```bash
mkdir -p datasets/s2v-videos datasets/s2v-audio
# Place your video files in datasets/s2v-videos/
# Place your audio files in datasets/s2v-audio/
```

Ensure each video has a matching audio file by filename stem.

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
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'

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

### Lowest VRAM config

Wan S2V is sensitive to quantisation, and cannot be used with NF4 or INT4 currently.

- OS: Ubuntu Linux 24
- GPU: A single NVIDIA CUDA device (24G recommended)
- System memory: 16G of system memory approximately
- Base model precision: `int8-quanto`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 480px
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: disabled / unconfigured
- PyTorch: 2.6
- Be sure to enable `--gradient_checkpointing` or nothing you do will stop it from OOMing
- Only train on images, or set `num_frames` to 1 for your video dataset

**NOTE**: Pre-caching of VAE embeds and text encoder outputs may use more memory and still OOM. As a result, `--offload_during_startup=true` is basically required. If so, text encoder quantisation and VAE tiling can be enabled. (Wan does not currently support VAE tiling/slicing)

### SageAttention

When using `--attention_mechanism=sageattention`, inference can be sped-up at validation time.

**Note**: This isn't compatible with the final VAE decode step, and will not speed that portion up.

### Masked loss

Don't use this with Wan S2V.

### Quantisation
- Quantisation may be needed to train this model in 24G depending on batch size

### Image artifacts
Wan requires the use of the Euler Betas flow-matching schedule or (by default) the UniPC multistep solver, a higher order scheduler which will make stronger predictions.

Like other DiT models, if you do these things (among others) some square grid artifacts **may** begin appearing in the samples:
- Overtrain with low quality data
- Use too high of a learning rate
- Overtraining (in general), a low-capacity network with too many images
- Undertraining (also), a high-capacity network with too few images
- Using weird aspect ratios or training data sizes

### Aspect bucketing
- Videos are bucketed like images.
- Training for too long on square crops probably won't damage this model too much. Go nuts, it's great and reliable.
- On the other hand, using the natural aspect buckets of your dataset might overly bias these shapes during inference time.
  - This could be a desirable quality, as it keeps aspect-dependent styles like cinematic stuff from bleeding into other resolutions too much.
  - However, if you're looking to improve results equally across many aspect buckets, you might have to experiment with `crop_aspect=random` which comes with its own downsides.
- Mixing dataset configurations by defining your image directory dataset multiple times has produced really good results and a nicely generalised model.

### Audio synchronization

For best results with S2V:
- Ensure audio duration matches video duration
- Audio is resampled to 16kHz internally
- The Wav2Vec2 encoder processes audio on-the-fly (~600MB VRAM overhead)
- Audio features are interpolated to match the number of video frames
