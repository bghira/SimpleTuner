## Wan 2.1 Quickstart

In this example, we'll be training a Wan 2.1 LoRA using Sayak Paul's [public domain Disney dataset](https://hf.co/datasets/sayakpaul/video-dataset-disney-organized).



https://github.com/user-attachments/assets/51e6cbfd-5c46-407c-9398-5932fa5fa561


### Hardware requirements

Wan 2.1 **1.3B** does not require much system **or** GPU memory. The **14B** model, also supported, is a lot more demanding.

Currently, image-to-video training is not supported for Wan, but T2V LoRA and Lycoris will run on the I2V models.

#### Text to Video

1.3B - https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- Resolution: 832x480
- Rank-16 LoRA uses a bit more than 12G (batch size 4)

14B - https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers
- Resolution: 832x480
- It'll fit in 24G, but you'll have to fiddle with the settings a bit.

<!--
#### Image to Video
14B (720p) - https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
- Resolution: 1280x720
-->

#### Image to Video (Wan 2.2)

Recent Wan 2.2 I2V checkpoints work with the same training flow:

- High stage: https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-Diffusers/tree/main/high_noise_model
- Low stage: https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-Diffusers/tree/main/low_noise_model

You can target the stage you want with the `model_flavour` and `wan_validation_load_other_stage` settings outlined later in this guide.

You'll need:
- **a realistic minimum** is 16GB or, a single 3090 or V100 GPU
- **ideally** multiple 4090, A6000, L40S, or better

If you encounter shape mismatches in the time embedding layers when running Wan 2.2 checkpoints, enable the new
`wan_force_2_1_time_embedding` flag. This forces the transformer to fall back to Wan 2.1 style time embeddings and
resolves the compatibility issue.

#### Stage presets & validation

- `model_flavour=i2v-14b-2.2-high` targets the Wan 2.2 high-noise stage.
- `model_flavour=i2v-14b-2.2-low` targets the low-noise stage (same checkpoints, different subfolder).
- Toggle `wan_validation_load_other_stage=true` to load the opposite stage alongside the one you train for validation renders.
- Leave the flavour unset (or use `t2v-480p-1.3b-2.1`) for the standard Wan 2.1 text-to-video run.

Apple silicon systems do not work super well with Wan 2.1 so far, something like 10 minutes for a single training step can be expected..

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

> ⚠️ For users located in countries where Hugging Face Hub is not readily accessible, you should add `HF_ENDPOINT=https://hf-mirror.com` to your `~/.bashrc` or `~/.zshrc` depending on which `$SHELL` your system uses.

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

This matches the new toggle in the configuration wizard (`Training → Memory Optimisation`). Smaller chunk sizes save more
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
  "data_backend_config": "config/wan/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "offload_during_startup": true,
  "disable_benchmark": false,
  "output_dir": "output/wan",
  "lora_type": "standard",
  "lycoris_config": "config/wan/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "wan-disney",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "wan-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "model_family": "wan",
  "train_batch_size": 2,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "832x480",
  "validation_prompt": "两只拟人化的猫咪身穿舒适的拳击装备，戴着鲜艳的手套，在聚光灯照射的舞台上激烈对战",
  "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
  "validation_guidance": 5.2,
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
  "webhook_config": "config/wan/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "validation_guidance_skip_layers": [9],
  "validation_guidance_skip_layers_start": 0.0,
  "validation_guidance_skip_layers_stop": 1.0,
  "lora_rank": 16,
  "flow_schedule_shift": 3,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

Of particular importance in this configuration are the validation settings. Without these, the outputs do not look super great.

### Advanced Experimental Features

<details>
<summary>Show advanced experimental details</summary>


SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> ⚠️ These features increase the computational overhead of training.

</details>

### TREAD training

> ⚠️ **Experimental**: TREAD is a newly implemented feature. While functional, optimal configurations are still being explored.

[TREAD](/documentation/TREAD.md) (paper) stands for **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion. It is a method that can accelerate Flux training by intelligently routing tokens through transformer layers. The speedup is proportional to how many tokens you drop.

#### Quick setup

Add this to your `config.json` for a simple and conservative approach to reach about 5 seconds per step with bs=2 and 480p (reduced from 10 seconds per step vanilla speed):

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

For Wan 1.3B we can enhance this approach using a progressive route setup over all 29 layers and hit a speed around 7.7 seconds per step at bs=2 and 480p:

<details>
<summary>View example config</summary>

```json
{
  "tread_config": {
      "routes": [
          { "selection_ratio": 0.1, "start_layer_idx": 2, "end_layer_idx": 8 },
          { "selection_ratio": 0.25, "start_layer_idx": 9, "end_layer_idx": 11 },
          { "selection_ratio": 0.35, "start_layer_idx": 12, "end_layer_idx": 15 },
          { "selection_ratio": 0.25, "start_layer_idx": 16, "end_layer_idx": 23 },
          { "selection_ratio": 0.1, "start_layer_idx": 24, "end_layer_idx": -2 }
      ]
  }
}
```
</details>

This configuration will attempt to use more aggressive token dropout in the inner layers of the model where semantic knowledge isn't as important.

For some datasets, more aggressive dropout may be tolerable, but a value of 0.5 is considerably high for Wan 2.1.

#### Key points

- **Limited architecture support** - TREAD is only implemented for Flux and Wan models
- **Best at high resolutions** - Biggest speedups at 1024x1024+ due to attention's O(n²) complexity
- **Compatible with masked loss** - Masked regions are automatically preserved (but this reduces speedup)
- **Works with quantization** - Can be combined with int8/int4/NF4 training
- **Expect initial loss spike** - When starting LoRA/LoKr training, loss will be higher initially but corrects quickly

#### Tuning tips

- **Conservative (quality-focused)**: Use `selection_ratio` of 0.1-0.3
- **Aggressive (speed-focused)**: Use `selection_ratio` of 0.3-0.5 and accept the quality impact
- **Avoid early/late layers**: Don't route in layers 0-1 or the final layer
- **For LoRA training**: May see slight slowdowns - experiment with different configs
- **Higher resolution = better speedup**: Most beneficial at 1024px and above

#### Known behavior

- The more tokens dropped (higher `selection_ratio`), the faster training but higher initial loss
- LoRA/LoKr training shows an initial loss spike that rapidly corrects as the network adapts
  - Using less-aggressive training configuration or multiple routes with inner layers having higher levels will alleviate this
- Some LoRA configurations may train slightly slower - optimal configs still being explored
- The RoPE (rotary position embedding) implementation is functional but may not be 100% correct

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

A set of diverse prompt will help determine whether the model is collapsing as it trains. In this example, the word `<token>` should be replaced with your subject name (instance_prompt).

<details>
<summary>View example config</summary>

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
</details>

> ℹ️ Wan 2.1 uses the UMT5 text encoder only, which has a lot of local information in its embeddings which means that shorter prompts may not have enough information for the model to do a good job. Be sure to use longer, more descriptive prompts.

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

Flow-matching models such as Flux, Sana, SD3, LTX Video and Wan 2.1 have a property called `shift` that allows us to shift the trained portion of the timestep schedule using a simple decimal value.

##### Defaults
By default, no schedule shift is applied, which results in a sigmoid bell-shape to the timestep sampling distribution, otherwise known as `logit_norm`.

##### Auto-shift
A commonly-recommended approach is to follow several recent works and enable resolution-dependent timestep shift, `--flow_schedule_auto_shift` which uses higher shift values for larger images, and lower shift values for smaller images. This results in stable but potentially mediocre training results.

##### Manual specification
_Thanks to General Awareness from Discord for the following examples_

> ℹ️ These examples show how the value works using Flux Dev, though Wan 2.1 should be very similar.

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

During initial exploration into adding Wan 2.1 into SimpleTuner, horrible nightmare fuel output was coming from Wan 2.1, and this boils down to a couple reasons:

- Not enough steps for inference
  - Unless you're using UniPC, you probably need at least 40 steps. UniPC can bring the number down a little, but you'll have to experiment.
- Incorrect scheduler configuration
  - It was using normal Euler flow matching schedule, but the Betas distribution seems to work best
  - If you haven't touched this setting, it should be fine now
- Incorrect resolution
  - Wan 2.1 only really works correctly on the resolutions it was trained on, you get lucky if it works, but it's common for it to be bad results
- Bad CFG value
  - Wan 2.1 1.3B in particular seems sensitive to CFG values, but a value around 4.0-5.0 seem safe
- Bad prompting
  - Of course, video models seem to require a team of mystics to spend months in the mountains on a zen retreat to learn the sacred art of prompting, because their datasets and caption style are guarded like the Holy Grail.
  - tl;dr try different prompts.

Despite all of this, unless your batch size is too low and / or your learning rate is too high, the model will run correctly in your favourite inference tool (assuming you already have one that you get good results from).

#### Dataset considerations

There are few limitations on the dataset size other than how much compute and time it will take to process and train.

You must ensure that the dataset is large enough to train your model effectively, but not too large for the amount of compute you have available.

Note that the bare minimum dataset size is `train_batch_size * gradient_accumulation_steps` as well as more than `vae_batch_size`. The dataset will not be useable if it is too small.

> ℹ️ With few enough samples, you might see a message **no samples detected in dataset** - increasing the `repeats` value will overcome this limitation.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently.

In this example, we will be using [video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized) as the dataset.

Create a `--data_backend_config` (`config/multidatabackend.json`) document containing this:

<details>
<summary>View example config</summary>

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
    "cache_dir_vae": "cache/vae/wan/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

- Wan 2.2 image-to-video runs create CLIP conditioning caches. In the **video** dataset entry, point at a dedicated backend and (optionally) override the cache path:

<details>
<summary>View example config</summary>

```json
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "conditioning_image_embeds": "disney-conditioning",
    "cache_dir_conditioning_image_embeds": "cache/conditioning_image_embeds/disney-black-and-white"
  }
```
</details>

- Define the conditioning backend once and reuse it across datasets if needed (full object shown here for clarity):

<details>
<summary>View example config</summary>

```json
  {
    "id": "disney-conditioning",
    "type": "local",
    "dataset_type": "conditioning_image_embeds",
    "cache_dir": "cache/conditioning_image_embeds/disney-conditioning",
    "disabled": false
  }
```
</details>

- In the `video` subsection, we have the following keys we can set:
  - `num_frames` (optional, int) is how many seconds of data we'll train on.
    - At 15 fps, 75 frames is 5 seconds of video, standard output. This should be your target.
  - `min_frames` (optional, int) determines the minimum length of a video that will be considered for training.
    - This should be at least equal to `num_frames`. Not setting it ensures it'll be equal.
  - `max_frames` (optional, int) determines the maximum length of a video that will be considered for training.
<!--  - `is_i2v` (optional, bool) determines whether i2v training will be done on a dataset.
    - This is set to True by default for Wan 2.1. You can disable it, however.
-->

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

> ℹ️ Append `--model_flavour i2v-14b-2.2-high` (or `low`) and, if desired, `--wan_validation_load_other_stage` inside `TRAINER_EXTRA_ARGS` or your CLI invocation when you train Wan 2.2. Add `--wan_force_2_1_time_embedding` only when the checkpoint reports a time-embedding shape mismatch.

**Option 3 (Legacy method - still works):**
```bash
./train.sh
```

This will begin the text embed and VAE output caching to disk.

For more information, see the [dataloader](/documentation/DATALOADER.md) and [tutorial](/documentation/TUTORIAL.md) documents.

## Notes & troubleshooting tips

### Lowest VRAM config

Wan 2.1 is sensitive to quantisation, and cannot be used with NF4 or INT4 currently.

- OS: Ubuntu Linux 24
- GPU: A single NVIDIA CUDA device (10G, 12G)
- System memory: 12G of system memory approximately
- Base model precision: `int8-quanto`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 480px
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: disabled / unconfigured
- PyTorch: 2.6
- Be sure to enable `--gradient_checkpointing` or nothing you do will stop it from OOMing
- Only train on images, or set `num_frames` to 1 for your video dataset

**NOTE**: Pre-caching of VAE embeds and text encoder outputs may use more memory and still OOM. As a result, `--offload_during_startup=true` is basically required. If so, text encoder quantisation and VAE tiling can be enabled. (Wan does not currently support VAE tiling/slicing)

Speeds:
- 665.8 sec/iter on an M3 Max Macbook Pro
- 2 sec/iter on a NVIDIA 4090 at a batch size of 1
- 11 sec/iter on NVIDIA 4090 with batch size of 4

### SageAttention

When using `--attention_mechanism=sageattention`, inference can be sped-up at validation time.

**Note**: This isn't compatible with the final VAE decode step, and will not speed that portion up.

### Masked loss

Don't use this with Wan 2.1.

### Quantisation
- Quantisation is not needed to train this model in 24G

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

### Training custom fine-tuned Wan 2.1 models

Some fine-tuned models on Hugging Face Hub lack the full directory structure, requiring specific options to be set.

<details>
<summary>View example config</summary>

```json
{
    "model_family": "wan",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_transformer_model_name_or_path": "path/to-the-other-model",
    "pretrained_vae_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_transformer_subfolder": "none",
}
```
</details>

> Note: You can provide a path to a single-file `.safetensors` for the `pretrained_transformer_name_or_path`
