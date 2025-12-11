## LTX Video Quickstart

In this example, we'll be training an LTX-Video LoRA using Sayak Paul's [public domain Disney dataset](https://hf.co/datasets/sayakpaul/video-dataset-disney-organized).

### Hardware requirements

LTX does not require much system **or** GPU memory.

When you're training every component of a rank-16 LoRA (MLP, projections, multimodal blocks), it ends up using a bit more than 12G on an M3 Mac (batch size 4).

You'll need:
- **a realistic minimum** is 16GB or, a single 3090 or V100 GPU
- **ideally** multiple 4090, A6000, L40S, or better

Apple silicon systems work great with LTX so far, albeit at a lower resolution due to limits inside the MPS backend used by Pytorch.

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
- Skip `--group_offload_to_disk_path` unless system RAM is <64 GB — disk staging is slower but keeps runs stable.
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


If you prefer to manually configure:

Copy `config/config.json.example` to `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

There, you will possibly need to modify the following variables:

- `model_type` - Set this to `lora`.
- `model_family` - Set this to `ltxvideo`.
- `pretrained_model_name_or_path` - Set this to `Lightricks/LTX-Video-0.9.5`.
- `pretrained_vae_model_name_or_path` - Set this to `Lightricks/LTX-Video-0.9.5`.
- `output_dir` - Set this to the directory where you want to store your checkpoints and validation images. It's recommended to use a full path here.
- `train_batch_size` - this can be increased for more stability, but a value of 4 should work alright to start with
- `validation_resolution` - This should be set to whatever you typically generate videos with when using LTX (`768x512`)
  - Multiple resolutions may be specified using commas to separate them: `1280x768,768x512`
- `validation_guidance` - Use whatever you are used to selecting at inference time for LTX.
- `validation_num_inference_steps` - Use somewhere around 25 to save time while still seeing decent quality.
- `--lora_rank=4` if you wish to substantially reduce the size of the LoRA being trained. This can help with VRAM use while reducing its capacity for learning.

- `gradient_accumulation_steps` - This option causes update steps to be accumulated over several steps.
  - This will increase the training runtime linearly, such that a value of 2 will make your training run half as quickly, and take twice as long.
- `optimizer` - Beginners are recommended to stick with adamw_bf16, though optimi-lion and optimi-stableadamw are also good choices.
- `mixed_precision` - Beginners should keep this in `bf16`
- `gradient_checkpointing` - set this to true in practically every situation on every device
- `gradient_checkpointing_interval` - this is not yet supported on LTX Video, and should be removed from your config.

Multi-GPU users can reference [this document](/documentation/OPTIONS.md#environment-configuration-variables) for information on configuring the number of GPUs to use.

At the end, your config should resemble mine:

<details>
<summary>View example config</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "data_backend_config": "config/ltxvideo/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "disable_benchmark": false,
  "offload_during_startup": true,
  "output_dir": "output/ltxvideo",
  "lora_type": "lycoris",
  "lycoris_config": "config/ltxvideo/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "ltxvideo-disney",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "ltxvideo-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5",
  "model_family": "ltxvideo",
  "train_batch_size": 8,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 800,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "768x512",
  "validation_negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "validation_guidance": 3.0,
  "validation_num_inference_steps": 40,
  "validation_prompt": "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day. And then the camera switch to a inding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region.",
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.0,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "fp8-torchao",
  "vae_batch_size": 1,
  "webhook_config": "config/ltxvideo/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "lora_rank": 128,
  "flow_schedule_shift": 1,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

### Optional: CREPA temporal regularizer

If your LTX runs show flicker or identity drift, try CREPA (cross-frame alignment):
- In the WebUI, go to **Training → Loss functions** and enable **CREPA**.
- Start with **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- Leave the default vision encoder (`dinov2_vitg14`, size `518`). Switch to `dinov2_vits14` + `224` only if you need lower VRAM.
- Needs internet (or a cached torch hub) the first time to fetch DINOv2 weights.
- Optional: if training purely from cached latents, enable **Drop VAE Encoder** to save memory; keep it off if you need to encode new videos.

### Advanced Experimental Features

<details>
<summary>Show advanced experimental details</summary>


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

A set of diverse prompt will help determine whether the model is collapsing as it trains. In this example, the word `<token>` should be replaced with your subject name (instance_prompt).

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

> ℹ️ LTX Video is a flow-matching model based on T5 XXL; shorter prompts may not have enough information for the model to do a good job. Be sure to use longer, more descriptive prompts.

#### CLIP score tracking

This should not be enabled for video model training, at the present time.

</details>

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

Flow-matching models such as Flux, Sana, SD3, and LTX Video have a property called `shift` that allows us to shift the trained portion of the timestep schedule using a simple decimal value.

##### Defaults
By default, no schedule shift is applied, which results in a sigmoid bell-shape to the timestep sampling distribution, otherwise known as `logit_norm`.

##### Auto-shift
A commonly-recommended approach is to follow several recent works and enable resolution-dependent timestep shift, `--flow_schedule_auto_shift` which uses higher shift values for larger images, and lower shift values for smaller images. This results in stable but potentially mediocre training results.

##### Manual specification
_Thanks to General Awareness from Discord for the following examples_

> ℹ️ These examples show how the value works using Flux Dev, though LTX Video should be very similar.

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
    "cache_dir_vae": "cache/vae/ltxvideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 125,
        "min_frames": 125
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

- In the `video` subsection, we have the following keys we can set:
  - `num_frames` (optional, int) is how many seconds of data we'll train on.
    - At 25 fps, 125 frames is 5 seconds of video, standard output. This should be your target.
  - `min_frames` (optional, int) determines the minimum length of a video that will be considered for training.
    - This should be at least equal to `num_frames`. Not setting it ensures it'll be equal.
  - `max_frames` (optional, int) determines the maximum length of a video that will be considered for training.
  - `is_i2v` (optional, bool) determines whether i2v training will be done on a dataset.
    - This is set to True by default for LTX. You can disable it, however.

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

### Lowest VRAM config

Like other models, it is possible that the lowest VRAM utilisation can be attained with:

- OS: Ubuntu Linux 24
- GPU: A single NVIDIA CUDA device (10G, 12G)
- System memory: 11G of system memory approximately
- Base model precision: `nf4-bnb`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 480px
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: disabled / unconfigured
- PyTorch: 2.6
- Be sure to enable `--gradient_checkpointing` or nothing you do will stop it from OOMing

**NOTE**: Pre-caching of VAE embeds and text encoder outputs may use more memory and still OOM. If so, text encoder quantisation and VAE tiling can be enabled. Beyond these options, `--offload_during_startup=true` will help avoid competition between VAE and text encoder memory use.

Speed was approximately 0.8 iterations per second on an M3 Max Macbook Pro.

### SageAttention

When using `--attention_mechanism=sageattention`, inference can be sped-up at validation time.

**Note**: This isn't compatible with _every_ model configuration, but it's worth trying.

### NF4-quantised training

In simplest terms, NF4 is a 4bit-_ish_ representation of the model, which means training has serious stability concerns to address.

In early tests, the following holds true:
- Lion optimiser causes model collapse but uses least VRAM; AdamW variants help to hold it together; bnb-adamw8bit, adamw_bf16 are great choices
  - AdEMAMix didn't fare well, but settings were not explored
- `--max_grad_norm=0.01` further helps reduce model breakage by preventing huge changes to the model in too short a time
- NF4, AdamW8bit, and a higher batch size all help to overcome the stability issues, at the cost of more time spent training or VRAM used
- Upping the resolution slows training down A LOT, and might harm the model
- Increasing the length of videos consumes a lot more memory as well. Reduce `num_frames` to beat this one.
- Anything that's difficult to train on int8 or bf16 becomes harder in NF4
- It's less compatible with options like SageAttention

NF4 does not work with torch.compile, so whatever you get for speed is what you get.

If VRAM is not a concern then int8 with torch.compile is your best, fastest option.

### Masked loss

Don't use this with LTX Video.


### Quantisation
- Quantisation is not needed to train this model

### Image artifacts
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

### Training custom fine-tuned LTX models

Some fine-tuned models on Hugging Face Hub lack the full directory structure, requiring specific options to be set.

<details>
<summary>View example config</summary>

```json
{
    "model_family": "ltxvideo",
    "pretrained_model_name_or_path": "Lightricks/LTX-Video",
    "pretrained_transformer_model_name_or_path": "path/to-the-other-model",
    "pretrained_vae_model_name_or_path": "Lightricks/LTX-Video",
    "pretrained_transformer_subfolder": "none",
}
```
</details>

## Credits

The [finetrainers](https://github.com/a-r-r-o-w/finetrainers) project and the Diffusers team.
- Originally used some design concepts from SimpleTuner
- Now contributes insight and code for making video training easily implemented
