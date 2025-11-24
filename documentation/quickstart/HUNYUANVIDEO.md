# Hunyuan Video 1.5 Quickstart

This guide walks through training a LoRA on Tencent's 8.3B **Hunyuan Video 1.5** release (`tencent/HunyuanVideo-1.5`) using SimpleTuner.

## Hardware requirements

Hunyuan Video 1.5 is a large model (8.3B parameters).

- **Minimum**: **24GB-32GB VRAM** is comfortable for a Rank-16 LoRA with full gradient checkpointing at 480p.
- **Recommended**: A6000 / A100 (48GB-80GB) for 720p training or larger batch sizes.
- **System RAM**: **64GB+** is recommended to handle model loading.

### Memory offloading (optional)

Add the following to your `config.json`:

```json
{
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "group_offload_blocks_per_group": 1,
  "group_offload_use_stream": true
}
```

- `--group_offload_use_stream`: Only works on CUDA devices.
- **Do not** combine this with `--enable_model_cpu_offload`.

## Prerequisites

Make sure that you have python installed; SimpleTuner does well with 3.10 through 3.12.

You can check this by running:

```bash
python --version
```

If you don't have python 3.12 installed on Ubuntu, you can try the following:

```bash
apt -y install python3.12 python3.12-venv
```

### Container image dependencies

For Vast, RunPod, and TensorDock (among others), the following will work on a CUDA 12.2-12.8 image to enable compiling of CUDA extensions:

```bash
apt -y install nvidia-cuda-toolkit
```

### AMD ROCm follow-up steps

The following must be executed for an AMD MI300X to be useable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## Installation

Install SimpleTuner via pip:

```bash
pip install simpletuner[cuda]
```

For manual installation or development setup, see the [installation documentation](/documentation/INSTALL.md).

### Required checkpoints

The main `tencent/HunyuanVideo-1.5` repo contains the transformer/vae/scheduler, but the **text encoder** (`text_encoder/llm`) and **vision encoder** (`vision_encoder/siglip`) live in separate downloads. Point SimpleTuner at your local copies before launching:

```bash
export HUNYUANVIDEO_TEXT_ENCODER_PATH=/path/to/text_encoder_root
export HUNYUANVIDEO_VISION_ENCODER_PATH=/path/to/vision_encoder_root
```

If these are unset, SimpleTuner tries to pull them from the model repo; most mirrors do not bundle them, so set the paths explicitly to avoid startup errors.

## Setting up the environment

### Web interface method

The SimpleTuner WebUI makes setup fairly straightforward. To run the server:

```bash
simpletuner server
```

This will create a webserver on port 8001 by default, which you can access by visiting http://localhost:8001.

### Manual / command-line method

To run SimpleTuner via command-line tools, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

#### Configuration file

An experimental script, `configure.py`, may allow you to entirely skip this section through an interactive step-by-step configuration.

**Note:** This doesn't configure your dataloader. You will still have to do that manually, later.

To run it:

```bash
simpletuner configure
```

If you prefer to manually configure:

Copy `config/config.json.example` to `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Key configuration overrides for HunyuanVideo:

```json
{
  "model_type": "lora",
  "model_family": "hunyuan_video",
  "pretrained_model_name_or_path": "tencent/HunyuanVideo",
  "model_flavour": "t2v-480p",
  "output_dir": "output/hunyuan-video",
  "validation_resolution": "854x480",
  "validation_num_video_frames": 61,
  "validation_guidance": 6.0,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "learning_rate": 1e-4,
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "lora_rank": 16,
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "dataset_backend_config": "config/multidatabackend.json"
}
```

- `model_flavour` options:
  - `t2v-480p` (Default)
  - `t2v-720p`
  - `i2v-480p` (Image-to-Video)
  - `i2v-720p` (Image-to-Video)
- `validation_num_video_frames`: Must be `(frames - 1) % 4 == 0`. E.g., 61, 129.

#### Dataset considerations

Create a `--data_backend_config` (`config/multidatabackend.json`) document containing this:

```json
[
  {
    "id": "my-video-dataset",
    "type": "local",
    "dataset_type": "video",
    "instance_data_dir": "datasets/videos",
    "caption_strategy": "textfile",
    "resolution": 480,
    "video": {
        "num_frames": 61,
        "min_frames": 61,
        "frame_rate": 24
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/hunyuan",
    "disabled": false
  }
]
```

- **Text Embed Caching**: Highly recommended. Hunyuan uses a large LLM text encoder. Caching saves significant VRAM during training.

#### Login to WandB and Huggingface Hub

```bash
wandb login
huggingface-cli login
```

### Executing the training run

From the SimpleTuner directory:

```bash
simpletuner train
```

## Notes & troubleshooting tips

### VRAM Optimization

- **Group Offload**: Essential for consumer GPUs. Ensure `enable_group_offload` is true.
- **Resolution**: Stick to 480p (`854x480` or similar) if you have limited VRAM. 720p (`1280x720`) increases memory usage significantly.
- **Quantization**: Not yet fully standardized for Hunyuan in SimpleTuner, but `base_model_precision` can be experimented with if available.

### Image-to-Video (I2V)

- Use `model_flavour="i2v-480p"`.
- SimpleTuner automatically uses the first frame of your video dataset samples as the conditioning image.
- Ensure your validation setup includes conditioning inputs or relies on the auto-extracted first frame.

### Text Encoders

Hunyuan uses a dual text encoder setup (LLM + CLIP). Ensure your system RAM can handle loading these during the caching phase.
