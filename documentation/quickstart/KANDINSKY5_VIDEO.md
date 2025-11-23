# Kandinsky 5.0 Video Quickstart

In this example, we'll train a Kandinsky 5.0 Video LoRA (Lite or Pro) using the HunyuanVideo VAE and dual text encoders.

## Hardware requirements

Kandinsky 5.0 Video is a heavy model. It combines:
1.  **Qwen2.5-VL (7B)**: A massive vision-language text encoder.
2.  **HunyuanVideo VAE**: A high-quality 3D VAE.
3.  **Video Transformer**: A complex DiT architecture.

This setup is VRAM-intensive, though the "Lite" and "Pro" variants have different requirements.

- **Lite Model Training**: Surprisingly efficient, capable of training on **~13GB VRAM**.
  - **Note**: The initial **VAE pre-caching step** requires significantly more VRAM due to the massive HunyuanVideo VAE. You may need to use CPU offloading or a larger GPU just for the caching phase.
  - **Tip**: Set `"offload_during_startup": true` in your `config.json` to ensure the VAE and text encoder are not loaded to the GPU at the same time, which significantly reduces pre-caching memory pressure.
- **Pro Model Training**: Requires **FSDP2** (multi-gpu) or aggressive **Group Offload** with LoRA to fit on consumer hardware. Specific VRAM/RAM requirements have not been established, but "the more, the merrier" applies.
- **System RAM**: Testing was comfortable on a system with **45GB** RAM for the Lite model. 64GB+ is recommended to be safe.

### Memory offloading (Critical)

For almost any single-GPU setup training the **Pro** model, you **must** enable grouped offloading. It is optional but recommended for **Lite** to save VRAM for larger batches/resolutions.

Add this to your `config.json`:

```json
{
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "group_offload_blocks_per_group": 1,
  "group_offload_use_stream": true
}
```

## Prerequisites

Ensure Python 3.10, 3.11, or 3.12 is installed.

```bash
python --version
```

## Installation

```bash
pip install simpletuner[cuda]
```

See [INSTALL.md](/documentation/INSTALL.md) for advanced installation options.

## Setting up the environment

### Web interface

```bash
simpletuner server
```
Access at http://localhost:8001.

### Manual configuration

Run the helper script:

```bash
simpletuner configure
```

Or copy the example and edit manually:

```bash
cp config/config.json.example config/config.json
```

#### Configuration parameters

Key settings for Kandinsky 5 Video:

- `model_family`: `kandinsky5-video`
- `model_flavour`:
  - `t2v-lite-sft-5s`: Lite model, ~5s output. (Default)
  - `t2v-lite-sft-10s`: Lite model, ~10s output.
  - `t2v-pro-sft-5s-hd`: Pro model, ~5s, higher definition training.
  - `t2v-pro-sft-10s-hd`: Pro model, ~10s, higher definition training.
  - *(Pretrain variants also available for all above)*
- `train_batch_size`: `1`. Do not increase this unless you have an A100/H100.
- `validation_resolution`:
  - `512x768` is a safe default for testing.
  - `720x1280` (720p) is possible but heavy.
- `validation_num_video_frames`: **Must be compatible with VAE compression (4x).**
  - For 5s (at ~12-24fps): Use `61` or `49`.
  - Formula: `(frames - 1) % 4 == 0`.
- `validation_guidance`: `5.0`.
- `frame_rate`: Default is 24.

#### Dataset considerations

Video datasets require careful setup. Create `config/multidatabackend.json`:

```json
[
  {
    "id": "my-video-dataset",
    "type": "local",
    "dataset_type": "video",
    "instance_data_dir": "datasets/videos",
    "caption_strategy": "textfile",
    "resolution": 512,
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
    "cache_dir": "cache/text/kandinsky5",
    "disabled": false
  }
]
```

#### Directory setup

```bash
mkdir -p datasets/videos
# Place .mp4 / .mov files here.
# Place corresponding .txt files with same filename for captions.
```

#### Login

```bash
wandb login
huggingface-cli login
```

### Executing the training

```bash
simpletuner train
```

## Notes & troubleshooting tips

### Out of Memory (OOM)

Video training is extremely demanding. If you OOM:

1.  **Reduce Resolution**: Try 480p (`480x854` or similar).
2.  **Reduce Frames**: Drop `validation_num_video_frames` and dataset `num_frames` to `33` or `49`.
3.  **Check Offload**: Ensure `--enable_group_offload` is active.

### Validation Video Quality

- **Black/Noise Videos**: Often caused by `validation_guidance` being too high (> 6.0) or too low (< 2.0). Stick to `5.0`.
- **Motion Jitter**: Check if your dataset frame rate matches the model's trained frame rate (often 24fps).
- **Stagnant/Static Video**: The model might be undertrained or the prompt isn't describing motion. Use prompts like "camera pans right", "zoom in", "running", etc.

### TREAD training

TREAD works for video too and is highly recommended to save compute.

Add to `config.json`:

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```

This can speed up training by ~25-40% depending on the ratio.

### I2V (Image-to-Video) Training

If using `i2v` flavours:
- SimpleTuner automatically extracts the first frame of training videos to use as the conditioning image.
- The pipeline automatically masks the first frame during training.
- Validation requires providing an input image, or SimpleTuner will use the first frame of the validation video generation as the conditioner.
