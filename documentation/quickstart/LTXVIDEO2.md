# LTX Video 2 Quickstart

In this example, we'll train an LTX Video 2 LoRA using the LTX-2 video/audio VAEs and a Gemma3 text encoder.

## Hardware requirements

LTX Video 2 is a heavy **19B** model. It combines:
1.  **Gemma3**: The text encoder.
2.  **LTX-2 Video VAE** (plus the Audio VAE when conditioning on audio).
3.  **19B Video Transformer**: A large DiT backbone.

This setup is VRAM-intensive, and the VAE pre-caching step can spike memory usage.

- **Single-GPU training**: Start with `train_batch_size: 1` and enable group offload.
  - **Note**: The initial **VAE pre-caching step** can require more VRAM. You may need CPU offloading or a larger GPU just for the caching phase.
  - **Tip**: Set `"offload_during_startup": true` in your `config.json` to ensure the VAE and text encoder are not loaded to the GPU at the same time, which significantly reduces pre-caching memory pressure.
- **Multi-GPU training**: **FSDP2** or aggressive **Group Offload** is recommended if you need more headroom.
- **System RAM**: 64GB+ is recommended for larger runs; more RAM helps with caching.

### Memory offloading (Critical)

For most single-GPU setups training LTX Video 2, you should enable grouped offloading. It is optional but recommended to keep VRAM headroom for larger batches/resolutions.

Add this to your `config.json`:

<details>
<summary>View example config</summary>

```json
{
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "group_offload_blocks_per_group": 1,
  "group_offload_use_stream": true
}
```
</details>

## Prerequisites

Ensure Python 3.12 is installed.

```bash
python --version
```

## Installation

```bash
pip install simpletuner[cuda]
```

See [INSTALL.md](../INSTALL.md) for advanced installation options.

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

Key settings for LTX Video 2:

- `model_family`: `ltxvideo2`
- `model_flavour`: `2.0` (default)
- `pretrained_model_name_or_path`: `Lightricks/LTX-2` (optional override)
- `train_batch_size`: `1`. Do not increase this unless you have an A100/H100.
- `validation_resolution`:
  - `512x768` is a safe default for testing.
  - `720x1280` (720p) is possible but heavy.
- `validation_num_video_frames`: **Must be compatible with VAE compression (4x).**
  - For 5s (at ~12-24fps): Use `61` or `49`.
  - Formula: `(frames - 1) % 4 == 0`.
- `validation_guidance`: `5.0`.
- `frame_rate`: Default is 25.

### Optional: VRAM optimizations

If you need more VRAM headroom:
- **Musubi block swap**: Set `musubi_blocks_to_swap` (try `4-8`) and optionally `musubi_block_swap_device` (default `cpu`) to stream the last transformer blocks from CPU. Expect lower throughput but lower peak VRAM.
- **VAE patch convolution**: Set `--vae_enable_patch_conv=true` to enable temporal chunking in the LTX-2 VAE; expect a small speed hit but lower peak VRAM.
- **VAE temporal roll**: Set `--vae_enable_temporal_roll=true` for more aggressive temporal chunking (larger speed hit).
- **VAE tiling**: Set `--vae_enable_tiling=true` to tile VAE encode/decode for large resolutions.

### Optional: CREPA temporal regularizer

To reduce flicker and keep subjects stable across frames:
- In **Training → Loss functions**, enable **CREPA**.
- Recommended starting values: **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- Keep the default vision encoder (`dinov2_vitg14`, size `518`) unless you need a smaller one (`dinov2_vits14` + `224`).
- Requires network (or a cached torch hub) to fetch DINOv2 weights the first time.
- Only enable **Drop VAE Encoder** if you are training entirely from cached latents; otherwise leave it off.

### Advanced Experimental Features

<details>
<summary>Show advanced experimental details</summary>


SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> ⚠️ These features increase the computational overhead of training.

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
        "frame_rate": 25,
        "bucket_strategy": "aspect_ratio"
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo2",
    "disabled": false
  }
]
```

In the `video` subsection:
- `num_frames`: Target frame count for training.
- `min_frames`: Minimum video length (shorter videos are discarded).
- `max_frames`: Maximum video length filter.
- `bucket_strategy`: How videos are grouped into buckets:
  - `aspect_ratio` (default): Group by spatial aspect ratio only.
  - `resolution_frames`: Group by `WxH@F` format (e.g., `1920x1080@61`) for mixed-resolution/duration datasets.
- `frame_interval`: When using `resolution_frames`, round frame counts to this interval.

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

#### Directory setup

```bash
mkdir -p datasets/videos
</details>

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
- **Motion Jitter**: Check if your dataset frame rate matches the model's trained frame rate (often 25fps).
- **Stagnant/Static Video**: The model might be undertrained or the prompt isn't describing motion. Use prompts like "camera pans right", "zoom in", "running", etc.

### TREAD training

TREAD works for video too and is highly recommended to save compute.

Add to `config.json`:

<details>
<summary>View example config</summary>

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
</details>

This can speed up training by ~25-40% depending on the ratio.

### Validation workflows (T2V vs I2V)

- **T2V (text-to-video)**: Leave `validation_using_datasets: false` and use `validation_prompt` or `validation_prompt_library`.
- **I2V (image-to-video)**: Set `validation_using_datasets: true` and point `eval_dataset_id` at a validation split that provides a reference image. Validation will switch to the image-to-video pipeline and use that image as the conditioner.
