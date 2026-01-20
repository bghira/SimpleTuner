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

### Observed performance and memory (field reports)

- **Baseline settings**: 480p, 17 frames, batch size 2 (minimal video length/resolution).
- **RamTorch (incl. text encoder)**: ~13 GB VRAM used on an AMD 7900XTX.
  - NVIDIA 3090/4090/5090+ should see similar or better VRAM headroom.
- **No offload (int8 TorchAO)**: ~29-30 GB VRAM used; 32 GB hardware recommended.
  - Peak system RAM: ~46 GB when loading bf16 Gemma3 then quantizing to int8 (~32 GB VRAM).
  - Peak system RAM: ~34 GB when loading bf16 LTX-2 transformer then quantizing to int8 (~30 GB VRAM).
- **No offload (full bf16)**: ~48 GB VRAM required for model training without any offload enabled.
- **Throughput**:
  - ~8 sec/step on A100-80G SXM4 (no compile).
  - ~16 sec/step on 7900XTX (local run).
  - ~30 min for 200 steps on A100-80G SXM4.

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
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
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
- `model_flavour`: `dev` (default), `dev-fp4`, or `dev-fp8`.
- `pretrained_model_name_or_path`: `Lightricks/LTX-2` (Hub repo with the combined checkpoint) or a local `.safetensors` file.
- `train_batch_size`: `1`. Do not increase this unless you have an A100/H100.
- `validation_resolution`:
  - `512x768` is a safe default for testing.
  - `720x1280` (720p) is possible but heavy.
- `validation_num_video_frames`: **Must be compatible with VAE compression (4x).**
  - For 5s (at ~12-24fps): Use `61` or `49`.
  - Formula: `(frames - 1) % 4 == 0`.
- `validation_guidance`: `5.0`.
- `frame_rate`: Default is 25.

LTX-2 ships as a single `.safetensors` checkpoint that includes the transformer, video VAE, audio VAE, and vocoder.
SimpleTuner loads from this combined file directly based on `model_flavour` (dev/dev-fp4/dev-fp8).

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
    "audio": {
        "auto_split": true,
        "sample_rate": 16000,
        "channels": 1,
        "duration_interval": 3.0
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

Audio auto-split is enabled by default for video datasets. Add an `audio` block to tune sample rate/channels, set
`audio.auto_split: false` to opt out, or provide a separate audio dataset and link it via `s2v_datasets`. SimpleTuner
will cache audio latents alongside video latents.

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

### Lowest VRAM use config (7900XTX)

Field-tested config that prioritizes minimal VRAM usage on LTX Video 2.

<details>
<summary>View 7900XTX config (lowest VRAM use)</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "checkpoint_step_interval": 100,
  "data_backend_config": "config/ltx2/multidatabackend.json",
  "disable_benchmark": true,
  "dynamo_mode": "",
  "evaluation_type": "none",
  "hub_model_id": "simpletuner-ltxvideo2-19b-t2v-lora-test",
  "learning_rate": 0.00006,
  "lr_warmup_steps": 50,
  "lycoris_config": "config/lycoris_config.json",
  "max_grad_norm": 0.1,
  "max_train_steps": 200,
  "minimum_image_size": 0,
  "model_family": "ltxvideo2",
  "model_flavour": "dev",
  "model_type": "lora",
  "num_train_epochs": 0,
  "offload_during_startup": true,
  "optimizer": "adamw_bf16",
  "output_dir": "output/examples/ltxvideo2-19b-t2v.peft-lora",
  "override_dataset_config": true,
  "ramtorch": true,
  "ramtorch_text_encoder": true,
  "report_to": "none",
  "resolution": 480,
  "scheduled_sampling_reflexflow": false,
  "seed": 42,
  "skip_file_discovery": "",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "example-training-run",
  "train_batch_size": 2,
  "vae_batch_size": 1,
  "vae_enable_patch_conv": true,
  "vae_enable_slicing": true,
  "vae_enable_temporal_roll": true,
  "vae_enable_tiling": true,
  "validation_disable": true,
  "validation_disable_unconditional": true,
  "validation_guidance": 5,
  "validation_num_inference_steps": 40,
  "validation_num_video_frames": 81,
  "validation_prompt": "🟫 is holding a sign that says hello world from ltxvideo2",
  "validation_resolution": "768x512",
  "validation_seed": 42,
  "validation_using_datasets": false
}
```
</details>

### Audio-Only Training

LTX-2 supports **audio-only training** where you train only the audio generation capability without video files. This is useful when you have audio datasets but no corresponding video content.

In audio-only mode:
- Video latents are automatically zeroed out
- Video loss is masked (not computed)
- Only audio generation is trained

#### Audio-only dataset configuration

```json
[
  {
    "id": "my-audio-dataset",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/audio",
    "caption_strategy": "textfile",
    "audio": {
      "audio_only": true,
      "sample_rate": 16000,
      "channels": 1,
      "min_duration_seconds": 1,
      "max_duration_seconds": 30,
      "duration_interval": 3.0
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

The key setting is `audio.audio_only: true`, which tells SimpleTuner to:
1. Use the audio VAE to cache audio latents
2. Generate zero video latents matching the audio duration
3. Mask video loss during training

Place your audio files (`.wav`, `.flac`, `.mp3`, etc.) in the `instance_data_dir` with corresponding `.txt` caption files.

### Validation workflows (T2V vs I2V)

- **T2V (text-to-video)**: Leave `validation_using_datasets: false` and use `validation_prompt` or `validation_prompt_library`.
- **I2V (image-to-video)**: Set `validation_using_datasets: true` and point `eval_dataset_id` at a validation split that provides a reference image. Validation will switch to the image-to-video pipeline and use that image as the conditioner.
- **S2V (audio-conditioned)**: With `validation_using_datasets: true`, point `eval_dataset_id` at a dataset with `s2v_datasets` (or the default `audio.auto_split` behavior). Validation will load cached audio latents automatically.

### Validation adapters (LoRAs)

Lightricks provides several LoRAs that can be applied during validation via `validation_adapter_path` (single) or
`validation_adapter_config` (multiple runs). These repos use nonstandard weight filenames, so include the filename
via `repo_id:weight_name`. See the LTX-2 collection for the latest filenames and related assets:
https://huggingface.co/collections/Lightricks/ltx-2
- `Lightricks/LTX-2-19b-IC-LoRA-Canny-Control:ltx-2-19b-ic-lora-canny-control.safetensors`
- `Lightricks/LTX-2-19b-IC-LoRA-Depth-Control:ltx-2-19b-ic-lora-depth-control.safetensors`
- `Lightricks/LTX-2-19b-IC-LoRA-Detailer:ltx-2-19b-ic-lora-detailer.safetensors`
- `Lightricks/LTX-2-19b-IC-LoRA-Pose-Control:ltx-2-19b-ic-lora-pose-control.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In:ltx-2-19b-lora-camera-control-dolly-in.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out:ltx-2-19b-lora-camera-control-dolly-out.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left:ltx-2-19b-lora-camera-control-dolly-left.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right:ltx-2-19b-lora-camera-control-dolly-right.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down:ltx-2-19b-lora-camera-control-jib-down.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up:ltx-2-19b-lora-camera-control-jib-up.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Static:ltx-2-19b-lora-camera-control-static.safetensors`

Example `validation_adapter_config`:

```json
{
  "validation_adapter_config": [
    { "label": "canny", "path": "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control:ltx-2-19b-ic-lora-canny-control.safetensors" },
    { "label": "pose", "path": "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control:ltx-2-19b-ic-lora-pose-control.safetensors" }
  ]
}
```

For faster validation, apply `Lightricks/LTX-2-19b-distilled-lora-384:ltx-2-19b-distilled-lora-384.safetensors`
as a validation adapter and set `validation_guidance: 1` plus `validation_num_inference_steps: 8`.
