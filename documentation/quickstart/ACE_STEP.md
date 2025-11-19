# ACE-Step Quickstart

In this example, we'll be training the ACE-Step v1 3.5B audio generation model.

## Overview

ACE-Step is a 3.5B parameter transformer-based flow-matching model designed for high-quality audio synthesis. It supports text-to-audio generation and can be conditioned on lyrics.

## Hardware Requirements

ACE-Step is a 3.5B parameter model, making it relatively lightweight compared to large image generation models like Flux.

- **Minimum:** NVIDIA GPU with 12GB+ VRAM (e.g., 3060, 4070).
- **Recommended:** NVIDIA GPU with 24GB+ VRAM (e.g., 3090, 4090, A10G) for larger batch sizes.
- **Mac:** Supported via MPS on Apple Silicon (Requires ~36GB+ Unified Memory).

> 💡 **Tip:** Using `int8-quanto` quantization allows training on GPUs with less VRAM (e.g., 12GB-16GB) with minimal quality loss.

## Prerequisites

Ensure you have a working Python 3.10+ environment.

```bash
pip install simpletuner
```

## Configuration

It is recommended to keep your configurations organized. We'll create a dedicated folder for this demo.

```bash
mkdir -p config/acestep-training-demo
```

### Critical Settings

Create `config/acestep-training-demo/config.json` with these values:

```json
{
  "model_family": "ace_step",
  "model_type": "lora",
  "model_flavour": "base",
  "pretrained_model_name_or_path": "ACE-Step/ACE-Step-v1-3.5B",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/acestep-training-demo/multidatabackend.json"
}
```

### Validation Settings

Add these to your `config.json` to monitor progress:

- **`validation_prompt`**: A text description of the audio you want to generate (e.g., "A catchy pop song with upbeat drums").
- **`validation_lyrics`**: (Optional) Lyrics for the model to sing.
- **`validation_audio_duration`**: Duration in seconds for validation clips (default: 30.0).
- **`validation_guidance`**: Guidance scale (default: ~3.0 - 5.0).
- **`validation_step_interval`**: How often to generate samples (e.g., every 100 steps).

## Dataset Configuration

ACE-Step requires an **audio-specific** dataset configuration.

### Option 1: Demo Dataset (Hugging Face)

For a quick start, you can use the prepared [ACEStep-Songs preset](/documentation/data_presets/preset_audio_dataset_with_lyrics.md).

Create `config/acestep-training-demo/multidatabackend.json`:

```json
[
  {
    "id": "acestep-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "cache_dir_vae": "cache/vae/{model_family}/acestep-demo-data"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```

### Option 2: Local Audio Files

Create `config/acestep-training-demo/multidatabackend.json`:

```json
[
  {
    "id": "my-audio-dataset",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/my_audio_files",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "disabled": false
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```

### Data Structure

Place your audio files in `datasets/my_audio_files`. SimpleTuner supports a wide range of formats including:
- **Lossless:** `.wav`, `.flac`, `.aiff`, `.alac`
- **Lossy:** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **Note:** To support formats like MP3, AAC, and WMA, you must have **FFmpeg** installed on your system.

For captions and lyrics, place corresponding text files next to your audio files:

- **Audio:** `track_01.wav`
- **Caption (Prompt):** `track_01.txt` (Contains the text description, e.g., "A slow jazz ballad")
- **Lyrics (Optional):** `track_01.lyrics` (Contains the lyrics text)

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```

> 💡 **Advanced:** If your dataset uses a different naming convention (e.g. `_lyrics.txt`), you can customize this in your dataset config:
> ```json
> "audio": {
>   "lyrics_filename_format": "{filename}_lyrics.txt"
> }
> ```

> ⚠️ **Note on Lyrics:** If a `.lyrics` file is not found for a sample, the lyric embeddings will be zeroed out. ACE-Step expects lyric conditioning; training heavily on data without lyrics (instrumentals) may require more training steps for the model to learn to generate high-quality instrumental audio with zeroed lyric inputs.

## Training

Start the training run by specifying your environment:

```bash
simpletuner train env=acestep-training-demo
```

This command tells SimpleTuner to look for `config.json` inside `config/acestep-training-demo/`.

## Troubleshooting

- **Validation Errors:** Ensure you are not trying to use image-centric validation features like `num_validation_images` > 1 (conceptually mapped to batch size for audio) or image-based metrics (CLIP score).
- **Memory Issues:** If running OOM, try reducing `train_batch_size` or enabling `gradient_checkpointing`.
