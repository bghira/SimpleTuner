# HeartMuLa Quickstart

In this example, we'll be training the HeartMuLa oss 3B audio generation model.

## Overview

HeartMuLa is a 3B parameter autoregressive transformer that predicts discrete audio tokens from tags and lyrics. The tokens are decoded with HeartCodec to produce waveforms.

## Hardware Requirements

HeartMuLa is a 3B parameter model, making it relatively lightweight compared to large image generation models like Flux.

- **Minimum:** NVIDIA GPU with 12GB+ VRAM (e.g., 3060, 4070).
- **Recommended:** NVIDIA GPU with 24GB+ VRAM (e.g., 3090, 4090, A10G) for larger batch sizes.
- **Mac:** Supported via MPS on Apple Silicon (Requires ~36GB+ Unified Memory).

### Storage Requirements

> ⚠️ **Token Dataset Warning:** HeartMuLa trains on precomputed audio tokens. SimpleTuner does not build tokens during training, so your dataset must supply `audio_tokens` or `audio_tokens_path` metadata. Token files can be large, so plan disk space accordingly.

> 💡 **Tip:** Using `int8-quanto` quantization allows training on GPUs with less VRAM (e.g., 12GB-16GB) with minimal quality loss.

## Prerequisites

Ensure you have a working Python 3.10+ environment.

```bash
pip install simpletuner
```

## Configuration

It is recommended to keep your configurations organized. We'll create a dedicated folder for this demo.

```bash
mkdir -p config/heartmula-training-demo
```

### Critical Settings

Create `config/heartmula-training-demo/config.json` with these values:

<details>
<summary>View example config</summary>

```json
{
  "model_family": "heartmula",
  "model_type": "lora",
  "model_flavour": "3b",
  "pretrained_model_name_or_path": "HeartMuLa/HeartMuLa-oss-3B",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/heartmula-training-demo/multidatabackend.json"
}
```
</details>

### Validation Settings

Add these to your `config.json` to monitor progress:

- **`validation_prompt`**: Tags or a text description of the audio (e.g., "Upbeat pop with bright synths").
- **`validation_lyrics`**: (Optional) Lyrics for the model to sing. Use an empty string for instrumentals.
- **`validation_audio_duration`**: Duration in seconds for validation clips (default: 30.0).
- **`validation_guidance`**: Guidance scale (start around 1.5 - 3.0).
- **`validation_step_interval`**: How often to generate samples (e.g., every 100 steps).

### Advanced Experimental Features

<details>
<summary>Show advanced experimental details</summary>


SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> ⚠️ These features increase the computational overhead of training.

</details>

## Dataset Configuration

HeartMuLa requires an **audio-specific** dataset with precomputed tokens.

Each sample must provide:

- `tags` (string)
- `lyrics` (string; can be empty)
- `audio_tokens` or `audio_tokens_path`

Token arrays must be 2D with shape `[frames, num_codebooks]` or `[num_codebooks, frames]`.

> 💡 **Note:** HeartMuLa does not use a separate text encoder, so a text-embeds backend is not required.

### Option 1: Hugging Face Dataset (Tokens in Columns)

Create `config/heartmula-training-demo/multidatabackend.json`:

<details>
<summary>View example config</summary>

```json
[
  {
    "id": "heartmula-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "your-org/heartmula-audio-tokens",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "config": {
      "audio_caption_fields": ["tags"],
      "lyrics_column": "lyrics"
    }
  }
]
```
</details>

Make sure your dataset includes `audio_tokens` or `audio_tokens_path` columns alongside the text fields.

### Option 2: Local Audio Files + Token Metadata

Create `config/heartmula-training-demo/multidatabackend.json`:

<details>
<summary>View example config</summary>

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
  }
]
```
</details>

Ensure your metadata backend supplies `audio_tokens` or `audio_tokens_path` for every sample.

### Data Structure

Place your audio files in `datasets/my_audio_files`. SimpleTuner supports a wide range of formats including:

- **Lossless:** `.wav`, `.flac`, `.aiff`, `.alac`
- **Lossy:** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **Note:** To support formats like MP3, AAC, and WMA, you must have **FFmpeg** installed on your system.

For tags and lyrics, place corresponding text files next to your audio files if you use `caption_strategy: textfile`:

- **Audio:** `track_01.wav`
- **Tags (Prompt):** `track_01.txt` (Contains the text description, e.g., "A slow jazz ballad")
- **Lyrics (Optional):** `track_01.lyrics` (Contains the lyrics text)

Provide token arrays via metadata (for example, `audio_tokens_path` entries pointing at `.npy` or `.npz` files).

<details>
<summary>Example dataset layout</summary>

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
├── track_01.lyrics
└── track_01.tokens.npy
```
</details>

> ⚠️ **Note on Lyrics:** HeartMuLa expects a lyrics string for every sample. For instrumental data, provide an empty string rather than omitting the field.

## Training

Start the training run by specifying your environment:

```bash
simpletuner train env=heartmula-training-demo
```

This command tells SimpleTuner to look for `config.json` inside `config/heartmula-training-demo/`.

> 💡 **Tip (Continue Training):** To continue fine-tuning from an existing LoRA, use the `--init_lora` option:
> ```bash
> simpletuner train env=heartmula-training-demo --init_lora=/path/to/existing_lora.safetensors
> ```

## Troubleshooting

- **Validation Errors:** Ensure you are not trying to use image-centric validation features like `num_validation_images` > 1 (conceptually mapped to batch size for audio) or image-based metrics (CLIP score).
- **Memory Issues:** If running OOM, try reducing `train_batch_size` or enabling `gradient_checkpointing`.
