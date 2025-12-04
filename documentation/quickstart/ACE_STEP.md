# ACE-Step Quickstart

In this example, we'll be training the ACE-Step v1 3.5B audio generation model.

## Overview

ACE-Step is a 3.5B parameter transformer-based flow-matching model designed for high-quality audio synthesis. It supports text-to-audio generation and can be conditioned on lyrics.

## Hardware Requirements

ACE-Step is a 3.5B parameter model, making it relatively lightweight compared to large image generation models like Flux.

- **Minimum:** NVIDIA GPU with 12GB+ VRAM (e.g., 3060, 4070).
- **Recommended:** NVIDIA GPU with 24GB+ VRAM (e.g., 3090, 4090, A10G) for larger batch sizes.
- **Mac:** Supported via MPS on Apple Silicon (Requires ~36GB+ Unified Memory).

### Storage Requirements

> ⚠️ **Disk Usage Warning:** The VAE cache for audio models can be substantial. For example, a single 60-second audio clip can result in a ~89MB cached latent file. This caching strategy is used to drastically reduce VRAM requirements during training. Ensure you have sufficient disk space for your dataset's cache.

> 💡 **Tip:** For larger datasets, you can use the `--vae_cache_disable` option to disable writing embeddings to disk. This will implicitly enable on-demand caching, which saves disk space but will increase training time and memory usage as encodings are performed during the training loop.

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

<details>
<summary>View example config</summary>

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
</details>

### Validation Settings

Add these to your `config.json` to monitor progress:

- **`validation_prompt`**: A text description of the audio you want to generate (e.g., "A catchy pop song with upbeat drums").
- **`validation_lyrics`**: (Optional) Lyrics for the model to sing.
- **`validation_audio_duration`**: Duration in seconds for validation clips (default: 30.0).
- **`validation_guidance`**: Guidance scale (default: ~3.0 - 5.0).
- **`validation_step_interval`**: How often to generate samples (e.g., every 100 steps).

### Advanced Experimental Features

<details>
<summary>Show advanced experimental details</summary>


SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> ⚠️ These features increase the computational overhead of training.

</details>

## Dataset Configuration

ACE-Step requires an **audio-specific** dataset configuration.

### Option 1: Demo Dataset (Hugging Face)

For a quick start, you can use the prepared [ACEStep-Songs preset](/documentation/data_presets/preset_audio_dataset_with_lyrics.md).

Create `config/acestep-training-demo/multidatabackend.json`:

<details>
<summary>View example config</summary>

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
</details>

### Option 2: Local Audio Files

Create `config/acestep-training-demo/multidatabackend.json`:

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
</details>

### Data Structure

Place your audio files in `datasets/my_audio_files`. SimpleTuner supports a wide range of formats including:

- **Lossless:** `.wav`, `.flac`, `.aiff`, `.alac`
- **Lossy:** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **Note:** To support formats like MP3, AAC, and WMA, you must have **FFmpeg** installed on your system.

For captions and lyrics, place corresponding text files next to your audio files:

- **Audio:** `track_01.wav`
- **Caption (Prompt):** `track_01.txt` (Contains the text description, e.g., "A slow jazz ballad")
- **Lyrics (Optional):** `track_01.lyrics` (Contains the lyrics text)

<details>
<summary>Example dataset layout</summary>

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```
</details>

> 💡 **Advanced:** If your dataset uses a different naming convention (e.g. `_lyrics.txt`), you can customize this in your dataset config.

<details>
<summary>View custom lyrics filename example</summary>

```json
"audio": {
  "lyrics_filename_format": "{filename}_lyrics.txt"
}
```
</details>

> ⚠️ **Note on Lyrics:** If a `.lyrics` file is not found for a sample, the lyric embeddings will be zeroed out. ACE-Step expects lyric conditioning; training heavily on data without lyrics (instrumentals) may require more training steps for the model to learn to generate high-quality instrumental audio with zeroed lyric inputs.

## Training

Start the training run by specifying your environment:

```bash
simpletuner train env=acestep-training-demo
```

This command tells SimpleTuner to look for `config.json` inside `config/acestep-training-demo/`.

> 💡 **Tip (Continue Training):** To continue fine-tuning from an existing LoRA (e.g. the official ACE-Step checkpoints or community adapters), use the `--init_lora` option:
> ```bash
> simpletuner train env=acestep-training-demo --init_lora=/path/to/existing_lora.safetensors
> ```

### Training the Lyrics Embedder (upstream-style)

The upstream ACE-Step trainer fine-tunes the lyrics embedder alongside the denoiser. To mirror that behaviour in SimpleTuner (full or standard LoRA only):

- Enable it: `lyrics_embedder_train: true`
- Optional overrides (otherwise the main optimizer/scheduler are reused):
  - `lyrics_embedder_lr`
  - `lyrics_embedder_optimizer`
  - `lyrics_embedder_lr_scheduler`

Example snippet:

<details>
<summary>View example config</summary>

```json
{
  "lyrics_embedder_train": true,
  "lyrics_embedder_lr": 5e-5,
  "lyrics_embedder_optimizer": "torch-adamw",
  "lyrics_embedder_lr_scheduler": "cosine_with_restarts"
}
```
</details>
Embedder weights are checkpointed with LoRA saves and restored on resume.

## Troubleshooting

- **Validation Errors:** Ensure you are not trying to use image-centric validation features like `num_validation_images` > 1 (conceptually mapped to batch size for audio) or image-based metrics (CLIP score).
- **Memory Issues:** If running OOM, try reducing `train_batch_size` or enabling `gradient_checkpointing`.

## Migrating from Upstream Trainer

If you are coming from the original ACE-Step training scripts, here is how the parameters map to SimpleTuner's `config.json`:

| Upstream Parameter | SimpleTuner `config.json` | Default / Notes |
| :--- | :--- | :--- |
| `--learning_rate` | `learning_rate` | `1e-4` |
| `--num_workers` | `dataloader_num_workers` | `8` |
| `--max_steps` | `max_train_steps` | `2000000` |
| `--every_n_train_steps` | `checkpointing_steps` | `2000` |
| `--precision` | `mixed_precision` | `"fp16"` or `"bf16"` (use `"no"` for fp32) |
| `--accumulate_grad_batches` | `gradient_accumulation_steps` | `1` |
| `--gradient_clip_val` | `max_grad_norm` | `0.5` |
| `--shift` | `flow_schedule_shift` | `3.0` (Specific to ACE-Step) |

### Converting Raw Data

If you have raw audio/text/lyrics files and want to use the Hugging Face dataset format (as used by the upstream `convert2hf_dataset.py` tool), you can use the resulting dataset directly in SimpleTuner.

The upstream converter produces a dataset with `tags` and `norm_lyrics` columns. To use these, configure your backend like this:

<details>
<summary>View example config</summary>

```json
{
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "path/to/converted/dataset",
    "config": {
        "audio_caption_fields": ["tags"],
        "lyrics_column": "norm_lyrics"
    }
}
```
</details>
