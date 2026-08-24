# MiniMax Music 3 Quickstart

This guide configures SimpleTuner for MiniMax Music 3 LoRA training.

## Overview

MiniMax Music 3 is a caption- and lyrics-conditioned music generator. The Diffusers layout uses a Qwen3 autoregressive language model for text/audio conditioning, a flow-matching transformer over 128-channel DAV latents, and a decoder/vocoder for waveform validation.

SimpleTuner supports:

- LoRA, LyCORIS, and full-rank transformer training
- VAECache encoding from raw audio through the original `dav.pth` autoencoder
- caption, lyrics, and duration metadata from audio datasets
- validation audio generation with `validation_prompt`, `validation_lyrics`, `validation_audio_duration`, and prompt libraries
- ComfyUI MiniMax Music LoRA import/export with `lora_format: "comfyui"`
- AnyFlow, TwinFlow, CREPA self-flow, and LayerSync

## Hardware Requirements

MiniMax Music 3 has a 2.4B flow transformer and an 8B Qwen3 AR text/audio conditioning model.

- **Minimum:** NVIDIA GPU with 24GB+ VRAM for conservative LoRA training.
- **Recommended:** 48GB+ VRAM, or CPU/RAM offload for larger rank, longer clips, and frequent validation.
- **Mac:** MPS may work for some components, but CUDA is the practical target for training and validation.

Start with `base_model_precision: "int8-quanto"`, `text_encoder_1_precision: "int8-quanto"`, and `gradient_checkpointing: true`. If the text encoder remains the bottleneck, use text encoder offload before increasing LoRA rank.

## Prerequisites

Install SimpleTuner and FFmpeg for audio loading:

```bash
pip install simpletuner
```

For manual installation or development setup, see the [installation documentation](../INSTALL.md).

## Configuration

Create a dedicated configuration folder:

```bash
mkdir -p config/minimaxmusic-training-demo
```

Create `config/minimaxmusic-training-demo/config.json`:

<details>
<summary>View example config</summary>

```json
{
  "model_family": "minimaxmusic",
  "model_type": "lora",
  "model_flavour": "music3",
  "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-Music3",
  "pretrained_vae_model_name_or_path": "SimpleTuner/MiniMax-Music-3-Encoder",
  "resolution": 512,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "int8-quanto",
  "gradient_checkpointing": true,
  "lora_rank": 64,
  "lora_format": "comfyui",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "train_batch_size": 1,
  "vae_batch_size": 1,
  "data_backend_config": "config/minimaxmusic-training-demo/multidatabackend.json",
  "validation_prompt": "bright synth pop with clean vocal melody and crisp percussion",
  "validation_lyrics": "[verse]\nturning sparks into a skyline\n[chorus]\nwe keep singing through the night",
  "validation_audio_duration": 30,
  "validation_guidance": 1.7,
  "validation_num_inference_steps": 30,
  "validation_steps": 50,
  "validation_disable_unconditional": true
}
```
</details>

Ready-made template files are available at:

- `simpletuner/examples/minimaxmusic-music3.peft-lora`
- `simpletuner/examples/minimaxmusic-audio.json`
- `simpletuner/examples/minimaxmusic-prompts.json`

You can launch the example with:

```bash
simpletuner train example=minimaxmusic-music3.peft-lora
```

## VAECache

MiniMax Music 3 raw audio caching uses the DAV audio autoencoder. The recommended SimpleTuner VAE repository is `SimpleTuner/MiniMax-Music-3-Encoder`, which stores the converted component in `audio_vae/` for Diffusers-style loading.

The upstream `MiniMaxAI/MiniMax-Music3` repository also includes the original `dav.pth`, and SimpleTuner can load that directly. If you use a converted local Diffusers directory, keep `dav.pth` at the checkpoint root or set `pretrained_vae_model_name_or_path` to a path or Hub repository containing `dav.pth` or an `audio_vae/` subfolder. A decoder-only `vocoder/` subfolder is enough for validation decode, but not for raw audio VAE caching.

## Dataset Configuration

MiniMax Music 3 requires an **audio** dataset plus a **text embeds** cache backend.

### Demo Dataset

Create `config/minimaxmusic-training-demo/multidatabackend.json`:

<details>
<summary>View example config</summary>

```json
[
  {
    "id": "minimaxmusic-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "audio": {
      "bucket_strategy": "duration",
      "duration_interval": 3.0,
      "max_duration_seconds": 30
    },
    "cache_dir_vae": "cache/vae/{model_family}/minimaxmusic-demo-data"
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

### Local Audio Files

For your own files, use a local audio backend:

```json
[
  {
    "id": "my-minimaxmusic-audio",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/minimaxmusic-audio",
    "metadata_backend": "discovery",
    "caption_strategy": "textfile",
    "audio": {
      "bucket_strategy": "duration",
      "duration_interval": 3.0,
      "max_duration_seconds": 60,
      "lyrics_filename_format": "{filename}.lyrics"
    },
    "cache_dir_vae": "cache/vae/{model_family}/my-minimaxmusic-audio"
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

Use this layout for local files:

```text
datasets/minimaxmusic-audio/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```

The `.txt` file is the music description. The `.lyrics` file is passed into the Qwen3 conditioning path. Structure tags such as `[verse]` and `[chorus]` are useful and should be on their own lines.

## Validation Settings

- **`validation_prompt`**: the music description or tags.
- **`validation_lyrics`**: lyrics for sung generations. Use an empty string for instrumental validation.
- **`validation_audio_duration`**: generated clip duration in seconds.
- **`validation_guidance`**: classifier-free guidance scale. Start near `1.5` to `2.0`.
- **`validation_num_inference_steps`**: validation sampling steps. Start around `30`.
- **`validation_steps`**: how often to render validation audio.
- **`validation_prompt_library`**: set to `"audio"` for the built-in music caption + lyrics library.
- **`user_prompt_library`**: path to a JSON library. Entries can use `prompt` or `caption`, plus optional multiline `lyrics`.

Example `user_prompt_library.json` entry:

```json
{
  "neon_pop_hook": {
    "caption": "neon synth pop, 120 bpm, bright lead vocal, pulsing bass, glossy drums",
    "lyrics": "[verse]\nwe found sparks in the city rain\n[chorus]\nlight it up and let it go"
  }
}
```

## Training

Start training:

```bash
simpletuner train env=minimaxmusic-training-demo
```

To start from an existing MiniMax Music 3 LoRA:

```bash
simpletuner train env=minimaxmusic-training-demo --init_lora=/path/to/adapter.safetensors --init_lora_step=0
```

If the adapter is in native ComfyUI format, keep `lora_format: "comfyui"` in the config. SimpleTuner will convert it for training and export in the same format.

## Advanced Features

MiniMax Music 3 uses SimpleTuner's flow-matching training path, so the same advanced tools are available:

- **AnyFlow** for endpoint-aware flow distillation
- **TwinFlow** for two-time consistency training
- **CREPA self-flow** for masked self-flow regularization
- **LayerSync** for hidden-state consistency

Start with standard LoRA first. Add one advanced feature at a time and keep validation clips short until memory use is understood.

## Language Model (AR Stage) Training

The Qwen3 language model that plans MiniMax Music 3's semantic codes can be trained instead of the music DiT — useful for dreambooth-style trigger words that bind a musical style to a keyword.

See [fiona crapple](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple) for a complete LM LoRA training example produced with this mode, including its settings, checkpoints, and audio comparisons.

```json
{
  "minimax_music_train_component": "language_model",
  "minimax_music_lm_max_frames": 0,
  "minimax_music_lm_window_mode": "prefix"
}
```

Requirements and differences from DiT training:

- Each dataset sample must provide `prompt` (or `tags`), `lyrics`, and `audio_tokens_path` metadata pointing at a `.pt` file of raw per-codebook RVQ codes shaped `[frames, codebooks]` (semantic codes `< 16384`, residual codes `< audio_vocab_size`, no vocabulary offsets baked in). Export them with `precompute_rvq_codes.py --raw-codes` from the dedicated `minimax-music3-latent-replanner` repository.
- The loss is next-token cross-entropy on the semantic codebook, masked to audio positions; the RVQ depth decoder stays frozen and supplies the residual-code input embeddings.
- Only standard PEFT LoRA is supported and `lora_format: "comfyui"` is rejected. Checkpoints save `pytorch_lora_weights.safetensors` with `language_model.`-prefixed adapter keys.
- In-trainer validation audio is disabled in this mode; render from saved checkpoints with the standard generation stack instead.
- No VAE or text-embed caching happens in this mode — training reads tokens directly, so `cache_dir_vae` and text embed backends are not used.
- Put your trigger keyword (e.g. `"fiona crapple"`) in the caption/`prompt` field of every sample; keep lyrics verbatim.
- For short capped runs, set `minimax_music_lm_window_mode: "random"` to sample positioned RVQ windows instead of always training on intros. Random windows add their start/end/duration to the prompt and omit full-track lyrics unless the sample provides `lyrics_window`.
- For song-structure training, use `minimax_music_lm_window_mode: "continuation"`. The final `minimax_music_lm_target_frames` receive loss while earlier visible frames are masked causal context. `full` continuation crops always begin at the song start; `random` continuation crops can move through the track while retaining at least one native 128-frame context segment. Minimum and maximum visible durations snap to the model's native 128-frame/5.12-second interval; a maximum of `0` uses the available track length.

A bounded full-prefix continuation configuration looks like this:

```json
{
  "minimax_music_lm_window_mode": "continuation",
  "minimax_music_lm_target_frames": 128,
  "minimax_music_lm_continuation_crop_mode": "full",
  "minimax_music_lm_min_duration_seconds": 5.12,
  "minimax_music_lm_max_duration_seconds": 30.72
}
```

Change the crop mode to `random` to train positioned continuations within the same memory cap. Positioned crops add their time range to the prompt and omit full-track lyrics unless `lyrics_window` is available. This sampling happens during LM collate over the complete cached RVQ sequence; it does not alter the dataset audio or cache.
- **Prior preservation**: add a second audio backend with `is_regularisation_data: true` containing unrelated songs
  (empty lyrics are allowed). On those batches the loss targets the frozen base model's own next-token distribution
  instead of the ground-truth codes, so the LoRA stays surgical: unrelated captions keep predicting exactly as the
  base model would, which sharply reduces style bleed.

## Troubleshooting

- **`VAE caching requires the original dav.pth checkpoint`**: use `SimpleTuner/MiniMax-Music-3-Encoder`, `MiniMaxAI/MiniMax-Music3`, keep `dav.pth` at your local checkpoint root, or set `pretrained_vae_model_name_or_path` to a location containing it.
- **Missing lyrics**: ensure the backend metadata contains `lyrics`, or place `.lyrics` sidecars next to audio files when using `caption_strategy: "textfile"`.
- **Text embedding or validation OOM**: lower validation duration, use int8 text encoder precision, or enable text encoder offload.

## Related MiniMax Music 3 experiments

- [Open RVQ encoders](https://huggingface.co/SimpleTuner/open-rvq-encoder-minimax-music3)
- [RVQ reference-audio integration](https://github.com/bghira/minimax-music3-rvq-reference-audio)
- [Fiona Crapple LM LoRA](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple)
- [Latent refiner](https://github.com/bghira/minimax-music3-latent-refiner) and [v0.10 weights](https://huggingface.co/terminusresearch/minimax-music3-latent-refiner-v0.10)
- [Latent replanner](https://github.com/bghira/minimax-music3-latent-replanner) and [experiment log](https://huggingface.co/terminusresearch/minimax-music3-replanner-experiment)
