# MiniMax Music 3 Quickstart

यह गाइड SimpleTuner में MiniMax Music 3 LoRA training कॉन्फिगर करता है।

## Overview

MiniMax Music 3 caption और lyrics conditioned music generator है। Diffusers layout में Qwen3 autoregressive language model text/audio conditioning बनाता है, flow-matching transformer 128-channel DAV latents पर train होता है, और decoder/vocoder validation audio बनाता है।

SimpleTuner support करता है:

- transformer के लिए LoRA, LyCORIS और full-rank training
- original `dav.pth` autoencoder से raw audio VAECache encoding
- audio dataset metadata से caption, lyrics और duration
- `validation_prompt`, `validation_lyrics`, `validation_audio_duration` और prompt libraries के साथ validation audio
- `lora_format: "comfyui"` के साथ ComfyUI MiniMax Music LoRA import/export
- AnyFlow, TwinFlow, CREPA self-flow और LayerSync

## Hardware Requirements

MiniMax Music 3 में 2.4B flow transformer और 8B Qwen3 AR conditioning model है।

- **Minimum:** conservative LoRA training के लिए 24GB+ VRAM वाली NVIDIA GPU।
- **Recommended:** बड़े rank, लंबे clips और frequent validation के लिए 48GB+ VRAM या CPU/RAM offload।
- **Mac:** कुछ हिस्सों में MPS काम कर सकता है, लेकिन training और validation के लिए practical target CUDA है।

`base_model_precision: "int8-quanto"`, `text_encoder_1_precision: "int8-quanto"` और `gradient_checkpointing: true` से शुरू करें। अगर text encoder bottleneck रहे, तो LoRA rank बढ़ाने से पहले text encoder offload इस्तेमाल करें।

## Prerequisites

SimpleTuner और audio loading के लिए FFmpeg install करें:

```bash
pip install simpletuner
```

Manual install या development setup के लिए [installation documentation](../INSTALL.md) देखें।

## Configuration

Dedicated config folder बनाएँ:

```bash
mkdir -p config/minimaxmusic-training-demo
```

`config/minimaxmusic-training-demo/config.json` बनाएँ:

<details>
<summary>Example config देखें</summary>

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

Ready-made templates:

- `simpletuner/examples/minimaxmusic-music3.peft-lora`
- `simpletuner/examples/minimaxmusic-audio.json`
- `simpletuner/examples/minimaxmusic-prompts.json`

Example run करें:

```bash
simpletuner train example=minimaxmusic-music3.peft-lora
```

## VAECache

MiniMax Music 3 raw audio caching DAV audio autoencoder इस्तेमाल करता है। Recommended SimpleTuner VAE repo `SimpleTuner/MiniMax-Music-3-Encoder` है, जहाँ converted component `audio_vae/` में Diffusers-style loading के लिए रखा गया है।

Upstream `MiniMaxAI/MiniMax-Music3` repo में original `dav.pth` भी है, और SimpleTuner उसे सीधे load कर सकता है। अगर local converted Diffusers directory इस्तेमाल कर रहे हैं, तो checkpoint root में `dav.pth` रखें या `pretrained_vae_model_name_or_path` को ऐसे path या Hub repo पर point करें जिसमें `dav.pth` या `audio_vae/` हो। केवल `vocoder/` subfolder validation decode के लिए काफी है, लेकिन raw audio VAE caching के लिए नहीं।

## Dataset

MiniMax Music 3 को एक **audio** dataset और एक **text embeds** cache backend चाहिए।

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

Local audio files के लिए `.txt` में description और `.lyrics` में lyrics रखें:

```text
datasets/minimaxmusic-audio/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```

## Validation

- **`validation_prompt`**: music description या tags।
- **`validation_lyrics`**: sung lyrics। instrumental validation के लिए empty string इस्तेमाल करें।
- **`validation_audio_duration`**: generated clip duration in seconds।
- **`validation_guidance`**: CFG scale। `1.5` से `2.0` के आसपास शुरू करें।
- **`validation_num_inference_steps`**: sampling steps। `30` के आसपास शुरू करें।
- **`validation_steps`**: कितनी training steps के बाद validation render हो।
- **`validation_prompt_library`**: built-in music caption + lyrics library के लिए `"audio"` इस्तेमाल करें।
- **`user_prompt_library`**: JSON library path। Entries `prompt` या `caption` और optional multiline `lyrics` इस्तेमाल कर सकती हैं।

## Training

```bash
simpletuner train env=minimaxmusic-training-demo
```

Existing MiniMax Music 3 LoRA से शुरू करने के लिए:

```bash
simpletuner train env=minimaxmusic-training-demo --init_lora=/path/to/adapter.safetensors --init_lora_step=0
```

अगर adapter native ComfyUI format में है, तो config में `lora_format: "comfyui"` रखें। SimpleTuner training के समय convert करेगा और उसी format में export करेगा।

## Advanced Features

MiniMax Music 3 SimpleTuner के flow-matching training path का उपयोग करता है, इसलिए AnyFlow, TwinFlow, CREPA self-flow और LayerSync उपलब्ध हैं। पहले standard LoRA से शुरू करें और advanced features एक-एक करके enable करें।

## Troubleshooting

- **`VAE caching requires the original dav.pth checkpoint`**: `SimpleTuner/MiniMax-Music-3-Encoder` या `MiniMaxAI/MiniMax-Music3` इस्तेमाल करें, local checkpoint root में `dav.pth` रखें, या `pretrained_vae_model_name_or_path` को ऐसे location पर point करें जहाँ यह मौजूद हो।
- **Lyrics missing हैं**: backend metadata में `lyrics` confirm करें, या `caption_strategy: "textfile"` के साथ audio files के पास `.lyrics` sidecar रखें।
- **Text embedding या validation OOM**: `validation_audio_duration` घटाएँ, int8 text encoder precision इस्तेमाल करें, या text encoder offload enable करें।
