# Cosmos3 快速开始

为 NVIDIA Cosmos3 训练 LyCORIS LoKr。

## 模型说明

- `model_family`: `cosmos3`
- 默认 `model_flavour`: `nano`
- 支持的 flavours:

| Flavour | Hub model | Notes |
| --- | --- | --- |
| `nano` | `nvidia/Cosmos3-Nano` | 16B omni model |
| `super` | `nvidia/Cosmos3-Super` | 65B omni model |
| `super-t2i` | `nvidia/Cosmos3-Super-Text2Image` | 65B text-to-image model |
| `super-i2v` | `nvidia/Cosmos3-Super-Image2Video` | 65B image-to-video model, silent video |

- Cosmos3 直接使用 tokenizer ID。
- 正向 prompts 会在 tokenization 中转换为 Cosmos3 JSON captions。
- 负向 prompts 不会转换为 JSON。
- 不要添加 `text_embeds` backend。
- 图像和视频样本使用普通 VAE cache。
- 视频 + 音频样本使用普通 VAE cache 和 audio VAE cache。
- `super-i2v` 需要 conditioning latents。
- 本指南不覆盖 action 和 policy dataset。

## Components

SimpleTuner 默认使用拆分的 Cosmos3 transformer components:

| Flavour | Reasoner | Generator |
| --- | --- | --- |
| `nano` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-nano` | `SimpleTuner/cosmos3-component-generation-layers-bf16-nano` |
| `super` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super` |
| `super-t2i` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-t2i` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super-t2i` |
| `super-i2v` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-i2v` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super-i2v` |

- 保持 `cosmos3_reasoner_component: auto`。
- 保持 `cosmos3_generator_component: auto`。
- Reasoner outputs 通过 text embed cache path 缓存。
- `text_cache_disable: true` 会在 training 中重新运行 frozen reasoner。

## Hardware

- 从 `model_flavour: nano` 开始。
- 使用 `mixed_precision: bf16`。
- 从 `base_model_precision: no_change` 开始。
- 保持 `train_batch_size: 1`。
- 启用 `gradient_checkpointing`。
- 使用拆分的 generator components 来降低 transformer 加载时内存。

可选 group offload:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

- Streams 仅适用于 CUDA。
- 不要与 `--enable_model_cpu_offload` 一起使用。
- 系统 RAM 不足时添加 `--group_offload_to_disk_path /fast-ssd/simpletuner-offload`。

## Installation

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Development install:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Example Configs

| Example | Flavour | Dataset | Media | Backend |
| --- | --- | --- | --- | --- |
| `cosmos3-image.lycoris-lokr` | `nano` | `RareConcepts/Domokun` | image | `multidatabackend-cosmos3-domokun-512px.json` |
| `cosmos3-video.lycoris-lokr` | `nano` | `sayakpaul/video-dataset-disney-organized` | video | `multidatabackend-cosmos3-disney-video-480p+49f.json` |
| `cosmos3-video-audio.lycoris-lokr` | `nano` | `bghira/Synchronised-Drumming-Gemini3Captions` | video + audio | `multidatabackend-cosmos3-drumming-video-audio-480p+49f.json` |
| `cosmos3-super-i2v.lycoris-lokr` | `super-i2v` | `sayakpaul/video-dataset-disney-organized` | image-to-video | `multidatabackend-cosmos3-disney-i2v-480p+49f.json` |

## Required Fields

- `model_family`: `cosmos3`
- `model_type`: `lora`
- `lora_type`: `lycoris`
- `base_model_precision`: `no_change`
- `mixed_precision`: `bf16`
- `train_batch_size`: `1`
- `gradient_checkpointing`: `true`

## Dataset Notes

### Image

- Dataset: [`RareConcepts/Domokun`](https://huggingface.co/datasets/RareConcepts/Domokun)
- Backend: `config/examples/multidatabackend-cosmos3-domokun-512px.json`
- Backend type: `huggingface`
- Caption strategy: `instanceprompt`
- Text embed cache: reasoner cache only
- Audio cache: not used

### Video

- Dataset: [`sayakpaul/video-dataset-disney-organized`](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)
- Backend: `config/examples/multidatabackend-cosmos3-disney-video-480p+49f.json`
- Backend type: `huggingface`
- Columns: `video`, `prompt`
- Text embed cache: reasoner cache only
- Audio cache: not used

### Video With Audio

- Dataset: [`bghira/Synchronised-Drumming-Gemini3Captions`](https://huggingface.co/datasets/bghira/Synchronised-Drumming-Gemini3Captions)
- Backend: `config/examples/multidatabackend-cosmos3-drumming-video-audio-480p+49f.json`
- Backend type: `local`
- Files: `.mpeg` videos with adjacent `.txt` captions
- Text embed cache: reasoner cache only
- Audio cache: generated from the video backend

训练前下载 dataset:

```bash
huggingface-cli download \
  --repo-type=dataset \
  bghira/Synchronised-Drumming-Gemini3Captions \
  --local-dir datasets/Synchronised-Drumming-Gemini3Captions
```

Audio backend block:

```json
"audio": {
  "auto_split": true,
  "sample_rate": 16000,
  "channels": 1,
  "duration_interval": 3.0,
  "allow_zero_audio": false
}
```

### Super-I2V

- Dataset: [`sayakpaul/video-dataset-disney-organized`](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)
- Backend: `config/examples/multidatabackend-cosmos3-disney-i2v-480p+49f.json`
- Backend type: `huggingface`
- Columns: `video`, `prompt`
- `video.is_i2v`: `true`
- Conditioning type: `reference_strict`
- Conditioning latents: required
- Audio cache: not used

I2V backend 在 video dataset 上设置:

```json
"video": {
  "num_frames": 49,
  "min_frames": 49,
  "is_i2v": true
}
```

SimpleTuner 会根据这个 flag 创建 paired strict reference conditioning backend。

## Run

```bash
simpletuner train example=cosmos3-image.lycoris-lokr
simpletuner train example=cosmos3-video.lycoris-lokr
simpletuner train example=cosmos3-video-audio.lycoris-lokr
simpletuner train example=cosmos3-super-i2v.lycoris-lokr
```

## Validation

- Image example: `validation_resolution: 512x512`
- Video examples: `validation_resolution: 768x432`
- Video examples: `validation_num_video_frames: 49`
- Super-I2V example: uses conditioning validation inputs.

## References

- [Diffusers Cosmos3 pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/cosmos3)
- [NVIDIA Cosmos3 collection](https://huggingface.co/collections/nvidia/cosmos3)
