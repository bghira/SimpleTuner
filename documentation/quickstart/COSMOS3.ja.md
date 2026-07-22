# Cosmos3 クイックスタート

NVIDIA Cosmos3 の LyCORIS LoKr をトレーニングします。

## モデルメモ

- `model_family`: `cosmos3`
- デフォルト `model_flavour`: `nano`
- 対応 flavour:

| Flavour | Hub model | Notes |
| --- | --- | --- |
| `nano` | `nvidia/Cosmos3-Nano` | 16B omni model |
| `super` | `nvidia/Cosmos3-Super` | 65B omni model |
| `super-t2i` | `nvidia/Cosmos3-Super-Text2Image` | 65B text-to-image model |
| `super-i2v` | `nvidia/Cosmos3-Super-Image2Video` | 65B image-to-video model, silent video |

- Cosmos3 は tokenizer ID を直接使います。
- Positive prompts は tokenization 中に Cosmos3 JSON captions へ変換されます。
- Negative prompts は JSON に変換されません。
- `text_embeds` backend は追加しません。
- 画像と動画サンプルは通常の VAE cache を使います。
- 動画 + 音声サンプルは通常の VAE cache と audio VAE cache を使います。
- `super-i2v` には conditioning latents が必要です。
- action と policy dataset は扱いません。

## Components

SimpleTuner はデフォルトで分割済み Cosmos3 transformer components を使います:

| Flavour | Reasoner | Generator |
| --- | --- | --- |
| `nano` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-nano` | `SimpleTuner/cosmos3-component-generation-layers-bf16-nano` |
| `super` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super` |
| `super-t2i` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-t2i` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super-t2i` |
| `super-i2v` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-i2v` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super-i2v` |

- `cosmos3_reasoner_component: auto` のままにします。
- `cosmos3_generator_component: auto` のままにします。
- Reasoner outputs は text embed cache path で cache されます。
- `text_cache_disable: true` は training 中に frozen reasoner を毎回実行します。

## Hardware

- `model_flavour: nano` から始めます。
- `mixed_precision: bf16` を使います。
- `base_model_precision: no_change` から始めます。
- `train_batch_size: 1` を維持します。
- `gradient_checkpointing` を有効にします。
- Transformer load-time memory を減らすには split generator components を使います。

任意の group offload:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

- Streams は CUDA のみです。
- `--enable_model_cpu_offload` と併用しません。
- system RAM が不足する場合は `--group_offload_to_disk_path /fast-ssd/simpletuner-offload` を追加します。

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
- Audio cache: video backend から生成

Training 前に dataset を download します:

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

I2V backend は video dataset に次を設定します:

```json
"video": {
  "num_frames": 49,
  "min_frames": 49,
  "is_i2v": true
}
```

SimpleTuner はこの flag から paired strict reference conditioning backend を作成します。

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
- Super-I2V example: conditioning validation inputs を使います。

## References

- [Diffusers Cosmos3 pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/cosmos3)
- [NVIDIA Cosmos3 collection](https://huggingface.co/collections/nvidia/cosmos3)
