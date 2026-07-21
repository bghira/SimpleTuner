## Cosmos3 クイックスタート

NVIDIA Cosmos3 の LyCORIS LoKr をトレーニングします。

## モデルメモ

- `model_family`: `cosmos3`
- デフォルト flavour: `nano`
- Flavours:
  - `nano`: `nvidia/Cosmos3-Nano`, 16B
  - `super`: `nvidia/Cosmos3-Super`, 65B
  - `super-t2i`: `nvidia/Cosmos3-Super-Text2Image`, 65B
- SimpleTuner の Cosmos3 は tokenizer ID を直接使います。
- Positive prompts は tokenization 中に Cosmos3 structured JSON captions へ変換されます。
- Negative prompts は JSON に変換しません。
- `text_embeds` backend は追加しません。
- これらの例では `image_embeds` backend も追加しません。
- 画像と動画サンプルは通常の VAE cache を使います。
- 音声付き動画は通常の VAE cache と audio VAE cache を使います。
- action と policy dataset はこのガイドでは扱いません。

## ハードウェア

- `model_flavour: nano` から始めます。
- `mixed_precision: bf16` を使います。
- まず `base_model_precision: no_change` を使います。
- `train_batch_size: 1` を維持します。
- `gradient_checkpointing` を有効にします。

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

## インストール

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

開発インストール:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## サンプル設定

| 例 | Dataset | Media | Backend |
| --- | --- | --- | --- |
| `cosmos3-image.lycoris-lokr` | `RareConcepts/Domokun` | image | `multidatabackend-cosmos3-domokun-512px.json` |
| `cosmos3-video.lycoris-lokr` | `sayakpaul/video-dataset-disney-organized` | video | `multidatabackend-cosmos3-disney-video-480p+49f.json` |
| `cosmos3-video-audio.lycoris-lokr` | `bghira/Synchronised-Drumming-Gemini3Captions` | video + audio | `multidatabackend-cosmos3-drumming-video-audio-480p+49f.json` |

## 必須フィールド

- `model_family`: `cosmos3`
- `model_flavour`: `nano`
- `model_type`: `lora`
- `lora_type`: `lycoris`
- `base_model_precision`: `no_change`
- `mixed_precision`: `bf16`
- `train_batch_size`: `1`
- `gradient_checkpointing`: `true`

## Dataset メモ

### Image

- Dataset: [`RareConcepts/Domokun`](https://huggingface.co/datasets/RareConcepts/Domokun)
- Backend: `config/examples/multidatabackend-cosmos3-domokun-512px.json`
- Backend type: `huggingface`
- Caption strategy: `instanceprompt`
- Text embed cache: 使用しません

### Video

- Dataset: [`sayakpaul/video-dataset-disney-organized`](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)
- Backend: `config/examples/multidatabackend-cosmos3-disney-video-480p+49f.json`
- Backend type: `huggingface`
- Columns: `video`, `prompt`
- Text embed cache: 使用しません
- Audio cache: 使用しません

### Video With Audio

- Dataset: [`bghira/Synchronised-Drumming-Gemini3Captions`](https://huggingface.co/datasets/bghira/Synchronised-Drumming-Gemini3Captions)
- Backend: `config/examples/multidatabackend-cosmos3-drumming-video-audio-480p+49f.json`
- Backend type: `local`
- Files: `.mpeg` videos with adjacent `.txt` captions
- Text embed cache: 使用しません
- Audio cache: video backend から自動生成します

学習前に dataset をダウンロードします:

```bash
huggingface-cli download \
  --repo-type=dataset \
  bghira/Synchronised-Drumming-Gemini3Captions \
  --local-dir datasets/Synchronised-Drumming-Gemini3Captions
```

Backend には次を含めます:

```json
"audio": {
  "auto_split": true,
  "sample_rate": 16000,
  "channels": 1,
  "duration_interval": 3.0,
  "allow_zero_audio": false
}
```

SimpleTuner はこの block から audio dataset を挿入し、audio latents を別の VAE cache に保存します。

## 実行

```bash
simpletuner train example=cosmos3-image.lycoris-lokr
simpletuner train example=cosmos3-video.lycoris-lokr
simpletuner train example=cosmos3-video-audio.lycoris-lokr
```

## Validation

- Image example: `validation_resolution: 512x512`
- Video examples: `validation_resolution: 768x432`
- Video examples: `validation_num_video_frames: 49`
- Audio generation validation には dataset-based validation 設定が必要になる場合があります。

## 参考

- [Diffusers Cosmos3 pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/cosmos3)
- [NVIDIA Cosmos3 collection](https://huggingface.co/collections/nvidia/cosmos3)
