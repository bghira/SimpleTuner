## Cosmos3 快速开始

为 NVIDIA Cosmos3 训练 LyCORIS LoKr。

## 模型说明

- `model_family`: `cosmos3`
- 默认 flavour: `nano`
- Flavours:
  - `nano`: `nvidia/Cosmos3-Nano`, 16B
  - `super`: `nvidia/Cosmos3-Super`, 65B
  - `super-t2i`: `nvidia/Cosmos3-Super-Text2Image`, 65B
- SimpleTuner 中的 Cosmos3 直接使用 tokenizer ID。
- 正向 prompts 会在 tokenization 中转换为 Cosmos3 structured JSON captions。
- 负向 prompts 不会转换为 JSON。
- 不要添加 `text_embeds` backend。
- 这些示例不添加 `image_embeds` backend。
- 图像和视频样本使用普通 VAE cache。
- 带音频的视频使用普通 VAE cache 和 audio VAE cache。
- 本指南不覆盖 action 和 policy dataset。

## 硬件

- 从 `model_flavour: nano` 开始。
- 使用 `mixed_precision: bf16`。
- 先使用 `base_model_precision: no_change`。
- 保持 `train_batch_size: 1`。
- 启用 `gradient_checkpointing`。

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

## 安装

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

开发安装:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## 示例配置

| 示例 | Dataset | Media | Backend |
| --- | --- | --- | --- |
| `cosmos3-image.lycoris-lokr` | `RareConcepts/Domokun` | image | `multidatabackend-cosmos3-domokun-512px.json` |
| `cosmos3-video.lycoris-lokr` | `sayakpaul/video-dataset-disney-organized` | video | `multidatabackend-cosmos3-disney-video-480p+49f.json` |
| `cosmos3-video-audio.lycoris-lokr` | `bghira/Synchronised-Drumming-Gemini3Captions` | video + audio | `multidatabackend-cosmos3-drumming-video-audio-480p+49f.json` |

## 必填字段

- `model_family`: `cosmos3`
- `model_flavour`: `nano`
- `model_type`: `lora`
- `lora_type`: `lycoris`
- `base_model_precision`: `no_change`
- `mixed_precision`: `bf16`
- `train_batch_size`: `1`
- `gradient_checkpointing`: `true`

## Dataset 说明

### Image

- Dataset: [`RareConcepts/Domokun`](https://huggingface.co/datasets/RareConcepts/Domokun)
- Backend: `config/examples/multidatabackend-cosmos3-domokun-512px.json`
- Backend type: `huggingface`
- Caption strategy: `instanceprompt`
- Text embed cache: 不使用

### Video

- Dataset: [`sayakpaul/video-dataset-disney-organized`](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)
- Backend: `config/examples/multidatabackend-cosmos3-disney-video-480p+49f.json`
- Backend type: `huggingface`
- Columns: `video`, `prompt`
- Text embed cache: 不使用
- Audio cache: 不使用

### Video With Audio

- Dataset: [`bghira/Synchronised-Drumming-Gemini3Captions`](https://huggingface.co/datasets/bghira/Synchronised-Drumming-Gemini3Captions)
- Backend: `config/examples/multidatabackend-cosmos3-drumming-video-audio-480p+49f.json`
- Backend type: `local`
- Files: `.mpeg` videos with adjacent `.txt` captions
- Text embed cache: 不使用
- Audio cache: 从 video backend 自动生成

训练前下载 dataset:

```bash
huggingface-cli download \
  --repo-type=dataset \
  bghira/Synchronised-Drumming-Gemini3Captions \
  --local-dir datasets/Synchronised-Drumming-Gemini3Captions
```

Backend 包含:

```json
"audio": {
  "auto_split": true,
  "sample_rate": 16000,
  "channels": 1,
  "duration_interval": 3.0,
  "allow_zero_audio": false
}
```

SimpleTuner 会从该 block 注入 audio dataset，并把 audio latents 存到单独的 VAE cache。

## 运行

```bash
simpletuner train example=cosmos3-image.lycoris-lokr
simpletuner train example=cosmos3-video.lycoris-lokr
simpletuner train example=cosmos3-video-audio.lycoris-lokr
```

## 验证

- Image example: `validation_resolution: 512x512`
- Video examples: `validation_resolution: 768x432`
- Video examples: `validation_num_video_frames: 49`
- 音频生成验证可能需要 dataset-based validation 设置。

## 参考

- [Diffusers Cosmos3 pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/cosmos3)
- [NVIDIA Cosmos3 collection](https://huggingface.co/collections/nvidia/cosmos3)
