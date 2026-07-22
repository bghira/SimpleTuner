# Cosmos3 Quickstart

Train a LyCORIS LoKr for NVIDIA Cosmos3.

## Model Notes

- `model_family`: `cosmos3`
- Default `model_flavour`: `nano`
- Supported flavours:

| Flavour | Hub model | Notes |
| --- | --- | --- |
| `edge` | `nvidia/Cosmos3-Edge` | 4B edge omni model |
| `nano` | `nvidia/Cosmos3-Nano` | 16B omni model |
| `super` | `nvidia/Cosmos3-Super` | 65B omni model |
| `super-t2i` | `nvidia/Cosmos3-Super-Text2Image` | 65B text-to-image model |
| `super-i2v` | `nvidia/Cosmos3-Super-Image2Video` | 65B image-to-video model, silent video |

- Cosmos3 uses tokenizer IDs directly.
- Positive prompts are converted to Cosmos3 JSON captions during tokenization.
- Negative prompts are not converted to JSON.
- Do not add a `text_embeds` backend.
- Image and video samples use the normal VAE cache.
- Video + audio samples use the normal VAE cache and an audio VAE cache.
- `super-i2v` requires conditioning latents.
- Action and policy datasets are not covered here.

## Components

SimpleTuner uses split Cosmos3 transformer components by default:

| Flavour | Reasoner | Generator |
| --- | --- | --- |
| `edge` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-edge` | `SimpleTuner/cosmos3-component-generation-layers-bf16-edge` |
| `nano` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-nano` | `SimpleTuner/cosmos3-component-generation-layers-bf16-nano` |
| `super` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super` |
| `super-t2i` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-t2i` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super-t2i` |
| `super-i2v` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-i2v` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super-i2v` |

- Keep `cosmos3_reasoner_component: auto`.
- Keep `cosmos3_generator_component: auto`.
- Reasoner outputs are cached through the text embed cache path.
- `text_cache_disable: true` reruns the frozen reasoner during training.

## Hardware

- Start with `model_flavour: nano`.
- Use `mixed_precision: bf16`.
- Start with `base_model_precision: no_change`.
- Keep `train_batch_size: 1`.
- Enable `gradient_checkpointing`.
- Use the split generator components for lower transformer load-time memory.

Optional group offload:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

- Streams are CUDA only.
- Do not combine with `--enable_model_cpu_offload`.
- Add `--group_offload_to_disk_path /fast-ssd/simpletuner-offload` when system RAM is limited.

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
| `cosmos3-image-48g.lycoris-lokr` | `nano` | `RareConcepts/Domokun` | image, 48 GB tuned | `multidatabackend-cosmos3-domokun-1024-arb.json` |
| `cosmos3-image-80g.lycoris-lokr` | `nano` | `RareConcepts/Domokun` | image, 80 GB tuned | `multidatabackend-cosmos3-domokun-1024-arb.json` |
| `cosmos3-video.lycoris-lokr` | `nano` | `sayakpaul/video-dataset-disney-organized` | video | `multidatabackend-cosmos3-disney-video-480p+49f.json` |
| `cosmos3-video-audio.lycoris-lokr` | `nano` | `bghira/Synchronised-Drumming-Gemini3Captions` | video + audio | `multidatabackend-cosmos3-drumming-video-audio-480p+49f.json` |
| `cosmos3-super-i2v.lycoris-lokr` | `super-i2v` | `sayakpaul/video-dataset-disney-organized` | image-to-video | `multidatabackend-cosmos3-disney-i2v-480p+49f.json` |

The `48g` and `80g` image examples are memory-size tuned variants of the nano image LoKr recipe. Both use the 1024px aspect-ratio backend. The `48g` config keeps gradient checkpointing enabled with `gradient_checkpointing_interval: 2`; the `80g` config disables gradient checkpointing and enables `flash-attn-3-hub`.

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

Download the dataset before training:

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

The I2V backend marks the video dataset with:

```json
"video": {
  "num_frames": 49,
  "min_frames": 49,
  "is_i2v": true
}
```

SimpleTuner creates the paired strict reference conditioning backend from this flag.

## Run

```bash
simpletuner train example=cosmos3-image.lycoris-lokr
simpletuner train example=cosmos3-image-48g.lycoris-lokr
simpletuner train example=cosmos3-image-80g.lycoris-lokr
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
