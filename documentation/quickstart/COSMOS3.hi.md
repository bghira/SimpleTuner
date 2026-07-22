# Cosmos3 क्विकस्टार्ट

NVIDIA Cosmos3 के लिए LyCORIS LoKr ट्रेन करें।

## मॉडल नोट्स

- `model_family`: `cosmos3`
- default `model_flavour`: `nano`
- supported flavours:

| Flavour | Hub model | Notes |
| --- | --- | --- |
| `edge` | `nvidia/Cosmos3-Edge` | 4B edge omni मॉडल |
| `nano` | `nvidia/Cosmos3-Nano` | 16B omni model |
| `super` | `nvidia/Cosmos3-Super` | 65B omni model |
| `super-t2i` | `nvidia/Cosmos3-Super-Text2Image` | 65B text-to-image model |
| `super-i2v` | `nvidia/Cosmos3-Super-Image2Video` | 65B image-to-video model, silent video |

- Cosmos3 tokenizer IDs सीधे उपयोग करता है।
- Positive prompts tokenization के दौरान Cosmos3 JSON captions में बदलते हैं।
- Negative prompts JSON में नहीं बदलते।
- `text_embeds` backend न जोड़ें।
- Image और video samples normal VAE cache उपयोग करते हैं।
- Video + audio samples normal VAE cache और audio VAE cache उपयोग करते हैं।
- `super-i2v` को conditioning latents चाहिए।
- Action और policy datasets यहां शामिल नहीं हैं।

## Components

SimpleTuner default रूप से split Cosmos3 transformer components उपयोग करता है:

| Flavour | Reasoner | Generator |
| --- | --- | --- |
| `edge` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-edge` | `SimpleTuner/cosmos3-component-generation-layers-bf16-edge` |
| `nano` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-nano` | `SimpleTuner/cosmos3-component-generation-layers-bf16-nano` |
| `super` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super` |
| `super-t2i` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-t2i` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super-t2i` |
| `super-i2v` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-i2v` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super-i2v` |

- `cosmos3_reasoner_component: auto` रखें।
- `cosmos3_generator_component: auto` रखें।
- Reasoner outputs text embed cache path से cache होते हैं।
- `text_cache_disable: true` training में frozen reasoner फिर चलाता है।

## Hardware

- `model_flavour: nano` से शुरू करें।
- `mixed_precision: bf16` उपयोग करें।
- `base_model_precision: no_change` से शुरू करें।
- `train_batch_size: 1` रखें।
- `gradient_checkpointing` enable करें।
- Transformer load-time memory घटाने के लिए split generator components उपयोग करें।

Optional group offload:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

- Streams केवल CUDA हैं।
- `--enable_model_cpu_offload` के साथ न मिलाएं।
- System RAM कम हो तो `--group_offload_to_disk_path /fast-ssd/simpletuner-offload` जोड़ें।

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
| `cosmos3-image-48g.lycoris-lokr` | `nano` | `RareConcepts/Domokun` | इमेज, 48 GB tuned | `multidatabackend-cosmos3-domokun-1024-arb.json` |
| `cosmos3-image-80g.lycoris-lokr` | `nano` | `RareConcepts/Domokun` | इमेज, 80 GB tuned | `multidatabackend-cosmos3-domokun-1024-arb.json` |
| `cosmos3-video.lycoris-lokr` | `nano` | `sayakpaul/video-dataset-disney-organized` | video | `multidatabackend-cosmos3-disney-video-480p+49f.json` |
| `cosmos3-video-audio.lycoris-lokr` | `nano` | `bghira/Synchronised-Drumming-Gemini3Captions` | video + audio | `multidatabackend-cosmos3-drumming-video-audio-480p+49f.json` |
| `cosmos3-super-i2v.lycoris-lokr` | `super-i2v` | `sayakpaul/video-dataset-disney-organized` | image-to-video | `multidatabackend-cosmos3-disney-i2v-480p+49f.json` |

`48g` और `80g` इमेज उदाहरण nano इमेज LoKr recipe के memory-size tuned variants हैं। दोनों 1024px aspect-ratio backend इस्तेमाल करते हैं। `48g` config `gradient_checkpointing_interval: 2` के साथ gradient checkpointing रखता है; `80g` config gradient checkpointing बंद करके `flash-attn-3-hub` enable करता है।

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
- Audio cache: video backend से generated

Training से पहले dataset download करें:

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

I2V backend video dataset में यह सेट करता है:

```json
"video": {
  "num_frames": 49,
  "min_frames": 49,
  "is_i2v": true
}
```

SimpleTuner इस flag से paired strict reference conditioning backend बनाता है।

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
- Super-I2V example: conditioning validation inputs उपयोग करता है।

## References

- [Diffusers Cosmos3 pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/cosmos3)
- [NVIDIA Cosmos3 collection](https://huggingface.co/collections/nvidia/cosmos3)
