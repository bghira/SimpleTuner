## Cosmos3 क्विकस्टार्ट

NVIDIA Cosmos3 के लिए LyCORIS LoKr ट्रेन करें।

## मॉडल नोट्स

- `model_family`: `cosmos3`
- डिफ़ॉल्ट flavour: `nano`
- Flavours:
  - `nano`: `nvidia/Cosmos3-Nano`, 16B
  - `super`: `nvidia/Cosmos3-Super`, 65B
  - `super-t2i`: `nvidia/Cosmos3-Super-Text2Image`, 65B
- SimpleTuner में Cosmos3 tokenizer IDs सीधे उपयोग करता है।
- Positive prompts tokenization के दौरान Cosmos3 structured JSON captions में बदलते हैं।
- Negative prompts JSON में नहीं बदलते।
- `text_embeds` backend न जोड़ें।
- ये उदाहरण `image_embeds` backend नहीं जोड़ते।
- image और video samples सामान्य VAE cache उपयोग करते हैं।
- audio वाला video सामान्य VAE cache और audio VAE cache उपयोग करता है।
- action और policy datasets इस गाइड में शामिल नहीं हैं।

## हार्डवेयर

- `model_flavour: nano` से शुरू करें।
- `mixed_precision: bf16` उपयोग करें।
- पहले `base_model_precision: no_change` उपयोग करें।
- `train_batch_size: 1` रखें।
- `gradient_checkpointing` सक्षम करें।

वैकल्पिक group offload:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

- Streams केवल CUDA पर हैं।
- इसे `--enable_model_cpu_offload` के साथ न मिलाएँ।
- system RAM कम हो तो `--group_offload_to_disk_path /fast-ssd/simpletuner-offload` जोड़ें।

## इंस्टॉलेशन

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

## उदाहरण configs

| उदाहरण | Dataset | Media | Backend |
| --- | --- | --- | --- |
| `cosmos3-image.lycoris-lokr` | `RareConcepts/Domokun` | image | `multidatabackend-cosmos3-domokun-512px.json` |
| `cosmos3-video.lycoris-lokr` | `sayakpaul/video-dataset-disney-organized` | video | `multidatabackend-cosmos3-disney-video-480p+49f.json` |
| `cosmos3-video-audio.lycoris-lokr` | `bghira/Synchronised-Drumming-Gemini3Captions` | video + audio | `multidatabackend-cosmos3-drumming-video-audio-480p+49f.json` |

## जरूरी fields

- `model_family`: `cosmos3`
- `model_flavour`: `nano`
- `model_type`: `lora`
- `lora_type`: `lycoris`
- `base_model_precision`: `no_change`
- `mixed_precision`: `bf16`
- `train_batch_size`: `1`
- `gradient_checkpointing`: `true`

## Dataset नोट्स

### Image

- Dataset: [`RareConcepts/Domokun`](https://huggingface.co/datasets/RareConcepts/Domokun)
- Backend: `config/examples/multidatabackend-cosmos3-domokun-512px.json`
- Backend type: `huggingface`
- Caption strategy: `instanceprompt`
- Text embed cache: उपयोग नहीं होता

### Video

- Dataset: [`sayakpaul/video-dataset-disney-organized`](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)
- Backend: `config/examples/multidatabackend-cosmos3-disney-video-480p+49f.json`
- Backend type: `huggingface`
- Columns: `video`, `prompt`
- Text embed cache: उपयोग नहीं होता
- Audio cache: उपयोग नहीं होता

### Video With Audio

- Dataset: [`bghira/Synchronised-Drumming-Gemini3Captions`](https://huggingface.co/datasets/bghira/Synchronised-Drumming-Gemini3Captions)
- Backend: `config/examples/multidatabackend-cosmos3-drumming-video-audio-480p+49f.json`
- Backend type: `local`
- Files: `.mpeg` videos with adjacent `.txt` captions
- Text embed cache: उपयोग नहीं होता
- Audio cache: video backend से auto-generated

Training से पहले dataset डाउनलोड करें:

```bash
huggingface-cli download \
  --repo-type=dataset \
  bghira/Synchronised-Drumming-Gemini3Captions \
  --local-dir datasets/Synchronised-Drumming-Gemini3Captions
```

Backend में यह शामिल है:

```json
"audio": {
  "auto_split": true,
  "sample_rate": 16000,
  "channels": 1,
  "duration_interval": 3.0,
  "allow_zero_audio": false
}
```

SimpleTuner उस block से audio dataset inject करता है और audio latents को अलग VAE cache में store करता है।

## चलाएँ

```bash
simpletuner train example=cosmos3-image.lycoris-lokr
simpletuner train example=cosmos3-video.lycoris-lokr
simpletuner train example=cosmos3-video-audio.lycoris-lokr
```

## Validation

- Image example: `validation_resolution: 512x512`
- Video examples: `validation_resolution: 768x432`
- Video examples: `validation_num_video_frames: 49`
- Audio generation validation के लिए dataset-based validation settings की जरूरत हो सकती है।

## References

- [Diffusers Cosmos3 pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/cosmos3)
- [NVIDIA Cosmos3 collection](https://huggingface.co/collections/nvidia/cosmos3)
