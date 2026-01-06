# LTX Video 2 क्विकस्टार्ट

इस उदाहरण में, हम LTX‑2 video/audio VAE और Gemma3 text encoder का उपयोग करके LTX Video 2 LoRA प्रशिक्षण करेंगे।

## हार्डवेयर आवश्यकताएँ

LTX Video 2 एक भारी **19B** मॉडल है। यह निम्न को जोड़ता है:
1. **Gemma3**: टेक्स्ट एन्कोडर।
2. **LTX‑2 Video VAE** (audio conditioning के लिए Audio VAE भी)।
3. **19B Video Transformer**: बड़ा DiT backbone।

यह सेटअप VRAM‑intensive है, और VAE pre‑caching मेमोरी usage को spike कर सकता है।

- **Single‑GPU प्रशिक्षण**: `train_batch_size: 1` से शुरू करें और group offload सक्षम करें।
  - **नोट**: शुरुआती **VAE pre‑caching चरण** में अधिक VRAM लग सकती है। caching चरण के लिए CPU offloading या बड़ा GPU चाहिए हो सकता है।
  - **टिप**: `config.json` में `"offload_during_startup": true` सेट करें ताकि VAE और text encoder एक साथ GPU पर लोड न हों, जिससे pre‑caching मेमोरी दबाव काफी कम हो जाता है।
- **Multi‑GPU प्रशिक्षण**: यदि अधिक headroom चाहिए तो **FSDP2** या आक्रामक **Group Offload** अनुशंसित है।
- **सिस्टम RAM**: बड़े रन के लिए 64GB+ अनुशंसित है; अधिक RAM caching में मदद करती है।

### मेमोरी ऑफ़लोडिंग (महत्वपूर्ण)

अधिकांश single‑GPU सेटअप पर LTX Video 2 प्रशिक्षण के लिए grouped offloading सक्षम करना चाहिए। बड़े batch/resolution के लिए VRAM headroom रखने हेतु यह वैकल्पिक लेकिन अनुशंसित है।

इसे अपने `config.json` में जोड़ें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "group_offload_blocks_per_group": 1,
  "group_offload_use_stream": true
}
```
</details>

## पूर्वापेक्षाएँ

Python 3.12 इंस्टॉल होना सुनिश्चित करें।

```bash
python --version
```

## इंस्टॉलेशन

```bash
pip install simpletuner[cuda]
```

अधिक उन्नत इंस्टॉलेशन विकल्पों के लिए [INSTALL.md](../INSTALL.md) देखें।

## वातावरण सेटअप

### Web interface

```bash
simpletuner server
```
http://localhost:8001 पर पहुँचें।

### Manual configuration

हेल्पर स्क्रिप्ट चलाएँ:

```bash
simpletuner configure
```

या उदाहरण कॉपी करके मैन्युअली संपादित करें:

```bash
cp config/config.json.example config/config.json
```

#### Configuration parameters

LTX Video 2 के लिए key settings:

- `model_family`: `ltxvideo2`
- `model_flavour`: `2.0` (डिफ़ॉल्ट)
- `pretrained_model_name_or_path`: `Lightricks/LTX-2` (वैकल्पिक override)
- `train_batch_size`: `1`। इसे तब तक न बढ़ाएँ जब तक आपके पास A100/H100 न हो।
- `validation_resolution`:
  - `512x768` परीक्षण के लिए सुरक्षित डिफ़ॉल्ट है।
  - `720x1280` (720p) संभव है लेकिन भारी है।
- `validation_num_video_frames`: **VAE compression (4x) के साथ संगत होना चाहिए।**
  - 5s (लगभग 12‑24fps पर): `61` या `49` उपयोग करें।
  - सूत्र: `(frames - 1) % 4 == 0`.
- `validation_guidance`: `5.0`.
- `frame_rate`: डिफ़ॉल्ट 25 है।

### वैकल्पिक: VRAM ऑप्टिमाइज़ेशन

VRAM headroom चाहिए तो:
- **Musubi block swap**: `musubi_blocks_to_swap` (`4-8` से शुरू करें) और वैकल्पिक `musubi_block_swap_device` (डिफ़ॉल्ट `cpu`) सेट करें ताकि आख़िरी Transformer blocks CPU से स्ट्रीम हों। throughput घटेगा लेकिन peak VRAM कम होगा।
- **VAE patch convolution**: `--vae_enable_patch_conv=true` से LTX-2 VAE में temporal chunking सक्षम करें; थोड़ी speed कम होगी लेकिन peak VRAM घटेगा।
- **VAE temporal roll**: `--vae_enable_temporal_roll=true` से अधिक aggressive temporal chunking (अधिक speed hit)।
- **VAE tiling**: `--vae_enable_tiling=true` से बड़े resolution पर VAE encode/decode को tiles में चलाएँ।

### वैकल्पिक: CREPA temporal regularizer

flicker कम करने और फ्रेम्स में विषय स्थिर रखने के लिए:
- **Training → Loss functions** में **CREPA** सक्षम करें।
- अनुशंसित प्रारंभिक मान: **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- जब तक छोटा encoder न चाहिए, डिफ़ॉल्ट vision encoder (`dinov2_vitg14`, size `518`) रखें (`dinov2_vits14` + `224` छोटा विकल्प है)।
- पहली बार DINOv2 weights पाने के लिए network (या cached torch hub) चाहिए।
- **Drop VAE Encoder** केवल तब सक्षम करें जब आप पूरी तरह cached latents से प्रशिक्षण कर रहे हों; अन्यथा बंद रखें।

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

#### डेटासेट विचार

वीडियो डेटासेट्स के लिए सावधानीपूर्वक सेटअप चाहिए। `config/multidatabackend.json` बनाएँ:

```json
[
  {
    "id": "my-video-dataset",
    "type": "local",
    "dataset_type": "video",
    "instance_data_dir": "datasets/videos",
    "caption_strategy": "textfile",
    "resolution": 512,
    "video": {
        "num_frames": 61,
        "min_frames": 61,
        "frame_rate": 25,
        "bucket_strategy": "aspect_ratio"
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo2",
    "disabled": false
  }
]
```

`video` उप‑सेक्शन में:
- `num_frames`: प्रशिक्षण के लिए target फ्रेम काउंट।
- `min_frames`: न्यूनतम वीडियो लंबाई (छोटे वीडियो discard हो जाते हैं)।
- `max_frames`: अधिकतम वीडियो लंबाई फ़िल्टर।
- `bucket_strategy`: videos को buckets में समूहित करने का तरीका:
  - `aspect_ratio` (डिफ़ॉल्ट): केवल spatial aspect ratio से समूहित।
  - `resolution_frames`: mixed resolution/duration datasets के लिए `WxH@F` फॉर्मैट (जैसे `1920x1080@61`) के अनुसार समूहित।
- `frame_interval`: `resolution_frames` उपयोग करते समय फ्रेम काउंट को इस इंटरवल तक राउंड करें।

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

#### Directory setup

```bash
mkdir -p datasets/videos
</details>

# Place .mp4 / .mov files here.
# Place corresponding .txt files with same filename for captions.
```

#### Login

```bash
wandb login
huggingface-cli login
```

### प्रशिक्षण चलाना

```bash
simpletuner train
```

## नोट्स और समस्या‑समाधान टिप्स

### Out of Memory (OOM)

वीडियो प्रशिक्षण बेहद मांग वाला है। यदि OOM हो:

1. **Resolution घटाएँ**: 480p (जैसे `480x854`) आज़माएँ।
2. **Frames घटाएँ**: `validation_num_video_frames` और dataset `num_frames` को `33` या `49` तक घटाएँ।
3. **Offload जांचें**: सुनिश्चित करें कि `--enable_group_offload` सक्रिय है।

### Validation Video Quality

- **Black/Noise Videos**: अक्सर `validation_guidance` बहुत अधिक (> 6.0) या बहुत कम (< 2.0) होने से होता है। `5.0` पर रहें।
- **Motion Jitter**: जांचें कि dataset frame rate मॉडल के प्रशिक्षित frame rate (अक्सर 25fps) से मेल खाता है।
- **Stagnant/Static Video**: मॉडल कम प्रशिक्षित हो सकता है या prompt में motion वर्णन नहीं है। "camera pans right", "zoom in", "running" जैसे prompts उपयोग करें।

### TREAD प्रशिक्षण

TREAD वीडियो के लिए भी काम करता है और compute बचाने के लिए अत्यधिक अनुशंसित है।

`config.json` में जोड़ें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

ratio पर निर्भर करते हुए यह प्रशिक्षण को ~25‑40% तक तेज़ कर सकता है।

### Validation workflows (T2V vs I2V)

- **T2V (text‑to‑video)**: `validation_using_datasets: false` रखें और `validation_prompt` या `validation_prompt_library` का उपयोग करें।
- **I2V (image‑to‑video)**: `validation_using_datasets: true` सेट करें और `eval_dataset_id` को ऐसे validation split पर पॉइंट करें जो reference image देता हो। Validation image‑to‑video pipeline पर स्विच करेगा और उसी image को conditioning के लिए उपयोग करेगा।
