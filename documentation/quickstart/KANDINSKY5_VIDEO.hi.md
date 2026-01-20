# Kandinsky 5.0 Video क्विकस्टार्ट

इस उदाहरण में, हम HunyuanVideo VAE और dual text encoders का उपयोग करके Kandinsky 5.0 Video LoRA (Lite या Pro) प्रशिक्षण करेंगे।

## हार्डवेयर आवश्यकताएँ

Kandinsky 5.0 Video एक भारी मॉडल है। यह निम्न को जोड़ता है:
1. **Qwen2.5‑VL (7B)**: एक विशाल vision‑language टेक्स्ट एन्कोडर।
2. **HunyuanVideo VAE**: उच्च‑गुणवत्ता 3D VAE।
3. **Video Transformer**: एक जटिल DiT आर्किटेक्चर।

यह सेटअप VRAM‑intensive है, हालांकि "Lite" और "Pro" variants की आवश्यकताएँ अलग हैं।

- **Lite मॉडल प्रशिक्षण**: आश्चर्यजनक रूप से कुशल, **~13GB VRAM** पर भी प्रशिक्षित हो सकता है।
  - **नोट**: शुरुआती **VAE pre‑caching चरण** में भारी HunyuanVideo VAE के कारण काफी अधिक VRAM लगती है। caching चरण के लिए आपको CPU offloading या बड़ा GPU चाहिए हो सकता है।
  - **टिप**: `config.json` में `"offload_during_startup": true` सेट करें ताकि VAE और text encoder एक साथ GPU पर लोड न हों, जिससे pre‑caching मेमोरी दबाव काफी कम हो जाता है।
  - **यदि VAE OOM करे**: HunyuanVideo VAE 3D convs को slice करने के लिए `--vae_enable_patch_conv=true` सेट करें; थोड़ी गति कम होगी लेकिन peak VRAM घटेगा।
- **Pro मॉडल प्रशिक्षण**: consumer hardware पर फिट कराने के लिए **FSDP2** (multi‑gpu) या LoRA के साथ आक्रामक **Group Offload** चाहिए। विशिष्ट VRAM/RAM आवश्यकताएँ तय नहीं हैं, लेकिन "the more, the merrier" लागू होता है।
- **सिस्टम RAM**: Lite मॉडल के लिए **45GB** RAM पर परीक्षण आरामदायक था। सुरक्षित रहने के लिए 64GB+ अनुशंसित है।

### मेमोरी ऑफ़लोडिंग (महत्वपूर्ण)

लगभग हर single‑GPU सेटअप पर **Pro** मॉडल प्रशिक्षण के लिए grouped offloading **आवश्यक** है। **Lite** के लिए यह वैकल्पिक है लेकिन बड़े batch/resolution के लिए VRAM बचाने हेतु अनुशंसित है।

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
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
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

Kandinsky 5 Video के लिए key settings:

- `model_family`: `kandinsky5-video`
- `model_flavour`:
  - `t2v-lite-sft-5s`: Lite मॉडल, ~5s आउटपुट। (डिफ़ॉल्ट)
  - `t2v-lite-sft-10s`: Lite मॉडल, ~10s आउटपुट।
  - `t2v-pro-sft-5s-hd`: Pro मॉडल, ~5s, उच्च‑परिभाषा प्रशिक्षण।
  - `t2v-pro-sft-10s-hd`: Pro मॉडल, ~10s, उच्च‑परिभाषा प्रशिक्षण।
  - `i2v-lite-5s`: Image‑to‑video Lite, 5s आउटपुट (conditioning images आवश्यक)।
  - `i2v-pro-sft-5s`: Image‑to‑video Pro SFT, 5s आउटपुट (conditioning images आवश्यक)।
  - *(Pretrain variants ऊपर सभी के लिए उपलब्ध हैं)*
- `train_batch_size`: `1`। इसे तब तक न बढ़ाएँ जब तक आपके पास A100/H100 न हो।
- `validation_resolution`:
  - `512x768` परीक्षण के लिए सुरक्षित डिफ़ॉल्ट है।
  - `720x1280` (720p) संभव है लेकिन भारी है।
- `validation_num_video_frames`: **VAE compression (4x) के साथ संगत होना चाहिए।**
  - 5s (लगभग 12‑24fps पर): `61` या `49` उपयोग करें।
  - सूत्र: `(frames - 1) % 4 == 0`.
- `validation_guidance`: `5.0`.
- `frame_rate`: डिफ़ॉल्ट 24 है।

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
        "frame_rate": 24,
        "bucket_strategy": "aspect_ratio"
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/kandinsky5",
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
- **Motion Jitter**: जांचें कि dataset frame rate मॉडल के प्रशिक्षित frame rate (अक्सर 24fps) से मेल खाता है।
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

### I2V (Image‑to‑Video) प्रशिक्षण

यदि `i2v` flavours उपयोग कर रहे हैं:
- SimpleTuner training videos के पहले फ्रेम को conditioning image के रूप में स्वतः निकालता है।
- पाइपलाइन training के दौरान पहले फ्रेम को अपने‑आप mask करती है।
- Validation के लिए एक input image देना होगा, या SimpleTuner validation video generation के पहले फ्रेम को conditioner के रूप में उपयोग करेगा।
