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

### देखी गई परफॉर्मेंस और मेमोरी (फील्ड रिपोर्ट)

- **बेसलाइन सेटिंग्स**: 480p, 17 frames, batch size 2 (न्यूनतम वीडियो लंबाई/रिज़ॉल्यूशन)।
- **RamTorch (text encoder सहित)**: AMD 7900XTX पर ~13 GB VRAM।
  - NVIDIA 3090/4090/5090+ पर समान या बेहतर VRAM हेडरूम मिलना चाहिए।
- **बिना offload (int8 TorchAO)**: ~29-30 GB VRAM; 32 GB हार्डवेयर अनुशंसित।
  - सिस्टम RAM पीक: bf16 Gemma3 लोड करके int8 में क्वांटाइज़ करने पर ~46 GB (~32 GB VRAM)।
  - सिस्टम RAM पीक: bf16 LTX-2 transformer लोड करके int8 में क्वांटाइज़ करने पर ~34 GB (~30 GB VRAM)।
- **बिना offload (पूर्ण bf16)**: बिना किसी offload के ट्रेनिंग के लिए ~48 GB VRAM चाहिए।
- **थ्रूपुट**:
  - A100-80G SXM4 पर ~8 sec/step (कम्पाइल बंद)।
  - 7900XTX पर ~16 sec/step (लोकल रन)।
  - A100-80G SXM4 पर 200 steps ~30 मिनट।

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
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
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
- `model_flavour`: `dev` (डिफ़ॉल्ट), `dev-fp4`, `dev-fp8`, `2.3-dev` या `2.3-distilled`।
- `pretrained_model_name_or_path`: `Lightricks/LTX-2`, `dg845/LTX-2.3-Diffusers`, `dg845/LTX-2.3-Distilled-Diffusers` या local `.safetensors` फ़ाइल।
- `train_batch_size`: `1`। इसे तब तक न बढ़ाएँ जब तक आपके पास A100/H100 न हो।
- `validation_resolution`:
  - `512x768` परीक्षण के लिए सुरक्षित डिफ़ॉल्ट है।
  - `720x1280` (720p) संभव है लेकिन भारी है।
- `validation_num_video_frames`: **VAE compression (4x) के साथ संगत होना चाहिए।**
  - 5s (लगभग 12‑24fps पर): `61` या `49` उपयोग करें।
  - सूत्र: `(frames - 1) % 4 == 0`.
- `validation_guidance`: `5.0`.
- `frame_rate`: डिफ़ॉल्ट 25 है।

LTX-2 2.0 variants एक `.safetensors` checkpoint के रूप में आते हैं जिनमें transformer, video VAE, audio VAE, और vocoder शामिल हैं।
LTX-2.3 के लिए, SimpleTuner `model_flavour` के आधार पर संबंधित Diffusers repo लोड करता है
(`2.3-dev` या `2.3-distilled`)।

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

LTX-2 बिना audio के video-only training को support करता है। Audio training enable करने के लिए, अपने video dataset configuration में `audio` block जोड़ें:

```json
"audio": {
    "auto_split": true,
    "sample_rate": 16000,
    "channels": 1,
    "duration_interval": 3.0,
    "allow_zero_audio": false
}
```

जब `audio` section मौजूद होता है, तो SimpleTuner आपकी video files से automatically audio dataset बनाता है और video latents
के साथ audio latents cache करता है। यदि आपकी videos में audio stream नहीं है तो `audio.allow_zero_audio: true` सेट करें।
`audio` section के बिना, LTX-2 केवल video पर train करता है और audio loss को automatically mask कर देता है।

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

### न्यूनतम VRAM उपयोग कॉन्फ़िग (7900XTX)

LTX Video 2 पर न्यूनतम VRAM उपयोग के लिए फील्ड‑टेस्टेड कॉन्फ़िग।

<details>
<summary>7900XTX कॉन्फ़िग देखें (न्यूनतम VRAM उपयोग)</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "checkpoint_step_interval": 100,
  "data_backend_config": "config/ltx2/multidatabackend.json",
  "disable_benchmark": true,
  "dynamo_mode": "",
  "evaluation_type": "none",
  "hub_model_id": "simpletuner-ltxvideo2-19b-t2v-lora-test",
  "learning_rate": 0.00006,
  "lr_warmup_steps": 50,
  "lycoris_config": "config/lycoris_config.json",
  "max_grad_norm": 0.1,
  "max_train_steps": 200,
  "minimum_image_size": 0,
  "model_family": "ltxvideo2",
  "model_flavour": "dev",
  "model_type": "lora",
  "num_train_epochs": 0,
  "offload_during_startup": true,
  "optimizer": "adamw_bf16",
  "output_dir": "output/examples/ltxvideo2-19b-t2v.peft-lora",
  "override_dataset_config": true,
  "ramtorch": true,
  "ramtorch_text_encoder": true,
  "report_to": "none",
  "resolution": 480,
  "scheduled_sampling_reflexflow": false,
  "seed": 42,
  "skip_file_discovery": "",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "example-training-run",
  "train_batch_size": 2,
  "vae_batch_size": 1,
  "vae_enable_patch_conv": true,
  "vae_enable_slicing": true,
  "vae_enable_temporal_roll": true,
  "vae_enable_tiling": true,
  "validation_disable": true,
  "validation_disable_unconditional": true,
  "validation_guidance": 5,
  "validation_num_inference_steps": 40,
  "validation_num_video_frames": 81,
  "validation_prompt": "🟫 is holding a sign that says hello world from ltxvideo2",
  "validation_resolution": "768x512",
  "validation_seed": 42,
  "validation_using_datasets": false
}
```
</details>

### केवल ऑडियो ट्रेनिंग

LTX-2 **केवल ऑडियो ट्रेनिंग** को सपोर्ट करता है, जहाँ आप बिना वीडियो फाइलों के केवल ऑडियो जेनरेशन क्षमता को ट्रेन करते हैं। यह तब उपयोगी है जब आपके पास ऑडियो डेटासेट हैं लेकिन कोई संबंधित वीडियो कंटेंट नहीं है।

केवल ऑडियो मोड में:
- वीडियो latents अपने आप शून्य हो जाते हैं (मेमोरी बचाने के लिए न्यूनतम 64x64 रिज़ॉल्यूशन)
- वीडियो loss मास्क हो जाता है (गणना नहीं होती)
- केवल ऑडियो जेनरेशन लेयर्स ट्रेन होती हैं

जब आपके डेटासेट कॉन्फ़िगरेशन में केवल ऑडियो डेटासेट होते हैं (कोई वीडियो या इमेज डेटासेट नहीं), तो केवल ऑडियो मोड **स्वचालित रूप से डिटेक्ट** हो जाता है। आप `audio.audio_only: true` से भी इसे स्पष्ट रूप से सक्षम कर सकते हैं।

#### केवल ऑडियो डेटासेट कॉन्फ़िगरेशन

```json
[
  {
    "id": "my-audio-dataset",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/audio",
    "caption_strategy": "textfile",
    "audio": {
      "sample_rate": 16000,
      "channels": 2,
      "duration_interval": 3.0,
      "truncation_mode": "beginning"
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

मुख्य ऑडियो सेटिंग्स:
- `channels`: LTX-2 Audio VAE के लिए **2 (स्टीरियो) होना चाहिए**
- `duration_interval`: ऑडियो को इंटरवल में बकेट करें (जैसे 3.0 सेकंड)। **मेमोरी मैनेजमेंट के लिए महत्वपूर्ण** - लंबी ऑडियो फाइलें कई वीडियो फ्रेम बनाती हैं भले ही वे शून्य हों
- `truncation_mode`: बकेट duration से अधिक ऑडियो को कैसे हैंडल करें (`beginning`, `end`, या `random`)

#### समर्थित ऑडियो फॉर्मेट

SimpleTuner सामान्य ऑडियो फॉर्मेट (`.wav`, `.flac`, `.mp3`, `.ogg`, `.opus`, आदि) के साथ-साथ कंटेनर फॉर्मेट जिनमें केवल ऑडियो हो सकता है (`.mp4`, `.mpeg`, `.mkv`, `.webm`) को सपोर्ट करता है। कंटेनर फॉर्मेट ffmpeg का उपयोग करके स्वचालित रूप से एक्सट्रैक्ट किए जाते हैं।

#### ऑडियो ट्रेनिंग के लिए LoRA टारगेट

जब आपके datasets में ऑडियो डेटा का पता चलता है, SimpleTuner स्वचालित रूप से ऑडियो-विशिष्ट मॉड्यूल को LoRA टारगेट में जोड़ देता है:
- `audio_proj_in` - ऑडियो इनपुट प्रोजेक्शन
- `audio_proj_out` - ऑडियो आउटपुट प्रोजेक्शन
- `audio_caption_projection.linear_1` - ऑडियो कैप्शन प्रोजेक्शन लेयर 1
- `audio_caption_projection.linear_2` - ऑडियो कैप्शन प्रोजेक्शन लेयर 2

यह केवल ऑडियो ट्रेनिंग और संयुक्त ऑडियो+वीडियो ट्रेनिंग दोनों के लिए स्वचालित रूप से होता है।

यदि आप LoRA टारगेट को मैन्युअल रूप से ओवरराइड करना चाहते हैं, तो `--peft_lora_target_modules` के साथ मॉड्यूल नामों की JSON लिस्ट का उपयोग करें।

अपनी ऑडियो फाइलें `instance_data_dir` में रखें और संबंधित `.txt` कैप्शन फाइलें प्रदान करें।

### Validation workflows (T2V vs I2V)

- **T2V (text‑to‑video)**: `validation_using_datasets: false` रखें और `validation_prompt` या `validation_prompt_library` का उपयोग करें।
- **I2V (image‑to‑video)**: `validation_using_datasets: true` सेट करें और `eval_dataset_id` को ऐसे validation split पर पॉइंट करें जो reference image देता हो। Validation image‑to‑video pipeline पर स्विच करेगा और उसी image को conditioning के लिए उपयोग करेगा।
- **S2V (audio‑conditioned)**: `validation_using_datasets: true` के साथ, `eval_dataset_id` को `s2v_datasets` (या default `audio.auto_split`) वाले dataset पर सेट करें। Validation cached audio latents अपने‑आप लोड करेगा।

### Validation adapters (LoRAs)

Lightricks के LoRAs को validation में `validation_adapter_path` (single) या `validation_adapter_config` (multiple runs) से लोड कर सकते हैं। इन repos में nonstandard weight filename हैं, इसलिए `repo_id:weight_name` के साथ दें। सही filenames और assets के लिए LTX-2 collection देखें:
https://huggingface.co/collections/Lightricks/ltx-2
- `Lightricks/LTX-2-19b-IC-LoRA-Canny-Control:ltx-2-19b-ic-lora-canny-control.safetensors`
- `Lightricks/LTX-2-19b-IC-LoRA-Depth-Control:ltx-2-19b-ic-lora-depth-control.safetensors`
- `Lightricks/LTX-2-19b-IC-LoRA-Detailer:ltx-2-19b-ic-lora-detailer.safetensors`
- `Lightricks/LTX-2-19b-IC-LoRA-Pose-Control:ltx-2-19b-ic-lora-pose-control.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In:ltx-2-19b-lora-camera-control-dolly-in.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out:ltx-2-19b-lora-camera-control-dolly-out.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left:ltx-2-19b-lora-camera-control-dolly-left.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right:ltx-2-19b-lora-camera-control-dolly-right.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down:ltx-2-19b-lora-camera-control-jib-down.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up:ltx-2-19b-lora-camera-control-jib-up.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Static:ltx-2-19b-lora-camera-control-static.safetensors`

तेज़ validation के लिए `Lightricks/LTX-2-19b-distilled-lora-384:ltx-2-19b-distilled-lora-384.safetensors` को
validation adapter के रूप में लगाएं और `validation_guidance: 1` तथा `validation_num_inference_steps: 8` सेट करें।
