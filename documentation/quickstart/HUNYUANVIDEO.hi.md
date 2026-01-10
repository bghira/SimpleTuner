# Hunyuan Video 1.5 क्विकस्टार्ट

यह गाइड Tencent के 8.3B **Hunyuan Video 1.5** रिलीज़ (`tencent/HunyuanVideo-1.5`) पर SimpleTuner का उपयोग करके LoRA प्रशिक्षण की प्रक्रिया बताती है।

## हार्डवेयर आवश्यकताएँ

Hunyuan Video 1.5 एक बड़ा मॉडल है (8.3B पैरामीटर)।

- **न्यूनतम**: 480p पर full gradient checkpointing के साथ Rank‑16 LoRA के लिए **24GB‑32GB VRAM** आरामदायक है।
- **अनुशंसित**: 720p प्रशिक्षण या बड़े batch sizes के लिए A6000 / A100 (48GB‑80GB)।
- **सिस्टम RAM**: मॉडल लोडिंग संभालने के लिए **64GB+** अनुशंसित है।

### मेमोरी ऑफ़लोडिंग (वैकल्पिक)

अपने `config.json` में यह जोड़ें:

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

- `--group_offload_use_stream`: केवल CUDA डिवाइसेस पर काम करता है।
- इसे `--enable_model_cpu_offload` के साथ **न** मिलाएँ।

## पूर्वापेक्षाएँ

सुनिश्चित करें कि Python इंस्टॉल है; SimpleTuner 3.10 से 3.12 के साथ अच्छा काम करता है।

आप इसे चलाकर जांच सकते हैं:

```bash
python --version
```

यदि आपके Ubuntu पर Python 3.12 इंस्टॉल नहीं है, तो आप यह प्रयास कर सकते हैं:

```bash
apt -y install python3.12 python3.12-venv
```

### Container image dependencies

Vast, RunPod, और TensorDock (आदि) के लिए, CUDA 12.2‑12.8 इमेज पर CUDA extensions कम्पाइल करने हेतु यह काम करेगा:

```bash
apt -y install nvidia-cuda-toolkit
```

### AMD ROCm follow‑up steps

AMD MI300X को उपयोगी बनाने के लिए निम्न चलाना आवश्यक है:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## इंस्टॉलेशन

pip के जरिए SimpleTuner इंस्टॉल करें:

```bash
pip install 'simpletuner[cuda]'
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [installation documentation](../INSTALL.md) देखें।

### आवश्यक checkpoints

मुख्य `tencent/HunyuanVideo-1.5` repo में transformer/vae/scheduler हैं, लेकिन **text encoder** (`text_encoder/llm`) और **vision encoder** (`vision_encoder/siglip`) अलग downloads में हैं। लॉन्च करने से पहले SimpleTuner को अपने लोकल कॉपी पाथ बताएं:

```bash
export HUNYUANVIDEO_TEXT_ENCODER_PATH=/path/to/text_encoder_root
export HUNYUANVIDEO_VISION_ENCODER_PATH=/path/to/vision_encoder_root
```

यदि ये unset हैं, तो SimpleTuner उन्हें मॉडल repo से खींचने की कोशिश करता है; अधिकांश mirrors इन्हें शामिल नहीं करते, इसलिए startup errors से बचने के लिए paths स्पष्ट रूप से सेट करें।

## वातावरण सेटअप

### Web interface method

SimpleTuner WebUI सेटअप को सरल बनाता है। सर्वर चलाने के लिए:

```bash
simpletuner server
```

यह डिफ़ॉल्ट रूप से पोर्ट 8001 पर वेब सर्वर बनाता है, जिसे आप http://localhost:8001 पर खोल सकते हैं।

### Manual / command‑line method

command‑line टूल्स से SimpleTuner चलाने के लिए, आपको एक configuration फ़ाइल, dataset और model directories, तथा एक dataloader configuration फ़ाइल सेट करनी होगी।

#### Configuration file

एक प्रयोगात्मक स्क्रिप्ट, `configure.py`, इंटरैक्टिव step‑by‑step कॉन्फ़िगरेशन के जरिए इस सेक्शन को पूरी तरह स्किप करने में मदद कर सकती है।

**नोट:** यह आपके dataloader को कॉन्फ़िगर नहीं करता। आपको उसे बाद में मैन्युअली करना होगा।

इसे चलाने के लिए:

```bash
simpletuner configure
```

यदि आप मैन्युअल कॉन्फ़िगर करना पसंद करते हैं:

`config/config.json.example` को `config/config.json` में कॉपी करें:

```bash
cp config/config.json.example config/config.json
```

HunyuanVideo के लिए key configuration overrides:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "model_type": "lora",
  "model_family": "hunyuanvideo",
  "pretrained_model_name_or_path": "tencent/HunyuanVideo-1.5",
  "model_flavour": "t2v-480p",
  "output_dir": "output/hunyuan-video",
  "validation_resolution": "854x480",
  "validation_num_video_frames": 61,
  "validation_guidance": 6.0,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "learning_rate": 1e-4,
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "lora_rank": 16,
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "dataset_backend_config": "config/multidatabackend.json"
}
```
</details>

- `model_flavour` विकल्प:
  - `t2v-480p` (डिफ़ॉल्ट)
  - `t2v-720p`
  - `i2v-480p` (Image‑to‑Video)
  - `i2v-720p` (Image‑to‑Video)
- `validation_num_video_frames`: `(frames - 1) % 4 == 0` होना चाहिए। जैसे 61, 129।

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

#### डेटासेट विचार

एक `--data_backend_config` (`config/multidatabackend.json`) दस्तावेज़ बनाएँ जिसमें यह हो:

```json
[
  {
    "id": "my-video-dataset",
    "type": "local",
    "dataset_type": "video",
    "instance_data_dir": "datasets/videos",
    "caption_strategy": "textfile",
    "resolution": 480,
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
    "cache_dir": "cache/text/hunyuan",
    "disabled": false
  }
]
```

`video` उप‑सेक्शन में:
- `num_frames`: प्रशिक्षण के लिए target फ्रेम काउंट। `(frames - 1) % 4 == 0` होना चाहिए।
- `min_frames`: न्यूनतम वीडियो लंबाई (छोटे वीडियो discard हो जाते हैं)।
- `max_frames`: अधिकतम वीडियो लंबाई फ़िल्टर।
- `bucket_strategy`: videos को buckets में समूहित करने का तरीका:
  - `aspect_ratio` (डिफ़ॉल्ट): केवल spatial aspect ratio से समूहित।
  - `resolution_frames`: mixed resolution/duration datasets के लिए `WxH@F` फॉर्मैट (जैसे `854x480@61`) के अनुसार समूहित।
- `frame_interval`: `resolution_frames` उपयोग करते समय फ्रेम काउंट को इस इंटरवल तक राउंड करें।

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

- **Text Embed Caching**: अत्यधिक अनुशंसित है। Hunyuan बड़ा LLM text encoder उपयोग करता है। caching से प्रशिक्षण के दौरान VRAM बचती है।

#### WandB और Huggingface Hub में लॉग‑इन

```bash
wandb login
huggingface-cli login
```

</details>

### प्रशिक्षण रन निष्पादित करना

SimpleTuner डायरेक्टरी से:

```bash
simpletuner train
```

## नोट्स और समस्या‑समाधान टिप्स

### VRAM Optimization

- **Group Offload**: consumer GPUs के लिए आवश्यक। सुनिश्चित करें कि `enable_group_offload` true है।
- **Resolution**: सीमित VRAM हो तो 480p (`854x480` या समान) पर टिकें। 720p (`1280x720`) मेमोरी उपयोग काफी बढ़ाता है।
- **Quantization**: `base_model_precision` उपयोग करें (`bf16` डिफ़ॉल्ट); अतिरिक्त बचत के लिए `int8-torchao` काम करता है, लेकिन गति कम होगी।
- **VAE patch convolution**: HunyuanVideo VAE OOMs के लिए `--vae_enable_patch_conv=true` सेट करें (या UI में टॉगल करें)। यह 3D conv/attention को slice करता है और peak VRAM कम करता है; throughput थोड़ा घटेगा।

### Image‑to‑Video (I2V)

- `model_flavour="i2v-480p"` उपयोग करें।
- SimpleTuner training videos के पहले फ्रेम को conditioning image के रूप में स्वतः उपयोग करता है।
- सुनिश्चित करें कि आपका validation setup conditioning inputs शामिल करता है या auto‑extracted first frame पर निर्भर रहता है।

### Text Encoders

Hunyuan dual text encoder सेटअप (LLM + CLIP) उपयोग करता है। caching फेज के दौरान इन्हें लोड करने के लिए पर्याप्त सिस्टम RAM सुनिश्चित करें।
