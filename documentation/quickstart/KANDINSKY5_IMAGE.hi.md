# Kandinsky 5.0 Image क्विकस्टार्ट

इस उदाहरण में, हम Kandinsky 5.0 Image LoRA प्रशिक्षण करेंगे।

## हार्डवेयर आवश्यकताएँ

Kandinsky 5.0 में एक **बहुत बड़ा 7B पैरामीटर वाला Qwen2.5‑VL टेक्स्ट एन्कोडर** होता है, साथ ही एक मानक CLIP एन्कोडर और Flux VAE भी। इससे VRAM और सिस्टम RAM दोनों पर भारी दबाव पड़ता है।

केवल Qwen एन्कोडर लोड करने में ही लगभग **14GB** मेमोरी लगती है। full gradient checkpointing के साथ rank‑16 LoRA ट्रेनिंग करते समय:

- **24GB VRAM** आरामदायक न्यूनतम है (RTX 3090/4090)।
- **16GB VRAM** संभव है, लेकिन इसके लिए आक्रामक offloading और संभवतः base model का `int8` quantization चाहिए।

आपको चाहिए:

- **सिस्टम RAM**: कम से कम 32GB, आदर्श रूप से 64GB, ताकि शुरुआती मॉडल लोड बिना क्रैश के हो।
- **GPU**: NVIDIA RTX 3090 / 4090 या प्रो‑ग्रेड कार्ड (A6000, A100, आदि)।

### मेमोरी ऑफ़लोडिंग (अनुशंसित)

टेक्स्ट एन्कोडर के आकार को देखते हुए, यदि आप consumer hardware पर हैं तो आपको लगभग निश्चित रूप से grouped offloading का उपयोग करना चाहिए। यह transformer blocks को CPU मेमोरी में ऑफ़लोड करता है जब वे सक्रिय रूप से compute नहीं हो रहे होते।

यह अपने `config.json` में जोड़ें:

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

इसके अलावा, initialization और caching फेज के दौरान VRAM उपयोग कम करने के लिए `config.json` में `"offload_during_startup": true` सेट करें। यह सुनिश्चित करता है कि text encoder और VAE एक साथ लोड न हों।

## पूर्वापेक्षाएँ

सुनिश्चित करें कि Python इंस्टॉल है; SimpleTuner 3.10 से 3.12 के साथ अच्छा काम करता है।

आप इसे चलाकर जांच सकते हैं:

```bash
python --version
```

यदि आपके Ubuntu पर Python 3.12 इंस्टॉल नहीं है, तो आप यह प्रयास कर सकते हैं:

```bash
apt -y install python3.13 python3.13-venv
```

## इंस्टॉलेशन

pip के जरिए SimpleTuner इंस्टॉल करें:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [installation documentation](../INSTALL.md) देखें।

## वातावरण सेटअप

### Web interface method

SimpleTuner WebUI सेटअप को सरल बनाता है। सर्वर चलाने के लिए:

```bash
simpletuner server
```

इसे http://localhost:8001 पर खोलें।

### Manual / command‑line method

command‑line टूल्स से SimpleTuner चलाने के लिए, आपको एक configuration फ़ाइल, dataset और model directories, तथा एक dataloader configuration फ़ाइल सेट करनी होगी।

#### Configuration file

एक प्रयोगात्मक स्क्रिप्ट, `configure.py`, इस सेक्शन को स्किप करने में मदद कर सकती है:

```bash
simpletuner configure
```

यदि आप मैन्युअल कॉन्फ़िगर करना पसंद करते हैं:

`config/config.json.example` को `config/config.json` में कॉपी करें:

```bash
cp config/config.json.example config/config.json
```

आपको निम्न वेरिएबल्स बदलने होंगे:

- `model_type`: `lora`
- `model_family`: `kandinsky5-image`
- `model_flavour`:
  - `t2i-lite-sft`: (डिफ़ॉल्ट) standard SFT checkpoint. स्टाइल/कैरक्टर fine‑tuning के लिए सर्वोत्तम।
  - `t2i-lite-pretrain`: pretrain checkpoint. बिल्कुल नए concepts सिखाने के लिए बेहतर।
  - `i2i-lite-sft` / `i2i-lite-pretrain`: image‑to‑image प्रशिक्षण के लिए। आपके डेटासेट में conditioning images चाहिए।
- `output_dir`: checkpoints सहेजने के लिए स्थान।
- `train_batch_size`: `1` से शुरू करें।
- `gradient_accumulation_steps`: बड़े batch का अनुकरण करने के लिए `1` या उससे अधिक रखें।
- `validation_resolution`: इस मॉडल के लिए `1024x1024` मानक है।
- `validation_guidance`: Kandinsky 5 के लिए अनुशंसित डिफ़ॉल्ट `5.0` है।
- `flow_schedule_shift`: डिफ़ॉल्ट `1.0` है। इसे बदलने से मॉडल विवरण बनाम composition को प्राथमिकता देता है (नीचे देखें)।

#### वैलिडेशन प्रॉम्प्ट्स

`config/config.json` के अंदर "primary validation prompt" होता है। आप `config/user_prompt_library.json` में prompts की लाइब्रेरी भी बना सकते हैं:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "portrait": "A high quality portrait of a woman, cinematic lighting, 8k",
  "landscape": "A beautiful mountain landscape at sunset, oil painting style"
}
```
</details>

इसे सक्षम करने के लिए अपने `config.json` में यह जोड़ें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "user_prompt_library": "config/user_prompt_library.json"
}
```
</details>

#### Flow schedule shifting

Kandinsky 5 एक flow‑matching मॉडल है। `shift` पैरामीटर training और inference के दौरान noise वितरण नियंत्रित करता है।

- **Shift 1.0 (डिफ़ॉल्ट)**: संतुलित प्रशिक्षण।
- **कम Shift (< 1.0)**: training का फोकस high‑frequency विवरण (texture, noise) पर बढ़ता है।
- **अधिक Shift (> 1.0)**: training का फोकस low‑frequency विवरण (composition, color, structure) पर बढ़ता है।

यदि मॉडल styles सीखता है लेकिन composition में असफल है, तो shift बढ़ाएँ। यदि composition सीखता है लेकिन texture कम है, तो shift घटाएँ।

#### Quantised model training

Transformer को 8‑bit में quantize करके आप VRAM उपयोग काफी कम कर सकते हैं।

`config.json` में:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "base_model_default_dtype": "bf16"
```
</details>

> **नोट**: हम टेक्स्ट एन्कोडर्स को quantize करने की सलाह नहीं देते (`no_change`), क्योंकि Qwen2.5‑VL quantization प्रभावों के प्रति संवेदनशील है और पाइपलाइन में पहले से ही सबसे भारी भाग है।

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

#### डेटासेट विचार

आपको एक dataset configuration फ़ाइल चाहिए, जैसे `config/multidatabackend.json`.

```json
[
  {
    "id": "my-image-dataset",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/my_images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "crop": true,
    "crop_aspect": "square",
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

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

फिर अपना dataset डायरेक्टरी बनाएँ:

```bash
mkdir -p datasets/my_images
</details>

# Copy your images and .txt caption files here
```

#### WandB और Huggingface Hub में लॉग‑इन

```bash
wandb login
huggingface-cli login
```

### प्रशिक्षण रन निष्पादित करना

**विकल्प 1 (अनुशंसित):**

```bash
simpletuner train
```

**विकल्प 2 (Legacy):**

```bash
./train.sh
```

## नोट्स और समस्या‑समाधान टिप्स

### सबसे कम VRAM कॉन्फ़िग

16GB या सीमित 24GB सेटअप पर चलाने के लिए:

1. **Group Offload सक्षम करें**: `--enable_group_offload`.
2. **Base Model quantize करें**: `"base_model_precision": "int8-quanto"`.
3. **Batch Size**: इसे `1` रखें।

### Artifacts और "Burnt" images

यदि validation images over‑saturated या noisy ("burnt") दिखें:

- **Guidance जांचें**: `validation_guidance` को लगभग `5.0` रखें। अधिक मान (जैसे 7.0+) इस मॉडल पर इमेज को "जलाते" हैं।
- **Flow Shift जांचें**: बहुत अधिक/कम `flow_schedule_shift` अस्थिरता ला सकता है। शुरुआत में `1.0` पर रहें।
- **Learning Rate**: LoRA के लिए 1e-4 मानक है, लेकिन यदि artifacts दिखें तो 5e-5 तक घटाएँ।

### TREAD प्रशिक्षण

Kandinsky 5 तेज़ प्रशिक्षण के लिए टोकन ड्रॉप करने हेतु [TREAD](../TREAD.md) का समर्थन करता है।

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

यह middle layers में 50% tokens गिराता है, जिससे transformer pass तेज़ होता है।
