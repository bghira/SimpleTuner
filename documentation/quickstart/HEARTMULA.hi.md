# HeartMuLa क्विकस्टार्ट

इस उदाहरण में, हम HeartMuLa oss 3B ऑडियो जेनरेशन मॉडल को प्रशिक्षित करेंगे।

## अवलोकन

HeartMuLa एक 3B पैरामीटर वाला ऑटोरिग्रेसिव ट्रांसफ़ॉर्मर है, जो टैग्स और लिरिक्स से डिस्क्रीट ऑडियो टोकन प्रेडिक्ट करता है। टोकन को HeartCodec के जरिए डिकोड करके वेवफॉर्म बनते हैं।

## हार्डवेयर आवश्यकताएँ

HeartMuLa 3B पैरामीटर वाला मॉडल है, इसलिए यह Flux जैसे बड़े इमेज जेनरेशन मॉडल्स की तुलना में अपेक्षाकृत हल्का है।

- **न्यूनतम:** 12GB+ VRAM वाला NVIDIA GPU (जैसे, 3060, 4070)।
- **अनुशंसित:** बड़े बैच साइज़ के लिए 24GB+ VRAM वाला NVIDIA GPU (जैसे, 3090, 4090, A10G)।
- **Mac:** Apple Silicon पर MPS के माध्यम से सपोर्टेड (लगभग 36GB+ यूनिफ़ाइड मेमोरी आवश्यक)।

### स्टोरेज आवश्यकताएँ

> ⚠️ **टोकन डेटासेट चेतावनी:** HeartMuLa प्रीकम्प्यूटेड ऑडियो टोकन पर ट्रेन करता है। SimpleTuner ट्रेनिंग के दौरान टोकन नहीं बनाता, इसलिए आपके डेटासेट में `audio_tokens` या `audio_tokens_path` मेटाडेटा होना चाहिए। टोकन फ़ाइलें बड़ी हो सकती हैं, इसलिए डिस्क स्पेस का ध्यान रखें।

> 💡 **टिप:** `int8-quanto` क्वांटाइज़ेशन का उपयोग कम VRAM (जैसे 12GB‑16GB) वाले GPUs पर भी न्यूनतम गुणवत्ता हानि के साथ प्रशिक्षण संभव बनाता है।

## पूर्वापेक्षाएँ

सुनिश्चित करें कि आपके पास Python 3.10+ का कार्यरत वातावरण है।

```bash
pip install simpletuner
```

## कॉन्फ़िगरेशन

अपनी कॉन्फ़िग्स को व्यवस्थित रखना अनुशंसित है। हम इस डेमो के लिए एक समर्पित फ़ोल्डर बनाएँगे।

```bash
mkdir -p config/heartmula-training-demo
```

### महत्वपूर्ण सेटिंग्स

`config/heartmula-training-demo/config.json` को इन वैल्यूज़ के साथ बनाएँ:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "model_family": "heartmula",
  "model_type": "lora",
  "model_flavour": "3b",
  "pretrained_model_name_or_path": "HeartMuLa/HeartMuLa-oss-3B",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/heartmula-training-demo/multidatabackend.json"
}
```
</details>

### वैलिडेशन सेटिंग्स

प्रोग्रेस मॉनिटर करने के लिए `config.json` में ये जोड़ें:

- **`validation_prompt`**: टैग्स या ऑडियो का टेक्स्ट विवरण (जैसे, "ब्राइट सिंथ्स के साथ अपबीट पॉप").
- **`validation_lyrics`**: (वैकल्पिक) मॉडल से गाने के लिए लिरिक्स। इंस्ट्रूमेंटल के लिए खाली स्ट्रिंग दें।
- **`validation_prompt_library`**: built-in caption + lyrics validation library के लिए `"audio"` इस्तेमाल करें।
- **`validation_audio_duration`**: वैलिडेशन क्लिप की अवधि (सेकंड, डिफ़ॉल्ट: 30.0)।
- **`validation_guidance`**: गाइडेंस स्केल (1.5 - 3.0 के आसपास से शुरू करें)।
- **`validation_step_interval`**: कितनी बार सैंपल जनरेट करने हैं (जैसे, हर 100 स्टेप)।

### एडवांस्ड एक्सपेरिमेंटल फीचर्स

<details>
<summary>एडवांस्ड एक्सपेरिमेंटल डिटेल्स दिखाएँ</summary>


SimpleTuner में एक्सपेरिमेंटल फीचर्स हैं जो ट्रेनिंग की स्थिरता और प्रदर्शन को काफी बेहतर बना सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** एक्सपोज़र बायस घटाता है और ट्रेनिंग के दौरान मॉडल को अपनी इनपुट खुद जनरेट करने देकर आउटपुट क्वालिटी सुधारता है।

> ⚠️ इन फीचर्स से ट्रेनिंग का कंप्यूटेशनल ओवरहेड बढ़ता है।

</details>

## डेटासेट कॉन्फ़िगरेशन

HeartMuLa को प्रीकम्प्यूटेड टोकन वाला **ऑडियो‑स्पेसिफ़िक** डेटासेट चाहिए।

किसी target vocal identity को styles या genres में expand करने के लिए [Voice Cloning Data Transforms](../experimental/VOICE_CLONING.hi.md) में दिए RVC `data_transforms` workflow को configure करें।

हर सैंपल में यह होना चाहिए:

- `tags` (स्ट्रिंग)
- `lyrics` (स्ट्रिंग; खाली हो सकती है)
- `audio_tokens` या `audio_tokens_path`

टोकन ऐरे 2D होना चाहिए और उसका आकार `[frames, num_codebooks]` या `[num_codebooks, frames]` होना चाहिए।

> 💡 **नोट:** HeartMuLa अलग टेक्स्ट एन्कोडर उपयोग नहीं करता, इसलिए text-embeds बैकएंड की जरूरत नहीं है।

### विकल्प 1: Hugging Face डेटासेट (कॉलम में टोकन)

`config/heartmula-training-demo/multidatabackend.json` बनाएँ:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "heartmula-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "your-org/heartmula-audio-tokens",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "config": {
      "audio_caption_fields": ["tags"],
      "lyrics_column": "lyrics"
    }
  }
]
```
</details>

सुनिश्चित करें कि आपके डेटासेट में टेक्स्ट फ़ील्ड्स के साथ `audio_tokens` या `audio_tokens_path` कॉलम भी हों।

### विकल्प 2: लोकल ऑडियो फ़ाइलें + टोकन मेटाडेटा

`config/heartmula-training-demo/multidatabackend.json` बनाएँ:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "my-audio-dataset",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/my_audio_files",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "disabled": false
  }
]
```
</details>

सुनिश्चित करें कि आपका मेटाडेटा बैकएंड हर सैंपल के लिए `audio_tokens` या `audio_tokens_path` प्रदान करता है।

### डेटा स्ट्रक्चर

अपने ऑडियो फ़ाइलें `datasets/my_audio_files` में रखें। SimpleTuner कई फ़ॉर्मेट्स को सपोर्ट करता है:

- **लॉसलेस:** `.wav`, `.flac`, `.aiff`, `.alac`
- **लॉसी:** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **नोट:** MP3, AAC और WMA जैसे फ़ॉर्मेट्स के लिए **FFmpeg** इंस्टॉल होना चाहिए।

अगर आप `caption_strategy: textfile` इस्तेमाल करते हैं, तो टैग्स और लिरिक्स की टेक्स्ट फ़ाइलें ऑडियो फ़ाइलों के साथ रखें:

- **ऑडियो:** `track_01.wav`
- **टैग्स (Prompt):** `track_01.txt` (जैसे, "धीमा जैज़ बैलेड")
- **लिरिक्स (वैकल्पिक):** `track_01.lyrics`

टोकन ऐरे मेटाडेटा के जरिए दें (उदाहरण के लिए, `.npy` या `.npz` फ़ाइलों की ओर इशारा करने वाले `audio_tokens_path` एंट्रीज़)।

<details>
<summary>उदाहरण डेटासेट लेआउट</summary>

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
├── track_01.lyrics
└── track_01.tokens.npy
```
</details>

> ⚠️ **लिरिक्स पर नोट:** HeartMuLa हर सैंपल में लिरिक्स स्ट्रिंग चाहता है। इंस्ट्रूमेंटल डेटा के लिए खाली स्ट्रिंग दें, फ़ील्ड हटाएँ नहीं।

## ट्रेनिंग

अपने एनवायरनमेंट के साथ ट्रेनिंग शुरू करें:

```bash
simpletuner train env=heartmula-training-demo
```

यह कमांड `config/heartmula-training-demo/` के अंदर `config.json` को ढूंढता है।

> 💡 **टिप (ट्रेनिंग जारी रखें):** किसी मौजूदा LoRA से फाइन‑ट्यूनिंग जारी रखने के लिए `--init_lora` विकल्प का उपयोग करें:
> ```bash
> simpletuner train env=heartmula-training-demo --init_lora=/path/to/existing_lora.safetensors
> ```

## ट्रबलशूटिंग

- **वैलिडेशन त्रुटियाँ:** `num_validation_images` > 1 जैसे इमेज‑केंद्रित वैलिडेशन फीचर्स (ऑडियो में यह बैच साइज़ का समकक्ष है) या CLIP स्कोर जैसी इमेज‑आधारित मीट्रिक्स का उपयोग न करें।
- **मेमोरी समस्याएँ:** अगर OOM हो रहा हो, `train_batch_size` कम करें या `gradient_checkpointing` सक्षम करें।
