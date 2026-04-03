# ACE-Step क्विकस्टार्ट

इस उदाहरण में, हम ACE‑Step ऑडियो जेनरेशन मॉडल को प्रशिक्षित करेंगे। SimpleTuner वर्तमान में मूल ACE‑Step v1 3.5B पाथ और ACE‑Step v1.5 bundle के लिए forward-compatible LoRA प्रशिक्षण, दोनों को सपोर्ट करता है।

## अवलोकन

ACE-Step उच्च गुणवत्ता वाली ऑडियो सिंथेसिस के लिए डिज़ाइन किया गया ट्रांसफ़ॉर्मर‑आधारित फ्लो‑मैचिंग ऑडियो मॉडल है। SimpleTuner में:

- `base` मूल ACE‑Step v1 3.5B प्रशिक्षण पाथ का उपयोग करता है।
- `v15-turbo`, `v15-base`, और `v15-sft` `ACE-Step/Ace-Step1.5` से लोड होने वाले ACE‑Step v1.5 bundle variants का उपयोग करते हैं।

## हार्डवेयर आवश्यकताएँ

ACE-Step 3.5B पैरामीटर वाला मॉडल है, इसलिए यह Flux जैसे बड़े इमेज जेनरेशन मॉडल्स की तुलना में अपेक्षाकृत हल्का है।

- **न्यूनतम:** 12GB+ VRAM वाला NVIDIA GPU (जैसे, 3060, 4070)।
- **अनुशंसित:** बड़े बैच साइज़ के लिए 24GB+ VRAM वाला NVIDIA GPU (जैसे, 3090, 4090, A10G)।
- **Mac:** Apple Silicon पर MPS के माध्यम से सपोर्टेड (लगभग 36GB+ यूनिफ़ाइड मेमोरी आवश्यक)।

### स्टोरेज आवश्यकताएँ

> ⚠️ **डिस्क उपयोग चेतावनी:** ऑडियो मॉडलों के लिए VAE कैश काफ़ी बड़ा हो सकता है। उदाहरण के लिए, 60‑सेकंड का एक ऑडियो क्लिप ~89MB कैश्ड लेटेंट फ़ाइल बना सकता है। यह कैशिंग रणनीति प्रशिक्षण के दौरान VRAM आवश्यकता को काफी हद तक घटाने के लिए उपयोग होती है। सुनिश्चित करें कि आपके डेटासेट के कैश के लिए पर्याप्त डिस्क स्पेस है।

> 💡 **टिप:** बड़े डेटासेट के लिए, आप `--vae_cache_disable` विकल्प का उपयोग कर डिस्क पर एम्बेडिंग्स लिखना बंद कर सकते हैं। इससे ऑन‑डिमांड कैशिंग सक्षम होगी, जो डिस्क स्पेस बचाती है लेकिन प्रशिक्षण समय और मेमोरी उपयोग बढ़ाती है क्योंकि एनकोडिंग्स प्रशिक्षण लूप के दौरान होती हैं।

> 💡 **टिप:** `int8-quanto` क्वांटाइज़ेशन का उपयोग कम VRAM (जैसे 12GB‑16GB) वाले GPUs पर भी न्यूनतम गुणवत्ता हानि के साथ प्रशिक्षण संभव बनाता है।

## पूर्वापेक्षाएँ

सुनिश्चित करें कि आपके पास Python 3.10+ का कार्यरत वातावरण है।

```bash
pip install simpletuner
```

## कॉन्फ़िगरेशन

अपनी कॉन्फ़िग्स को व्यवस्थित रखना अनुशंसित है। हम इस डेमो के लिए एक समर्पित फ़ोल्डर बनाएँगे।

```bash
mkdir -p config/acestep-training-demo
```

### महत्वपूर्ण सेटिंग्स

SimpleTuner वर्तमान में ACE-Step के लिए ये flavours सपोर्ट करता है:

- `base`: मूल ACE‑Step v1 3.5B
- `v15-turbo`, `v15-base`, `v15-sft`: ACE‑Step v1.5 bundle variants

अपनी लक्ष्य variant के अनुसार उपयुक्त कॉन्फ़िग चुनें।

तुरंत उपयोग के लिए example presets यहाँ उपलब्ध हैं:

- `simpletuner/examples/ace_step-v1-0.peft-lora`
- `simpletuner/examples/ace_step-v1-5.peft-lora`

आप इन्हें सीधे `simpletuner train example=ace_step-v1-0.peft-lora` या `simpletuner train example=ace_step-v1-5.peft-lora` से चला सकते हैं।

#### ACE-Step v1 उदाहरण

इन वैल्यूज़ के साथ `config/acestep-training-demo/config.json` बनाएँ:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "model_family": "ace_step",
  "model_type": "lora",
  "model_flavour": "base",
  "pretrained_model_name_or_path": "ACE-Step/ACE-Step-v1-3.5B",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/acestep-training-demo/multidatabackend.json"
}
```
</details>

#### ACE-Step v1.5 उदाहरण

ACE-Step v1.5 के लिए `model_family: "ace_step"` को वैसा ही रखें, कोई v1.5 flavour चुनें, और checkpoint root को साझा v1.5 bundle पर सेट करें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "model_family": "ace_step",
  "model_type": "lora",
  "model_flavour": "v15-base",
  "pretrained_model_name_or_path": "ACE-Step/Ace-Step1.5",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/acestep-training-demo/multidatabackend.json"
}
```
</details>

### वैलिडेशन सेटिंग्स

प्रगति मॉनिटर करने के लिए इन्हें अपने `config.json` में जोड़ें:

- **`validation_prompt`**: उस ऑडियो का टेक्स्ट विवरण जिसे आप जनरेट करना चाहते हैं (जैसे, "उत्साही ड्रम्स वाला आकर्षक पॉप सॉन्ग").
- **`validation_lyrics`**: (वैकल्पिक) मॉडल के गाने के लिए लिरिक्स।
- **`validation_audio_duration`**: वैलिडेशन क्लिप्स की अवधि सेकंड में (डिफ़ॉल्ट: 30.0)।
- **`validation_guidance`**: गाइडेंस स्केल (डिफ़ॉल्ट: ~3.0 - 5.0)।
- **`validation_step_interval`**: सैंपल कितनी बार जनरेट करना है (जैसे, हर 100 स्टेप पर)।

> ⚠️ **ACE-Step v1.5 सीमा:** वर्तमान SimpleTuner integration v1.5 प्रशिक्षण को सपोर्ट करता है, लेकिन बिल्ट‑इन ACE-Step validation/inference pipeline अभी भी केवल v1.0 के लिए है। v1.5 runs के लिए in-loop validation बंद करें या upstream/external inference tooling से validation करें।

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

</details>

## डेटासेट कॉन्फ़िगरेशन

ACE-Step के लिए **ऑडियो‑विशिष्ट** डेटासेट कॉन्फ़िगरेशन आवश्यक है।

### विकल्प 1: डेमो डेटासेट (Hugging Face)

क्विक स्टार्ट के लिए, आप तैयार किया गया [ACEStep-Songs प्रीसेट](../data_presets/preset_audio_dataset_with_lyrics.md) उपयोग कर सकते हैं।

`config/acestep-training-demo/multidatabackend.json` बनाएँ:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "acestep-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "cache_dir_vae": "cache/vae/{model_family}/acestep-demo-data"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```
</details>

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

### विकल्प 2: लोकल ऑडियो फ़ाइलें

`config/acestep-training-demo/multidatabackend.json` बनाएँ:

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
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```
</details>

### डेटा संरचना

अपनी ऑडियो फ़ाइलें `datasets/my_audio_files` में रखें। SimpleTuner निम्न फॉर्मैट्स सहित व्यापक रेंज को सपोर्ट करता है:

- **लॉसलेस:** `.wav`, `.flac`, `.aiff`, `.alac`
- **लॉसी:** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **नोट:** MP3, AAC, और WMA जैसे फॉर्मैट्स के लिए आपके सिस्टम पर **FFmpeg** इंस्टॉल होना चाहिए।

कैप्शन और लिरिक्स के लिए, संबंधित टेक्स्ट फ़ाइलों को अपनी ऑडियो फ़ाइलों के पास रखें:

- **ऑडियो:** `track_01.wav`
- **कैप्शन (प्रॉम्प्ट):** `track_01.txt` (टेक्स्ट विवरण शामिल, जैसे "धीमा जैज़ बैलाड")
- **लिरिक्स (वैकल्पिक):** `track_01.lyrics` (लिरिक्स टेक्स्ट शामिल)

<details>
<summary>उदाहरण डेटासेट लेआउट</summary>

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```
</details>

> 💡 **उन्नत:** यदि आपका डेटासेट अलग नामकरण कन्वेंशन का उपयोग करता है (जैसे `_lyrics.txt`), तो आप इसे अपने डेटासेट कॉन्फ़िग में कस्टमाइज़ कर सकते हैं।

<details>
<summary>कस्टम लिरिक्स फ़ाइलनाम का उदाहरण देखें</summary>

```json
"audio": {
  "lyrics_filename_format": "{filename}_lyrics.txt"
}
```
</details>

> ⚠️ **लिरिक्स नोट:** यदि किसी सैंपल के लिए `.lyrics` फ़ाइल नहीं मिलती, तो लिरिक एम्बेडिंग्स को शून्य किया जाएगा। ACE-Step लिरिक कंडीशनिंग की अपेक्षा करता है; बिना लिरिक्स (इंस्ट्रुमेंटल) डेटा पर भारी प्रशिक्षण करने पर मॉडल को शून्य लिरिक इनपुट्स के साथ उच्च‑गुणवत्ता इंस्ट्रुमेंटल ऑडियो जनरेट करना सीखने के लिए अधिक प्रशिक्षण स्टेप्स की आवश्यकता हो सकती है।

## प्रशिक्षण

अपने वातावरण को निर्दिष्ट करके प्रशिक्षण रन शुरू करें:

```bash
simpletuner train env=acestep-training-demo
```

यह कमांड SimpleTuner को `config/acestep-training-demo/` के अंदर `config.json` देखने के लिए बताता है।

> ⚠️ **ACE-Step v1.5 सीमा:** वर्तमान SimpleTuner integration v1.5 प्रशिक्षण को सपोर्ट करता है, लेकिन बिल्ट‑इन ACE-Step validation/inference pipeline अभी भी केवल v1.0 के लिए है। v1.5 runs के लिए in-loop validation बंद करें या upstream/external inference tooling से validation करें।

> ℹ️ **वर्ज़न नोट:** `lyrics_embedder_train` अभी केवल ACE-Step v1 प्रशिक्षण पाथ पर लागू होता है। SimpleTuner में v1.5 forward-compatible LoRA पाथ decoder-only है।
