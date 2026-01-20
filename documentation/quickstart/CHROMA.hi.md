# Chroma 1 क्विकस्टार्ट

![image](https://github.com/user-attachments/assets/3c8a12c6-9d45-4dd4-9fc8-6b7cd3ed51dd)

Chroma 1, Lodestone Labs द्वारा रिलीज़ किया गया Flux.1 Schnell का 8.9B‑पैरामीटर वाला trimmed वेरिएंट है। यह गाइड LoRA प्रशिक्षण के लिए SimpleTuner कॉन्फ़िगर करने की प्रक्रिया बताती है।

## हार्डवेयर आवश्यकताएँ

पैरामीटर कम होने के बावजूद मेमोरी उपयोग Flux Schnell के करीब है:

- बेस ट्रांसफ़ॉर्मर को quantise करने पर **≈40–50 GB** सिस्टम RAM लग सकती है।
- Rank‑16 LoRA प्रशिक्षण आमतौर पर:
  - बिना base quantisation के ~28 GB VRAM
  - int8 + bf16 के साथ ~16 GB VRAM
  - int4 + bf16 के साथ ~11 GB VRAM
  - NF4 + bf16 के साथ ~8 GB VRAM
- यथार्थवादी GPU न्यूनतम: **RTX 3090 / RTX 4090 / L40S** क्लास या बेहतर।
- **Apple M‑series (MPS)** पर LoRA प्रशिक्षण और AMD ROCm पर अच्छी तरह चलता है।
- फुल‑रैंक fine‑tuning के लिए 80 GB‑क्लास accelerators या multi‑GPU सेटअप अनुशंसित हैं।

## पूर्वापेक्षाएँ

Chroma का रनटाइम Flux गाइड के समान है:

- Python **3.10 – 3.12**
- समर्थित accelerator backend (CUDA, ROCm, या MPS)

Python संस्करण जांचें:

```bash
python3 --version
```

SimpleTuner इंस्टॉल करें (CUDA उदाहरण):

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

बैकएंड‑विशिष्ट सेटअप विवरण (CUDA, ROCm, Apple) के लिए [installation guide](../INSTALL.md) देखें।

## वेब UI लॉन्च करना

```bash
simpletuner server
```

UI http://localhost:8001 पर उपलब्ध होगा।

## CLI के जरिए कॉन्फ़िगरेशन

`simpletuner configure` आपको core settings में मार्गदर्शन करता है। Chroma के लिए key values:

- `model_type`: `lora`
- `model_family`: `chroma`
- `model_flavour`: इनमें से एक
  - `base` (डिफ़ॉल्ट, balanced quality)
  - `hd` (higher fidelity, ज़्यादा compute)
  - `flash` (तेज़ लेकिन unstable – production के लिए अनुशंसित नहीं)
- `pretrained_model_name_or_path`: ऊपर दिए flavour mapping के लिए खाली छोड़ें
- `model_precision`: डिफ़ॉल्ट `bf16` रखें
- `flux_fast_schedule`: **disabled** रखें; Chroma का अपना adaptive sampling है

### उदाहरण मैनुअल कॉन्फ़िग स्निपेट

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```jsonc
{
  "model_type": "lora",
  "model_family": "chroma",
  "model_flavour": "base",
  "output_dir": "/workspace/chroma-output",
  "network_rank": 16,
  "learning_rate": 2.0e-4,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "pretrained_model_name_or_path": null
}
```
</details>

> ⚠️ यदि आपके क्षेत्र में Hugging Face एक्सेस धीमा है, तो लॉन्च करने से पहले `HF_ENDPOINT=https://hf-mirror.com` export करें।

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

</details>

## डेटासेट और dataloader

Chroma वही dataloader फॉर्मैट उपयोग करता है जो Flux में है। डेटासेट तैयारी और prompt लाइब्रेरी के लिए [general tutorial](../TUTORIAL.md) या [web UI tutorial](../webui/TUTORIAL.md) देखें।

## Chroma‑विशिष्ट प्रशिक्षण विकल्प

- `flux_lora_target`: तय करता है कि किन transformer modules पर LoRA adapters लगेंगे (`all`, `all+ffs`, `context`, `tiny`, आदि)। डिफ़ॉल्ट्स Flux जैसे हैं और अधिकतर मामलों में अच्छे हैं।
- `flux_guidance_mode`: `constant` अच्छा काम करता है; Chroma guidance range एक्सपोज़ नहीं करता।
- Attention masking हमेशा enabled है — सुनिश्चित करें कि आपका text embedding cache padding masks के साथ जनरेट हुआ है (मौजूदा SimpleTuner रिलीज़ में डिफ़ॉल्ट व्यवहार)।
- Schedule shift विकल्प (`flow_schedule_shift` / `flow_schedule_auto_shift`) Chroma के लिए आवश्यक नहीं — helper पहले से tail timesteps बढ़ा देता है।
- `flux_t5_padding`: यदि आप masked करने से पहले padded tokens को zero करना चाहते हैं तो `zero` सेट करें।

## Automatic tail timestep sampling

Flux ने log‑normal शेड्यूल उपयोग किया था जिससे high‑noise / low‑noise extremes कम sample होते थे। Chroma का training helper sampled sigmas पर quadratic (`σ ↦ σ²` / `1-(1-σ)²`) remapping लगाता है ताकि tail क्षेत्रों में अधिक बार जाए। इसके लिए **कोई अतिरिक्त कॉन्फ़िगरेशन नहीं** चाहिए — यह `chroma` मॉडल फैमिली में बिल्ट‑इन है।

## वैलिडेशन और सैंपलिंग टिप्स

- `validation_guidance_real` सीधे pipeline के `guidance_scale` से मैप होता है। single‑pass sampling के लिए इसे `1.0` रखें, या validation renders के दौरान classifier‑free guidance चाहिए तो `2.0`–`3.0` तक बढ़ाएँ।
- तेज़ previews के लिए 20 inference steps; उच्च गुणवत्ता के लिए 28–32।
- Negative prompts वैकल्पिक हैं; बेस मॉडल पहले से de‑distilled है।
- मॉडल फिलहाल केवल text‑to‑image सपोर्ट करता है; img2img सपोर्ट बाद के अपडेट में आएगा।

## समस्या समाधान

- **Startup पर OOM**: `offload_during_startup` सक्षम करें या बेस मॉडल को quantise करें (`base_model_precision: int8-quanto`)।
- **Training जल्दी diverge हो**: gradient checkpointing ऑन रखें, `learning_rate` को `1e-4` तक घटाएँ, और कैप्शन्स में विविधता जाँचें।
- **Validation में वही pose दोहरता है**: prompts को लंबा करें; flow‑matching मॉडल कम विविधता पर collapse होते हैं।

उन्नत विषयों—DeepSpeed, FSDP2, evaluation metrics—के लिए README में लिंक किए गए साझा गाइड्स देखें।
