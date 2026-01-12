# LongCat‑Video क्विकस्टार्ट

LongCat‑Video एक 13.6B द्विभाषी (zh/en) टेक्स्ट‑टू‑वीडियो और इमेज‑टू‑वीडियो मॉडल है जो फ्लो मैचिंग, Qwen‑2.5‑VL टेक्स्ट एन्कोडर, और Wan VAE का उपयोग करता है। यह गाइड SimpleTuner के साथ सेटअप, डेटा तैयारी, और पहली ट्रेनिंग/वैलिडेशन चलाने की प्रक्रिया बताती है।

---

## 1) हार्डवेयर आवश्यकताएँ (क्या अपेक्षा करें)

- 13.6B ट्रांसफ़ॉर्मर + Wan VAE: इमेज मॉडलों से अधिक VRAM की अपेक्षा करें; `train_batch_size=1`, gradient checkpointing, और कम LoRA ranks से शुरू करें।
- सिस्टम RAM: multi‑frame clips के लिए 32 GB से अधिक मददगार है; डेटासेट तेज़ स्टोरेज पर रखें।
- Apple MPS: previews के लिए समर्थित; positional encodings अपने‑आप float32 में डाउनकास्ट हो जाते हैं।

---

## 2) पूर्वापेक्षाएँ

1. Python 3.12 सत्यापित करें (SimpleTuner डिफ़ॉल्ट रूप से `.venv` देता है):
   ```bash
   python --version
   ```
2. अपने हार्डवेयर के अनुसार बैकएंड के साथ SimpleTuner इंस्टॉल करें:
   ```bash
   pip install "simpletuner[cuda]"   # NVIDIA
   pip install "simpletuner[mps]"    # Apple Silicon
   pip install "simpletuner[cpu]"    # CPU-only
   ```
3. क्वांटाइज़ेशन बिल्ट‑इन है (`int8-quanto`, `int4-quanto`, `fp8-torchao`) और सामान्य सेटअप में अतिरिक्त इंस्टॉल की जरूरत नहीं।

---

## 3) वातावरण सेटअप

### Web UI
```bash
simpletuner server
```
http://localhost:8001 खोलें और मॉडल फैमिली `longcat_video` चुनें।

### CLI बेसलाइन (config/config.json)

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_video",
  "model_flavour": "final",
  "pretrained_model_name_or_path": null,      // auto-selected from flavour
  "base_model_precision": "bf16",             // int8-quanto/fp8-torchao also work for LoRA
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "learning_rate": 1e-4,
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0
}
```

**मुख्य डिफ़ॉल्ट्स जिन्हें रखना चाहिए**
- shift `12.0` के साथ फ्लो‑मैचिंग शेड्यूलर स्वचालित है; किसी custom noise फ़्लैग की जरूरत नहीं।
- आस्पेक्ट बकेट्स 64‑पिक्सेल अलाइन्ड रहते हैं; `aspect_bucket_alignment` 64 पर ही रखा जाता है।
- अधिकतम टोकन लंबाई 512 (Qwen‑2.5‑VL); CFG ऑन होने पर और नेगेटिव प्रॉम्प्ट न होने पर पाइपलाइन खाली नेगेटिव जोड़ती है।
- फ्रेम्स को `(num_frames - 1)` VAE temporal stride (डिफ़ॉल्ट 4) से विभाज्य होना चाहिए। डिफ़ॉल्ट 93 फ्रेम्स पहले से मैच करते हैं।

वैकल्पिक VRAM बचत विकल्प:
- `lora_rank` कम करें (4–8) और `int8-quanto` बेस प्रिसिजन उपयोग करें।
- group offload सक्षम करें: `--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`.
- यदि previews OOM करें तो पहले `validation_resolution`, frames, या steps घटाएँ।
- Attention डिफ़ॉल्ट: CUDA पर, LongCat‑Video उपलब्ध होने पर bundled block‑sparse Triton kernel का उपयोग करता है और अन्यथा standard dispatcher पर fallback करता है। कोई टॉगल ज़रूरी नहीं। यदि आपको खास तौर पर xFormers चाहिए, तो config/CLI में `attention_implementation: "xformers"` सेट करें।

### प्रशिक्षण शुरू करें (CLI)
```bash
simpletuner train --config config/config.json
```
या Web UI लॉन्च करके उसी कॉन्फ़िग के साथ जॉब सबमिट करें।

---

## 4) Dataloader मार्गदर्शन

- कैप्शन वाली वीडियो डेटासेट्स उपयोग करें; हर सैंपल में फ्रेम्स (या छोटा क्लिप) और टेक्स्ट कैप्शन होना चाहिए। `dataset_type: video` को `VideoToTensor` स्वतः संभालता है।
- फ्रेम आयाम 64px ग्रिड पर रखें (जैसे 480x832, 720p buckets)। height/width को Wan VAE stride (बिल्ट‑इन सेटिंग्स में 16px) और 64 से विभाज्य होना चाहिए।
- image‑to‑video रन के लिए, प्रति सैंपल एक conditioning इमेज शामिल करें; इसे पहले latent फ्रेम में रखा जाता है और sampling के दौरान स्थिर रखा जाता है।
- LongCat‑Video डिफ़ॉल्ट रूप से 30 fps है। 93 फ्रेम्स ~3.1 s होते हैं; यदि फ्रेम काउंट बदलें, तो `(frames - 1) % 4 == 0` रखें और अवधि fps के साथ स्केल होती है।

### वीडियो बकेट रणनीति

अपने डेटासेट के `video` सेक्शन में, आप वीडियो को कैसे समूहित किया जाए यह कॉन्फ़िगर कर सकते हैं:
- `bucket_strategy`: `aspect_ratio` (डिफ़ॉल्ट) spatial aspect ratio से समूहित करता है। `resolution_frames` `WxH@F` फॉर्मैट (जैसे `480x832@93`) से mixed resolution/duration डेटासेट्स के लिए समूह बनाता है।
- `frame_interval`: `resolution_frames` उपयोग करते समय फ्रेम काउंट को इस इंटरवल तक राउंड करें (जैसे VAE temporal stride से मैच करने के लिए 4)।

---

## 5) वैलिडेशन और इंफ़रेंस

- Guidance: 3.5–5.0 अच्छा काम करता है; CFG सक्षम होने पर खाली नेगेटिव प्रॉम्प्ट अपने‑आप बनते हैं।
- Steps: गुणवत्ता जांच के लिए 35–45; तेज़ previews के लिए कम रखें।
- Frames: डिफ़ॉल्ट 93 (VAE temporal stride 4 के अनुरूप)।
- Previews या training के लिए अधिक headroom चाहिए? `musubi_blocks_to_swap` (4–8 से शुरू करें) सेट करें और वैकल्पिक रूप से `musubi_block_swap_device` ताकि forward/backward के दौरान आख़िरी transformer blocks CPU से स्ट्रीम हों। इससे ट्रांसफ़र ओवरहेड बढ़ेगा लेकिन VRAM peak कम होंगे।

- Validation रन आपके कॉन्फ़िग के `validation_*` फ़ील्ड्स से या WebUI preview टैब से चलते हैं जब `simpletuner server` शुरू हो। standalone CLI subcommand के बजाय इन्हीं पाथ्स का उपयोग करें।
- dataset‑driven validation (I2V सहित) के लिए `validation_using_datasets: true` सेट करें और `eval_dataset_id` को अपने validation split पर पॉइंट करें। यदि वह split `is_i2v` मार्क है और conditioning frames जुड़े हैं, तो पाइपलाइन पहला फ्रेम अपने‑आप स्थिर रखती है।
- चैनल mismatch से बचने के लिए latent previews decode से पहले unpack होती हैं।

---

## 6) समस्या समाधान

- **Height/width errors**: सुनिश्चित करें कि दोनों 16 से विभाज्य हों और 64px ग्रिड पर रहें।
- **MPS float64 warnings**: आंतरिक रूप से संभाले जाते हैं; प्रिसिजन bf16/float32 रखें।
- **OOM**: पहले validation resolution/frames घटाएँ, फिर LoRA rank कम करें, group offload सक्षम करें, या `int8-quanto`/`fp8-torchao` पर जाएँ।
- **CFG के साथ blank negatives**: यदि नेगेटिव प्रॉम्प्ट न दें, तो पाइपलाइन खाली नेगेटिव अपने‑आप डालती है।

---

## 7) Flavours

- `final`: मुख्य LongCat‑Video रिलीज़ (text‑to‑video + image‑to‑video एक ही चेकपॉइंट में)।
