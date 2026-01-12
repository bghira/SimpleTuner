# LongCat‑Image क्विकस्टार्ट

LongCat‑Image एक 6B द्विभाषी (zh/en) टेक्स्ट‑टू‑इमेज मॉडल है जो फ्लो मैचिंग और Qwen‑2.5‑VL टेक्स्ट एन्कोडर का उपयोग करता है। यह गाइड SimpleTuner के साथ सेटअप, डेटा तैयारी, और पहली ट्रेनिंग/वैलिडेशन चलाने की प्रक्रिया बताती है।

---

## 1) हार्डवेयर आवश्यकताएँ (क्या अपेक्षा करें)

- VRAM: `int8-quanto` या `fp8-torchao` पर 1024px LoRA के लिए 16–24 GB पर्याप्त है। फुल bf16 रन के लिए ~24 GB लग सकते हैं।
- सिस्टम RAM: ~32 GB सामान्यतः पर्याप्त है।
- Apple MPS: inference/preview के लिए समर्थित; MPS पर dtype समस्याओं से बचने के लिए हम pos‑ids को float32 में डाउनकास्ट करते हैं।

---

## 2) पूर्वापेक्षाएँ (चरण‑दर‑चरण)

1. Python 3.10–3.12 सत्यापित करें:
   ```bash
   python --version
   ```
2. (Linux/CUDA) नए इमेज पर सामान्य build/toolchain पैकेज इंस्टॉल करें:
   ```bash
   apt -y update
   apt -y install build-essential nvidia-cuda-toolkit
   ```
3. अपने बैकएंड के अनुसार SimpleTuner इंस्टॉल करें:
   ```bash
   pip install "simpletuner[cuda]"   # CUDA
   pip install "simpletuner[mps]"    # Apple Silicon
   pip install "simpletuner[cpu]"    # CPU-only
   ```
4. क्वांटाइज़ेशन बिल्ट‑इन है (`int8-quanto`, `int4-quanto`, `fp8-torchao`) और सामान्य सेटअप में अतिरिक्त मैनुअल इंस्टॉल की आवश्यकता नहीं होती।

---

## 3) वातावरण सेटअप

### Web UI (सबसे निर्देशित)
```bash
simpletuner server
```
http://localhost:8001 खोलें और मॉडल फैमिली `longcat_image` चुनें।

### CLI बेसलाइन (config/config.json)

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_image",
  "model_flavour": "final",                // options: final, dev
  "pretrained_model_name_or_path": null,   // auto-selected from flavour; override with a local path if needed
  "base_model_precision": "int8-quanto",   // good default; fp8-torchao also works
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 16,
  "learning_rate": 1e-4,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 30
}
```

**मुख्य डिफ़ॉल्ट्स जिन्हें रखना चाहिए**
- फ्लो‑मैचिंग शेड्यूलर स्वचालित है; किसी विशेष शेड्यूल फ़्लैग की आवश्यकता नहीं।
- आस्पेक्ट बकेट्स 64‑पिक्सेल अलाइन्ड रहें; `aspect_bucket_alignment` न घटाएँ।
- अधिकतम टोकन लंबाई 512 (Qwen‑2.5‑VL)।

वैकल्पिक मेमोरी बचत विकल्प (अपने हार्डवेयर के अनुसार चुनें):
- `--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`
- `lora_rank` कम करें (4–8) और/या `int8-quanto` बेस प्रिसिजन इस्तेमाल करें।
- यदि वैलिडेशन OOM करे, पहले `validation_resolution` या स्टेप्स घटाएँ।

### तेज़ कॉन्फ़िग निर्माण (एक‑बार)
```bash
cp config/config.json.example config/config.json
```
ऊपर दिए गए फ़ील्ड्स (model_family, flavour, precision, paths) संपादित करें। `output_dir` और डेटासेट पाथ्स को अपने स्टोरेज की ओर इंगित करें।

### प्रशिक्षण शुरू करें (CLI)
```bash
simpletuner train --config config/config.json
```
या WebUI लॉन्च करके Jobs पेज से उसी कॉन्फ़िग के साथ रन शुरू करें।

---

## 4) Dataloader संकेत (क्या देना है)

- मानक कैप्शन वाली इमेज फ़ोल्डर्स (textfile/JSON/CSV) काम करते हैं। यदि आप द्विभाषी क्षमता बनाए रखना चाहते हैं, तो zh/en दोनों शामिल करें।
- बकेट किनारों को 64px ग्रिड पर रखें। यदि आप multi‑aspect ट्रेन करते हैं, तो कई resolutions सूचीबद्ध करें (जैसे `1024x1024,1344x768`)।
- VAE KL है shift+scale के साथ; कैश बिल्ट‑इन स्केलिंग फ़ैक्टर स्वतः उपयोग करता है।

---

## 5) वैलिडेशन और इंफ़रेंस

- Guidance: 4–6 अच्छा शुरुआती मान है; नेगेटिव प्रॉम्प्ट खाली रखें।
- Steps: तेज़ चेक्स के लिए ~30; सर्वोत्तम गुणवत्ता के लिए 40–50।
- Validation preview डिफ़ॉल्ट रूप से काम करता है; चैनल mismatch से बचने के लिए latents decode से पहले unpack होते हैं।

उदाहरण (CLI validate):
```bash
simpletuner validate \
  --model_family longcat_image \
  --model_flavour final \
  --validation_resolution 1024x1024 \
  --validation_num_inference_steps 30 \
  --validation_guidance 4.5
```

---

## 6) समस्या समाधान

- **MPS float64 errors**: आंतरिक रूप से संभाले जाते हैं; कॉन्फ़िग को float32/bf16 पर रखें।
- **Previews में चैनल mismatch**: decode से पहले latents unpack किए जाते हैं (इस गाइड वाले SimpleTuner संस्करण में शामिल)।
- **OOM**: `validation_resolution` घटाएँ, `lora_rank` कम करें, group offload सक्षम करें, या `int8-quanto` / `fp8-torchao` पर जाएँ।
- **धीमा टोकनाइज़ेशन**: Qwen‑2.5‑VL 512 टोकन तक सीमित है; बहुत लंबे प्रॉम्प्ट से बचें।

---

## 7) Flavour चयन
- `final`: मुख्य रिलीज़ (सर्वोत्तम गुणवत्ता)।
- `dev`: mid‑training चेकपॉइंट (प्रयोग/फाइन‑ट्यून के लिए)।
