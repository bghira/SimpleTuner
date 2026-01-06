# LongCat‑Image Edit क्विकस्टार्ट

यह LongCat‑Image का edit/img2img वेरिएंट है। पहले [LONGCAT_IMAGE.md](../quickstart/LONGCAT_IMAGE.md) पढ़ें; यह फ़ाइल केवल बताती है कि edit flavour के लिए क्या बदलता है।

---

## 1) बेस LongCat‑Image के मुकाबले मॉडल अंतर

|                               | बेस (text2img) | Edit |
| ----------------------------- | -------------- | ---- |
| Flavour                       | `final` / `dev` | `edit` |
| कंडीशनिंग                     | none           | **conditioning latents (reference image) आवश्यक** |
| टेक्स्ट एन्कोडर                | Qwen‑2.5‑VL     | Qwen‑2.5‑VL **with vision context** (प्रॉम्प्ट एन्कोडिंग को ref image चाहिए) |
| पाइपलाइन                      | TEXT2IMG        | IMG2IMG/EDIT |
| वैलिडेशन इनपुट्स              | केवल prompt     | prompt **और** reference |

---

## 2) कॉन्फ़िग बदलाव (CLI/WebUI)

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_image",
  "model_flavour": "edit",
  "base_model_precision": "int8-quanto",      // fp8-torchao also fine; helps fit 16–24 GB
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "learning_rate": 5e-5,
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 40,
  "validation_resolution": "768x768"
}
```

`aspect_bucket_alignment` को 64 पर रखें। conditioning latents को अक्षम न करें; edit पाइपलाइन इन्हें अपेक्षित करती है।

तेज़ कॉन्फ़िग निर्माण:
```bash
cp config/config.json.example config/config.json
```
फिर `model_family`, `model_flavour`, dataset paths, और output_dir सेट करें।

---

## 3) Dataloader: paired edit + reference

दो aligned datasets उपयोग करें: **edit images** (कैप्शन = edit instruction) और **reference images**। edit dataset का `conditioning_data` reference dataset ID की ओर इंगित करना चाहिए। फ़ाइलनाम 1‑to‑1 मैच होने चाहिए।

```jsonc
[
  {
    "id": "edit-images",
    "type": "local",
    "instance_data_dir": "/data/edits",
    "caption_strategy": "textfile",
    "resolution": 768,
    "cache_dir_vae": "/cache/vae/longcat/edit",
    "conditioning_data": ["ref-images"]
  },
  {
    "id": "ref-images",
    "type": "local",
    "instance_data_dir": "/data/refs",
    "caption_strategy": null,
    "resolution": 768,
    "cache_dir_vae": "/cache/vae/longcat/ref"
  }
]
```

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

नोट्स:
- आस्पेक्ट बकेट्स 64px ग्रिड पर रखें।
- रेफ़रेंस कैप्शन वैकल्पिक हैं; अगर मौजूद हों तो वे edit captions को बदल देते हैं (आमतौर पर अवांछित)।
- edit और reference के लिए VAE कैश अलग पाथ हों।
- यदि कैश मिस या shape errors दिखें, दोनों डेटासेट के VAE कैश साफ़ करके दोबारा जनरेट करें।

---

## 4) वैलिडेशन विशिष्टताएँ

- वैलिडेशन के लिए conditioning latents बनाने हेतु reference images चाहिए। `edit-images` के validation split को `conditioning_data` के जरिए `ref-images` से जोड़ें।
- Guidance: 4–6 अच्छा काम करता है; नेगेटिव प्रॉम्प्ट खाली रखें।
- Preview callbacks समर्थित हैं; latents डिकोडर के लिए अपने‑आप unpack होते हैं।
- यदि वैलिडेशन में conditioning latents गायब दिखें, तो सुनिश्चित करें कि वैलिडेशन dataloader में edit और reference दोनों entries हैं और फ़ाइलनाम मैच करते हैं।

---

## 5) इंफ़रेंस / वैलिडेशन कमांड्स

त्वरित CLI वैलिडेशन:
```bash
simpletuner validate \
  --model_family longcat_image \
  --model_flavour edit \
  --validation_resolution 768x768 \
  --validation_guidance 4.5 \
  --validation_num_inference_steps 40
```

WebUI: **Edit** पाइपलाइन चुनें, सोर्स इमेज और edit निर्देश दोनों दें।

---

## 6) प्रशिक्षण शुरू (CLI)

कॉन्फ़िग और dataloader सेट होने के बाद:
```bash
simpletuner train --config config/config.json
```
सुनिश्चित करें कि training के दौरान reference dataset उपलब्ध है, ताकि conditioning latents गणना किए जा सकें या कैश से लोड हों।

---

## 7) समस्या समाधान

- **Missing conditioning latents**: `conditioning_data` से reference dataset जोड़ें और फ़ाइलनाम मैच कराएँ।
- **MPS dtype errors**: पाइपलाइन MPS पर pos‑ids को float32 में auto‑downgrade करती है; बाकी को float32/bf16 रखें।
- **Previews में चैनल mismatch**: previews डिकोड से पहले latents को un‑patchify करते हैं (यह SimpleTuner संस्करण रखें)।
- **Edit में OOM**: validation resolution/steps घटाएँ, `lora_rank` कम करें, group offload सक्षम करें, और `int8-quanto`/`fp8-torchao` प्राथमिकता दें।
