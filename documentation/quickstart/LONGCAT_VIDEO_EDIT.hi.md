# LongCat‑Video Edit (Image‑to‑Video) क्विकस्टार्ट

यह गाइड LongCat‑Video के image‑to‑video वर्कफ़्लो को प्रशिक्षण और वैलिडेशन के लिए बताती है। आपको flavours बदलने की आवश्यकता नहीं है; वही `final` चेकपॉइंट text‑to‑video और image‑to‑video दोनों को कवर करता है। अंतर आपके datasets और validation सेटिंग्स से आता है।

---

## 1) बेस LongCat‑Video के मुकाबले मॉडल अंतर

|                               | बेस (text2video) | Edit / I2V |
| ----------------------------- | ---------------- | ---------- |
| Flavour                       | `final`          | `final` (same weights) |
| कंडीशनिंग                     | none             | **conditioning frame आवश्यक** (पहला latent स्थिर रहता है) |
| टेक्स्ट एन्कोडर                | Qwen‑2.5‑VL      | Qwen‑2.5‑VL (same) |
| पाइपलाइन                      | TEXT2IMG         | IMG2VIDEO |
| वैलिडेशन इनपुट्स              | केवल prompt       | prompt **और** conditioning image |
| Buckets / stride              | 64px buckets, `(frames-1)%4==0` | same |

**मुख्य डिफ़ॉल्ट्स जो आपको मिलते हैं**
- shift `12.0` के साथ फ्लो‑मैचिंग।
- आस्पेक्ट बकेट्स 64px पर लागू।
- Qwen‑2.5‑VL टेक्स्ट एन्कोडर; CFG ऑन होने पर खाली नेगेटिव अपने‑आप जोड़े जाते हैं।
- डिफ़ॉल्ट फ्रेम्स: 93 (`(frames-1)%4==0` को संतुष्ट करता है)।

---

## 2) कॉन्फ़िग बदलाव (CLI/WebUI)

```jsonc
{
  "model_family": "longcat_video",
  "model_flavour": "final",
  "model_type": "lora",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "learning_rate": 1e-4,
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0,
  "validation_using_datasets": true,
  "eval_dataset_id": "longcat-video-val"
}
```

`aspect_bucket_alignment` को 64 पर रखें। पहला latent फ्रेम स्टार्ट इमेज रखता है; उसे बदले नहीं। 93 फ्रेम्स पर बने रहें (VAE stride नियम `(frames - 1) % 4 == 0` से पहले से मेल खाता है) जब तक बदलने का मज़बूत कारण न हो।

त्वरित सेटअप:
```bash
cp config/config.json.example config/config.json
```
`model_family`, `model_flavour`, `output_dir`, `data_backend_config`, और `eval_dataset_id` भरें। ऊपर दिए डिफ़ॉल्ट्स को तब तक रखें जब तक आपको कुछ अलग न चाहिए।

CUDA attention विकल्प:
- CUDA पर, LongCat‑Video उपलब्ध होने पर bundled block‑sparse Triton kernel को प्राथमिकता देता है और अन्यथा standard dispatcher पर fallback करता है। कोई मैनुअल टॉगल आवश्यक नहीं।
- यदि आप xFormers मजबूरी से चाहते हैं, तो config/CLI में `attention_implementation: "xformers"` सेट करें।

---

## 3) Dataloader: क्लिप्स को start frames के साथ जोड़ें

- दो datasets बनाएँ:
  - **Clips**: target videos + captions (edit निर्देश)। इन्हें `is_i2v: true` मार्क करें और `conditioning_data` को start‑frame dataset ID पर सेट करें।
  - **Start frames**: प्रति क्लिप एक इमेज, वही फ़ाइलनाम, कोई कैप्शन नहीं।
- दोनों को 64px ग्रिड पर रखें (जैसे 480x832)। height/width 16 से विभाज्य होना चाहिए। frame counts को `(frames - 1) % 4 == 0` पूरा करना चाहिए; 93 पहले से मान्य है।
- clips और start frames के लिए अलग VAE caches रखें।

उदाहरण `multidatabackend.json`:
```jsonc
[
  {
    "id": "longcat-video-train",
    "type": "local",
    "dataset_type": "video",
    "is_i2v": true,
    "instance_data_dir": "/data/video-clips",
    "caption_strategy": "textfile",
    "resolution": 480,
    "cache_dir_vae": "/cache/vae/longcat/video",
    "conditioning_data": ["longcat-video-cond"]
  },
  {
    "id": "longcat-video-cond",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/data/video-start-frames",
    "caption_strategy": null,
    "resolution": 480,
    "cache_dir_vae": "/cache/vae/longcat/video-cond"
  }
]
```

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

---

## 4) वैलिडेशन विशिष्टताएँ

- प्रशिक्षण जैसी paired संरचना वाला एक छोटा validation split जोड़ें। `validation_using_datasets: true` सेट करें और `eval_dataset_id` को उस split पर पॉइंट करें (जैसे `longcat-video-val`), ताकि validation start frame स्वतः लाए।
- WebUI previews: `simpletuner server` शुरू करें, LongCat‑Video edit चुनें, और start frame + prompt अपलोड करें।
- Guidance: 3.5–5.0 अच्छा है; CFG ऑन होने पर खाली नेगेटिव अपने‑आप भरते हैं।
- कम VRAM previews/ट्रेनिंग के लिए `musubi_blocks_to_swap` (4–8 से शुरू करें) और वैकल्पिक रूप से `musubi_block_swap_device` सेट करें ताकि forward/backward के दौरान आख़िरी transformer blocks CPU से स्ट्रीम हों। इससे throughput कुछ घटेगा लेकिन VRAM peak कम होंगे।
- Sampling के दौरान conditioning frame स्थिर रहता है; केवल बाद वाले फ्रेम denoise होते हैं।

---

## 5) प्रशिक्षण शुरू (CLI)

कॉन्फ़िग और dataloader सेट होने के बाद:
```bash
simpletuner train --config config/config.json
```
सुनिश्चित करें कि training डेटा में conditioning frames मौजूद हों, ताकि पाइपलाइन conditioning latents बना सके।

---

## 6) समस्या समाधान

- **Missing conditioning image**: `conditioning_data` से conditioning dataset जोड़ें और फ़ाइलनाम मैच कराएँ; `eval_dataset_id` को validation split ID पर सेट करें।
- **Height/width errors**: आयाम 16 से विभाज्य रखें और 64px ग्रिड पर रखें।
- **First frame drifts**: guidance 3.5–4.0 करें या steps घटाएँ।
- **OOM**: validation resolution/frames घटाएँ, `lora_rank` कम करें, group offload सक्षम करें, या `int8-quanto`/`fp8-torchao` उपयोग करें।
