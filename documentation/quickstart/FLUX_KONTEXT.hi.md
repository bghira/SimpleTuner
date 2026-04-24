# Kontext [dev] Mini Quick‑start

> 📝  Kontext अपनी ट्रेनिंग workflow का 90 % हिस्सा Flux के साथ साझा करता है, इसलिए यह फ़ाइल केवल वे बातें सूचीबद्ध करती है जो *अलग* हैं। जब कोई स्टेप यहाँ **नहीं** दिया है, तो मूल [निर्देश](../quickstart/FLUX.md) फॉलो करें।


---

## 1. मॉडल ओवरव्यू

|                                                  | Flux‑dev               | Kontext‑dev                                 |
| ------------------------------------------------ | -------------------    | ------------------------------------------- |
| License                                          | Non‑commercial         | Non‑commercial                              |
| Guidance                                         | Distilled (CFG ≈ 1)     | Distilled (CFG ≈ 1)                         |
| Variants available                               | *dev*, schnell,[pro]    | *dev*, [pro, max]                           |
| T5 sequence length                               | 512 dev, 256 schnell   | 512 dev                                     |
| Typical 1024 px inference time<br>(4090 @ CFG 1)  | ≈ 20 s                  | **≈ 80 s**                                  |
| VRAM for 1024 px LoRA @ int8‑quanto               | 18 G                   | **24 G**                                    |

Kontext Flux transformer backbone को बनाए रखता है, लेकिन **paired‑reference conditioning** जोड़ता है।

Kontext के लिए दो `conditioning_type` मोड उपलब्ध हैं:

* `conditioning_type=reference_loose` (✅ stable) – reference का aspect‑ratio/size edit से अलग हो सकता है।
  - दोनों datasets का metadata स्कैन होता है, aspect bucketing और crop अलग‑अलग होते हैं, जिससे startup समय बढ़ सकता है।
  - यदि आप edit और reference इमेजेस का alignment सुनिश्चित करना चाहते हैं (जैसे एक ही फ़ाइल‑नाम से लोड करने वाला dataloader), तो यह समस्या हो सकती है।
* `conditioning_type=reference_strict` (✅ stable) – reference को edit crop की तरह ही pre‑transform किया जाता है।
  - जब edit और reference crops/aspect bucketing के बीच perfect alignment चाहिए, तब यही कॉन्फ़िगरेशन उपयोग करें।
  - पहले `--vae_cache_ondemand` और कुछ अतिरिक्त VRAM की जरूरत होती थी, अब नहीं।
  - startup पर source dataset से crop/aspect bucket metadata कॉपी करता है, इसलिए आपको यह काम नहीं करना पड़ता।

Field definitions के लिए [`conditioning_type`](../DATALOADER.md#conditioning_type) और [`conditioning_data`](../DATALOADER.md#conditioning_data) देखें। कई conditioning sets को कैसे sample करना है इसके लिए [OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling) में `conditioning_multidataset_sampling` देखें।


---

## 2. हार्डवेयर आवश्यकताएँ

* **System RAM**: quantisation के लिए अभी भी 50 GB चाहिए।
* **GPU**: 1024 px ट्रेनिंग के लिए **int8‑quanto** के साथ 3090 (24 G) वास्तविक न्यूनतम है।
  * Flash Attention 3 वाले Hopper H100/H200 सिस्टम `--fuse_qkv_projections` सक्षम कर सकते हैं, जिससे ट्रेनिंग काफी तेज़ होगी।
  * यदि आप 512 px पर ट्रेन करते हैं तो 12 G कार्ड में संभव है, लेकिन batch धीमे होंगे (sequence length बड़ा रहता है)।


---

## 3. Quick configuration diff

नीचे `config/config.json` में आपकी सामान्य Flux ट्रेनिंग कॉन्फ़िग की तुलना में आवश्यक *न्यूनतम* बदलाव दिए गए हैं।

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```jsonc
{
  "model_family":   "flux",
  "model_flavour": "kontext",                       // <‑‑ इसे "dev" से बदलकर "kontext" करें
  "base_model_precision": "int8-quanto",            // 1024 px पर 24 G में फिट होता है
  "gradient_checkpointing": true,
  "fuse_qkv_projections": false,                    // <‑‑ Hopper H100/H200 पर ट्रेनिंग तेज़ करने के लिए। चेतावनी: flash-attn मैन्युअली इंस्टॉल होना चाहिए।
  "lora_rank": 16,
  "learning_rate": 1e-5,
  "optimizer": "optimi-lion",                       // <‑‑ तेज़ परिणाम के लिए Lion; धीमे लेकिन संभवतः अधिक स्थिर परिणामों के लिए adamw_bf16।
  "max_train_steps": 10000,
  "validation_guidance": 2.5,                       // <‑‑ kontext वास्तव में 2.5 guidance पर सबसे अच्छा चलता है
  "validation_resolution": "1024x1024",
  "conditioning_multidataset_sampling": "random"    // <-- यदि दो conditioning datasets हैं तो "combined" सेट करने पर वे एक साथ दिखेंगे, switching की जगह।
}
```
</details>

### उन्नत प्रयोगात्मक फीचर्स

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में ऐसे प्रयोगात्मक फीचर्स शामिल हैं जो ट्रेनिंग स्थिरता और परफॉर्मेंस को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बेहतर करता है क्योंकि ट्रेनिंग के दौरान मॉडल अपने इनपुट्स स्वयं जनरेट करता है।

> ⚠️ ये फीचर्स ट्रेनिंग का कम्प्यूटेशनल ओवरहेड बढ़ाते हैं।

</details>

### Dataloader snippet (multi‑data‑backend)

यदि आपने manually curated image-pair dataset बनाया है, तो आप इसे दो अलग directories में कॉन्फ़िगर कर सकते हैं: एक edit images के लिए और दूसरी reference images के लिए।

Edit dataset में `conditioning_data` field को reference dataset की `id` की ओर पॉइंट करना चाहिए।

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```jsonc
[
  {
    "id": "my-edited-images",
    "type": "local",
    "cache_dir_vae": "/cache/vae/flux/kontext/edited-images",   // <-- VAE outputs यहाँ स्टोर होते हैं
    "instance_data_dir": "/datasets/edited-images",             // <-- absolute paths उपयोग करें
    "conditioning_data": [
      "my-reference-images"                                     // <‑‑ reference set की "id" यहाँ दें
                                                                // आप दूसरी सेट भी दे सकते हैं ताकि alternate या combine हो सकें, जैसे ["reference-images", "reference-images2"]
    ],
    "resolution": 1024,
    "caption_strategy": "textfile"                              // <-- इन captions में edit instructions होने चाहिए
  },
  {
    "id": "my-reference-images",
    "type": "local",
    "cache_dir_vae": "/cache/vae/flux/kontext/ref-images",      // <-- VAE outputs यहाँ स्टोर होते हैं; अन्य dataset VAE paths से अलग होना चाहिए।
    "instance_data_dir": "/datasets/reference-images",          // <-- absolute paths उपयोग करें
    "conditioning_type": "reference_strict",                    // <‑‑ reference_loose पर सेट करने पर images edit images से स्वतंत्र रूप से crop होंगी
    "resolution": 1024,
    "caption_strategy": null,                                   // <‑‑ references के लिए captions जरूरी नहीं; यदि उपलब्ध हैं तो edit captions की जगह उपयोग होंगे
                                                                // NOTE: conditioning_multidataset_sampling=combined के साथ अलग conditioning captions परिभाषित नहीं कर सकते।
                                                                // केवल edit datasets के captions उपयोग होंगे।
  }
]
```
</details>

> caption_strategy विकल्पों और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

*हर edit इमेज के लिए दोनों dataset folders में 1‑to‑1 matching फ़ाइल नाम और extension होना **अनिवार्य** है। SimpleTuner reference embedding को edit की conditioning में स्वतः जोड़ देता है।

एक तैयार उदाहरण [Kontext Max derived demo dataset](https://huggingface.co/datasets/terminusresearch/KontextMax-Edit-smol) उपलब्ध है जिसमें reference और edit images के साथ caption textfiles शामिल हैं, जिससे सेटअप समझने में मदद मिलेगी।

### Dedicated validation split सेट करना

यहाँ एक उदाहरण कॉन्फ़िगरेशन है जिसमें 200,000 samples का training set और कुछ samples का validation set है।
Kontext के लिए `validation_using_datasets` को unset या `false` ही रखें। Validation पहले से paired reference dataset को अपने आप use करती है; `eval_dataset_id` केवल यह चुनता है कि validation samples किस edit dataset से आएँगे।

अपने `config.json` में यह जोड़ें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "eval_dataset_id": "edited-images",
}
```
</details>

अपने `multidatabackend.json` में, `edited-images` और `reference-images` में validation data होना चाहिए और उनका लेआउट सामान्य training split जैसा होना चाहिए।

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
    {
        "id": "edited-images",
        "disabled": false,
        "type": "local",
        "instance_data_dir": "/datasets/edit/edited-images",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": "textfile",
        "cache_dir_vae": "cache/vae/flux-edit",
        "vae_cache_clear_each_epoch": false,
        "conditioning_data": ["reference-images"]
    },
    {
        "id": "reference-images",
        "disabled": false,
        "type": "local",
        "instance_data_dir": "/datasets/edit/reference-images",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": null,
        "cache_dir_vae": "cache/vae/flux-ref",
        "vae_cache_clear_each_epoch": false,
        "conditioning_type": "reference_strict"
    },
    {
        "id": "subjects200k-left",
        "disabled": false,
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "conditioning_data": ["subjects200k-right"],
        "huggingface": {
            "caption_column": "description.description_0",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 0
            }
        }
    },
    {
        "id": "subjects200k-right",
        "disabled": false,
        "type": "huggingface",
        "dataset_type": "conditioning",
        "conditioning_type": "reference_strict",
        "source_dataset_id": "subjects200k-left",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "huggingface": {
            "caption_column": "description.description_1",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 1
            }
        }
    },

    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/flux"
    }
]
```
</details>

### Automatic Reference-Edit Pair Generation

यदि आपके पास pre-existing reference-edit pairs नहीं हैं, तो SimpleTuner उन्हें एक single dataset से अपने आप जेनरेट कर सकता है। यह विशेष रूप से इन मॉडलों के लिए उपयोगी है:
- Image enhancement / super-resolution
- JPEG artifact removal
- Deblurring
- अन्य restoration tasks

#### उदाहरण: Deblurring Training Dataset

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```jsonc
[
  {
    "id": "high-quality-images",
    "type": "local",
    "instance_data_dir": "/path/to/sharp-images",
    "resolution": 1024,
    "caption_strategy": "textfile",
    "conditioning": [
      {
        "type": "superresolution",
        "blur_radius": 3.0,
        "blur_type": "gaussian",
        "add_noise": true,
        "noise_level": 0.02,
        "captions": ["enhance sharpness", "deblur", "increase clarity", "sharpen image"]
      }
    ]
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/kontext"
  }
]
```
</details>

यह कॉन्फ़िगरेशन:
1. आपके high-quality sharp images से blurred versions बनाता है (ये "reference" images बनते हैं)
2. मूल high-quality images को training loss target के रूप में उपयोग करता है
3. Kontext को poor-quality reference इमेज को enhance/deblur करना सिखाता है

> **नोट**: `conditioning_multidataset_sampling=combined` का उपयोग करते समय conditioning dataset पर `captions` परिभाषित नहीं कर सकते। इसके बजाय edit dataset के captions उपयोग होंगे।

#### उदाहरण: JPEG Artifact Removal

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```jsonc
[
  {
    "id": "pristine-images",
    "type": "local",
    "instance_data_dir": "/path/to/pristine-images",
    "resolution": 1024,
    "caption_strategy": "textfile",
    "conditioning": [
      {
        "type": "jpeg_artifacts",
        "quality_mode": "range",
        "quality_range": [10, 30],
        "compression_rounds": 2,
        "captions": ["remove compression artifacts", "restore quality", "fix jpeg artifacts"]
      }
    ]
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/kontext"
  }
]
```
</details>

#### महत्वपूर्ण नोट्स

1. **Generation startup पर होता है**: degraded versions ट्रेनिंग शुरू होते ही स्वतः बनते हैं
2. **Caching**: जेनरेटेड इमेजेस सेव होती हैं, इसलिए बाद की रन में दोबारा नहीं बनेंगी
3. **Caption strategy**: conditioning config में `captions` field task-specific prompts देता है जो generic image descriptions से बेहतर काम करते हैं
4. **Performance**: ये CPU-based generators (blur, JPEG) तेज़ हैं और multi-process उपयोग करते हैं
5. **Disk space**: जेनरेटेड इमेजेस बड़े हो सकते हैं, इसलिए पर्याप्त डिस्क स्पेस रखें! दुर्भाग्य से अभी on-demand बनाने का विकल्प नहीं है।

और अधिक conditioning types तथा advanced कॉन्फ़िगरेशन के लिए [ControlNet डॉक्यूमेंटेशन](../CONTROLNET.md) देखें।

---

## 4. Kontext के लिए विशेष ट्रेनिंग टिप्स

1. **लंबी sequences → धीमे steps.**  एक single 4090 पर 1024 px, rank‑1 LoRA, bf16 + int8 में ~0.4 it/s की उम्मीद करें।
2. **सही सेटिंग्स के लिए एक्सप्लोर करें।**  Kontext का fine-tuning अभी बहुत ज्ञात नहीं है; सुरक्षित रूप से `1e‑5` (Lion) या `5e‑4` (AdamW) पर रहें।
3. **VAE caching के दौरान VRAM spikes देखें।**  यदि OOM हो, `--offload_during_startup=true` जोड़ें, `resolution` कम करें, या `config.json` में VAE tiling सक्षम करें।
4. **आप reference images के बिना भी ट्रेन कर सकते हैं, लेकिन अभी SimpleTuner में नहीं।**  अभी चीज़ें कुछ हद तक conditional images की आवश्यकता पर hardcoded हैं, लेकिन आप सामान्य datasets को edit pairs के साथ जोड़ सकते हैं ताकि subjects और likeness सीख सके।
5. **Guidance re‑distillation.**  Flux‑dev की तरह, Kontext‑dev CFG‑distilled है; यदि आपको diversity चाहिए, तो `validation_guidance_real > 1` के साथ retrain करें और inference में Adaptive‑Guidance node उपयोग करें। यह बहुत अधिक समय लेगा और सफल होने के लिए बड़े rank LoRA या Lycoris LoKr चाहिए होंगे।
6. **Full-rank training शायद समय की बर्बादी है।**  Kontext low rank पर ट्रेन होने के लिए बनाया गया है; full rank training संभवतः Lycoris LoKr से बेहतर परिणाम नहीं देगा, जो आम तौर पर Standard LoRA से बेहतर और कम प्रयास में होता है। फिर भी यदि आप आज़माना चाहते हैं, तो DeepSpeed उपयोग करना होगा।
7. **आप ट्रेनिंग के लिए दो या अधिक reference images उपयोग कर सकते हैं।**  उदाहरण के तौर पर, यदि आपके पास subject-subject-scene इमेजेस हैं, तो आप सभी संबंधित इमेजेस को reference inputs के रूप में दे सकते हैं। बस सुनिश्चित करें कि फ़ाइल नाम सभी फ़ोल्डर्स में match हों।

---

## 5. Inference gotchas

- ट्रेनिंग और inference precision levels को मैच करें; int8 ट्रेनिंग int8 inference के साथ सबसे अच्छा काम करेगी, आदि।
- दो इमेजेस एक साथ सिस्टम से गुजरती हैं इसलिए यह बहुत धीमा होगा।  4090 पर 1024 px edit के लिए 80 s की उम्मीद करें।

---

## 6. Troubleshooting cheat‑sheet

| Symptom                                 | Likely cause               | Quick fix                                              |
| --------------------------------------- | -------------------------- | ------------------------------------------------------ |
| quantisation के दौरान OOM                 | पर्याप्त **system** RAM नहीं  | `quantize_via=cpu` उपयोग करें                                 |
| Ref image ignored / no edit applied     | Dataloader mis‑pairing     | identical filenames और `conditioning_data` field सुनिश्चित करें |
| Square grid artifacts                   | Low‑quality edits dominate | उच्च गुणवत्ता dataset बनाएं, LR कम करें, Lion से बचें      |

---

## 7. आगे पढ़ें

उन्नत tuning विकल्पों (LoKr, NF4 quant, DeepSpeed, आदि) के लिए [Flux quickstart](../quickstart/FLUX.md) देखें – ऊपर अलग से बताया न गया हो तो हर flag समान है।
