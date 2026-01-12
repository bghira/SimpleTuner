# Qwen Image Edit क्विकस्टार्ट

यह गाइड Qwen Image के **edit** flavours को कवर करती है जिन्हें SimpleTuner सपोर्ट करता है:

- `edit-v1` – प्रति प्रशिक्षण उदाहरण एक reference इमेज। reference इमेज को Qwen2.5‑VL टेक्स्ट एन्कोडर से encode किया जाता है और **conditioning image embeds** के रूप में कैश किया जाता है।
- `edit-v2` (“edit plus”) – प्रति सैंपल अधिकतम तीन reference इमेज, जिन्हें on‑the‑fly VAE latents में encode किया जाता है।

दोनों variants बेस [Qwen Image quickstart](./QWEN_IMAGE.md) का अधिकांश हिस्सा inherit करते हैं; नीचे के सेक्शन बताते हैं कि edit checkpoints को fine‑tune करते समय क्या *अलग* है।

---

## 1. हार्डवेयर चेकलिस्ट

बेस मॉडल अभी भी **20 B पैरामीटर** है:

| आवश्यकता | अनुशंसा |
|-------------|----------------|
| GPU VRAM    | 24 G न्यूनतम (int8/nf4 quantisation के साथ) • 40 G+ मजबूत रूप से अनुशंसित |
| Precision   | `mixed_precision=bf16`, `base_model_precision=int8-quanto` (या `nf4-bnb`) |
| Batch size  | `train_batch_size=1` रखना आवश्यक; effective batch के लिए gradient accumulation उपयोग करें |

[Qwen Image guide](./QWEN_IMAGE.md) से अन्य सभी प्रशिक्षण पूर्वापेक्षाएँ लागू रहती हैं (Python ≥ 3.10, CUDA 12.x image, आदि)।

---

## 2. कॉन्फ़िगरेशन हाइलाइट्स

`config/config.json` के अंदर:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```jsonc
{
  "model_type": "lora",
  "model_family": "qwen_image",
  "model_flavour": "edit-v1",      // or "edit-v2"
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.0,
  "validation_num_inference_steps": 30,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "base_model_precision": "int8-quanto",
  "quantize_via": "cpu",
  "quantize_activations": false,
  "flow_schedule_shift": 1.73,
  "data_backend_config": "config/qwen_edit/multidatabackend.json"
}
```
</details>

- EMA डिफ़ॉल्ट रूप से CPU पर चलता है और इसे सक्षम छोड़ना सुरक्षित है जब तक आपको तेज़ checkpoints की जरूरत न हो।
- `validation_resolution` को 24 G कार्ड्स पर घटाना चाहिए (जैसे `768x768`)।
- `edit-v2` में यदि आप चाहते हैं कि control images target resolution inherit करें (डिफ़ॉल्ट 1 MP packing के बजाय), तो `model_kwargs` के तहत `match_target_res` जोड़ें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```jsonc
"model_kwargs": {
  "match_target_res": true
}
```
</details>

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

---

</details>

## 3. Dataloader लेआउट

दोनों flavours **paired datasets** अपेक्षित करते हैं: एक edit image, वैकल्पिक edit caption, और एक या अधिक control/reference images जिनके **फ़ाइलनाम बिल्कुल समान** हों।

फील्ड विवरण के लिए [`conditioning_type`](../DATALOADER.md#conditioning_type) और [`conditioning_data`](../DATALOADER.md#conditioning_data) देखें। यदि आप multiple conditioning datasets देते हैं, तो [OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling) में `conditioning_multidataset_sampling` से उनके sampling का तरीका चुनें।

### 3.1 edit‑v1 (single control image)

मुख्य dataset को एक conditioning dataset **और** एक conditioning‑image‑embed cache का reference देना चाहिए:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```jsonc
[
  {
    "id": "qwen-edit-images",
    "type": "local",
    "instance_data_dir": "/datasets/qwen-edit/images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": ["qwen-edit-reference"],
    "conditioning_image_embeds": "qwen-edit-ref-embeds",
    "cache_dir_vae": "cache/vae/qwen-edit-images"
  },
  {
    "id": "qwen-edit-reference",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit/reference",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-reference"
  },
  {
    "id": "qwen-edit-ref-embeds",
    "type": "local",
    "dataset_type": "conditioning_image_embeds",
    "cache_dir": "cache/conditioning_image_embeds/qwen-edit"
  }
]
```
</details>

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

- `conditioning_type=reference_strict` यह सुनिश्चित करता है कि crops edit image से मैच करें। `reference_loose` का उपयोग तभी करें जब reference का aspect mismatch हो सकता है।
- `conditioning_image_embeds` entry हर reference के लिए बने Qwen2.5‑VL visual tokens स्टोर करती है। यदि इसे छोड़ा जाए, तो SimpleTuner `cache/conditioning_image_embeds/<dataset_id>` के तहत default cache बना देगा।

### 3.2 edit‑v2 (multi‑control)

`edit-v2` के लिए, हर control dataset को `conditioning_data` में सूचीबद्ध करें। प्रत्येक entry एक अतिरिक्त control frame देती है। आपको conditioning‑image‑embed cache की जरूरत नहीं है क्योंकि latents on‑the‑fly निकाले जाते हैं।

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```jsonc
[
  {
    "id": "qwen-edit-plus-images",
    "type": "local",
    "instance_data_dir": "/datasets/qwen-edit-plus/images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": [
      "qwen-edit-plus-reference-a",
      "qwen-edit-plus-reference-b",
      "qwen-edit-plus-reference-c"
    ],
    "cache_dir_vae": "cache/vae/qwen-edit-plus/images"
  },
  {
    "id": "qwen-edit-plus-reference-a",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit-plus/reference_a",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-plus/ref_a"
  },
  {
    "id": "qwen-edit-plus-reference-b",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit-plus/reference_b",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-plus/ref_b"
  },
  {
    "id": "qwen-edit-plus-reference-c",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit-plus/reference_c",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-plus/ref_c"
  }
]
```
</details>

जितनी reference images हों (1–3), उतने control datasets रखें। SimpleTuner फ़ाइलनाम मैच करके प्रति सैंपल उन्हें aligned रखता है।

---

## 4. ट्रेनर चलाना

सबसे तेज़ smoke test किसी example preset को चलाना है:

```bash
simpletuner train example=qwen_image.edit-v1-lora
# or
simpletuner train example=qwen_image.edit-v2-lora
```

जब मैन्युअली लॉन्च करें:

```bash
simpletuner train \
  --config config/config.json \
  --data config/qwen_edit/multidatabackend.json
```

### टिप्स

- `caption_dropout_probability` को `0.0` रखें जब तक आपके पास edit instruction के बिना ट्रेन करने का कारण न हो।
- लंबी training jobs के लिए validation cadence (`validation_step_interval`) कम करें ताकि महंगी edit validations runtime पर हावी न हों।
- Qwen edit checkpoints guidance head के बिना आते हैं; `validation_guidance` आमतौर पर **3.5–4.5** रेंज में रहता है।

---

## 5. वैलिडेशन प्रीव्यूज़

यदि आप validation आउटपुट के साथ reference image भी देखना चाहते हैं, तो validation edit/reference pairs को एक dedicated dataset में रखें (training split जैसी ही लेआउट) और सेट करें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```jsonc
{
  "eval_dataset_id": "qwen-edit-val"
}
```
</details>

SimpleTuner validation के दौरान उस dataset की conditioning images को पुन: उपयोग करेगा।

---

### समस्या समाधान

- **`ValueError: Control tensor list length does not match batch size`** – सुनिश्चित करें कि हर conditioning dataset में *सभी* edit images के लिए फ़ाइलें हों। खाली फ़ोल्डर या mismatched filenames यह त्रुटि ट्रिगर करते हैं।
- **Validation के दौरान OOM** – `validation_resolution`, `validation_num_inference_steps` घटाएँ, या आगे quantise करें (`base_model_precision=int2-quanto`) और फिर दोबारा प्रयास करें।
- **`edit-v1` में cache not found errors** – जाँचें कि मुख्य dataset का `conditioning_image_embeds` फ़ील्ड किसी मौजूदा cache dataset entry से मेल खाता है।

---

अब आप base Qwen Image quickstart को edit training के लिए अनुकूल बनाने के लिए तैयार हैं। full configuration options (text encoder caching, multi‑backend sampling, आदि) के लिए [FLUX_KONTEXT.md](./FLUX_KONTEXT.md) की guidance दोबारा उपयोग करें — dataset pairing workflow वही है, केवल model family `qwen_image` हो जाता है।
