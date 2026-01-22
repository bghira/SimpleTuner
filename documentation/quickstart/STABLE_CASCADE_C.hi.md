# Stable Cascade Stage C क्विकस्टार्ट

यह गाइड SimpleTuner को **Stable Cascade Stage C prior** को fine‑tune करने के लिए कॉन्फ़िगर करने की प्रक्रिया बताती है। Stage C वह text‑to‑image prior सीखता है जो Stage B/C decoder stack को फ़ीड करता है, इसलिए यहाँ अच्छी training hygiene सीधे downstream decoder आउटपुट्स को बेहतर बनाती है। हम LoRA प्रशिक्षण पर फोकस करेंगे, लेकिन यदि आपके पास पर्याप्त VRAM है तो वही चरण full fine‑tune पर भी लागू होते हैं।

> **Heads‑up:** Stage C 1B+ पैरामीटर वाले CLIP‑G/14 टेक्स्ट एन्कोडर और EfficientNet‑आधारित autoencoder का उपयोग करता है। सुनिश्चित करें कि torchvision इंस्टॉल है और बड़े text‑embed caches के लिए तैयार रहें (SDXL की तुलना में प्रति prompt ~5–6× बड़ा)।

## हार्डवेयर आवश्यकताएँ

- **LoRA प्रशिक्षण:** 20–24 GB VRAM (RTX 3090/4090, A6000, आदि)
- **फुल‑मॉडल प्रशिक्षण:** 48 GB+ VRAM अनुशंसित (A6000, A100, H100). DeepSpeed/FSDP2 offload आवश्यकता कम कर सकता है लेकिन जटिलता बढ़ाता है।
- **सिस्टम RAM:** 32 GB अनुशंसित ताकि CLIP‑G टेक्स्ट एन्कोडर और कैशिंग threads starving न हों।
- **डिस्क:** prompt‑cache फ़ाइलों के लिए कम से कम ~50 GB रखें। Stage C CLIP‑G embeddings ~4–6 MB प्रति आइटम होते हैं।

## पूर्वापेक्षाएँ

1. Python 3.13 (प्रोजेक्ट `.venv` से मेल)।
2. GPU acceleration के लिए CUDA 12.1+ या ROCm 5.7+ (या Apple Metal for M‑series Macs, हालांकि Stage C मुख्यतः CUDA पर टेस्ट हुआ है)।
3. `torchvision` (Stable Cascade autoencoder के लिए आवश्यक) और training launch के लिए `accelerate`.

Python संस्करण जांचें:

```bash
python --version
```

Missing पैकेज इंस्टॉल करें (Ubuntu उदाहरण):

```bash
sudo apt update && sudo apt install -y python3.13 python3.13-venv
```

## इंस्टॉलेशन

मानक SimpleTuner इंस्टॉलेशन (pip या source) का पालन करें। एक सामान्य CUDA वर्कस्टेशन के लिए:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

Contributors या जो भी सीधे repo पर काम कर रहे हैं, source से इंस्टॉल करें और फिर `pip install -e .[cuda,dev]` चलाएँ।

## वातावरण सेटअप

### 1. बेस कॉन्फ़िग कॉपी करें

```bash
cp config/config.json.example config/config.json
```

निम्न keys सेट करें (दिखाए गए मान Stage C के लिए अच्छा baseline हैं):

| Key | अनुशंसा | नोट्स |
| --- | ------- | ----- |
| `model_family` | `"stable_cascade"` | Stage C components लोड करने के लिए आवश्यक |
| `model_flavour` | `"stage-c"` (या `"stage-c-lite"`) | lite flavour पैरामीटर कम करता है यदि आपके पास केवल ~18 GB VRAM है |
| `model_type` | `"lora"` | Full fine‑tune संभव है लेकिन काफी अधिक मेमोरी चाहिए |
| `mixed_precision` | `"no"` | Stage C mixed precision में नहीं चलेगा जब तक `i_know_what_i_am_doing=true` न सेट करें; fp32 सुरक्षित विकल्प है |
| `gradient_checkpointing` | `true` | 3–4 GB VRAM बचाता है |
| `vae_batch_size` | `1` | Stage C autoencoder भारी है; छोटा रखें |
| `validation_resolution` | `"1024x1024"` | downstream decoder अपेक्षाओं से मेल खाता है |
| `stable_cascade_use_decoder_for_validation` | `true` | validation में संयुक्त prior+decoder pipeline उपयोग करता है |
| `stable_cascade_decoder_model_name_or_path` | `"stabilityai/stable-cascade"` | यदि आपके पास custom Stage B/C decoder है तो local path दें |
| `stable_cascade_validation_prior_num_inference_steps` | `20` | prior denoising steps |
| `stable_cascade_validation_prior_guidance_scale` | `3.0–4.0` | prior पर CFG |
| `stable_cascade_validation_decoder_guidance_scale` | `0.0–0.5` | decoder CFG (0.0 photorealistic, >0.0 अधिक prompt adherence) |

#### उदाहरण `config/config.json`

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "base_model_precision": "int8-torchao",
  "checkpoint_step_interval": 100,
  "data_backend_config": "config/stable_cascade/multidatabackend.json",
  "gradient_accumulation_steps": 2,
  "gradient_checkpointing": true,
  "hub_model_id": "stable-cascade-stage-c-lora",
  "learning_rate": 1e-4,
  "lora_alpha": 16,
  "lora_rank": 16,
  "lora_type": "standard",
  "lr_scheduler": "cosine",
  "max_train_steps": 30000,
  "mixed_precision": "no",
  "model_family": "stable_cascade",
  "model_flavour": "stage-c",
  "model_type": "lora",
  "optimizer": "adamw_bf16",
  "output_dir": "output/stable_cascade_stage_c",
  "report_to": "wandb",
  "seed": 42,
  "stable_cascade_decoder_model_name_or_path": "stabilityai/stable-cascade",
  "stable_cascade_decoder_subfolder": "decoder_lite",
  "stable_cascade_use_decoder_for_validation": true,
  "stable_cascade_validation_decoder_guidance_scale": 0.0,
  "stable_cascade_validation_prior_guidance_scale": 3.5,
  "stable_cascade_validation_prior_num_inference_steps": 20,
  "train_batch_size": 4,
  "use_ema": true,
  "vae_batch_size": 1,
  "validation_guidance": 4.0,
  "validation_negative_prompt": "ugly, blurry, low-res",
  "validation_num_inference_steps": 30,
  "validation_prompt": "a cinematic photo of a shiba inu astronaut",
  "validation_resolution": "1024x1024"
}
```
</details>

मुख्य निष्कर्ष:

- `model_flavour` `stage-c` और `stage-c-lite` स्वीकार करता है। यदि VRAM कम है या distilled prior चाहिए तो lite का उपयोग करें।
- `mixed_precision` को `"no"` पर रखें। यदि आप override करते हैं, तो `i_know_what_i_am_doing=true` सेट करें और NaNs के लिए तैयार रहें।
- `stable_cascade_use_decoder_for_validation` सक्षम करने पर prior आउटपुट Stage B/C decoder में जाता है ताकि validation gallery में prior latents की बजाय वास्तविक इमेज दिखें।

### 2. डेटा बैकएंड कॉन्फ़िगर करें

`config/stable_cascade/multidatabackend.json` बनाएँ:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "primary",
    "type": "local",
    "dataset_type": "images",
    "instance_data_dir": "/data/stable-cascade",
    "resolution": "1024x1024",
    "bucket_resolutions": ["1024x1024", "896x1152", "1152x896"],
    "crop": true,
    "crop_style": "random",
    "minimum_image_size": 768,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "caption_strategy": "filename",
    "prepend_instance_prompt": false,
    "repeats": 1
  },
  {
    "id": "stable-cascade-text-cache",
    "type": "local",
    "dataset_type": "text_embeds",
    "cache_dir": "/data/cache/stable-cascade/text",
    "default": true
  }
]
```
</details>

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

टिप्स:

- Stage C latents autoencoder से आते हैं, इसलिए 1024×1024 (या portrait/landscape का सीमित रेंज) पर टिके रहें। decoder 1024px इनपुट से ~24×24 latent grids अपेक्षित करता है।
- `target_downsample_size` को 1024 पर रखें ताकि narrow crops ~2:1 से अधिक आस्पेक्ट रेशियो में न जाएँ।
- एक dedicated text‑embed cache हमेशा कॉन्फ़िगर करें। इसके बिना, हर रन CLIP‑G के साथ captions re‑embed करने में 30–60 मिनट खर्च करेगा।

### 3. Prompt library (वैकल्पिक)

`config/stable_cascade/prompt_library.json` बनाएँ:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "portrait": "a cinematic portrait photograph lit by studio strobes",
  "landscape": "a sweeping ultra wide landscape with volumetric lighting",
  "product": "a product render on a seamless background, dramatic reflections",
  "stylized": "digital illustration in the style of a retro sci-fi book cover"
}
```
</details>

इसे सक्षम करने के लिए अपने कॉन्फ़िग में `"validation_prompt_library": "config/stable_cascade/prompt_library.json"` जोड़ें।

## प्रशिक्षण

1. अपना वातावरण सक्रिय करें और यदि पहले नहीं किया है तो Accelerate कॉन्फ़िगरेशन लॉन्च करें:

```bash
source .venv/bin/activate
accelerate config
```

2. प्रशिक्षण शुरू करें:

```bash
accelerate launch simpletuner/train.py \
  --config_file config/config.json \
  --data_backend_config config/stable_cascade/multidatabackend.json
```

पहले epoch के दौरान, यह मॉनिटर करें:

- **Text cache throughput** – Stage C कैश प्रगति लॉग करेगा। high‑end GPUs पर ~8–12 prompts/sec की अपेक्षा करें।
- **VRAM usage** – validation रन के दौरान OOM से बचने के लिए <95% utilization लक्षित रखें।
- **Validation outputs** – संयुक्त pipeline `output/<run>/validation/` में full‑resolution PNGs निकालेगा।

## वैलिडेशन और इंफ़रेंस नोट्स

- Stage C prior अकेले केवल image embeddings बनाता है। `stable_cascade_use_decoder_for_validation=true` होने पर SimpleTuner validation wrapper उन्हें अपने‑आप decoder से गुज़ारता है।
- decoder flavour बदलने के लिए `stable_cascade_decoder_subfolder` को `"decoder"`, `"decoder_lite"`, या Stage B/C weights वाले किसी custom फ़ोल्डर पर सेट करें।
- तेज़ previews के लिए `stable_cascade_validation_prior_num_inference_steps` को ~12 और `validation_num_inference_steps` को 20 तक घटाएँ। संतुष्ट होने पर गुणवत्ता के लिए इन्हें बढ़ाएँ।

## उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।
*   **[Diff2Flow](../experimental/DIFF2FLOW.md):** Stable Cascade को Flow Matching objective के साथ प्रशिक्षित करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

</details>

## समस्या समाधान

| लक्षण | समाधान |
| --- | --- |
| "Stable Cascade Stage C requires --mixed_precision=no" | `"mixed_precision": "no"` सेट करें या `"i_know_what_i_am_doing": true` जोड़ें (अनुशंसित नहीं) |
| Validation केवल priors दिखाता है (green noise) | सुनिश्चित करें कि `stable_cascade_use_decoder_for_validation` `true` है और decoder weights डाउनलोड हैं |
| Text embed caching में घंटों लगते हैं | cache डायरेक्टरी के लिए SSD/NVMe उपयोग करें और network mounts से बचें। prompts prune करें या `simpletuner-text-cache` CLI से pre‑compute करें |
| Autoencoder import error | अपनी `.venv` में torchvision इंस्टॉल करें (`pip install torchvision --extra-index-url https://download.pytorch.org/whl/cu124`)। Stage C को EfficientNet weights चाहिए |

## अगले कदम

- विषय की जटिलता के अनुसार `lora_rank` (8–32) और `learning_rate` (5e-5 से 2e-4) प्रयोग करें।
- prior प्रशिक्षण के बाद Stage B पर ControlNet/conditioning adapters जोड़ें।
- यदि तेज़ iteration चाहिए, `stage-c-lite` flavour ट्रेन करें और validation के लिए `decoder_lite` weights रखें।

सफल ट्यूनिंग!
