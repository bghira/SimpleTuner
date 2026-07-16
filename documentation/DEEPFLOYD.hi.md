# DeepFloyd IF

> 🤷🏽‍♂️ DeepFloyd ट्रेन करने के लिए LoRA के लिए कम से कम 24G VRAM चाहिए। यह गाइड 400M पैरामीटर वाले बेस मॉडल पर केंद्रित है, हालांकि 4.3B XL वेरिएंट को भी इन्हीं दिशानिर्देशों से ट्रेन किया जा सकता है।

## पृष्ठभूमि

2023 की वसंत में, StabilityAI ने DeepFloyd नाम का cascaded pixel diffusion मॉडल रिलीज़ किया।
![](https://tripleback.net/public/deepfloyd.png)

Stable Diffusion XL से संक्षिप्त तुलना:
- टेक्स्ट एन्कोडर
  - SDXL दो CLIP एन्कोडर्स का उपयोग करता है: "OpenCLIP G/14" और "OpenAI CLIP-L/14"
  - DeepFloyd एक single self‑supervised transformer मॉडल, Google का T5 XXL उपयोग करता है
- पैरामीटर count
  - DeepFloyd कई आकारों में आता है: 400M, 900M, और 4.3B पैरामीटर्स। हर बड़ा मॉडल क्रमशः महंगा ट्रेन होता है।
  - SDXL में सिर्फ़ एक है, ~3B पैरामीटर्स।
  - DeepFloyd का टेक्स्ट एन्कोडर अकेले 11B पैरामीटर्स का है, जिससे सबसे बड़ा कॉन्फ़िग लगभग 15.3B पैरामीटर्स हो जाता है।
- मॉडल count
  - DeepFloyd **तीन** स्टेज में चलता है: 64px -> 256px -> 1024px
    - हर स्टेज अपना denoising उद्देश्य पूरी तरह पूरा करता है
  - SDXL **दो** स्टेज में चलता है, रिफ़ाइनर सहित, 1024px -> 1024px
    - हर स्टेज केवल आंशिक रूप से अपना denoising उद्देश्य पूरा करता है
- डिज़ाइन
  - DeepFloyd के तीन मॉडल resolution और fine details बढ़ाते हैं
  - SDXL के दो मॉडल fine details और composition संभालते हैं

दोनों मॉडलों में, पहला स्टेज इमेज की अधिकतर composition तय करता है (जहाँ बड़े आइटम/शैडोज़ दिखाई देते हैं)।

## मॉडल मूल्यांकन

DeepFloyd का उपयोग करते समय आप यह अपेक्षा कर सकते हैं।

### सौंदर्य (Aesthetics)

SDXL या Stable Diffusion 1.x/2.x की तुलना में, DeepFloyd की aesthetics Stable Diffusion 2.x और SDXL के बीच कहीं आती हैं।


### नुकसान

यह मॉडल कई कारणों से लोकप्रिय नहीं है:

- inference‑time compute VRAM आवश्यकता अन्य मॉडलों से अधिक है
- training‑time compute VRAM आवश्यकताएँ अन्य मॉडलों से काफ़ी अधिक हैं
  - full u‑net tune के लिए 48G VRAM से अधिक चाहिए
  - rank‑32, batch‑4 LoRA को ~24G VRAM चाहिए
  - टेक्स्ट embed cache ऑब्जेक्ट्स बहुत बड़े हैं (प्रत्येक कई मेगाबाइट, जबकि SDXL के dual CLIP embeds केवल सैकड़ों किलाबाइट)
  - टेक्स्ट embed cache बहुत धीमे बनते हैं — अभी A6000 non‑Ada पर लगभग 9‑10 प्रति सेकंड
- डिफ़ॉल्ट aesthetic अन्य मॉडलों से खराब है (जैसे vanilla SD 1.5 ट्रेन करना)
- inference के दौरान **तीन** मॉडल फाइनट्यून/लोड करने पड़ते हैं (टेक्स्ट एन्कोडर जोड़ें तो चार)
- StabilityAI के वादे वास्तविक उपयोग अनुभव से मेल नहीं खाए (over‑hyped)
- DeepFloyd‑IF लाइसेंस commercial उपयोग के खिलाफ प्रतिबंधात्मक है
  - इससे NovelAI वेट्स पर असर नहीं पड़ा, जो अवैध रूप से लीक हुए थे। वाणिज्यिक लाइसेंस का हवाला अन्य बड़े मुद्दों की तुलना में एक सुविधाजनक बहाना लगता है।

### फायदे

हालाँकि, DeepFloyd के कुछ फायदे हैं जो अक्सर नज़रअंदाज़ होते हैं:

- inference समय पर, T5 टेक्स्ट एन्कोडर दुनिया की अच्छी समझ दिखाता है
- बहुत‑लंबे captions पर native training संभव है
- पहला स्टेज ~64x64 pixel area है, और multi‑aspect resolutions पर ट्रेन किया जा सकता है
  - ट्रेनिंग डेटा के low‑resolution होने का मतलब है कि DeepFloyd _एकमात्र मॉडल_ था जो LAION‑A के _सभी_ डेटा पर ट्रेन हो सकता था (LAION में बहुत कम इमेज 64x64 से छोटी हैं)
- हर स्टेज को अलग‑अलग उद्देश्य के लिए independently ट्यून किया जा सकता है
  - पहला स्टेज composition गुणों पर फोकस करता है, और बाद के स्टेज upscaled details बेहतर करने पर
- बड़े मेमोरी footprint के बावजूद यह तेज़ ट्रेन होता है
  - throughput के लिहाज़ से तेज़ — stage 1 tuning में उच्च samples‑per‑hour दिखते हैं
  - CLIP‑समकक्ष मॉडल्स की तुलना में तेज़ सीखता है, जो उन लोगों के लिए चुनौती हो सकता है जो CLIP मॉडल ट्रेन करने के आदी हैं
    - मतलब आपको learning rates और training schedules को लेकर अपेक्षाएँ समायोजित करनी होंगी
- इसमें VAE नहीं है; training samples सीधे target size में downscale होते हैं और pixels सीधे U‑net को दिए जाते हैं
- यह ControlNet LoRAs और अन्य कई ट्रिक्स को सपोर्ट करता है जो सामान्य linear CLIP u‑nets पर काम करते हैं

## LoRA को फाइन‑ट्यून करना

> ⚠️ DeepFloyd के सबसे छोटे 400M मॉडल में भी full u‑net backpropagation की compute आवश्यकताएँ बहुत अधिक हैं, इसलिए इसे टेस्ट नहीं किया गया है। इस दस्तावेज़ में LoRA उपयोग होगा, हालांकि full u‑net tuning भी काम कर सकती है।

ये निर्देश SimpleTuner से बेसिक परिचितता मानते हैं। नए उपयोगकर्ताओं के लिए [Kwai Kolors](quickstart/KOLORS.md) जैसे बेहतर‑समर्थित मॉडल से शुरू करना अनुशंसित है।

हालाँकि, यदि आप DeepFloyd ट्रेन करना चाहते हैं, तो आपको `model_flavour` कॉन्फ़िगरेशन विकल्प का उपयोग करके यह बताना होगा कि आप कौन सा मॉडल ट्रेन कर रहे हैं।

### config.json

```bash
"model_family": "deepfloyd",

# Possible values:
# - i-medium-400m
# - i-large-900m
# - i-xlarge-4.3b
# - ii-medium-450m
# - ii-large-1.2b
"model_flavour": "i-medium-400m",

# DoRA isn't tested a whole lot yet. It's still new and experimental.
"use_dora": false,
# Bitfit hasn't been tested for efficacy on DeepFloyd.
# It will probably work, but no idea what the outcome is.
"use_bitfit": false,

# Highest learning rate to use.
"learning_rate": 4e-5,
# For schedules that decay or oscillate, this will be the end LR or the bottom of the valley.
"lr_end": 4e-6,
```

- `model_family` का मान deepfloyd है
- `model_flavour` Stage I या II की ओर इंगित करता है
- `resolution` अब `64` है और `resolution_type` `pixel` है
- `attention_mechanism` को `xformers` सेट किया जा सकता है, लेकिन AMD और Apple उपयोगकर्ता इसे सेट नहीं कर पाएँगे, जिससे अधिक VRAM की आवश्यकता होगी
  - **Note** ~~Apple MPS में फिलहाल एक बग है जो DeepFloyd tuning को काम नहीं करने देता था।~~ PyTorch 2.6 या उससे पहले के संस्करणों से, stage I और II दोनों Apple MPS पर ट्रेन होते हैं।

अधिक thorough validation के लिए, `validation_resolution` का मान इस तरह सेट किया जा सकता है:

- `validation_resolution=64` 64x64 square image देगा।
- `validation_resolution=96x64` 3:2 widescreen image देगा।
- `validation_resolution=64,96,64x96,96x64` प्रत्येक validation पर चार इमेज बनाएगा:
  - 64x64
  - 96x96
  - 64x96
  - 96x64

### multidatabackend_deepfloyd.json

अब DeepFloyd प्रशिक्षण के लिए dataloader कॉन्फ़िगरेशन पर चलते हैं। यह SDXL या legacy मॉडल datasets की कॉन्फ़िगरेशन जैसा ही होगा, फ़ोकस resolution पैरामीटर्स पर रहेगा।

```json
[
    {
        "id": "primary-dataset",
        "type": "local",
        "instance_data_dir": "/training/data/primary-dataset",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "random",
        "resolution": 64,
        "resolution_type": "pixel",
        "minimum_image_size": 64,
        "maximum_image_size": 256,
        "target_downsample_size": 128,
        "prepend_instance_prompt": false,
        "instance_prompt": "Your Subject Trigger Phrase or Word",
        "caption_strategy": "instanceprompt",
        "repeats": 1
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "disable": false,
        "type": "local",
        "cache_dir": "/training/cache/deepfloyd/text/dreambooth"
    }
]
```

ऊपर DeepFloyd के लिए एक बेसिक Dreambooth कॉन्फ़िगरेशन दिया गया है:

- `resolution` और `resolution_type` के मान क्रमशः `64` और `pixel` हैं
- `minimum_image_size` को 64 pixels तक घटाया गया है ताकि हम छोटे images को गलती से upsample न करें
- `maximum_image_size` को 256 pixels रखा गया है ताकि बड़े images का crop ratio 4:1 से अधिक न हो, जो scene context loss का कारण बन सकता है
- `target_downsample_size` को 128 pixels सेट किया गया है ताकि 256 pixels से बड़े images पहले 128 pixels तक resize हों और फिर crop हों

Note: images को 25% के चरणों में downsample किया जाता है ताकि image size में बड़े jump से scene details का गलत averaging न हो।

## Validation pipeline modes

DeepFloyd validation अब SimpleTuner में stages को सीधे chain कर सकता है। `deepfloyd_validation_pipeline_mode=auto` default है: prompt validation stage I -> stage II चलाता है, जबकि dataset-image validation trained stage पर रहता है। single-stage validation force करने के लिए `trained-stage` उपयोग करें, या fixed peer stage हमेशा load करने के लिए `full-pipeline` उपयोग करें। peer checkpoints को `deepfloyd_validation_stage1_model` और `deepfloyd_validation_stage2_model` से override करें।

Optional stage III validation को `deepfloyd_validation_stage3_mode=sd-x4-upscaler` से enable किया जा सकता है; यह stage II के बाद `deepfloyd_validation_stage3_model` को terminal 4x upscaler के रूप में उपयोग करता है।

## इनफेरेंस चलाना

फिलहाल, SimpleTuner toolkit में DeepFloyd के लिए कोई dedicated inference scripts नहीं हैं।

built‑in validation प्रक्रिया के अलावा, आप Hugging Face के [इस दस्तावेज़](https://huggingface.co/docs/diffusers/v0.23.1/en/training/dreambooth#if) को देख सकते हैं, जिसमें बाद में inference चलाने का छोटा उदाहरण है:

```py
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", use_safetensors=True)
pipe.load_lora_weights("<lora weights path>")
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
```

> ⚠️ ध्यान दें कि `DiffusionPipeline.from_pretrained(...)` का पहला मान `IF-I-M-v1.0` है, लेकिन आपको इसे उस base model path से बदलना होगा जिस पर आपने अपना LoRA ट्रेन किया है।

> ⚠️ ध्यान दें कि Hugging Face की सभी सिफ़ारिशें SimpleTuner पर लागू नहीं होतीं। उदाहरण के लिए, हम DeepFloyd stage I LoRA को Diffusers के उदाहरण dreambooth scripts के 28G के बजाय केवल 22G VRAM में ट्यून कर सकते हैं, efficient pre‑caching और pure‑bf16 optimiser states के कारण।

## Super‑resolution stage II मॉडल को फाइन‑ट्यून करना

DeepFloyd का stage II मॉडल लगभग 64x64 (या 96x64) इनपुट लेता है, और `VALIDATION_RESOLUTION` सेटिंग के अनुसार upscaled image लौटाता है।

eval images आपके datasets से स्वतः चुनी जाती हैं, ताकि `--num_eval_images` यह बताए कि प्रत्येक dataset से कितनी upscale images चुनी जाएँ। images अभी रैंडम चुनते हैं — लेकिन हर session में वही बनी रहती हैं।

कुछ अतिरिक्त checks यह सुनिश्चित करते हैं कि आप गलत size सेटिंग्स के साथ गलती से रन न करें।

stage II ट्रेन करने के लिए, ऊपर दिए गए स्टेप्स का पालन करें और `MODEL_TYPE` के लिए `deepfloyd-lora` की जगह `deepfloyd-stage2-lora` उपयोग करें:

```bash
export MODEL_TYPE="deepfloyd-stage2-lora"
```
