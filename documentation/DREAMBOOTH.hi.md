# Dreambooth (single-subject प्रशिक्षण)

## पृष्ठभूमि

Dreambooth शब्द Google द्वारा विकसित एक तकनीक को दर्शाता है, जिसमें उच्च गुणवत्ता वाली कुछ इमेजेज़ का उपयोग करके विषयों को मॉडल में फाइनट्यून करके इंजेक्ट किया जाता है ([पेपर](https://dreambooth.github.io))।

फाइन‑ट्यूनिंग के संदर्भ में, Dreambooth मॉडल collapse (जैसे overfitting या artifacts) को रोकने में मदद करने के लिए नई तकनीकें जोड़ता है।

### Regularisation images

Regularisation images आमतौर पर उसी मॉडल द्वारा बनाई जाती हैं जिसे आप ट्रेन कर रहे हैं, ऐसे token का उपयोग करके जो आपके class से मिलता‑जुलता हो।

ये **जरूरी नहीं** कि मॉडल द्वारा बनाई गई synthetic इमेजेज़ हों, लेकिन यह वास्तविक डेटा (जैसे असली लोगों की फोटोज़) की तुलना में बेहतर प्रदर्शन दे सकता है।

उदाहरण: यदि आप किसी पुरुष विषय की इमेजेज़ पर प्रशिक्षण कर रहे हैं, तो आपकी regularisation data में अन्य यादृच्छिक पुरुषों की फोटोज़ या synthetic samples होंगे।

> 🟢 Regularisation images को अलग dataset के रूप में कॉन्फ़िगर किया जा सकता है, जिससे वे आपके प्रशिक्षण डेटा के साथ समान रूप से mix हों।

### Rare token training

मूल पेपर का एक संदिग्ध मूल्य वाला विचार था मॉडल के tokenizer vocabulary में reverse search करके एक "rare" string ढूँढना, जिस पर बहुत कम training हुई हो।

उस समय से यह विचार विकसित और बहस का विषय बना है, और एक विरोधी पक्ष किसी समान दिखने वाले celebrity नाम पर ट्रेन करने को प्राथमिकता देता है, क्योंकि इसमें कम compute लगता है।

> 🟡 SimpleTuner में rare token training समर्थित है, लेकिन इसे ढूँढने का कोई टूल उपलब्ध नहीं है।

### Prior preservation loss

मॉडल में एक "prior" होता है जिसे सिद्धांततः Dreambooth training के दौरान संरक्षित किया जा सकता है। लेकिन Stable Diffusion प्रयोगों में यह मददगार नहीं लगा — मॉडल अपनी ही जानकारी पर overfit हो जाता है।

> 🟢 ([#1031](https://github.com/bghira/SimpleTuner/issues/1031)) SimpleTuner में LyCORIS adapters ट्रेन करते समय `is_regularisation_data` सेट करके prior preservation loss समर्थित है।

### Masked loss

इमेज masks को image डेटा के साथ जोड़ा जा सकता है। mask के गहरे हिस्से loss calculations को उन हिस्सों को अनदेखा करने के लिए बाध्य करते हैं।

एक उदाहरण [script](/scripts/datasets/masked_loss/generate_dataset_masks.py) उपलब्ध है, जो input_dir और output_dir दिए जाने पर masks बना सकता है:

```bash
python generate_dataset_masks.py --input_dir /images/input \
                      --output_dir /images/output \
                      --text_input "person"
```

हालाँकि, इसमें mask padding blurring जैसी advanced functionality नहीं है।

अपने image mask dataset को परिभाषित करते समय:

- हर image के लिए mask होना चाहिए। यदि mask नहीं चाहिए तो all‑white image उपयोग करें।
- conditioning (mask) data folder पर `dataset_type=conditioning` सेट करें
- mask dataset पर `conditioning_type=mask` सेट करें
- image dataset पर `conditioning_data=` में अपने conditioning dataset का `id` सेट करें

```json
[
    {
        "id": "dreambooth-data",
        "type": "local",
        "dataset_type": "image",
        "conditioning_data": "dreambooth-conditioning",
        "instance_data_dir": "/training/datasets/test_datasets/dreambooth",
        "cache_dir_vae": "/training/cache/vae/sdxl/dreambooth-data",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an dreambooth",
        "metadata_backend": "discovery",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution_type": "pixel_area"
    },
    {
        "id": "dreambooth-conditioning",
        "type": "local",
        "dataset_type": "conditioning",
        "instance_data_dir": "/training/datasets/test_datasets/dreambooth_mask",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution_type": "pixel_area",
        "conditioning_type": "mask"
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "/training/cache/text/sdxl-base/masked_loss"
    }
]
```

## सेटअप

Dreambooth‑specific कॉन्फ़िगरेशन पर जाने से पहले [ट्यूटोरियल](TUTORIAL.md) का पालन करना आवश्यक है।

DeepFloyd tuning के लिए, उस मॉडल के setup संबंधी विशेष टिप्स के लिए [यह पेज](DEEPFLOYD.md) देखें।

### Quantised मॉडल प्रशिक्षण (केवल LoRA/LyCORIS)

Apple और NVIDIA सिस्टम्स पर परीक्षण किए गए Hugging Face Optimum-Quanto से precision और VRAM आवश्यकताएँ कम की जा सकती हैं।

अपने SimpleTuner venv के अंदर:

```bash
pip install optimum-quanto
```

उपलब्ध precision स्तर आपके हार्डवेयर और उसकी क्षमताओं पर निर्भर करते हैं।

- int2-quanto, int4-quanto, **int8-quanto** (अनुशंसित)
- fp8-quanto, fp8-torchao (केवल CUDA >= 8.9, जैसे 4090 या H100)
- nf4-bnb (low‑VRAM उपयोगकर्ताओं के लिए आवश्यक)

अपने config.json में निम्न मान बदलें या जोड़ें:
```json
{
    "base_model_precision": "int8-quanto",
    "text_encoder_1_precision": "no_change",
    "text_encoder_2_precision": "no_change",
    "text_encoder_3_precision": "no_change"
}
```

हमारी dataloader कॉन्फ़िग `multidatabackend-dreambooth.json` में यह कुछ ऐसा दिखेगा:

```json
[
    {
        "id": "subjectname-data-512px",
        "type": "local",
        "instance_data_dir": "/training/datasets/subjectname",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "subjectname",
        "cache_dir_vae": "/training/vae_cache/subjectname",
        "repeats": 100,
        "crop": false,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "minimum_image_size": 192
    },
    {
        "id": "subjectname-data-1024px",
        "type": "local",
        "instance_data_dir": "/training/datasets/subjectname",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "subjectname",
        "cache_dir_vae": "/training/vae_cache/subjectname-1024px",
        "repeats": 100,
        "crop": false,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "minimum_image_size": 768
    },
    {
        "id": "regularisation-data",
        "type": "local",
        "instance_data_dir": "/training/datasets/regularisation",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "a picture of a man",
        "cache_dir_vae": "/training/vae_cache/regularisation",
        "repeats": 0,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "minimum_image_size": 192,
        "is_regularisation_data": true
    },
    {
        "id": "regularisation-data-1024px",
        "type": "local",
        "instance_data_dir": "/training/datasets/regularisation",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "a picture of a man",
        "cache_dir_vae": "/training/vae_cache/regularisation-1024px",
        "repeats": 0,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "minimum_image_size": 768,
        "is_regularisation_data": true
    },
    {
        "id": "textembeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/training/text_cache/sdxl_base"
    }
]
```

कुछ प्रमुख मानों को single subject ट्रेनिंग आसान बनाने के लिए बदला गया है:

- अब दो datasets को दो‑दो बार कॉन्फ़िगर किया गया है, कुल चार datasets। Regularisation data वैकल्पिक है और बिना इसके भी training बेहतर हो सकती है। चाहें तो इस dataset को सूची से हटा सकते हैं।
- Resolution को 512px और 1024px mixed bucketing पर सेट किया गया है, जिससे training speed और convergence बेहतर हो सकते हैं
- Minimum image size 192px या 768px रखी गई है ताकि कुछ छोटे लेकिन महत्वपूर्ण low‑resolution images को upsample किया जा सके, जो कुछ datasets में आवश्यक हो सकता है।
- `caption_strategy` अब `instanceprompt` है, यानी dataset की हर image के लिए `instance_prompt` को caption के रूप में उपयोग किया जाएगा।
  - **Note:** instance prompt उपयोग करना Dreambooth का पारंपरिक तरीका है, लेकिन छोटे captions बेहतर काम कर सकते हैं। यदि मॉडल generalise नहीं कर रहा, तो captions उपयोग करने का प्रयास करें।

### Regularisation dataset विचार

Regularisation dataset के लिए:

- अपने Dreambooth subject पर `repeats` बहुत ऊँचा रखें ताकि Dreambooth डेटा का image count `repeats` गुना होकर regularisation set से अधिक हो जाए
  - यदि Regularisation set में 1000 images हैं और आपके training set में 10 images हैं, तो तेज़ परिणामों के लिए `repeats` कम से कम 100 होना चाहिए
- `minimum_image_size` बढ़ाई गई है ताकि बहुत अधिक low‑quality artifacts न आएं
- इसी तरह, अधिक descriptive captions भूलने से बचने में मदद कर सकते हैं। `instanceprompt` से `textfile` या अन्य रणनीतियों पर स्विच करने के लिए हर image के लिए `.txt` फाइलें बनानी होंगी।
- जब `is_regularisation_data` (या अमेरिकी users के लिए 🇺🇸 `is_regularization_data` with z) सेट किया जाता है, तो इस set का डेटा base model में feed होता है ताकि एक prediction बने जिसे student LyCORIS मॉडल के लिए loss target के रूप में उपयोग किया जा सके।
  - ध्यान दें, यह फिलहाल केवल LyCORIS adapter पर काम करता है।

## Instance prompt चुनना

जैसा पहले बताया, Dreambooth का मूल फोकस rare tokens चुनने पर था।

वैकल्पिक रूप से, कोई अपने subject का वास्तविक नाम या किसी "पर्याप्त समान" celebrity का नाम इस्तेमाल कर सकता है।

कई प्रशिक्षण प्रयोगों के बाद, ऐसा लगता है कि "पर्याप्त समान" celebrity सबसे अच्छा विकल्प है, खासकर यदि मॉडल को वास्तविक नाम से prompt करने पर वह असमान दिखता हो।

# Scheduled Sampling (Rollout)

Dreambooth जैसे छोटे datasets पर training करते समय, मॉडल जल्दी "perfect" noise पर overfit हो सकता है। इससे **exposure bias** होता है: मॉडल perfect inputs को denoise करना सीखता है, लेकिन inference के दौरान अपने ही थोड़े‑से imperfect outputs के सामने फेल हो जाता है।

**Scheduled Sampling (Rollout)** इसे इस तरह संबोधित करता है कि training loop के दौरान कभी‑कभी मॉडल को कुछ steps के लिए अपने ही noisy latents generate करने देता है। pure Gaussian noise + signal पर ट्रेन करने के बजाय, यह "rollout" samples पर ट्रेन होता है जिनमें मॉडल की पिछली गलतियाँ शामिल होती हैं। इससे मॉडल खुद को सुधारना सीखता है और subject generation अधिक robust और stable होता है।

> 🟢 यह फीचर प्रयोगात्मक है लेकिन छोटे datasets के लिए जहाँ overfitting या "frying" सामान्य है, अत्यधिक अनुशंसित है।
> ⚠️ rollout सक्षम करने से compute requirements बढ़ते हैं क्योंकि training loop में extra inference steps चलाने पड़ते हैं।

इसे सक्षम करने के लिए अपने `config.json` में ये keys जोड़ें:

```json
{
  "scheduled_sampling_max_step_offset": 10,
  "scheduled_sampling_probability": 1.0,
  "scheduled_sampling_ramp_steps": 1000,
  "scheduled_sampling_sampler": "unipc"
}
```

*   `scheduled_sampling_max_step_offset`: कितने steps generate करने हैं। छोटा मान (जैसे 5‑10) अक्सर पर्याप्त होता है।
*   `scheduled_sampling_probability`: इस तकनीक को कितनी बार लागू करना है (0.0 से 1.0)।
*   `scheduled_sampling_ramp_steps`: शुरुआती N steps में probability धीरे‑धीरे बढ़ाएँ ताकि training destabilize न हो।

# Exponential moving average (EMA)

एक दूसरा मॉडल आपके checkpoint के साथ parallel में लगभग मुफ्त में train किया जा सकता है — इसमें केवल system memory (डिफ़ॉल्ट रूप से) उपयोग होती है, अधिक VRAM नहीं।

अपने config file में `use_ema=true` सेट करके यह फीचर सक्षम करें।

# CLIP score tracking

यदि आप evaluations से मॉडल प्रदर्शन स्कोर करना चाहते हैं, तो CLIP scores कॉन्फ़िगर करने और समझने के लिए [यह दस्तावेज़](evaluation/CLIP_SCORES.md) देखें।

# स्थिर evaluation loss

यदि आप मॉडल प्रदर्शन स्कोर करने के लिए स्थिर MSE loss उपयोग करना चाहते हैं, तो evaluation loss कॉन्फ़िगर करने और समझने के लिए [यह दस्तावेज़](evaluation/EVAL_LOSS.md) देखें।

# Validation previews

SimpleTuner Tiny AutoEncoder मॉडल्स का उपयोग करके generation के दौरान intermediate validation previews streaming सपोर्ट करता है। यह feature आपको webhook callbacks के जरिए real‑time में validation images step‑by‑step देखने देता है, बजाय पूरी generation के खत्म होने का इंतज़ार करने के।

## Validation previews सक्षम करना

Validation previews सक्षम करने के लिए अपने `config.json` में निम्न जोड़ें:

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

## आवश्यकताएँ

- Tiny AutoEncoder समर्थन वाला model family (Flux, SDXL, SD3, आदि)
- preview images पाने के लिए webhook configuration
- validation सक्षम होना चाहिए (`validation_disable` true नहीं होना चाहिए)

## Configuration विकल्प

- `--validation_preview`: preview फीचर enable/disable (डिफ़ॉल्ट: false)
- `--validation_preview_steps`: sampling के दौरान previews कितनी बार decode हों (डिफ़ॉल्ट: 1)
  - 1 सेट करने पर हर sampling step पर preview मिलेगा
  - higher values (जैसे 3 या 5) से Tiny AutoEncoder decoding का overhead कम होगा

## उदाहरण

`validation_num_inference_steps=20` और `validation_preview_steps=5` के साथ, हर validation generation के दौरान steps 5, 10, 15, और 20 पर preview images मिलेंगी।

# Refiner tuning

यदि आप SDXL refiner पसंद करते हैं, तो हो सकता है कि यह आपकी Dreamboothed model outputs को "ruin" कर दे।

SimpleTuner SDXL refiner को LoRA और full rank दोनों से ट्रेन करने का समर्थन करता है।

इसमें कुछ विचार आवश्यक हैं:
- इमेजेस पूरी तरह उच्च‑गुणवत्ता वाली हों
- text embeds को base model के साथ share नहीं किया जा सकता
- VAE embeds **share** किए जा सकते हैं

अपनी dataloader configuration `multidatabackend.json` में `cache_dir` अपडेट करें:

```json
[
    {
        "id": "textembeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/training/text_cache/sdxl_refiner"
    }
]
```

यदि आप अपने डेटा के लिए विशिष्ट aesthetic score लक्ष्य करना चाहते हैं, तो `config/config.json` में यह जोड़ें:

```bash
"--data_aesthetic_score": 5.6,
```

**5.6** को अपने लक्षित score से बदलें। डिफ़ॉल्ट **7.0** है।

> ⚠️ SDXL refiner ट्रेन करते समय आपके validation prompts अनदेखे किए जाएंगे। इसके बजाय आपके datasets से रैंडम इमेजेस refine की जाएंगी।
