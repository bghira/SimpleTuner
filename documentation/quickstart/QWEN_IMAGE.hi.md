## Qwen Image 2.1

Qwen Image 2.1 अब डिफ़ॉल्ट है (`model_flavour: "v2.1"`) और `Qwen/Qwen-Image-2.1` का उपयोग करता है। इसमें 32 ब्लॉक वाला transformer, Qwen3-VL टेक्स्ट एन्कोडर और 16× स्थानिक संपीड़न वाला 64-चैनल VAE है।

Qwen Image 2.1 अब वैलिडेशन और अन्य VAE डिकोडिंग के लिए डिफ़ॉल्ट रूप से [Ollin का टेक्सचर-सुधारित VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) इस्तेमाल करता है। केवल डिकोडर को फ़ाइन-ट्यून किया गया है; एनकोडर अपरिवर्तित है, इसलिए मौजूदा 2.1 ट्रेनिंग लैटेंट्स संगत रहते हैं। VAE का रिविज़न बेस मॉडल से स्वतंत्र रूप से पिन किया गया है। मूल VAE के आउटपुट दोहराने के लिए `pretrained_vae_model_name_or_path: "Qwen/Qwen-Image-2.1"` सेट करें। स्पष्ट रूप से दिए गए VAE और पुराने फ़्लेवर का व्यवहार नहीं बदलता।

`qwen_image.peft-lora` उदाहरण `RareConcepts/Domokun` पर 512px में प्रशिक्षण देता है और ट्रिगर `🟫` का उपयोग करता है। BF16 (`base_model_precision: "no_change"`) से शुरू करें और मेमोरी कम होने पर gradient checkpointing का उपयोग करें। उदाहरण में 2.1 के लिए अलग latent और टेक्स्ट कैश हैं; पुराने संस्करणों के कैश दोबारा उपयोग न करें।

```bash
simpletuner train example=qwen_image.peft-lora
```

सत्यापन के लिए `validation_guidance: 1.0`, `validation_guidance_real: 1.0` और `validation_num_inference_steps: 40` रखें। यह जाँचने के लिए कि मॉडल ने विषय सीखा है या नहीं, सत्यापन प्रॉम्प्ट में ट्रिगर शामिल करें।

Qwen Image 2.1 एकल छवियों को डिकोड करते समय अनुपयोगी कालिक फीचर कैश नहीं रखता। H200 पर BF16 में एक 2048×2048 छवि को अलग से डिकोड करने पर, इससे आवंटित मेमोरी का पीक 26.87 से घटकर 15.28 GiB हुआ और आउटपुट बिल्कुल समान रहा। टाइल आधारित डिकोडिंग अधिक मेमोरी बचाती है, लेकिन स्थानिक संदर्भ सीमित होने से रंगीन जोड़ दिख सकते हैं; अनुपयोगी कैश हटाने से ये जोड़ ठीक नहीं होते।

पुराने संस्करण उपलब्ध हैं: `v1.0` से Qwen-Image, `v2.0` से Qwen-Image-2512 चुना जाता है और `edit-*` संस्करण अपने मौजूदा checkpoints रखते हैं। इनके adapters और latent कैश 2.1 के साथ अदला-बदली नहीं किए जा सकते।

250-step Domokun recipe throughput मापने का उदाहरण है, भरोसेमंद convergence recipe नहीं। पहले के एक checkpoint ने दोबारा लोड करने पर पहचानने योग्य Domokun बनाया, लेकिन नए 250-step runs वह परिणाम दोहरा नहीं पाए। Padding masks बनाए रखने, compilation बंद करने और पुराने RoPE expression का उपयोग करने वाले controls भी विफल रहे। Cached latents सही विषय में decode होते हैं। Training deterioration का कारण अभी स्पष्ट नहीं है; timing tables अलग attention backends की समान image quality साबित नहीं करतीं।

### VRAM प्रीसेट

इन उदाहरणों में 512px पर BF16, rank-32 LoRA, Optimi Lion और regional compilation उपयोग होते हैं; gradient checkpointing बंद है। पहली बार compilation में समय लगता है, इसलिए warm-up के बाद के training steps की तुलना करें। 24 GB और 32 GB मेमोरी बजट L40S पर जाँचे गए हैं, उन क्षमताओं वाले अलग GPU पर नहीं।

| VRAM बजट | उदाहरण | डेटासेट बैच आकार | पीक VRAM (GiB) | warm-up के बाद step (सेकंड) |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 1 | 20.6 | 0.238 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 2 | 26.5 | 0.390 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 2 | 26.5 | 0.390 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 10 | 71.8 | 0.639 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 20 | 128.4 | 1.223 |

L40S (24/32/48 GB प्रीसेट), H100 (80 GB) और H200 (144 GB) पर 20 steps मापे गए; timing से पहले पाँच steps हटाए गए। peak VRAM में तैयारी शामिल है। ये 512px और संबंधित बैच आकार के परिणाम हैं, बड़े चित्रों या लंबे prompts के लिए गारंटी नहीं।

48 GB प्रीसेट भी बैच 2 उपयोग करता है: L40S पर प्रति-चित्र throughput बैच 3, 4 और 5 से बेहतर था। बैच 5, 43.3 GiB में फिट हुआ लेकिन 0.991 सेकंड/step लगा; बैच 2 में 0.390 सेकंड/step लगा।

```bash
simpletuner train example=qwen_image-2.1-48g.peft-lora
```

हर उदाहरण के साथ दिया गया डेटासेट फ़ाइल उपयोग करें: उसमें बैच आकार स्पष्ट है। बैच आकार या डेटासेट सेटिंग बदलने पर नया प्रशिक्षण शुरू करें; असंगत training-state checkpoint का पुनः उपयोग न करें।

कम VRAM के लिए `gradient_checkpointing: true` और `gradient_checkpointing_interval: 2` सक्षम करें। अब यह लगातार दो blocks के समूह पर checkpoint लागू करता है। मापी गई तुलना के लिए [Qwen Image 2.1 checkpoint और attention परिणाम](../experimental/SEGMENTED_CHECKPOINTING.hi.md#qwen-image-21) देखें; हर दूसरे block पर checkpoint करने वाला पुराना परिणाम अब लागू नहीं है। इन presets में BF16 फिट होता है और int8 checkpoint आवश्यक नहीं है।

टेक्स्ट-टू-इमेज पथ tensor के मान पर निर्भर sequence assembly से बचता है, जिससे graph break के बिना capture होता है। वास्तविक संख्या वाले RoPE से Inductor normalization और rotation को fuse कर सकता है; modulation, residual और MLP epilogues भी compile होते हैं। इन प्रशिक्षण उदाहरणों में मौजूदा Hopper CuTe ConvRot GEMM और केवल inference के लिए बने LTX RoPE kernels उपयोग नहीं होते।


### प्रयोगात्मक AnyFlow पायलट

तीन `qwen_image-2.1-anyflow-stage*.peft-lora` उदाहरण जाँचते हैं कि interval distillation से `🟫` जोड़ते हुए base model का व्यवहार बचाया जा सकता है या नहीं। ये प्रयोगात्मक हैं; Qwen का model card यह सिद्ध नहीं करता कि पहले की गिरावट guidance distillation के कारण हुई थी।

सभी stages में 1024px, BF16, AdamW, rank 32 और batch 4 हैं। Stage 1 में H200 पर gradient checkpointing बंद है; stages 2 और 3 लगातार दो blocks पर checkpointing इस्तेमाल करते हैं। यह ऊपर दिए 512px throughput presets से अलग workload है। चलाने से पहले `webshart` इंस्टॉल करें।

इन AnyFlow उदाहरणों में स्पष्ट रूप से `grad_clip_method: "value"` और `max_grad_norm: 0.01` उपयोग होते हैं: हर gradient element को ±0.01 तक सीमित किया जाता है। यह global norm की सीमा नहीं है। 1.0 पर norm clipping का परीक्षण करने के लिए `grad_clip_method: "norm"` और `max_grad_norm: 1.0` दोनों सेट करें।

1. **Stage 1:** `1e-5` पर 10,000 forward AnyFlow updates। CC12M के लिए `webshart/cc12m-structured-captions` और `caption_key: "long_caption"`; e621 के लिए `webshart/e621-2024-webp-4Mpixel-webshart-indices` है। दोनों prior datasets में अधिकतम 4,096 स्वीकृत images और प्रत्येक का sampling weight 0.49 है। `RareConcepts/Domokun` का trigger `🟫`, weight 0.02 और `repeats: 0` है।
2. **Stage 2:** `2e-6` पर 2,000 on-policy DMD updates, उसी मिश्रण पर forward objective का co-training। यह AnyFlow DMD है, preference-pair DPO नहीं। दोनों stages में character की वास्तविक images शामिल हैं; frozen teacher अकेले नया character नहीं सिखा सकता।
3. **Stage 3:** `5e-7` पर 100 updates, Domokun का weight 0.5 और दो regularisation datasets का प्रत्येक weight 0.25 तथा अधिकतम 64 images है। सभी intervals `r=t` और raw flow target इस्तेमाल करते हैं, जिससे छोटे supervised refinement में interval embedder बना रहता है। Regularisation batches adapter बंद करके base prediction इस्तेमाल करते हैं।

Stage 1 को 20,000 updates तक बढ़ाने से पहले 2,000, 5,000 और 10,000-update checkpoints की तुलना करें। Stage 2 का budget 2,000 updates है; समान validation settings पर images बिगड़ें तो पहले रोकें। केवल step count से मॉडल को final नहीं माना जा सकता।

अलग-अलग caption lengths के लिए dynamic compilation चालू है। `schedule_shift: 2.000802574061872` जारी scheduler की 1024px (4,096 latent tokens) सेटिंग से मेल खाता है; resolution बदलने पर इसे दोबारा गणना करें।

```bash
simpletuner train example=qwen_image-2.1-anyflow-stage1.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage2.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage3.peft-lora
```

Stages 2 और 3 पिछले stage का अंतिम adapter `init_lora` से लोड करते हैं और optimizer तथा dataloader की नई state शुरू करते हैं। Sampling weights उपलब्ध datasets के बीच चयन नियंत्रित करते हैं; अंतिम sample प्रतिशत की गारंटी नहीं देते। यह सीमित पायलट पूरे prior corpora को कवर नहीं करता। String वाला `long_caption` मौजूदा Webshart caption selector के साथ काम करता है; native JSON-object caption support आवश्यक नहीं है।

Stages 1 और 2 में `diffusion_target: "base_prediction"` तथा `fuse_guidance_scale: 1.0` हैं: diffusion branch frozen conditional field बचाती है और अन्य branches data से सीखती हैं। यह preservation की परिकल्पना है, character सीखने की गारंटी नहीं। दिए गए character और prior prompts को 4, 16 और 40 inference steps पर जाँचें, और base model के 40-step output से तुलना करें; automatic validation 4 steps इस्तेमाल करता है। Objective और checkpoint आवश्यकताओं के लिए [AnyFlow](../experimental/ANYFLOW.hi.md) देखें।

पूरा हुए पायलट में चरण 1 ने 10,000 और चरण 2 ने 2,000 अपडेट पूरे किए। एडेप्टर दोबारा लोड करने पर 40-step लोमड़ी और चरित्र-प्रॉम्प्ट की तस्वीरें सुसंगत रहीं, लेकिन चरित्र वाले प्रॉम्प्ट ने Domokun के बजाय लोगों को बनाया। चार-step तस्वीरें धुंधली या शोरयुक्त रहीं। अलग L40S तुलना में दोनों checkpoints को 1024px, seed 42, CFG 1/2/4/6 और खाली negative prompt के साथ जाँचा गया। Forward-call assertions ने CFG 1 से ऊपर हर step पर दो passes की पुष्टि की। देखी गई लोमड़ी और समुद्र तट की तस्वीरें ठीक नहीं हुईं; ऊँचे CFG ने saturation और artifacts बढ़ाए। इन परिणामों से कारण तय नहीं होता; चरण 3 अभी सत्यापित नहीं है।

एक ही checkpoint से स्टेज 1 की दो शाखाओं को 1,000-1,000 अतिरिक्त updates तक चलाकर value-clipping सीमा 0.01 और 1.0 की तुलना की गई। दोबारा लोड करने पर दोनों की चार-step fox images में शोर रहा और 40-step समुद्र तट के चित्रों में अब भी लोग दिखे। 0.01 पर 65/1,000 updates में clipping हुई, जबकि 1.0 पर 0/1,000 में। दोनों शाखाओं ने बिना सुधार वाला optimizer इस्तेमाल किया और clipping शुरू होने से पहले ही शुरुआती gradients में अंतर था, इसलिए छोटे बदलावों को केवल सीमा का प्रभाव नहीं माना जा सकता।

<a id="qwen21-optimizer-correction"></a>

यहाँ दिए गए स्टेज 1 और असिस्टेंट के पायलट AdamW BF16 के stochastic-add helper को ठीक करने से पहले चलाए गए थे: वह `input + alpha * other` के बजाय `other + alpha * input` गणना करता था। β₁ = 0.9 पर पहला मोमेंट `m = 0.9 * m + 0.1 * g` के बजाय `m = 0.09 * m + g` के अनुसार अपडेट होता था। सटीक अंकगणित वाले regression tests CPU, MPS और CUDA पर पास हुए हैं। चित्रों के अवलोकन उन checkpoints के लिए सही हैं, लेकिन उनसे केवल आर्किटेक्चर या distillation objective का निष्कर्ष नहीं निकाला जा सकता; सुधारे गए optimizer के साथ प्रशिक्षण की दोबारा जाँच आवश्यक है।

सुधारे गए optimizer, उसी शुरुआती checkpoint और value clipping 1.0 के साथ 1,000 अतिरिक्त updates दोहराने पर भी जाँचे गए चार-step fox और समुद्र तट के चित्र ठीक नहीं हुए। 40 steps पर fox सुसंगत रहा, जबकि समुद्र तट के prompt ने अब भी एक व्यक्ति बनाया। यह मौजूदा checkpoint की रिकवरी का परीक्षण है, सुधारे गए optimizer से शुरुआत से प्रशिक्षण का नहीं; समग्र विफलता का कारण अभी स्पष्ट नहीं है।

### पुराने Qwen Image की सेटिंग (v1.0 / v2.0)

> 🆕 Edit checkpoints चाहिए? paired‑reference training निर्देशों के लिए [Qwen Image Edit quickstart](./QWEN_EDIT.md) देखें।

इस उदाहरण में, हम Qwen Image के लिए LoRA प्रशिक्षण करेंगे, जो 20B पैरामीटर वाला vision‑language मॉडल है। इसके आकार के कारण हमें आक्रामक मेमोरी ऑप्टिमाइज़ेशन तकनीकों की आवश्यकता होगी।

24GB GPU पूर्ण न्यूनतम है, और फिर भी व्यापक quantization और सावधानीपूर्ण कॉन्फ़िगरेशन चाहिए। 40GB+ अधिक सहज अनुभव के लिए मज़बूती से अनुशंसित है।

24G पर training करते समय, validations कम resolution या int8 से आगे की aggressive quant level के बिना OOM होंगी।

### हार्डवेयर आवश्यकताएँ

Qwen Image एक 20B पैरामीटर मॉडल है जिसमें एक परिष्कृत text encoder है जो अकेला ~16GB VRAM खपत करता है (quantization से पहले)। यह मॉडल 16 latent channels वाला custom VAE उपयोग करता है।

**महत्वपूर्ण सीमाएँ:**
- **AMD ROCm या MacOS पर समर्थित नहीं** क्योंकि efficient flash attention उपलब्ध नहीं
- Batch size > 1 अभी सही तरीके से काम नहीं करता; इसके बजाय gradient accumulation उपयोग करें
- TREAD (Text‑Representation Enhanced Adversarial Diffusion) अभी समर्थित नहीं है

### पूर्वापेक्षाएँ

सुनिश्चित करें कि Python इंस्टॉल है; SimpleTuner 3.10 से 3.12 के साथ अच्छा काम करता है।

आप इसे चलाकर जांच सकते हैं:

```bash
python --version
```

यदि आपके Ubuntu पर Python 3.12 इंस्टॉल नहीं है, तो आप यह प्रयास कर सकते हैं:

```bash
apt -y install python3.13 python3.13-venv
```

#### Container image dependencies

Vast, RunPod, और TensorDock (आदि) के लिए, CUDA 12.2‑12.8 इमेज पर CUDA extensions कम्पाइल करने हेतु यह काम करेगा:

```bash
apt -y install nvidia-cuda-toolkit
```

### इंस्टॉलेशन

pip के जरिए SimpleTuner इंस्टॉल करें:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [installation documentation](../INSTALL.md) देखें।

### वातावरण सेटअप

SimpleTuner चलाने के लिए, आपको एक configuration फ़ाइल, dataset और model directories, तथा एक dataloader configuration फ़ाइल सेट करनी होगी।

#### Configuration file

एक प्रयोगात्मक स्क्रिप्ट, `configure.py`, इंटरैक्टिव step‑by‑step कॉन्फ़िगरेशन के जरिए इस सेक्शन को पूरी तरह स्किप करने में मदद कर सकती है। इसमें कुछ सुरक्षा फीचर्स हैं जो सामान्य pitfalls से बचाते हैं।

**नोट:** यह आपके dataloader को कॉन्फ़िगर नहीं करता। आपको उसे बाद में मैन्युअली करना होगा।

इसे चलाने के लिए:

```bash
simpletuner configure
```

> ⚠️ जिन देशों में Hugging Face Hub आसानी से उपलब्ध नहीं है, वहाँ `HF_ENDPOINT=https://hf-mirror.com` को अपने `~/.bashrc` या `~/.zshrc` में जोड़ें, यह आपके सिस्टम के `$SHELL` पर निर्भर करता है।

यदि आप मैन्युअल कॉन्फ़िगर करना पसंद करते हैं:

`config/config.json.example` को `config/config.json` में कॉपी करें:

```bash
cp config/config.json.example config/config.json
```

फिर, आपको संभवतः निम्न वेरिएबल्स बदलने होंगे:

- `model_type` - इसे `lora` पर सेट करें।
- `lora_type` - PEFT LoRA के लिए `standard` या LoKr के लिए `lycoris` सेट करें।
- `model_family` - इसे `qwen_image` पर सेट करें।
- `model_flavour` - इसे `v1.0` पर सेट करें।
- `output_dir` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप अपने checkpoints और validation images रखना चाहते हैं। यहाँ full path उपयोग करने की सलाह है।
- `train_batch_size` - इसे उपलब्ध VRAM के अनुसार सेट करें। SimpleTuner के मौजूदा Qwen overrides में batch size > 1 समर्थित है।
- `gradient_accumulation_steps` - यदि per-step VRAM बढ़ाए बिना effective batch बढ़ाना हो, तो इसे 2‑8 पर सेट करें।
- `validation_resolution` - मेमोरी सीमाओं के लिए `1024x1024` या उससे कम रखें।
  - 24G अभी 1024x1024 validations संभाल नहीं सकता — आकार घटाएँ
  - अन्य resolutions को कॉमा से अलग कर सकते हैं: `1024x1024,768x768,512x512`
- `validation_guidance` - अच्छे परिणामों के लिए 3.0‑4.0 के आसपास रखें।
- `validation_num_inference_steps` - लगभग 30 रखें।
- `use_ema` - इसे `true` सेट करने से स्मूद परिणाम मिलते हैं लेकिन मेमोरी बढ़ती है।

- `optimizer` - अच्छे परिणामों के लिए `optimi-lion`, या यदि मेमोरी उपलब्ध हो तो `adamw-bf16`।
- `mixed_precision` - Qwen Image के लिए `bf16` आवश्यक है।
- `gradient_checkpointing` - उचित मेमोरी उपयोग के लिए इसे **अनिवार्य** (`true`) रखें।
- `base_model_precision` - 24GB कार्ड्स के लिए `int8-quanto` या `nf4-bnb` **मज़बूती से अनुशंसित** है।
- `quantize_via` - छोटे GPUs पर quantization के दौरान OOM से बचने के लिए `cpu` सेट करें।
- `quantize_activations` - प्रशिक्षण गुणवत्ता बनाए रखने के लिए `false` रखें।

24GB GPUs के लिए memory optimization settings:
- `lora_rank` - 8 या कम रखें।
- `lora_alpha` - इसे `lora_rank` के बराबर रखें।
- `flow_schedule_shift` - 1.73 पर सेट करें (या 1.0‑3.0 के बीच प्रयोग करें)।

न्यूनतम सेटअप के लिए आपका config.json कुछ ऐसा दिखेगा:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "model_type": "lora",
    "model_family": "qwen_image",
    "model_flavour": "v1.0",
    "lora_type": "standard",
    "lora_rank": 8,
    "lora_alpha": 8,
    "output_dir": "output/models-qwen_image",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "validation_resolution": "1024x1024",
    "validation_guidance": 4.0,
    "validation_num_inference_steps": 30,
    "validation_seed": 42,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_step_interval": 100,
    "vae_batch_size": 1,
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 1000,
    "max_grad_norm": 0.01,
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "learning_rate": "1e-4",
    "gradient_checkpointing": "true",
    "base_model_precision": "int2-quanto",
    "quantize_via": "cpu",
    "quantize_activations": false,
    "flow_schedule_shift": 1.73,
    "disable_benchmark": false,
    "data_backend_config": "config/qwen_image/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ Multi‑GPU उपयोगकर्ता उपयोग किए जाने वाले GPU की संख्या कॉन्फ़िगर करने के लिए [इस दस्तावेज़](../OPTIONS.md#environment-configuration-variables) को देखें।

> ⚠️ **24GB GPUs के लिए महत्वपूर्ण**: text encoder अकेला ~16GB VRAM उपयोग करता है। `int2-quanto` या `nf4-bnb` quantization के साथ इसे काफी कम किया जा सकता है।

काम करने वाले कॉन्फ़िग के साथ त्वरित sanity check:

**विकल्प 1 (अनुशंसित - pip install):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train example=qwen_image.peft-lora
```

**विकल्प 2 (Git clone विधि):**
```bash
simpletuner train env=examples/qwen_image.peft-lora
```

**विकल्प 3 (Legacy विधि - अभी भी काम करता है):**
```bash
ENV=examples/qwen_image.peft-lora ./train.sh
```

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

#### वैलिडेशन प्रॉम्प्ट्स

`config/config.json` के अंदर "primary validation prompt" होता है, जो आमतौर पर आपके single subject या style के लिए मुख्य instance_prompt होता है। इसके अतिरिक्त, एक JSON फ़ाइल बनाई जा सकती है जिसमें वैलिडेशन के दौरान चलाने के लिए अतिरिक्त प्रॉम्प्ट्स हों।

उदाहरण config फ़ाइल `config/user_prompt_library.json.example` का फ़ॉर्मैट:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

nicknames validation के लिए फ़ाइलनाम होते हैं, इसलिए इन्हें छोटा और फ़ाइलसिस्टम‑अनुकूल रखें।

ट्रेनर को इस prompt library की ओर इंगित करने के लिए, अपने config.json में यह जोड़ें:
```json
  "validation_prompt_library": "config/user_prompt_library.json",
```

विविध प्रॉम्प्ट्स का सेट यह निर्धारित करने में मदद करेगा कि मॉडल सही तरह से सीख रहा है या नहीं:

```json
{
    "anime_style": "a breathtaking anime-style portrait with vibrant colors and expressive features",
    "chef_cooking": "a high-quality, detailed photograph of a sous-chef immersed in culinary creation",
    "portrait": "a lifelike and intimate portrait showcasing unique personality and charm",
    "cinematic": "a cinematic, visually stunning photo with dramatic and captivating presence",
    "elegant": "an elegant and timeless portrait exuding grace and sophistication",
    "adventurous": "a dynamic and adventurous photo captured in an exciting moment",
    "mysterious": "a mysterious and enigmatic portrait shrouded in shadows and intrigue",
    "vintage": "a vintage-style portrait evoking the charm and nostalgia of a bygone era",
    "artistic": "an artistic and abstract representation blending creativity with visual storytelling",
    "futuristic": "a futuristic and cutting-edge portrayal set against advanced technology"
}
```

#### CLIP score ट्रैकिंग

यदि आप मॉडल प्रदर्शन स्कोर करने के लिए evaluations सक्षम करना चाहते हैं, तो CLIP scores को कॉन्फ़िगर और इंटरप्रेट करने के लिए [यह दस्तावेज़](../evaluation/CLIP_SCORES.md) देखें।

#### स्थिर evaluation loss

यदि आप मॉडल प्रदर्शन स्कोर करने के लिए stable MSE loss उपयोग करना चाहते हैं, तो evaluation loss को कॉन्फ़िगर और इंटरप्रेट करने के लिए [यह दस्तावेज़](../evaluation/EVAL_LOSS.md) देखें।

#### Validation previews

SimpleTuner Tiny AutoEncoder मॉडलों का उपयोग करके generation के दौरान intermediate validation previews स्ट्रीम करने का समर्थन करता है। इससे आप webhook callbacks के जरिए real‑time में step‑by‑step validation images देख सकते हैं।

सक्रिय करने के लिए:
```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

**आवश्यकताएँ:**
- Webhook configuration
- Validation सक्षम होना

`validation_preview_steps` को ऊँचा मान (जैसे 3 या 5) रखें ताकि Tiny AutoEncoder का ओवरहेड कम हो। `validation_num_inference_steps=20` और `validation_preview_steps=5` के साथ, आपको steps 5, 10, 15, और 20 पर preview images मिलेंगी।

#### Flow schedule shifting

Qwen Image, एक flow‑matching मॉडल के रूप में, generation प्रक्रिया के किस हिस्से पर प्रशिक्षण हो यह नियंत्रित करने के लिए timestep schedule shifting सपोर्ट करता है।

`flow_schedule_shift` पैरामीटर इसे नियंत्रित करता है:
- कम मान (0.1‑1.0): fine details पर फोकस
- मध्यम मान (1.0‑3.0): संतुलित प्रशिक्षण (अनुशंसित)
- अधिक मान (3.0‑6.0): बड़े compositional features पर फोकस

##### Auto‑shift
`--flow_schedule_auto_shift` के साथ resolution‑dependent timestep shift सक्षम कर सकते हैं, जो बड़े images के लिए उच्च shift मान और छोटे images के लिए कम shift मान उपयोग करता है। इससे स्थिर लेकिन संभवतः औसत प्रशिक्षण परिणाम मिलते हैं।

##### Manual specification
Qwen Image के लिए `--flow_schedule_shift` का मान 1.73 एक अच्छा शुरुआती बिंदु है, लेकिन डेटासेट और लक्ष्यों के अनुसार प्रयोग करना पड़ सकता है।

#### Dataset considerations

अपने मॉडल को प्रशिक्षित करने के लिए पर्याप्त बड़ा डेटासेट होना महत्वपूर्ण है। डेटासेट आकार पर सीमाएँ हैं, और आपको सुनिश्चित करना होगा कि आपका डेटासेट पर्याप्त बड़ा हो।

> ℹ️ बहुत कम images होने पर आपको **no images detected in dataset** संदेश दिख सकता है — `repeats` मान बढ़ाना इस सीमा को पार करेगा।

> ⚠️ **महत्वपूर्ण**: वर्तमान सीमाओं के कारण `train_batch_size` को 1 रखें और बड़े batch का अनुकरण करने के लिए `gradient_accumulation_steps` उपयोग करें।

एक `--data_backend_config` (`config/multidatabackend.json`) दस्तावेज़ बनाएँ जिसमें यह हो:

```json
[
  {
    "id": "pseudo-camera-10k-qwen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/qwen_image",
    "disabled": false,
    "write_batch_size": 16
  }
]
```

> ℹ️ यदि आपके पास captions वाली `.txt` फ़ाइलें हैं तो `caption_strategy=textfile` उपयोग करें।
> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।
> ℹ️ OOM से बचने के लिए text embeds का `write_batch_size` कम रखा गया है।

फिर, `datasets` डायरेक्टरी बनाएँ:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

यह लगभग 10k फोटोग्राफ सैंपल्स को आपकी `datasets/pseudo-camera-10k` डायरेक्टरी में डाउनलोड करेगा, जो अपने‑आप बन जाएगी।

आपकी Dreambooth images को `datasets/dreambooth-subject` डायरेक्टरी में जाना चाहिए।

#### WandB और Huggingface Hub में लॉग‑इन

प्रशिक्षण शुरू करने से पहले WandB और HF Hub में लॉग‑इन करना बेहतर है, खासकर यदि आप `--push_to_hub` और `--report_to=wandb` उपयोग कर रहे हैं।

यदि आप Git LFS रिपॉज़िटरी में मैन्युअली आइटम्स push करने वाले हैं, तो `git config --global credential.helper store` भी चलाएँ।

निम्न कमांड चलाएँ:

```bash
wandb login
```

और

```bash
huggingface-cli login
```

निर्देशों का पालन करके दोनों सेवाओं में लॉग‑इन करें।

</details>

### प्रशिक्षण रन निष्पादित करना

SimpleTuner डायरेक्टरी से, बस यह चलाएँ:

```bash
./train.sh
```

इससे text embed और VAE आउटपुट कैशिंग डिस्क पर शुरू होगी।

अधिक जानकारी के लिए [dataloader](../DATALOADER.md) और [tutorial](../TUTORIAL.md) दस्तावेज़ देखें।

### मेमोरी optimization टिप्स

#### सबसे कम VRAM कॉन्फ़िग (24GB न्यूनतम)

सबसे कम VRAM वाला Qwen Image कॉन्फ़िग लगभग 24GB मांगता है:

- OS: Ubuntu Linux 24
- GPU: एक NVIDIA CUDA डिवाइस (कम से कम 24GB)
- System memory: 64GB+ अनुशंसित
- Base model precision:
  - NVIDIA सिस्टम्स के लिए: `int2-quanto` या `nf4-bnb` (24GB कार्ड्स के लिए आवश्यक)
  - `int4-quanto` काम कर सकता है लेकिन गुणवत्ता कम हो सकती है
- Optimizer: मेमोरी दक्षता के लिए `optimi-lion` या `bnb-lion8bit-paged`
- Resolution: 512px या 768px से शुरू करें, मेमोरी अनुमति दे तो 1024px तक जाएँ
- Batch size: 1 (वर्तमान सीमाओं के कारण अनिवार्य)
- Gradient accumulation steps: 2‑8 से बड़े batch का अनुकरण करें
- `--gradient_checkpointing` सक्षम करें (अनिवार्य)
- Startup पर OOM से बचने के लिए `--quantize_via=cpu` उपयोग करें
- छोटा LoRA rank (1‑8) उपयोग करें
- environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` सेट करने से VRAM उपयोग कम होता है

**नोट**: VAE embeds और text encoder outputs की pre‑caching काफी मेमोरी उपयोग करेगी। यदि OOM हो तो `offload_during_startup=true` सक्षम करें।

### बाद में LoRA पर inference चलाना

क्योंकि Qwen Image एक नया मॉडल है, यहाँ inference के लिए काम करने वाला उदाहरण है:

<details>
<summary>Show Python inference example</summary>

```python
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

model_id = 'Qwen/Qwen-Image'
adapter_id = 'your-username/your-lora-name'

# Load the pipeline
pipeline = QwenImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
pipeline.load_lora_weights(adapter_id)

# Optional: quantize the model to save VRAM
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

# Move to device
pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

# Generate an image
prompt = "Your test prompt here"
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=4.0,
    generator=torch.Generator(device='cuda').manual_seed(42),
    width=1024,
    height=1024,
).images[0]

image.save("output.png", format="PNG")
```
</details>

### Notes & troubleshooting tips

#### Batch size limitations

पुराने diffusers Qwen builds में text embed padding और attention mask handling की वजह से batch size > 1 पर समस्याएँ थीं। SimpleTuner के मौजूदा Qwen overrides दोनों paths को patch करते हैं, इसलिए यदि VRAM अनुमति दे तो बड़े batches काम करते हैं।
- `train_batch_size` केवल तभी बढ़ाएँ जब आपकी memory headroom पर्याप्त हो।
- यदि किसी पुराने install पर artifacts दिखें, तो update करें और पुराने text embeds दोबारा generate करें।

#### Quantization

- `int2-quanto` सबसे आक्रामक मेमोरी बचत देता है लेकिन गुणवत्ता प्रभावित हो सकती है
- `nf4-bnb` मेमोरी और गुणवत्ता के बीच अच्छा संतुलन देता है
- `int4-quanto` मध्यम विकल्प है
- 40GB+ VRAM न हो तो `int8` से बचें

#### Learning rates

LoRA प्रशिक्षण के लिए:
- छोटे LoRAs (rank 1‑8): लगभग 1e‑4 learning rate
- बड़े LoRAs (rank 16‑32): लगभग 5e‑5 learning rate
- Prodigy optimizer के साथ: 1.0 से शुरू करें और इसे adapt करने दें

#### Image artifacts

यदि artifacts दिखें:
- learning rate घटाएँ
- gradient accumulation steps बढ़ाएँ
- उच्च‑गुणवत्ता और सही तरह से preprocessed images सुनिश्चित करें
- शुरुआत में कम resolutions उपयोग करने पर विचार करें

#### Multiple‑resolution training

शुरुआत में कम resolution (512px या 768px) पर training करें, फिर 1024px पर fine‑tune करें। अलग resolutions पर training करते समय `--flow_schedule_auto_shift` सक्षम करें।

### Platform limitations

**इन पर समर्थित नहीं:**
- AMD ROCm (efficient flash attention implementation नहीं)
- Apple Silicon/MacOS (memory और attention सीमाएँ)
- 24GB VRAM से कम वाले consumer GPUs

### Current known issues

1. Batch size > 1 सही काम नहीं करता (gradient accumulation उपयोग करें)
2. TREAD अभी समर्थित नहीं है
3. text encoder से उच्च मेमोरी उपयोग (~16GB quantization से पहले)
4. Sequence length handling समस्याएँ ([upstream issue](https://github.com/huggingface/diffusers/issues/12075))

अधिक सहायता और troubleshooting के लिए [SimpleTuner documentation](/documentation) देखें या community Discord जॉइन करें।

<a id="assistant-lora"></a>

### कैप्शन से सहायक LoRA का प्रशिक्षण

`qwen_image-2.1-assistant-lora.peft-lora` L40S के लिए प्रयोगात्मक प्रारंभिक सेटिंग है: BF16, बैच 1, अंतराल 2 पर ग्रेडिएंट चेकपॉइंटिंग, rank 32 और `1e-4` पर AdamW BF16। इसमें 1,000 अपडेट का बजट है और हर 50 अपडेट पर सत्यापन तथा सेव होता है। वास्तविक प्रशिक्षण से पहले बारह परीक्षण कैप्शन को विविध कैप्शन से बदलें; इस सेटिंग के अभिसरण की पुष्टि नहीं हुई है।

`grad_clip_method: "norm"`, `max_grad_norm: 1.0`.

`distillation_method: assistant_lora` चुनें। कैप्शन बैकएंड पहले टेक्स्ट एम्बेडिंग बनाता है। हर बैच में अडैप्टर बंद करके बेस मॉडल से 40 मूल इन्फरेंस चरणों और CFG 1 पर नए लेटेंट बनाए जाते हैं। निजी पाइपलाइन उसी transformer का उपयोग करती है और VAE, processor या टेक्स्ट एन्कोडर लोड नहीं करती। फिर अडैप्टर बहाल होता है और सामान्य डीनॉइज़िंग प्रशिक्षण चलता है। अंतिम लेटेंट कैश नहीं होते। अभी केवल Qwen Image 2.1 के टेक्स्ट-से-इमेज मोड का समर्थन है।

`distillation_config.assistant_lora` में `num_inference_steps` (डिफ़ॉल्ट 40), `resolutions` (`[चौड़ाई, ऊँचाई]` की गैर-रिक्त सूची, डिफ़ॉल्ट `[[1024, 1024]]`) और `seed` (42) आते हैं। आयाम 32 के गुणज होने चाहिए। रिज़ॉल्यूशन बैच के अनुसार चक्रीय बदलते हैं और बीज हर नमूने पर आगे बढ़ते हैं, छोटे अंतिम बैच में भी। चेकपॉइंट ये काउंटर सहेजते हैं। पुनः शुरू करते समय डेटा, बैच, ग्रेडिएंट संचय या वितरित टोपोलॉजी न बदलें। ऑन-डिमांड टेक्स्ट कैश समर्थित नहीं है।

प्रवाह की जाँच के लिए 8 अपडेट और शिक्षक के 2 चरण रखें; छवियों का मूल्यांकन करने से पहले 40 चरण बहाल करें। 100, 250, 500 और 1,000 अपडेट पर समान प्रॉम्प्ट और बीज वाली बेस छवियों से तुलना करें। सहायक की उपयोगिता की जाँच अलग कॉन्सेप्ट प्रशिक्षण में भी आवश्यक है।

Qwen Image 2.1 LoRA प्रशिक्षण अब डिफ़ॉल्ट रूप से [प्रशिक्षण सहायक v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2) लोड करता है। यह प्रशिक्षण के दौरान स्थिर रहता है और वैलिडेशन में निष्क्रिय रहता है। इसे बंद करने के लिए `disable_assistant_lora: true` सेट करें। दूसरा अडैप्टर चुनने के लिए `assistant_lora_path` दें। पुराने फ़्लेवर का व्यवहार नहीं बदलता। नया सहायक प्रशिक्षित करते समय संबंधित उदाहरणों की तरह `disable_assistant_lora: true` बनाए रखें।

इसे एकमात्र डिस्टिलेशन विधि के रूप में इस्तेमाल करें; अन्य डिस्टिलर के साथ संयोजन समर्थित नहीं है।

कैप्शन डेटासेट के लिए `dataloader_prefetch: false` आवश्यक है, ताकि चेकपॉइंट कर्सर उपयोग किए गए कैप्शन को दर्शाए। पुनः शुरू करते समय कैप्शन पहचान या पाठ, बैच, दोहराव, शफ़ल, बीज, ग्रेडिएंट संचय या वितरित व्यवस्था में बदलाव स्वीकार नहीं होते। सहायक LoRA के चेकपॉइंट जनरेशन बीज, रिज़ॉल्यूशन सूची या शिक्षक के इन्फरेंस चरणों में बदलाव भी अस्वीकार करते हैं।

<a id="assistant-lora-multires"></a>

#### चार बेस रिज़ॉल्यूशन और आस्पेक्ट रेशियो बकेट वाला असिस्टेंट प्रयोग

`qwen_image-2.1-assistant-lora-multires.peft-lora` 512, 1024, 1536 और 2048 बेस रिज़ॉल्यूशन के कुल 12 आस्पेक्ट रेशियो बकेट को क्रम से दोहराता है। हर बेस पर वर्गाकार, 4:7 पोर्ट्रेट और 7:4 लैंडस्केप बकेट हैं; हर बैच एक बकेट इस्तेमाल करता है और पूरे चक्र में हर बेस और आस्पेक्ट को बराबर एक्सपोज़र मिलता है। इसमें मूल उदाहरण के 40 teacher steps, BF16 batch 1, हर 2 blocks पर checkpointing, rank 32, AdamW BF16 और 1,000 updates बने रहते हैं। यह वही caption backend और validation prompts इस्तेमाल करता है। परीक्षण captions को baseline में इस्तेमाल किए गए उसी विविध caption सेट से बदलें। नए output directory में नया adapter और optimizer शुरू करें; `resume_from_checkpoint: ""` से resume बंद रहता है। तुलना के लिए पहले run के weights सुरक्षित रखें।

यह प्रयोग जाँचता है कि कई image sizes से assistant बेहतर होता है या नहीं; यह native resolution की आवश्यकता या गुणवत्ता लाभ का प्रमाण नहीं है। L40S पर 1,000 अपडेट सभी 12 buckets के साथ पूरे हुए और 1024×1024 तथा 2048×2048 पर validation हुआ। अंतिम लोमड़ी और portrait की तस्वीरें सुसंगत रहीं, लेकिन tiled VAE decoding की ज्ञात रंगीन सीमाएँ मौजूद थीं। Teacher बिना VAE decoding के latent targets बनाता है। बाद के concept training में लाभ अभी सत्यापित नहीं है।

Assistant LoRA में AnyFlow की तरह run के दायरे में Dynamo cache की न्यूनतम सीमा 32 entries होती है, ताकि teacher, student और validation variants समा सकें। उपयोगकर्ता की अधिक सीमा बनी रहती है और run समाप्त होने पर पुरानी सीमा लौट आती है। रिज़ॉल्यूशन या लंबे captions जोड़ते समय recompilations पर नज़र रखें।

इसके बाद Domokun के एक तुलनात्मक परीक्षण में दो नए adapters को 2048px, batch 1, LR `1e-5` और value clipping 1.0 पर 250-250 updates तक प्रशिक्षित किया गया। नियंत्रण में प्रशिक्षण के दौरान assistant strength 0 और दूसरे रन में 1 थी; inference के दौरान दोनों में assistant बंद था। शुरुआती adapter weights और validation images बिल्कुल समान थे। 1024px और 40 inference steps पर दोनों अंतिम रन चरित्र prompts के लिए अब भी लोगों को बनाते थे, जबकि fox और portrait के चित्र सुसंगत रहे; assistant का लाभ सिद्ध नहीं हुआ। दोनों रन और assistant की तैयारी में [ऊपर](#qwen21-optimizer-correction) बताया गया बिना सुधार वाला optimizer इस्तेमाल हुआ था।

<a id="assistant-lora-offline"></a>

#### दोबारा उपयोग की जा सकने वाली जनरेटेड छवियों से सहायक LoRA

`qwen_image-2.1-assistant-lora-offline.peft-lora` Webshart से [10,000 जनरेटेड छवियों](https://huggingface.co/datasets/webshart/qwen-image-2.1-generated-images) पर प्रशिक्षण करता है। डेटासेट CC12M के `long_caption` प्रॉम्प्ट, शिक्षक के 40 नेटिव स्टेप, CFG 1 और पूरी छवि की VAE डिकोडिंग का उपयोग करता है। बारह इमेज बैकएंड 512, 1024, 1536 और 2048 बेस रिज़ॉल्यूशन पर वर्गाकार, पोर्ट्रेट और लैंडस्केप बकेट देते हैं। सभी का सैंपलिंग भार समान है और repeats शून्य हैं।

यह नई शुरुआत वाली विधि सुधारे गए `adamw_bf16`, LR `1e-4`, `grad_clip_method: "norm"`, `max_grad_norm: 1.0`, BF16 बैच 1, रैंक 32 और हर दो ब्लॉक पर चेकपॉइंटिंग का उपयोग करती है। इसमें 1,000 अपडेट, हर 50 पर सेव और हर 100 पर 1024 में वैलिडेशन है। प्रकाशित करने से पहले अलग से 2048px प्रीव्यू बनाएँ। VAE tiling बंद है और एन्कोडिंग बैच 1 है। `vae_cache_ondemand: true` सैंपल चुने जाने पर छवियों को एन्कोड और कैश करता है, जिससे 1,000 अपडेट से पहले सभी 10,000 छवियों को एन्कोड नहीं करना पड़ता। सामान्य इमेज प्रशिक्षण ऑनलाइन शिक्षक जनरेशन की जगह लेता है; नया सहायक बनाते समय `distillation_method` छोड़ें और `disable_assistant_lora: true` रखें।

डेटासेट को दूसरे प्रयोगों में भी इस्तेमाल कर सकते हैं। PNG को फिर से VAE से एन्कोड करना पड़ता है, इसलिए ये लक्ष्य शिक्षक के अंतिम latent के समान नहीं हैं। प्रकाशित करने से पहले वैलिडेशन छवियाँ देखें और अलग कॉन्सेप्ट प्रशिक्षण में सहायक का लाभ जाँचें। कैप्शन वाली विधि से बदलते समय पुराने ऑप्टिमाइज़र या डेटासेट स्टेट को resume न करें; नया प्रशिक्षण शुरू करें।

[बदले गए v1 सहायक](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v1) ने L40S पर इस विधि के 1,000 अपडेट पूरे किए; अंतिम छवियों की 1024 और 2048 पर समीक्षा की गई। इसके बाद Domokun की समान परिस्थितियों वाली तुलना में 2048, LR `1e-4` और norm clipping 1.0 पर दोनों रन को 1,000 अपडेट दिए गए। सहायक प्रशिक्षण में frozen और validation में बंद रहा। नियंत्रण और सहायक वाले दोनों रन ने चरित्र के दोनों प्रॉम्प्ट पर अब भी लोगों की छवियाँ बनाईं। नियंत्रण ने एक असंबंधित लोमड़ी प्रॉम्प्ट में Domokun के स्पष्ट लक्षण डाल दिए; सहायक वाले रन ने पहचानने योग्य लोमड़ी बनाए रखी। दोनों में पोर्ट्रेट सुसंगत रहा। यह अवधारणा के दूसरे प्रॉम्प्ट में फैलने में कमी का सीमित प्रमाण है, सफल चरित्र सीखने या सामान्य गुणवत्ता लाभ का नहीं; मूल्यांकन में एक training seed और चार प्रॉम्प्ट हैं।

```bash
simpletuner train example=qwen_image-2.1-assistant-lora-offline.peft-lora
```
