# Qwen Image 2.1 क्विकस्टार्ट

[Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) में 7B, 32-ब्लॉक image transformer, Qwen3-VL text encoder और 16× spatial compression वाला 64-channel VAE है। SimpleTuner का डिफ़ॉल्ट `model_family: "qwen_image"` और `model_flavour: "v2.1"` है। यह गाइड text-to-image LoRA प्रशिक्षण के लिए है।

20B `v1.0` / `v2.0` मॉडल के लिए [पुराना Qwen Image गाइड](QWEN_IMAGE.hi.md) देखें। पुराने `edit-*` प्रकारों का paired-reference प्रशिक्षण [Qwen Edit](QWEN_EDIT.hi.md) में है। उनके adapters, text embeddings और latent caches 2.1 के साथ इस्तेमाल नहीं किए जा सकते।

[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)

## इंस्टॉलेशन

Python 3.12–3.14 इस्तेमाल करें और अपनी मशीन के लिए [इंस्टॉलेशन गाइड](../INSTALL.hi.md) देखें। नीचे के presets NVIDIA GPUs पर जाँचे गए हैं। SimpleTuner की dependencies में `webshart` शामिल है; regularisation डेटा के लिए नेटवर्क और स्थानीय cache स्थान चाहिए।

```bash
pip install 'simpletuner[cuda]'
```

<a id="vram-presets"></a>

## VRAM के अनुसार preset चुनें

मानक उदाहरण [assistant v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3), synthetic regularisation, REPA और automatic flow shift जोड़ते हैं। डिफ़ॉल्ट `qwen_image.peft-lora` में 48 GB preset वाली 512px + 1024px विधि है। 24/32 GB presets में validation सहित केवल 512px है।

| VRAM बजट | उदाहरण | आधार रिज़ॉल्यूशन | अपडेट | ग्रेडिएंट चेकपॉइंटिंग अंतराल |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 512px | 2000 | 1 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 512px | 2000 | 2 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 512px + 1024px | 4000 | 2 |

```bash
simpletuner train example=qwen_image-2.1-48g.peft-lora
```

सभी प्रीसेट BF16, rank/alpha 32, batch 1, `1e-4` पर AdamW BF16, 25 वार्मअप अपडेट और 1.0 नॉर्म क्लिपिंग इस्तेमाल करते हैं। REPA में `dinov2_vitg14`, ब्लॉक 8, वज़न 0.5, एन्कोडर आकार 518, स्थानिक संरेखण और समय दूरी 0 है। ऑटो शिफ्ट चालू और स्थिर शिफ्ट 0 है। सहायक प्रशिक्षण में स्थिर और वैलिडेशन में निष्क्रिय रहता है। नियमितीकरण का लक्ष्य दोनों अडैप्टर बंद करके बेस मॉडल की भविष्यवाणी से बनता है।

सामान्य संभाव्य सैम्पलर आधा वज़न `🟫` ट्रिगर वाले `RareConcepts/Domokun` को और आधा `is_regularisation_data: true` वाले `webshart/qwen-image-2.1-generated-images` को देता है। मल्टी-स्केल में प्रत्येक आधा 512px और 1024px में बराबर बँटता है। क्षेत्रफल-आधारित बकेट अनुपात बनाए रखते हैं: 0.262144 और 1.048576 मेगापिक्सेल। वर्गाकार, पोर्ट्रेट और लैंडस्केप सिंथेटिक उपसमूह अपने रिज़ॉल्यूशन का नियमितीकरण वज़न बराबर बाँटते हैं। यह निश्चित बारी-बारी चयन नहीं है। हर 250 अपडेट पर वैलिडेशन और चेकपॉइंट होता है।

हर synthetic aspect subset अधिकतम 1,024 चित्रों तक सीमित है। प्रत्येक रिज़ॉल्यूशन और स्रोत के latent caches अलग हैं। VAE tiling बंद है। अपडेट बजट शुरुआती सीमा है; प्रशिक्षण बढ़ाने से पहले checkpoint चित्र देखें।

<details markdown="1">
<summary>मेमोरी माप</summary>

L40S मेमोरी परीक्षण में हर बैकएंड से चार छवियाँ, 16 अपडेट, वैलिडेशन और सेव शामिल थे। 512px और अंतराल 1 वाला सेटअप 24 GiB सीमा में सफल रहा: PyTorch का अधिकतम आवंटन / आरक्षण 20.06 / 21.10 GiB था। मल्टी-स्केल और अंतराल 2 वाला सेटअप L40S पर 32.49 / 41.21 GiB में सफल रहा, लेकिन 32 GiB सीमा पर मेमोरी समाप्त हुई। ये छोटे उपसमूह के मेमोरी परीक्षण हैं, गति या अभिसरण के माप नहीं; छोटी सीमाएँ अलग कार्ड की जगह L40S पर लागू की गई थीं। अंतिम 32 GB प्रीसेट, 512px और अंतराल 2 के साथ, सफल रहा: 22.12 GiB आवंटित / 23.68 GiB आरक्षित।

</details>

## अपना डेटा इस्तेमाल करें

चुने हुए उदाहरण के `config.json` और `dataloader.json` को अपने training environment में कॉपी करें और अपना prompt library बनाएँ। `data_backend_config`, `user_prompt_library` और `output_dir` को उन फ़ाइलों के अनुसार बदलें। संरचना के लिए [प्रशिक्षण ट्यूटोरियल](../TUTORIAL.hi.md) और [dataloader संदर्भ](../DATALOADER.hi.md) देखें।

Domokun backends को अपने concept या photo dataset से बदलें। Multi-scale में 512px और 1024px के अलग backends, IDs और VAE cache paths रखें। `resolution_type: "area"` में megapixels का मान **0.262144** और **1.048576** होता है। `crop: false` अनुपात सुरक्षित रखता है। प्रशिक्षण config में `aspect_bucket_alignment: 32` रखें।

संयुक्त विधि में synthetic backends के लिए `is_regularisation_data: true` रखें और प्रशिक्षण backends में false करें। आधा sampling weight training डेटा और आधा regularisation को दें; दोनों हिस्सों को रिज़ॉल्यूशन में बाँटें। Probabilistic sampler इस्तेमाल होता है, सख्त बारी-बारी चयन नहीं। Repeats dataset की उपलब्धता तय करता है, sampling probability नहीं। डेटा, रिज़ॉल्यूशन, batches या distributed topology बदलने पर नया प्रशिक्षण शुरू करें।

## प्रशिक्षण assistant चुनें

Main और उदाहरण अभी **assistant v3** इस्तेमाल करते हैं। प्रशिक्षण में वह frozen रहता है, strength 1 होती है, और validation में बंद होता है। Regularisation batches का parent target **बिना adapters वाले base model की prediction** है; trainable LoRA और assistant दोनों बंद होते हैं।

[Assistant v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3) डिफ़ॉल्ट rank-64 adapter है, जिसे 512/1024/1536/2048 पर 30k updates तक प्रशिक्षित किया गया है। Presets की memory जाँच v2 के साथ हुई थी; v3 इस्तेमाल करते समय उपलब्ध VRAM जाँचें। पुराने runs दोहराने के लिए नीचे दिए override से [assistant v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2) स्पष्ट रूप से चुनें:

```json
{
  "assistant_lora_path": "SimpleTuner/Qwen-Image-2.1-training-assistant-v2",
  "assistant_lora_strength": 1.0,
  "assistant_lora_inference_strength": 0.0
}
```

Assistant के बिना प्रशिक्षण के लिए `disable_assistant_lora: true` रखें। नया assistant बनाते समय भी यही सेटिंग रखें। Downstream inference में केवल अपनी प्रशिक्षित LoRA लोड करें; assistant और REPA projector प्रशिक्षण के घटक हैं।

## सीखने और गुणवत्ता की जाँच

उदाहरण हर 250 updates पर validation और save करते हैं। 40 inference steps, true CFG 1 और seed 42 इस्तेमाल होते हैं। 24/32 GB पर 512px और बड़े presets पर 512px तथा 1024px दोनों में validation होता है। अपनी config में ये सेटिंग रखें:

```json
{
  "validation_guidance": 1.0,
  "validation_guidance_real": 1.0,
  "validation_num_inference_steps": 40,
  "validation_seed": 42,
  "validation_step_interval": 250,
  "checkpoint_step_interval": 250
}
```

Concept prompts में trigger रखें, और coherence व quality जाँचने के लिए असंबंधित subjects भी रखें। Checkpoints की तुलना में prompt, seed, decoder, resolution और guidance समान रखें। Training और output resolution अलग हैं: 512px-trained adapter को 1024px पर भी जाँचें। इन विधियों के लिए 2048px concept डेटा आवश्यक नहीं है।

[प्रयोग संग्रह](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-experiments) में assistant, regularisation, REPA, auto shift और लंबे प्रशिक्षण की तुलना है। कुछ runs खराब होने के बाद सुधरते हैं; एक कमजोर checkpoint से पूरी प्रगति तय नहीं होती। [Photo-aesthetics v3 तुलना](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-photo-aesthetics-v3) में 512px पर 50k updates वाला मॉडल 1024px पर 10k वाले से 1MP output में भी अधिक scene detail देता है; दोनों के update बजट अलग हैं।

## VAE और inference

Qwen Image 2.1 अब वैलिडेशन और अन्य VAE डिकोडिंग के लिए डिफ़ॉल्ट रूप से [Ollin का टेक्सचर-सुधारित VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) इस्तेमाल करता है। केवल डिकोडर को फ़ाइन-ट्यून किया गया है; एनकोडर अपरिवर्तित है, इसलिए मौजूदा 2.1 ट्रेनिंग लैटेंट्स संगत रहते हैं। VAE का रिविज़न बेस मॉडल से स्वतंत्र रूप से पिन किया गया है। मूल VAE के आउटपुट दोहराने के लिए `pretrained_vae_model_name_or_path: "Qwen/Qwen-Image-2.1"` सेट करें। स्पष्ट रूप से दिए गए VAE और पुराने फ़्लेवर का व्यवहार नहीं बदलता।

Standard PEFT adapters के लिए SimpleTuner में शामिल pipeline से Diffusers का Git install आवश्यक नहीं है। नीचे LoRA export पढ़कर केवल प्रशिक्षण में काम आने वाले REPA projector tensors हटाए जाते हैं। Full-frame decode होता है और assistant लोड नहीं होता:

मॉडल CPU offload से text encoder, transformer और VAE एक साथ GPU मेमोरी में नहीं रहते। पर्याप्त VRAM हो तो तेज़ repeated inference के लिए `pipe.enable_model_cpu_offload()` की जगह `pipe.to("cuda")` इस्तेमाल करें।

```python
import torch
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from simpletuner.helpers.models.qwen_image.autoencoder_21 import AutoencoderKLQwenImage21
from simpletuner.helpers.models.qwen_image.pipeline_21 import QwenImage21Pipeline
from simpletuner.helpers.models.qwen_image.transformer_21 import QwenImage21Transformer2DModel

vae = AutoencoderKLQwenImage21.from_pretrained(
    "madebyollin/texture-fix-vae-for-qwen-image-2.1",
    revision="e9f84623d22c47f8bc9fb799bc54201fa53cf80b",
    torch_dtype=torch.bfloat16,
)
model_id = "Qwen/Qwen-Image-2.1"
pipe = QwenImage21Pipeline(
    transformer=QwenImage21Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    ),
    vae=vae,
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", dtype=torch.bfloat16,
    ),
    processor=Qwen3VLProcessor.from_pretrained(model_id, subfolder="processor"),
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler"),
)
pipe.enable_model_cpu_offload()
pipe.vae.disable_tiling()
pipe.vae.enable_slicing()
weights = load_file("output/my-qwen21/pytorch_lora_weights.safetensors")
weights = {
    key: value for key, value in weights.items()
    if ".lora_A." in key or ".lora_B." in key
}
pipe.load_lora_weights(weights, adapter_name="trained")
pipe.set_adapters("trained", adapter_weights=1.0)
image = pipe(
    prompt="🟫 wearing a red scarf, standing in a snowy forest",
    width=1024, height=1024, num_inference_steps=40, true_cfg_scale=1.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]
image.save("output.png")
```

## मेमोरी और चित्र के artifacts

BF16 (`base_model_precision: "no_change"`) से शुरू करें। Quantisation से पहले batch घटाएँ या छोटा preset लें; quantisation मेमोरी बचाता है लेकिन तेज प्रशिक्षण की गारंटी नहीं देता। उदाहरण regional compilation चालू करते हैं। लंबे captions, अतिरिक्त resolutions, validation और REPA peak VRAM बदल सकते हैं। Batch बढ़ाने से पहले खाली VRAM मापें।

मेमोरी हो तो `vae_enable_tiling: false` रखें। Fixed decoder canvas-जैसी texture ठीक करता है; tiled colour seams अलग spatial-context समस्या है। Norm clipping 1.0 के लिए `grad_clip_method: "norm"` और `max_grad_norm: 1.0` रखें; यह पुराने AnyFlow pilot का प्रति-element 0.01 clipping नहीं है।

<details markdown="1">
<summary>मेमोरी माप: VAE</summary>

Qwen Image 2.1 एकल छवियों को डिकोड करते समय अनुपयोगी कालिक फीचर कैश नहीं रखता। H200 पर BF16 में एक 2048×2048 छवि को अलग से डिकोड करने पर, इससे आवंटित मेमोरी का पीक 26.87 से घटकर 15.28 GiB हुआ और आउटपुट बिल्कुल समान रहा। टाइल आधारित डिकोडिंग अधिक मेमोरी बचाती है, लेकिन स्थानिक संदर्भ सीमित होने से रंगीन जोड़ दिख सकते हैं; अनुपयोगी कैश हटाने से ये जोड़ ठीक नहीं होते।

</details>

## पुरानी और प्रयोगात्मक विधियाँ

नीचे assistant बनाने या distillation जाँचने की विधियाँ हैं। वे ऊपर की सुझाई downstream विधि से अलग हैं; पुराने pilot settings और परिणाम संदर्भ के लिए रखे गए हैं।

<details markdown="1">
<summary>AnyFlow pilot और optimizer का इतिहास</summary>

<a id="experimental-anyflow-pilot"></a>

### प्रयोगात्मक AnyFlow पायलट

तीन `qwen_image-2.1-anyflow-stage*.peft-lora` उदाहरण जाँचते हैं कि interval distillation से `🟫` जोड़ते हुए base model का व्यवहार बचाया जा सकता है या नहीं। ये प्रयोगात्मक हैं; Qwen का model card यह सिद्ध नहीं करता कि पहले की गिरावट guidance distillation के कारण हुई थी।

सभी stages में 1024px, BF16, AdamW, rank 32 और batch 4 हैं। Stage 1 में H200 पर gradient checkpointing बंद है; stages 2 और 3 लगातार दो blocks पर checkpointing इस्तेमाल करते हैं। यह ऊपर दिए सहायक REPA प्रीसेट से अलग डिस्टिलेशन प्रशिक्षण है।

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

</details>

<details markdown="1">
<summary>Assistant बनाना: online, multi-resolution और offline विधियाँ</summary>

<a id="assistant-lora"></a>

### कैप्शन से सहायक LoRA का प्रशिक्षण

`qwen_image-2.1-assistant-lora.peft-lora` L40S के लिए प्रयोगात्मक प्रारंभिक सेटिंग है: BF16, बैच 1, अंतराल 2 पर ग्रेडिएंट चेकपॉइंटिंग, rank 32 और `1e-4` पर AdamW BF16। इसमें 1,000 अपडेट का बजट है और हर 50 अपडेट पर सत्यापन तथा सेव होता है। वास्तविक प्रशिक्षण से पहले बारह परीक्षण कैप्शन को विविध कैप्शन से बदलें; इस सेटिंग के अभिसरण की पुष्टि नहीं हुई है।

`grad_clip_method: "norm"`, `max_grad_norm: 1.0`.

`distillation_method: assistant_lora` चुनें। कैप्शन बैकएंड पहले टेक्स्ट एम्बेडिंग बनाता है। हर बैच में अडैप्टर बंद करके बेस मॉडल से 40 मूल इन्फरेंस चरणों और CFG 1 पर नए लेटेंट बनाए जाते हैं। निजी पाइपलाइन उसी transformer का उपयोग करती है और VAE, processor या टेक्स्ट एन्कोडर लोड नहीं करती। फिर अडैप्टर बहाल होता है और सामान्य डीनॉइज़िंग प्रशिक्षण चलता है। अंतिम लेटेंट कैश नहीं होते। अभी केवल Qwen Image 2.1 के टेक्स्ट-से-इमेज मोड का समर्थन है।

`distillation_config.assistant_lora` में `num_inference_steps` (डिफ़ॉल्ट 40), `resolutions` (`[चौड़ाई, ऊँचाई]` की गैर-रिक्त सूची, डिफ़ॉल्ट `[[1024, 1024]]`) और `seed` (42) आते हैं। आयाम 32 के गुणज होने चाहिए। रिज़ॉल्यूशन बैच के अनुसार चक्रीय बदलते हैं और बीज हर नमूने पर आगे बढ़ते हैं, छोटे अंतिम बैच में भी। चेकपॉइंट ये काउंटर सहेजते हैं। पुनः शुरू करते समय डेटा, बैच, ग्रेडिएंट संचय या वितरित टोपोलॉजी न बदलें। ऑन-डिमांड टेक्स्ट कैश समर्थित नहीं है।

प्रवाह की जाँच के लिए 8 अपडेट और शिक्षक के 2 चरण रखें; छवियों का मूल्यांकन करने से पहले 40 चरण बहाल करें। 100, 250, 500 और 1,000 अपडेट पर समान प्रॉम्प्ट और बीज वाली बेस छवियों से तुलना करें। सहायक की उपयोगिता की जाँच अलग कॉन्सेप्ट प्रशिक्षण में भी आवश्यक है।

Qwen Image 2.1 LoRA प्रशिक्षण अब डिफ़ॉल्ट रूप से [प्रशिक्षण सहायक v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3) लोड करता है। यह प्रशिक्षण के दौरान स्थिर रहता है और वैलिडेशन में निष्क्रिय रहता है। इसे बंद करने के लिए `disable_assistant_lora: true` सेट करें। दूसरा अडैप्टर चुनने के लिए `assistant_lora_path` दें। पुराने फ़्लेवर का व्यवहार नहीं बदलता। नया सहायक प्रशिक्षित करते समय संबंधित उदाहरणों की तरह `disable_assistant_lora: true` बनाए रखें।

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

</details>
