# MOSS-Image त्वरित शुरुआत

अपनी तस्वीरों से MOSS-Image LoRA प्रशिक्षित करें। LoRA एक छोटा अडैप्टर है जो मूल चेकपॉइंट को अलग रखते हुए मॉडल को विषय या शैली सिखाता है।

[MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) के लिए `model_family: "moss_image"`, `model_flavour: "base"` और `model_type: "lora"` चुनें। चुने हुए डेटा पर परिष्कृत [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) के लिए `model_flavour: "sft"` चुनें। दोनों में 5B इमेज Transformer, Qwen3.5-9B टेक्स्ट एन्कोडर और FLUX.2 VAE हैं। चेकपॉइंट [Apache-2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/LICENSE) के अंतर्गत जारी हैं। प्रशिक्षण में टेक्स्ट एन्कोडर फ़्रीज़ रहता है।

## SimpleTuner इंस्टॉल करें

नीचे दी गई CUDA रेसिपी में Python 3.13 और NVIDIA GPU का उपयोग करें। 48 GB GPU शुरुआती संदर्भ है; कैप्शन की लंबाई, बैच आकार और अतिरिक्त सुविधाएँ मेमोरी को प्रभावित करती हैं। अन्य प्लेटफ़ॉर्म और CUDA संस्करणों के लिए [इंस्टॉलेशन गाइड](../INSTALL.md) देखें।

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda]'
```

## चित्र और कैप्शन तैयार करें

`datasets/moss-image` बनाएँ और चित्रों के साथ उसी नाम की `.txt` कैप्शन फ़ाइलें रखें। लगभग एक मेगापिक्सेल प्रशिक्षण के लिए पर्याप्त बारीकी वाले साफ़ चित्र चुनें। वर्गाकार चित्र 1024×1024 होता है; चौड़े और लंबे चित्र अपना अनुपात बनाए रखते हैं।

```text
datasets/moss-image/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

विषय के लिए `TOK` जैसा शब्द चुनें और उस विषय के हर कैप्शन में शामिल करें। दिख रहे विषय, रचना, प्रकाश, परिवेश और शैली का वर्णन करें। परियोजना अंग्रेज़ी से शुरुआत की सलाह देती है। चित्र में दिखता पाठ और पंक्ति-विराम ठीक से लिखें। उदाहरण: `A photograph of TOK, a small blue ceramic owl on a wooden table beside a window.`

## प्रशिक्षण कॉन्फ़िगरेशन बनाएँ

Base रेसिपी और नमूना प्रॉम्प्ट कॉपी करें। SFT से शुरू करने के लिए कमांड में `moss_image.peft-lora` की जगह `moss_image-sft.peft-lora` रखें।

```bash
mkdir -p config/examples datasets/moss-image
cp -R simpletuner/examples/moss_image.peft-lora config/examples/
cp simpletuner/examples/moss_image-prompts.json config/examples/
cp config/examples/moss_image.peft-lora/config.json config/config.json
```

रेसिपी में rank-32 BF16 LoRA, `1e-4` लर्निंग रेट, gradient checkpointing, REPA, 4,000 ऑप्टिमाइज़र स्टेप और 1024-स्तर प्रशिक्षण है। `config/config.json` खोलकर मौजूदा कॉन्फ़िगरेशन में ये प्रविष्टियाँ बदलें:

```json
{
  "output_dir": "output/my-moss-lora",
  "train_batch_size": 1,
  "validation_prompt": "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
  "user_prompt_library": null
}
```

कॉपी किया गया `dataloader.json`, `datasets/moss-image` की टेक्स्ट फ़ाइलें पढ़ता है और latent व टेक्स्ट embeddings कैश करता है। `resolution_type: "area"` के साथ `resolution: 1.048576` का अर्थ 1024×1024 के बराबर क्षेत्र है। आयाम 16 के गुणज में होते हैं। अतिरिक्त डेटासेट और नियमितीकरण चित्रों के लिए [डेटालोडर गाइड](../DATALOADER.md) देखें।

## प्रशिक्षण और प्रीव्यू तुलना

वर्चुअल एनवायरनमेंट सक्रिय करके SimpleTuner डायरेक्टरी में `simpletuner train` चलाएँ। पहली बार मॉडल और REPA एन्कोडर डाउनलोड होते हैं और कैश बनता है। रेसिपी हर 250 स्टेप में चेकपॉइंट और वैलिडेशन चित्र सहेजती है। विषय, रचना और बारीकी की तुलना करें तथा अपने डेटासेट के अनुसार `max_train_steps` बदलें।

```bash
simpletuner train
```

## सैंपलिंग सेटिंग

**50 स्टेप**, **CFG 3** और **1024×1024** वैलिडेशन से शुरू करें। सैंपलिंग के लिए `moss_image_validation_shift` का डिफ़ॉल्ट **5** है; प्रशिक्षण मूल बिना-शिफ़्ट ग्रिड का उपयोग करता है। `moss_image_uncond_skip_layers` का डिफ़ॉल्ट **[22]** है, जिससे नकारात्मक CFG शाखा में लेयर 22 छोड़ी जाती है। मानक CFG के लिए `[]` दें। सकारात्मक शाखा की अतिरिक्त skip-layer guidance को `validation_guidance_skip_layers`, `validation_guidance_skip_scale` तथा start/stop सेटिंग नियंत्रित करती हैं; [OPTIONS](../OPTIONS.md) देखें। Python में `use_cfg_zero_star=True` से CFG-Zero* सक्षम करें।

## सैंपलिंग के दौरान प्रीव्यू

WebUI में denoising के दौरान प्रगति चित्र देखने के लिए नीचे की सेटिंग जोड़ें। प्रीव्यू TAEF2 और अंतिम वैलिडेशन चित्र पूर्ण VAE इस्तेमाल करते हैं। CLI कॉन्फ़िगर किए गए webhook से प्रीव्यू भेज सकता है।

```json
{
  "validation_preview": true,
  "validation_preview_steps": 5
}
```

## मेमोरी और प्रशिक्षण विकल्प

पहली बार `train_batch_size: 1` रखें। पर्याप्त चित्र और GPU मेमोरी मिलने पर इसे बढ़ाएँ। `gradient_checkpointing_backend` में `torch`, `unsloth`, `torch-ffn` या `unsloth-ffn` चुनें; interval और `gradient_checkpointing_segment_stride` कवरेज नियंत्रित करते हैं। Musubi, `musubi_blocks_to_swap` से ब्लॉक CPU और GPU के बीच ले जाता है। RamTorch के लिए `ramtorch: true` तथा चयनात्मक ऑफ़लोड के लिए `ramtorch_transformer_percent` है। ऑफ़लोड में सिस्टम RAM और ट्रांसफ़र समय लगता है; एक समय में एक तरीका सेट करें। [OPTIONS](../OPTIONS.md) देखें।

NVIDIA GPU के लिए वैकल्पिक CuTE attention निर्भरताएँ `pip install -e '.[cuda,attention-cute]'` से इंस्टॉल करें, फिर `attention_mechanism: "kohaku-fa"` चुनें। साझा चयनकर्ता अन्य इंस्टॉल किए गए बैकएंड भी स्वीकार करता है। backward के लिए सहेजे गए attention tensors ऑफ़लोड करने के लिए `gradient_checkpointing_offload_attention: true` तथा ट्रांसफ़र प्रीफ़ेच के लिए `gradient_checkpointing_offload_prefetch: true` दें।

रेसिपी [REPA](../experimental/IMAGE_REPA.md) सक्षम करती है। बाद की लेयर की इमेज विशेषताओं को पहले की लेयर से मिलाने के लिए [LayerSync](../experimental/LAYERSYNC.md) जोड़ें:

```json
{
  "layersync_enabled": true,
  "layersync_student_block": 20,
  "layersync_teacher_block": 8,
  "layersync_lambda": 0.2
}
```

पोस्ट-ट्रेनिंग के लिए [AnyFlow](../quickstart/ANYFLOW.md) या [TwinFlow](../distillation/TWINFLOW.md) गाइड देखें। AnyFlow में `lora_dropout: 0.0` रखें। Python में इसका LoRA लोड करते समय प्रशिक्षण सेटिंग `load_lora_weights` को दें, जैसे `flowmap_deltatime_type="r", flowmap_gate_value=0.25`।

## प्रॉम्प्ट सुधार और लेआउट

MOSS-Image विस्तृत कैप्शन, टैग और JSON लेआउट स्वीकार करता है। परियोजना की [प्रॉम्प्ट सुधार skill](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer) इन्हें लिखने में एजेंट की सहायता करती है। तैयार टेक्स्ट को कैप्शन या जनरेशन प्रॉम्प्ट की तरह उपयोग करें। [परियोजना गाइड](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md) में कैप्शन की सलाह है।

लेआउट दृश्य, पाठ और वस्तुओं की स्थिति बताता है। निर्देशांक पिक्सेल में `[y_min, x_min, y_max, x_max]` हैं, जिनका मूल कैनवास के केंद्र में है। 1024×1024 पर सीमा −512 से +512 है। `json.dumps(layout, ensure_ascii=False)` से स्ट्रिंग बनाकर प्रॉम्प्ट दें और लेआउट के समान जनरेशन आयाम रखें।

```json
{
  "summary": {"caption": "A cream travel poster with the title EXPLORE above a blue mountain."},
  "items": [
    {"label": "text", "text": "EXPLORE", "caption": "Large navy title", "bbox": [-416, -384, -288, 384]},
    {"label": "object", "caption": "Blue mountain", "bbox": [-192, -288, 384, 288]}
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

## अपने LoRA से चित्र बनाएँ

अंतिम अडैप्टर आउटपुट डायरेक्टरी में `pytorch_lora_weights.safetensors` है। प्रशिक्षण वाले Base या SFT चेकपॉइंट के साथ इसे लोड करें। प्रॉम्प्ट में विषय का शब्द और इच्छित दृश्य रखें:

```python
import torch
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline

pipe = MOSSImagePipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-Image-Base", torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("output/my-moss-lora")
pipe.to("cuda")
image = pipe(
    "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
    height=1024, width=1024, num_inference_steps=50, guidance_scale=3.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("my-moss-image.png")
```
