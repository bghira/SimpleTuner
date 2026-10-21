# MOSS-Image त्वरित शुरुआत

अपनी तस्वीरों से MOSS-Image LoRA प्रशिक्षित करें। LoRA एक छोटा अडैप्टर है जो मूल चेकपॉइंट को अलग रखते हुए मॉडल को विषय या शैली सिखाता है।

`model_family: "moss_image"`, `model_flavour: "sft"` और `model_type: "lora"` चुनकर डिफ़ॉल्ट [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) से शुरू करें, जिसे चुने हुए डेटा पर परिष्कृत किया गया है। [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) के लिए `model_flavour: "base"` चुनें। दोनों में 5B इमेज Transformer, Qwen3.5-9B टेक्स्ट एन्कोडर और FLUX.2 VAE हैं। चेकपॉइंट [Apache-2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/LICENSE) के अंतर्गत जारी हैं। प्रशिक्षण में टेक्स्ट एन्कोडर फ़्रीज़ रहता है।

## VRAM प्रीसेट चुनें

GPU मेमोरी के अनुसार रेसिपी चुनें। हर स्तर 512×512 के बराबर क्षेत्र पर प्रशिक्षण करता है, चित्रों का अनुपात बनाए रखता है और rank-32 BF16 LoRA तथा REPA सक्षम रखता है।

| GPU मेमोरी | Base उदाहरण | SFT उदाहरण | प्रति बैच चित्र | `gradient_checkpointing_interval` |
| --- | --- | --- | ---: | ---: |
| 24 GB | `moss_image/base/24g/peft-lora` | `moss_image/sft/24g/peft-lora` | 1 | 1 |
| 32 GB | `moss_image/base/32g/peft-lora` | `moss_image/sft/32g/peft-lora` | 1 | 2 |
| 48 GB | `moss_image/base/48g/peft-lora` | `moss_image/sft/48g/peft-lora` | 2 | 2 |
| 80 GB | `moss_image/base/80g/peft-lora` | `moss_image/sft/80g/peft-lora` | 4 | 2 |
| 144 GB | `moss_image/base/144g/peft-lora` | `moss_image/sft/144g/peft-lora` | 8 | 2 |

डिफ़ॉल्ट `moss_image/sft/default/default.peft-lora` और `moss_image/sft/default/peft-lora` में 48 GB सेटिंग हैं। बैच आकार शुरुआती संदर्भ हैं: लंबे कैप्शन, अतिरिक्त सुविधाओं या अनुपात बकेट में बैच से कम चित्र होने पर `train_batch_size` घटाएँ। 144 GB स्तर H200 श्रेणी के GPU के लिए है।

## SimpleTuner इंस्टॉल करें

नीचे दी गई CUDA रेसिपी में Python 3.13 और NVIDIA GPU का उपयोग करें। अन्य प्लेटफ़ॉर्म और CUDA संस्करणों के लिए [इंस्टॉलेशन गाइड](../INSTALL.md) देखें।

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda,attention-cute]'
```

## चित्र और कैप्शन तैयार करें

`datasets/moss-image` बनाएँ और चित्रों के साथ उसी नाम की `.txt` कैप्शन फ़ाइलें रखें। 512px प्रशिक्षण के लिए पर्याप्त बारीकी वाले साफ़ चित्र चुनें। वर्गाकार चित्र 512×512 होता है; चौड़े और लंबे चित्र अपना अनुपात बनाए रखते हैं।

```text
datasets/moss-image/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

विषय के लिए `TOK` जैसा शब्द चुनें और उस विषय के हर कैप्शन में शामिल करें। दिख रहे विषय, रचना, प्रकाश, परिवेश और शैली का वर्णन करें। परियोजना अंग्रेज़ी से शुरुआत की सलाह देती है। चित्र में दिखता पाठ और पंक्ति-विराम ठीक से लिखें। उदाहरण: `A photograph of TOK, a small blue ceramic owl on a wooden table beside a window.`

## प्रशिक्षण कॉन्फ़िगरेशन बनाएँ

48 GB SFT रेसिपी और नमूना प्रॉम्प्ट कॉपी करें। `48g` की जगह अपना चुना स्तर रखें। Base के लिए कमांड में `moss_image/sft/48g/peft-lora` को `moss_image/base/48g/peft-lora` से बदलें।

```bash
mkdir -p config/examples/_shared datasets/moss-image
mkdir -p config/examples/moss_image/sft/48g
cp -R simpletuner/examples/moss_image/sft/48g/peft-lora config/examples/moss_image/sft/48g/
mkdir -p config/examples/_shared
cp simpletuner/examples/_shared/moss_image-prompts.json config/examples/_shared/
cp config/examples/moss_image/sft/48g/peft-lora/config.json config/config.json
```

रेसिपी में rank-32 BF16 LoRA, `1e-4` लर्निंग रेट, gradient checkpointing, REPA, 4,000 ऑप्टिमाइज़र स्टेप और 512-स्तर प्रशिक्षण है। `config/config.json` खोलकर मौजूदा कॉन्फ़िगरेशन में ये प्रविष्टियाँ बदलें:

```json
{
  "output_dir": "output/my-moss-lora",
  "validation_prompt": "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
  "user_prompt_library": null
}
```

कॉपी किया गया `dataloader.json`, `datasets/moss-image` की टेक्स्ट फ़ाइलें पढ़ता है और latent व टेक्स्ट embeddings कैश करता है। `resolution_type: "area"` के साथ `resolution: 0.262144` का अर्थ 512×512 के बराबर क्षेत्र है। आयाम 16 के गुणज में होते हैं। अतिरिक्त डेटासेट और नियमितीकरण चित्रों के लिए [डेटालोडर गाइड](../DATALOADER.md) देखें।

## प्रशिक्षण और प्रीव्यू तुलना

वर्चुअल एनवायरनमेंट सक्रिय करके SimpleTuner डायरेक्टरी में `simpletuner train` चलाएँ। पहली बार मॉडल और REPA एन्कोडर डाउनलोड होते हैं और कैश बनता है। रेसिपी हर 250 स्टेप में चेकपॉइंट और वैलिडेशन चित्र सहेजती है। विषय, रचना और बारीकी की तुलना करें तथा अपने डेटासेट के अनुसार `max_train_steps` बदलें।

```bash
simpletuner train
```

## सैंपलिंग सेटिंग

**50 स्टेप**, **CFG 3** और **1024×1024** वैलिडेशन से शुरू करें। सैंपलिंग के लिए `moss_image_validation_shift` का डिफ़ॉल्ट **5** है; प्रशिक्षण मूल बिना-शिफ़्ट ग्रिड का उपयोग करता है। `moss_image_uncond_skip_layers` का डिफ़ॉल्ट **[22]** है, जिससे नकारात्मक CFG शाखा में लेयर 22 छोड़ी जाती है। मानक CFG के लिए `[]` दें। सकारात्मक शाखा की अतिरिक्त skip-layer guidance को `validation_guidance_skip_layers`, `validation_guidance_skip_scale` तथा start/stop सेटिंग नियंत्रित करती हैं; [OPTIONS](../OPTIONS.md) देखें। Python में `use_cfg_zero_star=True` से CFG-Zero* सक्षम करें।

प्रशिक्षण अपने आप MOSS-Image का मूल noise वितरण और 1% uniform मिश्रण उपयोग करता है। वैलिडेशन sampling shift को केवल एक बार लागू करता है और अंतिम zero-noise बिंदु शामिल करता है। AnyFlow में वैलिडेशन और प्रशिक्षण रोलआउट के लिए समान shift रखें।

## सैंपलिंग के दौरान प्रीव्यू

WebUI में denoising के दौरान प्रगति चित्र देखने के लिए नीचे की सेटिंग जोड़ें। प्रीव्यू TAEF2 और अंतिम वैलिडेशन चित्र पूर्ण VAE इस्तेमाल करते हैं। CLI कॉन्फ़िगर किए गए webhook से प्रीव्यू भेज सकता है।

```json
{
  "validation_preview": true,
  "validation_preview_steps": 5
}
```

## मेमोरी और प्रशिक्षण विकल्प

रेसिपी चुने हुए VRAM स्तर के अनुसार `train_batch_size` सेट करती है। छोटे डेटासेट या लंबे कैप्शन और अतिरिक्त सुविधाओं के लिए इसे घटाएँ। `gradient_checkpointing_backend` में `torch`, `unsloth`, `torch-ffn` या `unsloth-ffn` चुनें; interval और `gradient_checkpointing_segment_stride` कवरेज नियंत्रित करते हैं। Musubi, `musubi_blocks_to_swap` से ब्लॉक CPU और GPU के बीच ले जाता है। RamTorch के लिए `ramtorch: true` तथा चयनात्मक ऑफ़लोड के लिए `ramtorch_transformer_percent` है। ऑफ़लोड में सिस्टम RAM और ट्रांसफ़र समय लगता है; एक समय में एक तरीका सेट करें। [OPTIONS](../OPTIONS.md) देखें।

CUDA इंस्टॉलेशन में CuTE attention शामिल है। प्रशिक्षण के दौरान डिफ़ॉल्ट `attention_mechanism: "kohaku-fa-auto"` Ada, Hopper और RTX Blackwell GPU पर CuTE तथा अन्य आर्किटेक्चर पर मूल SDPA चुनता है। backward के लिए सहेजे गए attention tensors ऑफ़लोड करने के लिए `gradient_checkpointing_offload_attention: true` तथा ट्रांसफ़र प्रीफ़ेच के लिए `gradient_checkpointing_offload_prefetch: true` दें। अन्य attention बैकएंड के लिए [OPTIONS](../OPTIONS.md) देखें।

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

प्रीप्रोसेसिंग में captions को मूल JSON में विस्तृत करने के लिए प्रशिक्षण कॉन्फ़िगरेशन में ये सेटिंग जोड़ें:

```json
{
  "moss_image_auto_json": true,
  "moss_image_prompt_upsample": true
}
```

डिफ़ॉल्ट caption मॉडल `Qwen/Qwen3.5-27B` है, जिसे परियोजना सुझाती है। 80 GB GPU और अतिरिक्त system RAM रखें। SimpleTuner हर प्रशिक्षण चित्र और उसके bucket dimensions से दृश्य विवरण और केंद्र से मापे गए pixel bounding boxes बनाता है, फिर text embeddings कैश करता है। Validation prompts भी विस्तृत होते हैं; खाली और नकारात्मक prompts नहीं बदलते। नया caption मॉडल लोड किए बिना मौजूदा captions को JSON में रखने के लिए केवल `moss_image_auto_json` सक्रिय करें।

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
    "OpenMOSS-Team/MOSS-Image-SFT", torch_dtype=torch.bfloat16
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

## एडिट या रेफ़रेंस LoRA ट्रेन करें

रेफ़रेंस इमेज के साथ ट्रेन करने के लिए `model_flavour: "sft-edit"` या `"base-edit"` चुनें। ये फ़्लेवर संबंधित SFT या Base checkpoint से शुरू होते हैं और आपके adapter के माध्यम से रेफ़रेंस कंडीशनिंग सीखते हैं। `simpletuner/examples/moss_image/sft-edit/default/peft-lora` या `moss_image/base-edit/default/peft-lora` से शुरू करें; इनमें प्रति बैच एक लक्ष्य इमेज, 512-स्केल ट्रेनिंग, REPA और डेटासेट वैलिडेशन शामिल हैं।

```bash
mkdir -p config/examples/_shared datasets/moss-edits datasets/moss-references
mkdir -p config/examples/moss_image/sft-edit/default
cp -R simpletuner/examples/moss_image/sft-edit/default/peft-lora config/examples/moss_image/sft-edit/default/
cp config/examples/moss_image/sft-edit/default/peft-lora/config.json config/config.json
```

वांछित आउटपुट `datasets/moss-edits/sample.png`, उसका निर्देश `datasets/moss-edits/sample.txt` और इनपुट रेफ़रेंस `datasets/moss-references/sample.png` में रखें। दोनों फ़ोल्डर में इमेज के नाम और एक्सटेंशन समान होने चाहिए। उदाहरण निर्देश: `Change the blue mug to red while keeping the background unchanged.` एडिट उदाहरणों में यह dataloader कॉन्फ़िगरेशन है:

```json
[
  {
    "id": "edits",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/moss-edits",
    "caption_strategy": "textfile",
    "resolution": 0.262144,
    "resolution_type": "area",
    "crop": false,
    "conditioning_data": [
      "references"
    ],
    "cache_dir_vae": "cache/vae/moss-image/sft-edit/targets",
    "text_embeds": "text",
    "repeats": 10
  },
  {
    "id": "references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "datasets/moss-references",
    "conditioning_type": "reference_loose",
    "resolution": 0.262144,
    "resolution_type": "area",
    "crop": false,
    "cache_dir_vae": "cache/vae/moss-image/sft-edit/references"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/moss-image/sft-edit"
  }
]
```

`reference_loose` हर रेफ़रेंस का आस्पेक्ट रेशियो बनाए रखता है और रेफ़रेंस तथा लक्ष्य के अलग आकार स्वीकार करता है। संरेखित जोड़ों के लिए `reference_strict` उपयोग करें। कई रेफ़रेंस साथ देने के लिए `conditioning_data` में और रेफ़रेंस डेटासेट जोड़ें और `conditioning_multidataset_sampling: "combined"` रखें। `"random"` हर सैंपल के लिए एक रेफ़रेंस डेटासेट चुनता है; निर्देश अलग हों तो हर रेफ़रेंस डेटासेट में संबंधित निर्देश captions रखें। लक्ष्य, रेफ़रेंस और टेक्स्ट की कैश अलग रखें। रेफ़रेंस को अतिरिक्त मेमोरी चाहिए: आवश्यकतानुसार बैच घटाएं या ऊपर दिए गए ऑफ़लोड विकल्प उपयोग करें।

वैलिडेशन जोड़ीदार डेटासेट की रेफ़रेंस इमेज और निर्देश उपयोग करता है। रेफ़रेंस पिक्सेल फ़्रोज़न Qwen3.5 vision tower से गुजरते हैं और साफ़ VAE latents इमेज विवरण देते हैं; ट्रेनिंग लॉस केवल लक्ष्य इमेज पर होता है। Caption dropout रेफ़रेंस बनाए रखता है और केवल निर्देश हटाता है। Qwen image processor अपने आप `Qwen/Qwen3.5-9B` से डाउनलोड होता है।

ट्रेनिंग के बाद एडिट adapter लोड करें और `image` में एक रेफ़रेंस इमेज या रेफ़रेंस की सूची दें। Prompt बैच के लिए हर prompt का एक रेफ़रेंस समूह रखने वाली nested सूची दें। सैंपलिंग में रेफ़रेंस K/V कैश चालू है; हर चरण में रेफ़रेंस tokens दोबारा गणना करने के लिए `cache_reference_kv=False` दें।

```python
import torch
from diffusers.utils import load_image
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline

pipe = MOSSImagePipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-Image-SFT", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("output/examples/moss_image-sft-edit.peft-lora")
image = pipe(
    "Change the blue mug to red while keeping the background unchanged.",
    image=[load_image("reference.png")],
    height=512, width=512, num_inference_steps=50, guidance_scale=3.0,
).images[0]
image.save("edited.png")
```
