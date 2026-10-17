# FLUX.3 Image Klein क्विकस्टार्ट

इस गाइड में आप अपनी इमेज से FLUX.3 Image Klein LoRA ट्रेन करेंगे। LoRA एक छोटा अडैप्टर है जो मूल मॉडल के weights अलग रखते हुए उसे कोई विषय या शैली सिखाता है।

`model_family: "flux3_image"`, `model_flavour: "klein-base"` और `model_type: "lora"` से शुरू करें। इससे [black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) लोड होता है। डाउनलोड से पहले मॉडल पेज पर एक्सेस की शर्तें स्वीकार करें और [FLUX [Kom]munity लाइसेंस](https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/LICENSE) पढ़ें।

## इंस्टॉलेशन

Python 3.13, Git और मॉडल, इमेज व कैश के लिए खाली डिस्क स्पेस चाहिए। नीचे के कमांड NVIDIA CUDA GPU के लिए हैं। अन्य प्लेटफ़ॉर्म और CUDA 13 के लिए [इंस्टॉलेशन गाइड](../INSTALL.hi.md) देखें।

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda]'
hf auth login
```

बाकी कमांड सक्रिय virtual environment के साथ `SimpleTuner` डायरेक्टरी में चलाएँ। `hf auth login` में मॉडल पढ़ने की अनुमति वाला Hugging Face token दें।

## अपनी GPU के लिए रेसिपी चुनें

GPU मेमोरी (VRAM) तय करती है कि एक साथ कितनी इमेज प्रोसेस कर सकते हैं। अपनी क्षमता के अनुसार उदाहरण चुनें। 24 और 32 GB Base रेसिपी 512 रेज़ोल्यूशन इस्तेमाल करती हैं; बड़ी रेसिपी में 1024 डेटासेट भी हैं।

| GPU मेमोरी | Base उदाहरण | 512 / 1024 पर प्रति बैच इमेज |
| --- | --- | --- |
| 24 GB | `flux3_image-24g.peft-lora` | 3 / — |
| 32 GB | `flux3_image-32g.peft-lora` | 4 / — |
| 48 GB | `flux3_image-48g.peft-lora` | 9 / 2 |
| 80 GB | `flux3_image-80g.peft-lora` | 18 / 5 |
| 144 GB | `flux3_image-144g.peft-lora` | 36 / 10 |

डिस्टिल्ड `klein` के लिए इनमें से एक रेसिपी चुनें। नीचे दिए गए कॉन्फ़िगरेशन कॉपी कमांड में उसका पूरा उदाहरण नाम रखें। प्रशिक्षण स्केल वर्गाकार चित्र के बराबर क्षेत्र बताता है; चित्रों का अनुपात बना रहता है।

| GPU मेमोरी | डिस्टिल्ड उदाहरण | प्रशिक्षण स्केल | प्रति बैच चित्र |
| --- | --- | --- | ---: |
| 24 GB | `flux3_image-klein-24g.peft-lora` | 768px | 1 |
| 80 GB | `flux3_image-klein-80g.peft-lora` | 1024px | 4 |

तालिका के बैच आकार शुरुआती मान हैं। लंबे captions और रेफ़रेंस इमेज अधिक मेमोरी लेते हैं; मेमोरी कम पड़ने पर `train_batch_size` घटाएँ। रेसिपी rank-32 LoRA, `1e-4` learning rate और REPA इस्तेमाल करती हैं, जो प्रशिक्षण के दौरान इमेज features की तुलना करता है। साथ दिए dataloaders में कॉन्सेप्ट और regularisation इमेज हैं, जो मॉडल का सामान्य इमेज ज्ञान बनाए रखने में मदद करती हैं।

## इमेज और captions तैयार करें

SimpleTuner में `datasets/my-concept` फ़ोल्डर बनाकर अपनी प्रशिक्षण इमेज रखें। हर इमेज के साथ उसी नाम की टेक्स्ट फ़ाइल में उसका caption लिखें:

```text
datasets/my-concept/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

किसी विषय की LoRA के लिए `TOK` जैसा अलग शब्द चुनें और captions में लगातार इस्तेमाल करें। विषय, स्थान और मुद्रा का वर्णन करें। उदाहरण के लिए, `image01.txt` में लिख सकते हैं:

```text
लकड़ी की मेज़ पर रखे नीले पंखों वाले छोटे सिरेमिक उल्लू TOK की तस्वीर।
```

## प्रशिक्षण कॉन्फ़िगरेशन बनाएँ

चुनी हुई रेसिपी कॉपी करें। इन कमांड में `flux3_image-48g.peft-lora` है; इस पूरे उदाहरण नाम को अपनी चुनी हुई Base या डिस्टिल्ड रेसिपी से बदलें:

```bash
mkdir -p config/examples
cp -R simpletuner/examples/flux3_image-48g.peft-lora config/examples/
cp simpletuner/examples/flux3_image-prompts.json config/examples/
cp config/examples/flux3_image-48g.peft-lora/config.json config/config.json
```

टेक्स्ट एडिटर में `config/config.json` खोलें। ऊपर की स्थानीय इमेज से पहली रन के लिए मौजूदा कॉन्फ़िगरेशन में ये मान बदलें:

```json
{
  "data_backend_config": "config/dataloader.json",
  "output_dir": "output/my-flux3-lora",
  "train_batch_size": 1
}
```

बैच आकार 1 से कम इमेज के साथ शुरुआत हो सकती है। पहली रन सफल होने के बाद पर्याप्त इमेज और खाली मेमोरी मिलने पर इसे बढ़ाएँ। `max_train_steps` प्रशिक्षण की अवधि तय करता है। `config/examples/flux3_image-prompts.json` में अपने विषय के शब्द वाले prompts डालें; इनसे प्रशिक्षण की प्रगति दिखाने वाली इमेज बनेंगी।

नीचे का JSON `config/dataloader.json` के रूप में सेव करें। यह बताता है कि इमेज कहाँ हैं, captions कैसे पढ़ने हैं और कैश कहाँ रखना है:

```json
[
  {
    "id": "my-concept",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/my-concept",
    "caption_strategy": "textfile",
    "resolution": 0.589824,
    "resolution_type": "area",
    "crop": false,
    "cache_dir_vae": "cache/vae/flux3-image/my-concept",
    "text_embeds": "text"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux3-image/my-concept"
  }
]
```

इस स्थानीय डेटासेट में `resolution: 0.589824` और `resolution_type: "area"` से 768×768 के बराबर क्षेत्र मिलता है। 80 GB डिस्टिल्ड रेसिपी के 1024 स्केल वाले डेटासेट के लिए `1.048576` रखें।

`resolution` प्रशिक्षण इमेज का आकार तय करता है; आयाम 16 के गुणज में समायोजित होते हैं। कैश दोबारा इस्तेमाल होने वाले latents और text embeddings रखता है, जिससे अगली रन जल्दी शुरू होती है। अलग मॉडल परिवारों के कैश अलग रखें। अधिक डेटासेट या regularisation इमेज के लिए [dataloader गाइड](../DATALOADER.hi.md) देखें।

## वैकल्पिक: prompt upsampler से captions विस्तार करें

छोटे प्रशिक्षण captions और validation prompts को [FLUX.3 Image Klein prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) से विस्तार करने के लिए `config/config.json` में यह सेटिंग जोड़ें:

```json
{
  "flux3_prompt_upsample": true
}
```

यह डिफ़ॉल्ट रूप से बंद है। चालू करने पर SimpleTuner अलग Qwen3.5-9B मॉडल (लगभग 19 GB weights) डाउनलोड करता है और text embeddings कैश करने से पहले positive prompts को विस्तार देता है। एडिट के लिए जोड़ी गई रेफ़रेंस इमेज इस्तेमाल होती हैं। Negative prompts, dropout के खाली captions और मूल caption फ़ाइलें वैसे ही रहती हैं।

अतिरिक्त system RAM और preprocessing समय रखें। Upsampler और text encoder बारी-बारी से GPU इस्तेमाल करते हैं; बनी embeddings प्रशिक्षण में दोबारा इस्तेमाल होती हैं। SimpleTuner expanded और मूल embeddings अलग रखता है और वही references दोबारा चुनने पर कैश परिणाम फिर इस्तेमाल करता है।

## प्रशिक्षण शुरू करें

SimpleTuner डायरेक्टरी से चलाएँ:

```bash
simpletuner train
```

पहली रन में SimpleTuner मॉडल डाउनलोड करता है, इमेज पढ़ता है और प्रशिक्षण से पहले टेक्स्ट व इमेज कैश बनाता है। Compilation से स्टार्टअप समय बढ़ सकता है। इन चरणों के दौरान प्रक्रिया चलने दें। प्रगति की इमेज और checkpoints `output/my-flux3-lora` में मिलेंगे; इमेज देखकर विषय सीखने का आकलन करें।

`klein-base` के लिए 50 validation steps और guidance 3.5 रखें। डिस्टिल्ड मॉडल के लिए `model_flavour: "klein"` चुनें; यह `black-forest-labs/flux-3-image-klein` लोड करता है और 16 steps व guidance 1 इस्तेमाल करता है। शुरू करने से पहले उस रिपॉज़िटरी की एक्सेस लें।

## एडिट और रेफ़रेंस प्रशिक्षण

दोनों वेरिएंट में रेफ़रेंस वैकल्पिक है। टार्गेट डेटासेट में वांछित एडिट और निर्देश captions होते हैं; conditioning डेटासेट में स्रोत या रेफ़रेंस इमेज होती हैं। नीचे का JSON अलग सेव करें, `data_backend_config` में उसका पथ रखें और `train_batch_size: 1` से शुरू करें:

```json
[
  {
    "id": "edits",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/edits",
    "caption_strategy": "textfile",
    "resolution": 512,
    "resolution_type": "pixel",
    "conditioning_data": ["references"],
    "cache_dir_vae": "cache/vae/flux3-image/edits",
    "text_embeds": "text"
  },
  {
    "id": "references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "datasets/references",
    "conditioning_type": "reference_loose",
    "resolution": 512,
    "resolution_type": "pixel",
    "cache_dir_vae": "cache/vae/flux3-image/references"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux3-image-edit"
  }
]
```

फ़ाइल नाम मेल खाने चाहिए: जैसे `edits/sample.png`, `edits/sample.txt` और `references/sample.png`। `reference_loose` अलग आकार और अनुपात की अनुमति देता है; `reference_strict` aligned crops के लिए है। `conditioning_data` में और IDs जोड़कर प्रति सैंपल दस रेफ़रेंस तक दे सकते हैं; सभी को साथ देने के लिए `conditioning_multidataset_sampling: "combined"` रखें। `conditioning_data` के बिना text-to-image प्रशिक्षण होता है। रेफ़रेंस, टार्गेट और अन्य मॉडल परिवारों के कैश अलग रखें।

`conditioning_multidataset_sampling: "random"` के साथ, हर प्रशिक्षण नमूना एक संदर्भ डेटासेट और उसके कैप्शन का उपयोग करता है। हर संदर्भ डेटासेट में `caption_strategy: "textfile"` सेट करें और संपादन निर्देश संबंधित `.txt` फ़ाइल में लिखें, जैसे `references/sample.txt`।

## अपनी LoRA से इमेज बनाएँ

वह checkpoint डायरेक्टरी खोजें जिसमें `pytorch_lora_weights.safetensors` है। नीचे `output/my-flux3-lora/checkpoint-1000` को उससे बदलें और वही मॉडल वेरिएंट इस्तेमाल करें जिस पर ट्रेन किया था। यह उदाहरण base मॉडल और CFG-Zero-Star guidance इस्तेमाल करता है:

```python
import torch
from simpletuner.helpers.models.flux3_image.pipeline import Flux3ImagePipeline

pipe = Flux3ImagePipeline.from_pretrained(
    "black-forest-labs/flux-3-image-klein-base", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("output/my-flux3-lora/checkpoint-1000")
image = pipe(
    "A photo of TOK, a small ceramic owl, sitting on a wooden shelf",
    num_inference_steps=50,
    guidance_scale=3.5,
    use_cfg_zero_star=True,
).images[0]
image.save("my-flux3-image.png")
```

## सहायक LoRA

`model_flavour: "klein"` के साथ LoRA प्रशिक्षण अपने आप [प्रकाशित प्रशिक्षण सहायक](https://huggingface.co/SimpleTuner/flux-3-klein-image-training-assistant) को फ़्रीज़ किए गए अडैप्टर के रूप में लोड करता है। प्रशिक्षण में `assistant_lora_strength` का डिफ़ॉल्ट `1.0` और validation में `assistant_lora_inference_strength` का डिफ़ॉल्ट `0.0` है। इसे बंद करने के लिए `disable_assistant_lora: true` रखें। दूसरा अडैप्टर इस्तेमाल करने के लिए `assistant_lora_path` और ज़रूरत हो तो `assistant_lora_weight_name` दें।

अपना सहायक प्रशिक्षित करने के लिए `flux3_image-assistant-lora.peft-lora` चुनें। यह रेसिपी मौजूदा सहायक को बंद करती है और मॉडल की बनाई samples से सीखती है। इसके बाद अपनी डिस्टिल्ड LoRA कॉन्फ़िगरेशन में `assistant_lora_path` को सहेजे गए अडैप्टर फ़ोल्डर पर रखें।

अधिक सेटिंग्स [options reference](../OPTIONS.hi.md) में हैं। TwinFlow के लिए `twinflow_enabled: true` और `use_ema: true` दोनों रखें; EMA इस प्रशिक्षण विधि का teacher मॉडल बनाए रखता है।
