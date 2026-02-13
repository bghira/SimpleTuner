# T-LoRA (Timestep-dependent LoRA)

## पृष्ठभूमि

मानक LoRA फाइन-ट्यूनिंग सभी डिफ्यूज़न टाइमस्टेप्स पर समान रूप से एक निश्चित लो-रैंक अनुकूलन लागू करती है। जब प्रशिक्षण डेटा सीमित होता है (विशेष रूप से एकल-छवि अनुकूलन), तो यह ओवरफिटिंग की ओर ले जाता है — मॉडल उच्च-नॉइज़ टाइमस्टेप्स पर नॉइज़ पैटर्न याद कर लेता है जहाँ बहुत कम सिमेंटिक जानकारी मौजूद होती है।

**T-LoRA** ([Soboleva et al., 2025](https://arxiv.org/abs/2507.05964)) इस समस्या को वर्तमान डिफ्यूज़न टाइमस्टेप के आधार पर सक्रिय LoRA रैंक की संख्या को गतिशील रूप से समायोजित करके हल करता है:

- **उच्च नॉइज़** (शुरुआती डीनॉइज़िंग, $t \to T$): कम रैंक सक्रिय होते हैं, जिससे मॉडल को गैर-सूचनात्मक नॉइज़ पैटर्न याद करने से रोका जा सके।
- **कम नॉइज़** (बाद की डीनॉइज़िंग, $t \to 0$): अधिक रैंक सक्रिय होते हैं, जिससे मॉडल सूक्ष्म कॉन्सेप्ट विवरण को कैप्चर कर सके।

SimpleTuner का T-LoRA समर्थन [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) लाइब्रेरी पर बना है और इसके लिए एक LyCORIS संस्करण आवश्यक है जिसमें `lycoris.modules.tlora` मॉड्यूल शामिल हो।

> **प्रयोगात्मक:** वीडियो मॉडल के साथ T-LoRA अपेक्षाकृत कमज़ोर परिणाम दे सकता है क्योंकि टेम्पोरल कम्प्रेशन फ्रेम को टाइमस्टेप सीमाओं के पार मिला देता है।

## त्वरित सेटअप

### 1. अपना प्रशिक्षण कॉन्फ़िग सेट करें

अपने `config.json` में, एक अलग T-LoRA कॉन्फ़िग फ़ाइल के साथ LyCORIS का उपयोग करें:

```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "lycoris_config": "config/lycoris_tlora.json",
    "validation_lycoris_strength": 1.0
}
```

### 2. LyCORIS T-LoRA कॉन्फ़िग बनाएँ

`config/lycoris_tlora.json` बनाएँ:

```json
{
    "algo": "tlora",
    "multiplier": 1.0,
    "linear_dim": 64,
    "linear_alpha": 32,
    "apply_preset": {
        "target_module": ["Attention", "FeedForward"]
    }
}
```

प्रशिक्षण शुरू करने के लिए बस इतना ही चाहिए। नीचे दिए गए अनुभाग वैकल्पिक ट्यूनिंग और इन्फ़रेंस को कवर करते हैं।

## कॉन्फ़िगरेशन संदर्भ

### आवश्यक फ़ील्ड

| फ़ील्ड | प्रकार | विवरण |
|-------|------|-------------|
| `algo` | string | `"tlora"` होना चाहिए |
| `multiplier` | float | LoRA शक्ति गुणक। `1.0` पर रखें जब तक आपको पता न हो कि आप क्या कर रहे हैं |
| `linear_dim` | int | LoRA रैंक। यह मास्किंग शेड्यूल में `max_rank` बन जाता है |
| `linear_alpha` | int | LoRA स्केलिंग फ़ैक्टर (`tlora_alpha` से अलग) |

### वैकल्पिक फ़ील्ड

| फ़ील्ड | प्रकार | डिफ़ॉल्ट | विवरण |
|-------|------|---------|-------------|
| `tlora_min_rank` | int | `1` | उच्चतम नॉइज़ स्तर पर न्यूनतम सक्रिय रैंक |
| `tlora_alpha` | float | `1.0` | मास्किंग शेड्यूल एक्सपोनेंट। `1.0` रैखिक है; `1.0` से ऊपर के मान अधिक क्षमता सूक्ष्म-विवरण चरणों की ओर स्थानांतरित करते हैं |
| `apply_preset` | object | — | `target_module` और `module_algo_map` के माध्यम से मॉड्यूल टार्गेटिंग |

### मॉडल-विशिष्ट मॉड्यूल लक्ष्य

अधिकांश मॉडलों के लिए सामान्य `["Attention", "FeedForward"]` लक्ष्य काम करते हैं। Flux 2 (Klein) के लिए, कस्टम क्लास नामों का उपयोग करें:

```json
{
    "algo": "tlora",
    "multiplier": 1.0,
    "linear_dim": 64,
    "linear_alpha": 32,
    "apply_preset": {
        "target_module": [
            "Flux2Attention", "Flux2FeedForward", "Flux2ParallelSelfAttention"
        ]
    }
}
```

प्रति-मॉडल मॉड्यूल लक्ष्यों की पूरी सूची के लिए [LyCORIS दस्तावेज़ीकरण](../LYCORIS.md) देखें।

## ट्यूनिंग नॉब

### `linear_dim` (रैंक)

उच्च रैंक = अधिक पैरामीटर और अभिव्यक्ति क्षमता, लेकिन सीमित डेटा के साथ ओवरफिटिंग की अधिक संभावना। मूल T-LoRA पेपर SDXL एकल-छवि अनुकूलन के लिए रैंक 64 का उपयोग करता है।

### `tlora_min_rank`

सबसे अधिक नॉइज़ वाले टाइमस्टेप पर रैंक सक्रियण की न्यूनतम सीमा को नियंत्रित करता है। इसे बढ़ाने से मॉडल मोटे ढाँचे को सीख पाता है लेकिन ओवरफिटिंग लाभ कम हो जाता है। डिफ़ॉल्ट `1` से शुरू करें और केवल तभी बढ़ाएँ जब कन्वर्जेन्स बहुत धीमा हो।

### `tlora_alpha` (शेड्यूल एक्सपोनेंट)

मास्किंग शेड्यूल के वक्र आकार को नियंत्रित करता है:

- `1.0` — `min_rank` और `max_rank` के बीच रैखिक इंटरपोलेशन
- `> 1.0` — उच्च नॉइज़ पर अधिक आक्रामक मास्किंग; अधिकांश रैंक केवल डीनॉइज़िंग के अंत के पास ही सक्रिय होते हैं
- `< 1.0` — कोमल मास्किंग; रैंक पहले सक्रिय होते हैं

<details>
<summary>शेड्यूल विज़ुअलाइज़ेशन (रैंक बनाम टाइमस्टेप)</summary>

`linear_dim=64`, `tlora_min_rank=1` के साथ, 1000-स्टेप शेड्यूलर के लिए:

```
alpha=1.0 (linear):
  t=0   (clean)  → 64 active ranks
  t=250 (25%)    → 48 active ranks
  t=500 (50%)    → 32 active ranks
  t=750 (75%)    → 16 active ranks
  t=999 (noise)  →  1 active rank

alpha=2.0 (quadratic — biased toward detail):
  t=0   (clean)  → 64 active ranks
  t=250 (25%)    → 60 active ranks
  t=500 (50%)    → 48 active ranks
  t=750 (75%)    → 20 active ranks
  t=999 (noise)  →  1 active rank

alpha=0.5 (sqrt — biased toward structure):
  t=0   (clean)  → 64 active ranks
  t=250 (25%)    → 55 active ranks
  t=500 (50%)    → 46 active ranks
  t=750 (75%)    → 33 active ranks
  t=999 (noise)  →  1 active rank
```

</details>

## SimpleTuner पाइपलाइनों के साथ इन्फ़रेंस

SimpleTuner की वेंडर्ड पाइपलाइनों में अंतर्निहित T-LoRA समर्थन है। वैलिडेशन के दौरान, प्रशिक्षण के मास्किंग पैरामीटर प्रत्येक डीनॉइज़िंग चरण पर स्वचालित रूप से पुन: उपयोग किए जाते हैं — किसी अतिरिक्त कॉन्फ़िगरेशन की आवश्यकता नहीं है।

प्रशिक्षण के बाहर स्वतंत्र इन्फ़रेंस के लिए, आप सीधे SimpleTuner की पाइपलाइन आयात कर सकते हैं और `_tlora_config` एट्रिब्यूट सेट कर सकते हैं। यह सुनिश्चित करता है कि प्रति-चरण मास्किंग उससे मेल खाती है जिसके साथ मॉडल को प्रशिक्षित किया गया था।

### SDXL उदाहरण

```py
import torch
from lycoris import create_lycoris_from_weights

# Use SimpleTuner's vendored SDXL pipeline (has T-LoRA support built in)
from simpletuner.helpers.models.sdxl.pipeline import StableDiffusionXLPipeline
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
dtype = torch.bfloat16
device = "cuda"

# Load pipeline components
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)

# Load and apply LyCORIS T-LoRA weights
lora_path = "path/to/pytorch_lora_weights.safetensors"
wrapper, _ = create_lycoris_from_weights(1.0, lora_path, unet)
wrapper.merge_to()

unet.to(device)

pipe = StableDiffusionXLPipeline(
    scheduler=scheduler,
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    unet=unet,
)

# Enable T-LoRA inference masking — must match training config
pipe._tlora_config = {
    "max_rank": 64,      # linear_dim from your lycoris config
    "min_rank": 1,       # tlora_min_rank (default 1)
    "alpha": 1.0,        # tlora_alpha (default 1.0)
}

with torch.inference_mode():
    image = pipe(
        prompt="a sks dog riding a surfboard",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=5.0,
    ).images[0]

image.save("tlora_output.png")
```

### Flux उदाहरण

```py
import torch
from lycoris import create_lycoris_from_weights

# Use SimpleTuner's vendored Flux pipeline (has T-LoRA support built in)
from simpletuner.helpers.models.flux.pipeline import FluxPipeline
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16
device = "cuda"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2")
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype)

# Load and apply LyCORIS T-LoRA weights
lora_path = "path/to/pytorch_lora_weights.safetensors"
wrapper, _ = create_lycoris_from_weights(1.0, lora_path, transformer)
wrapper.merge_to()

transformer.to(device)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
)

# Enable T-LoRA inference masking
pipe._tlora_config = {
    "max_rank": 64,
    "min_rank": 1,
    "alpha": 1.0,
}

with torch.inference_mode():
    image = pipe(
        prompt="a sks dog riding a surfboard",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=3.5,
    ).images[0]

image.save("tlora_flux_output.png")
```

> **नोट:** आपको SimpleTuner की वेंडर्ड पाइपलाइन (जैसे `simpletuner.helpers.models.flux.pipeline.FluxPipeline`) का उपयोग करना होगा, स्टॉक Diffusers पाइपलाइन का नहीं। केवल वेंडर्ड पाइपलाइनों में प्रति-चरण T-LoRA मास्किंग लॉजिक होता है।

### केवल `merge_to()` का उपयोग करके मास्किंग क्यों नहीं छोड़ सकते?

`merge_to()` LoRA वेट को बेस मॉडल में स्थायी रूप से समाहित कर देता है — यह आवश्यक है ताकि फ़ॉरवर्ड पास के दौरान LoRA पैरामीटर सक्रिय रहें। हालाँकि, T-LoRA को टाइमस्टेप-निर्भर रैंक मास्किंग के साथ **प्रशिक्षित** किया गया था: नॉइज़ स्तर के आधार पर कुछ रैंक शून्य कर दिए गए थे। इन्फ़रेंस के दौरान उसी मास्किंग को पुन: लागू किए बिना, सभी रैंक हर टाइमस्टेप पर सक्रिय हो जाते हैं, जिससे अति-संतृप्त या जली हुई दिखने वाली छवियाँ बनती हैं।

पाइपलाइन पर `_tlora_config` सेट करना डीनॉइज़िंग लूप को प्रत्येक मॉडल फ़ॉरवर्ड पास से पहले सही मास्क लागू करने और बाद में उसे साफ़ करने का निर्देश देता है।

<details>
<summary>मास्किंग आंतरिक रूप से कैसे काम करती है</summary>

प्रत्येक डीनॉइज़िंग चरण पर, पाइपलाइन यह कॉल करती है:

```python
from simpletuner.helpers.training.lycoris import apply_tlora_inference_mask, clear_tlora_mask

_tlora_cfg = getattr(self, "_tlora_config", None)
if _tlora_cfg:
    apply_tlora_inference_mask(
        timestep=int(t),
        max_timestep=self.scheduler.config.num_train_timesteps,
        max_rank=_tlora_cfg["max_rank"],
        min_rank=_tlora_cfg["min_rank"],
        alpha=_tlora_cfg["alpha"],
    )
try:
    noise_pred = self.unet(...)  # or self.transformer(...)
finally:
    if _tlora_cfg:
        clear_tlora_mask()
```

`apply_tlora_inference_mask` निम्न सूत्र का उपयोग करके `(1, max_rank)` आकार का एक बाइनरी मास्क गणना करता है:

$$r = \left\lfloor\left(\frac{T - t}{T}\right)^\alpha \cdot (R_{\max} - R_{\min})\right\rfloor + R_{\min}$$

जहाँ $T$ अधिकतम शेड्यूलर टाइमस्टेप है, $R_{\max}$ `linear_dim` है, और $R_{\min}$ `tlora_min_rank` है। मास्क के पहले $r$ तत्वों को `1.0` पर और शेष को `0.0` पर सेट किया जाता है। फिर यह मास्क LyCORIS के `set_timestep_mask()` के माध्यम से सभी T-LoRA मॉड्यूल पर वैश्विक रूप से सेट किया जाता है।

फ़ॉरवर्ड पास पूरा होने के बाद, `clear_tlora_mask()` मास्क स्थिति को हटा देता है ताकि यह बाद के ऑपरेशनों में लीक न हो।

</details>

<details>
<summary>वैलिडेशन के दौरान SimpleTuner कॉन्फ़िग कैसे पास करता है</summary>

प्रशिक्षण के दौरान, T-LoRA कॉन्फ़िग डिक्ट (`max_rank`, `min_rank`, `alpha`) Accelerator ऑब्जेक्ट पर संग्रहीत होती है। जब वैलिडेशन चलता है, तो `validation.py` इस कॉन्फ़िग को पाइपलाइन पर कॉपी करता है:

```python
# setup_pipeline()
if getattr(self.accelerator, "_tlora_active", False):
    self.model.pipeline._tlora_config = self.accelerator._tlora_config

# clean_pipeline()
if hasattr(self.model.pipeline, "_tlora_config"):
    del self.model.pipeline._tlora_config
```

यह पूरी तरह से स्वचालित है — वैलिडेशन छवियों के लिए सही मास्किंग का उपयोग करने हेतु किसी उपयोगकर्ता कॉन्फ़िगरेशन की आवश्यकता नहीं है।

</details>

## अपस्ट्रीम: T-LoRA पेपर

<details>
<summary>पेपर विवरण और एल्गोरिथम</summary>

**T-LoRA: Single Image Diffusion Model Customization Without Overfitting**
Vera Soboleva, Aibek Alanov, Andrey Kuznetsov, Konstantin Sobolev
[arXiv:2507.05964](https://arxiv.org/abs/2507.05964) — AAAI 2026 में स्वीकृत

यह पेपर दो पूरक नवाचार प्रस्तुत करता है:

### 1. टाइमस्टेप-निर्भर रैंक मास्किंग

मुख्य अंतर्दृष्टि यह है कि उच्च डिफ्यूज़न टाइमस्टेप्स (अधिक नॉइज़ वाले इनपुट) निम्न टाइमस्टेप्स की तुलना में ओवरफिटिंग के प्रति अधिक संवेदनशील होते हैं। उच्च नॉइज़ पर, लेटेंट में अधिकतर रैंडम नॉइज़ होता है जिसमें बहुत कम सिमेंटिक सिग्नल होता है — इस पर फुल-रैंक एडेप्टर को प्रशिक्षित करना मॉडल को लक्ष्य कॉन्सेप्ट सीखने के बजाय नॉइज़ पैटर्न याद करना सिखाता है।

T-LoRA इसे एक गतिशील मास्किंग शेड्यूल के साथ संबोधित करता है जो वर्तमान टाइमस्टेप के आधार पर सक्रिय LoRA रैंक को प्रतिबंधित करता है।

### 2. ऑर्थोगोनल वेट पैरामीटराइज़ेशन (वैकल्पिक)

पेपर मूल मॉडल वेट के SVD विघटन के माध्यम से LoRA वेट को इनिशियलाइज़ करने का भी प्रस्ताव करता है, जो एक रेगुलराइज़ेशन लॉस के माध्यम से ऑर्थोगोनैलिटी लागू करता है। यह एडेप्टर घटकों के बीच स्वतंत्रता सुनिश्चित करता है।

SimpleTuner का LyCORIS एकीकरण टाइमस्टेप मास्किंग घटक पर केंद्रित है, जो ओवरफिटिंग में कमी का प्राथमिक चालक है। ऑर्थोगोनल इनिशियलाइज़ेशन स्वतंत्र T-LoRA कार्यान्वयन का हिस्सा है लेकिन वर्तमान में LyCORIS `tlora` एल्गोरिथम द्वारा उपयोग नहीं किया जाता है।

### संदर्भ

```bibtex
@misc{soboleva2025tlorasingleimagediffusion,
      title={T-LoRA: Single Image Diffusion Model Customization Without Overfitting},
      author={Vera Soboleva and Aibek Alanov and Andrey Kuznetsov and Konstantin Sobolev},
      year={2025},
      eprint={2507.05964},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05964},
}
```

</details>

## सामान्य गलतियाँ

- **इन्फ़रेंस के दौरान `_tlora_config` भूल गए:** छवियाँ अति-संतृप्त या जली हुई दिखती हैं। प्रशिक्षित मास्किंग शेड्यूल का पालन करने के बजाय सभी रैंक हर टाइमस्टेप पर सक्रिय हो जाते हैं।
- **स्टॉक Diffusers पाइपलाइन का उपयोग:** स्टॉक पाइपलाइनों में T-LoRA मास्किंग लॉजिक नहीं होता। आपको SimpleTuner की वेंडर्ड पाइपलाइनों का उपयोग करना होगा।
- **`linear_dim` बेमेल:** `_tlora_config` में `max_rank` प्रशिक्षण के दौरान उपयोग किए गए `linear_dim` से मेल खाना चाहिए, अन्यथा मास्क आयाम गलत होंगे।
- **वीडियो मॉडल:** टेम्पोरल कम्प्रेशन फ्रेम को टाइमस्टेप सीमाओं के पार मिला देता है, जो टाइमस्टेप-निर्भर मास्किंग सिग्नल को कमज़ोर कर सकता है। परिणाम अपेक्षाकृत कमज़ोर हो सकते हैं।
- **SDXL + FeedForward मॉड्यूल:** SDXL पर LyCORIS के साथ FeedForward मॉड्यूल को प्रशिक्षित करने से NaN लॉस हो सकता है — यह एक सामान्य LyCORIS समस्या है, T-LoRA के लिए विशिष्ट नहीं। विवरण के लिए [LyCORIS दस्तावेज़ीकरण](../LYCORIS.md#potential-problems) देखें।
