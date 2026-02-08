# LyCORIS

## पृष्ठभूमि

[LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) पैरामीटर‑एफ़िशिएंट फाइन‑ट्यूनिंग (PEFT) तरीकों का एक व्यापक सूट है, जो कम VRAM में मॉडल को फाइनट्यून करने देता है और छोटे, वितरित करने योग्य वेट्स बनाता है।

## LyCORIS का उपयोग

LyCORIS उपयोग करने के लिए, `--lora_type=lycoris` सेट करें और फिर `--lycoris_config=config/lycoris_config.json` सेट करें, जहाँ `config/lycoris_config.json` आपकी LyCORIS कॉन्फ़िगरेशन फ़ाइल का स्थान है।

यह आपके `config.json` में जाएगा:
```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "lycoris_config": "config/lycoris_config.json",
    "validation_lycoris_strength": 1.0,
    ...the rest of your settings...
}
```


LyCORIS कॉन्फ़िगरेशन फ़ाइल का फ़ॉर्मैट इस प्रकार है:

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 10,
    "apply_preset": {
        "target_module": [
            "Attention",
            "FeedForward"
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 10
            },
            "FeedForward": {
                "factor": 4
            }
        }
    }
}
```

### फ़ील्ड्स

वैकल्पिक फ़ील्ड्स:
- LycorisNetwork.apply_preset के लिए apply_preset
- चुने गए algorithm के लिए विशिष्ट keyword arguments, अंत में

अनिवार्य फ़ील्ड्स:
- multiplier, जो 1.0 ही होना चाहिए जब तक आपको अपेक्षित परिणामों की स्पष्ट जानकारी न हो
- linear_dim
- linear_alpha

LyCORIS के बारे में अधिक जानकारी के लिए, [लाइब्रेरी के दस्तावेज़](https://github.com/KohakuBlueleaf/LyCORIS/tree/main/docs) देखें।

### Flux 2 (Klein) मॉड्यूल टारगेट

Flux 2 मॉडल generic `Attention` और `FeedForward` नामों के बजाय कस्टम मॉड्यूल क्लासेस का उपयोग करते हैं। Flux 2 LoKR config में निम्नलिखित को टारगेट करें:

- `Flux2Attention` — डबल-स्ट्रीम अटेंशन ब्लॉक
- `Flux2FeedForward` — डबल-स्ट्रीम फीडफॉरवर्ड ब्लॉक
- `Flux2ParallelSelfAttention` — सिंगल-स्ट्रीम पैरेलल अटेंशन+फीडफॉरवर्ड ब्लॉक (फ्यूज्ड QKV और MLP प्रोजेक्शन)

`Flux2ParallelSelfAttention` को शामिल करने से सिंगल-स्ट्रीम ब्लॉक भी ट्रेन होते हैं, जो convergence में सुधार कर सकता है लेकिन overfitting का जोखिम बढ़ सकता है। यदि Flux 2 पर LyCORIS LoKR को converge करने में कठिनाई हो रही है, तो इस टारगेट को जोड़ने की सिफारिश की जाती है।

Flux 2 LoKR config का उदाहरण:

```json
{
    "bypass_mode": true,
    "algo": "lokr",
    "multiplier": 1.0,
    "full_matrix": true,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 4,
    "apply_preset": {
        "target_module": [
            "Flux2Attention", "Flux2FeedForward", "Flux2ParallelSelfAttention"
        ],
        "module_algo_map": {
            "Flux2FeedForward": {
                "factor": 4
            },
            "Flux2Attention": {
                "factor": 2
            },
            "Flux2ParallelSelfAttention": {
                "factor": 2
            }
        }
    }
}
```

## संभावित समस्याएँ

SDXL पर Lycoris उपयोग करते समय यह नोट किया गया है कि FeedForward मॉड्यूल्स को ट्रेन करने से मॉडल टूट सकता है और loss `NaN` (Not‑a‑Number) हो सकता है।

यह समस्या SageAttention (`--sageattention_usage=training` के साथ) का उपयोग करने पर और बढ़ सकती है, जिससे मॉडल का तुरंत फेल होना लगभग तय हो जाता है।

समाधान यह है कि lycoris कॉन्फ़िग से `FeedForward` मॉड्यूल्स हटा दें और केवल `Attention` ब्लॉक्स ट्रेन करें।

## LyCORIS इनफेरेंस उदाहरण

यह एक सरल FLUX.1-dev इनफेरेंस स्क्रिप्ट है, जो दिखाता है कि अपने unet या transformer को create_lycoris_from_weights के साथ कैसे wrap करें और फिर इनफेरेंस में उपयोग करें।

```py
import torch

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import AutoModelForCausalLM, CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

from lycoris import create_lycoris_from_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
bfl_repo = "black-forest-labs/FLUX.1-dev"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer")

lycoris_safetensors_path = 'pytorch_lora_weights.safetensors'
lycoris_strength = 1.0
wrapper, _ = create_lycoris_from_weights(lycoris_strength, lycoris_safetensors_path, transformer)
wrapper.merge_to() # using apply_to() will be slower.

transformer.to(device, dtype=dtype)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
)

pipe.enable_sequential_cpu_offload()

with torch.inference_mode():
    image = pipe(
        prompt="a pokemon that looks like a pizza is eating a popsicle",
        width=1280,
        height=768,
        num_inference_steps=15,
        generator=generator,
        guidance_scale=3.5,
    ).images[0]
image.save('image.png')

# optionally, save a merged pipeline containing the LyCORIS baked-in:
pipe.save_pretrained('/path/to/output/pipeline')
```
