# ControlNet प्रशिक्षण गाइड

## पृष्ठभूमि

ControlNet मॉडल कई कार्य कर सकते हैं, जो प्रशिक्षण समय दिए गए conditioning डेटा पर निर्भर करते हैं।

शुरुआत में इन्हें ट्रेन करना बहुत resource‑intensive था, लेकिन अब हम PEFT LoRA या Lycoris का उपयोग करके वही कार्य बहुत कम संसाधनों में कर सकते हैं।

उदाहरण (Diffusers ControlNet मॉडल कार्ड से लिया गया):

![उदाहरण](https://tripleback.net/public/controlnet-example-1.png)

बाईं ओर, आप conditioning इनपुट के रूप में दिया गया "canny edge map" देख सकते हैं। इसके दाईं ओर ControlNet मॉडल द्वारा बेस SDXL मॉडल से गाइड किए गए आउटपुट हैं।

जब मॉडल इस तरह उपयोग होता है, तो prompt लगभग कोई भी composition तय नहीं करता—यह सिर्फ़ विवरण भरता है।

## ControlNet प्रशिक्षण कैसा दिखता है

शुरुआत में, ControlNet ट्रेन करते समय नियंत्रण का कोई संकेत नहीं दिखता:

![उदाहरण](https://tripleback.net/public/controlnet-example-2.png)
(_ControlNet को Stable Diffusion 2.1 मॉडल पर केवल 4 steps के लिए ट्रेन किया गया_)

antelope prompt अभी भी composition पर अधिक नियंत्रण रखता है, और ControlNet conditioning इनपुट अनदेखा होता है।

समय के साथ, control इनपुट का सम्मान होना चाहिए:

![उदाहरण](https://tripleback.net/public/controlnet-example-3.png)
(_ControlNet को Stable Diffusion XL मॉडल पर केवल 100 steps के लिए ट्रेन किया गया_)

उस बिंदु पर ControlNet का प्रभाव दिखना शुरू होता है, लेकिन परिणाम बेहद असंगत रहते हैं।

इसके लिए 100 steps से कहीं अधिक steps की आवश्यकता होगी!

## उदाहरण dataloader कॉन्फ़िगरेशन

dataloader कॉन्फ़िगरेशन सामान्य text‑to‑image dataset कॉन्फ़िगरेशन के काफ़ी करीब रहता है:

- मुख्य image डेटा `antelope-data` सेट है
  - `conditioning_data` की को अब सेट किया जाता है, और इसे उस conditioning डेटा के `id` पर सेट करना चाहिए जो इस सेट के साथ पेयर होता है।
  - बेस सेट के लिए `dataset_type` को `image` होना चाहिए
- दूसरा dataset कॉन्फ़िगर किया जाता है, जिसका नाम `antelope-conditioning` है
  - नाम महत्वपूर्ण नहीं है — उदाहरण में केवल समझाने के लिए `-data` और `-conditioning` जोड़ा गया है।
  - `dataset_type` को `conditioning` सेट करना चाहिए, ताकि ट्रेनर जान सके कि इसे evaluation और conditioned input प्रशिक्षण के लिए उपयोग करना है।
- SDXL ट्रेनिंग में conditioning inputs VAE‑encoded नहीं होते, बल्कि प्रशिक्षण के दौरान सीधे pixel values के रूप में मॉडल में जाते हैं। इसका मतलब है कि प्रशिक्षण की शुरुआत में VAE embeds प्रोसेस करने में समय नहीं लगता!
- Flux, SD3, Auraflow, HiDream, या अन्य MMDiT मॉडल्स ट्रेन करते समय conditioning inputs latents में encode होते हैं, और ये प्रशिक्षण के दौरान on‑demand compute किए जाते हैं।
- यहाँ सब कुछ `-controlnet` के रूप में explicitly लेबल है, लेकिन आप वही text embeds उपयोग कर सकते हैं जो आपने सामान्य full/LoRA tuning के लिए उपयोग किए थे। ControlNet inputs prompt embeds को संशोधित नहीं करते।
- aspect bucketing और random cropping के साथ, conditioning samples भी मुख्य image samples की तरह ही crop होंगे, इसलिए इसकी चिंता नहीं करनी पड़ेगी।

```json
[
    {
        "id": "antelope-data",
        "type": "local",
        "dataset_type": "image",
        "conditioning_data": "antelope-conditioning",
        "instance_data_dir": "datasets/animals/antelope-data",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an antelope",
        "metadata_backend": "discovery",
        "minimum_image_size": 512,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "cache_dir_vae": "cache/vae/sdxl/antelope-data",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "cache_file_suffix": "controlnet-sdxl"
    },
    {
        "id": "antelope-conditioning",
        "type": "local",
        "dataset_type": "conditioning",
        "instance_data_dir": "datasets/animals/antelope-conditioning",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an antelope",
        "metadata_backend": "discovery",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "minimum_image_size": 512,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel_area",
        "cache_file_suffix": "controlnet-sdxl"
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/sdxl-base/controlnet"
    }
]
```

## Conditioning इमेज इनपुट बनाना

SimpleTuner में ControlNet समर्थन जितना नया है, अभी प्रशिक्षण सेट बनाने के लिए हमारे पास केवल एक विकल्प है:

- [create_canny_edge.py](/scripts/toolkit/datasets/controlnet/create_canny_edge.py)
  - Canny मॉडल प्रशिक्षण के लिए एक बेहद बुनियादी उदाहरण।
  - आपको स्क्रिप्ट में `input_dir` और `output_dir` मान बदलने होंगे

यह 100 से कम इमेज वाले छोटे डेटासेट के लिए लगभग 30 सेकंड लेगा।

## ControlNet मॉडल ट्रेन करने के लिए कॉन्फ़िगरेशन बदलना

सिर्फ़ dataloader कॉन्फ़िगरेशन सेट करना पर्याप्त नहीं होगा।

`config/config.json` के अंदर आपको निम्न मान सेट करने होंगे:

```bash
"model_type": 'lora',
"controlnet": true,

# You may have to reduce TRAIN_BATCH_SIZE and RESOLUTION more than usual
"train_batch_size": 1
```

अंत में आपका कॉन्फ़िगरेशन कुछ ऐसा दिखेगा:

```json
{
    "aspect_bucket_rounding": 2,
    "caption_dropout_probability": 0.1,
    "checkpoint_step_interval": 100,
    "checkpoints_total_limit": 5,
    "controlnet": true,
    "data_backend_config": "config/controlnet-sdxl/multidatabackend.json",
    "disable_benchmark": false,
    "gradient_checkpointing": true,
    "hub_model_id": "simpletuner-controlnet-sdxl-lora-test",
    "learning_rate": 3e-5,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 100,
    "max_train_steps": 1000,
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "sdxl",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "bnb-lion8bit",
    "output_dir": "output/controlnet-sdxl/models",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 42,
    "train_batch_size": 1,
    "use_ema": false,
    "vae_cache_ondemand": true,
    "validation_guidance": 4.2,
    "validation_guidance_rescale": 0.0,
    "validation_num_inference_steps": 20,
    "validation_resolution": "1024x1024",
    "validation_seed": 42,
    "validation_step_interval": 10,
    "validation_torch_compile": false
}
```

## तैयार ControlNet मॉडल पर इनफेरेंस

यहाँ **full** ControlNet मॉडल (ControlNet LoRA नहीं) पर इनफेरेंस के लिए एक SDXL उदाहरण दिया गया है:

```py
# Update these values:
base_model = "stabilityai/stable-diffusion-xl-base-1.0"         # This is the model you used as `--pretrained_model_name_or_path`
controlnet_model_path = "diffusers/controlnet-canny-sdxl-1.0"   # This is the path to the resulting ControlNet checkpoint
# controlnet_model_path = "/path/to/controlnet/checkpoint-100"

# Leave the rest alone:
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'

image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    controlnet_model_path,
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

images = pipe(
    prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

images[0].save(f"hug_lab.png")
```
(_डेमो कोड [Hugging Face SDXL ControlNet उदाहरण](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0) से लिया गया_)


## स्वचालित डेटा ऑग्मेंटेशन और Conditioning जनरेशन

SimpleTuner स्टार्टअप के दौरान स्वतः conditioning datasets बना सकता है, जिससे मैन्युअल preprocessing की ज़रूरत नहीं रहती। यह खास तौर पर उपयोगी है:
- Super‑resolution प्रशिक्षण
- JPEG artifacts हटाना
- Depth‑guided generation
- Edge detection (Canny)

### यह कैसे काम करता है

manually conditioning datasets बनाने की बजाय, आप अपनी मुख्य dataset कॉन्फ़िगरेशन में एक `conditioning` array निर्दिष्ट कर सकते हैं। SimpleTuner:
1. स्टार्टअप पर conditioning images generate करता है
2. उपयुक्त metadata के साथ अलग datasets बनाता है
3. उन्हें आपके मुख्य dataset से स्वतः लिंक करता है

### परफ़ॉर्मेंस विचार

कुछ generators CPU‑bound होने पर धीमे चलेंगे, जबकि कुछ GPU संसाधन मांग सकते हैं और main process में चलेंगे, जिससे startup time बढ़ सकता है।

**CPU‑based generators (तेज़):**
- `superresolution` - Blur और noise ऑपरेशन्स
- `jpeg_artifacts` - Compression simulation
- `random_masks` - Mask generation
- `canny` - Edge detection

**GPU‑based generators (धीमे):**
- `depth` / `depth_midas` - Transformer मॉडल्स लोड करना पड़ता है
- `segmentation` - Semantic segmentation मॉडल्स
- `optical_flow` - Motion estimation

GPU‑based generators main process में चलते हैं और बड़े datasets के लिए startup time को काफ़ी बढ़ा सकते हैं।

### उदाहरण: Multi‑Task Conditioning Dataset

यह एक पूर्ण उदाहरण है जो एक ही source dataset से कई conditioning प्रकार बनाता है:

```json
[
  {
    "id": "multitask-training",
    "type": "local",
    "instance_data_dir": "/datasets/high-quality-images",
    "caption_strategy": "filename",
    "resolution": 512,
    "conditioning": [
      {
        "type": "superresolution",
        "blur_radius": 2.0,
        "noise_level": 0.02,
        "captions": ["enhance image quality", "increase resolution", "sharpen"]
      },
      {
        "type": "jpeg_artifacts",
        "quality_range": [20, 40],
        "captions": ["remove compression", "fix jpeg artifacts"]
      },
      {
        "type": "canny",
        "low_threshold": 50,
        "high_threshold": 150
      }
    ]
  },
  {
    "id": "text-embed-cache",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/sdxl"
  }
]
```

यह कॉन्फ़िगरेशन:
1. `/datasets/high-quality-images` से आपकी high‑quality images लोड करेगा
2. तीन conditioning datasets स्वतः जनरेट करेगा
3. super‑resolution और JPEG कार्यों के लिए specific captions उपयोग करेगा
4. Canny edge dataset के लिए मूल image captions उपयोग करेगा

#### जनरेट किए गए datasets के लिए caption रणनीतियाँ

आपके पास conditioning data के captions के लिए दो विकल्प हैं:

1. **source captions का उपयोग** (डिफ़ॉल्ट): `captions` फ़ील्ड छोड़ दें
2. **कस्टम captions**: एक string या strings का array दें

कार्य‑विशिष्ट प्रशिक्षण (जैसे "enhance" या "remove artifacts") के लिए custom captions अक्सर मूल image विवरणों से बेहतर काम करते हैं।

### Startup समय ऑप्टिमाइज़ेशन

बड़े datasets के लिए conditioning generation समय‑साध्य हो सकता है। ऑप्टिमाइज़ करने के लिए:

1. **एक बार जनरेट करें**: conditioning data cache होता है और पहले से मौजूद होने पर फिर से generate नहीं होगा
2. **CPU generators उपयोग करें**: ये तेज़ generation के लिए multiple processes का उपयोग कर सकते हैं
3. **unused types बंद करें**: केवल वही generate करें जिसकी आपको प्रशिक्षण के लिए आवश्यकता है
4. **Pre‑generate**: discovery और conditioning data generation छोड़ने के लिए `--skip_file_discovery=true` के साथ चला सकते हैं
5. **disk scans से बचें**: किसी भी बड़े dataset कॉन्फ़िगरेशन पर `preserve_data_backend_cache=True` उपयोग करके disk को दोबारा scan होने से बचा सकते हैं। इससे startup time काफ़ी तेज़ हो जाएगा, खासकर बड़े datasets में।

Generation प्रक्रिया progress bars दिखाती है और interrupt होने पर resume को सपोर्ट करती है।
