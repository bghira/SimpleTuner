## Qwen Image क्विकस्टार्ट

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
pip install 'simpletuner[cuda13]'
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
- `train_batch_size` - इसे 1 पर सेट करना अनिवार्य है (batch size > 1 अभी सही काम नहीं करता)।
- `gradient_accumulation_steps` - बड़े batch का अनुकरण करने के लिए 2‑8 पर सेट करें।
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
pip install 'simpletuner[cuda13]'
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

वर्तमान में Qwen Image में टेक्स्ट एन्कोडर की sequence length handling के कारण batch size > 1 में समस्याएँ हैं। हमेशा उपयोग करें:
- `train_batch_size: 1`
- बड़े batch का अनुकरण करने के लिए `gradient_accumulation_steps: 2-8`

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
