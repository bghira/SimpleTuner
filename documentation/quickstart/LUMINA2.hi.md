## Lumina2 क्विकस्टार्ट

इस उदाहरण में, हम Lumina2 LoRA या full model fine‑tune प्रशिक्षण करेंगे।

### हार्डवेयर आवश्यकताएँ

Lumina2 एक 2B पैरामीटर मॉडल है, जो Flux या SD3 जैसे बड़े मॉडलों की तुलना में काफी अधिक सुलभ है। मॉडल का छोटा आकार मतलब:

rank‑16 LoRA प्रशिक्षण पर यह उपयोग करता है:
- लगभग 12‑14GB VRAM (LoRA प्रशिक्षण)
- लगभग 16‑20GB VRAM (full fine‑tuning)
- startup के दौरान लगभग 20‑30GB सिस्टम RAM

आपको चाहिए:
- **न्यूनतम**: एक RTX 3060 12GB या RTX 4060 Ti 16GB
- **अनुशंसित**: तेज़ प्रशिक्षण के लिए RTX 3090, RTX 4090, या A100
- **सिस्टम RAM**: कम से कम 32GB अनुशंसित

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

Vast, RunPod, और TensorDock (आदि) के लिए, CUDA 12.2‑12.8 इमेज पर यह काम करेगा:

```bash
apt -y install nvidia-cuda-toolkit
```

### इंस्टॉलेशन

pip के जरिए SimpleTuner इंस्टॉल करें:

```bash
pip install 'simpletuner[cuda]' --extra-index-url https://download.pytorch.org/whl/cu132

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu132
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [installation documentation](../INSTALL.md) देखें।

### वातावरण सेटअप

SimpleTuner चलाने के लिए, आपको एक configuration फ़ाइल, dataset और model directories, तथा एक dataloader configuration फ़ाइल सेट करनी होगी।

#### Configuration file

`config/config.json.example` को `config/config.json` में कॉपी करें:

```bash
cp config/config.json.example config/config.json
```

वहाँ आपको निम्न वेरिएबल्स बदलने होंगे:

- `model_type` - LoRA प्रशिक्षण के लिए `lora` या full fine‑tuning के लिए `full` सेट करें।
- `model_family` - इसे `lumina2` पर सेट करें।
- `output_dir` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप अपने checkpoints और validation images रखना चाहते हैं। यहाँ full path उपयोग करने की सलाह है।
- `train_batch_size` - आपके GPU memory और dataset size पर निर्भर करते हुए 1‑4 हो सकता है।
- `validation_resolution` - Lumina2 कई resolutions सपोर्ट करता है। सामान्य विकल्प: `1024x1024`, `512x512`, `768x768`।
- `validation_guidance` - Lumina2 classifier‑free guidance उपयोग करता है। 3.5‑7.0 के मान अच्छे चलते हैं।
- `validation_num_inference_steps` - 20‑30 स्टेप्स Lumina2 के लिए अच्छे हैं।
- `gradient_accumulation_steps` - बड़े batch का अनुकरण करने के लिए उपयोग करें। 2‑4 अच्छा काम करता है।
- `optimizer` - `adamw_bf16` अनुशंसित है। `lion` और `optimi-stableadamw` भी अच्छे काम करते हैं।
- `mixed_precision` - सर्वोत्तम परिणामों के लिए इसे `bf16` रखें।
- `gradient_checkpointing` - VRAM बचाने के लिए `true` सेट करें।
- `learning_rate` - LoRA के लिए: `1e-4` से `5e-5`। Full fine‑tuning के लिए: `1e-5` से `1e-6`।

#### Lumina2 उदाहरण कॉन्फ़िगरेशन

यह `config.json` में जाता है:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "base_model_precision": "int8-torchao",
    "checkpoint_step_interval": 50,
    "data_backend_config": "config/lumina2/multidatabackend.json",
    "disable_bucket_pruning": true,
    "eval_steps_interval": 50,
    "evaluation_type": "clip",
    "flow_schedule_auto_shift": true,
    "gradient_checkpointing": true,
    "hub_model_id": "lumina2-lora",
    "learning_rate": 1e-4,
    "lora_alpha": 16,
    "lora_rank": 16,
    "lora_type": "standard",
    "lr_scheduler": "constant",
    "max_train_steps": 400000,
    "model_family": "lumina2",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/lumina2",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "quantize_via": "cpu",
    "report_to": "wandb",
    "seed": 42,
    "tracker_project_name": "lumina2-training",
    "tracker_run_name": "lumina2-lora",
    "train_batch_size": 4,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_disable_unconditional": true,
    "validation_guidance": 4.0,
    "validation_guidance_rescale": 0.0,
    "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
    "validation_num_inference_steps": 40,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_prompt_library": false,
    "validation_resolution": "1024x1024",
    "validation_seed": 42,
    "validation_step_interval": 50
}
```
</details>

Lycoris प्रशिक्षण के लिए, `lora_type` को `lycoris` पर स्विच करें।

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

#### वैलिडेशन प्रॉम्प्ट्स

`config/config.json` के अंदर "primary validation prompt" होता है। इसके अतिरिक्त, एक prompt library फ़ाइल बनाएँ:

```json
{
  "portrait": "a high-quality portrait photograph with natural lighting",
  "landscape": "a breathtaking landscape photograph with dramatic lighting",
  "artistic": "an artistic rendering with vibrant colors and creative composition",
  "detailed": "a highly detailed image with sharp focus and rich textures",
  "stylized": "a stylized illustration with unique artistic flair"
}
```

अपने कॉन्फ़िग में जोड़ें:
```json
{
  "--user_prompt_library": "config/user_prompt_library.json"
}
```

#### डेटासेट विचार

Lumina2 उच्च गुणवत्ता वाले प्रशिक्षण डेटा से लाभ लेता है। एक `--data_backend_config` (`config/multidatabackend.json`) बनाएँ:

> 💡 **टिप:** बड़े डेटासेट्स में जहाँ डिस्क स्पेस चिंता का विषय हो, आप `--vae_cache_disable` का उपयोग करके VAE एन्कोडिंग ऑनलाइन कर सकते हैं ताकि परिणाम डिस्क पर कैश न हों।

```json
[
  {
    "id": "lumina2-training",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 2048,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/lumina2/training",
    "instance_data_dir": "/datasets/training",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/lumina2",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

अपनी dataset डायरेक्टरी बनाएँ। इस पाथ को अपने वास्तविक स्थान से अपडेट करें।

```bash
mkdir -p /datasets/training
</details>

# Place your images and caption files in /datasets/training/
```

Caption फाइलों का नाम इमेज के समान होना चाहिए और `.txt` एक्सटेंशन होना चाहिए।

#### WandB में लॉग‑इन

SimpleTuner में **वैकल्पिक** tracker सपोर्ट है, मुख्यतः Weights & Biases पर केंद्रित। इसे `report_to=none` से बंद किया जा सकता है।

wandb सक्षम करने के लिए:

```bash
wandb login
```

#### Huggingface Hub में लॉग‑इन

Huggingface Hub पर checkpoints push करने के लिए सुनिश्चित करें कि:
```bash
huggingface-cli login
```

### प्रशिक्षण रन निष्पादित करना

SimpleTuner डायरेक्टरी से, प्रशिक्षण शुरू करने के लिए आपके पास कई विकल्प हैं:

**विकल्प 1 (अनुशंसित - pip install):**
```bash
pip install 'simpletuner[cuda]' --extra-index-url https://download.pytorch.org/whl/cu132

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu132
simpletuner train
```

**विकल्प 2 (Git clone विधि):**
```bash
simpletuner train
```

**विकल्प 3 (Legacy विधि - अभी भी काम करता है):**
```bash
./train.sh
```

इससे text embed और VAE आउटपुट कैशिंग डिस्क पर शुरू होगी।

## Lumina2 के लिए प्रशिक्षण टिप्स

### Learning rates

#### LoRA Training
- `1e-4` से शुरू करें और परिणामों के आधार पर समायोजित करें
- Lumina2 तेज़ी से ट्रेन होता है, इसलिए शुरुआती iterations को क़रीब से मॉनिटर करें
- अधिकतर उपयोग‑मामलों के लिए rank 8‑32 अच्छा है; 64‑128 के लिए करीबी मॉनिटरिंग चाहिए, और 256‑512 नए कार्यों को मॉडल में सिखाने के लिए उपयोगी हो सकता है

#### Full Fine‑tuning
- कम learning rates उपयोग करें: `1e-5` से `5e-6`
- स्थिरता के लिए EMA (Exponential Moving Average) पर विचार करें
- Gradient clipping (`max_grad_norm`) 1.0 अनुशंसित है

### Resolution considerations

Lumina2 flexible resolutions सपोर्ट करता है:
- 1024x1024 पर training सबसे अच्छी गुणवत्ता देता है
- Mixed resolution training (512px, 768px, 1024px) का गुणवत्ता प्रभाव field‑tested नहीं है
- Aspect ratio bucketing Lumina2 के साथ अच्छा काम करता है

### Training duration

Lumina2 का 2B पैरामीटर आकार इसे कुशल बनाता है:
- LoRA training अक्सर 500‑2000 steps में converge हो जाता है
- Full fine‑tuning को 2000‑5000 steps लग सकते हैं
- मॉडल तेज़ी से ट्रेन होता है, इसलिए validation images अक्सर मॉनिटर करें

### सामान्य समस्याएँ और समाधान

1. **मॉडल बहुत जल्दी converge होना**: learning rate घटाएँ, Lion optimiser से AdamW पर स्विच करें
2. **जनरेटेड इमेज में artifacts**: उच्च‑गुणवत्ता प्रशिक्षण डेटा सुनिश्चित करें और learning rate घटाने पर विचार करें
3. **Out of memory**: gradient checkpointing सक्षम करें और batch size घटाएँ
4. **आसानी से overfitting**: regularisation datasets उपयोग करें

## Inference टिप्स

### अपने प्रशिक्षित मॉडल का उपयोग

Lumina2 मॉडलों का उपयोग किया जा सकता है:
- सीधे Diffusers लाइब्रेरी के साथ
- उचित nodes के साथ ComfyUI में
- अन्य inference frameworks जो Gemma2‑आधारित मॉडलों को सपोर्ट करते हैं

### Optimal inference settings

- Guidance scale: 4.0‑6.0
- Inference steps: 20‑50
- सर्वोत्तम परिणामों के लिए वही resolution उपयोग करें जिस पर ट्रेन किया था

## नोट्स

### Lumina2 के फायदे

- 2B पैरामीटर आकार के कारण तेज़ प्रशिक्षण
- गुणवत्ता‑to‑size ratio अच्छा
- विभिन्न training modes सपोर्ट (LoRA, LyCORIS, full)
- कुशल मेमोरी उपयोग

### वर्तमान सीमाएँ

- अभी ControlNet सपोर्ट नहीं
- केवल text‑to‑image generation तक सीमित
- सर्वोत्तम परिणामों के लिए कैप्शन गुणवत्ता उच्च होना आवश्यक

### मेमोरी optimization

बड़े मॉडलों के विपरीत, Lumina2 आमतौर पर यह नहीं मांगता:
- Model quantization
- अत्यधिक memory optimization तकनीकें
- जटिल mixed precision रणनीतियाँ
