# SDXL LCM Distillation Quickstart (SimpleTuner)

इस उदाहरण में हम **4-8 step SDXL student** को **LCM (Latent Consistency Model) distillation** से प्रशिक्षित करेंगे, एक pre-trained SDXL teacher मॉडल से।

> **नोट**: अन्य मॉडल्स को भी आधार के रूप में उपयोग किया जा सकता है, यहाँ SDXL सिर्फ LCM कॉन्फ़िगरेशन कॉन्सेप्ट्स को समझाने के लिए उपयोग किया गया है।

LCM सक्षम करता है:
* Ultra-fast inference (4-8 steps बनाम 25-50)
* timesteps के बीच consistency
* कम स्टेप्स में उच्च गुणवत्ता आउटपुट

## 📦 इंस्टॉलेशन

मानक SimpleTuner इंस्टॉलेशन [गाइड](../INSTALL.md) का पालन करें:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.12 -m venv .venv && source .venv/bin/activate

# Install with automatic platform detection
pip install -e .
```

**नोट:** setup.py आपके प्लेटफ़ॉर्म (CUDA/ROCm/Apple) को स्वतः पहचानकर उपयुक्त dependencies इंस्टॉल करता है।

कंटेनर एनवायरनमेंट (Vast, RunPod, आदि) के लिए:
```bash
apt -y install nvidia-cuda-toolkit
```

---

## 📁 कॉन्फ़िगरेशन

SDXL LCM के लिए `config/config.json` बनाएं:

```json
{
  "model_type": "lora",
  "model_family": "sdxl",
  "output_dir": "/home/user/output/sdxl-lcm",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",

  "distillation_method": "lcm",
  "distillation_config": {
    "lcm": {
      "num_ddim_timesteps": 50,
      "w_min": 1.0,
      "w_max": 12.0,
      "loss_type": "l2",
      "huber_c": 0.001,
      "timestep_scaling_factor": 10.0
    }
  },

  "resolution": 1024,
  "resolution_type": "pixel",
  "validation_resolution": "1024x1024,1280x768,768x1280",
  "aspect_bucket_rounding": 64,
  "minimum_image_size": 0.5,
  "maximum_image_size": 1.0,

  "learning_rate": 1e-4,
  "lr_scheduler": "constant_with_warmup",
  "lr_warmup_steps": 1000,
  "max_train_steps": 10000,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "gradient_checkpointing": true,
  "mixed_precision": "bf16",

  "lora_type": "standard",
  "lora_rank": 64,
  "lora_alpha": 64,
  "lora_dropout": 0.0,

  "validation_step_interval": 250,
  "validation_num_inference_steps": 4,
  "validation_guidance": 0.0,
  "validation_prompt": "A portrait of a woman with flowers in her hair, highly detailed, professional photography",
  "validation_negative_prompt": "blurry, low quality, distorted, amateur",

  "checkpoint_step_interval": 500,
  "checkpoints_total_limit": 5,
  "resume_from_checkpoint": "latest",

  "optimizer": "adamw_bf16",
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_weight_decay": 1e-2,
  "adam_epsilon": 1e-8,
  "max_grad_norm": 1.0,

  "seed": 42,
  "push_to_hub": true,
  "hub_model_id": "your-username/sdxl-lcm-distilled",
  "report_to": "wandb",
  "tracker_project_name": "sdxl-lcm-distillation",
  "tracker_run_name": "sdxl-lcm-4step"
}
```

### प्रमुख LCM कॉन्फ़िगरेशन विकल्प:

- **`num_ddim_timesteps`**: DDIM solver में timesteps की संख्या (50-100 सामान्य)
- **`w_min/w_max`**: ट्रेनिंग के लिए guidance scale रेंज (SDXL के लिए 1.0-12.0)
- **`loss_type`**: "l2" या "huber" उपयोग करें (huber outliers के लिए अधिक robust)
- **`timestep_scaling_factor`**: boundary conditions के लिए scaling (डिफ़ॉल्ट 10.0)
- **`validation_num_inference_steps`**: अपने target step count के साथ टेस्ट करें (4-8)
- **`validation_guidance`**: LCM के लिए 0.0 रखें (inference में कोई CFG नहीं)

### Quantized Training के लिए (कम VRAM):

मेमोरी उपयोग कम करने के लिए ये विकल्प जोड़ें:
```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "optimizer": "optimi-lion"
}
```

---

## 🎬 डेटासेट कॉन्फ़िगरेशन

अपने output डायरेक्टरी में `multidatabackend.json` बनाएं:

```json
[
  {
    "id": "your-dataset-name",
    "type": "local",
    "crop": false,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1.0,
    "minimum_image_size": 0.5,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/sdxl/your-dataset",
    "instance_data_dir": "/path/to/your/dataset",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sdxl/your-dataset",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> **महत्वपूर्ण**: LCM distillation को विविध, उच्च-गुणवत्ता डेटा चाहिए। अच्छे परिणामों के लिए कम से कम 10k+ इमेजेस की सलाह है।

---

## 🚀 ट्रेनिंग

1. **सेवाओं में लॉगिन करें** (यदि hub फीचर्स का उपयोग कर रहे हैं):
   ```bash
   huggingface-cli login
   wandb login
   ```

2. **ट्रेनिंग शुरू करें**:
   ```bash
   bash train.sh
   ```

3. **प्रगति मॉनिटर करें**:
   - LCM loss के कम होने पर नज़र रखें
   - 4-8 steps पर validation इमेजेस गुणवत्ता बनाए रखें
   - ट्रेनिंग आमतौर पर 5k-10k steps लेती है

---

## 📊 अपेक्षित परिणाम

| Metric | Expected Value | Notes |
| ------ | -------------- | ----- |
| LCM Loss | < 0.1 | लगातार घटता रहना चाहिए |
| Validation Quality | 4 steps पर अच्छा | guidance=0 की जरूरत हो सकती है |
| Training Time | 5-10 घंटे | single A100 पर |
| Final Inference | 4-8 steps | base SDXL के 25-50 की तुलना में |

---

## 🧩 ट्रबलशूटिंग

| समस्या | समाधान |
| ------- | -------- |
| **OOM errors** | batch size घटाएँ, gradient checkpointing सक्षम करें, int8 quantization उपयोग करें |
| **Blurry outputs** | `num_ddim_timesteps` बढ़ाएँ, डेटा गुणवत्ता जांचें, learning rate कम करें |
| **Slow convergence** | learning rate को 2e-4 तक बढ़ाएँ, विविध डेटासेट सुनिश्चित करें |
| **Validation खराब दिख रहा** | `validation_guidance: 0.0` उपयोग करें, सही scheduler इस्तेमाल हो रहा है या नहीं जांचें |
| **Artifacts कम steps पर** | <4 steps पर सामान्य है, अधिक समय ट्रेन करें या `w_min/w_max` समायोजित करें |

---

## 🔧 उन्नत टिप्स

1. **Multi-resolution training**: SDXL कई aspect ratios पर ट्रेनिंग से लाभ लेता है:
   ```json
   "validation_resolution": "1024x1024,1280x768,768x1280,1152x896,896x1152"
   ```

2. **Progressive training**: पहले अधिक timesteps, फिर कम करें:
   - Week 1: `validation_num_inference_steps: 8` के साथ ट्रेन करें
   - Week 2: `validation_num_inference_steps: 4` के साथ fine-tune करें

3. **Inference के लिए scheduler**: ट्रेनिंग के बाद LCM scheduler का उपयोग करें:
   ```python
   from diffusers import LCMScheduler
   scheduler = LCMScheduler.from_pretrained(
       "stabilityai/stable-diffusion-xl-base-1.0",
       subfolder="scheduler"
   )
   ```

4. **ControlNet के साथ संयोजन**: LCM कम steps पर guided generation के लिए ControlNet के साथ अच्छा काम करता है।

---

## 📚 अतिरिक्त संसाधन

- [LCM Paper](https://arxiv.org/abs/2310.04378)
- [Diffusers LCM Docs](https://huggingface.co/docs/diffusers/using-diffusers/inference_with_lcm)
- [More SimpleTuner Docs](../quickstart/SDXL.md)

---

## 🎯 अगला कदम

LCM distillation सफल होने के बाद:
1. विभिन्न prompts पर 4-8 steps पर अपने मॉडल को टेस्ट करें
2. अलग base models पर LCM-LoRA आज़माएँ
3. विशेष use cases के लिए और कम steps (2-3) पर प्रयोग करें
4. domain-specific data पर fine-tuning पर विचार करें
