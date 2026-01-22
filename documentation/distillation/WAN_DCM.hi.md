# DCM Distillation Quickstart (SimpleTuner)

इस उदाहरण में हम एक **4-step student** ट्रेन करेंगे **DCM distillation** के जरिए, एक बड़े flow-matching teacher मॉडल से जैसे [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)।

DCM सपोर्ट करता है:

* **Semantic** मोड: standard flow-matching जिसमें CFG baked-in होता है।
* **Fine** मोड: वैकल्पिक GAN-आधारित adversarial supervision (experimental)।

---

## ✅ हार्डवेयर आवश्यकताएँ

| Model     | Batch Size | Min VRAM | Notes                                  |
| --------- | ---------- | -------- | -------------------------------------- |
| Wan 1.3B  | 1          | 12 GB    | A5000 / 3090+ tier GPU                 |
| Wan 14B   | 1          | 24 GB    | धीमा; `--offload_during_startup` उपयोग करें |
| Fine mode | 1          | +10%     | Discriminator प्रति GPU चलता है        |

> ⚠️ Mac और Apple silicon धीमे हैं और अनुशंसित नहीं हैं। semantic mode में भी आपको 10 min/step रनटाइम मिलेंगे।

---

## 📦 इंस्टॉलेशन

Wan गाइड जैसे ही स्टेप्स:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv && source .venv/bin/activate

# Install with automatic platform detection
pip install -e .
```

**नोट:** setup.py आपके प्लेटफ़ॉर्म (CUDA/ROCm/Apple) को स्वतः पहचानकर उपयुक्त dependencies इंस्टॉल करता है।

---

## 📁 कॉन्फ़िगरेशन

अपना `config/config.json` एडिट करें:

```json
{
    "aspect_bucket_rounding": 2,
    "attention_mechanism": "diffusers",
    "base_model_precision": "int8-quanto",
    "caption_dropout_probability": 0.1,
    "checkpoint_step_interval": 100,
    "checkpoints_total_limit": 5,
    "compress_disk_cache": true,
    "data_backend_config": "config/wan/multidatabackend.json",
    "delete_problematic_images": false,
    "disable_benchmark": false,
    "disable_bucket_pruning": true,
    "distillation_method": "dcm",
    "distillation_config": {
      "mode": "semantic",
      "euler_steps": 100
    },
    "ema_update_interval": 2,
    "ema_validation": "ema_only",
    "flow_schedule_shift": 17,
    "grad_clip_method": "value",
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": true,
    "hub_model_id": "wan-disney-DCM-distilled",
    "ignore_final_epochs": true,
    "learning_rate": 1e-4,
    "lora_alpha": 128,
    "lora_rank": 128,
    "lora_type": "standard",
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 400000,
    "lycoris_config": "config/wan/lycoris_config.json",
    "max_grad_norm": 0.01,
    "max_train_steps": 400000,
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "wan",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/wan",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "prodigy_steps": 100000,
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "quantize_via": "cpu",
    "report_to": "wandb",
    "resolution": 480,
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 42,
    "text_encoder_1_precision": "int8-quanto",
    "tracker_project_name": "lora-training",
    "tracker_run_name": "wan-AdamW-DCM",
    "train_batch_size": 2,
    "use_ema": false,
    "vae_batch_size": 1,
    "validation_guidance": 1.0,
    "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "validation_num_inference_steps": 8,
    "validation_num_video_frames": 16,
    "validation_prompt": "A black and white animated scene unfolds featuring a distressed upright cow with prominent horns and expressive eyes, suspended by its legs from a hook on a static background wall. A smaller Mickey Mouse-like character enters, standing near a wooden bench, initiating interaction between the two. The cow's posture changes as it leans, stretches, and falls, while the mouse watches with a concerned expression, its face a mixture of curiosity and worry, in a world devoid of color.",
    "validation_prompt_library": false,
    "validation_resolution": "832x480",
    "validation_seed": 42,
    "validation_step_interval": 4,
    "webhook_config": "config/wan/webhook.json"
}
```

### वैकल्पिक:

* **fine mode** के लिए, केवल `"mode": "fine"` बदलें।
  - यह मोड फिलहाल SimpleTuner में प्रयोगात्मक है और इसे उपयोग करने के लिए कुछ अतिरिक्त चरण चाहिए होंगे, जो इस गाइड में अभी नहीं दिए गए हैं।

---

## 🎬 डेटासेट और डाटालोडर

Wan quickstart से Disney dataset और `data_backend_config` JSON को reuse करें।

> **नोट**: यह डेटासेट distillation के लिए पर्याप्त नहीं है, सफल होने के लिए **कहीं अधिक** विविध और बड़े डेटा की आवश्यकता है।

सुनिश्चित करें:

* `num_frames`: 75–81
* `resolution`: 480
* `crop`: false (वीडियो को बिना crop के रखें)
* `repeats`: अभी के लिए 0

---

## 📌 नोट्स

* **Semantic mode** स्थिर है और अधिकांश उपयोग मामलों के लिए अनुशंसित है।
* **Fine mode** यथार्थवाद बढ़ाता है, लेकिन अधिक स्टेप्स और ट्यूनिंग की जरूरत होती है और SimpleTuner में इसका सपोर्ट अभी बहुत अच्छा नहीं है।

---

## 🧩 ट्रबलशूटिंग

| समस्या                      | समाधान                                                                  |
| ---------------------------- | -------------------------------------------------------------------- |
| **Results धुंधले हैं**       | अधिक euler_steps उपयोग करें, या `multiphase` बढ़ाएँ                       |
| **Validation घट रहा है**  | `validation_guidance: 1.0` उपयोग करें                                       |
| **Fine mode में OOM**         | `train_batch_size` घटाएँ, precision levels कम करें, या बड़ा GPU उपयोग करें |
| **Fine mode converge नहीं हो रहा** | fine mode उपयोग न करें, SimpleTuner में यह पर्याप्त रूप से टेस्टेड नहीं है      |
