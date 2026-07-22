# DMD Distillation Quickstart (SimpleTuner)

इस उदाहरण में हम **3-step student** को **DMD (Distribution Matching Distillation)** से ट्रेन करेंगे, एक बड़े flow-matching teacher मॉडल जैसे [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) से।

> **Note:** Distillation methods को `--train_text_encoder` के साथ combine नहीं किया जा सकता; text encoder training disabled रखें।

DMD फीचर्स:

* **Generator (Student)**: कम स्टेप्स में teacher को मैच करना सीखता है
* **Fake Score Transformer**: teacher और student आउटपुट्स के बीच भेद करता है
* **Multi-step simulation**: वैकल्पिक train-inference consistency मोड

---

## ✅ हार्डवेयर आवश्यकताएँ


⚠️ DMD मेमोरी-इंटेंसिव है क्योंकि fake score transformer के लिए base मॉडल की दूसरी पूरी कॉपी मेमोरी में रखनी पड़ती है।

यदि आपके पास आवश्यक VRAM नहीं है, तो 14B Wan मॉडल के लिए DMD की जगह LCM या DCM distillation विधियों का प्रयास करने की सलाह है।

Sparse attention सपोर्ट के बिना 14B मॉडल distill करने पर NVIDIA B200 की आवश्यकता हो सकती है।

LoRA student ट्रेनिंग से आवश्यकताएँ काफी कम हो सकती हैं, लेकिन फिर भी भारी रहती हैं।

---

## 📦 इंस्टॉलेशन

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
    "checkpoint_step_interval": 200,
    "checkpoints_total_limit": 3,
    "compress_disk_cache": true,
    "data_backend_config": "config/wan/multidatabackend.json",
    "delete_problematic_images": false,
    "disable_benchmark": false,
    "disable_bucket_pruning": true,
    "distillation_method": "dmd",
    "distillation_config": {
        "dmd_denoising_steps": "1000,757,522",
        "generator_update_interval": 1,
        "real_score_guidance_scale": 3.0,
        "fake_score_lr": 1e-5,
        "fake_score_weight_decay": 0.01,
        "fake_score_betas": [0.9, 0.999],
        "fake_score_eps": 1e-8,
        "fake_score_grad_clip": 1.0,
        "fake_score_guidance_scale": 0.0,
        "min_timestep_ratio": 0.02,
        "max_timestep_ratio": 0.98,
        "num_frame_per_block": 3,
        "independent_first_frame": false,
        "same_step_across_blocks": false,
        "last_step_only": false,
        "num_training_frames": 21,
        "context_noise": 0,
        "ts_schedule": true,
        "ts_schedule_max": false,
        "min_score_timestep": 0,
        "timestep_shift": 1.0
    },
    "ema_update_interval": 5,
    "ema_validation": "ema_only",
    "flow_schedule_shift": 5,
    "grad_clip_method": "value",
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": true,
    "hub_model_id": "wan-disney-DMD-3step",
    "strict_epoch_limit": false,
    "learning_rate": 2e-5,
    "lora_alpha": 128,
    "lora_rank": 128,
    "lora_type": "standard",
    "lr_scheduler": "cosine_with_min_lr",
    "lr_warmup_steps": 100,
    "max_grad_norm": 1.0,
    "max_train_steps": 4000,
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "wan",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/wan-dmd",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "quantize_via": "cpu",
    "report_to": "wandb",
    "resolution": 480,
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 1000,
    "text_encoder_1_precision": "int8-quanto",
    "tracker_project_name": "dmd-training",
    "tracker_run_name": "wan-DMD-3step",
    "train_batch_size": 1,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_guidance": 1.0,
    "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "validation_num_inference_steps": 3,
    "validation_num_video_frames": 121,
    "validation_prompt": "A black and white animated scene unfolds featuring a distressed upright cow with prominent horns and expressive eyes, suspended by its legs from a hook on a static background wall. A smaller Mickey Mouse-like character enters, standing near a wooden bench, initiating interaction between the two. The cow's posture changes as it leans, stretches, and falls, while the mouse watches with a concerned expression, its face a mixture of curiosity and worry, in a world devoid of color.",
    "validation_prompt_library": "config/wan/validation_prompts_dmd.json",
    "validation_resolution": "1280x704",
    "validation_seed": 42,
    "validation_step_interval": 200,
    "webhook_config": "config/wan/webhook.json"
}
```

### प्रमुख DMD सेटिंग्स:

* **`dmd_denoising_steps`** – backward simulation के लिए target timesteps (डिफ़ॉल्ट: 3-step student के लिए `1000,757,522`)।
* **`generator_update_interval`** – महँगा generator replay हर _N_ trainer steps पर चलाएँ। गुणवत्ता बनाम गति के लिए इसे बढ़ाएँ।
* **`fake_score_lr` / `fake_score_weight_decay` / `fake_score_betas`** – fake score transformer के optimiser hyperparameters।
* **`fake_score_guidance_scale`** – fake score नेटवर्क पर वैकल्पिक classifier-free guidance (डिफ़ॉल्ट off)।
* **`num_frame_per_block`, `same_step_across_blocks`, `last_step_only`** – self-forcing rollout के दौरान temporal blocks scheduling को नियंत्रित करते हैं।
* **`num_training_frames`** – backward simulation के दौरान generate किए जाने वाले अधिकतम frames (बड़े मान fidelity बढ़ाते हैं लेकिन मेमोरी लागत भी बढ़ती है)।
* **`min_timestep_ratio`, `max_timestep_ratio`, `timestep_shift`** – KL sampling window को आकार देते हैं। यदि आप defaults से हटते हैं तो इन्हें अपने teacher के flow schedule से match करें।

---

## 🎬 डेटासेट और डाटालोडर

DMD को अच्छा चलाने के लिए आपको **विविध, उच्च-गुणवत्ता डेटा** चाहिए:

```json
{
  "dataset_type": "video",
  "cache_dir": "cache/wan-dmd",
  "resolution_type": "pixel_area",
  "crop": false,
  "num_frames": 121,
  "frame_interval": 1,
  "resolution": 480,
  "minimum_image_size": 0,
  "repeats": 0
}
```

> **नोट**: Disney डेटासेट DMD के लिए **अपर्याप्त** है। **इसे उपयोग न करें!** यह सिर्फ उदाहरण के लिए दिया गया है।

आपको चाहिए:
> - उच्च वॉल्यूम (कम से कम 10k+ वीडियो)
> - विविध सामग्री (अलग-अलग स्टाइल, motions, subjects)
> - उच्च गुणवत्ता (कोई compression artifacts नहीं)

ये parent मॉडल से जेनरेट किए जा सकते हैं।

---

## 🚀 ट्रेनिंग टिप्स

1. **generator interval छोटा रखें**: `1. **generator interval छोटा रखें**: शुरुआत में `"generator_update_interval": 1` रखें। केवल तब बढ़ाएँ जब आपको throughput चाहिए और आप अधिक noisy updates सहन कर सकते हों।
2. **दोनों losses मॉनिटर करें**: wandb में `dmd_loss` और `fake_score_loss` देखें
3. **Validation frequency**: DMD जल्दी converge होता है, अक्सर validate करें
4. **मेमोरी प्रबंधन**:
   - `gradient_checkpointing` उपयोग करें
   - `train_batch_size` को 1 तक घटाएँ
   - `base_model_precision: "int8-quanto"` पर विचार करें

---

## 📌 DMD बनाम DCM

| Feature | DCM | DMD |
|---------|-----|-----|
| Memory usage | कम | अधिक (fake score मॉडल) |
| Training time | लंबा | छोटा (आमतौर पर 4k steps) |
| Quality | अच्छी | उत्कृष्ट |
| Inference steps | 4-8+ | 3-8 |
| Stability | स्थिर | ट्यूनिंग की आवश्यकता |

---

## 🧩 ट्रबलशूटिंग

| समस्या | समाधान |
|---------|-----|
| **OOM errors** | `num_training_frames` घटाएँ, `generator_update_interval` घटाएँ, या batch size कम करें |
| **Fake score सीख नहीं रहा** | `fake_score_lr` बढ़ाएँ या अलग scheduler उपयोग करें |
| **Generator overfitting** | `generator_update_interval` को 10 तक बढ़ाएँ |
| **3-step गुणवत्ता खराब** | पहले 2-step के लिए "1000,500" आज़माएँ |
| **ट्रेनिंग अस्थिर** | learning rates कम करें, डेटा गुणवत्ता जांचें |

---

## 🔬 उन्नत विकल्प

प्रयोग करने वालों के लिए:

```json
"distillation_config": {
    "dmd_denoising_steps": "1000,666,333",
    "generator_update_interval": 4,
    "fake_score_guidance_scale": 1.2,
    "num_training_frames": 28,
    "same_step_across_blocks": true,
    "timestep_shift": 7.0
}
```

> ⚠️ संसाधन-सीमित प्रोजेक्ट्स के लिए DMD की मूल FastVideo इम्प्लीमेंटेशन का उपयोग करने की सलाह है, क्योंकि यह sequence-parallel और video-sparse attention (VSA) सपोर्ट करती है जिससे रनटाइम काफी अधिक कुशल होता है।
