# TwinFlow (RCGM) Few-Step Training

TwinFlow एक हल्का, standalone few-step recipe है जो **recursive consistency gradient matching (RCGM)** पर आधारित है। यह **मुख्य `distillation_method` विकल्पों का हिस्सा नहीं** है—आप इसे सीधे `twinflow_*` flags के जरिए सक्षम करते हैं। hub से खींचे गए configs में loader `twinflow_enabled` को डिफ़ॉल्ट रूप से `false` रखता है ताकि सामान्य transformer configs अप्रभावित रहें।

SimpleTuner में TwinFlow:
* Flow-matching केवल तभी, जब तक आप diffusion मॉडल्स को `diff2flow_enabled` + `twinflow_allow_diff2flow` से ब्रिज न करें।
* EMA teacher डिफ़ॉल्ट है; teacher/CFG passes के आसपास RNG capture/restore **हमेशा चालू** रहता है ताकि reference TwinFlow run जैसा व्यवहार मिले।
* negative-time semantics के लिए optional sign embeddings transformers पर wired हैं, लेकिन केवल तब उपयोग होते हैं जब `twinflow_enabled` true हो; HF configs में flag न होने पर कोई व्यवहार परिवर्तन नहीं होता।
* मौजूदा losses केवल RCGM + real-velocity का उपयोग करते हैं (adversarial/fake branch बंद रहता है); guidance `0.0` पर 1–4 step generation अपेक्षित है।
* W&B logging डिबग के लिए experimental TwinFlow trajectory scatter (थ्योरी अभी unverified) emit कर सकता है।

---

## क्विक कॉन्फ़िग (flow-matching मॉडल)

अपने सामान्य config में TwinFlow bits जोड़ें (`distillation_method` unset/null रखें):

```json
{
  "model_family": "sd3",
  "model_type": "lora",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
  "output_dir": "output/sd3-twinflow",

  "distillation_method": null,
  "use_ema": true,

  "twinflow_enabled": true,
  "twinflow_target_step_count": 2,
  "twinflow_estimate_order": 2,
  "twinflow_enhanced_ratio": 0.5,
  "twinflow_delta_t": 0.01,
  "twinflow_target_clamp": 1.0,

  "learning_rate": 1e-4,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "mixed_precision": "bf16",
  "validation_guidance": 0.0,
  "validation_num_inference_steps": 2
}
```

Diffusion मॉडल्स (epsilon/v prediction) के लिए स्पष्ट opt-in करें:

```json
{
  "prediction_type": "epsilon",
  "diff2flow_enabled": true,
  "twinflow_allow_diff2flow": true
}
```

> TwinFlow जानबूझकर सरल रखा गया है: कोई अतिरिक्त discriminator या fake branch wired नहीं है; केवल RCGM और real-velocity terms उपयोग होते हैं।

---

## क्या उम्मीद करें (पेपर डेटा)

arXiv:2512.05150 (PDF text) से:
* Inference benchmarks **single A100 (BF16)** पर 1024×1024 में throughput (batch=10) और latency (batch=1) के साथ मापे गए थे। सटीक नंबर टेक्स्ट में नहीं थे, सिर्फ हार्डवेयर सेटिंग थी।
* Qwen-Image-20B (LoRA) और SANA-1.6B के लिए **GPU memory तुलना** (1024×1024) दिखाती है कि TwinFlow वहाँ फिट होता है जहाँ DMD2 / SANA-Sprint OOM हो सकते हैं।
* Training configs (Table 6) में **batch sizes 128/64/32/24** और **training steps 30k–60k (या 7k–10k छोटे रन)** सूचीबद्ध हैं; constant LR, EMA decay अक्सर 0.99।
* PDF में कुल GPU counts, node layouts, या wall-clock time रिपोर्ट नहीं किए गए हैं।

इन्हें दिशात्मक अपेक्षाएँ मानें, गारंटी नहीं। सटीक hardware/runtime के लिए लेखक पुष्टि की आवश्यकता होगी।

---

## प्रमुख विकल्प

* `twinflow_enabled`: RCGM auxiliary loss चालू करता है; `distillation_method` खाली रखें और scheduled sampling बंद रखें। यदि config में नहीं है तो डिफ़ॉल्ट `false`।
* `twinflow_target_step_count` (1–4 अनुशंसित): ट्रेनिंग को गाइड करता है और validation/inference में reuse होता है। guidance को `0.0` पर मजबूर किया जाता है क्योंकि CFG baked in है।
* `twinflow_estimate_order`: RCGM rollout के लिए integration order (डिफ़ॉल्ट 2)। उच्च मान अधिक teacher passes जोड़ते हैं।
* `twinflow_enhanced_ratio`: teacher cond/uncond predictions से optional CFG-style target refinement (डिफ़ॉल्ट 0.5; 0.0 सेट करें ताकि disable हो)। captured RNG का उपयोग करता है ताकि cond/uncond align रहें।
* `twinflow_delta_t` / `twinflow_target_clamp`: recursive target को आकार देते हैं; डिफ़ॉल्ट्स पेपर की स्थिर सेटिंग्स के अनुरूप हैं।
* `use_ema` + `twinflow_require_ema` (डिफ़ॉल्ट true): EMA weights teacher के रूप में उपयोग होते हैं। `twinflow_allow_no_ema_teacher: true` केवल तब सेट करें जब आप student-as-teacher गुणवत्ता स्वीकार करते हों।
* `twinflow_allow_diff2flow`: epsilon/v-prediction मॉडल्स को ब्रिज करने देता है जब `diff2flow_enabled` भी true हो।
* RNG capture/restore: reference TwinFlow implementation जैसा व्यवहार पाने के लिए हमेशा सक्षम रहता है। opt-out स्विच नहीं है।
* Sign embeddings: जब `twinflow_enabled` true होता है, मॉडल `twinflow_time_sign` को उन transformers में पास करते हैं जो `timestep_sign` सपोर्ट करते हैं; अन्यथा कोई अतिरिक्त embedding नहीं।

---

## ट्रेनिंग और वैलिडेशन फ्लो

1. सामान्य flow-matching रन की तरह ट्रेन करें (कोई distiller जरूरी नहीं)। EMA मौजूद होना चाहिए जब तक आप स्पष्ट रूप से opt out न करें; RNG alignment स्वचालित है।
2. Validation स्वचालित रूप से **TwinFlow/UCGM scheduler** पर स्वैप करता है और `twinflow_target_step_count` steps के साथ `guidance_scale=0.0` उपयोग करता है।
3. Exported pipelines के लिए, scheduler को मैन्युअली attach करें:

```python
from simpletuner.helpers.training.custom_schedule import TwinFlowScheduler

pipe = ...  # your loaded diffusers pipeline
pipe.scheduler = TwinFlowScheduler(num_train_timesteps=1000, prediction_type="flow_matching", shift=1.0)
pipe.scheduler.set_timesteps(num_inference_steps=2, device=pipe.device)
result = pipe(prompt="A cinematic portrait, 35mm", guidance_scale=0.0, num_inference_steps=2).images
```

---

## लॉगिंग

* जब `report_to=wandb` और `twinflow_enabled=true` हो, trainer एक experimental TwinFlow trajectory scatter (σ vs tt vs sign) लॉग कर सकता है। यह केवल डिबगिंग के लिए है और UI में “experimental/theory unverified” टैग के साथ दिखता है।

---

## ट्रबलशूटिंग

* **flow-matching त्रुटि**: TwinFlow के लिए `prediction_type=flow_matching` आवश्यक है जब तक आप `diff2flow_enabled` + `twinflow_allow_diff2flow` सक्षम न करें।
* **EMA आवश्यक**: `use_ema` सक्षम करें या `twinflow_allow_no_ema_teacher: true` / `twinflow_require_ema: false` सेट करें यदि आप student-teacher fallback स्वीकार करते हैं।
* **1 step पर गुणवत्ता फ्लैट**: `twinflow_target_step_count: 2`–`4` आज़माएँ, guidance `0.0` रखें, और यदि overfitting हो तो `twinflow_enhanced_ratio` कम करें।
* **Teacher/Student drift**: RNG alignment हमेशा सक्षम है; drift मॉडल mismatch से होना चाहिए, stochastic differences से नहीं। यदि आपका transformer `timestep_sign` सपोर्ट नहीं करता, तो `twinflow_enabled` बंद रखें या इसे consume करने के लिए मॉडल अपडेट करें।
