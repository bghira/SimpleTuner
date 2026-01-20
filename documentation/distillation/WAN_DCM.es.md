# Inicio rápido de destilación DCM (SimpleTuner)

En este ejemplo, entrenaremos un **estudiante de 4 pasos** usando **destilación DCM** a partir de un modelo teacher de flow-matching grande como [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).

DCM soporta:

* Modo **Semantic**: flow-matching estándar con CFG integrado.
* Modo **Fine**: supervisión adversarial basada en GAN (experimental).

---

## ✅ Requisitos de hardware

| Modelo     | Batch size | VRAM mín. | Notas                                  |
| --------- | ---------- | -------- | -------------------------------------- |
| Wan 1.3B  | 1          | 12 GB    | GPU tier A5000 / 3090+                 |
| Wan 14B   | 1          | 24 GB    | Más lento; usa `--offload_during_startup` |
| Modo fine | 1          | +10%     | Discriminador se ejecuta por GPU       |

> ⚠️ Mac y Apple silicon son lentos y no se recomiendan. Tendrás runtimes de 10 min/paso incluso en modo semantic.

---

## 📦 Instalación

Los mismos pasos que en la guía de Wan:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv && source .venv/bin/activate

# Install with automatic platform detection
pip install -e .
```

**Nota:** setup.py detecta automáticamente tu plataforma (CUDA/ROCm/Apple) e instala las dependencias apropiadas.

---

## 📁 Configuración

Edita tu `config/config.json`:

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

### Opcional:

* Para **modo fine**, solo cambia `"mode": "fine"`.
  - Este modo es actualmente experimental en SimpleTuner y requiere pasos extra para usarlo, que aún no se describen en esta guía.

---

## 🎬 Dataset y dataloader

Reutiliza el dataset Disney y el JSON `data_backend_config` del quickstart de Wan.

> **Nota**: Este dataset es inadecuado para destilación; se requiere **mucha** más diversidad y volumen de datos para tener éxito.

Asegúrate de:

* `num_frames`: 75–81
* `resolution`: 480
* `crop`: false (deja los videos sin recorte)
* `repeats`: 0 por ahora

---

## 📌 Notas

* El **modo semantic** es estable y recomendado para la mayoría de casos.
* El **modo fine** añade realismo, pero necesita más pasos y ajuste y el nivel de soporte actual en SimpleTuner no es muy bueno.

---

## 🧩 Solución de problemas

| Problema                      | Solución                                                                  |
| ---------------------------- | -------------------------------------------------------------------- |
| **Resultados borrosos**       | Usa más `euler_steps` o aumenta `multiphase`                       |
| **La validación empeora**  | Usa `validation_guidance: 1.0`                                       |
| **OOM en modo fine**         | Baja `train_batch_size`, reduce niveles de precisión o usa una GPU más grande |
| **Modo fine no converge** | No uses modo fine; no está muy probado en SimpleTuner      |
