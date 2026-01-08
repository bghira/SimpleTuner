# Inicio rápido de destilación DMD (SimpleTuner)

En este ejemplo, entrenaremos un **estudiante de 3 pasos** usando **DMD (Distribution Matching Distillation)** a partir de un modelo teacher grande de flow-matching como [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).

Características de DMD:

* **Generador (Estudiante)**: Aprende a igualar al teacher en menos pasos
* **Fake Score Transformer**: Discrimina entre salidas del teacher y del estudiante
* **Simulación multi-step**: Modo opcional de consistencia entrenamiento-inferencia

---

## ✅ Requisitos de hardware


⚠️ DMD consume mucha memoria debido al fake score transformer, que requiere mantener en memoria una segunda copia completa del modelo base.

Se recomienda intentar métodos de destilación LCM o DCM para el modelo Wan 14B en lugar de DMD si no tienes la VRAM necesaria.

Puede requerirse una NVIDIA B200 al destilar el modelo 14B sin soporte de sparse attention.

Usar entrenamiento de estudiante con LoRA puede reducir los requisitos sustancialmente, pero sigue siendo bastante exigente.

---

## 📦 Instalación

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.12 -m venv .venv && source .venv/bin/activate

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
    "ignore_final_epochs": true,
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

### Ajustes clave de DMD:

* **`dmd_denoising_steps`** – Timesteps objetivo para la simulación inversa (predeterminado: `1000,757,522` para un estudiante de 3 pasos).
* **`generator_update_interval`** – Ejecuta el replay caro del generador cada _N_ pasos del trainer. Auméntalo para intercambiar calidad por velocidad.
* **`fake_score_lr` / `fake_score_weight_decay` / `fake_score_betas`** – Hiperparámetros del optimizador para el fake score transformer.
* **`fake_score_guidance_scale`** – Guidance opcional classifier-free en la red de fake score (por defecto apagado).
* **`num_frame_per_block`, `same_step_across_blocks`, `last_step_only`** – Controlan cómo se programan los bloques temporales durante el rollout de auto-forzado.
* **`num_training_frames`** – Máximo de frames generados durante la simulación inversa (valores mayores mejoran fidelidad con costo de memoria).
* **`min_timestep_ratio`, `max_timestep_ratio`, `timestep_shift`** – Dan forma a la ventana de muestreo KL. Empareja estos valores con el schedule de flujo del teacher si te desvías de los defaults.

---

## 🎬 Dataset y dataloader

Para que DMD funcione bien, necesitas **datos diversos y de alta calidad**:

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
  "repeats": 0
}
```

> **Nota**: El dataset de Disney es **inadecuado** para DMD. **¡NO lo uses!** Está proporcionado solo con fines ilustrativos.

Necesitas:
> - Alto volumen (10k+ videos mínimo)
> - Contenido diverso (estilos, movimientos, sujetos distintos)
> - Alta calidad (sin artefactos de compresión)

Estos pueden generarse a partir del modelo parent.

---

## 🚀 Consejos de entrenamiento

1. **Mantén pequeño el intervalo del generador**: Empieza con `"generator_update_interval": 1`. Auméntalo solo si necesitas throughput y puedes tolerar actualizaciones más ruidosas.
2. **Monitorea ambas pérdidas**: Observa `dmd_loss` y `fake_score_loss` en wandb
3. **Frecuencia de validación**: DMD converge rápido, valida a menudo
4. **Gestión de memoria**:
   - Usa `gradient_checkpointing`
   - Reduce `train_batch_size` a 1
   - Considera `base_model_precision: "int8-quanto"`

---

## 📌 DMD vs DCM

| Característica | DCM | DMD |
|---------|-----|-----|
| Uso de memoria | Más bajo | Más alto (modelo de fake score) |
| Tiempo de entrenamiento | Más largo | Más corto (4k pasos típico) |
| Calidad | Buena | Excelente |
| Pasos de inferencia | 4-8+ | 3-8 |
| Estabilidad | Estable | Requiere ajuste |

---

## 🧩 Solución de problemas

| Problema | Solución |
|---------|-----|
| **Errores OOM** | Reduce `num_training_frames`, baja `generator_update_interval` o reduce batch size |
| **Fake score no aprende** | Aumenta `fake_score_lr` o usa otro scheduler |
| **Sobreajuste del generador** | Aumenta `generator_update_interval` a 10 |
| **Baja calidad en 3 pasos** | Prueba "1000,500" para 2 pasos primero |
| **Entrenamiento inestable** | Reduce tasas de aprendizaje, revisa la calidad de datos |

---

## 🔬 Opciones avanzadas

Para valientes que quieran experimentar:

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

> ⚠️ Se recomienda usar la implementación original de FastVideo de DMD para proyectos con recursos limitados, ya que soporta sequence-parallel y video-sparse attention (VSA) para un uso de runtime mucho más eficiente.
