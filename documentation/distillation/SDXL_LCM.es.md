# Inicio rápido de destilación LCM de SDXL (SimpleTuner)

En este ejemplo, entrenaremos un **estudiante SDXL de 4-8 pasos** usando **destilación LCM (Latent Consistency Model)** a partir de un modelo teacher SDXL preentrenado.

> **NOTA**: Se pueden usar otros modelos como base; SDXL solo se usa para ilustrar los conceptos de configuración para LCM.

LCM permite:
* Inferencia ultra rápida (4-8 pasos vs 25-50)
* Consistencia entre timesteps
* Salidas de alta calidad con pocos pasos

## 📦 Instalación

Sigue la [guía](../INSTALL.md) estándar de instalación de SimpleTuner:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv && source .venv/bin/activate

# Install with automatic platform detection
pip install -e .
```

**Nota:** setup.py detecta automáticamente tu plataforma (CUDA/ROCm/Apple) e instala las dependencias apropiadas.

Para entornos de contenedor (Vast, RunPod, etc.):
```bash
apt -y install nvidia-cuda-toolkit
```

---

## 📁 Configuración

Crea tu `config/config.json` para SDXL LCM:

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

### Opciones clave de configuración LCM:

- **`num_ddim_timesteps`**: Número de timesteps en el solver DDIM (50-100 típico)
- **`w_min/w_max`**: Rango de guidance scale para entrenamiento (1.0-12.0 para SDXL)
- **`loss_type`**: Usa "l2" o "huber" (huber es más robusto a outliers)
- **`timestep_scaling_factor`**: Escalado para condiciones de borde (predeterminado 10.0)
- **`validation_num_inference_steps`**: Prueba con tu conteo objetivo de pasos (4-8)
- **`validation_guidance`**: Establece en 0.0 para LCM (sin CFG en inferencia)

### Para entrenamiento cuantizado (menos VRAM):

Añade estas opciones para reducir el uso de memoria:
```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "optimizer": "optimi-lion"
}
```

---

## 🎬 Configuración del dataset

Crea `multidatabackend.json` en tu directorio de salida:

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

> **Importante**: La destilación LCM requiere datos diversos y de alta calidad. Se recomienda un mínimo de 10k+ imágenes para buenos resultados.

---

## 🚀 Entrenamiento

1. **Inicia sesión en servicios** (si usas funciones del hub):
   ```bash
   huggingface-cli login
   wandb login
   ```

2. **Inicia el entrenamiento**:
   ```bash
   bash train.sh
   ```

3. **Monitorea el progreso**:
   - Observa la reducción de la pérdida LCM
   - Las imágenes de validación deben mantener calidad a 4-8 pasos
   - El entrenamiento típicamente toma 5k-10k pasos

---

## 📊 Resultados esperados

| Métrica | Valor esperado | Notas |
| ------ | -------------- | ----- |
| Pérdida LCM | < 0.1 | Debería disminuir de forma constante |
| Calidad de validación | Buena a 4 pasos | Puede requerir guidance=0 |
| Tiempo de entrenamiento | 5-10 horas | En una sola A100 |
| Inferencia final | 4-8 pasos | vs 25-50 para SDXL base |

---

## 🧩 Solución de problemas

| Problema | Solución |
| ------- | -------- |
| **Errores OOM** | Reduce batch size, habilita gradient checkpointing, usa cuantización int8 |
| **Salidas borrosas** | Aumenta `num_ddim_timesteps`, revisa calidad de datos, reduce tasa de aprendizaje |
| **Convergencia lenta** | Aumenta la tasa de aprendizaje a 2e-4, asegura dataset diverso |
| **Validación se ve mal** | Usa `validation_guidance: 0.0`, revisa si estás usando el scheduler correcto |
| **Artefactos con pocos pasos** | Normal con <4 pasos, prueba entrenar más o ajustar `w_min/w_max` |

---

## 🔧 Consejos avanzados

1. **Entrenamiento multi-resolución**: SDXL se beneficia de entrenar en múltiples aspectos:
   ```json
   "validation_resolution": "1024x1024,1280x768,768x1280,1152x896,896x1152"
   ```

2. **Entrenamiento progresivo**: Empieza con más timesteps y luego reduce:
   - Semana 1: Entrena con `validation_num_inference_steps: 8`
   - Semana 2: Fine-tune con `validation_num_inference_steps: 4`

3. **Scheduler para inferencia**: Después de entrenar, usa el scheduler LCM:
   ```python
   from diffusers import LCMScheduler
   scheduler = LCMScheduler.from_pretrained(
       "stabilityai/stable-diffusion-xl-base-1.0",
       subfolder="scheduler"
   )
   ```

4. **Combinar con ControlNet**: LCM funciona bien con ControlNet para generación guiada a pocos pasos.

---

## 📚 Recursos adicionales

- [LCM Paper](https://arxiv.org/abs/2310.04378)
- [Docs de LCM en Diffusers](https://huggingface.co/docs/diffusers/using-diffusers/inference_with_lcm)
- [Más docs de SimpleTuner](../quickstart/SDXL.md)

---

## 🎯 Próximos pasos

Después de una destilación LCM exitosa:
1. Prueba tu modelo con varios prompts a 4-8 pasos
2. Prueba LCM-LoRA en diferentes modelos base
3. Experimenta con aún menos pasos (2-3) para casos de uso específicos
4. Considera fine-tuning con datos de dominio específico
