## Guía rápida de Lumina2

En este ejemplo, entrenaremos un LoRA de Lumina2 o un ajuste fino de modelo completo.

### Requisitos de hardware

Lumina2 es un modelo de 2B parámetros, lo que lo hace mucho más accesible que modelos más grandes como Flux o SD3. El tamaño más pequeño del modelo implica:

Al entrenar un LoRA de rango 16, usa:
- Aproximadamente 12-14GB de VRAM para entrenamiento LoRA
- Aproximadamente 16-20GB de VRAM para ajuste fino completo
- Aproximadamente 20-30GB de RAM del sistema durante el arranque

Necesitarás:
- **Mínimo**: Una sola RTX 3060 12GB o RTX 4060 Ti 16GB
- **Recomendado**: RTX 3090, RTX 4090 o A100 para entrenamiento más rápido
- **RAM del sistema**: Al menos 32GB recomendados

### Requisitos previos

Asegúrate de tener python instalado; SimpleTuner funciona bien con 3.10 a 3.12.

Puedes comprobarlo ejecutando:

```bash
python --version
```

Si no tienes python 3.12 instalado en Ubuntu, puedes intentar lo siguiente:

```bash
apt -y install python3.13 python3.13-venv
```

#### Dependencias de la imagen de contenedor

Para Vast, RunPod y TensorDock (entre otros), lo siguiente funcionará en una imagen CUDA 12.2-12.8:

```bash
apt -y install nvidia-cuda-toolkit
```

### Instalación

Instala SimpleTuner vía pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

Para instalación manual o entorno de desarrollo, consulta la [documentación de instalación](../INSTALL.md).

### Configuración del entorno

Para ejecutar SimpleTuner, necesitas configurar un archivo de configuración, los directorios del dataset y del modelo, y un archivo de configuración del dataloader.

#### Archivo de configuración

Copia `config/config.json.example` a `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Allí, necesitarás modificar las siguientes variables:

- `model_type` - Configúralo en `lora` para entrenamiento LoRA o `full` para ajuste fino completo.
- `model_family` - Configúralo en `lumina2`.
- `output_dir` - Configúralo al directorio donde quieres guardar tus checkpoints y las imágenes de validación. Se recomienda usar una ruta completa aquí.
- `train_batch_size` - Puede ser 1-4 según la memoria de tu GPU y el tamaño del dataset.
- `validation_resolution` - Lumina2 soporta múltiples resoluciones. Opciones comunes: `1024x1024`, `512x512`, `768x768`.
- `validation_guidance` - Lumina2 usa classifier-free guidance. Valores de 3.5-7.0 funcionan bien.
- `validation_num_inference_steps` - 20-30 pasos funcionan bien para Lumina2.
- `gradient_accumulation_steps` - Puede usarse para simular tamaños de lote mayores. Un valor de 2-4 funciona bien.
- `optimizer` - Se recomienda `adamw_bf16`. `lion` y `optimi-stableadamw` también funcionan bien.
- `mixed_precision` - Mantén esto en `bf16` para mejores resultados.
- `gradient_checkpointing` - Configúralo en `true` para ahorrar VRAM.
- `learning_rate` - Para LoRA: `1e-4` a `5e-5`. Para ajuste fino completo: `1e-5` a `1e-6`.

#### Configuración de ejemplo de Lumina2

Esto va en `config.json`

<details>
<summary>Ver ejemplo de config</summary>

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

Para entrenamiento Lycoris, cambia `lora_type` a `lycoris`

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

#### Prompts de validación

Dentro de `config/config.json` está el "prompt de validación principal". Además, crea un archivo de biblioteca de prompts:

```json
{
  "portrait": "a high-quality portrait photograph with natural lighting",
  "landscape": "a breathtaking landscape photograph with dramatic lighting",
  "artistic": "an artistic rendering with vibrant colors and creative composition",
  "detailed": "a highly detailed image with sharp focus and rich textures",
  "stylized": "a stylized illustration with unique artistic flair"
}
```

Agrega a tu config:
```json
{
  "--user_prompt_library": "config/user_prompt_library.json"
}
```

#### Consideraciones del dataset

Lumina2 se beneficia de datos de entrenamiento de alta calidad. Crea un `--data_backend_config` (`config/multidatabackend.json`):

> 💡 **Tip:** Para datasets grandes donde el espacio en disco es un problema, puedes usar `--vae_cache_disable` para realizar codificación VAE en línea sin cachear los resultados en disco.

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

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

Crea el directorio de tu dataset. Asegúrate de actualizar esta ruta con tu ubicación real.

```bash
mkdir -p /datasets/training
</details>

# Place your images and caption files in /datasets/training/
```

Los archivos de caption deben tener el mismo nombre que la imagen con extensión `.txt`.

#### Iniciar sesión en WandB

SimpleTuner tiene soporte de tracking **opcional**, centrado principalmente en Weights & Biases. Puedes desactivarlo con `report_to=none`.

Para habilitar wandb, ejecuta los siguientes comandos:

```bash
wandb login
```

#### Iniciar sesión en Huggingface Hub

Para subir checkpoints a Huggingface Hub, asegúrate de:
```bash
huggingface-cli login
```

### Ejecutar el entrenamiento

Desde el directorio de SimpleTuner, tienes varias opciones para iniciar el entrenamiento:

**Opción 1 (Recomendada - instalación con pip):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
simpletuner train
```

**Opción 2 (Método de git clone):**
```bash
simpletuner train
```

**Opción 3 (Método heredado - aún funciona):**
```bash
./train.sh
```

Esto iniciará el caché a disco de text embeds y salidas del VAE.

## Consejos de entrenamiento para Lumina2

### Tasas de aprendizaje

#### Entrenamiento LoRA
- Comienza con `1e-4` y ajusta según resultados
- Lumina2 entrena rápido, así que monitorea de cerca las primeras iteraciones
- Rangos 8-32 funcionan bien para la mayoría de casos, 64-128 pueden requerir monitoreo cercano, y 256-512 pueden ser útiles para entrenar tareas nuevas en el modelo

#### Ajuste fino completo
- Usa tasas de aprendizaje más bajas: `1e-5` a `5e-6`
- Considera usar EMA (Exponential Moving Average) para estabilidad
- Se recomienda clipping de gradiente (`max_grad_norm`) de 1.0

### Consideraciones de resolución

Lumina2 soporta resoluciones flexibles:
- Entrenar a 1024x1024 proporciona la mejor calidad
- El entrenamiento de resolución mixta (512px, 768px, 1024px) no ha sido probado en campo para impacto de calidad
- El bucketing de relación de aspecto funciona bien con Lumina2

### Duración del entrenamiento

Debido al tamaño eficiente de 2B parámetros de Lumina2:
- El entrenamiento LoRA a menudo converge en 500-2000 pasos
- El ajuste fino completo puede necesitar 2000-5000 pasos
- Monitorea imágenes de validación con frecuencia ya que el modelo entrena rápido

### Problemas comunes y soluciones

1. **El modelo converge demasiado rápido**: baja la tasa de aprendizaje, cambia de Lion a AdamW
2. **Artefactos en imágenes generadas**: asegura datos de entrenamiento de alta calidad y considera bajar la tasa de aprendizaje
3. **Out of memory**: habilita gradient checkpointing y reduce el tamaño de lote
4. **Se sobreajusta fácilmente**: usa datasets de regularización

## Consejos de inferencia

### Usar tu modelo entrenado

Los modelos Lumina2 pueden usarse con:
- Librería Diffusers directamente
- ComfyUI con nodos apropiados
- Otros frameworks de inferencia que soporten modelos basados en Gemma2

### Ajustes óptimos de inferencia

- Guidance scale: 4.0-6.0
- Pasos de inferencia: 20-50
- Usa la misma resolución con la que entrenaste para mejores resultados

## Notas

### Ventajas de Lumina2

- Entrenamiento rápido gracias al tamaño de 2B parámetros
- Buena relación calidad/tamaño
- Soporta varios modos de entrenamiento (LoRA, LyCORIS, full)
- Uso eficiente de memoria

### Limitaciones actuales

- Aún no hay soporte de ControlNet
- Limitado a generación texto‑a‑imagen
- Requiere alta calidad de captions para mejores resultados

### Optimización de memoria

A diferencia de modelos más grandes, Lumina2 normalmente no requiere:
- Cuantización del modelo
- Técnicas extremas de optimización de memoria
- Estrategias complejas de precisión mixta
