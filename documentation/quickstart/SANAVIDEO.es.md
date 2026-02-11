## Guía rápida de Sana Video

En este ejemplo, entrenaremos el modelo Sana Video 2B 480p.

### Requisitos de hardware

Sana Video usa el autoencoder Wan y procesa secuencias de 81 frames a 480p por defecto. Espera un uso de memoria similar al de otros modelos de video; habilita gradient checkpointing temprano y aumenta `train_batch_size` solo después de verificar margen de VRAM.

### Offloading de memoria (opcional)

Si estás cerca del límite de VRAM, habilita offloading agrupado en tu config:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Los usuarios de CUDA se benefician de `--group_offload_use_stream`; otros backends lo ignoran automáticamente.
- Omite `--group_offload_to_disk_path` a menos que la RAM del sistema sea limitada — el staging a disco es más lento pero mantiene ejecuciones estables.
- Desactiva `--enable_model_cpu_offload` al usar group offloading.

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

Para Vast, RunPod y TensorDock (entre otros), lo siguiente funcionará en una imagen CUDA 12.2-12.8 para habilitar la compilación de extensiones CUDA:

```bash
apt -y install nvidia-cuda-toolkit
```

### Instalación

Instala SimpleTuner vía pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Para instalación manual o entorno de desarrollo, consulta la [documentación de instalación](../INSTALL.md).

### Configuración del entorno

Para ejecutar SimpleTuner, necesitas configurar un archivo de configuración, los directorios del dataset y del modelo, y un archivo de configuración del dataloader.

#### Archivo de configuración

Un script experimental, `configure.py`, puede permitirte omitir por completo esta sección mediante una configuración interactiva paso a paso. Incluye funciones de seguridad que ayudan a evitar errores comunes.

**Nota:** Esto no configura tu dataloader. Aún tendrás que hacerlo manualmente más adelante.

Para ejecutarlo:

```bash
simpletuner configure
```

> ⚠️ Para usuarios ubicados en países donde Hugging Face Hub no es fácilmente accesible, debes agregar `HF_ENDPOINT=https://hf-mirror.com` a tu `~/.bashrc` o `~/.zshrc` dependiendo de cuál `$SHELL` use tu sistema.

Si prefieres configurarlo manualmente:

Copia `config/config.json.example` a `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Allí, probablemente necesitarás modificar las siguientes variables:

- `model_type` - Configúralo en `full`.
- `model_family` - Configúralo en `sanavideo`.
- `pretrained_model_name_or_path` - Configúralo en `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`.
- `pretrained_vae_model_name_or_path` - Configúralo en `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`.
- `output_dir` - Configúralo al directorio donde quieres guardar tus checkpoints y los videos de validación. Se recomienda usar una ruta completa aquí.
- `train_batch_size` - empieza bajo para entrenamiento de video y aumenta solo después de confirmar el uso de VRAM.
- `validation_resolution` - Sana Video se entrega como modelo 480p; usa `832x480` o los buckets de aspecto que planeas validar.
- `validation_num_video_frames` - Configúralo en `81` para coincidir con la longitud del sampler por defecto.
- `validation_guidance` - Usa el valor que suelas seleccionar en inferencia para Sana Video.
- `validation_num_inference_steps` - Usa alrededor de 50 para calidad estable.
- `framerate` - Si se omite, Sana Video usa 16 fps; configúralo para que coincida con tu dataset.

- `optimizer` - Puedes usar cualquier optimizador con el que te sientas cómodo, pero usaremos `optimi-adamw` para este ejemplo.
- `mixed_precision` - Se recomienda configurarlo en `bf16` para la configuración de entrenamiento más eficiente, o `no` (pero consumirá más memoria y será más lento).
- `gradient_checkpointing` - Habilítalo para controlar el uso de VRAM.
- `use_ema` - Configurar esto en `true` ayudará mucho a obtener un resultado más suavizado junto con tu checkpoint principal.

Usuarios multi-GPU pueden consultar [este documento](../OPTIONS.md#environment-configuration-variables) para información sobre cómo configurar la cantidad de GPUs a usar.

Al final, tu config debería parecerse a:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "data_backend_config": "config/sanavideo/multidatabackend.json",
  "seed": 42,
  "output_dir": "output/sanavideo",
  "max_train_steps": 400000,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "tracker_project_name": "video-training",
  "tracker_run_name": "sanavideo-2b-480p",
  "report_to": "wandb",
  "model_type": "full",
  "pretrained_model_name_or_path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
  "pretrained_vae_model_name_or_path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
  "model_family": "sanavideo",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 200,
  "validation_resolution": "832x480",
  "validation_negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "validation_guidance": 6.0,
  "validation_num_inference_steps": 50,
  "validation_num_video_frames": 81,
  "validation_prompt": "A short video of a small, fluffy animal exploring a sunny room with soft window light and gentle camera motion.",
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.0,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "bf16",
  "vae_batch_size": 1,
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "framerate": 16,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

### Opcional: regularizador temporal CREPA

Si tus videos muestran flicker o sujetos que se desvían, habilita CREPA:
- En **Training → Loss functions**, activa **CREPA**.
- Valores sugeridos: **Block Index = 10**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- Mantén el codificador por defecto (`dinov2_vitg14`, tamaño `518`) a menos que necesites una opción más pequeña (`dinov2_vits14` + `224`) para ahorrar VRAM.
- La primera ejecución descarga DINOv2 vía torch hub; cachea o precarga si estás offline.
- Solo activa **Drop VAE Encoder** cuando entrenes completamente desde latentes en caché; déjalo apagado si aún codificas píxeles.

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

#### Prompts de validación

Dentro de `config/config.json` está el "prompt de validación principal", que suele ser el instance_prompt principal en el que estás entrenando para tu único sujeto o estilo. Además, se puede crear un archivo JSON que contiene prompts adicionales para ejecutar durante las validaciones.

El archivo de ejemplo `config/user_prompt_library.json.example` tiene el siguiente formato:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

Los apodos son el nombre de archivo de la validación, así que mantenlos cortos y compatibles con tu sistema de archivos.

Para indicar al entrenador esta librería de prompts, añádela a TRAINER_EXTRA_ARGS agregando una nueva línea al final de `config.json`:

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

Un conjunto de prompts diverso ayudará a determinar si el modelo colapsa a medida que entrena. En este ejemplo, la palabra `<token>` debe reemplazarse por el nombre de tu sujeto (instance_prompt).

```json
{
    "anime_<token>": "a breathtaking anime-style video featuring <token>, capturing her essence with vibrant colors, dynamic motion, and expressive storytelling",
    "chef_<token>": "a high-quality, detailed video of <token> as a sous-chef, immersed in the art of culinary creation with captivating close-ups and engaging sequences",
    "just_<token>": "a lifelike and intimate video portrait of <token>, showcasing her unique personality and charm through nuanced movement and expression",
    "cinematic_<token>": "a cinematic, visually stunning video of <token>, emphasizing her dramatic and captivating presence through fluid camera movements and atmospheric effects",
    "elegant_<token>": "an elegant and timeless video portrait of <token>, exuding grace and sophistication with smooth transitions and refined visuals",
    "adventurous_<token>": "a dynamic and adventurous video featuring <token>, captured in an exciting, action-filled sequence that highlights her energy and spirit",
    "mysterious_<token>": "a mysterious and enigmatic video portrait of <token>, shrouded in shadows and intrigue with a narrative that unfolds in subtle, cinematic layers",
    "vintage_<token>": "a vintage-style video of <token>, evoking the charm and nostalgia of a bygone era through sepia tones and period-inspired visual storytelling",
    "artistic_<token>": "an artistic and abstract video representation of <token>, blending creativity with visual storytelling through experimental techniques and fluid visuals",
    "futuristic_<token>": "a futuristic and cutting-edge video portrayal of <token>, set against a backdrop of advanced technology with sleek, high-tech visuals",
    "woman": "a beautifully crafted video portrait of a woman, highlighting her natural beauty and unique features through elegant motion and storytelling",
    "man": "a powerful and striking video portrait of a man, capturing his strength and character with dynamic sequences and compelling visuals",
    "boy": "a playful and spirited video portrait of a boy, capturing youthful energy and innocence through lively scenes and engaging motion",
    "girl": "a charming and vibrant video portrait of a girl, emphasizing her bright personality and joy with colorful visuals and fluid movement",
    "family": "a heartwarming and cohesive family video, showcasing the bonds and connections between loved ones through intimate moments and shared experiences"
}
```

> ℹ️ Sana Video es un modelo de flow-matching; los prompts más cortos podrían no tener suficiente información para que el modelo haga un buen trabajo. Usa prompts descriptivos siempre que sea posible.

#### Seguimiento de puntuaciones CLIP

Esto no debería habilitarse para entrenamiento de modelos de video por ahora.

</details>

# Pérdida de evaluación estable

Si deseas usar pérdida MSE estable para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/EVAL_LOSS.md) para información sobre cómo configurar e interpretar la pérdida de evaluación.

#### Vistas previas de validación

SimpleTuner admite la transmisión de vistas previas de validación intermedias durante la generación usando modelos Tiny AutoEncoder. Esto te permite ver los videos de validación generándose paso a paso en tiempo real mediante callbacks de webhook.

Para habilitarlo:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```
</details>

**Requisitos:**

- Configuración de webhook
- Validación habilitada

Configura `validation_preview_steps` en un valor más alto (p. ej., 3 o 5) para reducir la sobrecarga del Tiny AutoEncoder. Con `validation_num_inference_steps=20` y `validation_preview_steps=5`, recibirás frames de vista previa en los pasos 5, 10, 15 y 20.

#### Calendario de flow-matching

Sana Video usa el calendario canónico de flow-matching del checkpoint. Los overrides de shift proporcionados por el usuario se ignoran; deja `flow_schedule_shift` y `flow_schedule_auto_shift` sin configurar para este modelo.

#### Entrenamiento con modelo cuantizado

Las opciones de precisión (bf16, int8, fp8) están disponibles en la config; ajústalas a tu hardware y vuelve a mayor precisión si encuentras inestabilidades.

#### Consideraciones del dataset

Hay pocas limitaciones en el tamaño del dataset aparte de cuánto cómputo y tiempo tomará procesar y entrenar.

Debes asegurarte de que el dataset sea lo suficientemente grande para entrenar tu modelo de forma efectiva, pero no tan grande para el cómputo disponible.

Ten en cuenta que el tamaño mínimo del dataset es `train_batch_size * gradient_accumulation_steps` y también mayor que `vae_batch_size`. El dataset no será utilizable si es demasiado pequeño.

> ℹ️ Con pocas muestras, podrías ver el mensaje **no samples detected in dataset** - aumentar el valor de `repeats` superará esta limitación.

Dependiendo del dataset que tengas, necesitarás configurar el directorio del dataset y el archivo de configuración del dataloader de manera diferente.

En este ejemplo, usaremos [video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized) como dataset.

Crea un documento `--data_backend_config` (`config/multidatabackend.json`) que contenga esto:

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/sanavideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 81,
        "min_frames": 81,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sanavideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

- En la subsección `video`, tenemos las siguientes claves:
  - `num_frames` (opcional, int) es cuántos frames de datos entrenaremos.
  - `min_frames` (opcional, int) determina la longitud mínima de un video que se considerará para entrenamiento.
  - `max_frames` (opcional, int) determina la longitud máxima de un video que se considerará para entrenamiento.
  - `is_i2v` (opcional, bool) determina si se hará entrenamiento i2v en un dataset.
  - `bucket_strategy` (opcional, string) determina cómo se agrupan los videos en buckets:
    - `aspect_ratio` (predeterminado): Agrupar solo por relación de aspecto espacial (p. ej., `1.78`, `0.75`).
    - `resolution_frames`: Agrupar por resolución y recuento de frames en formato `WxH@F` (p. ej., `832x480@81`). Útil para datasets de resolución/duración mixta.
  - `frame_interval` (opcional, int) al usar `resolution_frames`, redondea recuentos de frames a este intervalo.

Luego, crea un directorio `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset sayakpaul/video-dataset-disney-organized --local-dir=disney-black-and-white
popd
```

Esto descargará todas las muestras de video de Disney a tu directorio `datasets/disney-black-and-white`, que se creará automáticamente.

#### Iniciar sesión en WandB y Huggingface Hub

Querrás iniciar sesión en WandB y HF Hub antes de empezar el entrenamiento, especialmente si usas `--push_to_hub` y `--report_to=wandb`.

Si vas a subir elementos a un repositorio Git LFS manualmente, también deberías ejecutar `git config --global credential.helper store`

Ejecuta los siguientes comandos:

```bash
wandb login
```

y

```bash
huggingface-cli login
```

Sigue las instrucciones para iniciar sesión en ambos servicios.

### Ejecutar el entrenamiento

Desde el directorio de SimpleTuner, tienes varias opciones para iniciar el entrenamiento:

**Opción 1 (Recomendada - instalación con pip):**

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
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

Para más información, consulta los documentos de [dataloader](../DATALOADER.md) y [tutorial](../TUTORIAL.md) documents.

## Notas y consejos de solución de problemas

### Valores predeterminados de validación

- Sana Video usa por defecto 81 frames y 16 fps cuando no se proporcionan ajustes de validación.
- La ruta del autoencoder Wan debe coincidir con la ruta del modelo base; mantenlas alineadas para evitar errores de carga.

### Pérdida con máscara

Si estás entrenando un sujeto o estilo y te gustaría enmascarar uno u otro, consulta la sección de [entrenamiento con pérdida enmascarada](../DREAMBOOTH.md#masked-loss) de la guía de Dreambooth.
