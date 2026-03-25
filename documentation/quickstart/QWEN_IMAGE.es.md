## Guía rápida de Qwen Image

> 🆕 ¿Buscas los checkpoints de edición? Consulta la [guía rápida de Qwen Image Edit](./QWEN_EDIT.md) para instrucciones de entrenamiento con referencia emparejada.

En este ejemplo, entrenaremos un LoRA para Qwen Image, un modelo visión‑lenguaje de 20B parámetros. Debido a su tamaño, necesitaremos técnicas agresivas de optimización de memoria.

Una GPU de 24GB es el mínimo absoluto, y aun así necesitarás cuantización extensa y configuración cuidadosa. Se recomiendan encarecidamente 40GB+ para una experiencia más fluida.

Al entrenar en 24G, las validaciones se quedarán sin memoria a menos que uses menor resolución o un nivel de cuantización agresivo más allá de int8.

### Requisitos de hardware

Qwen Image es un modelo de 20B parámetros con un codificador de texto sofisticado que por sí solo consume ~16GB de VRAM antes de la cuantización. El modelo usa un VAE personalizado con 16 canales latentes.

**Limitaciones importantes:**
- **No está soportado en AMD ROCm ni MacOS** por falta de flash attention eficiente
- Tamaño de lote > 1 no funciona correctamente por ahora; usa acumulación de gradiente en su lugar
- TREAD (Text-Representation Enhanced Adversarial Diffusion) aún no está soportado

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

- `model_type` - Configúralo en `lora`.
- `lora_type` - Configúralo en `standard` para PEFT LoRA o `lycoris` para LoKr.
- `model_family` - Configúralo en `qwen_image`.
- `model_flavour` - Configúralo en `v1.0`.
- `output_dir` - Configúralo al directorio donde quieres guardar tus checkpoints y las imágenes de validación. Se recomienda usar una ruta completa aquí.
- `train_batch_size` - Ajústalo según la VRAM disponible. Los overrides actuales de Qwen en SimpleTuner admiten tamaños de lote mayores que 1.
- `gradient_accumulation_steps` - Configúralo en 2-8 si quieres un batch efectivo mayor sin subir la VRAM por paso.
- `validation_resolution` - Debes configurarlo en `1024x1024` o menor por restricciones de memoria.
  - 24G no puede manejar validaciones 1024x1024 actualmente - tendrás que reducir el tamaño
  - Se pueden especificar otras resoluciones separándolas con comas: `1024x1024,768x768,512x512`
- `validation_guidance` - Usa un valor alrededor de 3.0-4.0 para buenos resultados.
- `validation_num_inference_steps` - Usa alrededor de 30.
- `use_ema` - Configurar esto en `true` ayudará a obtener resultados más suaves pero usa más memoria.

- `optimizer` - Usa `optimi-lion` para buenos resultados, o `adamw-bf16` si tienes memoria de sobra.
- `mixed_precision` - Debe configurarse en `bf16` para Qwen Image.
- `gradient_checkpointing` - **Obligatorio** habilitar (`true`) para un uso razonable de memoria.
- `base_model_precision` - **Muy recomendado** configurar en `int8-quanto` o `nf4-bnb` para tarjetas de 24GB.
- `quantize_via` - Configúralo en `cpu` para evitar OOM durante la cuantización en GPUs más pequeñas.
- `quantize_activations` - Mantén esto en `false` para conservar la calidad de entrenamiento.

Ajustes de optimización de memoria para GPUs de 24GB:
- `lora_rank` - Usa 8 o menos.
- `lora_alpha` - Iguala esto a tu valor de lora_rank.
- `flow_schedule_shift` - Configura en 1.73 (o experimenta entre 1.0-3.0).

Tu config.json se verá algo así para un setup mínimo:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "model_type": "lora",
    "model_family": "qwen_image",
    "model_flavour": "v1.0",
    "lora_type": "standard",
    "lora_rank": 8,
    "lora_alpha": 8,
    "output_dir": "output/models-qwen_image",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "validation_resolution": "1024x1024",
    "validation_guidance": 4.0,
    "validation_num_inference_steps": 30,
    "validation_seed": 42,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_step_interval": 100,
    "vae_batch_size": 1,
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 1000,
    "max_grad_norm": 0.01,
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "learning_rate": "1e-4",
    "gradient_checkpointing": "true",
    "base_model_precision": "int2-quanto",
    "quantize_via": "cpu",
    "quantize_activations": false,
    "flow_schedule_shift": 1.73,
    "disable_benchmark": false,
    "data_backend_config": "config/qwen_image/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ Usuarios multi-GPU pueden consultar [este documento](../OPTIONS.md#environment-configuration-variables) para información sobre cómo configurar la cantidad de GPUs a usar.

> ⚠️ **Crítico para GPUs de 24GB**: El codificador de texto por sí solo usa ~16GB de VRAM. Con cuantización `int2-quanto` o `nf4-bnb`, esto se puede reducir significativamente.

Para una comprobación rápida con una configuración conocida que funciona:

**Opción 1 (Recomendada - instalación con pip):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train example=qwen_image.peft-lora
```

**Opción 2 (Método de git clone):**
```bash
simpletuner train env=examples/qwen_image.peft-lora
```

**Opción 3 (Método heredado - aún funciona):**
```bash
ENV=examples/qwen_image.peft-lora ./train.sh
```

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

Para indicar al entrenador esta librería de prompts, añádela a tu config.json:
```json
  "validation_prompt_library": "config/user_prompt_library.json",
```

Un conjunto de prompts diverso ayudará a determinar si el modelo está aprendiendo correctamente:

```json
{
    "anime_style": "a breathtaking anime-style portrait with vibrant colors and expressive features",
    "chef_cooking": "a high-quality, detailed photograph of a sous-chef immersed in culinary creation",
    "portrait": "a lifelike and intimate portrait showcasing unique personality and charm",
    "cinematic": "a cinematic, visually stunning photo with dramatic and captivating presence",
    "elegant": "an elegant and timeless portrait exuding grace and sophistication",
    "adventurous": "a dynamic and adventurous photo captured in an exciting moment",
    "mysterious": "a mysterious and enigmatic portrait shrouded in shadows and intrigue",
    "vintage": "a vintage-style portrait evoking the charm and nostalgia of a bygone era",
    "artistic": "an artistic and abstract representation blending creativity with visual storytelling",
    "futuristic": "a futuristic and cutting-edge portrayal set against advanced technology"
}
```

#### Seguimiento de puntuaciones CLIP

Si deseas habilitar evaluaciones para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/CLIP_SCORES.md) para información sobre cómo configurar e interpretar las puntuaciones CLIP.

#### Pérdida de evaluación estable

Si deseas usar pérdida MSE estable para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/EVAL_LOSS.md) para información sobre cómo configurar e interpretar la pérdida de evaluación.

#### Vistas previas de validación

SimpleTuner admite la transmisión de vistas previas de validación intermedias durante la generación usando modelos Tiny AutoEncoder. Esto te permite ver las imágenes de validación generándose paso a paso en tiempo real mediante callbacks de webhook.

Para habilitarlo:
```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

**Requisitos:**
- Configuración de webhook
- Validación habilitada

Configura `validation_preview_steps` en un valor más alto (p. ej., 3 o 5) para reducir la sobrecarga del Tiny AutoEncoder. Con `validation_num_inference_steps=20` y `validation_preview_steps=5`, recibirás imágenes de vista previa en los pasos 5, 10, 15 y 20.

#### Desplazamiento del calendario de flujo

Qwen Image, como modelo de flow-matching, admite el desplazamiento del calendario de timesteps para controlar qué partes del proceso de generación se entrenan.

El parámetro `flow_schedule_shift` controla esto:
- Valores bajos (0.1-1.0): Enfoque en detalles finos
- Valores medios (1.0-3.0): Entrenamiento equilibrado (recomendado)
- Valores altos (3.0-6.0): Enfoque en grandes rasgos compositivos

##### Auto-shift
Puedes habilitar el shift de timesteps dependiente de la resolución con `--flow_schedule_auto_shift`, que usa valores de shift más altos para imágenes grandes y valores más bajos para imágenes pequeñas. Esto puede dar resultados de entrenamiento estables pero potencialmente mediocres.

##### Especificación manual
Se recomienda un valor `--flow_schedule_shift` de 1.73 como punto de partida para Qwen Image, aunque quizá necesites experimentar según tu dataset y objetivos.

#### Consideraciones del dataset

Es crucial contar con un dataset sustancial para entrenar tu modelo. Hay limitaciones en el tamaño del dataset, y debes asegurarte de que sea lo suficientemente grande para entrenar tu modelo de forma efectiva.

> ℹ️ Con pocas imágenes, podrías ver el mensaje **no images detected in dataset** - aumentar el valor de `repeats` superará esta limitación.

> ⚠️ **Importante**: Debido a limitaciones actuales, mantén `train_batch_size` en 1 y usa `gradient_accumulation_steps` para simular tamaños de lote mayores.

Crea un documento `--data_backend_config` (`config/multidatabackend.json`) que contenga esto:

```json
[
  {
    "id": "pseudo-camera-10k-qwen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/qwen_image",
    "disabled": false,
    "write_batch_size": 16
  }
]
```

> ℹ️ Usa `caption_strategy=textfile` si tienes archivos `.txt` que contienen captions.
> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).
> ℹ️ Nota el `write_batch_size` reducido para text embeds para evitar OOM.

Luego, crea un directorio `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

Esto descargará alrededor de 10k muestras de fotografías a tu directorio `datasets/pseudo-camera-10k`, que se creará automáticamente.

Tus imágenes de Dreambooth deben ir en el directorio `datasets/dreambooth-subject`.

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

</details>

### Ejecutar el entrenamiento

Desde el directorio de SimpleTuner, solo hay que ejecutar:

```bash
./train.sh
```

Esto iniciará el caché a disco de text embeds y salidas del VAE.

Para más información, consulta los documentos de [dataloader](../DATALOADER.md) y [tutorial](../TUTORIAL.md).

### Consejos de optimización de memoria

#### Configuración de VRAM más baja (mínimo 24GB)

La configuración de VRAM más baja de Qwen Image requiere aproximadamente 24GB:

- OS: Ubuntu Linux 24
- GPU: Un solo dispositivo NVIDIA CUDA (mínimo 24GB)
- Memoria del sistema: Se recomienda 64GB+
- Precisión del modelo base:
  - Para sistemas NVIDIA: `int2-quanto` o `nf4-bnb` (requerido para tarjetas de 24GB)
  - `int4-quanto` puede funcionar pero con menor calidad
- Optimizador: `optimi-lion` o `bnb-lion8bit-paged` para eficiencia de memoria
- Resolución: Comienza con 512px o 768px, sube a 1024px si la memoria lo permite
- Tamaño de lote: 1 (obligatorio por limitaciones actuales)
- Pasos de acumulación de gradiente: 2-8 para simular lotes más grandes
- Habilitar `--gradient_checkpointing` (obligatorio)
- Usa `--quantize_via=cpu` para evitar OOM durante el arranque
- Usa un rango LoRA pequeño (1-8)
- Configurar la variable de entorno `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ayuda a minimizar el uso de VRAM

**NOTA**: El pre-caché de embeddings del VAE y salidas del codificador de texto usará memoria significativa. Habilita `offload_during_startup=true` si encuentras problemas de OOM.

### Ejecutar inferencia en el LoRA después

Como Qwen Image es un modelo más nuevo, aquí hay un ejemplo funcional de inferencia:

<details>
<summary>Mostrar ejemplo de inferencia en Python</summary>

```python
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

model_id = 'Qwen/Qwen-Image'
adapter_id = 'your-username/your-lora-name'

# Load the pipeline
pipeline = QwenImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
pipeline.load_lora_weights(adapter_id)

# Optional: quantize the model to save VRAM
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

# Move to device
pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

# Generate an image
prompt = "Your test prompt here"
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=4.0,
    generator=torch.Generator(device='cuda').manual_seed(42),
    width=1024,
    height=1024,
).images[0]

image.save("output.png", format="PNG")
```
</details>

### Notas y consejos de solución de problemas

#### Limitaciones de tamaño de lote

Las compilaciones antiguas de diffusers para Qwen tenían problemas con batch size > 1 por el padding de embeddings de texto y el enmascarado de atención. Los overrides actuales de Qwen en SimpleTuner corrigen ambos puntos, así que los lotes mayores funcionan si tu VRAM lo permite.
- Aumenta `train_batch_size` solo después de confirmar que tienes memoria suficiente.
- Si todavía ves artefactos en una instalación antigua, actualiza y regenera cualquier embedding de texto obsoleto.

#### Cuantización

- `int2-quanto` brinda el mayor ahorro de memoria pero puede afectar la calidad
- `nf4-bnb` ofrece un buen equilibrio entre memoria y calidad
- `int4-quanto` es una opción intermedia
- Evita `int8` a menos que tengas 40GB+ de VRAM

#### Tasas de aprendizaje

Para entrenamiento LoRA:
- LoRAs pequeñas (rango 1-8): usa tasas alrededor de 1e-4
- LoRAs grandes (rango 16-32): usa tasas alrededor de 5e-5
- Con optimizador Prodigy: empieza con 1.0 y deja que se adapte

#### Artefactos de imagen

Si encuentras artefactos:
- Baja la tasa de aprendizaje
- Aumenta los pasos de acumulación de gradiente
- Asegúrate de que tus imágenes sean de alta calidad y estén bien preprocesadas
- Considera usar resoluciones más bajas al inicio

#### Entrenamiento de múltiples resoluciones

Comienza el entrenamiento en resoluciones más bajas (512px o 768px) para acelerar el aprendizaje inicial, luego ajusta fino en 1024px. Habilita `--flow_schedule_auto_shift` al entrenar en diferentes resoluciones.

### Limitaciones de plataforma

**No soportado en:**
- AMD ROCm (no tiene implementación eficiente de flash attention)
- Apple Silicon/MacOS (limitaciones de memoria y atención)
- GPUs de consumo con menos de 24GB de VRAM

### Problemas conocidos actuales

1. Tamaño de lote > 1 no funciona correctamente (usa acumulación de gradiente)
2. TREAD aún no está soportado
3. Alto uso de memoria del codificador de texto (~16GB antes de cuantización)
4. Problemas de manejo de longitud de secuencia ([issue upstream](https://github.com/huggingface/diffusers/issues/12075))

Para ayuda adicional y solución de problemas, consulta la [documentación de SimpleTuner](/documentation) o únete al Discord de la comunidad.
