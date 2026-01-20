# Guía rápida de Z-Image [base / turbo]

En este ejemplo, entrenaremos un LoRA de Z-Image Turbo. Z-Image es un transformer de flow‑matching de 6B (aprox. la mitad del tamaño de Flux) con flavours base y turbo. Turbo espera un adaptador asistente; SimpleTuner puede cargarlo automáticamente.

## Requisitos de hardware

Z-Image necesita menos memoria que Flux pero aún se beneficia de GPUs potentes. Cuando entrenas cada componente de un LoRA de rango 16 (MLP, proyecciones, bloques del transformer), típicamente usa:

- ~32-40G de VRAM cuando no cuantizas el modelo base
- ~16-24G de VRAM al cuantizar a int8 + pesos base/LoRA en bf16
- ~10–12G de VRAM al cuantizar a NF4 + pesos base/LoRA en bf16

Además, Ramtorch y group offload pueden usarse para bajar aún más el uso de VRAM. Para usuarios multi-GPU, FSDP2 también permite correr en varias GPUs pequeñas.

Necesitarás:

- **el mínimo absoluto** es una sola **3080 10G** (con cuantización/offload agresivo)
- **un mínimo realista** es una sola 3090/4090 o V100/A6000
- **idealmente** varias 4090, A6000, L40S o mejor

Las GPUs de Apple no se recomiendan para entrenamiento.

### Offloading de memoria (opcional)

El offloading agrupado de módulos reduce drásticamente la presión de VRAM cuando el cuello de botella son los pesos del transformer. Puedes habilitarlo agregando los siguientes flags a `TRAINER_EXTRA_ARGS` (o en la página Hardware de la WebUI):

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Los streams solo son efectivos en CUDA; SimpleTuner los desactiva automáticamente en ROCm, MPS y CPU.
- No combines esto con otras estrategias de offload a CPU.
- Group offload no es compatible con cuantización Quanto.
- Prefiere un SSD/NVMe local rápido al hacer offload a disco.

## Requisitos previos

Asegúrate de tener python instalado; SimpleTuner funciona bien con 3.10 a 3.12.

Puedes comprobarlo ejecutando:

```bash
python --version
```

Si no tienes python 3.12 instalado en Ubuntu, puedes intentar lo siguiente:

```bash
apt -y install python3.13 python3.13-venv
```

### Dependencias de la imagen de contenedor

Para Vast, RunPod y TensorDock (entre otros), lo siguiente funcionará en una imagen CUDA 12.x para habilitar la compilación de extensiones CUDA:

```bash
apt -y install nvidia-cuda-toolkit-12-8
```

## Instalación

Instala SimpleTuner vía pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

Para instalación manual o entorno de desarrollo, consulta la [documentación de instalación](../INSTALL.md).

### Pasos adicionales para AMD ROCm

Lo siguiente debe ejecutarse para que una AMD MI300X sea utilizable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## Configuración del entorno

### Método de interfaz web

La WebUI de SimpleTuner hace que la configuración sea sencilla. Para ejecutar el servidor:

```bash
simpletuner server
```

Esto creará un servidor web en el puerto 8001 por defecto, al que puedes acceder visitando http://localhost:8001.

### Método manual / línea de comandos

Para ejecutar SimpleTuner mediante herramientas de línea de comandos, necesitarás configurar un archivo de configuración, los directorios del dataset y del modelo, y un archivo de configuración del dataloader.

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
- `model_family` - Configúralo en `z-image`.
- `model_flavour` - configúralo en `turbo` (o `turbo-ostris-v2` para el adaptador asistente v2); el flavour base apunta a un checkpoint actualmente no disponible.
- `pretrained_model_name_or_path` - Configúralo en `TONGYI-MAI/Z-Image-Turbo`.
- `output_dir` - Configúralo al directorio donde quieres guardar tus checkpoints y las imágenes de validación. Se recomienda usar una ruta completa aquí.
- `train_batch_size` - mantén en 1, especialmente si tienes un dataset muy pequeño.
- `validation_resolution` - Z-Image es 1024px; usa `1024x1024` o buckets multi-aspecto: `1024x1024,1280x768,2048x2048`.
- `validation_guidance` - Guidance bajo (0–1) es típico para Z-Image Turbo, pero el flavour base requiere un rango entre 4-6.
- `validation_num_inference_steps` - Turbo requiere solo 8, pero Base puede funcionar con alrededor de 50-100.
- `--lora_rank=4` si deseas reducir sustancialmente el tamaño del LoRA entrenado. Esto puede ayudar con el uso de VRAM.
- Para turbo, proporciona el adaptador asistente (ver abajo) o desactívalo explícitamente.

- `gradient_accumulation_steps` - aumenta el tiempo de ejecución linealmente; úsalo si necesitas alivio de VRAM.
- `optimizer` - A los principiantes se les recomienda quedarse con adamw_bf16, aunque otras variantes adamw/lion también son buenas opciones.
- `mixed_precision` - `bf16` en GPUs modernas; `fp16` en caso contrario.
- `gradient_checkpointing` - configúralo en true prácticamente en todas las situaciones y en todos los dispositivos.
- `gradient_checkpointing_interval` - puede configurarse en 2+ en GPUs más grandes para hacer checkpoint cada _n_ bloques.

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

</details>

### LoRA asistente (Turbo)

Turbo espera un adaptador asistente:

- `assistant_lora_path`: `ostris/zimage_turbo_training_adapter`
- `assistant_lora_weight_name`:
  - `turbo`: `zimage_turbo_training_adapter_v1.safetensors`
  - `turbo-ostris-v2`: `zimage_turbo_training_adapter_v2.safetensors`

SimpleTuner completa estos valores automáticamente para flavours turbo a menos que los sobrescribas. Desactívalo con `--disable_assistant_lora` si aceptas la pérdida de calidad.

### Prompts de validación

Dentro de `config/config.json` está el "prompt de validación principal", que suele ser el instance_prompt principal en el que estás entrenando para tu único sujeto o estilo. Además, se puede crear un archivo JSON que contiene prompts adicionales para ejecutar durante las validaciones.

El archivo de ejemplo `config/user_prompt_library.json.example` contiene el siguiente formato:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

Los apodos son el nombre de archivo de la validación, así que mantenlos cortos y compatibles con tu sistema de archivos.

Para indicar al entrenador esta librería de prompts, añádela a TRAINER_EXTRA_ARGS agregando una nueva línea al final de `config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

Un conjunto de prompts diverso ayudará a determinar si el modelo colapsa a medida que entrena. En este ejemplo, la palabra `<token>` debe reemplazarse por el nombre de tu sujeto (instance_prompt).

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing her essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing her unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing her dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```
</details>

> ℹ️ Z-Image es un modelo de flow-matching y los prompts más cortos con similitudes fuertes producirán prácticamente la misma imagen. Usa prompts más largos y descriptivos.

### Seguimiento de puntuaciones CLIP

Si deseas habilitar evaluaciones para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/CLIP_SCORES.md) para información sobre cómo configurar e interpretar las puntuaciones CLIP.

### Pérdida de evaluación estable

Si deseas usar pérdida MSE estable para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/EVAL_LOSS.md) para información sobre cómo configurar e interpretar la pérdida de evaluación.

### Vistas previas de validación

SimpleTuner admite la transmisión de vistas previas de validación intermedias durante la generación usando modelos Tiny AutoEncoder. Esto te permite ver las imágenes de validación generándose paso a paso en tiempo real mediante callbacks de webhook.

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

Configura `validation_preview_steps` en un valor más alto (p. ej., 3 o 5) para reducir la sobrecarga del Tiny AutoEncoder.

### Desplazamiento del calendario de flujo (flow matching)

Los modelos de flow-matching como Z-Image tienen un parámetro "shift" para mover la parte entrenada del calendario de timesteps. El auto-shift basado en resolución es un valor seguro por defecto. Aumentar shift manualmente desplaza el aprendizaje hacia rasgos gruesos; reducirlo favorece detalles finos. Para el modelo turbo, es posible que modificar estos valores perjudique el modelo.

### Entrenamiento con modelo cuantizado

TorchAO u otras cuantizaciones pueden reducir precisión y requisitos de VRAM - Optimum Quanto está en modo mantenimiento, pero también está disponible.

Para usuarios de `config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
  "base_model_precision": "int8-torchao",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

### Consideraciones del dataset

> ⚠️ La calidad de imagen para el entrenamiento es crítica; Z-Image absorberá los artefactos temprano. Puede requerirse un pase final con datos de alta calidad.

Mantén tu dataset lo suficientemente grande (al menos `train_batch_size * gradient_accumulation_steps`, y más que `vae_batch_size`). Aumenta `repeats` si ves **no images detected in dataset**.

Ejemplo de configuración multi-backend (`config/multidatabackend.json`):

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-zimage",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zimage/pseudo-camera-10k",
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
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zimage/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "dreambooth-subject-512",
    "type": "local",
    "crop": false,
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zimage/dreambooth-subject-512",
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
    "cache_dir": "cache/text/zimage",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

Ejecutar datasets de 512px y 1024px de forma concurrente está soportado y puede mejorar la convergencia.

Crea el directorio de datasets:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

### Iniciar sesión en WandB y Huggingface Hub

Inicia sesión antes de entrenar, especialmente si usas `--push_to_hub` y `--report_to=wandb`:

```bash
wandb login
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

Para más información, consulta los documentos de [dataloader](../DATALOADER.md) y [tutorial](../TUTORIAL.md) documents.

## Configuración multi-GPU

SimpleTuner incluye **detección automática de GPU** mediante la WebUI. Durante el onboarding, configurarás:

- **Modo Auto**: usa automáticamente todas las GPUs detectadas con ajustes óptimos
- **Modo Manual**: selecciona GPUs específicas o configura un número de procesos personalizado
- **Modo Deshabilitado**: entrenamiento con una sola GPU

La WebUI detecta tu hardware y configura `--num_processes` y `CUDA_VISIBLE_DEVICES` automáticamente.

Para configuración manual o setups avanzados, consulta la [sección de entrenamiento multi-GPU](../INSTALL.md#multiple-gpu-training) en la guía de instalación.

## Consejos de inferencia

### Ajustes de guidance

Z-Image es flow-matching; valores bajos de guidance (alrededor de 0–1) tienden a preservar calidad y diversidad. Si entrenas con vectores de guidance más altos, asegúrate de que tu pipeline de inferencia soporte CFG y espera generación más lenta o mayor uso de VRAM con CFG en batch.

## Notas y consejos de solución de problemas

### Configuración de VRAM más baja

- GPU: un solo dispositivo NVIDIA CUDA (10–12G) con cuantización/offload agresivo
- Memoria del sistema: ~32–48G
- Precisión del modelo base: `nf4-bnb` o `int8`
- Optimizador: Lion 8Bit Paged, `bnb-lion8bit-paged` o variantes adamw
- Resolución: 512px (1024px requiere más VRAM)
- Tamaño de lote: 1, cero pasos de acumulación de gradiente
- DeepSpeed: deshabilitado / sin configurar
- Usa `--quantize_via=cpu` si el arranque hace OOM en tarjetas <=16G
- Habilita `--gradient_checkpointing`
- Habilita Ramtorch o group offload

La etapa de pre-caché puede quedarse sin memoria; la cuantización del codificador de texto y el tiling del VAE pueden habilitarse vía `--text_encoder_precision=int8-torchao` y `--vae_enable_tiling=true`. Se puede ahorrar más memoria al inicio con `--offload_during_startup=true`, que mantendrá solo el codificador de texto o el VAE cargado, y no ambos.

### Cuantización

- A menudo se requiere cuantización mínima de 8 bits para que una tarjeta de 16G entrene este modelo.
- Cuantizar el modelo a 8 bits generalmente no perjudica el entrenamiento y permite tamaños de lote más altos.
- **int8** se beneficia de aceleración por hardware; **nf4-bnb** reduce VRAM aún más pero es más sensible.
- Al cargar el LoRA más tarde, **idealmente** deberías usar la misma precisión de modelo base con la que entrenaste.

### Aspect bucketing

- Entrenar solo en recortes cuadrados suele funcionar, pero buckets multi‑aspecto pueden mejorar la generalización.
- Usar buckets de aspecto naturales puede sesgar formas; el recorte aleatorio puede ayudar si necesitas cobertura más amplia.
- Mezclar configuraciones de dataset definiendo tu directorio de imágenes múltiples veces ha producido buena generalización.

### Tasas de aprendizaje

#### LoRA (--lora_type=standard)

- Tasas de aprendizaje más bajas suelen comportarse mejor en transformers grandes.
- Comienza con rangos modestos (4–16) antes de probar rangos muy altos.
- Reduce `max_grad_norm` si el modelo se desestabiliza; aumenta si el aprendizaje se estanca.

#### LoKr (--lora_type=lycoris)

- Tasas de aprendizaje más altas (p. ej., `1e-3` con AdamW, `2e-4` con Lion) pueden funcionar bien; ajusta a gusto.
- Marca datasets de regularización con `is_regularisation_data` para ayudar a preservar el modelo base.

### Artefactos de imagen

Z-Image absorberá artefactos de imagen malos temprano. Puede ser necesario un pase final con datos de alta calidad para limpiar. Vigila artefactos de cuadrícula si la tasa de aprendizaje es demasiado alta, los datos son de baja calidad o el manejo de aspecto es deficiente.

### Entrenar modelos Z-Image ajustados personalizados

Algunos checkpoints ajustados pueden carecer de estructura completa de directorios. Configura estos campos si es necesario:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "model_family": "z-image",
    "pretrained_model_name_or_path": "TONGYI-MAI/Z-Image-Base",
    "pretrained_transformer_model_name_or_path": "your-custom-transformer",
    "pretrained_vae_model_name_or_path": "TONGYI-MAI/Z-Image-Base",
    "pretrained_transformer_subfolder": "none"
}
```
</details>

## Solución de problemas

- OOM al inicio: habilita group offload (no con Quanto), baja el rango LoRA o cuantiza (`--base_model_precision int8`/`nf4`).
- Salidas borrosas: aumenta `validation_num_inference_steps` (p. ej., 24–28) o sube guidance hacia 1.0.
- Artefactos/sobreentrenamiento: reduce rango o tasa de aprendizaje, agrega prompts más diversos o acorta el entrenamiento.
- Problemas con adaptador asistente: turbo espera la ruta/peso del adaptador; desactívalo solo si aceptas pérdida de calidad.
- Validaciones lentas: recorta resoluciones o pasos de validación; el flow-matching converge rápido.
