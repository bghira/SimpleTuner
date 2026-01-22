## Guía rápida de Wan 2.2 S2V

En este ejemplo, entrenaremos una LoRA Wan 2.2 S2V (Speech-to-Video). Los modelos S2V generan video condicionado por entrada de audio, lo que permite la generación de video impulsada por audio.

### Requisitos de hardware

Wan 2.2 S2V **14B** es un modelo exigente que requiere mucha memoria de GPU.

#### Speech to Video

14B - https://huggingface.co/tolgacangoz/Wan2.2-S2V-14B-Diffusers
- Resolución: 832x480
- Cabe en 24G, pero tendrás que ajustar un poco la configuración.

Necesitarás:
- **un mínimo realista** es 24GB o, una sola GPU 4090 o A6000
- **idealmente** varias 4090, A6000, L40S o mejores

Los sistemas con Apple Silicon no funcionan tan bien con Wan 2.2 por ahora; se pueden esperar algo como 10 minutos para un solo paso de entrenamiento.

### Requisitos previos

Asegúrate de tener Python instalado; SimpleTuner funciona bien con 3.10 a 3.12.

Puedes comprobarlo ejecutando:

```bash
python --version
```

Si no tienes Python 3.12 instalado en Ubuntu, puedes intentar lo siguiente:

```bash
apt -y install python3.13 python3.13-venv
```

#### Dependencias de imagen de contenedor

Para Vast, RunPod y TensorDock (entre otros), lo siguiente funcionará en una imagen CUDA 12.2-12.8 para habilitar la compilación de extensiones CUDA:

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

Para instalación manual o configuración de desarrollo, consulta la [documentación de instalación](/documentation/INSTALL.md).
#### SageAttention 2

Si deseas usar SageAttention 2, se deben seguir algunos pasos.

> Nota: SageAttention ofrece una aceleración mínima, no es muy efectivo; no sé por qué. Probado en 4090.

Ejecuta lo siguiente mientras sigues dentro de tu venv de Python:
```bash
git clone https://github.com/thu-ml/SageAttention
pushd SageAttention
  pip install . --no-build-isolation
popd
```

#### Pasos posteriores para AMD ROCm

Lo siguiente debe ejecutarse para que un AMD MI300X sea utilizable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

### Configuración del entorno

Para ejecutar SimpleTuner, tendrás que configurar un archivo de configuración, los directorios de dataset y modelo, y un archivo de configuración del dataloader.

#### Archivo de configuración

Un script experimental, `configure.py`, puede permitirte saltarte por completo esta sección mediante una configuración interactiva paso a paso. Contiene algunas funciones de seguridad que ayudan a evitar errores comunes.

**Nota:** Esto no configura tu dataloader. Aún tendrás que hacerlo manualmente más adelante.

Para ejecutarlo:

```bash
simpletuner configure
```

> Para usuarios ubicados en países donde Hugging Face Hub no es fácilmente accesible, debes agregar `HF_ENDPOINT=https://hf-mirror.com` a tu `~/.bashrc` o `~/.zshrc` según el `$SHELL` que use tu sistema.

### Offloading de memoria (opcional)

Wan es uno de los modelos más pesados que admite SimpleTuner. Habilita el offloading agrupado si estás cerca del límite de VRAM:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Solo los dispositivos CUDA respetan `--group_offload_use_stream`; ROCm/MPS vuelven automáticamente.
- Deja el staging a disco comentado a menos que la memoria de CPU sea el cuello de botella.
- `--enable_model_cpu_offload` es mutuamente excluyente con el offload agrupado.

### Fragmentación de feed-forward (opcional)

Si los checkpoints 14B todavía hacen OOM durante el gradient checkpointing, fragmenta las capas feed-forward de Wan:

```bash
--enable_chunked_feed_forward \
--feed_forward_chunk_size 2 \
```

Esto coincide con el nuevo interruptor del asistente de configuración (`Training -> Memory Optimisation`). Los tamaños de fragmento más pequeños ahorran más memoria pero ralentizan cada paso. También puedes establecer `WAN_FEED_FORWARD_CHUNK_SIZE=2` en tu entorno para experimentos rápidos.


Si prefieres configurar manualmente:

Copia `config/config.json.example` a `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Los usuarios multi-GPU pueden consultar [este documento](/documentation/OPTIONS.md#environment-configuration-variables) para información sobre cómo configurar el número de GPU a usar.

Tu configuración al final se verá como la mía:

<details>
<summary>View example config</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "attention_mechanism": "sageattention",
  "data_backend_config": "config/wan_s2v/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "offload_during_startup": true,
  "disable_benchmark": false,
  "output_dir": "output/wan_s2v",
  "lora_type": "standard",
  "lycoris_config": "config/wan_s2v/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "wan-s2v-lora",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "wan-s2v-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
  "pretrained_t5_model_name_or_path": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
  "model_family": "wan_s2v",
  "model_flavour": "s2v-14b-2.2",
  "train_batch_size": 2,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "832x480",
  "validation_prompt": "A person speaking with natural gestures",
  "validation_negative_prompt": "blurry, low quality, distorted",
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 40,
  "validation_num_video_frames": 81,
  "mixed_precision": "bf16",
  "optimizer": "optimi-lion",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.01,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "no_change",
  "vae_batch_size": 1,
  "webhook_config": "config/wan_s2v/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "lora_rank": 16,
  "flow_schedule_shift": 3,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

De particular importancia en esta configuración son los ajustes de validación. Sin ellos, los resultados no se ven muy bien.

### Opcional: regularizador temporal CREPA

Para movimiento más suave y menos deriva de identidad en Wan S2V:
- En **Training -> Loss functions**, habilita **CREPA**.
- Empieza con **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- El codificador por defecto (`dinov2_vitg14`, tamaño `518`) funciona bien; cambia a `dinov2_vits14` + `224` solo si necesitas reducir VRAM.
- La primera ejecución descarga DINOv2 vía torch hub; cachea o precarga si entrenas sin conexión.
- Solo habilita **Drop VAE Encoder** cuando entrenes totalmente desde latentes en caché; de lo contrario, déjalo desactivado para que las codificaciones de píxeles sigan funcionando.

### Funciones experimentales avanzadas

<details>
<summary>Show advanced experimental details</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.

> Estas funciones aumentan la sobrecarga computacional del entrenamiento.

</details>

### Entrenamiento TREAD

> **Experimental**: TREAD es una función implementada recientemente. Aunque funciona, aún se están explorando configuraciones óptimas.

[TREAD](/documentation/TREAD.md) (paper) significa **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion. Es un método que puede acelerar el entrenamiento de Wan S2V al enrutar inteligentemente tokens a través de las capas del transformer. La aceleración es proporcional a cuántos tokens se eliminan.

#### Configuración rápida

Añade esto a tu `config.json` para un enfoque simple y conservador:

<details>
<summary>View example config</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.1,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

Esta configuración hará lo siguiente:
- Mantener solo el 50% de los tokens de imagen durante las capas 2 hasta la penúltima
- Los tokens de texto nunca se eliminan
- Aceleración de entrenamiento de ~25% con un impacto mínimo en la calidad
- Potencialmente mejora la calidad del entrenamiento y la convergencia

#### Puntos clave

- **Soporte de arquitectura limitado** - TREAD solo está implementado para modelos Flux y Wan (incluido S2V)
- **Mejor en altas resoluciones** - Las mayores aceleraciones en 1024x1024+ debido a la complejidad O(n^2) de la atención
- **Compatible con pérdida enmascarada** - Las regiones enmascaradas se conservan automáticamente (pero esto reduce la aceleración)
- **Funciona con cuantización** - Se puede combinar con entrenamiento int8/int4/NF4
- **Espera un pico inicial de pérdida** - Al iniciar entrenamiento LoRA/LoKr, la pérdida será mayor al principio pero se corrige rápido

#### Consejos de ajuste

- **Conservador (enfocado en calidad)**: Usa `selection_ratio` de 0.1-0.3
- **Agresivo (enfocado en velocidad)**: Usa `selection_ratio` de 0.3-0.5 y acepta el impacto en calidad
- **Evita capas tempranas/tardías**: No enrutes en las capas 0-1 o en la última capa
- **Para entrenamiento LoRA**: Podrías ver pequeñas ralentizaciones; experimenta con distintas configuraciones
- **Mayor resolución = mejor aceleración**: Más beneficioso en 1024px y superiores

Para opciones de configuración y solución de problemas detalladas, consulta la [documentación completa de TREAD](/documentation/TREAD.md).


#### Prompts de validación

Dentro de `config/config.json` está el "primary validation prompt", que normalmente es el instance_prompt principal con el que estás entrenando para tu sujeto o estilo único. Además, se puede crear un archivo JSON con prompts adicionales para ejecutar durante las validaciones.

El archivo de ejemplo `config/user_prompt_library.json.example` contiene el siguiente formato:

<details>
<summary>View example config</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

Los apodos son el nombre de archivo para la validación, así que mantenlos cortos y compatibles con tu sistema de archivos.

Para indicar al trainer esta biblioteca de prompts, añádela a TRAINER_EXTRA_ARGS agregando una nueva línea al final de `config.json`:
<details>
<summary>View example config</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

> S2V usa el codificador de texto UMT5, que tiene mucha información local en sus embeddings, lo que significa que los prompts cortos pueden no tener suficiente información para que el modelo haga un buen trabajo. Asegúrate de usar prompts más largos y descriptivos.

#### Seguimiento de puntuación CLIP

Esto no debe habilitarse para el entrenamiento de modelos de video, por el momento.

# Pérdida de evaluación estable

Si deseas usar pérdida MSE estable para puntuar el rendimiento del modelo, consulta [este documento](/documentation/evaluation/EVAL_LOSS.md) para información sobre cómo configurar e interpretar la pérdida de evaluación.

#### Vistas previas de validación

SimpleTuner admite previsualizaciones de validación intermedias en streaming durante la generación usando modelos Tiny AutoEncoder. Esto te permite ver imágenes de validación generadas paso a paso en tiempo real mediante callbacks de webhook.

Para habilitarlo:
<details>
<summary>View example config</summary>

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

Establece `validation_preview_steps` en un valor más alto (por ejemplo, 3 o 5) para reducir la sobrecarga de Tiny AutoEncoder. Con `validation_num_inference_steps=20` y `validation_preview_steps=5`, recibirás imágenes de vista previa en los pasos 5, 10, 15 y 20.

#### Desplazamiento del cronograma de flow-matching

Los modelos de flow-matching como Flux, Sana, SD3, LTX Video y Wan S2V tienen una propiedad llamada `shift` que nos permite desplazar la parte entrenada del cronograma de timesteps usando un valor decimal simple.

##### Valores predeterminados
De forma predeterminada, no se aplica desplazamiento de cronograma, lo que da como resultado una forma de campana sigmoide en la distribución de muestreo de timesteps, conocida como `logit_norm`.

##### Auto-desplazamiento
Un enfoque recomendado comúnmente es seguir varios trabajos recientes y habilitar el desplazamiento de timesteps dependiente de la resolución, `--flow_schedule_auto_shift`, que usa valores de desplazamiento más altos para imágenes grandes y valores más bajos para imágenes pequeñas. Esto produce resultados estables pero potencialmente mediocres.

##### Especificación manual
_Agradecimientos a General Awareness de Discord por los siguientes ejemplos_

> Estos ejemplos muestran cómo funciona el valor usando Flux Dev, aunque Wan S2V debería ser muy similar.

Al usar un valor de `--flow_schedule_shift` de 0.1 (muy bajo), solo se ven afectados los detalles finos de la imagen:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

Al usar un valor de `--flow_schedule_shift` de 4.0 (muy alto), se ven afectados los grandes rasgos compositivos y posiblemente el espacio de color del modelo:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### Entrenamiento de modelo cuantizado

Probado en sistemas Apple y NVIDIA, Hugging Face Optimum-Quanto puede usarse para reducir la precisión y los requisitos de VRAM, entrenando con solo 16GB.



Para usuarios de `config.json`:
<details>
<summary>View example config</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

#### Configuración de validación

Durante la exploración inicial, la baja calidad de salida puede venir de Wan S2V, y esto se reduce a un par de razones:

- No hay suficientes pasos de inferencia
  - A menos que uses UniPC, probablemente necesites al menos 40 pasos. UniPC puede reducir el número un poco, pero tendrás que experimentar.
- Configuración incorrecta del scheduler
  - Estaba usando el cronograma normal de flow matching de Euler, pero la distribución Betas parece funcionar mejor
  - Si no has tocado esta configuración, debería estar bien ahora
- Resolución incorrecta
  - Wan S2V solo funciona correctamente en las resoluciones con las que fue entrenado, si funciona es por suerte, pero es común que los resultados sean malos
- Valor CFG incorrecto
  - Un valor alrededor de 4.0-5.0 parece seguro
- Mal prompting
  - Por supuesto, los modelos de video parecen requerir un equipo de místicos que pasen meses en las montañas en un retiro zen para aprender el arte sagrado del prompting, porque sus datasets y estilo de captions están custodiados como el Santo Grial.
  - tl;dr prueba distintos prompts.
- Audio faltante o desajustado
  - S2V requiere entrada de audio para validación: asegúrate de que tus muestras de validación tengan archivos de audio correspondientes

A pesar de todo esto, a menos que tu batch size sea demasiado bajo y/o tu learning rate sea demasiado alto, el modelo se ejecutará correctamente en tu herramienta de inferencia favorita (suponiendo que ya tienes una con buenos resultados).

#### Consideraciones del dataset

El entrenamiento S2V requiere datos de video y audio emparejados. Por defecto, SimpleTuner hace auto-split de audio
desde datasets de video, así que no necesitas definir un dataset de audio separado a menos que quieras un
procesamiento personalizado. Usa `audio.auto_split: false` para desactivar y define `s2v_datasets` manualmente.

Hay pocas limitaciones en el tamaño del dataset, aparte de cuánto cómputo y tiempo tomará procesar y entrenar.

Debes asegurarte de que el dataset sea lo suficientemente grande para entrenar tu modelo de forma efectiva, pero no tan grande para la cantidad de cómputo que tienes disponible.

Ten en cuenta que el tamaño mínimo del dataset es `train_batch_size * gradient_accumulation_steps` así como más que `vae_batch_size`. El dataset no será utilizable si es demasiado pequeño.

> Con suficientes pocas muestras, podrías ver un mensaje **no samples detected in dataset** - aumentar el valor de `repeats` superará esta limitación.

#### Configuración del dataset de audio

##### Extracción automática de audio desde videos (Recomendado)

Si tus videos ya contienen pistas de audio, SimpleTuner puede extraer y procesar audio automáticamente sin requerir un dataset de audio separado. Este es el enfoque más simple y el predeterminado:

```json
[
  {
    "id": "s2v-videos",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan_s2v/videos",
    "instance_data_dir": "datasets/s2v-videos",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    },
    "audio": {
        "auto_split": true,
        "sample_rate": 16000,
        "channels": 1
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan_s2v",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

Con auto-split de audio habilitado (por defecto), SimpleTuner:
1. Genera automáticamente una configuración de dataset de audio (`s2v-videos_audio`)
2. Extrae audio de cada video durante el descubrimiento de metadatos
3. Cachea latentes VAE de audio en un directorio dedicado
4. Enlaza automáticamente el dataset de audio vía `s2v_datasets`

**Opciones de configuración de audio:**
- `audio.auto_split` (bool): Habilita la extracción automática de audio desde videos (predeterminado: true)
- `audio.sample_rate` (int): Tasa de muestreo objetivo en Hz (predeterminado: 16000 para Wav2Vec2)
- `audio.channels` (int): Número de canales de audio (predeterminado: 1 para mono)
- `audio.allow_zero_audio` (bool): Genera audio con ceros para videos sin streams de audio (predeterminado: false)
- `audio.max_duration_seconds` (float): Duración máxima del audio; los archivos más largos se omiten
- `audio.duration_interval` (float): Intervalo de duración para agrupación en buckets en segundos (predeterminado: 3.0)
- `audio.truncation_mode` (string): Cómo truncar audio largo: "beginning", "end", "random" (predeterminado: "beginning")

**Nota**: Los videos sin pistas de audio se omiten automáticamente para entrenamiento S2V a menos que se establezca `audio.allow_zero_audio: true`.

##### Dataset de audio manual (Alternativa)

Si prefieres archivos de audio separados, necesitas un procesamiento de audio personalizado o desactivas el auto-split,
los modelos S2V también pueden usar archivos de audio preextraídos que coincidan con tus archivos de video por nombre de archivo. Por ejemplo:
- `video_001.mp4` debe tener un `video_001.wav` (o `.mp3`, `.flac`, `.ogg`, `.m4a`)

Los archivos de audio deben estar en un directorio separado que configurarás como un backend `s2v_datasets`.

##### Extraer audio de videos (Manual)

Si tus videos ya contienen audio, usa el script proporcionado para extraerlo:

```bash
# Extract audio only (keeps original videos unchanged)
python scripts/generate_s2v_audio.py \
    --input-dir datasets/s2v-videos \
    --output-dir datasets/s2v-audio

# Extract audio and remove it from source videos (recommended to avoid redundant data)
python scripts/generate_s2v_audio.py \
    --input-dir datasets/s2v-videos \
    --output-dir datasets/s2v-audio \
    --strip-audio
```

El script:
- Extrae audio en WAV mono de 16kHz (tasa nativa de Wav2Vec2)
- Empareja nombres de archivo automáticamente (p. ej., `video.mp4` -> `video.wav`)
- Omite videos sin streams de audio
- Requiere que `ffmpeg` esté instalado

##### Configuración del dataset (Manual)

Crea un documento `--data_backend_config` (`config/multidatabackend.json`) que contenga esto:

<details>
<summary>View example config</summary>

```json
[
  {
    "id": "s2v-videos",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan_s2v/videos",
    "instance_data_dir": "datasets/s2v-videos",
    "s2v_datasets": ["s2v-audio"],
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "s2v-audio",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/s2v-audio",
    "disabled": false
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan_s2v",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

Puntos clave para la configuración del dataset S2V:
- El campo `s2v_datasets` en tu dataset de video apunta al/los backend(s) de audio
- Los archivos de audio se emparejan por el stem del nombre de archivo (p. ej., `video_001.mp4` coincide con `video_001.wav`)
- El audio se codifica en tiempo real usando Wav2Vec2 (~600MB de VRAM), no requiere caché
- El tipo de dataset de audio es `audio`

- En la subsección `video`, tenemos las siguientes claves que podemos configurar:
  - `num_frames` (opcional, int) es cuántos frames de datos entrenaremos.
    - A 15 fps, 75 frames son 5 segundos de video, salida estándar. Ese debería ser tu objetivo.
  - `min_frames` (opcional, int) determina la longitud mínima de un video que se considerará para entrenamiento.
    - Debe ser al menos igual a `num_frames`. Si no se establece, se asegura que sea igual.
  - `max_frames` (opcional, int) determina la longitud máxima de un video que se considerará para entrenamiento.
  - `bucket_strategy` (opcional, string) determina cómo se agrupan los videos en buckets:
    - `aspect_ratio` (predeterminado): Agrupa solo por relación de aspecto espacial (p. ej., `1.78`, `0.75`).
    - `resolution_frames`: Agrupa por resolución y cantidad de frames en formato `WxH@F` (p. ej., `832x480@75`). Útil para datasets con resoluciones/duraciones mixtas.
  - `frame_interval` (opcional, int) cuando se usa `resolution_frames`, redondea el recuento de frames a este intervalo.

Luego, crea un directorio `datasets` con tus archivos de video y audio:

```bash
mkdir -p datasets/s2v-videos datasets/s2v-audio
# Place your video files in datasets/s2v-videos/
# Place your audio files in datasets/s2v-audio/
```

Asegúrate de que cada video tenga un archivo de audio coincidente por stem de nombre de archivo.

#### Iniciar sesión en WandB y Huggingface Hub

Querrás iniciar sesión en WandB y HF Hub antes de comenzar el entrenamiento, especialmente si usas `--push_to_hub` y `--report_to=wandb`.

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

### Ejecución del entrenamiento

Desde el directorio de SimpleTuner, tienes varias opciones para iniciar el entrenamiento:

**Opción 1 (Recomendado - instalación con pip):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
simpletuner train
```

**Opción 2 (Método git clone):**
```bash
simpletuner train
```

**Opción 3 (Método heredado - aún funciona):**
```bash
./train.sh
```

Esto iniciará el cacheo en disco de los embeddings de texto y las salidas de VAE.

Para más información, consulta los documentos de [dataloader](/documentation/DATALOADER.md) y [tutorial](/documentation/TUTORIAL.md).

## Notas y consejos de solución de problemas

### Configuración de VRAM mínima

Wan S2V es sensible a la cuantización y actualmente no puede usarse con NF4 o INT4.

- OS: Ubuntu Linux 24
- GPU: Un solo dispositivo NVIDIA CUDA (24G recomendado)
- Memoria del sistema: aproximadamente 16G de memoria del sistema
- Precisión del modelo base: `int8-quanto`
- Optimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolución: 480px
- Batch size: 1, cero pasos de acumulación de gradiente
- DeepSpeed: deshabilitado / no configurado
- PyTorch: 2.6
- Asegúrate de habilitar `--gradient_checkpointing` o nada evitará el OOM
- Entrena solo con imágenes, o establece `num_frames` a 1 para tu dataset de video

**NOTA**: El precacheo de embeds de VAE y salidas del codificador de texto puede usar más memoria y aún así hacer OOM. Como resultado, `--offload_during_startup=true` es básicamente obligatorio. Si es así, se puede habilitar la cuantización del codificador de texto y el tiling de VAE. (Wan no admite actualmente VAE tiling/slicing)

### SageAttention

Al usar `--attention_mechanism=sageattention`, la inferencia puede acelerarse en tiempo de validación.

**Nota**: Esto no es compatible con el paso final de decodificación VAE y no acelerará esa parte.

### Pérdida enmascarada

No uses esto con Wan S2V.

### Cuantización
- La cuantización puede ser necesaria para entrenar este modelo en 24G dependiendo del batch size

### Artefactos de imagen
Wan requiere el uso del cronograma de flow-matching Euler Betas o (por defecto) el solver multistep UniPC, un scheduler de orden superior que hará predicciones más fuertes.

Como otros modelos DiT, si haces estas cosas (entre otras) podrían empezar a aparecer algunos artefactos en cuadrícula:
- Sobreentrenar con datos de baja calidad
- Usar una tasa de aprendizaje demasiado alta
- Sobreentrenamiento (en general), una red de baja capacidad con demasiadas imágenes
- Subentrenamiento (también), una red de alta capacidad con muy pocas imágenes
- Usar relaciones de aspecto extrañas o tamaños de datos de entrenamiento

### Bucketización de aspecto
- Los videos se agrupan en buckets como las imágenes.
- Entrenar demasiado tiempo con recortes cuadrados probablemente no dañará mucho este modelo. Adelante, es genial y confiable.
- Por otro lado, usar los buckets de aspecto natural de tu dataset podría sesgar excesivamente esas formas durante la inferencia.
  - Esto podría ser una cualidad deseable, ya que evita que estilos dependientes del aspecto como lo cinematográfico se filtren a otras resoluciones.
  - Sin embargo, si buscas mejorar resultados por igual en muchos buckets de aspecto, quizás tengas que experimentar con `crop_aspect=random`, lo que trae sus propias desventajas.
- Mezclar configuraciones de dataset definiendo tu directorio de imágenes varias veces ha producido resultados muy buenos y un modelo bien generalizado.

### Sincronización de audio

Para mejores resultados con S2V:
- Asegúrate de que la duración del audio coincida con la duración del video
- El audio se remuestrea internamente a 16kHz
- El codificador Wav2Vec2 procesa audio en tiempo real (~600MB de sobrecarga de VRAM)
- Las características de audio se interpolan para coincidir con la cantidad de frames de video
