# Guía rápida de LTX Video 2

En este ejemplo, entrenaremos un LoRA de LTX Video 2 usando los VAE de video/audio de LTX-2 y un codificador de texto Gemma3.

## Requisitos de hardware

LTX Video 2 es un modelo pesado de **19B**. Combina:
1.  **Gemma3**: El codificador de texto.
2.  **VAE de LTX-2 Video** (más el VAE de audio cuando se condiciona con audio).
3.  **Video Transformer de 19B**: Un backbone DiT grande.

Esta configuración es intensiva en VRAM, y el pre-caché del VAE puede disparar el uso de memoria.

- **Entrenamiento en una sola GPU**: Empieza con `train_batch_size: 1` y habilita group offload.
  - **Nota**: El **pre-caché del VAE** puede requerir más VRAM. Podrías necesitar offloading a CPU o una GPU más grande solo para la fase de caché.
  - **Consejo**: Configura `"offload_during_startup": true` en tu `config.json` para asegurar que el VAE y el codificador de texto no se carguen en la GPU al mismo tiempo, lo que reduce mucho la presión de memoria durante el pre-caché.
- **Entrenamiento multi-GPU**: Se recomienda **FSDP2** o **Group Offload** agresivo si necesitas más margen.
- **RAM del sistema**: Se recomiendan 64GB+ para ejecuciones grandes; más RAM ayuda con el caché.

### Rendimiento y memoria observados (reportes de campo)

- **Configuración base**: 480p, 17 frames, batch size 2 (mínima duración/resolución de video).
- **RamTorch (incl. codificador de texto)**: ~13 GB de VRAM en AMD 7900XTX.
  - NVIDIA 3090/4090/5090+ debería ver margen similar o mejor.
- **Sin offload (int8 TorchAO)**: ~29-30 GB de VRAM; se recomienda hardware de 32 GB.
  - Pico de RAM del sistema: ~46 GB al cargar Gemma3 bf16 y luego cuantizar a int8 (~32 GB VRAM).
  - Pico de RAM del sistema: ~34 GB al cargar el transformer LTX-2 bf16 y luego cuantizar a int8 (~30 GB VRAM).
- **Sin offload (bf16 completo)**: ~48 GB de VRAM necesarios para entrenar el modelo sin offload.
- **Rendimiento**:
  - ~8 s/step en A100-80G SXM4 (sin compilación).
  - ~16 s/step en 7900XTX (ejecución local).
  - ~30 min por 200 steps en A100-80G SXM4.

### Offloading de memoria (crítico)

Para la mayoría de configuraciones de una sola GPU entrenando LTX Video 2, deberías habilitar offloading agrupado. Es opcional pero recomendado para mantener margen de VRAM con lotes/resoluciones mayores.

Agrega esto a tu `config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "group_offload_blocks_per_group": 1,
  "group_offload_use_stream": true
}
```
</details>

## Requisitos previos

Asegúrate de que Python 3.12 esté instalado.

```bash
python --version
```

## Instalación

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Consulta [INSTALL.md](../INSTALL.md) para opciones de instalación avanzadas.

## Configuración del entorno

### Interfaz web

```bash
simpletuner server
```
Accede en http://localhost:8001.

### Configuración manual

Ejecuta el script de ayuda:

```bash
simpletuner configure
```

O copia el ejemplo y edita manualmente:

```bash
cp config/config.json.example config/config.json
```

#### Parámetros de configuración

Ajustes clave para LTX Video 2:

- `model_family`: `ltxvideo2`
- `model_flavour`: `dev` (predeterminado), `dev-fp4`, `dev-fp8`, `2.3-dev` o `2.3-distilled`.
- `pretrained_model_name_or_path`: `Lightricks/LTX-2`, `dg845/LTX-2.3-Diffusers`, `dg845/LTX-2.3-Distilled-Diffusers` o un archivo `.safetensors` local.
- `train_batch_size`: `1`. No aumentes esto a menos que tengas una A100/H100.
- `validation_resolution`:
  - `512x768` es un valor seguro para pruebas.
  - `720x1280` (720p) es posible pero pesado.
- `validation_num_video_frames`: **Debe ser compatible con la compresión del VAE (4x).**
  - Para 5s (a ~12-24fps): Usa `61` o `49`.
  - Fórmula: `(frames - 1) % 4 == 0`.
- `validation_guidance`: `5.0`.
- `frame_rate`: Por defecto es 25.

Las variantes LTX-2 2.0 se distribuyen como un único checkpoint `.safetensors` que incluye el transformer, el VAE de video,
el VAE de audio y el vocoder. Para LTX-2.3, SimpleTuner carga el repositorio Diffusers correspondiente según `model_flavour`
(`2.3-dev` o `2.3-distilled`).

### Opcional: optimizaciones de VRAM

Si necesitas más margen de VRAM:
- **Block swap (Musubi)**: Establece `musubi_blocks_to_swap` (prueba `4-8`) y opcionalmente `musubi_block_swap_device` (por defecto `cpu`) para stream de los últimos bloques del transformer desde CPU. Menor throughput, menor pico de VRAM.
- **Convolución de parches del VAE**: Configura `--vae_enable_patch_conv=true` para habilitar chunking temporal en el VAE de LTX-2; espera una pequeña pérdida de velocidad pero menor pico de VRAM.
- **VAE temporal roll**: Configura `--vae_enable_temporal_roll=true` para un chunking temporal más agresivo (mayor pérdida de velocidad).
- **VAE tiling**: Configura `--vae_enable_tiling=true` para dividir el encode/decode del VAE en resoluciones grandes.

### Opcional: regularizador temporal CREPA

Para reducir el parpadeo y mantener sujetos estables entre frames:
- En **Training → Loss functions**, habilita **CREPA**.
- Valores iniciales recomendados: **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- Mantén el codificador de visión por defecto (`dinov2_vitg14`, tamaño `518`) a menos que necesites uno más pequeño (`dinov2_vits14` + `224`).
- Requiere red (o un torch hub en caché) para obtener los pesos de DINOv2 la primera vez.
- Solo habilita **Drop VAE Encoder** si entrenas completamente desde latentes en caché; de lo contrario déjalo desactivado.

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

#### Consideraciones del dataset

Los datasets de video requieren una configuración cuidadosa. Crea `config/multidatabackend.json`:

```json
[
  {
    "id": "my-video-dataset",
    "type": "local",
    "dataset_type": "video",
    "instance_data_dir": "datasets/videos",
    "caption_strategy": "textfile",
    "resolution": 512,
    "video": {
        "num_frames": 61,
        "min_frames": 61,
        "frame_rate": 25,
        "bucket_strategy": "aspect_ratio"
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo2",
    "disabled": false
  }
]
```

En la subsección `video`:
- `num_frames`: Recuento objetivo de frames para entrenamiento.
- `min_frames`: Longitud mínima del video (los videos más cortos se descartan).
- `max_frames`: Filtro de longitud máxima del video.
- `bucket_strategy`: Cómo se agrupan los videos en buckets:
  - `aspect_ratio` (predeterminado): Agrupar solo por relación de aspecto espacial.
  - `resolution_frames`: Agrupar por formato `WxH@F` (p. ej., `1920x1080@61`) para datasets de resolución/duración mixta.
- `frame_interval`: Al usar `resolution_frames`, redondea recuentos de frames a este intervalo.

LTX-2 soporta entrenamiento solo de video sin audio. Para habilitar el entrenamiento con audio, agrega un bloque `audio` a la configuración de tu dataset de video:

```json
"audio": {
    "auto_split": true,
    "sample_rate": 16000,
    "channels": 1,
    "duration_interval": 3.0,
    "allow_zero_audio": false
}
```

Cuando la sección `audio` está presente, SimpleTuner genera automáticamente un dataset de audio a partir de tus archivos de video y cachea los latentes de audio junto con los latentes de video. Establece `audio.allow_zero_audio: true` si tus videos no tienen stream de audio. Sin una sección `audio`, LTX-2 entrena solo video y enmascara la pérdida de audio automáticamente.

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

#### Configuración de directorios

```bash
mkdir -p datasets/videos
</details>

# Place .mp4 / .mov files here.
# Place corresponding .txt files with same filename for captions.
```

#### Inicio de sesión

```bash
wandb login
huggingface-cli login
```

### Ejecutar el entrenamiento

```bash
simpletuner train
```

## Notas y consejos de solución de problemas

### Out of Memory (OOM)

El entrenamiento de video es extremadamente exigente. Si haces OOM:

1.  **Reduce la resolución**: Prueba 480p (`480x854` o similar).
2.  **Reduce frames**: Baja `validation_num_video_frames` y `num_frames` del dataset a `33` o `49`.
3.  **Revisa el offload**: Asegúrate de que `--enable_group_offload` esté activo.

### Calidad del video de validación

- **Videos negros/ruido**: A menudo causado por `validation_guidance` demasiado alto (> 6.0) o demasiado bajo (< 2.0). Mantén `5.0`.
- **Temblores de movimiento**: Revisa si la tasa de frames de tu dataset coincide con la del modelo entrenado (a menudo 25fps).
- **Video estancado/estático**: El modelo podría estar subentrenado o el prompt no describe movimiento. Usa prompts como "camera pans right", "zoom in", "running", etc.

### Entrenamiento TREAD

TREAD también funciona para video y se recomienda mucho para ahorrar cómputo.

Agrega a `config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

Esto puede acelerar el entrenamiento en ~25-40% dependiendo del ratio.

### Configuración de menor uso de VRAM (7900XTX)

Configuración probada en campo que prioriza el uso mínimo de VRAM en LTX Video 2.

<details>
<summary>Ver configuración 7900XTX (menor uso de VRAM)</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "checkpoint_step_interval": 100,
  "data_backend_config": "config/ltx2/multidatabackend.json",
  "disable_benchmark": true,
  "dynamo_mode": "",
  "evaluation_type": "none",
  "hub_model_id": "simpletuner-ltxvideo2-19b-t2v-lora-test",
  "learning_rate": 0.00006,
  "lr_warmup_steps": 50,
  "lycoris_config": "config/lycoris_config.json",
  "max_grad_norm": 0.1,
  "max_train_steps": 200,
  "minimum_image_size": 0,
  "model_family": "ltxvideo2",
  "model_flavour": "dev",
  "model_type": "lora",
  "num_train_epochs": 0,
  "offload_during_startup": true,
  "optimizer": "adamw_bf16",
  "output_dir": "output/examples/ltxvideo2-19b-t2v.peft-lora",
  "override_dataset_config": true,
  "ramtorch": true,
  "ramtorch_text_encoder": true,
  "report_to": "none",
  "resolution": 480,
  "scheduled_sampling_reflexflow": false,
  "seed": 42,
  "skip_file_discovery": "",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "example-training-run",
  "train_batch_size": 2,
  "vae_batch_size": 1,
  "vae_enable_patch_conv": true,
  "vae_enable_slicing": true,
  "vae_enable_temporal_roll": true,
  "vae_enable_tiling": true,
  "validation_disable": true,
  "validation_disable_unconditional": true,
  "validation_guidance": 5,
  "validation_num_inference_steps": 40,
  "validation_num_video_frames": 81,
  "validation_prompt": "🟫 is holding a sign that says hello world from ltxvideo2",
  "validation_resolution": "768x512",
  "validation_seed": 42,
  "validation_using_datasets": false
}
```
</details>

### Entrenamiento solo con audio

LTX-2 soporta **entrenamiento solo con audio**, donde entrenas únicamente la capacidad de generación de audio sin archivos de video. Esto es útil cuando tienes datasets de audio pero ningún contenido de video correspondiente.

En modo solo audio:
- Los latentes de video se ponen a cero automáticamente (resolución mínima de 64x64 para ahorrar memoria)
- La pérdida de video se enmascara (no se calcula)
- Solo se entrenan las capas de generación de audio

El modo solo audio se **detecta automáticamente** cuando tu configuración de dataset contiene únicamente datasets de audio (sin datasets de video o imagen). También puedes habilitarlo explícitamente con `audio.audio_only: true`.

#### Configuración del dataset solo audio

```json
[
  {
    "id": "my-audio-dataset",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/audio",
    "caption_strategy": "textfile",
    "audio": {
      "sample_rate": 16000,
      "channels": 2,
      "duration_interval": 3.0,
      "truncation_mode": "beginning"
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo2",
    "disabled": false
  }
]
```

Configuraciones clave de audio:
- `channels`: **Debe ser 2** (estéreo) para el Audio VAE de LTX-2
- `duration_interval`: Agrupa el audio en intervalos (ej. 3.0 segundos). **Importante para la gestión de memoria** - archivos de audio largos crean muchos frames de video aunque sean ceros
- `truncation_mode`: Cómo manejar audio más largo que la duración del bucket (`beginning`, `end`, o `random`)

#### Formatos de audio soportados

SimpleTuner soporta formatos de audio comunes (`.wav`, `.flac`, `.mp3`, `.ogg`, `.opus`, etc.) así como formatos contenedor que pueden contener solo audio (`.mp4`, `.mpeg`, `.mkv`, `.webm`). Los formatos contenedor se extraen automáticamente usando ffmpeg.

#### Objetivos LoRA para entrenamiento de audio

Cuando se detectan datos de audio en tus datasets, SimpleTuner añade automáticamente módulos específicos de audio a los objetivos LoRA:
- `audio_proj_in` - Proyección de entrada de audio
- `audio_proj_out` - Proyección de salida de audio
- `audio_caption_projection.linear_1` - Capa de proyección de subtítulos de audio 1
- `audio_caption_projection.linear_2` - Capa de proyección de subtítulos de audio 2

Esto sucede automáticamente tanto para el entrenamiento solo con audio como para el entrenamiento conjunto de audio+video.

Si deseas sobrescribir los objetivos LoRA manualmente, usa `--peft_lora_target_modules` con una lista JSON de nombres de módulos.

Coloca tus archivos de audio en `instance_data_dir` con los archivos `.txt` de subtítulos correspondientes.

### Flujos de validación (T2V vs I2V)

- **T2V (texto a video)**: Deja `validation_using_datasets: false` y usa `validation_prompt` o `validation_prompt_library`.
- **I2V (imagen a video)**: Configura `validation_using_datasets: true` y apunta `eval_dataset_id` a un split de validación que proporcione una imagen de referencia. La validación cambiará al pipeline de imagen a video y usará esa imagen como condicionamiento.
- **S2V (condicionado por audio)**: Con `validation_using_datasets: true`, asegúrate de que `eval_dataset_id` apunte a un dataset con `s2v_datasets` (o el comportamiento por defecto de `audio.auto_split`). La validación cargará los latentes de audio en caché automáticamente.

### Adaptadores de validación (LoRAs)

Lightricks publica varios LoRAs que se pueden aplicar en validación con `validation_adapter_path` (uno) o
`validation_adapter_config` (varias ejecuciones). Estos repos usan nombres de pesos no estándar, así que incluye
el filename con `repo_id:weight_name`. Consulta la colección de LTX-2 para ver los filenames correctos y los assets:
https://huggingface.co/collections/Lightricks/ltx-2
- `Lightricks/LTX-2-19b-IC-LoRA-Canny-Control:ltx-2-19b-ic-lora-canny-control.safetensors`
- `Lightricks/LTX-2-19b-IC-LoRA-Depth-Control:ltx-2-19b-ic-lora-depth-control.safetensors`
- `Lightricks/LTX-2-19b-IC-LoRA-Detailer:ltx-2-19b-ic-lora-detailer.safetensors`
- `Lightricks/LTX-2-19b-IC-LoRA-Pose-Control:ltx-2-19b-ic-lora-pose-control.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In:ltx-2-19b-lora-camera-control-dolly-in.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out:ltx-2-19b-lora-camera-control-dolly-out.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left:ltx-2-19b-lora-camera-control-dolly-left.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right:ltx-2-19b-lora-camera-control-dolly-right.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down:ltx-2-19b-lora-camera-control-jib-down.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up:ltx-2-19b-lora-camera-control-jib-up.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Static:ltx-2-19b-lora-camera-control-static.safetensors`

Para validación más rápida, aplica `Lightricks/LTX-2-19b-distilled-lora-384:ltx-2-19b-distilled-lora-384.safetensors`
como adaptador y configura `validation_guidance: 1` junto con `validation_num_inference_steps: 8`.
