# Guía rápida de Hunyuan Video 1.5

Esta guía recorre el entrenamiento de un LoRA sobre la versión **Hunyuan Video 1.5** de Tencent (8.3B) (`tencent/HunyuanVideo-1.5`) usando SimpleTuner.

## Requisitos de hardware

Hunyuan Video 1.5 es un modelo grande (8.3B parámetros).

- **Mínimo**: **24GB-32GB de VRAM** es cómodo para un LoRA de rango 16 con gradient checkpointing completo a 480p.
- **Recomendado**: A6000 / A100 (48GB-80GB) para entrenamiento a 720p o tamaños de lote mayores.
- **RAM del sistema**: Se recomienda **64GB+** para manejar la carga del modelo.

### Offloading de memoria (opcional)

Agrega lo siguiente a tu `config.json`:

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

- `--group_offload_use_stream`: Solo funciona en dispositivos CUDA.
- **No** combines esto con `--enable_model_cpu_offload`.

## Requisitos previos

Asegúrate de tener python instalado; SimpleTuner funciona bien con 3.10 a 3.12.

Puedes comprobarlo ejecutando:

```bash
python --version
```

Si no tienes python 3.12 instalado en Ubuntu, puedes intentar lo siguiente:

```bash
apt -y install python3.12 python3.12-venv
```

### Dependencias de la imagen de contenedor

Para Vast, RunPod y TensorDock (entre otros), lo siguiente funcionará en una imagen CUDA 12.2-12.8 para habilitar la compilación de extensiones CUDA:

```bash
apt -y install nvidia-cuda-toolkit
```

### Pasos adicionales para AMD ROCm

Lo siguiente debe ejecutarse para que una AMD MI300X sea utilizable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## Instalación

Instala SimpleTuner vía pip:

```bash
pip install simpletuner[cuda]
```

Para instalación manual o entorno de desarrollo, consulta la [documentación de instalación](../INSTALL.md).

### Checkpoints requeridos

El repo principal `tencent/HunyuanVideo-1.5` contiene transformer/vae/scheduler, pero el **codificador de texto** (`text_encoder/llm`) y el **codificador de visión** (`vision_encoder/siglip`) viven en descargas separadas. Apunta SimpleTuner a tus copias locales antes de iniciar:

```bash
export HUNYUANVIDEO_TEXT_ENCODER_PATH=/path/to/text_encoder_root
export HUNYUANVIDEO_VISION_ENCODER_PATH=/path/to/vision_encoder_root
```

Si no se configuran, SimpleTuner intenta obtenerlos del repo del modelo; la mayoría de los mirrors no los incluyen, así que configura las rutas explícitamente para evitar errores de arranque.

## Configuración del entorno

### Método de interfaz web

La WebUI de SimpleTuner hace que la configuración sea bastante sencilla. Para ejecutar el servidor:

```bash
simpletuner server
```

Esto creará un servidor web en el puerto 8001 por defecto, al que puedes acceder visitando http://localhost:8001.

### Método manual / línea de comandos

Para ejecutar SimpleTuner mediante herramientas de línea de comandos, necesitarás configurar un archivo de configuración, los directorios del dataset y del modelo, y un archivo de configuración del dataloader.

#### Archivo de configuración

Un script experimental, `configure.py`, puede permitirte omitir por completo esta sección mediante una configuración interactiva paso a paso.

**Nota:** Esto no configura tu dataloader. Aún tendrás que hacerlo manualmente más adelante.

Para ejecutarlo:

```bash
simpletuner configure
```

Si prefieres configurarlo manualmente:

Copia `config/config.json.example` a `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Overrides clave de configuración para HunyuanVideo:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "model_type": "lora",
  "model_family": "hunyuanvideo",
  "pretrained_model_name_or_path": "tencent/HunyuanVideo-1.5",
  "model_flavour": "t2v-480p",
  "output_dir": "output/hunyuan-video",
  "validation_resolution": "854x480",
  "validation_num_video_frames": 61,
  "validation_guidance": 6.0,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "learning_rate": 1e-4,
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "lora_rank": 16,
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "dataset_backend_config": "config/multidatabackend.json"
}
```
</details>

- Opciones de `model_flavour`:
  - `t2v-480p` (Predeterminado)
  - `t2v-720p`
  - `i2v-480p` (Image-to-Video)
  - `i2v-720p` (Image-to-Video)
- `validation_num_video_frames`: Debe cumplir `(frames - 1) % 4 == 0`. Por ejemplo, 61, 129.

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

#### Consideraciones del dataset

Crea un documento `--data_backend_config` (`config/multidatabackend.json`) que contenga esto:

```json
[
  {
    "id": "my-video-dataset",
    "type": "local",
    "dataset_type": "video",
    "instance_data_dir": "datasets/videos",
    "caption_strategy": "textfile",
    "resolution": 480,
    "video": {
        "num_frames": 61,
        "min_frames": 61,
        "frame_rate": 24,
        "bucket_strategy": "aspect_ratio"
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/hunyuan",
    "disabled": false
  }
]
```

En la subsección `video`:
- `num_frames`: Recuento objetivo de frames para entrenamiento. Debe cumplir `(frames - 1) % 4 == 0`.
- `min_frames`: Longitud mínima del video (los videos más cortos se descartan).
- `max_frames`: Filtro de longitud máxima del video.
- `bucket_strategy`: Cómo se agrupan los videos en buckets:
  - `aspect_ratio` (predeterminado): Agrupar solo por relación de aspecto espacial.
  - `resolution_frames`: Agrupar por formato `WxH@F` (p. ej., `854x480@61`) para datasets de resolución/duración mixta.
- `frame_interval`: Al usar `resolution_frames`, redondea los recuentos de frames a este intervalo.

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

- **Caché de text embeds**: Muy recomendado. Hunyuan usa un codificador de texto LLM grande. El caché ahorra VRAM significativa durante el entrenamiento.

#### Iniciar sesión en WandB y Huggingface Hub

```bash
wandb login
huggingface-cli login
```

</details>

### Ejecutar el entrenamiento

Desde el directorio de SimpleTuner:

```bash
simpletuner train
```

## Notas y consejos de solución de problemas

### Optimización de VRAM

- **Group Offload**: Esencial para GPUs de consumo. Asegúrate de que `enable_group_offload` sea true.
- **Resolución**: Mantente en 480p (`854x480` o similar) si tienes VRAM limitada. 720p (`1280x720`) aumenta significativamente el uso de memoria.
- **Cuantización**: Usa `base_model_precision` (`bf16` por defecto); `int8-torchao` funciona para mayor ahorro a costa de velocidad.
- **Convolución por parches del VAE**: Para OOM del VAE de HunyuanVideo, configura `--vae_enable_patch_conv=true` (o cambia en la UI). Esto divide el trabajo de conv/atención 3D para reducir el pico de VRAM; espera una pequeña pérdida de throughput.

### Image-to-Video (I2V)

- Usa `model_flavour="i2v-480p"`.
- SimpleTuner usa automáticamente el primer frame de tus muestras del dataset de video como la imagen de condicionamiento.
- Asegúrate de que tu validación incluya inputs de condicionamiento o dependa del primer frame extraído automáticamente.

### Codificadores de texto

Hunyuan usa un setup de doble codificador de texto (LLM + CLIP). Asegúrate de que tu RAM del sistema pueda manejar la carga durante la fase de caché.
