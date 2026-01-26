# Guía rápida de Kandinsky 5.0 Video

En este ejemplo, entrenaremos un LoRA de Kandinsky 5.0 Video (Lite o Pro) usando el VAE de HunyuanVideo y codificadores de texto duales.

## Requisitos de hardware

Kandinsky 5.0 Video es un modelo pesado. Combina:
1.  **Qwen2.5-VL (7B)**: Un enorme codificador de texto visión-lenguaje.
2.  **VAE de HunyuanVideo**: Un VAE 3D de alta calidad.
3.  **Video Transformer**: Una arquitectura DiT compleja.

Esta configuración es intensiva en VRAM, aunque las variantes "Lite" y "Pro" tienen requisitos diferentes.

- **Entrenamiento del modelo Lite**: Sorprendentemente eficiente, capaz de entrenar con **~13GB de VRAM**.
  - **Nota**: El **pre-caché del VAE** requiere significativamente más VRAM debido al enorme VAE de HunyuanVideo. Podrías necesitar usar offloading a CPU o una GPU más grande solo para la fase de caché.
  - **Consejo**: Configura `"offload_during_startup": true` en tu `config.json` para asegurar que el VAE y el codificador de texto no se carguen en la GPU al mismo tiempo, lo que reduce mucho la presión de memoria durante el pre-caché.
  - **Si el VAE hace OOM**: Configura `--vae_enable_patch_conv=true` para dividir las conv 3D del VAE de HunyuanVideo; espera una pequeña pérdida de velocidad pero menor pico de VRAM.
- **Entrenamiento del modelo Pro**: Requiere **FSDP2** (multi-gpu) u **Group Offload** agresivo con LoRA para entrar en hardware de consumo. Los requisitos específicos de VRAM/RAM no están establecidos, pero aplica "cuanto más, mejor".
- **RAM del sistema**: Las pruebas fueron cómodas en un sistema con **45GB** de RAM para el modelo Lite. Se recomienda 64GB+ por seguridad.

### Offloading de memoria (crítico)

Para casi cualquier configuración de una sola GPU entrenando el modelo **Pro**, **debes** habilitar offloading agrupado. Es opcional pero recomendado para **Lite** para ahorrar VRAM y permitir lotes/resoluciones mayores.

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
pip install 'simpletuner[cuda13]'
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

Ajustes clave para Kandinsky 5 Video:

- `model_family`: `kandinsky5-video`
- `model_flavour`:
  - `t2v-lite-sft-5s`: Modelo Lite, salida ~5s. (Predeterminado)
  - `t2v-lite-sft-10s`: Modelo Lite, salida ~10s.
  - `t2v-pro-sft-5s-hd`: Modelo Pro, ~5s, entrenamiento de mayor definición.
  - `t2v-pro-sft-10s-hd`: Modelo Pro, ~10s, entrenamiento de mayor definición.
  - `i2v-lite-5s`: Image-to-video Lite, salidas de 5s (requiere imágenes de condicionamiento).
  - `i2v-pro-sft-5s`: Image-to-video Pro SFT, salidas de 5s (requiere imágenes de condicionamiento).
  - *(También hay variantes pretrain para todo lo anterior)*
- `train_batch_size`: `1`. No aumentes esto a menos que tengas una A100/H100.
- `validation_resolution`:
  - `512x768` es un valor seguro para pruebas.
  - `720x1280` (720p) es posible pero pesado.
- `validation_num_video_frames`: **Debe ser compatible con la compresión del VAE (4x).**
  - Para 5s (a ~12-24fps): Usa `61` o `49`.
  - Fórmula: `(frames - 1) % 4 == 0`.
- `validation_guidance`: `5.0`.
- `frame_rate`: Por defecto es 24.

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
    "cache_dir": "cache/text/kandinsky5",
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
- **Temblores de movimiento**: Revisa si la tasa de frames de tu dataset coincide con la del modelo entrenado (a menudo 24fps).
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

### Entrenamiento I2V (Image-to-Video)

Si usas sabores `i2v`:
- SimpleTuner extrae automáticamente el primer frame de los videos de entrenamiento para usarlo como imagen de condicionamiento.
- El pipeline enmascara automáticamente el primer frame durante el entrenamiento.

#### Opciones de Validación I2V

Para validación con modelos i2v, tienes dos opciones:

1. **Primer frame extraído automáticamente**: Por defecto, la validación usa el primer frame de las muestras de video.

2. **Dataset de imágenes separado** (setup más simple): Usa `--validation_using_datasets=true` con `--eval_dataset_id` apuntando a un dataset de imágenes:

```json
{
  "validation_using_datasets": true,
  "eval_dataset_id": "my-image-dataset"
}
```

Esto permite usar cualquier dataset de imágenes como input de condicionamiento del primer frame para videos de validación, sin necesidad de configurar el emparejamiento complejo de datasets de condicionamiento usado durante el entrenamiento.
