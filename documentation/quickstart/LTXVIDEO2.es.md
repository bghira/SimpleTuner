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
pip install simpletuner[cuda]
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
- `model_flavour`: `2.0` (predeterminado)
- `pretrained_model_name_or_path`: `Lightricks/LTX-2` (anulación opcional)
- `train_batch_size`: `1`. No aumentes esto a menos que tengas una A100/H100.
- `validation_resolution`:
  - `512x768` es un valor seguro para pruebas.
  - `720x1280` (720p) es posible pero pesado.
- `validation_num_video_frames`: **Debe ser compatible con la compresión del VAE (4x).**
  - Para 5s (a ~12-24fps): Usa `61` o `49`.
  - Fórmula: `(frames - 1) % 4 == 0`.
- `validation_guidance`: `5.0`.
- `frame_rate`: Por defecto es 25.

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

### Flujos de validación (T2V vs I2V)

- **T2V (texto a video)**: Deja `validation_using_datasets: false` y usa `validation_prompt` o `validation_prompt_library`.
- **I2V (imagen a video)**: Configura `validation_using_datasets: true` y apunta `eval_dataset_id` a un split de validación que proporcione una imagen de referencia. La validación cambiará al pipeline de imagen a video y usará esa imagen como condicionamiento.
