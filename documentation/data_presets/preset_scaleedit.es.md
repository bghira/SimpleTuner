# ScaleEdit-12M: imágenes Parquet emparejadas con metadatos Webshart

## Detalles

- **Dataset de origen**: [InternVL-U/ScaleEdit-12M](https://huggingface.co/datasets/InternVL-U/ScaleEdit-12M).
- **Metadatos complementarios**: [webshart/ScaleEdit-12M-metadata](https://huggingface.co/datasets/webshart/ScaleEdit-12M-metadata).
- **Descripción**: 11.845.023 filas de edición en 251 shards Parquet y 23 categorías. Los metadatos ocupan unos 423 MB; las imágenes permanecen en el origen.
- **Formato de captions**: cadenas `edit_instruction`. Son instrucciones de edición, no descripciones independientes de escenas.
- **Licencia**: CC-BY-NC-SA-4.0, heredada del origen. Consulta su ficha para los derechos y las restricciones de uso.

## Preprocesamiento necesario

No hace falta extraer tar, crear captions por imagen ni compilar metadatos. Usa SimpleTuner con el [backend Parquet de Webshart](../DATALOADER.md#webshart-datasets), incluido en [PR #3325](https://github.com/bghira/SimpleTuner/pull/3325), e instala la integración publicada con el Hub:

```bash
python -m pip install --upgrade "webshart[hub]>=0.8.0"
```

El acceso público al Hub no requiere token. Configura `HF_TOKEN` para acceso autenticado. El manifiesto fija el origen en `05d2eea7b80ba72c077d39be43b7aca1a7857e6a` y verifica las sumas de los metadatos.

## Estructura del dataset

Cada fila de origen contiene `source_image`, `edited_image`, `edit_instruction`, dimensiones de ambas imágenes y puntuaciones de calidad. `webshart-parquet.json` relaciona los shards con `metadata/<task>/<part>.parquet`. Los metadatos no contienen imágenes. Como `id` se repite entre tareas, las uniones usan la ruta del shard y la posición física de la fila.

## Ejemplo de configuración del dataloader

Coloca este array en `multidatabackend.json`. Empieza por los 458 pares de `3.4_count_change`, un shard de unos 54 MB. Todas las imágenes originales y editadas están embebidas en este subconjunto fijado. Usa un modelo de edición que admita `reference_strict`, como [Flux Kontext](../quickstart/FLUX_KONTEXT.md); la configuración del modelo se establece aparte.

```json
[
  {
    "id": "scaleedit-edits",
    "type": "webshart",
    "source": "InternVL-U/ScaleEdit-12M",
    "metadata": "webshart/ScaleEdit-12M-metadata",
    "subfolder": "3.4_count_change",
    "caption_strategy": "webshart",
    "metadata_backend": "webshart",
    "resolution": 512,
    "resolution_type": "pixel",
    "crop": false,
    "dataset_type": "image",
    "conditioning_data": [
      "scaleedit-sources"
    ],
    "cache_dir_vae": "cache/vae/scaleedit/edited_image",
    "webshart": {
      "format": "parquet",
      "cache_dir": "cache/webshart/scaleedit/edited_image",
      "shard_cache_gb": 0,
      "parallel_downloads": 4,
      "columns": {
        "image": "edited_image",
        "captions": "edit_instruction",
        "width": "edited_image_width",
        "height": "edited_image_height"
      }
    }
  },
  {
    "id": "scaleedit-sources",
    "type": "webshart",
    "source": "InternVL-U/ScaleEdit-12M",
    "metadata": "webshart/ScaleEdit-12M-metadata",
    "subfolder": "3.4_count_change",
    "caption_strategy": "webshart",
    "metadata_backend": "webshart",
    "resolution": 512,
    "resolution_type": "pixel",
    "crop": false,
    "dataset_type": "conditioning",
    "conditioning_type": "reference_strict",
    "source_dataset_id": "scaleedit-edits",
    "cache_dir_vae": "cache/vae/scaleedit/source_image",
    "webshart": {
      "format": "parquet",
      "cache_dir": "cache/webshart/scaleedit/source_image",
      "shard_cache_gb": 0,
      "parallel_downloads": 4,
      "columns": {
        "image": "source_image",
        "captions": "edit_instruction",
        "width": "source_image_width",
        "height": "source_image_height"
      }
    }
  },
  {
    "id": "scaleedit-text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/scaleedit"
  }
]
```

## Explicación de la configuración

- `scaleedit-edits` usa `edited_image` como objetivo y `edit_instruction` como caption. Sus columnas de dimensiones corresponden a la imagen editada.
- `scaleedit-sources` usa `source_image` como condicionamiento. `conditioning_data` conecta el objetivo con las referencias y `source_dataset_id` las conecta de vuelta. Se seleccionan explícitamente las dimensiones originales.
- Ambas entradas comparten origen, metadatos y `subfolder`, conservando las identidades shard/fila. Mantén separados sus directorios de caché VAE y Webshart.
- `webshart.shard_cache_gb: 0` mantiene las lecturas por rangos. La mayoría de los shards completos rondan los 31 GB; la caché completa puede descargar un shard entero en la primera solicitud de imagen.
- `crop: false` evita recortes independientes en esta receta inicial. `reference_strict` aplica la geometría de preparación del objetivo a su referencia. Conserva un preprocesamiento compatible con el modelo de edición.

## Ampliar el ejemplo

Antes de cambiar o quitar `subfolder`, procesa las 702.078 imágenes originales disponibles solo mediante URL en el conjunto completo. Los metadatos conservan URL, SHA-256 y fecha de descarga, pero no las recuperan: un `source_image` ausente sigue siendo `None`. Descárgalas y verifícalas explícitamente, o crea subconjuntos alineados por ruta y fila.

No limites ni filtres los objetivos y las referencias de forma independiente: deben seleccionar las mismas muestras. Usar el mismo `max_num_samples` no garantiza los mismos pares cuando difieren los buckets.

Los grupos de filas y diccionarios grandes todavía pueden consumir cientos de MiB al leer imágenes, incluso sin caché completa. El bucketing usa los metadatos complementarios; acceder a imágenes tiene otro coste de memoria y red. El [benchmark del loader Webshart](../DATALOADER.md#parquet-image-bytes-on-s3) permite medir una muestra acotada antes de ampliar.
