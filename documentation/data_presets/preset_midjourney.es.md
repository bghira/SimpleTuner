# Midjourney v6 520k

## Detalles

- **Enlace en Hub**: [terminusresearch/midjourney-v6-520k-raw](https://huggingface.co/datasets/terminusresearch/midjourney-v6-520k-raw)
- **Descripción**: ~520.000 salidas de alta calidad en las que cualquier prompt en japonés se volvió a captionar con GPT-3.5-Turbo.
- **Formato(s) de caption**: Parquet

## Almacenamiento requerido

Este dataset contiene todos los datos de imagen, y por lo tanto será difícil de extraer sin suficiente espacio en disco. **Asegúrate de tener al menos 1,5 TB de espacio disponible para extraerlo.**

Los embeddings de texto T5-XXL para este modelo consumirán ~520 GB incluso con `--compress_disk_cache` habilitado.
Los embeddings VAE consumirán entre 80 y 100 GB de espacio, según el modelo entrenado y la resolución de los embeddings.


## Descarga

```bash
huggingface-cli download --repo-type=dataset terminusresearch/midjourney-v6-520k-raw --local-dir=midjourney-v6-520k-raw
```

Esto descargará simultáneamente los segmentos tar fragmentados desde Hugging Face Hub.

## Extracción

```bash
cd midjourney-v6-520k-raw
cat *.tar | tar x
```

Esto creará una carpeta que contiene todas las muestras dentro del directorio actual.

## Ejemplo de configuración del dataloader

```json
{
    "id": "midjourney-v6-520k-raw",
    "type": "local",
    "cache_dir_vae": "cache/vae-mj-520k/",
    "crop": true,
    "crop_aspect": "square",
    "resolution": 1.0,
    "maximum_image_size": 1.0,
    "minimum_image_size": 0.75,
    "target_downsample_size": 1.00,
    "resolution_type": "area",
    "caption_strategy": "parquet",
    "metadata_backend": "parquet",
    "parquet": {
        "path": "/path/to/midjourney-v6-520k-raw/train.parquet",
        "caption_column": "gpt_caption",
        "filename_column": "id",
        "width_column": "width",
        "height_column": "height",
        "identifier_includes_extension": false
    }
}
```
