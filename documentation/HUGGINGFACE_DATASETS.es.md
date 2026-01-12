# Integración con Hugging Face Datasets

SimpleTuner soporta cargar datasets directamente desde Hugging Face Hub, permitiendo entrenamiento eficiente en datasets a gran escala sin descargar todas las imágenes localmente.

## Resumen

El backend de datasets de Hugging Face te permite:
- Cargar datasets directamente desde Hugging Face Hub
- Aplicar filtros basados en metadatos o métricas de calidad
- Extraer captions de columnas del dataset
- Manejar imágenes compuestas/en cuadrícula
- Cachear solo los embeddings procesados localmente

**Importante**: SimpleTuner requiere acceso completo al dataset para construir buckets de relación de aspecto y calcular tamaños de batch. Aunque Hugging Face soporta datasets en streaming, esta función no es compatible con la arquitectura de SimpleTuner. Usa filtros para reducir datasets muy grandes a tamaños manejables.

## Configuración básica

Para usar un dataset de Hugging Face, configura tu dataloader con `"type": "huggingface"`:

```json
{
  "id": "my-hf-dataset",
  "type": "huggingface",
  "dataset_name": "username/dataset-name",
  "split": "train",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "text",
  "image_column": "image",
  "cache_dir": "cache/my-hf-dataset"
}
```

### Campos requeridos

- `type`: Debe ser `"huggingface"`
- `dataset_name`: El identificador del dataset en Hugging Face (p. ej., "laion/laion-aesthetic")
- `caption_strategy`: Debe ser `"huggingface"`
- `metadata_backend`: Debe ser `"huggingface"`

### Campos opcionales

- `split`: Split del dataset a usar (default: "train")
- `revision`: Revisión específica del dataset
- `image_column`: Columna que contiene imágenes (default: "image")
- `caption_column`: Columna(s) que contienen captions
- `cache_dir`: Directorio de caché local para archivos del dataset
- `streaming`: ⚠️ **Actualmente no funcional** - SimpleTuner intenta escanear eficientemente el dataset para construir metadatos y cachés del encoder.
- `num_proc`: Número de procesos para filtrado (default: 16)

## Configuración de captions

El backend de Hugging Face soporta extracción flexible de captions:

### Columna de caption única
```json
{
  "caption_column": "caption"
}
```

### Múltiples columnas de caption
```json
{
  "caption_column": ["short_caption", "detailed_caption", "tags"]
}
```

### Acceso a columna anidada
```json
{
  "caption_column": "metadata.caption",
  "fallback_caption_column": "basic_caption"
}
```

### Configuración avanzada de captions
```json
{
  "huggingface": {
    "caption_column": "caption",
    "fallback_caption_column": "description",
    "description_column": "detailed_description",
    "width_column": "width",
    "height_column": "height"
  }
}
```

## Filtrar datasets

Aplica filtros para seleccionar solo muestras de alta calidad:

### Filtrado por calidad
```json
{
  "huggingface": {
    "filter_func": {
      "quality_thresholds": {
        "clip_score": 0.3,
        "aesthetic_score": 5.0,
        "resolution": 0.8
      },
      "quality_column": "quality_assessment"
    }
  }
}
```

### Filtrado por colección/subconjunto
```json
{
  "huggingface": {
    "filter_func": {
      "collection": ["photo", "artwork"],
      "min_width": 512,
      "min_height": 512
    }
  }
}
```

## Soporte de imágenes compuestas

Maneja datasets con múltiples imágenes en una cuadrícula:

```json
{
  "huggingface": {
    "composite_image_config": {
      "enabled": true,
      "image_count": 4,
      "select_index": 0
    }
  }
}
```

Esta configuración:
- Detectará grids de 4 imágenes
- Extraerá solo la primera imagen (índice 0)
- Ajustará las dimensiones en consecuencia

## Configuraciones completas de ejemplo

### Dataset básico de fotos
```json
{
  "id": "aesthetic-photos",
  "type": "huggingface",
  "dataset_name": "aesthetic-foundation/aesthetic-photos",
  "split": "train",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "caption",
  "image_column": "image",
  "resolution": 1024,
  "resolution_type": "pixel",
  "minimum_image_size": 512,
  "cache_dir": "cache/aesthetic-photos"
}
```

### Dataset filtrado de alta calidad
```json
{
  "id": "high-quality-art",
  "type": "huggingface",
  "dataset_name": "example/art-dataset",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "huggingface": {
    "caption_column": ["title", "description", "tags"],
    "fallback_caption_column": "filename",
    "width_column": "original_width",
    "height_column": "original_height",
    "filter_func": {
      "quality_thresholds": {
        "aesthetic_score": 6.0,
        "technical_quality": 0.8
      },
      "min_width": 768,
      "min_height": 768
    }
  },
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "crop": true,
  "crop_aspect": "square"
}
```

### Dataset de video
```json
{
  "id": "video-dataset",
  "type": "huggingface",
  "dataset_type": "video",
  "dataset_name": "example/video-clips",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "huggingface": {
    "caption_column": "description",
    "num_frames_column": "frame_count",
    "fps_column": "fps"
  },
  "video": {
    "num_frames": 125,
    "min_frames": 100
  },
  "resolution": 480,
  "resolution_type": "pixel"
}
```

## Sistema de archivos virtual

El backend de Hugging Face usa un sistema de archivos virtual donde las imágenes se referencian por su índice en el dataset:
- `0.jpg` → Primer ítem del dataset
- `1.jpg` → Segundo ítem del dataset
- etc.

Esto permite que el pipeline estándar de SimpleTuner funcione sin modificaciones.

## Comportamiento de caché

- **Archivos del dataset**: Cacheados según los defaults de la librería Hugging Face datasets
- **Embeddings VAE**: Almacenados en `cache_dir/vae/{backend_id}/`
- **Text embeddings**: Usan la configuración estándar de caché de text embeds
- **Metadatos**: Almacenados en `cache_dir/huggingface_metadata/{backend_id}/`

## Consideraciones de rendimiento

1. **Escaneo inicial**: La primera ejecución descargará metadatos del dataset y construirá buckets de relación de aspecto
2. **Tamaño del dataset**: Deben cargarse todos los metadatos del dataset para construir listas de archivos y calcular longitudes
3. **Filtrado**: Se aplica durante la carga inicial - los ítems filtrados no se descargarán
4. **Reutilización de caché**: Las ejecuciones posteriores reutilizan metadatos y embeddings cacheados

**Nota**: Aunque Hugging Face datasets soporta streaming, SimpleTuner requiere acceso completo al dataset para construir buckets de aspecto y calcular tamaños de batch. Los datasets muy grandes deben filtrarse a un tamaño manejable.

## Limitaciones

- Acceso de solo lectura (no puede modificar el dataset fuente)
- Requiere conexión a internet para acceso inicial al dataset
- Algunos formatos de dataset pueden no estar soportados
- El modo streaming no está soportado - SimpleTuner requiere acceso completo al dataset
- Los datasets muy grandes deben filtrarse a tamaños manejables
- La carga inicial de metadatos puede ser intensiva en memoria para datasets enormes

## Troubleshooting

### Dataset no encontrado
```
Error: Dataset 'username/dataset' not found
```
- Verifica que el dataset exista en Hugging Face Hub
- Comprueba si el dataset es privado (requiere autenticación)
- Asegúrate de que el nombre del dataset esté bien escrito

### Carga inicial lenta
- Los datasets grandes tardan en cargar metadatos y construir buckets
- Usa filtrado agresivo para reducir el tamaño del dataset
- Considera usar un subconjunto o una versión filtrada del dataset
- Los archivos de caché acelerarán ejecuciones posteriores

### Problemas de memoria
- Usa filtros para reducir el tamaño del dataset antes de cargar
- Reduce `num_proc` para operaciones de filtrado
- Considera dividir datasets muy grandes en chunks más pequeños
- Usa umbrales de calidad para limitar el dataset a muestras de alta calidad

### Problemas de extracción de captions
- Verifica que los nombres de columna coincidan con el esquema del dataset
- Comprueba estructuras de columnas anidadas
- Usa `fallback_caption_column` para captions faltantes

## Uso avanzado

### Funciones de filtro personalizadas

Aunque la configuración soporta filtrado básico, puedes implementar filtros más complejos modificando el código. La función de filtro recibe cada ítem del dataset y devuelve True/False.

### Entrenamiento multi-dataset

Combina datasets de Hugging Face con datos locales:

```json
[
  {
    "id": "hf-dataset",
    "type": "huggingface",
    "dataset_name": "laion/laion-art",
    "probability": 0.7
  },
  {
    "id": "local-dataset",
    "type": "local",
    "instance_data_dir": "/path/to/local/data",
    "probability": 0.3
  }
]
```

Esta configuración muestreará 70% del dataset de Hugging Face y 30% de datos locales.

## Datasets de audio

Para modelos de audio (como ACE-Step), puedes especificar `dataset_type: "audio"`.

```json
{
    "id": "audio-dataset",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "my-audio-data",
    "audio_column": "audio",
    "config": {
        "audio_caption_fields": ["tags"],
        "lyrics_column": "lyrics"
    }
}
```

*   **`audio_column`**: La columna que contiene datos de audio (decodificados o bytes). Default: `"audio"`.
*   **`audio_caption_fields`**: Lista de nombres de columnas que se combinan para formar el **prompt** (condicionamiento de texto). Default: `["prompt", "tags"]`.
*   **`lyrics_column`**: La columna que contiene las letras. Default: `"lyrics"`. Si falta esta columna, SimpleTuner buscará `"norm_lyrics"` como fallback.

### Columnas esperadas
*   **`audio`**: Los datos de audio.
*   **`prompt`** / **`tags`**: Tags descriptivos o prompts usados por el encoder de texto.
*   **`lyrics`**: Letras de canciones usadas por el encoder de letras.
