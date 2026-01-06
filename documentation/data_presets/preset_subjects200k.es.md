# Subjects200K

## Detalles

- **Enlace en Hub**: [Yuanshi/Subjects200K](https://huggingface.co/datasets/Yuanshi/Subjects200K)
- **Descripción**: Más de 200K imágenes compuestas de alta calidad con descripciones emparejadas y evaluaciones de calidad. Cada muestra contiene dos imágenes lado a lado del mismo sujeto en contextos diferentes.
- **Formato(s) de caption**: JSON estructurado con descripciones separadas para cada mitad de la imagen
- **Características especiales**: Puntuaciones de evaluación de calidad, etiquetas de colección, imágenes compuestas

## Estructura del dataset

El dataset Subjects200K es único porque:
- Cada campo `image` contiene **dos imágenes combinadas lado a lado** en una sola imagen ancha
- Cada muestra tiene **dos captions separados**: uno para la imagen izquierda (`description.description_0`) y otro para la derecha (`description.description_1`)
- Los metadatos de evaluación de calidad permiten filtrar por métricas de calidad
- Las imágenes están pre-organizadas en colecciones

Ejemplo de estructura de datos:
```python
{
    'image': PIL.Image,  # Combined image (e.g., 1056x528 for two 528x528 images)
    'collection': 'collection_1',
    'quality_assessment': {
        'compositeStructure': 5,
        'objectConsistency': 5,
        'imageQuality': 5
    },
    'description': {
        'item': 'Eames Lounge Chair',
        'description_0': 'The Eames Lounge Chair is placed in a modern city living room...',
        'description_1': 'Nestled in a cozy nook of a rustic cabin...',
        'category': 'Furniture',
        'description_valid': True
    }
}
```

## Uso directo (sin preprocesamiento requerido)

A diferencia de datasets que requieren extracción y preprocesamiento, Subjects200K puede usarse directamente desde HuggingFace. El dataset ya está formateado y alojado correctamente.

Solo asegúrate de tener la librería `datasets` instalada:
```bash
pip install datasets
```

## Configuración del dataloader

Como cada muestra contiene dos imágenes, necesitamos configurar **dos entradas de dataset separadas**: una para cada mitad de la imagen compuesta:

```json
[
    {
        "id": "subjects200k-left",
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/subjects-left",
        "huggingface": {
            "caption_column": "description.description_0",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 0
            },
            "filter_func": {
                "collection": "collection_1",
                "quality_thresholds": {
                    "compositeStructure": 4.5,
                    "objectConsistency": 4.5,
                    "imageQuality": 4.5
                }
            }
        }
    },
    {
        "id": "subjects200k-right",
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/subjects-right",
        "huggingface": {
            "caption_column": "description.description_1",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 1
            },
            "filter_func": {
                "collection": "collection_1",
                "quality_thresholds": {
                    "compositeStructure": 4.5,
                    "objectConsistency": 4.5,
                    "imageQuality": 4.5
                }
            }
        }
    },
    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/flux"
    }
]
```

### Configuración explicada

#### Configuración de imagen compuesta
- `composite_image_config.enabled`: Activa el manejo de imágenes compuestas
- `composite_image_config.image_count`: Número de imágenes en la compuesta (2 para lado a lado)
- `composite_image_config.select_index`: Qué imagen extraer (0 = izquierda, 1 = derecha)

#### Filtrado por calidad
El `filter_func` permite filtrar muestras basado en:
- `collection`: Usar solo imágenes de colecciones específicas
- `quality_thresholds`: Puntuaciones mínimas para métricas de calidad:
  - `compositeStructure`: Qué tan bien funcionan juntas las dos imágenes
  - `objectConsistency`: Consistencia del sujeto en ambas imágenes
  - `imageQuality`: Calidad general de la imagen

#### Selección de captions
- La imagen izquierda usa: `"caption_column": "description.description_0"`
- La imagen derecha usa: `"caption_column": "description.description_1"`

### Opciones de personalización

1. **Ajustar umbrales de calidad**: Valores más bajos (p. ej., 4.0) incluyen más imágenes, valores más altos (p. ej., 4.8) son más selectivos

2. **Usar distintas colecciones**: Cambia `"collection": "collection_1"` por otras colecciones disponibles en el dataset

3. **Cambiar resolución**: Ajusta el valor de `resolution` según tus necesidades de entrenamiento

4. **Desactivar filtrado**: Elimina la sección `filter_func` para usar todas las imágenes

5. **Usar nombres de ítem como captions**: Cambia la columna de caption a `"description.item"` para usar solo el nombre del sujeto

### Tips

- El dataset se descargará y cacheará automáticamente en el primer uso
- Cada "mitad" se trata como un dataset independiente, duplicando efectivamente tus muestras de entrenamiento
- Considera usar distintos umbrales de calidad para cada mitad si quieres variedad
- Los directorios de caché VAE deben ser diferentes para cada mitad para evitar conflictos
