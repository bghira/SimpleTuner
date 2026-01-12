# Archivo de configuración del dataloader

Aquí está el ejemplo más básico de un archivo de configuración del dataloader, como `multidatabackend.example.json`.

```json
[
  {
    "id": "something-special-to-remember-by",
    "type": "local",
    "instance_data_dir": "/path/to/data/tree",
    "crop": true,
    "crop_style": "center",
    "crop_aspect": "square",
    "resolution": 1024,
    "minimum_image_size": 768,
    "maximum_image_size": 2048,
    "minimum_aspect_ratio": 0.50,
    "maximum_aspect_ratio": 3.00,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "prepend_instance_prompt": false,
    "instance_prompt": "something to label every image",
    "only_instance_prompt": false,
    "caption_strategy": "textfile",
    "cache_dir_vae": "/path/to/vaecache",
    "repeats": 0
  },
  {
    "id": "an example backend for text embeds.",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "aws",
    "aws_bucket_name": "textembeds-something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir": ""
  }
]
```

## Opciones de configuración

### `id`

- **Descripción:** Identificador único para el dataset. Debe permanecer constante una vez definido, ya que lo vincula a sus entradas de seguimiento de estado.

### `disabled`

- **Valores:** `true` | `false`
- **Descripción:** Cuando se establece en `true`, este dataset se omite por completo durante el entrenamiento. Útil para excluir temporalmente un dataset sin eliminar su configuración.
- **Nota:** También acepta la grafía `disable`.

### `dataset_type`

- **Valores:** `image` | `video` | `audio` | `text_embeds` | `image_embeds` | `conditioning_image_embeds` | `conditioning`
- **Descripción:** Los datasets `image`, `video` y `audio` contienen muestras de entrenamiento primarias. `text_embeds` contiene las salidas de la caché del encoder de texto, `image_embeds` contiene los latentes del VAE (cuando un modelo usa uno), y `conditioning_image_embeds` almacena embeddings de imagen de condicionamiento cacheados (por ejemplo, features de visión de CLIP). Cuando un dataset está marcado como `conditioning`, se puede emparejar con tu dataset `image` mediante [la opción conditioning_data](#conditioning_data)
- **Nota:** Los datasets de text y image embeds se definen de forma diferente a los datasets de imagen. Un dataset de text embeds almacena SOLO los objetos de text embed. Un dataset de imagen almacena los datos de entrenamiento.
- **Nota:** No combines imágenes y video en un **único** dataset. Sepáralos.

### `default`

- **Solo aplica a `dataset_type=text_embeds`**
- Si se establece en `true`, este dataset de text embeds será donde SimpleTuner almacene la caché de text embeds para, p. ej., embeds de prompts de validación. Como no se emparejan con datos de imagen, se necesita una ubicación específica donde terminen.

### `cache_dir`

- **Solo aplica a `dataset_type=text_embeds` y `dataset_type=image_embeds`**
- **Descripción:** Especifica dónde se almacenan los archivos de caché de embeds para este dataset. Para `text_embeds`, aquí se escriben las salidas del encoder de texto. Para `image_embeds`, aquí se almacenan los latentes del VAE.
- **Nota:** Es diferente de `cache_dir_vae`, que se configura en datasets de imagen/video principales para indicar dónde va su caché VAE.

### `write_batch_size`

- **Solo aplica a `dataset_type=text_embeds`**
- **Descripción:** Número de text embeds que se escriben en una sola operación por batch. Valores más altos pueden mejorar el throughput de escritura pero usan más memoria.
- **Default:** Usa el argumento `--write_batch_size` del trainer (normalmente 128).

### `text_embeds`

- **Solo aplica a `dataset_type=image`**
- Si no se configura, se usará el dataset `text_embeds` marcado como `default`. Si se configura con el `id` existente de un dataset `text_embeds`, se usará ese en su lugar. Permite asociar datasets de text embeds específicos a un dataset de imagen.

### `image_embeds`

- **Solo aplica a `dataset_type=image`**
- Si no se configura, las salidas del VAE se almacenarán en el backend de imagen. En caso contrario, puedes establecerlo al `id` de un dataset `image_embeds`, y las salidas del VAE se almacenarán allí. Permite asociar el dataset image_embed a los datos de imagen.

### `conditioning_image_embeds`

- **Aplica a `dataset_type=image` y `dataset_type=video`**
- Cuando un modelo reporta `requires_conditioning_image_embeds`, establece esto al `id` de un dataset `conditioning_image_embeds` para almacenar embeddings de imagen de condicionamiento cacheados (por ejemplo, features de visión de CLIP para Wan 2.2 I2V). Si se omite, SimpleTuner escribe la caché en `cache/conditioning_image_embeds/<dataset_id>` por defecto, garantizando que ya no colisione con la caché VAE.
- Los modelos que necesitan estos embeds deben exponer un encoder de imagen a través de su pipeline principal. Si el modelo no puede suministrar el encoder, el preprocesamiento fallará pronto en lugar de generar archivos vacíos silenciosamente.

#### `cache_dir_conditioning_image_embeds`

- **Override opcional para el destino de caché de conditioning image embed.**
- Configúralo cuando quieras fijar la caché a una ubicación específica del filesystem o tener un backend remoto dedicado (`dataset_type=conditioning_image_embeds`). Si se omite, se usa automáticamente la ruta de caché descrita arriba.

#### `conditioning_image_embed_batch_size`

- **Override opcional para el tamaño de batch usado al generar conditioning image embeds.**
- Por defecto usa el argumento `conditioning_image_embed_batch_size` del trainer o el tamaño de batch del VAE cuando no se proporciona explícitamente.

### Configuración de dataset de audio (`dataset_type=audio`)

Los backends de audio admiten un bloque `audio` dedicado para que los metadatos y el cálculo de buckets tengan en cuenta la duración. Ejemplo:

```json
"audio": {
  "max_duration_seconds": 90,
  "channels": 2,
  "bucket_strategy": "duration",
  "duration_interval": 15,
  "truncation_mode": "beginning"
}
```

- **`bucket_strategy`** – actualmente `duration` es el valor por defecto y recorta clips en buckets espaciados uniformemente para que el muestreo por GPU respete el cálculo de batch.
- **`duration_interval`** – redondeo del bucket en segundos (por defecto **3** si no se configura). Con `15`, un clip de 77 s queda en el bucket de 75 s. Esto evita que un clip largo deje a otras ranks sin muestras y fuerza el truncado al mismo intervalo.
- **`max_duration_seconds`** – los clips más largos que esto se omiten por completo durante el descubrimiento de metadatos, así que pistas excepcionalmente largas no consumen buckets inesperadamente.
- **`truncation_mode`** – determina qué parte del clip se conserva cuando ajustamos al intervalo del bucket. Opciones: `beginning`, `end` o `random` (default: `beginning`).
- La configuración estándar de audio (número de canales, directorio de caché) se mapea directamente al backend de audio en tiempo de ejecución creado por `simpletuner.helpers.data_backend.factory`. Se evita el padding intencionalmente: los clips se truncan en lugar de extenderse para mantener el comportamiento consistente con entrenadores de difusión como ACE-Step.

### Captions de audio (Hugging Face)
Para datasets de audio de Hugging Face, puedes especificar qué columnas forman el caption (prompt) y qué columna contiene las letras:
```json
"config": {
    "audio_caption_fields": ["prompt", "tags"],
    "lyrics_column": "lyrics"
}
```
*   `audio_caption_fields`: Une varias columnas para formar el prompt de texto (usado por el encoder de texto).
*   `lyrics_column`: Especifica la columna de letras (usado por el encoder de letras).

Durante el descubrimiento de metadatos, el loader registra `sample_rate`, `num_samples`, `num_channels` y `duration_seconds` para cada archivo. Los reportes de buckets en la CLI ahora hablan en **muestras** en lugar de **imágenes**, y los diagnósticos de dataset vacío enumerarán el `bucket_strategy`/`duration_interval` activo (más cualquier límite de `max_duration_seconds`) para que puedas ajustar intervalos sin entrar en los logs.

### `type`

- **Valores:** `aws` | `local` | `csv` | `huggingface`
- **Descripción:** Determina el backend de almacenamiento (local, csv o cloud) usado para este dataset.

### `conditioning_type`

- **Valores:** `controlnet` | `mask` | `reference_strict` | `reference_loose`
- **Descripción:** Especifica el tipo de condicionamiento para un dataset `conditioning`.
  - **controlnet**: Entradas de condicionamiento ControlNet para entrenamiento con señal de control.
  - **mask**: Máscaras binarias para entrenamiento de inpainting.
  - **reference_strict**: Imágenes de referencia con alineación estricta de píxeles (para modelos de edición como Qwen Edit).
  - **reference_loose**: Imágenes de referencia con alineación flexible.

### `source_dataset_id`

- **Solo aplica a `dataset_type=conditioning`** con `conditioning_type` de `reference_strict`, `reference_loose` o `mask`
- **Descripción:** Vincula un dataset de condicionamiento con su dataset fuente de imagen/video para alineación de píxeles. Cuando se establece, SimpleTuner duplica metadatos del dataset fuente para garantizar que las imágenes de condicionamiento se alineen con sus objetivos.
- **Nota:** Requerido para modos de alineación estricta; opcional para alineación flexible.

### `conditioning_data`

- **Valores:** valor `id` de dataset de condicionamiento o un arreglo de valores `id`
- **Descripción:** Como se describe en [la guía de ControlNet](CONTROLNET.md), un dataset `image` puede emparejarse con sus datos de ControlNet o máscaras de imagen mediante esta opción.
- **Nota:** Si tienes varios datasets de condicionamiento, puedes especificarlos como un arreglo de valores `id`. Al entrenar Flux Kontext, esto permite cambiar aleatoriamente entre condiciones o unir entradas para entrenar tareas avanzadas de composición multi-imagen.

### `instance_data_dir` / `aws_data_prefix`

- **Local:** Ruta a los datos en el filesystem.
- **AWS:** Prefijo S3 de los datos en el bucket.

### `caption_strategy`

- **textfile** requiere que tu image.png esté junto a un image.txt que contenga uno o más captions, separados por saltos de línea. Estos pares imagen+texto **deben estar en el mismo directorio**.
- **instanceprompt** requiere que se proporcione un valor para `instance_prompt` y usará **solo** este valor para el caption de cada imagen en el conjunto.
- **filename** usará una versión convertida y limpiada del nombre de archivo como su caption, p. ej., sustituyendo guiones bajos por espacios.
- **parquet** extraerá captions de la tabla parquet que contiene el resto de los metadatos de imagen. Usa el campo `parquet` para configurarlo. Consulta [Estrategia de caption parquet](#parquet-caption-strategy-json-lines-datasets).

Tanto `textfile` como `parquet` soportan multi-captions:
- Los textfiles se dividen por saltos de línea. Cada nueva línea será un caption independiente.
- Las tablas parquet pueden tener un tipo iterable en el campo.

### `metadata_backend`

- **Valores:** `discovery` | `parquet` | `huggingface`
- **Descripción:** Controla cómo SimpleTuner descubre dimensiones de imagen y otros metadatos durante la preparación del dataset.
  - **discovery** (default): Escanea archivos de imagen reales para leer dimensiones. Funciona con cualquier backend de almacenamiento, pero puede ser lento para datasets grandes.
  - **parquet**: Lee dimensiones desde `width_column` y `height_column` en un archivo parquet/JSONL, omitiendo acceso a archivos. Consulta [Estrategia de caption parquet](#parquet-caption-strategy-json-lines-datasets).
  - **huggingface**: Usa metadatos de datasets de Hugging Face. Consulta [Hugging Face Datasets Support](#hugging-face-datasets-support).
- **Nota:** Cuando usas `parquet`, también debes configurar el bloque `parquet` con `width_column` y `height_column`. Esto acelera drásticamente el inicio en datasets grandes.

### `metadata_update_interval`

- **Valores:** Integer (segundos)
- **Descripción:** Con qué frecuencia (en segundos) refrescar metadatos del dataset durante el entrenamiento. Útil para datasets que puedan cambiar durante una ejecución larga.
- **Default:** Usa el argumento `--metadata_update_interval` del trainer.

### Opciones de recorte

- `crop`: Habilita o deshabilita el recorte de imágenes.
- `crop_style`: Selecciona el estilo de recorte (`random`, `center`, `corner`, `face`).
- `crop_aspect`: Elige el aspecto de recorte (`closest`, `random`, `square` o `preserve`).
- `crop_aspect_buckets`: Cuando `crop_aspect` se establece en `closest` o `random`, se seleccionará un bucket de esta lista. Por defecto, todos los buckets están disponibles (permitiendo escalado ilimitado). Usa `max_upscale_threshold` para limitar el escalado si es necesario.

### `resolution`

- **resolution_type=area:** El tamaño final de la imagen se determina por el conteo de megapíxeles: un valor de 1.05 aquí corresponderá a buckets de aspecto alrededor de 1024^2 (1024x1024) de área total de píxeles, ~1_050_000 píxeles.
- **resolution_type=pixel_area:** Como `area`, el tamaño final se determina por su área, pero mide en píxeles en lugar de megapíxeles. Un valor de 1024 aquí generará buckets de aspecto alrededor de 1024^2 (1024x1024) de área total de píxeles, ~1_050_000 píxeles.
- **resolution_type=pixel:** El tamaño final de la imagen se determinará por el borde más corto siendo este valor.

> **NOTA**: Que las imágenes se escalen, reduzcan o recorten depende de los valores de `minimum_image_size`, `maximum_target_size`, `target_downsample_size`, `crop` y `crop_aspect`.

### `minimum_image_size`

- Cualquier imagen cuyo tamaño final quede por debajo de este valor será **excluida** del entrenamiento.
- Cuando `resolution` se mide en megapíxeles (`resolution_type=area`), esto debe estar en megapíxeles también (p. ej., `1.05` megapíxeles para excluir imágenes bajo 1024x1024 de **área**)
- Cuando `resolution` se mide en píxeles, usa la misma unidad aquí (p. ej., `1024` para excluir imágenes con el **borde más corto** menor a 1024 px)
- **Recomendación**: Mantén `minimum_image_size` igual a `resolution` salvo que quieras arriesgarte a entrenar con imágenes mal escaladas.

### `minimum_aspect_ratio`

- **Descripción:** La relación de aspecto mínima de la imagen. Si la relación de aspecto de la imagen es menor que este valor, se excluirá del entrenamiento.
- **Nota**: Si el número de imágenes que califican para exclusión es excesivo, esto podría desperdiciar tiempo en el arranque porque el trainer intentará escanearlas y hacer bucketing si faltan en las listas de buckets.

> **Nota**: Una vez que las listas de aspecto y metadatos están construidas para tu dataset, usar `skip_file_discovery="vae aspect metadata"` evitará que el trainer escanee el dataset en el arranque, ahorrando mucho tiempo.

### `maximum_aspect_ratio`

- **Descripción:** La relación de aspecto máxima de la imagen. Si la relación de aspecto de la imagen es mayor que este valor, se excluirá del entrenamiento.
- **Nota**: Si el número de imágenes que califican para exclusión es excesivo, esto podría desperdiciar tiempo en el arranque porque el trainer intentará escanearlas y hacer bucketing si faltan en las listas de buckets.

> **Nota**: Una vez que las listas de aspecto y metadatos están construidas para tu dataset, usar `skip_file_discovery="vae aspect metadata"` evitará que el trainer escanee el dataset en el arranque, ahorrando mucho tiempo.

### `conditioning`

- **Valores:** Arreglo de objetos de configuración de condicionamiento
- **Descripción:** Genera automáticamente datasets de condicionamiento a partir de tus imágenes fuente. Cada tipo de condicionamiento crea un dataset separado que puede usarse para entrenamiento de ControlNet u otras tareas de condicionamiento.
- **Nota:** Cuando se especifica, SimpleTuner creará automáticamente datasets de condicionamiento con IDs como `{source_id}_conditioning_{type}`

Cada objeto de condicionamiento puede contener:
- `type`: El tipo de condicionamiento a generar (requerido)
- `params`: Parámetros específicos del tipo (opcional)
- `captions`: Estrategia de captions para el dataset generado (opcional)
  - Puede ser `false` (sin captions)
  - Una sola cadena (usada como instance prompt para todas las imágenes)
  - Un arreglo de cadenas (seleccionadas aleatoriamente para cada imagen)
  - Si se omite, se usan los captions del dataset de origen

#### Tipos de condicionamiento disponibles

##### `superresolution`
Genera versiones de baja calidad de las imágenes para entrenamiento de super-resolución:
```json
{
  "type": "superresolution",
  "blur_radius": 2.5,
  "blur_type": "gaussian",
  "add_noise": true,
  "noise_level": 0.03,
  "jpeg_quality": 85,
  "downscale_factor": 2
}
```

##### `jpeg_artifacts`
Crea artefactos de compresión JPEG para entrenamiento de eliminación de artefactos:
```json
{
  "type": "jpeg_artifacts",
  "quality_mode": "range",
  "quality_range": [10, 30],
  "compression_rounds": 1,
  "enhance_blocks": false
}
```

##### `depth` / `depth_midas`
Genera mapas de profundidad usando modelos DPT:
```json
{
  "type": "depth_midas",
  "model_type": "DPT"
}
```
**Nota:** La generación de profundidad requiere GPU y se ejecuta en el proceso principal, lo que puede ser más lento que los generadores basados en CPU.

##### `random_masks` / `inpainting`
Crea máscaras aleatorias para entrenamiento de inpainting:
```json
{
  "type": "random_masks",
  "mask_types": ["rectangle", "circle", "brush", "irregular"],
  "min_coverage": 0.1,
  "max_coverage": 0.5,
  "output_mode": "mask"
}
```

##### `canny` / `edges`
Genera mapas de detección de bordes Canny:
```json
{
  "type": "canny",
  "low_threshold": 100,
  "high_threshold": 200
}
```

Consulta [la guía de ControlNet](CONTROLNET.md) para más detalles sobre cómo usar estos datasets de condicionamiento.

#### Ejemplos

##### Dataset de video

Un dataset de video debería ser una carpeta de archivos de video (p. ej., mp4) y los métodos habituales de almacenamiento de captions.

```json
[
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/ltxvideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 125,
        "min_frames": 125
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

- En la subsección `video`, tenemos las siguientes claves que podemos configurar:
  - `num_frames` (opcional, int) es cuántos frames de datos entrenaremos.
    - A 25 fps, 125 frames son 5 segundos de video, salida estándar. Este debería ser tu objetivo.
  - `min_frames` (opcional, int) determina la longitud mínima de un video que se considerará para entrenamiento.
    - Esto debería ser al menos igual a `num_frames`. No configurarlo garantiza que sea igual.
  - `max_frames` (opcional, int) determina la longitud máxima de un video que se considerará para entrenamiento.
  - `is_i2v` (opcional, bool) determina si el entrenamiento i2v se realizará en un dataset.
    - Esto se establece en True por defecto para LTX. Puedes desactivarlo, sin embargo.
  - `bucket_strategy` (opcional, string) determina cómo se agrupan los videos en buckets:
    - `aspect_ratio` (default): Bucketing solo por relación de aspecto espacial (p. ej., `1.78`, `0.75`). Mismo comportamiento que los datasets de imagen.
    - `resolution_frames`: Bucketing por resolución y conteo de frames en formato `WxH@F` (p. ej., `1920x1080@125`). Útil para entrenar en datasets con resoluciones y duraciones variables.
  - `frame_interval` (opcional, int) cuando se usa `bucket_strategy: "resolution_frames"`, los conteos de frames se redondean hacia abajo al múltiplo más cercano de este valor. Configúralo al factor de conteo de frames requerido por tu modelo (algunos modelos requieren que `num_frames - 1` sea divisible por cierto valor).

**Ajuste Automático de Conteo de Frames:** SimpleTuner ajusta automáticamente los conteos de frames de video para satisfacer las restricciones específicas del modelo. Por ejemplo, LTX-2 requiere conteos de frames que satisfagan `frames % 8 == 1` (p. ej., 49, 57, 65, 73, 81, etc.). Si tus videos tienen conteos de frames diferentes (p. ej., 119 frames), se recortan automáticamente al conteo de frames válido más cercano (p. ej., 113 frames). Los videos que se acortan a menos de `min_frames` después del ajuste se omiten con un mensaje de advertencia. Este ajuste automático evita errores de entrenamiento y no requiere ninguna configuración de tu parte.

**Nota:** Al usar `bucket_strategy: "resolution_frames"` con `num_frames` configurado, obtendrás un único bucket de frames y los videos más cortos que `num_frames` se descartarán. Quita `num_frames` si quieres múltiples buckets de frames con menos descartes.

Ejemplo de bucketing `resolution_frames` para datasets de video con resoluciones mezcladas:

```json
{
  "id": "mixed-resolution-videos",
  "type": "local",
  "dataset_type": "video",
  "resolution": 720,
  "resolution_type": "pixel_area",
  "instance_data_dir": "datasets/videos",
  "video": {
      "bucket_strategy": "resolution_frames",
      "frame_interval": 25,
      "min_frames": 25,
      "max_frames": 250
  }
}
```

Esta configuración creará buckets como `1280x720@100`, `1920x1080@125`, `640x480@75`, etc. Los videos se agrupan por su resolución de entrenamiento y conteo de frames (redondeado a los 25 frames más cercanos).


##### Configuración
```json
    "minimum_image_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel"
```
##### Resultado
- Cualquier imagen con un borde más corto menor a **1024px** se excluirá por completo del entrenamiento.
- Imágenes como `768x1024` o `1280x768` serían excluidas, pero `1760x1024` y `1024x1024` no.
- Ninguna imagen se escalará hacia arriba, porque `minimum_image_size` es igual a `resolution`

##### Configuración
```json
    "minimum_image_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel_area" # different from the above configuration, which is 'pixel'
```
##### Resultado
- Si el área total de la imagen (ancho * alto) es menor que el área mínima (1024 * 1024), se excluirá del entrenamiento.
- Imágenes como `1280x960` **no** se excluirían porque `(1280 * 960)` es mayor que `(1024 * 1024)`
- Ninguna imagen se escalará hacia arriba, porque `minimum_image_size` es igual a `resolution`

##### Configuración
```json
    "minimum_image_size": 0, # or completely unset, not present in the config
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": false
```

##### Resultado
- Las imágenes se redimensionarán para que su borde más corto sea 1024px manteniendo la relación de aspecto
- No se excluirán imágenes por tamaño
- Las imágenes pequeñas se escalarán hacia arriba usando métodos ingenuos de `PIL.resize` que no se ven bien
  - Se recomienda evitar el escalado hacia arriba salvo que lo hagas manualmente con un upscaler de tu elección antes de empezar el entrenamiento

### `maximum_image_size` y `target_downsample_size`

Las imágenes no se redimensionan antes del recorte **a menos que** `maximum_image_size` y `target_downsample_size` estén configurados. En otras palabras, una imagen de `4096x4096` se recortará directamente a un objetivo de `1024x1024`, lo cual puede ser indeseable.

- `maximum_image_size` especifica el umbral a partir del cual comenzará el redimensionado. Reducirá imágenes antes del recorte si son mayores que esto.
- `target_downsample_size` especifica qué tan grande será la imagen después del reescalado y antes del recorte.

#### Ejemplos

##### Configuración
```json
    "resolution_type": "pixel_area",
    "resolution": 1024,
    "maximum_image_size": 1536,
    "target_downsample_size": 1280,
    "crop": true,
    "crop_aspect": "square"
```

##### Resultado
- Cualquier imagen con un área de píxeles mayor que `(1536 * 1536)` se redimensionará para que su área de píxeles sea aproximadamente `(1280 * 1280)` manteniendo su relación de aspecto original
- El tamaño final de la imagen se recortará aleatoriamente a un área de píxeles de `(1024 * 1024)`
- Útil para entrenar con, p. ej., datasets de 20 megapíxeles que necesitan reducirse de forma significativa antes del recorte para evitar una gran pérdida de contexto en la escena (como recortar una foto de una persona y quedarte solo con una pared de azulejos o una sección borrosa del fondo)

### `max_upscale_threshold`

Por defecto, SimpleTuner escalará hacia arriba imágenes pequeñas para cumplir la resolución objetivo, lo que puede degradar la calidad. La opción `max_upscale_threshold` permite limitar este comportamiento de escalado.

- **Default**: `null` (permite escalado ilimitado)
- **Cuando se configura**: Filtra buckets de aspecto que requerirían escalado más allá del umbral especificado
- **Rango de valores**: Entre 0 y 1 (p. ej., `0.2` = permitir hasta 20% de escalado)
- **Aplica a**: Selección de buckets de aspecto cuando `crop_aspect` se establece en `closest` o `random`

#### Ejemplos

##### Configuración
```json
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": true,
    "crop_aspect": "random",
    "crop_aspect_buckets": [1.0, 0.5, 2.0],
    "max_upscale_threshold": null
```

##### Resultado
- Todos los buckets de aspecto están disponibles para selección
- Una imagen de 256x256 puede escalarse a 1024x1024 (4x de escala)
- Puede degradar la calidad en imágenes muy pequeñas

##### Configuración
```json
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": true,
    "crop_aspect": "random",
    "crop_aspect_buckets": [1.0, 0.5, 2.0],
    "max_upscale_threshold": 0.2
```

##### Resultado
- Solo están disponibles buckets de aspecto que requieran ≤20% de escalado
- Una imagen de 256x256 intentando escalar a 1024x1024 (4x = 300% de escalado) no tendría buckets disponibles
- Una imagen de 850x850 puede usar todos los buckets ya que 1024/850 ≈ 1.2 (20% de escalado)
- Ayuda a mantener la calidad de entrenamiento excluyendo imágenes mal escaladas

---

### `prepend_instance_prompt`

- Cuando está habilitado, todos los captions incluirán el valor `instance_prompt` al inicio.

### `only_instance_prompt`

- Además de `prepend_instance_prompt`, reemplaza todos los captions del dataset con una sola frase o palabra trigger.

### `repeats`

- Especifica cuántas veces se ven todas las muestras del dataset durante una época. Útil para dar más impacto a datasets pequeños o maximizar el uso de objetos de caché VAE.
- Si tienes un dataset de 1000 imágenes frente a otro de 100 imágenes, probablemente quieras dar al dataset menor un `repeats` de `9` **o más** para llevarlo a 1000 imágenes muestreadas en total.

> ℹ️ Este valor se comporta distinto a la misma opción en los scripts de Kohya, donde un valor de 1 significa sin repeats. **Para SimpleTuner, un valor de 0 significa sin repeats**. Resta uno de tu valor de configuración de Kohya para obtener el equivalente en SimpleTuner, de ahí que un valor de **9** resulte del cálculo `(dataset_length + repeats * dataset_length)`.

#### Entrenamiento multi-GPU y tamaño del dataset

Cuando entrenas con múltiples GPUs, tu dataset debe ser lo bastante grande como para acomodar el **tamaño efectivo de batch**, calculado como:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

Por ejemplo, con 4 GPUs, `train_batch_size=4` y `gradient_accumulation_steps=1`, necesitas al menos **16 muestras** (después de aplicar repeats) en cada bucket de aspecto.

**Importante:** SimpleTuner lanzará un error si tu configuración de dataset produce cero batches utilizables. El mensaje de error mostrará:
- Valores actuales de configuración (batch size, número de GPUs, repeats)
- Qué buckets de aspecto tienen muestras insuficientes
- El mínimo exacto de repeats requerido para cada bucket
- Soluciones sugeridas

##### Sobresuscripción automática del dataset

Para ajustar automáticamente `repeats` cuando tu dataset es más pequeño que el tamaño efectivo de batch, usa el flag `--allow_dataset_oversubscription` (documentado en [OPTIONS.md](OPTIONS.md#--allow_dataset_oversubscription)).

Cuando está habilitado, SimpleTuner:
- Calcula los repeats mínimos necesarios para el entrenamiento
- Aumenta automáticamente `repeats` para cumplir el requisito
- Registra una advertencia mostrando el ajuste
- **Respeta valores de repeats configurados manualmente** - si configuras `repeats` explícitamente en tu configuración de dataset, se omitirá el ajuste automático

Esto es especialmente útil cuando:
- Entrenas datasets pequeños (< 100 imágenes)
- Usas un alto número de GPUs con datasets pequeños
- Experimentas con distintos tamaños de batch sin reconfigurar datasets

**Escenario de ejemplo:**
- Dataset: 25 imágenes
- Configuración: 8 GPUs, `train_batch_size=4`, `gradient_accumulation_steps=1`
- Tamaño efectivo de batch: se requieren 32 muestras
- Sin sobresuscripción: se lanza un error
- Con `--allow_dataset_oversubscription`: repeats ajustado automáticamente a 1 (25 × 2 = 50 muestras)

### `start_epoch` / `start_step`

- Programa cuándo un dataset empieza a muestrear.
- `start_epoch` (default: `1`) controla por número de época; `start_step` (default: `0`) controla por paso de optimizador (después de la acumulación de gradientes). Ambas condiciones deben cumplirse antes de extraer muestras.
- Al menos un dataset debe tener `start_epoch<=1` **y** `start_step<=1`; de lo contrario, el entrenamiento fallará porque no hay datos disponibles al inicio.
- Los datasets que nunca cumplen su condición de inicio (por ejemplo, `start_epoch` más allá de `--num_train_epochs`) se omitirán y se anotarán en la model card.
- Las estimaciones de pasos en la barra de progreso son aproximadas cuando los datasets programados se activan a mitad de ejecución porque la longitud de la época puede aumentar cuando nuevos datos entran en línea.

### `is_regularisation_data`

- También puede escribirse `is_regularization_data`
- Habilita entrenamiento parent-teacher para adaptadores LyCORIS de modo que el objetivo de predicción prefiera el resultado del modelo base para un dataset dado.
  - LoRA estándar no está soportado actualmente.

### `delete_unwanted_images`

- **Valores:** `true` | `false`
- **Descripción:** Cuando está habilitado, las imágenes que fallen los filtros de tamaño o relación de aspecto (p. ej., por debajo de `minimum_image_size` o fuera de `minimum_aspect_ratio`/`maximum_aspect_ratio`) se eliminan permanentemente del directorio del dataset.
- **Advertencia:** Esto es destructivo y no se puede deshacer. Úsalo con cuidado.
- **Default:** Usa el argumento `--delete_unwanted_images` del trainer (default: `false`).

### `delete_problematic_images`

- **Valores:** `true` | `false`
- **Descripción:** Cuando está habilitado, las imágenes que fallen durante la codificación VAE (archivos corruptos, formatos no compatibles, etc.) se eliminan permanentemente del directorio del dataset.
- **Advertencia:** Esto es destructivo y no se puede deshacer. Úsalo con cuidado.
- **Default:** Usa el argumento `--delete_problematic_images` del trainer (default: `false`).

### `slider_strength`

- **Valores:** Cualquier valor float (positivo, negativo o cero)
- **Descripción:** Marca un dataset para entrenamiento de slider LoRA, que aprende "opuestos" contrastivos para crear adaptadores de conceptos controlables.
  - **Valores positivos** (p. ej., `0.5`): "Más del concepto" — ojos más brillantes, sonrisa más marcada, etc.
  - **Valores negativos** (p. ej., `-0.5`): "Menos del concepto" — ojos más apagados, expresión neutra, etc.
  - **Cero u omitido**: Ejemplos neutrales que no empujan el concepto en ninguna dirección.
- **Nota:** Cuando los datasets tienen valores `slider_strength`, SimpleTuner rota los batches en un ciclo fijo: positivo → negativo → neutral. Dentro de cada grupo, las probabilidades estándar del backend siguen aplicando.
- **Ver también:** [SLIDER_LORA.md](SLIDER_LORA.md) para una guía completa sobre cómo configurar entrenamiento de slider LoRA.

### `vae_cache_clear_each_epoch`

- Cuando está habilitado, todos los objetos de caché VAE se eliminan del filesystem al final de cada ciclo de repeats del dataset. Esto puede ser intensivo en recursos para datasets grandes, pero combinado con `crop_style=random` y/o `crop_aspect=random` querrás tenerlo habilitado para asegurar que muestreas un rango completo de recortes de cada imagen.
- De hecho, esta opción está **habilitada por defecto** cuando se usa bucketing aleatorio o recortes aleatorios.

### `vae_cache_disable`

- **Valores:** `true` | `false`
- **Descripción:** Cuando está habilitado (mediante el argumento de línea de comandos `--vae_cache_disable`), esta opción habilita implícitamente el cacheo VAE bajo demanda, pero desactiva la escritura de los embeddings generados a disco. Esto es útil para datasets grandes donde el espacio en disco es una preocupación o escribir no es práctico.
- **Nota:** Este es un argumento a nivel de trainer, no una opción de configuración por dataset, pero afecta cómo el dataloader interactúa con la caché VAE.

### `skip_file_discovery`

- Probablemente no quieras configurar esto nunca; solo es útil para datasets muy grandes.
- Este parámetro acepta una lista separada por comas o espacios, p. ej., `vae metadata aspect text` para omitir el descubrimiento de archivos en una o más etapas de la configuración del loader.
- Es equivalente a la opción de línea de comandos `--skip_file_discovery`
- Es útil si tienes datasets que no necesitas que el trainer escanee en cada arranque, p. ej., sus latentes/embeds ya están completamente cacheados. Esto permite un inicio más rápido y reanudar el entrenamiento.

### `preserve_data_backend_cache`

- Probablemente no quieras configurar esto nunca; solo es útil para datasets muy grandes en AWS.
- Al igual que `skip_file_discovery`, esta opción puede configurarse para evitar escaneos del filesystem innecesarios, largos y costosos al inicio.
- Toma un valor booleano, y si se establece en `true`, el archivo de caché de lista de filesystem generado no se eliminará al iniciar.
- Es útil para sistemas de almacenamiento muy grandes y lentos como S3 o discos duros locales SMR con tiempos de respuesta extremadamente lentos.
- Además, en S3, el listado del backend puede acumular costos y debería evitarse.

> ⚠️ **Desafortunadamente, esto no puede configurarse si los datos se están cambiando activamente.** El trainer no verá ningún dato nuevo que se agregue al conjunto; tendrá que hacer otro escaneo completo.

### `hash_filenames`

- Los nombres de archivo de las entradas de caché VAE siempre se hashean. Esto no es configurable por el usuario y garantiza que datasets con nombres de archivo muy largos puedan usarse sin problemas de longitud de ruta. Cualquier configuración `hash_filenames` en tu configuración será ignorada.

## Filtrado de captions

### `caption_filter_list`

- **Solo para datasets de text embeds.** Puede ser una lista JSON, una ruta a un archivo txt o una ruta a un documento JSON. Las cadenas de filtro pueden ser términos simples para eliminar de todos los captions, o pueden ser expresiones regulares. Además, se pueden usar entradas estilo sed `s/buscar/reemplazar/` para _reemplazar_ cadenas en el caption en lugar de simplemente eliminarlas.

#### Ejemplo de lista de filtros

Un ejemplo completo se encuentra [aquí](/config/caption_filter_list.txt.example). Contiene cadenas comunes repetitivas y negativas que podrían devolver BLIP (variedad común), LLaVA y CogVLM.

Este es un ejemplo abreviado, que se explicará a continuación:

```
arafed
this .* has a
^this is the beginning of the string
s/this/will be found and replaced/
```

En orden, las líneas se comportan así:

- `arafed ` (con un espacio al final) se eliminará de cualquier caption en el que se encuentre. Incluir un espacio al final hace que el caption se vea mejor, ya que no quedarán dobles espacios. Esto es innecesario, pero se ve bien.
- `this .* has a` es una expresión regular que eliminará cualquier cosa que contenga "this ... has a", incluyendo cualquier texto entre esas dos cadenas; `.*` es una expresión regular que significa "todo lo que encontremos" hasta que encuentre la cadena "has a", momento en el que deja de coincidir.
- `^this is the beginning of the string` eliminará la frase "this is the beginning of the string" de cualquier caption, pero solo cuando aparezca al inicio del caption.
- `s/this/will be found and replaced/` hará que la primera instancia del término "this" en cualquier caption sea reemplazada por "will be found and replaced".

> ❗Usa [regex 101](https://regex101.com) para ayuda al depurar y probar expresiones regulares.

# Técnicas avanzadas

## Configuración avanzada de ejemplo

```json
[
  {
    "id": "something-special-to-remember-by",
    "type": "local",
    "instance_data_dir": "/path/to/data/tree",
    "crop": false,
    "crop_style": "random|center|corner|face",
    "crop_aspect": "square|preserve|closest|random",
    "crop_aspect_buckets": [0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    "resolution": 1.0,
    "resolution_type": "area|pixel",
    "minimum_image_size": 1.0,
    "prepend_instance_prompt": false,
    "instance_prompt": "something to label every image",
    "only_instance_prompt": false,
    "caption_strategy": "filename|instanceprompt|parquet|textfile",
    "cache_dir_vae": "/path/to/vaecache",
    "vae_cache_clear_each_epoch": true,
    "probability": 1.0,
    "repeats": 0,
    "start_epoch": 1,
    "start_step": 0,
    "text_embeds": "alt-embed-cache",
    "image_embeds": "vae-embeds-example",
    "conditioning_image_embeds": "conditioning-embeds-example"
  },
  {
    "id": "another-special-name-for-another-backend",
    "type": "aws",
    "aws_bucket_name": "something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir_vae": "s3prefix/for/vaecache",
    "vae_cache_clear_each_epoch": true,
    "repeats": 0
  },
  {
      "id": "vae-embeds-example",
      "type": "local",
      "dataset_type": "image_embeds",
      "disabled": false,
  },
  {
      "id": "conditioning-embeds-example",
      "type": "local",
      "dataset_type": "conditioning_image_embeds",
      "disabled": false
  },
  {
    "id": "an example backend for text embeds.",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "aws",
    "aws_bucket_name": "textembeds-something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir": ""
  },
  {
    "id": "alt-embed-cache",
    "dataset_type": "text_embeds",
    "default": false,
    "type": "local",
    "cache_dir": "/path/to/textembed_cache"
  }
]
```

## Entrenar directamente desde una lista de URLs en CSV

**Nota: Tu CSV debe contener los captions de tus imágenes.**

> ⚠️ Esta es una función avanzada **y** experimental, y puedes encontrarte con problemas. Si ocurre, por favor abre un [issue](https://github.com/bghira/simpletuner/issues).

En lugar de descargar manualmente tus datos desde una lista de URLs, quizá quieras conectarlos directamente al trainer.

**Nota:** Siempre es mejor descargar manualmente los datos de imagen. Otra estrategia para ahorrar espacio local en disco podría ser intentar [usar almacenamiento en la nube con cachés de encoder locales](#local-cache-with-cloud-dataset).

### Ventajas

- No es necesario descargar los datos directamente
- Puedes usar el toolkit de captions de SimpleTuner para captionar directamente la lista de URLs
- Ahorra espacio en disco, ya que solo se almacenan los image embeds (si aplica) y text embeds

### Desventajas

- Requiere un escaneo de buckets de aspecto costoso y potencialmente lento donde cada imagen se descarga y se recolectan sus metadatos
- Las imágenes descargadas se cachean en disco, lo que puede crecer muchísimo. Esto es un área de mejora, ya que la gestión de caché en esta versión es muy básica, write-only/delete-never
- Si tu dataset tiene muchas URLs inválidas, esto puede desperdiciar tiempo al reanudar ya que, actualmente, las muestras malas **nunca** se eliminan de la lista de URLs
  - **Sugerencia:** Ejecuta una tarea de validación de URLs previamente y elimina cualquier muestra inválida.

### Configuración

Claves requeridas:

- `type: "csv"`
- `csv_caption_column`
- `csv_cache_dir`
- `caption_strategy: "csv"`

```json
[
    {
        "id": "csvtest",
        "type": "csv",
        "csv_caption_column": "caption",
        "csv_file": "/Volumes/ml/dataset/test_list.csv",
        "csv_cache_dir": "/Volumes/ml/cache/csv/test",
        "cache_dir_vae": "/Volumes/ml/cache/vae/sdxl",
        "caption_strategy": "csv",
        "image_embeds": "image-embeds",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel",
        "minimum_image_size": 0,
        "disabled": false,
        "skip_file_discovery": "",
        "preserve_data_backend_cache": false
    },
    {
      "id": "image-embeds",
      "type": "local"
    },
    {
        "id": "text-embeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/Volumes/ml/cache/text/sdxl",
        "disabled": false,
        "preserve_data_backend_cache": false,
        "skip_file_discovery": "",
        "write_batch_size": 128
    }
]
```

## Estrategia de captions parquet / datasets JSON Lines

> ⚠️ Esta es una función avanzada y no será necesaria para la mayoría de usuarios.

Cuando entrenas un modelo con un dataset muy grande que supera cientos de miles o millones de imágenes, lo más rápido es almacenar tus metadatos dentro de una base de datos parquet en lugar de archivos txt, especialmente cuando tus datos de entrenamiento están en un bucket S3.

Usar la estrategia de captions parquet te permite nombrar todos tus archivos por su valor `id`, y cambiar su columna de caption mediante un valor de configuración en lugar de actualizar muchos archivos de texto o renombrar archivos para actualizar sus captions.

Aquí tienes un ejemplo de configuración del dataloader que usa los captions y datos del dataset [photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket):

```json
{
  "id": "photo-concept-bucket",
  "type": "local",
  "instance_data_dir": "/models/training/datasets/photo-concept-bucket-downloads",
  "caption_strategy": "parquet",
  "metadata_backend": "parquet",
  "parquet": {
    "path": "photo-concept-bucket.parquet",
    "filename_column": "id",
    "caption_column": "cogvlm_caption",
    "fallback_caption_column": "tags",
    "width_column": "width",
    "height_column": "height",
    "identifier_includes_extension": false
  },
  "resolution": 1.0,
  "minimum_image_size": 0.75,
  "maximum_image_size": 2.0,
  "target_downsample_size": 1.5,
  "prepend_instance_prompt": false,
  "instance_prompt": null,
  "only_instance_prompt": false,
  "disable": false,
  "cache_dir_vae": "/models/training/vae_cache/photo-concept-bucket",
  "probability": 1.0,
  "skip_file_discovery": "",
  "preserve_data_backend_cache": false,
  "vae_cache_clear_each_epoch": true,
  "repeats": 1,
  "crop": true,
  "crop_aspect": "closest",
  "crop_style": "random",
  "crop_aspect_buckets": [1.0, 0.75, 1.23],
  "resolution_type": "area"
}
```

En esta configuración:

- `caption_strategy` se configura como `parquet`.
- `metadata_backend` se configura como `parquet`.
- Se debe definir una nueva sección, `parquet`:
  - `path` es la ruta al archivo parquet o JSONL.
  - `filename_column` es el nombre de la columna en la tabla que contiene los nombres de archivo. En este caso, usamos la columna numérica `id` (recomendado).
  - `caption_column` es el nombre de la columna en la tabla que contiene los captions. En este caso, usamos la columna `cogvlm_caption`. Para datasets LAION, sería el campo TEXT.
  - `width_column` y `height_column` pueden ser una columna que contenga cadenas, int o incluso un tipo Series de una sola entrada, midiendo las dimensiones reales de la imagen. Esto mejora notablemente el tiempo de preparación del dataset, ya que no necesitamos acceder a las imágenes reales para descubrir esta información.
  - `fallback_caption_column` es un nombre opcional de una columna en la tabla que contiene captions de respaldo. Se usan si el campo de caption principal está vacío. En este caso, usamos la columna `tags`.
  - `identifier_includes_extension` debe establecerse en `true` cuando tu columna de nombre de archivo contiene la extensión de imagen. De lo contrario, se asumirá la extensión `.png`. Se recomienda incluir extensiones de archivo en tu columna de nombres.

> ⚠️ La capacidad de soporte de parquet se limita a leer captions. Debes poblar por separado una fuente de datos con tus muestras de imagen usando "{id}.png" como nombre de archivo. Consulta scripts en el directorio [scripts/toolkit/datasets](scripts/toolkit/datasets) para ideas.

Como en otras configuraciones de dataloader:

- `prepend_instance_prompt` y `instance_prompt` se comportan como de costumbre.
- Actualizar el caption de una muestra entre ejecuciones de entrenamiento cacheará el nuevo embed, pero no eliminará la unidad vieja (huérfana).
- Cuando una imagen no existe en un dataset, su nombre de archivo se usará como caption y se emitirá un error.

## Caché local con dataset en la nube

Para maximizar el uso del costoso almacenamiento local NVMe, quizá quieras almacenar solo los archivos de imagen (png, jpg) en un bucket S3 y usar el almacenamiento local para cachear tus feature maps extraídos del/los encoder(es) de texto y del VAE (si aplica).

En este ejemplo de configuración:

- Los datos de imagen se almacenan en un bucket compatible con S3
- Los datos del VAE se almacenan en /local/path/to/cache/vae
- Los text embeds se almacenan en /local/path/to/cache/textencoder

> ⚠️ Recuerda configurar las otras opciones del dataset, como `resolution` y `crop`

```json
[
    {
        "id": "data",
        "type": "aws",
        "aws_bucket_name": "text-vae-embeds",
        "aws_endpoint_url": "https://storage.provider.example",
        "aws_access_key_id": "exampleAccessKey",
        "aws_secret_access_key": "exampleSecretKey",
        "aws_region_name": null,
        "cache_dir_vae": "/local/path/to/cache/vae/",
        "caption_strategy": "parquet",
        "metadata_backend": "parquet",
        "parquet": {
            "path": "train.parquet",
            "caption_column": "caption",
            "filename_column": "filename",
            "width_column": "width",
            "height_column": "height",
            "identifier_includes_extension": true
        },
        "preserve_data_backend_cache": false,
        "image_embeds": "vae-embed-storage"
    },
    {
        "id": "vae-embed-storage",
        "type": "local",
        "dataset_type": "image_embeds"
    },
    {
        "id": "text-embed-storage",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/local/path/to/cache/textencoder/",
        "write_batch_size": 128
    }
]
```

**Nota:** El dataset `image_embeds` no tiene opciones para configurar rutas de datos. Esas se configuran mediante `cache_dir_vae` en el backend de imagen.

### Soporte de Hugging Face Datasets

SimpleTuner ahora soporta cargar datasets directamente desde Hugging Face Hub sin descargar todo el dataset localmente. Esta función experimental es ideal para:

- Datasets a gran escala alojados en Hugging Face
- Datasets con metadatos y evaluaciones de calidad integrados
- Experimentación rápida sin requerimientos de almacenamiento local

Para documentación detallada sobre esta función, consulta [este documento](HUGGINGFACE_DATASETS.md).

Para un ejemplo básico de cómo usar un dataset de Hugging Face, configura `"type": "huggingface"` en tu configuración del dataloader:

```json
{
  "id": "my-hf-dataset",
  "type": "huggingface",
  "dataset_name": "username/dataset-name",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "caption",
  "image_column": "image"
}
```

## Mapeo personalizado de relación de aspecto a resolución

Cuando SimpleTuner inicia por primera vez, genera listas de mapeo de aspecto específicas de resolución que vinculan un valor decimal de relación de aspecto con su tamaño de píxel objetivo.

Es posible crear un mapeo personalizado que fuerce al trainer a ajustarse a tu resolución objetivo en lugar de sus propios cálculos. Esta funcionalidad se ofrece bajo tu propio riesgo, ya que obviamente puede causar gran daño si se configura incorrectamente.

Para crear el mapeo personalizado:

- Crea un archivo que siga el ejemplo (abajo)
- Nombra el archivo usando el formato `aspect_ratio_map-{resolution}.json`
  - Para un valor de configuración `resolution=1.0` / `resolution_type=area`, el nombre del mapeo será `aspect_resolution_map-1.0.json`
- Coloca este archivo en la ubicación especificada como `--output_dir`
  - Esta es la misma ubicación donde se encontrarán tus checkpoints e imágenes de validación.
- No se requieren flags ni opciones adicionales de configuración. Se descubrirá y usará automáticamente, siempre que el nombre y la ubicación sean correctos.

### Configuración de ejemplo del mapeo

Este es un ejemplo de mapeo de relación de aspecto generado por SimpleTuner. No necesitas configurarlo manualmente, ya que el trainer creará uno automáticamente. Sin embargo, para control total sobre las resoluciones resultantes, estos mapeos se proporcionan como punto de partida para su modificación.

- El dataset tenía más de 1 millón de imágenes
- El `resolution` del dataloader se configuró en `1.0`
- El `resolution_type` del dataloader se configuró en `area`

Esta es la configuración más común y la lista de buckets de aspecto entrenables para un modelo de 1 megapíxel.

```json
{
    "0.07": [320, 4544],    "0.38": [640, 1664],    "0.88": [960, 1088],    "1.92": [1472, 768],    "3.11": [1792, 576],    "5.71": [2560, 448],
    "0.08": [320, 3968],    "0.4": [640, 1600],     "0.89": [1024, 1152],   "2.09": [1472, 704],    "3.22": [1856, 576],    "6.83": [2624, 384],
    "0.1": [320, 3328],     "0.41": [704, 1728],    "0.94": [1024, 1088],   "2.18": [1536, 704],    "3.33": [1920, 576],    "7.0": [2688, 384],
    "0.11": [384, 3520],    "0.42": [704, 1664],    "1.06": [1088, 1024],   "2.27": [1600, 704],    "3.44": [1984, 576],    "8.0": [3072, 384],
    "0.12": [384, 3200],    "0.44": [704, 1600],    "1.12": [1152, 1024],   "2.5": [1600, 640],     "3.88": [1984, 512],
    "0.14": [384, 2688],    "0.46": [704, 1536],    "1.13": [1088, 960],    "2.6": [1664, 640],     "4.0": [2048, 512],
    "0.15": [448, 3008],    "0.48": [704, 1472],    "1.2": [1152, 960],     "2.7": [1728, 640],     "4.12": [2112, 512],
    "0.16": [448, 2816],    "0.5": [768, 1536],     "1.36": [1216, 896],    "2.8": [1792, 640],     "4.25": [2176, 512],
    "0.19": [448, 2304],    "0.52": [768, 1472],    "1.46": [1216, 832],    "3.11": [1792, 576],    "4.38": [2240, 512],
    "0.24": [512, 2112],    "0.55": [768, 1408],    "1.54": [1280, 832],    "3.22": [1856, 576],    "5.0": [2240, 448],
    "0.26": [512, 1984],    "0.59": [832, 1408],    "1.83": [1408, 768],    "3.33": [1920, 576],    "5.14": [2304, 448],
    "0.29": [576, 1984],    "0.62": [832, 1344],    "1.92": [1472, 768],    "3.44": [1984, 576],    "5.71": [2560, 448],
    "0.31": [576, 1856],    "0.65": [832, 1280],    "2.09": [1472, 704],    "3.88": [1984, 512],    "6.83": [2624, 384],
    "0.34": [640, 1856],    "0.68": [832, 1216],    "2.18": [1536, 704],    "4.0": [2048, 512],     "7.0": [2688, 384],
    "0.38": [640, 1664],    "0.74": [896, 1216],    "2.27": [1600, 704],    "4.12": [2112, 512],    "8.0": [3072, 384],
    "0.4": [640, 1600],     "0.83": [960, 1152],    "2.5": [1600, 640],     "4.25": [2176, 512],
    "0.41": [704, 1728],    "0.88": [960, 1088],    "2.6": [1664, 640],     "4.38": [2240, 512],
    "0.42": [704, 1664],    "0.89": [1024, 1152],   "2.7": [1728, 640],     "5.0": [2240, 448],
    "0.44": [704, 1600],    "0.94": [1024, 1088],   "2.8": [1792, 640],     "5.14": [2304, 448]
}
```

Para modelos de Stable Diffusion 1.5 / 2.0-base (512px), el siguiente mapeo funciona:

```json
{
    "1.3": [832, 640], "1.0": [768, 768], "2.0": [1024, 512],
    "0.64": [576, 896], "0.77": [640, 832], "0.79": [704, 896],
    "0.53": [576, 1088], "1.18": [832, 704], "0.85": [704, 832],
    "0.56": [576, 1024], "0.92": [704, 768], "1.78": [1024, 576],
    "1.56": [896, 576], "0.67": [640, 960], "1.67": [960, 576],
    "0.5": [512, 1024], "1.09": [768, 704], "1.08": [832, 768],
    "0.44": [512, 1152], "0.71": [640, 896], "1.4": [896, 640],
    "0.39": [448, 1152], "2.25": [1152, 512], "2.57": [1152, 448],
    "0.4": [512, 1280], "3.5": [1344, 384], "2.12": [1088, 512],
    "0.3": [448, 1472], "2.71": [1216, 448], "8.25": [2112, 256],
    "0.29": [384, 1344], "2.86": [1280, 448], "6.2": [1984, 320],
    "0.6": [576, 960]
}
```
