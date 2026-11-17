# Dreambooth (entrenamiento de un solo sujeto)

## Antecedentes

El término Dreambooth se refiere a una técnica desarrollada por Google para inyectar sujetos afinándolos en un modelo usando un pequeño conjunto de imágenes de alta calidad ([paper](https://dreambooth.github.io)).

En el contexto del fine-tuning, Dreambooth agrega nuevas técnicas para ayudar a prevenir el colapso del modelo debido a, p. ej., overfitting o artefactos.

### Imágenes de regularización

Las imágenes de regularización suelen ser generadas por el modelo que estás entrenando, usando un token que se parece a tu clase.

No **tienen** que ser imágenes sintéticas generadas por el modelo, pero esto posiblemente tenga mejor rendimiento que usar datos reales (p. ej., fotos de personas reales).

Ejemplo: Si estás entrenando imágenes de un sujeto masculino, tus datos de regularización serían fotos o muestras sintéticas generadas de sujetos masculinos aleatorios.

> 🟢 Las imágenes de regularización pueden configurarse como un dataset separado, lo que permite mezclarlas de forma uniforme con tus datos de entrenamiento.

### Entrenamiento con token raro

Un concepto de valor dudoso del paper original era hacer una búsqueda inversa en el vocabulario del tokenizer del modelo para encontrar una cadena "rara" con muy poco entrenamiento asociado.

Desde entonces, la idea ha evolucionado y se ha debatido, con un bando opuesto decidiendo entrenar contra el nombre de una celebridad suficientemente similar, ya que esto requiere menos cómputo.

> 🟡 El entrenamiento con token raro está soportado en SimpleTuner, pero no hay una herramienta disponible para ayudarte a encontrar uno.

### Pérdida de preservación del prior

El modelo contiene algo llamado "prior" que, en teoría, podría preservarse durante el entrenamiento de Dreambooth. Sin embargo, en experimentos con Stable Diffusion no pareció ayudar: el modelo simplemente sobreajusta su propio conocimiento.

> 🟢 ([#1031](https://github.com/bghira/SimpleTuner/issues/1031)) La pérdida de preservación del prior está soportada en SimpleTuner cuando se entrenan adaptadores LyCORIS estableciendo `is_regularisation_data` en ese dataset.

### Pérdida enmascarada

Las máscaras de imagen pueden definirse en pares con los datos de imagen. Las partes oscuras de la máscara harán que los cálculos de pérdida ignoren esas partes de la imagen.

Existe un [script](/scripts/datasets/masked_loss/generate_dataset_masks.py) para generar estas máscaras, dado un input_dir y output_dir:

```bash
python generate_dataset_masks.py --input_dir /images/input \
                      --output_dir /images/output \
                      --text_input "person"
```

Sin embargo, esto no tiene funcionalidades avanzadas como el difuminado de padding de máscara.

Al definir tu dataset de máscaras:

- Cada imagen debe tener una máscara. Usa una imagen completamente blanca si no quieres enmascarar.
- Configura `dataset_type=conditioning` en tu carpeta de datos de condicionamiento (máscara)
- Configura `conditioning_type=mask` en tu dataset de máscaras
- Configura `conditioning_data=` con el `id` de tu dataset de condicionamiento en tu dataset de imágenes

```json
[
    {
        "id": "dreambooth-data",
        "type": "local",
        "dataset_type": "image",
        "conditioning_data": "dreambooth-conditioning",
        "instance_data_dir": "/training/datasets/test_datasets/dreambooth",
        "cache_dir_vae": "/training/cache/vae/sdxl/dreambooth-data",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an dreambooth",
        "metadata_backend": "discovery",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution_type": "pixel_area"
    },
    {
        "id": "dreambooth-conditioning",
        "type": "local",
        "dataset_type": "conditioning",
        "instance_data_dir": "/training/datasets/test_datasets/dreambooth_mask",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution_type": "pixel_area",
        "conditioning_type": "mask"
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "/training/cache/text/sdxl-base/masked_loss"
    }
]
```

## Setup

Seguir el [tutorial](TUTORIAL.md) es necesario antes de continuar con la configuración específica de Dreambooth.

Para ajuste de DeepFloyd, se recomienda visitar [esta página](DEEPFLOYD.md) para tips específicos relacionados con la configuración de ese modelo.

### Entrenamiento con modelos cuantizados (solo LoRA/LyCORIS)

Probado en sistemas Apple y NVIDIA, Hugging Face Optimum-Quanto puede usarse para reducir la precisión y los requisitos de VRAM.

Dentro de tu venv de SimpleTuner:

```bash
pip install optimum-quanto
```

Los niveles de precisión disponibles dependen de tu hardware y sus capacidades.

- int2-quanto, int4-quanto, **int8-quanto** (recomendado)
- fp8-quanto, fp8-torchao (solo para CUDA >= 8.9, p. ej., 4090 o H100)
- nf4-bnb (requerido para usuarios con baja VRAM)

Dentro de tu config.json, los siguientes valores deberían modificarse o añadirse:
```json
{
    "base_model_precision": "int8-quanto",
    "text_encoder_1_precision": "no_change",
    "text_encoder_2_precision": "no_change",
    "text_encoder_3_precision": "no_change"
}
```

Dentro de nuestro dataloader config `multidatabackend-dreambooth.json`, se verá algo así:

```json
[
    {
        "id": "subjectname-data-512px",
        "type": "local",
        "instance_data_dir": "/training/datasets/subjectname",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "subjectname",
        "cache_dir_vae": "/training/vae_cache/subjectname",
        "repeats": 100,
        "crop": false,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "minimum_image_size": 192
    },
    {
        "id": "subjectname-data-1024px",
        "type": "local",
        "instance_data_dir": "/training/datasets/subjectname",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "subjectname",
        "cache_dir_vae": "/training/vae_cache/subjectname-1024px",
        "repeats": 100,
        "crop": false,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "minimum_image_size": 768
    },
    {
        "id": "regularisation-data",
        "type": "local",
        "instance_data_dir": "/training/datasets/regularisation",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "a picture of a man",
        "cache_dir_vae": "/training/vae_cache/regularisation",
        "repeats": 0,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "minimum_image_size": 192,
        "is_regularisation_data": true
    },
    {
        "id": "regularisation-data-1024px",
        "type": "local",
        "instance_data_dir": "/training/datasets/regularisation",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "a picture of a man",
        "cache_dir_vae": "/training/vae_cache/regularisation-1024px",
        "repeats": 0,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "minimum_image_size": 768,
        "is_regularisation_data": true
    },
    {
        "id": "textembeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/training/text_cache/sdxl_base"
    }
]
```

Algunos valores clave se ajustaron para facilitar el entrenamiento de un solo sujeto:

- Ahora tenemos dos datasets configurados dos veces, para un total de cuatro datasets. Los datos de regularización son opcionales y el entrenamiento puede funcionar mejor sin ellos. Puedes eliminar ese dataset de la lista si lo deseas.
- La resolución se establece en 512px y 1024px con bucketing mixto, lo que puede ayudar a mejorar la velocidad de entrenamiento y la convergencia.
- El tamaño mínimo de imagen se establece en 192px o 768px, lo que permitirá escalar hacia arriba algunas imágenes pequeñas, lo cual puede ser necesario para datasets con unas pocas imágenes importantes pero de baja resolución.
- `caption_strategy` ahora es `instanceprompt`, lo que significa que usaremos el valor `instance_prompt` para cada imagen en el dataset como su caption.
  - **Nota:** Usar el instance prompt es el método tradicional de entrenamiento Dreambooth, pero captions cortas pueden funcionar mejor. Si descubres que el modelo no generaliza, quizá valga la pena intentar usar captions.

### Consideraciones del dataset de regularización

Para un dataset de regularización:

- Establece `repeats` muy alto en tu sujeto Dreambooth para que el conteo de imágenes en los datos Dreambooth se multiplique `repeats` veces y supere el conteo de imágenes de tu conjunto de regularización
  - Si tu conjunto de regularización tiene 1000 imágenes y tienes 10 imágenes en tu conjunto de entrenamiento, querrás un valor de repeats de al menos 100 para obtener resultados rápidos
- `minimum_image_size` se ha incrementado para asegurar que no introducimos demasiados artefactos de baja calidad
- De manera similar, usar captions más descriptivos puede ayudar a evitar el olvido. Cambiar de `instanceprompt` a `textfile` u otras estrategias requerirá crear archivos `.txt` para cada imagen.
- Cuando `is_regularisation_data` (o 🇺🇸 `is_regularization_data` con z, para usuarios estadounidenses) se establece, los datos de este conjunto se alimentarán al modelo base para obtener una predicción que pueda usarse como objetivo de pérdida para el modelo LyCORIS estudiante.
  - Nota: actualmente esto solo funciona con un adaptador LyCORIS.

## Seleccionar un instance prompt

Como se mencionó antes, el enfoque original de Dreambooth era la selección de tokens raros para entrenar.

Alternativamente, se podría usar el nombre real del sujeto o el de una celebridad "suficientemente similar".

Después de varios experimentos de entrenamiento, parece que una celebridad "suficientemente similar" es la mejor opción, especialmente si al pedir el nombre real de la persona el resultado se ve disímil.

# Scheduled Sampling (Rollout)

Al entrenar con datasets pequeños como en Dreambooth, los modelos pueden sobreajustarse rápidamente al ruido "perfecto" añadido durante el entrenamiento. Esto lleva a **sesgo de exposición**: el modelo aprende a denoising entradas perfectas pero falla cuando se enfrenta a sus propias salidas ligeramente imperfectas durante la inferencia.

**Scheduled Sampling (Rollout)** aborda esto permitiendo ocasionalmente que el modelo genere sus propios latentes ruidosos por unos pasos durante el bucle de entrenamiento. En lugar de entrenar con ruido gaussiano puro + señal, entrena con muestras "rollout" que contienen los errores previos del modelo. Esto enseña al modelo a corregirse, resultando en una generación de sujetos más robusta y estable.

> 🟢 Esta función es experimental pero muy recomendada para datasets pequeños donde el sobreajuste o el "frying" es común.
> ⚠️ Habilitar rollout aumenta los requisitos de cómputo, ya que el modelo debe realizar pasos de inferencia extra durante el bucle de entrenamiento.

Para habilitarlo, agrega estas claves a tu `config.json`:

```json
{
  "scheduled_sampling_max_step_offset": 10,
  "scheduled_sampling_probability": 1.0,
  "scheduled_sampling_ramp_steps": 1000,
  "scheduled_sampling_sampler": "unipc"
}
```

*   `scheduled_sampling_max_step_offset`: Cuántos pasos generar. Un valor pequeño (p. ej., 5-10) suele ser suficiente.
*   `scheduled_sampling_probability`: Con qué frecuencia aplicar esta técnica (0.0 a 1.0).
*   `scheduled_sampling_ramp_steps`: Incrementa la probabilidad durante los primeros N pasos para evitar desestabilizar el entrenamiento temprano.

# Media móvil exponencial (EMA)

Un segundo modelo puede entrenarse en paralelo a tu checkpoint, casi gratis: solo se consume memoria del sistema (por defecto), no más VRAM.

Aplicar `use_ema=true` en tu archivo de configuración habilitará esta función.

# Seguimiento de puntuaciones CLIP

Si deseas habilitar evaluaciones para puntuar el rendimiento del modelo, consulta [este documento](evaluation/CLIP_SCORES.md) para información sobre configuración e interpretación de puntuaciones CLIP.

# Pérdida de evaluación estable

Si deseas usar pérdida MSE estable para puntuar el rendimiento del modelo, consulta [este documento](evaluation/EVAL_LOSS.md) para información sobre configuración e interpretación de la pérdida de evaluación.

# Previsualizaciones de validación

SimpleTuner admite streaming de previsualizaciones intermedias de validación durante la generación usando modelos Tiny AutoEncoder. Esta función te permite ver tus imágenes de validación generadas paso a paso en tiempo real vía callbacks de webhook, en lugar de esperar a la generación completa.

## Habilitar previsualizaciones de validación

Para habilitar previsualizaciones de validación, añade lo siguiente a tu `config.json`:

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

## Requisitos

- Familia de modelos con soporte de Tiny AutoEncoder (Flux, SDXL, SD3, etc.)
- Configuración de webhook para recibir las imágenes de preview
- La validación debe estar habilitada (`validation_disable` no debe establecerse en true)

## Opciones de configuración

- `--validation_preview`: Habilita/deshabilita la función de preview (default: false)
- `--validation_preview_steps`: Controla con qué frecuencia se decodifican previsualizaciones durante el muestreo (default: 1)
  - Establece 1 para recibir un preview en cada paso de muestreo
  - Establece valores más altos (p. ej., 3 o 5) para reducir el overhead del decodificado de Tiny AutoEncoder

## Ejemplo

Con `validation_num_inference_steps=20` y `validation_preview_steps=5`, recibirás previsualizaciones en los pasos 5, 10, 15 y 20 durante cada generación de validación.

# Ajuste de refiner

Si eres fan del refiner de SDXL, puede que descubras que hace que tus generaciones "arruinen" los resultados de tu modelo Dreamboothed.

SimpleTuner soporta entrenar el refiner de SDXL usando LoRA y full rank.

Esto requiere un par de consideraciones:
- Las imágenes deben ser exclusivamente de alta calidad
- Los text embeds no pueden compartirse con los del modelo base
- Los VAE embeds **sí** pueden compartirse con los del modelo base

Necesitarás actualizar `cache_dir` en tu configuración de dataloader, `multidatabackend.json`:

```json
[
    {
        "id": "textembeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/training/text_cache/sdxl_refiner"
    }
]
```

Si deseas apuntar a una puntuación estética específica con tus datos, puedes añadir esto a `config/config.json`:

```bash
"--data_aesthetic_score": 5.6,
```

Actualiza **5.6** al score que quieras targetear. El default es **7.0**.

> ⚠️ Al entrenar el refiner de SDXL, tus prompts de validación serán ignorados. En su lugar, se refinarán imágenes aleatorias de tus datasets.
