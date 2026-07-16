# DeepFloyd IF

> 🤷🏽‍♂️ Entrenar DeepFloyd requiere al menos 24G de VRAM para un LoRA. Esta guía se centra en el modelo base de 400M parámetros, aunque la variante XL de 4.3B puede entrenarse con las mismas pautas.

## Antecedentes

En la primavera de 2023, StabilityAI lanzó un modelo de difusión de píxeles en cascada llamado DeepFloyd.
![](https://tripleback.net/public/deepfloyd.png)

Comparado brevemente con Stable Diffusion XL:
- Encoder de texto
  - SDXL usa dos encoders CLIP, "OpenCLIP G/14" y "OpenAI CLIP-L/14"
  - DeepFloyd usa un único transformer auto-supervisado, T5 XXL de Google
- Cantidad de parámetros
  - DeepFloyd viene en múltiples densidades: 400M, 900M y 4.3B parámetros. Cada unidad más grande es sucesivamente más costosa de entrenar.
  - SDXL tiene solo uno, ~3B parámetros.
  - El encoder de texto de DeepFloyd tiene 11B parámetros por sí solo, haciendo que la configuración más grande sea de aproximadamente 15.3B parámetros.
- Cantidad de modelos
  - DeepFloyd funciona en **tres** etapas: 64px -> 256px -> 1024px
    - Cada etapa completa por completo su objetivo de denoising
  - SDXL funciona en **dos** etapas, incluyendo su refiner, de 1024px -> 1024px
    - Cada etapa solo completa parcialmente su objetivo de denoising
- Diseño
  - Los tres modelos de DeepFloyd aumentan la resolución y los detalles finos
  - Los dos modelos de SDXL gestionan los detalles finos y la composición

Para ambos modelos, la primera etapa define la mayor parte de la composición de la imagen (dónde aparecen objetos grandes / sombras).

## Evaluación del modelo

Esto es lo que puedes esperar al usar DeepFloyd para entrenamiento o inferencia.

### Estética

Comparado con SDXL o Stable Diffusion 1.x/2.x, la estética de DeepFloyd se ubica en algún punto entre Stable Diffusion 2.x y SDXL.


### Desventajas

No es un modelo popular, por varias razones:

- El requisito de VRAM para inferencia es más pesado que el de otros modelos
- Los requisitos de VRAM para entrenamiento superan con creces a otros modelos
  - Un ajuste completo de u-net requiere más de 48G de VRAM
  - LoRA en rank-32, batch-4 necesita ~24G de VRAM
  - Los objetos de caché de text embeds son ENORMES (varios megabytes cada uno, vs cientos de kilobytes para los embeds CLIP duales de SDXL)
  - Los objetos de caché de text embeds son LENTOS DE CREAR, alrededor de 9-10 por segundo en un A6000 no-Ada.
- La estética por defecto es peor que en otros modelos (como intentar entrenar SD 1.5 vanilla)
- Hay **tres** modelos que ajustar o cargar en tu sistema durante la inferencia (cuatro si contamos el encoder de texto)
- Las promesas de StabilityAI no coincidieron con la realidad de uso del modelo (sobreprometido)
- La licencia de DeepFloyd-IF es restrictiva frente al uso comercial.
  - Esto no afectó a los pesos de NovelAI, que de hecho se filtraron de forma ilícita. La naturaleza de licencia comercial parece una excusa conveniente, considerando los otros problemas más grandes.

### Ventajas

Sin embargo, DeepFloyd tiene ventajas que a menudo se pasan por alto:

- En inferencia, el encoder de texto T5 demuestra una fuerte comprensión del mundo
- Puede entrenarse de forma nativa con captions muy largos
- La primera etapa es de área ~64x64 píxeles, y puede entrenarse con resoluciones multi-aspecto
  - La naturaleza de baja resolución de los datos de entrenamiento significa que DeepFloyd fue _el único modelo_ capaz de entrenar con _TODO_ LAION-A (pocas imágenes están por debajo de 64x64 en LAION)
- Cada etapa puede ajustarse de forma independiente, enfocándose en objetivos distintos
  - La primera etapa puede ajustarse enfocándose en cualidades compositivas, y las etapas posteriores se ajustan para mejores detalles al reescalar
- Entrena muy rápido a pesar de su mayor huella de memoria
  - Entrena más rápido en términos de throughput: se observa una alta tasa de muestras por hora en el ajuste de la etapa 1
  - Aprende más rápido que un modelo equivalente a CLIP, quizá en detrimento de quienes están acostumbrados a entrenar modelos CLIP
    - En otras palabras, tendrás que ajustar tus expectativas de tasas de aprendizaje y cronogramas de entrenamiento
- No hay VAE; las muestras de entrenamiento se reducen directamente a su tamaño objetivo y los píxeles se consumen por el U-net
- Soporta ControlNet LoRAs y muchos otros trucos que funcionan en u-nets CLIP lineales típicos.

## Ajuste fino de un LoRA

> ⚠️ Debido a los requisitos de cómputo del backprop completo de u-net incluso en el modelo más pequeño de 400M de DeepFloyd, no se ha probado. Se usará LoRA para este documento, aunque el ajuste completo de u-net también debería funcionar.

Estas instrucciones asumen familiaridad básica con SimpleTuner. Para recién llegados, se recomienda empezar con un modelo mejor soportado como [Kwai Kolors](quickstart/KOLORS.md).

Sin embargo, si deseas entrenar DeepFloyd, requiere el uso de la opción de configuración `model_flavour` para indicar qué modelo estás entrenando.

### config.json

```bash
"model_family": "deepfloyd",

# Possible values:
# - i-medium-400m
# - i-large-900m
# - i-xlarge-4.3b
# - ii-medium-450m
# - ii-large-1.2b
"model_flavour": "i-medium-400m",

# DoRA isn't tested a whole lot yet. It's still new and experimental.
"use_dora": false,
# Bitfit hasn't been tested for efficacy on DeepFloyd.
# It will probably work, but no idea what the outcome is.
"use_bitfit": false,

# Highest learning rate to use.
"learning_rate": 4e-5,
# For schedules that decay or oscillate, this will be the end LR or the bottom of the valley.
"lr_end": 4e-6,
```

- `model_family` es deepfloyd
- `model_flavour` apunta a la etapa I o II
- `resolution` ahora es `64` y `resolution_type` es `pixel`
- `attention_mechanism` se puede establecer en `xformers`, pero usuarios de AMD y Apple no podrán configurarlo, lo que requiere más VRAM.
  - **Nota** ~~Apple MPS actualmente tiene un bug que impide que el ajuste de DeepFloyd funcione en absoluto.~~ A partir de Pytorch 2.6 o antes, las etapas I y II entrenan en Apple MPS.

Para validaciones más exhaustivas, el valor de `validation_resolution` puede establecerse en:

- `validation_resolution=64` dará como resultado una imagen cuadrada de 64x64.
- `validation_resolution=96x64` dará como resultado una imagen panorámica 3:2.
- `validation_resolution=64,96,64x96,96x64` dará como resultado cuatro imágenes generadas para cada validación:
  - 64x64
  - 96x96
  - 64x96
  - 96x64

### multidatabackend_deepfloyd.json

Ahora pasemos a configurar el dataloader para entrenamiento de DeepFloyd. Esto será casi idéntico a la configuración de datasets de SDXL o modelos legacy, con un enfoque en parámetros de resolución.

```json
[
    {
        "id": "primary-dataset",
        "type": "local",
        "instance_data_dir": "/training/data/primary-dataset",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "random",
        "resolution": 64,
        "resolution_type": "pixel",
        "minimum_image_size": 64,
        "maximum_image_size": 256,
        "target_downsample_size": 128,
        "prepend_instance_prompt": false,
        "instance_prompt": "Your Subject Trigger Phrase or Word",
        "caption_strategy": "instanceprompt",
        "repeats": 1
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "disable": false,
        "type": "local",
        "cache_dir": "/training/cache/deepfloyd/text/dreambooth"
    }
]
```

Arriba se proporciona una configuración básica de Dreambooth para DeepFloyd:

- Los valores de `resolution` y `resolution_type` se configuran en `64` y `pixel`, respectivamente
- El valor de `minimum_image_size` se reduce a 64 píxeles para asegurar que no escalemos hacia arriba imágenes más pequeñas por accidente
- El valor de `maximum_image_size` se establece en 256 píxeles para asegurar que imágenes grandes no se recorten con una proporción mayor a 4:1, lo que puede provocar una pérdida catastrófica del contexto de la escena
- El valor de `target_downsample_size` se establece en 128 píxeles para que cualquier imagen mayor que `maximum_image_size` de 256 píxeles primero se redimensione a 128 píxeles antes de recortar

Nota: las imágenes se reducen un 25% cada vez para evitar saltos extremos de tamaño que provoquen un promedio incorrecto de los detalles de la escena.

## Modos de pipeline de validación

La validación de DeepFloyd ahora puede encadenar stages directamente en SimpleTuner. `deepfloyd_validation_pipeline_mode=auto` es el valor predeterminado: prompt validation ejecuta stage I -> stage II, mientras que dataset-image validation permanece en el stage entrenado. Usa `trained-stage` para forzar validación single-stage, o `full-pipeline` para cargar siempre el peer stage fijo. Sobrescribe los peer checkpoints con `deepfloyd_validation_stage1_model` y `deepfloyd_validation_stage2_model`.

La validación opcional de stage III puede activarse con `deepfloyd_validation_stage3_mode=sd-x4-upscaler`; esto usa `deepfloyd_validation_stage3_model` como upscaler terminal 4x después de stage II.

## Ejecutar inferencia

Actualmente, DeepFloyd no tiene scripts de inferencia dedicados en el toolkit de SimpleTuner.

Aparte del proceso de validaciones integrado, quizá quieras consultar [este documento de Hugging Face](https://huggingface.co/docs/diffusers/v0.23.1/en/training/dreambooth#if) que contiene un pequeño ejemplo para ejecutar inferencia después:

```py
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", use_safetensors=True)
pipe.load_lora_weights("<lora weights path>")
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
```

> ⚠️ Ten en cuenta que el primer valor de `DiffusionPipeline.from_pretrained(...)` está establecido en `IF-I-M-v1.0`, pero debes actualizarlo para usar la ruta del modelo base sobre la que entrenaste tu LoRA.

> ⚠️ Ten en cuenta que no todas las recomendaciones de Hugging Face aplican a SimpleTuner. Por ejemplo, podemos ajustar un LoRA de DeepFloyd etapa I con solo 22G de VRAM frente a 28G en los scripts de dreambooth de ejemplo de Diffusers gracias al pre-cacheo eficiente y estados de optimizador pure-bf16.

## Ajuste fino del modelo de super-resolución de la etapa II

El modelo de etapa II de DeepFloyd toma entradas de imágenes alrededor de 64x64 (o 96x64) y devuelve la imagen reescalada resultante usando la configuración `VALIDATION_RESOLUTION`.

Las imágenes de evaluación se recopilan automáticamente de tus datasets, de modo que `--num_eval_images` especifica cuántas imágenes de reescalado seleccionar de cada dataset. Actualmente se seleccionan al azar, pero permanecerán iguales en cada sesión.

Hay algunas comprobaciones adicionales para asegurar que no ejecutes accidentalmente con tamaños incorrectos.

Para entrenar la etapa II, solo necesitas seguir los pasos anteriores, usando `deepfloyd-stage2-lora` en lugar de `deepfloyd-lora` para `MODEL_TYPE`:

```bash
export MODEL_TYPE="deepfloyd-stage2-lora"
```
