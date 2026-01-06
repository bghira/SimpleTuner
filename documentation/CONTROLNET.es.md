# Guía de entrenamiento de ControlNet

## Antecedentes

Los modelos ControlNet son capaces de realizar muchas tareas, que dependen de los datos de condicionamiento entregados durante el entrenamiento.

Al principio eran muy costosos en recursos, pero ahora podemos usar PEFT LoRA o Lycoris para entrenar las mismas tareas con muchos menos recursos.

Ejemplo (tomado de la ficha del modelo ControlNet de Diffusers):

![ejemplo](https://tripleback.net/public/controlnet-example-1.png)

A la izquierda puedes ver el "mapa de bordes canny" dado como entrada de condicionamiento. A la derecha están las salidas que el modelo ControlNet guió a partir del modelo base SDXL.

Cuando el modelo se usa de esta manera, el prompt casi no controla la composición, solo rellena los detalles.

## Cómo se ve entrenar un ControlNet

Al principio, al entrenar un ControlNet, no tiene ninguna indicación de control:

![ejemplo](https://tripleback.net/public/controlnet-example-2.png)
(_ControlNet entrenado solo 4 pasos en un modelo Stable Diffusion 2.1_)

El prompt del antílope sigue teniendo la mayor parte del control sobre la composición, y la entrada de condicionamiento de ControlNet se ignora.

Con el tiempo, la entrada de control debería ser respetada:

![ejemplo](https://tripleback.net/public/controlnet-example-3.png)
(_ControlNet entrenado solo 100 pasos en un modelo Stable Diffusion XL_)

En ese punto empezaron a aparecer algunas señales de la influencia de ControlNet, pero los resultados eran increíblemente inconsistentes.

¡Se necesitarán muchos más de 100 pasos para que esto funcione!

## Ejemplo de configuración del dataloader

La configuración del dataloader se mantiene bastante cerca de una configuración típica de dataset de texto a imagen:

- Los datos de imagen principales son el conjunto `antelope-data`
  - La clave `conditioning_data` ahora está configurada y debe apuntar al valor `id` de tus datos de condicionamiento que se emparejan con este conjunto.
  - `dataset_type` debe ser `image` para el conjunto base
- Se configura un dataset secundario, llamado `antelope-conditioning`
  - El nombre no es importante; agregar `-data` y `-conditioning` en este ejemplo solo es para fines ilustrativos.
  - `dataset_type` debe establecerse en `conditioning`, lo que indica al entrenador que esto se usará para evaluación y entrenamiento con entradas condicionadas.
- Al entrenar SDXL, las entradas de condicionamiento no se codifican con VAE, sino que se pasan directamente al modelo durante el entrenamiento como valores de píxel. Esto significa que no gastamos más tiempo procesando embeddings del VAE al inicio del entrenamiento.
- Al entrenar Flux, SD3, Auraflow, HiDream u otros modelos MMDiT, las entradas de condicionamiento se codifican en latentes y se calculan bajo demanda durante el entrenamiento.
- Aunque aquí todo está etiquetado explícitamente como `-controlnet`, puedes reutilizar los mismos embeddings de texto que usaste para el ajuste completo/LoRA normal. Las entradas de ControlNet no modifican los embeddings del prompt.
- Cuando se usan buckets de aspecto y recorte aleatorio, las muestras de condicionamiento se recortarán de la misma manera que las muestras de imagen principales, así que no tienes que preocuparte por eso.

```json
[
    {
        "id": "antelope-data",
        "type": "local",
        "dataset_type": "image",
        "conditioning_data": "antelope-conditioning",
        "instance_data_dir": "datasets/animals/antelope-data",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an antelope",
        "metadata_backend": "discovery",
        "minimum_image_size": 512,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "cache_dir_vae": "cache/vae/sdxl/antelope-data",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "cache_file_suffix": "controlnet-sdxl"
    },
    {
        "id": "antelope-conditioning",
        "type": "local",
        "dataset_type": "conditioning",
        "instance_data_dir": "datasets/animals/antelope-conditioning",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an antelope",
        "metadata_backend": "discovery",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "minimum_image_size": 512,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel_area",
        "cache_file_suffix": "controlnet-sdxl"
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/sdxl-base/controlnet"
    }
]
```

## Generar entradas de imagen de condicionamiento

Aunque el soporte de ControlNet es nuevo en SimpleTuner, por ahora solo tenemos una opción disponible para generar tu conjunto de entrenamiento:

- [create_canny_edge.py](/scripts/toolkit/datasets/controlnet/create_canny_edge.py)
  - Un ejemplo extremadamente básico para generar un conjunto de entrenamiento para modelos Canny.
  - Tendrás que modificar los valores `input_dir` y `output_dir` en el script

Esto tarda unos 30 segundos para un dataset pequeño de menos de 100 imágenes.

## Modificar la configuración para entrenar modelos ControlNet

Solo configurar el dataloader no será suficiente para comenzar a entrenar modelos ControlNet.

Dentro de `config/config.json`, tendrás que establecer los siguientes valores:

```bash
"model_type": 'lora',
"controlnet": true,

# You may have to reduce TRAIN_BATCH_SIZE and RESOLUTION more than usual
"train_batch_size": 1
```

Tu configuración se verá algo así al final:

```json
{
    "aspect_bucket_rounding": 2,
    "caption_dropout_probability": 0.1,
    "checkpoint_step_interval": 100,
    "checkpoints_total_limit": 5,
    "controlnet": true,
    "data_backend_config": "config/controlnet-sdxl/multidatabackend.json",
    "disable_benchmark": false,
    "gradient_checkpointing": true,
    "hub_model_id": "simpletuner-controlnet-sdxl-lora-test",
    "learning_rate": 3e-5,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 100,
    "max_train_steps": 1000,
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "sdxl",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "bnb-lion8bit",
    "output_dir": "output/controlnet-sdxl/models",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 42,
    "train_batch_size": 1,
    "use_ema": false,
    "vae_cache_ondemand": true,
    "validation_guidance": 4.2,
    "validation_guidance_rescale": 0.0,
    "validation_num_inference_steps": 20,
    "validation_resolution": "1024x1024",
    "validation_seed": 42,
    "validation_step_interval": 10,
    "validation_torch_compile": false
}
```

## Inferencia con los modelos ControlNet resultantes

Aquí se proporciona un ejemplo de SDXL para inferencia con un modelo ControlNet **completo** (no ControlNet LoRA):

```py
# Update these values:
base_model = "stabilityai/stable-diffusion-xl-base-1.0"         # This is the model you used as `--pretrained_model_name_or_path`
controlnet_model_path = "diffusers/controlnet-canny-sdxl-1.0"   # This is the path to the resulting ControlNet checkpoint
# controlnet_model_path = "/path/to/controlnet/checkpoint-100"

# Leave the rest alone:
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'

image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    controlnet_model_path,
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

images = pipe(
    prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

images[0].save(f"hug_lab.png")
```
(_Código de demostración tomado del [ejemplo de ControlNet SDXL en Hugging Face](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)_)


## Aumento automático de datos y generación de condicionamiento

SimpleTuner puede generar automáticamente datasets de condicionamiento durante el inicio, eliminando la necesidad de preprocesamiento manual. Esto es especialmente útil para:
- Entrenamiento de super-resolución
- Eliminación de artefactos JPEG
- Generación guiada por profundidad
- Detección de bordes (Canny)

### Cómo funciona

En lugar de crear datasets de condicionamiento manualmente, puedes especificar un arreglo `conditioning` en la configuración de tu dataset principal. SimpleTuner:
1. Genera las imágenes de condicionamiento al inicio
2. Crea datasets separados con los metadatos adecuados
3. Los enlaza automáticamente con tu dataset principal

### Consideraciones de rendimiento

Algunos generadores se ejecutarán más lento si dependen de CPU y tu sistema tiene problemas de carga de CPU, mientras que otros pueden requerir recursos de GPU y por lo tanto se ejecutan en el proceso principal, lo que puede incrementar el tiempo de inicio.

**Generadores basados en CPU (rápidos):**
- `superresolution` - Operaciones de desenfoque y ruido
- `jpeg_artifacts` - Simulación de compresión
- `random_masks` - Generación de máscaras
- `canny` - Detección de bordes

**Generadores basados en GPU (más lentos):**
- `depth` / `depth_midas` - Requiere cargar modelos transformer
- `segmentation` - Modelos de segmentación semántica
- `optical_flow` - Estimación de movimiento

Los generadores basados en GPU se ejecutan en el proceso principal y pueden aumentar significativamente el tiempo de inicio para datasets grandes.

### Ejemplo: Dataset de condicionamiento multitarea

Aquí hay un ejemplo completo que genera múltiples tipos de condicionamiento a partir de un solo dataset fuente:

```json
[
  {
    "id": "multitask-training",
    "type": "local",
    "instance_data_dir": "/datasets/high-quality-images",
    "caption_strategy": "filename",
    "resolution": 512,
    "conditioning": [
      {
        "type": "superresolution",
        "blur_radius": 2.0,
        "noise_level": 0.02,
        "captions": ["enhance image quality", "increase resolution", "sharpen"]
      },
      {
        "type": "jpeg_artifacts",
        "quality_range": [20, 40],
        "captions": ["remove compression", "fix jpeg artifacts"]
      },
      {
        "type": "canny",
        "low_threshold": 50,
        "high_threshold": 150
      }
    ]
  },
  {
    "id": "text-embed-cache",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/sdxl"
  }
]
```

Esta configuración:
1. Carga tus imágenes de alta calidad desde `/datasets/high-quality-images`
2. Genera tres datasets de condicionamiento automáticamente
3. Usa captions específicos para tareas de super-resolución y JPEG
4. Usa las captions originales de las imágenes para el dataset de bordes Canny

#### Estrategias de captions para datasets generados

Tienes dos opciones para las captions de los datos de condicionamiento generados:

1. **Usar captions de origen** (predeterminado): Omite el campo `captions`
2. **Captions personalizadas**: Proporciona una cadena o un arreglo de cadenas

Para entrenamiento específico de tareas (como "enhance" o "remove artifacts"), las captions personalizadas suelen funcionar mejor que las descripciones originales de las imágenes.

### Optimización del tiempo de inicio

Para datasets grandes, la generación de condicionamiento puede consumir mucho tiempo. Para optimizar:

1. **Generar una vez**: Los datos de condicionamiento se cachean y no se regeneran si ya existen
2. **Usar generadores de CPU**: Estos pueden usar múltiples procesos para generar más rápido
3. **Deshabilitar tipos no usados**: Solo genera lo que necesitas para tu entrenamiento
4. **Pre-generar**: Puedes ejecutar con `--skip_file_discovery=true` para omitir el descubrimiento y la generación de datos de condicionamiento
5. **Evitar escaneos de disco**: Puedes usar `preserve_data_backend_cache=True` en cualquier configuración de dataset grande para evitar volver a escanear el disco en busca de datos de condicionamiento existentes. Esto acelerará significativamente el tiempo de inicio, especialmente en datasets grandes.

El proceso de generación muestra barras de progreso y admite reanudación si se interrumpe.
