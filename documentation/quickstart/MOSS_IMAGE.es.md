# Guía rápida de MOSS-Image

Entrena un LoRA de MOSS-Image con tus propias imágenes. Un LoRA es un pequeño adaptador que enseña un sujeto o estilo al modelo sin mezclarlo con el checkpoint original.

Empieza con `model_family: "moss_image"`, `model_flavour: "sft"` y `model_type: "lora"` para usar [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), la variante predeterminada, refinada con datos seleccionados. Selecciona `model_flavour: "base"` para [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base). Ambos comparten un transformador de imágenes de 5B, el codificador de texto Qwen3.5-9B y un VAE de FLUX.2. Los checkpoints tienen licencia [Apache-2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/LICENSE). El codificador de texto permanece congelado durante el entrenamiento.

## Elegir un preset de VRAM

Elige la receta según la memoria de tu GPU. Todos los niveles entrenan con un área equivalente a 512×512, conservan las proporciones y mantienen LoRA BF16 de rango 32 y REPA.

| Memoria GPU | Ejemplo Base | Ejemplo SFT | Imágenes por lote | `gradient_checkpointing_interval` |
| --- | --- | --- | ---: | ---: |
| 24 GB | `moss_image/base/24g/peft-lora` | `moss_image/sft/24g/peft-lora` | 1 | 1 |
| 32 GB | `moss_image/base/32g/peft-lora` | `moss_image/sft/32g/peft-lora` | 1 | 2 |
| 48 GB | `moss_image/base/48g/peft-lora` | `moss_image/sft/48g/peft-lora` | 2 | 2 |
| 80 GB | `moss_image/base/80g/peft-lora` | `moss_image/sft/80g/peft-lora` | 4 | 2 |
| 144 GB | `moss_image/base/144g/peft-lora` | `moss_image/sft/144g/peft-lora` | 8 | 2 |

Los ejemplos predeterminados `moss_image/sft/default/default.peft-lora` y `moss_image/sft/default/peft-lora` usan los ajustes de 48 GB. Los lotes son puntos de partida: reduce `train_batch_size` para descripciones más largas, funciones adicionales o datasets con menos imágenes por grupo de proporción que el tamaño del lote. El nivel de 144 GB está pensado para GPU de clase H200.

## Instalar SimpleTuner

Usa Python 3.13 y una GPU NVIDIA para la receta CUDA siguiente. Consulta la [instalación](../INSTALL.md) para otras plataformas y versiones de CUDA.

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda,attention-cute]'
```

## Preparar imágenes y descripciones

Crea `datasets/moss-image` y añade imágenes con archivos `.txt` del mismo nombre. Usa imágenes nítidas con detalle suficiente para entrenar a 512px. Una imagen cuadrada equivale a 512×512; las imágenes anchas y altas conservan sus proporciones.

```text
datasets/moss-image/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

Elige una palabra como `TOK` e inclúyela en todas las descripciones de ese sujeto. Describe el sujeto visible, la composición, la luz, el entorno y el estilo. El proyecto recomienda empezar en inglés. Copia el texto visible con precisión y conserva los saltos de línea. Ejemplo: `A photograph of TOK, a small blue ceramic owl on a wooden table beside a window.`

## Configurar el entrenamiento

Copia la receta SFT de 48 GB y sus prompts. Sustituye `48g` por el nivel elegido. Para Base, sustituye `moss_image/sft/48g/peft-lora` por `moss_image/base/48g/peft-lora` en los comandos.

```bash
mkdir -p config/examples/_shared datasets/moss-image
mkdir -p config/examples/moss_image/sft/48g
cp -R simpletuner/examples/moss_image/sft/48g/peft-lora config/examples/moss_image/sft/48g/
mkdir -p config/examples/_shared
cp simpletuner/examples/_shared/moss_image-prompts.json config/examples/_shared/
cp config/examples/moss_image/sft/48g/peft-lora/config.json config/config.json
```

La configuración utiliza LoRA BF16 de rango 32, tasa de aprendizaje `1e-4`, gradient checkpointing, REPA, 4.000 pasos del optimizador y escala 512. Abre `config/config.json` y modifica estas entradas dentro de la configuración existente:

```json
{
  "output_dir": "output/my-moss-lora",
  "validation_prompt": "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
  "user_prompt_library": null
}
```

El `dataloader.json` copiado apunta a `datasets/moss-image`, lee los archivos de texto correspondientes y almacena latentes y embeddings en caché. `resolution: 0.262144` con `resolution_type: "area"` equivale al área de 512×512. Las dimensiones se alinean a múltiplos de 16. Consulta la [guía del dataloader](../DATALOADER.md) para añadir datasets e imágenes de regularización.

## Entrenar y comparar vistas previas

Ejecuta `simpletuner train` desde SimpleTuner con el entorno virtual activo. La primera ejecución descarga el modelo y el codificador REPA y crea las cachés. La receta guarda checkpoints e imágenes de validación cada 250 pasos. Compara el aprendizaje del sujeto, la composición y el detalle; ajusta `max_train_steps` a tu dataset.

```bash
simpletuner train
```

## Ajustes de muestreo

Empieza con **50 pasos**, **CFG 3** y validación de **1024×1024**. `moss_image_validation_shift` vale **5** para el muestreo; el entrenamiento utiliza la cuadrícula original sin desplazamiento. `moss_image_uncond_skip_layers` vale **[22]** y omite esa capa en la rama negativa de CFG. Usa `[]` para CFG estándar. La omisión adicional en la rama positiva usa `validation_guidance_skip_layers`, `validation_guidance_skip_scale` y los límites de inicio y fin de [OPTIONS](../OPTIONS.md). En Python, `use_cfg_zero_star=True` activa CFG-Zero*.

El entrenamiento usa automáticamente la distribución de ruido nativa de MOSS-Image, incluida su mezcla uniforme del 1%. La validación aplica el desplazamiento una sola vez e incluye el extremo final sin ruido. Para AnyFlow, usa el mismo desplazamiento en validación y en las trayectorias de entrenamiento.

## Vistas previas durante el muestreo

Añade estos ajustes para ver fotogramas de progreso durante el denoising en la WebUI. Las vistas previas usan TAEF2 y las imágenes finales usan el VAE completo. La CLI puede enviar vistas previas a un webhook configurado.

```json
{
  "validation_preview": true,
  "validation_preview_steps": 5
}
```

## Memoria y opciones de entrenamiento

La receta ajusta `train_batch_size` al nivel de VRAM elegido. Redúcelo para un dataset pequeño o si las descripciones y funciones adicionales necesitan más memoria. `gradient_checkpointing_backend` acepta `torch`, `unsloth`, `torch-ffn` y `unsloth-ffn`; el intervalo y `gradient_checkpointing_segment_stride` controlan la cobertura. Musubi mueve bloques entre GPU y CPU con `musubi_blocks_to_swap`. RamTorch usa `ramtorch: true` y permite descarga selectiva mediante `ramtorch_transformer_percent`. La descarga consume RAM y añade transferencias; configura un método cada vez. Consulta [OPTIONS](../OPTIONS.md).

La instalación CUDA incluye atención CuTE. Durante el entrenamiento, el valor predeterminado `attention_mechanism: "kohaku-fa-auto"` selecciona CuTE en GPU Ada y Hopper y SDPA nativo en otras arquitecturas. Activa `gradient_checkpointing_offload_attention: true` para descargar los tensores de atención guardados para backward; `gradient_checkpointing_offload_prefetch: true` activa la precarga de transferencias. Consulta [OPTIONS](../OPTIONS.md) para otros backends de atención.

La receta activa [REPA](../experimental/IMAGE_REPA.md). Para alinear características de una capa posterior con una anterior, añade [LayerSync](../experimental/LAYERSYNC.md):

```json
{
  "layersync_enabled": true,
  "layersync_student_block": 20,
  "layersync_teacher_block": 8,
  "layersync_lambda": 0.2
}
```

Para postentrenamiento, sigue [AnyFlow](../quickstart/ANYFLOW.md) o [TwinFlow](../distillation/TWINFLOW.md). Mantén `lora_dropout: 0.0` con AnyFlow. Al cargar su LoRA en Python, pasa los ajustes de entrenamiento a `load_lora_weights`, por ejemplo `flowmap_deltatime_type="r", flowmap_gate_value=0.25`.

## Mejorar prompts y usar layouts

MOSS-Image acepta descripciones detalladas, etiquetas y layouts JSON. La [skill para mejorar prompts](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer) ayuda a un agente a redactarlos. Usa el resultado como descripción o prompt. La [guía del proyecto](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md) incluye recomendaciones de captioning.

Un layout describe la escena, el texto y las posiciones. Las coordenadas son `[y_min, x_min, y_max, x_max]` en píxeles, con origen en el centro del lienzo. En 1024×1024 van de −512 a +512. Serializa con `json.dumps(layout, ensure_ascii=False)` y pasa esa cadena como prompt; usa dimensiones de generación iguales a las del layout.

Para ampliar las descripciones a JSON nativo durante el preprocesamiento, añade estos ajustes a tu configuración:

```json
{
  "moss_image_auto_json": true,
  "moss_image_prompt_upsample": true
}
```

El modelo predeterminado es `Qwen/Qwen3.5-27B`, uno de los VLM recomendados por el proyecto. Usa una GPU de 80 GB y reserva RAM adicional. SimpleTuner utiliza cada imagen y las dimensiones de su bucket para generar descripciones y cajas en píxeles centrados, y almacena los embeddings. También amplía los prompts de validación; conserva los vacíos y negativos. Para envolver las descripciones existentes en JSON sin cargar otro modelo, activa solo `moss_image_auto_json`.

```json
{
  "summary": {"caption": "A cream travel poster with the title EXPLORE above a blue mountain."},
  "items": [
    {"label": "text", "text": "EXPLORE", "caption": "Large navy title", "bbox": [-416, -384, -288, 384]},
    {"label": "object", "caption": "Blue mountain", "bbox": [-192, -288, 384, 288]}
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

## Generar imágenes con tu LoRA

El adaptador final es `pytorch_lora_weights.safetensors`, en el directorio de salida. Cárgalo con el mismo checkpoint Base o SFT usado para entrenar. Incluye la palabra del sujeto y describe la escena:

```python
import torch
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline

pipe = MOSSImagePipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-Image-SFT", torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("output/my-moss-lora")
pipe.to("cuda")
image = pipe(
    "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
    height=1024, width=1024, num_inference_steps=50, guidance_scale=3.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("my-moss-image.png")
```

## Entrenar una LoRA de edición o referencia

Elige `model_flavour: "sft-edit"` o `"base-edit"` para entrenar con imágenes de referencia. Estas variantes parten del checkpoint SFT o Base correspondiente y aprenden el condicionamiento mediante tu adaptador. Empieza con `simpletuner/examples/moss_image/sft-edit/default/peft-lora` o `moss_image/base-edit/default/peft-lora`; usan una imagen objetivo por lote, entrenamiento a escala 512, REPA y validación del conjunto de datos.

```bash
mkdir -p config/examples/_shared datasets/moss-edits datasets/moss-references
mkdir -p config/examples/moss_image/sft-edit/default
cp -R simpletuner/examples/moss_image/sft-edit/default/peft-lora config/examples/moss_image/sft-edit/default/
cp config/examples/moss_image/sft-edit/default/peft-lora/config.json config/config.json
```

Guarda el resultado deseado en `datasets/moss-edits/sample.png`, la instrucción en `datasets/moss-edits/sample.txt` y la referencia de entrada en `datasets/moss-references/sample.png`. Los nombres de las imágenes, incluidas las extensiones, deben coincidir en ambas carpetas. Una instrucción de ejemplo es `Change the blue mug to red while keeping the background unchanged.` Los ejemplos de edición incluyen esta configuración del dataloader:

```json
[
  {
    "id": "edits",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/moss-edits",
    "caption_strategy": "textfile",
    "resolution": 0.262144,
    "resolution_type": "area",
    "crop": false,
    "conditioning_data": [
      "references"
    ],
    "cache_dir_vae": "cache/vae/moss-image/sft-edit/targets",
    "text_embeds": "text",
    "repeats": 10
  },
  {
    "id": "references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "datasets/moss-references",
    "conditioning_type": "reference_loose",
    "resolution": 0.262144,
    "resolution_type": "area",
    "crop": false,
    "cache_dir_vae": "cache/vae/moss-image/sft-edit/references"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/moss-image/sft-edit"
  }
]
```

`reference_loose` conserva la relación de aspecto de cada referencia y permite tamaños distintos para referencias y objetivos. Usa `reference_strict` para pares alineados. Añade más conjuntos de referencia a `conditioning_data` y mantén `conditioning_multidataset_sampling: "combined"` para proporcionar varias referencias juntas. `"random"` selecciona un conjunto de referencia por muestra; añade instrucciones propias a cada conjunto cuando sean diferentes. Separa las cachés de objetivos, referencias y texto. Las referencias requieren memoria adicional: reduce el lote o utiliza las opciones de descarga anteriores según sea necesario.

La validación utiliza las referencias e instrucciones de los conjuntos emparejados. Los píxeles de referencia pasan por la torre visual congelada de Qwen3.5 y los latentes limpios del VAE aportan los detalles; la pérdida se calcula sobre la imagen objetivo. El descarte de captions conserva las referencias y elimina solo la instrucción. El procesador de imágenes Qwen se descarga automáticamente de `Qwen/Qwen3.5-9B`.

Tras entrenar, carga el adaptador de edición y pasa una imagen de referencia o una lista a `image`. Para un lote de prompts, usa una lista anidada con un conjunto de referencias por prompt. La caché K/V de referencias está activada durante el muestreo; usa `cache_reference_kv=False` para recalcular los tokens de referencia en cada paso.

```python
import torch
from diffusers.utils import load_image
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline

pipe = MOSSImagePipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-Image-SFT", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("output/examples/moss_image-sft-edit.peft-lora")
image = pipe(
    "Change the blue mug to red while keeping the background unchanged.",
    image=[load_image("reference.png")],
    height=512, width=512, num_inference_steps=50, guidance_scale=3.0,
).images[0]
image.save("edited.png")
```
