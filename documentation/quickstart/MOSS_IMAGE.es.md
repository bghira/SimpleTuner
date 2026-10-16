# Guía rápida de MOSS-Image

Entrena un LoRA de MOSS-Image con tus propias imágenes. Un LoRA es un pequeño adaptador que enseña un sujeto o estilo al modelo sin mezclarlo con el checkpoint original.

Empieza con `model_family: "moss_image"`, `model_flavour: "base"` y `model_type: "lora"` para usar [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base). Selecciona `model_flavour: "sft"` para [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), refinado con datos seleccionados. Ambos comparten un transformador de imágenes de 5B, el codificador de texto Qwen3.5-9B y un VAE de FLUX.2. Los checkpoints tienen licencia [Apache-2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/LICENSE). El codificador de texto permanece congelado durante el entrenamiento.

## Instalar SimpleTuner

Usa Python 3.13 y una GPU NVIDIA para la receta CUDA siguiente. Una GPU de 48 GB es un buen punto de partida; la longitud de las descripciones, el lote y las funciones adicionales afectan al consumo de memoria. Consulta la [instalación](../INSTALL.md) para otras plataformas y versiones de CUDA.

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda]'
```

## Preparar imágenes y descripciones

Crea `datasets/moss-image` y añade imágenes con archivos `.txt` del mismo nombre. Usa imágenes nítidas con detalle suficiente para entrenar a aproximadamente un megapíxel. Una imagen cuadrada equivale a 1024×1024; las imágenes anchas y altas conservan sus proporciones.

```text
datasets/moss-image/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

Elige una palabra como `TOK` e inclúyela en todas las descripciones de ese sujeto. Describe el sujeto visible, la composición, la luz, el entorno y el estilo. El proyecto recomienda empezar en inglés. Copia el texto visible con precisión y conserva los saltos de línea. Ejemplo: `A photograph of TOK, a small blue ceramic owl on a wooden table beside a window.`

## Configurar el entrenamiento

Copia la receta Base y sus prompts de ejemplo. Para empezar con SFT, sustituye `moss_image.peft-lora` por `moss_image-sft.peft-lora` en los comandos.

```bash
mkdir -p config/examples datasets/moss-image
cp -R simpletuner/examples/moss_image.peft-lora config/examples/
cp simpletuner/examples/moss_image-prompts.json config/examples/
cp config/examples/moss_image.peft-lora/config.json config/config.json
```

La configuración utiliza LoRA BF16 de rango 32, tasa de aprendizaje `1e-4`, gradient checkpointing, REPA, 4.000 pasos del optimizador y escala 1024. Abre `config/config.json` y modifica estas entradas dentro de la configuración existente:

```json
{
  "output_dir": "output/my-moss-lora",
  "train_batch_size": 1,
  "validation_prompt": "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
  "user_prompt_library": null
}
```

El `dataloader.json` copiado apunta a `datasets/moss-image`, lee los archivos de texto correspondientes y almacena latentes y embeddings en caché. `resolution: 1.048576` con `resolution_type: "area"` equivale al área de 1024×1024. Las dimensiones se alinean a múltiplos de 16. Consulta la [guía del dataloader](../DATALOADER.md) para añadir datasets e imágenes de regularización.

## Entrenar y comparar vistas previas

Ejecuta `simpletuner train` desde SimpleTuner con el entorno virtual activo. La primera ejecución descarga el modelo y el codificador REPA y crea las cachés. La receta guarda checkpoints e imágenes de validación cada 250 pasos. Compara el aprendizaje del sujeto, la composición y el detalle; ajusta `max_train_steps` a tu dataset.

```bash
simpletuner train
```

## Ajustes de muestreo

Empieza con **50 pasos**, **CFG 3** y validación de **1024×1024**. `moss_image_validation_shift` vale **5** para el muestreo; el entrenamiento utiliza la cuadrícula original sin desplazamiento. `moss_image_uncond_skip_layers` vale **[22]** y omite esa capa en la rama negativa de CFG. Usa `[]` para CFG estándar. La omisión adicional en la rama positiva usa `validation_guidance_skip_layers`, `validation_guidance_skip_scale` y los límites de inicio y fin de [OPTIONS](../OPTIONS.md). En Python, `use_cfg_zero_star=True` activa CFG-Zero*.

## Vistas previas durante el muestreo

Añade estos ajustes para ver fotogramas de progreso durante el denoising en la WebUI. Las vistas previas usan TAEF2 y las imágenes finales usan el VAE completo. La CLI puede enviar vistas previas a un webhook configurado.

```json
{
  "validation_preview": true,
  "validation_preview_steps": 5
}
```

## Memoria y opciones de entrenamiento

Mantén `train_batch_size: 1` al empezar. Auméntalo cuando tengas suficientes imágenes y memoria. `gradient_checkpointing_backend` acepta `torch`, `unsloth`, `torch-ffn` y `unsloth-ffn`; el intervalo y `gradient_checkpointing_segment_stride` controlan la cobertura. Musubi mueve bloques entre GPU y CPU con `musubi_blocks_to_swap`. RamTorch usa `ramtorch: true` y permite descarga selectiva mediante `ramtorch_transformer_percent`. La descarga consume RAM y añade transferencias; configura un método cada vez. Consulta [OPTIONS](../OPTIONS.md).

En GPU NVIDIA, instala las dependencias opcionales de atención CuTE con `pip install -e '.[cuda,attention-cute]'` y selecciona `attention_mechanism: "kohaku-fa"`. El selector compartido también acepta otros backends instalados. Activa `gradient_checkpointing_offload_attention: true` para descargar los tensores de atención guardados para backward; `gradient_checkpointing_offload_prefetch: true` activa la precarga de transferencias.

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
    "OpenMOSS-Team/MOSS-Image-Base", torch_dtype=torch.bfloat16
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
