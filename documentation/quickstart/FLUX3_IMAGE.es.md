# Inicio rápido de FLUX.3 Image Klein

Esta guía explica cómo entrenar una LoRA de FLUX.3 Image Klein con tus imágenes. Una LoRA es un adaptador pequeño que enseña un sujeto o estilo al modelo y mantiene sus pesos originales separados.

Empieza con `model_family: "flux3_image"`, `model_flavour: "klein-base"` y `model_type: "lora"`. Se cargará [black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base). Visita la página, acepta las condiciones de acceso y lee la [licencia FLUX [Kom]munity](https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/LICENSE) antes de descargar.

## Instalación

Necesitas Python 3.13, Git y espacio libre para el modelo, las imágenes y la caché. Los comandos siguientes instalan SimpleTuner para una GPU NVIDIA CUDA. Consulta la [guía de instalación](../INSTALL.es.md) para otras plataformas y CUDA 13.

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda]'
hf auth login
```

Ejecuta el resto desde la carpeta `SimpleTuner`, con el entorno virtual activo. En `hf auth login`, introduce un token de Hugging Face con permiso de lectura para el modelo.

## Elige una receta para tu GPU

La memoria GPU (VRAM) determina cuántas imágenes puedes procesar juntas. Elige el ejemplo correspondiente a tu memoria. Las recetas Base de 24 y 32 GB usan resolución 512; las mayores también incluyen datasets a 1024.

| Memoria GPU | Ejemplo Base | Imágenes por lote a 512 / 1024 |
| --- | --- | --- |
| 24 GB | `flux3_image-24g.peft-lora` | 3 / — |
| 32 GB | `flux3_image-32g.peft-lora` | 4 / — |
| 48 GB | `flux3_image-48g.peft-lora` | 9 / 2 |
| 80 GB | `flux3_image-80g.peft-lora` | 18 / 5 |
| 144 GB | `flux3_image-144g.peft-lora` | 36 / 10 |

Para la variante destilada `klein`, elige una de estas recetas. Usa su nombre completo en los comandos de copia de configuración siguientes. La escala corresponde al área de una imagen cuadrada; se conservan las proporciones.

| Memoria GPU | Ejemplo destilado | Escala de entrenamiento | Imágenes por lote |
| --- | --- | --- | ---: |
| 24 GB | `flux3_image-klein-24g.peft-lora` | 768px | 1 |
| 80 GB | `flux3_image-klein-80g.peft-lora` | 1024px | 4 |

Los tamaños de lote son valores iniciales. Las descripciones largas y las referencias consumen más memoria; reduce `train_batch_size` si se agota. Las recetas usan LoRA de rango 32, tasa de aprendizaje `1e-4` y REPA, que compara características visuales durante el entrenamiento. Sus dataloaders combinan concepto y regularización para conservar el conocimiento general del modelo.

## Prepara imágenes y descripciones

Crea `datasets/my-concept` dentro de SimpleTuner y coloca allí tus imágenes. Cada imagen debe tener un archivo de texto con el mismo nombre y su descripción:

```text
datasets/my-concept/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

Para una LoRA de sujeto, elige una palabra distintiva como `TOK` y úsala en todas sus descripciones. Describe el sujeto, el entorno y la postura. Por ejemplo, `image01.txt` puede contener:

```text
Una foto de TOK, un pequeño búho de cerámica con plumas azules, sobre una mesa de madera.
```

## Configura el entrenamiento

Copia la receta elegida. Estos comandos muestran `flux3_image-48g.peft-lora`; sustituye ese nombre completo por tu receta Base o destilada:

```bash
mkdir -p config/examples
cp -R simpletuner/examples/flux3_image-48g.peft-lora config/examples/
cp simpletuner/examples/flux3_image-prompts.json config/examples/
cp config/examples/flux3_image-48g.peft-lora/config.json config/config.json
```

Abre `config/config.json` con un editor de texto. Para la primera ejecución con tus imágenes locales, cambia estos valores dentro de la configuración existente:

```json
{
  "data_backend_config": "config/dataloader.json",
  "output_dir": "output/my-flux3-lora",
  "train_batch_size": 1
}
```

Un lote de 1 permite empezar con pocos datos. Auméntalo cuando la primera ejecución funcione y tengas suficientes imágenes y memoria libre. `max_train_steps` controla la duración. Edita `config/examples/flux3_image-prompts.json` e incluye la palabra de tu sujeto; esos prompts generan imágenes de progreso.

Guarda lo siguiente como `config/dataloader.json`. Indica dónde están las imágenes, cómo leer sus descripciones y dónde almacenar la caché:

```json
[
  {
    "id": "my-concept",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/my-concept",
    "caption_strategy": "textfile",
    "resolution": 0.589824,
    "resolution_type": "area",
    "crop": false,
    "cache_dir_vae": "cache/vae/flux3-image/my-concept",
    "text_embeds": "text"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux3-image/my-concept"
  }
]
```

En este dataset local, `resolution: 0.589824` con `resolution_type: "area"` equivale al área de 768×768. Usa `1.048576` para el dataset a escala 1024 de la receta destilada de 80 GB.

`resolution` controla el tamaño de entrenamiento; las dimensiones se ajustan a múltiplos de 16. La caché guarda latentes y embeddings reutilizables para acelerar ejecuciones posteriores. Usa cachés distintas para cada familia. Consulta la [guía de datos](../DATALOADER.es.md) para añadir datasets o regularización.

## Opcional: amplía las descripciones con el prompt upsampler

Añade este ajuste a `config/config.json` para ampliar descripciones cortas y prompts de validación con el [upsampler de FLUX.3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup):

```json
{
  "flux3_prompt_upsample": true
}
```

Está desactivado por defecto. Al activarlo, SimpleTuner descarga el modelo Qwen3.5-9B separado (unos 19 GB de pesos) y reescribe los prompts positivos antes de guardar sus embeddings. Para ediciones utiliza tus imágenes de referencia emparejadas. Conserva los prompts negativos, las descripciones vacías de dropout y los archivos originales.

Prepara RAM adicional y más tiempo de preprocesamiento. El upsampler y el codificador de texto se turnan en la GPU; sus embeddings se reutilizan durante el entrenamiento. SimpleTuner separa los embeddings ampliados de los originales y reutiliza los resultados cuando vuelven a elegirse las mismas referencias.

## Inicia el entrenamiento

Desde la carpeta SimpleTuner, ejecuta:

```bash
simpletuner train
```

La primera ejecución descarga el modelo, lee las imágenes y crea las cachés antes de entrenar. La compilación puede alargar el inicio. Deja el proceso activo mientras termina estos pasos. Encontrarás imágenes de progreso y checkpoints en `output/my-flux3-lora`; las imágenes permiten evaluar el aprendizaje del sujeto.

Para `klein-base`, usa 50 pasos de validación y guía 3.5. Para entrenar el modelo destilado, elige `model_flavour: "klein"`: carga `black-forest-labs/flux-3-image-klein` y usa 16 pasos con guía 1. Acepta el acceso a ese repositorio antes de comenzar.

## Entrenamiento de edición y referencias

El condicionamiento por referencia es opcional en ambas variantes. El dataset objetivo contiene las ediciones deseadas y sus instrucciones; los datasets de condicionamiento contienen las imágenes de origen o referencia. Guarda este JSON por separado, indica su ruta en `data_backend_config` y empieza con `train_batch_size: 1`:

```json
[
  {
    "id": "edits",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/edits",
    "caption_strategy": "textfile",
    "resolution": 512,
    "resolution_type": "pixel",
    "conditioning_data": ["references"],
    "cache_dir_vae": "cache/vae/flux3-image/edits",
    "text_embeds": "text"
  },
  {
    "id": "references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "datasets/references",
    "conditioning_type": "reference_loose",
    "resolution": 512,
    "resolution_type": "pixel",
    "cache_dir_vae": "cache/vae/flux3-image/references"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux3-image-edit"
  }
]
```

Los nombres deben coincidir: por ejemplo, `edits/sample.png`, `edits/sample.txt` y `references/sample.png`. `reference_loose` permite tamaños y proporciones diferentes; `reference_strict` conserva recortes alineados. Añade más IDs en `conditioning_data` para usar hasta diez referencias por muestra; `conditioning_multidataset_sampling: "combined"` las proporciona juntas. Sin `conditioning_data`, se entrena texto a imagen. Separa las cachés de referencias, objetivos y otras familias.

Con `conditioning_multidataset_sampling: "random"`, cada muestra de entrenamiento utiliza un conjunto de referencias y el texto de ese conjunto. Establece `caption_strategy: "textfile"` en cada conjunto de referencias y escribe la instrucción de edición en el archivo `.txt` correspondiente, por ejemplo `references/sample.txt`.

## Genera una imagen con tu LoRA

Busca el directorio de checkpoint que contiene `pytorch_lora_weights.safetensors`. Sustituye `output/my-flux3-lora/checkpoint-1000` por ese directorio y usa la misma variante con la que entrenaste. Este ejemplo usa base y guía CFG-Zero-Star:

```python
import torch
from simpletuner.helpers.models.flux3_image.pipeline import Flux3ImagePipeline

pipe = Flux3ImagePipeline.from_pretrained(
    "black-forest-labs/flux-3-image-klein-base", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("output/my-flux3-lora/checkpoint-1000")
image = pipe(
    "A photo of TOK, a small ceramic owl, sitting on a wooden shelf",
    num_inference_steps=50,
    guidance_scale=3.5,
    use_cfg_zero_star=True,
).images[0]
image.save("my-flux3-image.png")
```

## LoRA asistente

El entrenamiento LoRA con `model_flavour: "klein"` carga automáticamente el [asistente publicado](https://huggingface.co/SimpleTuner/flux-3-klein-image-training-assistant) como adaptador congelado. `assistant_lora_strength` vale `1.0` durante el entrenamiento y `assistant_lora_inference_strength`, `0.0` en validación. Usa `disable_assistant_lora: true` para desactivarlo. Para otro adaptador, indica `assistant_lora_path` y, si hace falta, `assistant_lora_weight_name`.

Para entrenar tu propio asistente, selecciona `flux3_image-assistant-lora.peft-lora`. Esta receta desactiva el asistente existente y aprende de muestras generadas por el modelo. Después, indica su directorio en `assistant_lora_path` dentro de tu configuración LoRA destilada.

Consulta los ajustes adicionales en la [referencia de opciones](../OPTIONS.es.md). Para TwinFlow, activa `twinflow_enabled: true` y `use_ema: true`; EMA mantiene el modelo profesor utilizado por este método.
