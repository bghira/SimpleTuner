## Guía rápida de Auraflow

En este ejemplo, entrenaremos un Lycoris LoKr para Auraflow.

El ajuste fino completo para este modelo requerirá mucha VRAM debido a sus 6B parámetros, y tendrías que usar [DeepSpeed](../DEEPSPEED.md) para que funcione.

### Requisitos de hardware

Auraflow v0.3 se lanzó como un MMDiT de 6B parámetros que usa Pile T5 para su representación de texto codificada y el VAE SDXL de 4 canales para su representación de imagen latente.

Este modelo es algo lento para inferencia, pero entrena a una velocidad decente.

### Offloading de memoria (opcional)

Auraflow se beneficia mucho de la nueva ruta de offloading agrupado. Agrega lo siguiente a tus flags de entrenamiento si estás limitado a una sola GPU de 24G (o menor):

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Los streams se desactivan automáticamente en backends no CUDA, así que el comando es seguro para reutilizar en ROCm y MPS.
- No combines esto con `--enable_model_cpu_offload`.
- El offload a disco reduce el throughput para bajar la presión de RAM del host; mantenlo en un SSD local para mejores resultados.

### Requisitos previos

Asegúrate de tener python instalado; SimpleTuner funciona bien con 3.10 a 3.12.

Puedes comprobarlo ejecutando:

```bash
python --version
```

Si no tienes python 3.12 instalado en Ubuntu, puedes intentar lo siguiente:

```bash
apt -y install python3.12 python3.12-venv
```

#### Dependencias de la imagen de contenedor

Para Vast, RunPod y TensorDock (entre otros), lo siguiente funcionará en una imagen CUDA 12.2-12.8 para habilitar la compilación de extensiones CUDA:

```bash
apt -y install nvidia-cuda-toolkit
```

### Instalación

Instala SimpleTuner vía pip:

```bash
pip install 'simpletuner[cuda]'
```

Para instalación manual o entorno de desarrollo, consulta la [documentación de instalación](../INSTALL.md).

### Configuración del entorno

Para ejecutar SimpleTuner, necesitas configurar un archivo de configuración, los directorios del dataset y del modelo, y un archivo de configuración del dataloader.

#### Archivo de configuración

Un script experimental, `configure.py`, puede permitirte omitir por completo esta sección mediante una configuración interactiva paso a paso. Incluye funciones de seguridad que ayudan a evitar errores comunes.

**Nota:** Esto no configura tu dataloader. Aún tendrás que hacerlo manualmente más adelante.

Para ejecutarlo:

```bash
simpletuner configure
```

> ⚠️ Para usuarios ubicados en países donde Hugging Face Hub no es fácilmente accesible, debes agregar `HF_ENDPOINT=https://hf-mirror.com` a tu `~/.bashrc` o `~/.zshrc` dependiendo de cuál `$SHELL` use tu sistema.


Si prefieres configurarlo manualmente:

Copia `config/config.json.example` a `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Allí, probablemente necesitarás modificar las siguientes variables:

- `model_type` - Configúralo en `lora`.
- `lora_type` - Configúralo en `lycoris`.
- `model_family` - Configúralo en `auraflow`.
- `model_flavour` - Configúralo en `pony`, o déjalo sin definir para usar el modelo predeterminado.
- `output_dir` - Configúralo al directorio donde quieres guardar tus checkpoints y las imágenes de validación. Se recomienda usar una ruta completa aquí.
- `train_batch_size` - 1 a 4 debería funcionar para una tarjeta de 24G.
- `validation_resolution` - Debes configurarlo a `1024x1024` o a otra de las resoluciones compatibles de Auraflow.
  - Se pueden especificar otras resoluciones separándolas con comas: `1024x1024,1280x768,1536x1536`
  - Ten en cuenta que los positional embeds de Auraflow son un poco extraños y entrenar con imágenes multi-escala (múltiples resoluciones base) tiene un resultado incierto.
- `validation_guidance` - Usa el valor que suelas elegir en inferencia para Auraflow; un valor más bajo alrededor de 3.5-4.0 da resultados más realistas.
- `validation_num_inference_steps` - Usa un valor alrededor de 30-50.
- `use_ema` - establecer esto en `true` ayudará mucho a obtener un resultado más suavizado junto con tu checkpoint principal.

- `optimizer` - Puedes usar cualquier optimizador con el que te sientas cómodo, pero usaremos `optimi-lion` para este ejemplo.
  - El autor de Pony Flow recomienda usar `adamw_bf16` para tener menos problemas y resultados de entrenamiento más estables y confiables.
  - Usamos Lion en esta demostración para ayudarte a ver que el modelo entrena más rápido, pero para ejecuciones largas, `adamw_bf16` es una apuesta segura.
- `learning_rate` - Para el optimizador Lion con Lycoris LoKr, un valor de `4e-5` es un buen punto de partida.
  - Si usaste `adamw_bf16`, querrás que el LR sea aproximadamente 10 veces mayor, o `2.5e-4`.
  - Rangos Lycoris/LoRA más pequeños requieren **tasas de aprendizaje más altas** y rangos Lycoris/LoRA más grandes requieren **tasas de aprendizaje más bajas**.
- `mixed_precision` - Se recomienda configurarlo en `bf16` para la configuración de entrenamiento más eficiente, o `no` para mejores resultados (pero consumirá más memoria y será más lento).
- `gradient_checkpointing` - Desactivarlo será lo más rápido, pero limita el tamaño del lote. Es necesario activarlo para obtener el menor uso de VRAM.


El impacto de estas opciones es actualmente desconocido.

Tu config.json se verá algo así al final:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "validation_torch_compile": "false",
    "validation_step_interval": 200,
    "validation_seed": 42,
    "validation_resolution": "1024x1024",
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_num_inference_steps": "20",
    "validation_guidance": 2.0,
    "validation_guidance_rescale": "0.0",
    "vae_cache_ondemand": true,
    "vae_batch_size": 1,
    "train_batch_size": 1,
    "tracker_run_name": "eval_loss_test1",
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "output_dir": "output/models-auraflow",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "auraflow",
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 10000,
    "max_grad_norm": 0.01,
    "lycoris_config": "config/lycoris_config.json",
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant",
    "lora_type": "lycoris",
    "learning_rate": "4e-5",
    "gradient_checkpointing": "true",
    "grad_clip_method": "value",
    "eval_steps_interval": 100,
    "disable_benchmark": false,
    "data_backend_config": "config/auraflow/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "int8-quanto",
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ Usuarios multi-GPU pueden consultar [este documento](../OPTIONS.md#environment-configuration-variables) para información sobre cómo configurar la cantidad de GPUs a usar.

Y un archivo simple `config/lycoris_config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 16,
    "apply_preset": {
        "target_module": [
            "Attention",
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 8
            },
        }
    }
}
```
</details>

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

#### Prompts de validación

Dentro de `config/config.json` está el "prompt de validación principal", que suele ser el instance_prompt principal en el que estás entrenando para tu único sujeto o estilo. Además, se puede crear un archivo JSON que contiene prompts adicionales para ejecutar durante las validaciones.

El archivo de ejemplo `config/user_prompt_library.json.example` tiene el siguiente formato:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

Los apodos son el nombre de archivo de la validación, así que mantenlos cortos y compatibles con tu sistema de archivos.

Para indicar al entrenador esta librería de prompts, añádela a TRAINER_EXTRA_ARGS agregando una nueva línea al final de `config.json`:
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

Un conjunto de prompts diverso ayudará a determinar si el modelo colapsa a medida que entrena. En este ejemplo, la palabra `<token>` debe reemplazarse por el nombre de tu sujeto (instance_prompt).

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing her essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing her unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing her dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```

> ℹ️ Auraflow usa por defecto 128 tokens y luego trunca.

#### Seguimiento de puntuaciones CLIP

Si deseas habilitar evaluaciones para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/CLIP_SCORES.md) para información sobre cómo configurar e interpretar las puntuaciones CLIP.

</details>

# Pérdida de evaluación estable

Si deseas usar pérdida MSE estable para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/EVAL_LOSS.md) para información sobre cómo configurar e interpretar la pérdida de evaluación.

#### Vistas previas de validación

SimpleTuner admite la transmisión de vistas previas de validación intermedias durante la generación usando modelos Tiny AutoEncoder. Esto te permite ver las imágenes de validación generándose paso a paso en tiempo real mediante callbacks de webhook.

Para habilitarlo:
<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```
</details>

**Requisitos:**
- Configuración de webhook
- Validación habilitada

Configura `validation_preview_steps` en un valor más alto (p. ej., 3 o 5) para reducir la sobrecarga del Tiny AutoEncoder. Con `validation_num_inference_steps=20` y `validation_preview_steps=5`, recibirás imágenes de vista previa en los pasos 5, 10, 15 y 20.

#### Desplazamiento del calendario de flujo

Los modelos de flow-matching como OmniGen, Sana, Flux y SD3 tienen una propiedad llamada "shift" que permite desplazar la parte entrenada del calendario de timesteps usando un simple valor decimal.

##### Auto-shift

Un enfoque comúnmente recomendado es seguir varios trabajos recientes y habilitar el shift de timesteps dependiente de la resolución, `--flow_schedule_auto_shift`, que usa valores de shift más altos para imágenes grandes y valores más bajos para imágenes pequeñas. Esto da resultados de entrenamiento estables pero potencialmente mediocres.

##### Especificación manual

_Gracias a General Awareness de Discord por los siguientes ejemplos_

Al usar un valor `--flow_schedule_shift` de 0.1 (muy bajo), solo se ven afectados los detalles finos de la imagen:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

Al usar un valor `--flow_schedule_shift` de 4.0 (muy alto), se ven afectados los grandes rasgos compositivos y potencialmente el espacio de color del modelo:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### Consideraciones del dataset

Es crucial contar con un dataset sustancial para entrenar tu modelo. Hay limitaciones en el tamaño del dataset, y debes asegurarte de que sea lo suficientemente grande para entrenar tu modelo de forma efectiva. Ten en cuenta que el tamaño mínimo del dataset es `train_batch_size * gradient_accumulation_steps` y también mayor que `vae_batch_size`. El dataset no será utilizable si es demasiado pequeño.

> ℹ️ Con pocas imágenes, podrías ver el mensaje **no images detected in dataset** - aumentar el valor de `repeats` superará esta limitación.

Dependiendo del dataset que tengas, necesitarás configurar el directorio del dataset y el archivo de configuración del dataloader de manera diferente. En este ejemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

Crea un documento `--data_backend_config` (`config/multidatabackend.json`) que contenga esto:

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-auraflow",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/auraflow/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/auraflow/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/auraflow",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> ℹ️ Usa `caption_strategy=textfile` si tienes archivos `.txt` que contienen captions.
> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

Luego, crea un directorio `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

Esto descargará alrededor de 10k muestras de fotografías a tu directorio `datasets/pseudo-camera-10k`, que se creará automáticamente.

Tus imágenes de Dreambooth deben ir en el directorio `datasets/dreambooth-subject`.

#### Iniciar sesión en WandB y Huggingface Hub

Querrás iniciar sesión en WandB y HF Hub antes de empezar el entrenamiento, especialmente si usas `--push_to_hub` y `--report_to=wandb`.

Si vas a subir elementos a un repositorio Git LFS manualmente, también deberías ejecutar `git config --global credential.helper store`

Ejecuta los siguientes comandos:

```bash
wandb login
```

y

```bash
huggingface-cli login
```

Sigue las instrucciones para iniciar sesión en ambos servicios.

### Ejecutar el entrenamiento

Desde el directorio de SimpleTuner, tienes varias opciones para iniciar el entrenamiento:

**Opción 1 (Recomendada - instalación con pip):**
```bash
pip install 'simpletuner[cuda]'
simpletuner train
```

**Opción 2 (Método de git clone):**
```bash
simpletuner train
```

**Opción 3 (Método heredado - aún funciona):**
```bash
./train.sh
```

Esto iniciará el caché a disco de text embeds y salidas del VAE.

Para más información, consulta los documentos de [dataloader](../DATALOADER.md) y [tutorial](../TUTORIAL.md).

### Ejecutar inferencia en el LoKr después

Como es un modelo nuevo, el ejemplo necesita algunos ajustes para funcionar. Aquí hay un ejemplo funcional:

<details>
<summary>Mostrar ejemplo de inferencia en Python</summary>

```py
import torch
from helpers.models.auraflow.pipeline import AuraFlowPipeline
from helpers.models.auraflow.transformer import AuraFlowTransformer2DModel
from lycoris import create_lycoris_from_weights
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

model_id = 'terminusresearch/auraflow-v0.3'
adapter_repo_id = 'bghira/auraflow-photo-1mp-Prodigy'
adapter_filename = 'pytorch_lora_weights.safetensors'

def download_adapter(repo_id: str):
    import os
    from huggingface_hub import hf_hub_download
    adapter_filename = "pytorch_lora_weights.safetensors"
    cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
    cleaned_adapter_path = repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
    path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
    os.makedirs(path_to_adapter, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
    )

    return path_to_adapter_file

adapter_file_path = download_adapter(repo_id=adapter_repo_id)
transformer = AuraFlowTransformer2DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, subfolder="transformer")
pipeline = AuraFlowPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    transformer=transformer,
)
lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

prompt = "Place your test prompt here."
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

## Optional: quantise the model to save on vram.
## Note: The model was quantised during training, and so it is recommended to do the same during inference time.
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # the pipeline is already in its target precision level
t5_embeds, negative_t5_embeds, attention_mask, negative_attention_mask = pipeline.encode_prompt(
    prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt, num_images_per_prompt=1
)
# We'll nuke the text encoders to save memory.
pipeline.text_encoder.to("meta")
pipeline.text_encoder_2.to("meta")
pipeline.text_encoder_3.to("meta")
model_output = pipeline(
    prompt_embeds=t5_embeds,
    prompt_attention_mask=attention_mask,
    negative_prompt_embeds=negative_t5_embeds,
    negative_prompt_attention_mask=negative_attention_mask,
    num_inference_steps=30,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=3.2,
).images[0]

model_output.save("output.png", format="PNG")

```
</details>

## Notas y consejos de solución de problemas

### Configuración de VRAM más baja

La configuración de VRAM más baja para Auraflow es de alrededor de 20-22G:

- OS: Ubuntu Linux 24
- GPU: Un solo dispositivo NVIDIA CUDA (10G, 12G)
- Memoria del sistema: alrededor de 50G de memoria del sistema (podría ser más o menos)
- Precisión del modelo base:
  - Para sistemas Apple y AMD, `int8-quanto` (o `fp8-torchao`, `int8-torchao` siguen perfiles de uso de memoria similares)
    - `int4-quanto` también funciona, pero podrías tener menor precisión / peores resultados
  - Para sistemas NVIDIA, se reporta que `nf4-bnb` funciona bien, pero será más lento que `int8-quanto`
- Optimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolución: 1024px
- Tamaño de lote: 1, cero pasos de acumulación de gradiente
- DeepSpeed: deshabilitado / sin configurar
- PyTorch: 2.7+
- Usar `--quantize_via=cpu` para evitar error outOfMemory durante el arranque en tarjetas <=16G.
- Habilitar `--gradient_checkpointing`
- Usar una configuración LoRA o Lycoris pequeña (p. ej., rango LoRA 1 o factor Lokr 25)

**NOTA**: El pre-caché de embeddings del VAE y salidas del codificador de texto puede usar más memoria y aun así producir OOM. El tiling y slicing del VAE están habilitados por defecto. Si ves OOM, podrías necesitar habilitar `offload_during_startup=true`; de lo contrario, puede que no haya suerte.

La velocidad fue de aproximadamente 3 iteraciones por segundo en una NVIDIA 4090 usando Pytorch 2.7 y CUDA 12.8

### Pérdida con máscara

Si estás entrenando un sujeto o estilo y te gustaría enmascarar uno u otro, consulta la sección de [entrenamiento con pérdida enmascarada](../DREAMBOOTH.md#masked-loss) de la guía de Dreambooth.

### Cuantización

Auraflow tiende a responder bien hasta el nivel de precisión `int4`, aunque `int8` será un punto óptimo para calidad y estabilidad si no puedes permitirte `bf16`.

### Tasas de aprendizaje

#### LoRA (--lora_type=standard)

*No compatible.*

#### LoKr (--lora_type=lycoris)
- Tasas de aprendizaje suaves son mejores para LoKr (`1e-4` con AdamW, `2e-5` con Lion)
- Otros algoritmos necesitan más exploración.
- Configurar `is_regularisation_data` tiene impacto/efecto desconocido con Auraflow (no probado, pero, ¿debería estar bien?)

### Artefactos de imagen

Auraflow tiene una respuesta desconocida a artefactos de imagen, aunque usa el VAE de Flux y tiene limitaciones similares de detalle fino.

Si surge algún problema de calidad de imagen, por favor abre un issue en Github.

### Aspect bucketing

Algunas limitaciones con la implementación de patch embed del modelo hacen que ciertas resoluciones provoquen un error.

La experimentación será útil, al igual que informes de bugs exhaustivos.

### Ajuste de rango completo

DeepSpeed usará MUCHA memoria del sistema con Auraflow, y el ajuste completo podría no rendir como esperas en términos de aprender conceptos o evitar el colapso del modelo.

Se recomienda Lycoris LoKr en lugar del ajuste de rango completo, ya que es más estable y tiene menor huella de memoria.
