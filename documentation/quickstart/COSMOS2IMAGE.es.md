## Guía rápida de Cosmos2 Predict (Image)

En este ejemplo, entrenaremos un Lycoris LoKr para Cosmos2 Predict (Image), un modelo de flow-matching de NVIDIA.

### Requisitos de hardware

Cosmos2 Predict (Image) es un modelo basado en vision transformer que usa flow matching.

**Nota**: Debido a su arquitectura, realmente no debería cuantizarse durante el entrenamiento, lo que significa que necesitarás VRAM suficiente para admitir la precisión completa en bf16.

Se recomienda una GPU de 24GB como mínimo para un entrenamiento cómodo sin optimizaciones extensas.

### Offloading de memoria (opcional)

Para encajar Cosmos2 en GPUs más pequeñas, habilita el offloading agrupado:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Los streams solo se respetan en CUDA; otros dispositivos hacen fallback automáticamente.
- No combines esto con `--enable_model_cpu_offload`.
- El staging en disco es opcional y ayuda cuando la RAM del sistema es el cuello de botella.

### Requisitos previos

Asegúrate de tener python instalado; SimpleTuner funciona bien con 3.10 a 3.12.

Puedes comprobarlo ejecutando:

```bash
python --version
```

Si no tienes python 3.12 instalado en Ubuntu, puedes intentar lo siguiente:

```bash
apt -y install python3.13 python3.13-venv
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

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
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
- `model_family` - Configúralo en `cosmos2image`.
- `base_model_precision` - **Importante**: Configúralo en `no_change` - Cosmos2 no debe cuantizarse.
- `output_dir` - Configúralo al directorio donde quieres guardar tus checkpoints y las imágenes de validación. Se recomienda usar una ruta completa aquí.
- `train_batch_size` - Comienza con 1 y aumenta si tienes VRAM suficiente.
- `validation_resolution` - El valor predeterminado es `1024x1024` para Cosmos2.
  - Se pueden especificar otras resoluciones separándolas con comas: `1024x1024,768x768`
- `validation_guidance` - Usa un valor alrededor de 4.0 para Cosmos2.
- `validation_num_inference_steps` - Usa alrededor de 20 pasos.
- `use_ema` - Configurar esto en `true` ayudará mucho a obtener un resultado más suavizado junto con tu checkpoint principal.
- `optimizer` - El ejemplo usa `adamw_bf16`.
- `mixed_precision` - Se recomienda configurarlo en `bf16` para la configuración de entrenamiento más eficiente.
- `gradient_checkpointing` - Habilítalo para reducir el uso de VRAM a costa de velocidad de entrenamiento.

Tu config.json se verá algo así:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "base_model_precision": "no_change",
    "checkpoint_step_interval": 500,
    "data_backend_config": "config/cosmos2image/multidatabackend.json",
    "disable_bucket_pruning": true,
    "flow_schedule_shift": 0.0,
    "flow_schedule_auto_shift": true,
    "gradient_checkpointing": true,
    "hub_model_id": "cosmos2image-lora",
    "learning_rate": 6e-5,
    "lora_type": "lycoris",
    "lycoris_config": "config/cosmos2image/lycoris_config.json",
    "lr_scheduler": "constant",
    "lr_warmup_steps": 100,
    "max_train_steps": 10000,
    "model_family": "cosmos2image",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/cosmos2image",
    "push_checkpoints_to_hub": false,
    "push_to_hub": false,
    "quantize_via": "cpu",
    "report_to": "tensorboard",
    "seed": 42,
    "tracker_project_name": "cosmos2image-training",
    "tracker_run_name": "cosmos2image-lora",
    "train_batch_size": 1,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_disable_unconditional": true,
    "validation_guidance": 4.0,
    "validation_guidance_rescale": 0.0,
    "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
    "validation_num_inference_steps": 20,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_prompt_library": false,
    "validation_resolution": "512x512",
    "validation_seed": 42,
    "validation_step_interval": 500
}
```
</details>

> ℹ️ Usuarios multi-GPU pueden consultar [este documento](../OPTIONS.md#environment-configuration-variables) para información sobre cómo configurar la cantidad de GPUs a usar.

Y un archivo `config/cosmos2image/lycoris_config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "bypass_mode": true,
    "algo": "lokr",
    "multiplier": 1.0,
    "full_matrix": true,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 4,
    "apply_preset": {
        "target_module": [
            "Attention"
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 4
            }
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

Para indicar al entrenador esta librería de prompts, agrégala a tu config configurando:
```json
"validation_prompt_library": "config/user_prompt_library.json"
```

Un conjunto de prompts diverso ayudará a determinar si el modelo colapsa a medida que entrena. En este ejemplo, la palabra `<token>` debe reemplazarse por el nombre de tu sujeto (instance_prompt).

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```

#### Seguimiento de puntuaciones CLIP

Si deseas habilitar evaluaciones para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/CLIP_SCORES.md) para información sobre cómo configurar e interpretar las puntuaciones CLIP.

#### Vistas previas de validación

SimpleTuner admite la transmisión de vistas previas de validación intermedias durante la generación usando modelos Tiny AutoEncoder. Esto te permite ver las imágenes de validación generándose paso a paso en tiempo real mediante callbacks de webhook.

Para habilitarlo:
```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

**Requisitos:**
- Configuración de webhook
- Validación habilitada

Configura `validation_preview_steps` en un valor más alto (p. ej., 3 o 5) para reducir la sobrecarga del Tiny AutoEncoder. Con `validation_num_inference_steps=20` y `validation_preview_steps=5`, recibirás imágenes de vista previa en los pasos 5, 10, 15 y 20.

#### Desplazamiento del calendario de flujo

Como modelo de flow-matching, Cosmos2 tiene una propiedad llamada "shift" que permite desplazar la parte entrenada del calendario de timesteps usando un simple valor decimal.

La configuración incluye `flow_schedule_auto_shift` habilitado por defecto, que usa el shift de timesteps dependiente de la resolución: valores de shift más altos para imágenes grandes, y valores más bajos para imágenes pequeñas.

#### Consideraciones del dataset

Es crucial contar con un dataset sustancial para entrenar tu modelo. Hay limitaciones en el tamaño del dataset, y debes asegurarte de que sea lo suficientemente grande para entrenar tu modelo de forma efectiva. Ten en cuenta que el tamaño mínimo del dataset es `train_batch_size * gradient_accumulation_steps` y también mayor que `vae_batch_size`. El dataset no será utilizable si es demasiado pequeño.

> ℹ️ Con pocas imágenes, podrías ver el mensaje **no images detected in dataset** - aumentar el valor de `repeats` superará esta limitación.

Dependiendo del dataset que tengas, necesitarás configurar el directorio del dataset y el archivo de configuración del dataloader de manera diferente. En este ejemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

Crea un documento `--data_backend_config` (`config/cosmos2image/multidatabackend.json`) que contenga esto:

```json
[
  {
    "id": "pseudo-camera-10k-cosmos2",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/cosmos2/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/cosmos2/dreambooth-subject",
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
    "cache_dir": "cache/text/cosmos2",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

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

</details>

### Ejecutar el entrenamiento

Desde el directorio de SimpleTuner, tienes varias opciones para iniciar el entrenamiento:

**Opción 1 (Recomendada - instalación con pip):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
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

Como Cosmos2 es un modelo más nuevo con documentación limitada, los ejemplos de inferencia pueden necesitar ajustes. Una estructura básica de ejemplo sería:

<details>
<summary>Mostrar ejemplo de inferencia en Python</summary>

```py
import torch
from lycoris import create_lycoris_from_weights

# Model and adapter paths
model_id = 'nvidia/Cosmos-1.0-Predict-Image-Text2World-12B'
adapter_repo_id = 'your-username/your-cosmos2-lora'
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

# Load the model and adapter

import torch
from diffusers import Cosmos2TextToImagePipeline

# Available checkpoints: nvidia/Cosmos-Predict2-2B-Text2Image, nvidia/Cosmos-Predict2-14B-Text2Image
model_id = "nvidia/Cosmos-Predict2-2B-Text2Image"
adapter_repo_id = "youruser/your-repo-name"

adapter_file_path = download_adapter(repo_id=adapter_repo_id)
pipe = Cosmos2TextToImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

pipe.to("cuda")

prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

output = pipe(
    prompt=prompt, negative_prompt=negative_prompt, generator=torch.Generator().manual_seed(1)
).images[0]
output.save("output.png")

```

</details>

## Notas y consejos de solución de problemas

### Consideraciones de memoria

Como Cosmos2 no puede cuantizarse durante el entrenamiento, el uso de memoria será mayor que en modelos cuantizados. Ajustes clave para menor uso de VRAM:

- Habilitar `gradient_checkpointing`
- Usar tamaño de lote 1
- Considerar usar el optimizador `adamw_8bit` si la memoria es ajustada
- Configurar la variable de entorno `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ayuda a minimizar el uso de VRAM al entrenar múltiples relaciones de aspecto
- Usa `--vae_cache_disable` para realizar codificación VAE en línea sin caché a disco, lo que puede ahorrar espacio en disco pero aumenta el tiempo de entrenamiento y la presión de memoria.

### Consideraciones de entrenamiento

Como Cosmos2 es un modelo más nuevo, los parámetros de entrenamiento óptimos aún se están explorando:

- El ejemplo usa una tasa de aprendizaje de `6e-5` con AdamW
- El auto-shift de flow schedule está habilitado para manejar diferentes resoluciones
- Se usa evaluación CLIP para monitorear el progreso del entrenamiento

### Aspect bucketing

La configuración tiene `disable_bucket_pruning` en true, lo cual puede ajustarse según las características de tu dataset.

### Entrenamiento de múltiples resoluciones

El modelo puede entrenarse inicialmente a 512px, con posibilidad de entrenar a resoluciones más altas después. El ajuste `flow_schedule_auto_shift` ayuda con entrenamiento de múltiples resoluciones.

### Pérdida con máscara

Si estás entrenando un sujeto o estilo y te gustaría enmascarar uno u otro, consulta la sección de [entrenamiento con pérdida enmascarada](../DREAMBOOTH.md#masked-loss) de la guía de Dreambooth.

### Limitaciones conocidas

- El manejo del system prompt aún no está implementado
- Las características de entrenabilidad del modelo todavía se están explorando
- La cuantización no está soportada y debe evitarse
