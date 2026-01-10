## Guía rápida de NVLabs Sana

En este ejemplo, haremos entrenamiento de rango completo del modelo NVLabs Sana.

### Requisitos de hardware

Sana es muy liviano y quizá ni siquiera necesite gradient checkpointing completo en una tarjeta de 24G, ¡lo que significa que entrena muy rápido!

- **el mínimo absoluto** es alrededor de 12G de VRAM, aunque esta guía puede que no te lleve hasta ahí totalmente
- **un mínimo realista** es una sola GPU 3090 o V100
- **idealmente** varias 4090, A6000, L40S o mejor

Sana tiene una arquitectura extraña respecto a otros modelos entrenables por SimpleTuner;

- Inicialmente, a diferencia de otros modelos, Sana requería entrenamiento fp16 y fallaba con bf16
  - Los autores del modelo en NVIDIA tuvieron la gentileza de publicar pesos compatibles con bf16 para ajuste fino
- La cuantización podría ser más sensible en esta familia de modelos debido a los problemas con bf16/fp16
- SageAttention no funciona con Sana (aún) debido a su forma de head_dim, actualmente no soportada
- El valor de pérdida al entrenar Sana es muy alto, y podría necesitar una tasa de aprendizaje mucho más baja que otros modelos (p. ej., `1e-5` o similar)
- El entrenamiento puede producir valores NaN, y no está claro por qué

El gradient checkpointing puede liberar VRAM, pero ralentiza el entrenamiento. Un gráfico de resultados de prueba en una 4090 con 5800X3D:

![image](https://github.com/user-attachments/assets/310bf099-a077-4378-acf4-f60b4b82fdc4)

El código de modelado de Sana en SimpleTuner permite especificar `--gradient_checkpointing_interval` para hacer checkpoint cada _n_ bloques y lograr los resultados vistos en el gráfico anterior.

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

- `model_type` - Configúralo en `full`.
- `model_family` - Configúralo en `sana`.
- `pretrained_model_name_or_path` - Configúralo en `terminusresearch/sana-1.6b-1024px`
- `output_dir` - Configúralo al directorio donde quieres guardar tus checkpoints y las imágenes de validación. Se recomienda usar una ruta completa aquí.
- `train_batch_size` - para una tarjeta de 24G con gradient checkpointing completo, puede ser tan alto como 6.
- `validation_resolution` - Este checkpoint de Sana es un modelo 1024px, debes configurarlo en `1024x1024` o una de las otras resoluciones compatibles de Sana.
  - Se pueden especificar otras resoluciones separándolas con comas: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - Usa el valor que suelas seleccionar en inferencia para Sana.
- `validation_num_inference_steps` - Usa alrededor de 50 para la mejor calidad, aunque puedes aceptar menos si estás conforme con los resultados.
- `use_ema` - Configurar esto en `true` ayudará mucho a obtener un resultado más suavizado junto con tu checkpoint principal.

- `optimizer` - Puedes usar cualquier optimizador con el que te sientas cómodo, pero usaremos `optimi-adamw` para este ejemplo.
- `mixed_precision` - Se recomienda configurarlo en `bf16` para la configuración de entrenamiento más eficiente, o `no` (pero consumirá más memoria y será más lento).
  - Un valor de `fp16` no se recomienda aquí pero puede ser requerido para ciertos ajustes de Sana (e introduce otros problemas nuevos)
- `gradient_checkpointing` - Desactivarlo será lo más rápido, pero limita el tamaño de los lotes. Es necesario activarlo para obtener el menor uso de VRAM.
- `gradient_checkpointing_interval` - Si `gradient_checkpointing` se siente demasiado agresivo en tu GPU, puedes configurarlo en un valor de 2 o mayor para hacer checkpoint cada _n_ bloques. Un valor de 2 haría checkpoint de la mitad de los bloques, y 3 de un tercio.

Usuarios multi-GPU pueden consultar [este documento](../OPTIONS.md#environment-configuration-variables) para información sobre cómo configurar la cantidad de GPUs a usar.

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

> ℹ️ Sana usa una configuración de codificador de texto extraña que significa que prompts más cortos podrían verse muy mal.

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

#### Desplazamiento del calendario temporal de Sana

Los modelos de flow-matching como Sana, Flux y SD3 tienen una propiedad llamada "shift" que permite desplazar la parte entrenada del calendario de timesteps usando un simple valor decimal.

##### Auto-shift
Un enfoque comúnmente recomendado es seguir varios trabajos recientes y habilitar el shift de timesteps dependiente de la resolución, `--flow_schedule_auto_shift`, que usa valores de shift más altos para imágenes grandes y valores más bajos para imágenes pequeñas. Esto da resultados de entrenamiento estables pero potencialmente mediocres.

##### Especificación manual
_Gracias a General Awareness de Discord por los siguientes ejemplos_

Al usar un valor `--flow_schedule_shift` de 0.1 (muy bajo), solo se ven afectados los detalles finos de la imagen:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

Al usar un valor `--flow_schedule_shift` de 4.0 (muy alto), se ven afectados los grandes rasgos compositivos y potencialmente el espacio de color del modelo:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)

#### Consideraciones del dataset

> ⚠️ La calidad de imagen para el entrenamiento es más importante para Sana que para la mayoría de los otros modelos, ya que absorberá los artefactos de tus imágenes *primero*, y luego aprenderá el concepto/sujeto.

Es crucial contar con un dataset sustancial para entrenar tu modelo. Hay limitaciones en el tamaño del dataset, y debes asegurarte de que sea lo suficientemente grande para entrenar tu modelo de forma efectiva. Ten en cuenta que el tamaño mínimo del dataset es `train_batch_size * gradient_accumulation_steps` y también mayor que `vae_batch_size`. El dataset no será utilizable si es demasiado pequeño.

> ℹ️ Con pocas imágenes, podrías ver el mensaje **no images detected in dataset** - aumentar el valor de `repeats` superará esta limitación.

Dependiendo del dataset que tengas, necesitarás configurar el directorio del dataset y el archivo de configuración del dataloader de manera diferente. En este ejemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

Crea un documento `--data_backend_config` (`config/multidatabackend.json`) que contenga esto:

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sana",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/sana/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/sana/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "dreambooth-subject-512",
    "type": "local",
    "crop": false,
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/sana/dreambooth-subject-512",
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
    "cache_dir": "cache/text/sana",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

> ℹ️ Se admite ejecutar datasets de 512px y 1024px de forma concurrente, y podría resultar en mejor convergencia para Sana.

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

Para más información, consulta los documentos de [dataloader](../DATALOADER.md) y [tutorial](../TUTORIAL.md) documents.

## Notas y consejos de solución de problemas

### Configuración de VRAM más baja

Actualmente, el uso de VRAM más bajo se puede lograr con:

- OS: Ubuntu Linux 24
- GPU: Un solo dispositivo NVIDIA CUDA (10G, 12G)
- Memoria del sistema: alrededor de 50G de memoria del sistema
- Precisión del modelo base: `nf4-bnb`
- Optimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolución: 1024px
- Tamaño de lote: 1, cero pasos de acumulación de gradiente
- DeepSpeed: deshabilitado / sin configurar
- PyTorch: 2.5.1
- Usar `--quantize_via=cpu` para evitar error outOfMemory durante el arranque en tarjetas <=16G.
- Habilitar `--gradient_checkpointing`

**NOTA**: El pre-caché de embeddings del VAE y salidas del codificador de texto puede usar más memoria y aún así producir OOM. Si sucede, se puede habilitar cuantización del codificador de texto. El tiling del VAE puede no funcionar para Sana por ahora. Para datasets grandes donde el espacio en disco es un problema, puedes usar `--vae_cache_disable` para codificar en línea sin cachear a disco.

La velocidad fue de aproximadamente 1.4 iteraciones por segundo en una 4090.

### Pérdida con máscara

Si estás entrenando un sujeto o estilo y te gustaría enmascarar uno u otro, consulta la sección de [entrenamiento con pérdida enmascarada](../DREAMBOOTH.md#masked-loss) de la guía de Dreambooth.

### Cuantización

No probado a fondo (aún).

### Tasas de aprendizaje

#### LoRA (--lora_type=standard)

*No compatible.*

#### LoKr (--lora_type=lycoris)
- Tasas de aprendizaje suaves son mejores para LoKr (`1e-4` con AdamW, `2e-5` con Lion)
- Otros algoritmos necesitan más exploración.
- Configurar `is_regularisation_data` tiene impacto/efecto desconocido con Sana (no probado)

### Artefactos de imagen

Sana tiene una respuesta desconocida a artefactos de imagen.

Actualmente no se sabe si se producirán artefactos comunes de entrenamiento ni cuál podría ser la causa.

Si surge algún problema de calidad de imagen, por favor abre un issue en Github.

### Aspect bucketing

Este modelo tiene una respuesta desconocida a datos bucketed por aspecto. La experimentación será útil.
