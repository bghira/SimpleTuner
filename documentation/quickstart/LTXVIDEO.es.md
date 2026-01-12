## Guía rápida de LTX Video

En este ejemplo, entrenaremos un LoRA de LTX-Video usando el [dataset Disney de dominio público](https://hf.co/datasets/sayakpaul/video-dataset-disney-organized) de Sayak Paul.

### Requisitos de hardware

LTX no requiere mucha memoria del sistema **ni** de GPU.

Cuando entrenas cada componente de un LoRA de rango 16 (MLP, proyecciones, bloques multimodales), termina usando un poco más de 12G en un Mac M3 (batch size 4).

Necesitarás:
- **un mínimo realista**: 16GB o una sola GPU 3090 o V100
- **idealmente** varias 4090, A6000, L40S o mejor

Los sistemas Apple silicon funcionan muy bien con LTX hasta ahora, aunque a una resolución menor debido a límites en el backend MPS usado por Pytorch.

### Offloading de memoria (opcional)

Si estás cerca del límite de VRAM, habilita offloading agrupado en tu config:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Los usuarios de CUDA se benefician de `--group_offload_use_stream`; otros backends lo ignoran automáticamente.
- Omite `--group_offload_to_disk_path` a menos que la RAM del sistema sea <64 GB — el staging a disco es más lento pero mantiene ejecuciones estables.
- Desactiva `--enable_model_cpu_offload` al usar group offloading.

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

#### Pasos adicionales para AMD ROCm

Lo siguiente debe ejecutarse para que una AMD MI300X sea utilizable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

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
- `model_family` - Configúralo en `ltxvideo`.
- `pretrained_model_name_or_path` - Configúralo en `Lightricks/LTX-Video-0.9.5`.
- `pretrained_vae_model_name_or_path` - Configúralo en `Lightricks/LTX-Video-0.9.5`.
- `output_dir` - Configúralo al directorio donde quieres guardar tus checkpoints y las imágenes de validación. Se recomienda usar una ruta completa aquí.
- `train_batch_size` - esto puede aumentarse para más estabilidad, pero un valor de 4 debería funcionar bien para empezar
- `validation_resolution` - Debe configurarse a lo que normalmente generas con LTX (`768x512`)
  - Se pueden especificar múltiples resoluciones separándolas con comas: `1280x768,768x512`
- `validation_guidance` - Usa el valor que sueles seleccionar en inferencia para LTX.
- `validation_num_inference_steps` - Usa alrededor de 25 para ahorrar tiempo y aún ver calidad decente.
- `--lora_rank=4` si deseas reducir sustancialmente el tamaño del LoRA entrenado. Esto puede ayudar con el uso de VRAM mientras reduce su capacidad de aprendizaje.

- `gradient_accumulation_steps` - Esta opción hace que los pasos de actualización se acumulen durante varios pasos.
  - Esto aumentará el tiempo de entrenamiento linealmente; un valor de 2 hará que tu ejecución sea la mitad de rápida y tome el doble de tiempo.
- `optimizer` - A los principiantes se les recomienda quedarse con adamw_bf16, aunque optimi-lion y optimi-stableadamw también son buenas opciones.
- `mixed_precision` - Los principiantes deberían mantener esto en `bf16`
- `gradient_checkpointing` - configúralo en true prácticamente en todas las situaciones y en todos los dispositivos
- `gradient_checkpointing_interval` - aún no está soportado en LTX Video y debe eliminarse de tu config.

Usuarios multi-GPU pueden consultar [este documento](../OPTIONS.md#environment-configuration-variables) para información sobre cómo configurar la cantidad de GPUs a usar.

Al final, tu config debería parecerse a la mía:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "data_backend_config": "config/ltxvideo/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "disable_benchmark": false,
  "offload_during_startup": true,
  "output_dir": "output/ltxvideo",
  "lora_type": "lycoris",
  "lycoris_config": "config/ltxvideo/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "ltxvideo-disney",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "ltxvideo-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5",
  "model_family": "ltxvideo",
  "train_batch_size": 8,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 800,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "768x512",
  "validation_negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "validation_guidance": 3.0,
  "validation_num_inference_steps": 40,
  "validation_prompt": "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day. And then the camera switch to a inding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region.",
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.0,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "fp8-torchao",
  "vae_batch_size": 1,
  "webhook_config": "config/ltxvideo/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "lora_rank": 128,
  "flow_schedule_shift": 1,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

### Opcional: regularizador temporal CREPA

Si tus ejecuciones LTX muestran flicker o drift de identidad, prueba CREPA (alineación entre frames):
- En la WebUI, ve a **Training → Loss functions** y habilita **CREPA**.
- Empieza con **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- Deja el codificador de visión por defecto (`dinov2_vitg14`, tamaño `518`). Cambia a `dinov2_vits14` + `224` solo si necesitas menos VRAM.
- Requiere internet (o un torch hub en caché) la primera vez para obtener los pesos de DINOv2.
- Opcional: si entrenas solo desde latentes en caché, habilita **Drop VAE Encoder** para ahorrar memoria; déjalo desactivado si necesitas codificar videos nuevos.

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
    "anime_<token>": "a breathtaking anime-style video featuring <token>, capturing her essence with vibrant colors, dynamic motion, and expressive storytelling",
    "chef_<token>": "a high-quality, detailed video of <token> as a sous-chef, immersed in the art of culinary creation with captivating close-ups and engaging sequences",
    "just_<token>": "a lifelike and intimate video portrait of <token>, showcasing her unique personality and charm through nuanced movement and expression",
    "cinematic_<token>": "a cinematic, visually stunning video of <token>, emphasizing her dramatic and captivating presence through fluid camera movements and atmospheric effects",
    "elegant_<token>": "an elegant and timeless video portrait of <token>, exuding grace and sophistication with smooth transitions and refined visuals",
    "adventurous_<token>": "a dynamic and adventurous video featuring <token>, captured in an exciting, action-filled sequence that highlights her energy and spirit",
    "mysterious_<token>": "a mysterious and enigmatic video portrait of <token>, shrouded in shadows and intrigue with a narrative that unfolds in subtle, cinematic layers",
    "vintage_<token>": "a vintage-style video of <token>, evoking the charm and nostalgia of a bygone era through sepia tones and period-inspired visual storytelling",
    "artistic_<token>": "an artistic and abstract video representation of <token>, blending creativity with visual storytelling through experimental techniques and fluid visuals",
    "futuristic_<token>": "a futuristic and cutting-edge video portrayal of <token>, set against a backdrop of advanced technology with sleek, high-tech visuals",
    "woman": "a beautifully crafted video portrait of a woman, highlighting her natural beauty and unique features through elegant motion and storytelling",
    "man": "a powerful and striking video portrait of a man, capturing his strength and character with dynamic sequences and compelling visuals",
    "boy": "a playful and spirited video portrait of a boy, capturing youthful energy and innocence through lively scenes and engaging motion",
    "girl": "a charming and vibrant video portrait of a girl, emphasizing her bright personality and joy with colorful visuals and fluid movement",
    "family": "a heartwarming and cohesive family video, showcasing the bonds and connections between loved ones through intimate moments and shared experiences"
}
```

> ℹ️ LTX Video es un modelo de flow-matching basado en T5 XXL; los prompts más cortos podrían no tener suficiente información para que el modelo haga un buen trabajo. Asegúrate de usar prompts más largos y descriptivos.

#### Seguimiento de puntuaciones CLIP

Esto no debería habilitarse para entrenamiento de modelos de video por ahora.

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

#### Desplazamiento del calendario de flow-matching

Los modelos de flow-matching como Flux, Sana, SD3 y LTX Video tienen una propiedad llamada `shift` que permite desplazar la parte entrenada del calendario de timesteps usando un valor decimal simple.

##### Valores predeterminados
Por defecto, no se aplica shift de calendario, lo que da una forma de campana sigmoide a la distribución de muestreo de timesteps, también conocida como `logit_norm`.

##### Auto-shift
Un enfoque comúnmente recomendado es seguir varios trabajos recientes y habilitar el shift de timesteps dependiente de la resolución, `--flow_schedule_auto_shift`, que usa valores de shift más altos para imágenes grandes y valores más bajos para imágenes pequeñas. Esto da resultados de entrenamiento estables pero potencialmente mediocres.

##### Especificación manual
_Gracias a General Awareness de Discord por los siguientes ejemplos_

> ℹ️ Estos ejemplos muestran cómo funciona el valor usando Flux Dev, aunque LTX Video debería ser muy similar.

Al usar un valor `--flow_schedule_shift` de 0.1 (muy bajo), solo se ven afectados los detalles finos de la imagen:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

Al usar un valor `--flow_schedule_shift` de 4.0 (muy alto), se ven afectados los grandes rasgos compositivos y potencialmente el espacio de color del modelo:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### Entrenamiento con modelo cuantizado

Probado en sistemas Apple y NVIDIA, Hugging Face Optimum-Quanto puede usarse para reducir la precisión y los requisitos de VRAM, entrenando con solo 16GB.



Para usuarios de `config.json`:
<details>
<summary>Ver ejemplo de config</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

#### Consideraciones del dataset

Hay pocas limitaciones en el tamaño del dataset aparte de cuánto cómputo y tiempo tomará procesar y entrenar.

Debes asegurarte de que el dataset sea lo suficientemente grande para entrenar tu modelo de forma efectiva, pero no tan grande para el cómputo disponible.

Ten en cuenta que el tamaño mínimo del dataset es `train_batch_size * gradient_accumulation_steps` y también mayor que `vae_batch_size`. El dataset no será utilizable si es demasiado pequeño.

> ℹ️ Con pocas muestras, podrías ver el mensaje **no samples detected in dataset** - aumentar el valor de `repeats` superará esta limitación.

Dependiendo del dataset que tengas, necesitarás configurar el directorio del dataset y el archivo de configuración del dataloader de manera diferente.

En este ejemplo, usaremos [video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized) como dataset.

Crea un documento `--data_backend_config` (`config/multidatabackend.json`) que contenga esto:

<details>
<summary>Ver ejemplo de config</summary>

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
        "min_frames": 125,
        "bucket_strategy": "aspect_ratio"
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
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

- En la subsección `video`, tenemos las siguientes claves:
  - `num_frames` (opcional, int) es cuántos frames de datos entrenaremos.
    - A 25 fps, 125 frames son 5 segundos de video, salida estándar. Este debería ser tu objetivo.
  - `min_frames` (opcional, int) determina la longitud mínima de un video que se considerará para entrenamiento.
    - Esto debería ser al menos igual a `num_frames`. Si no se configura, será igual.
  - `max_frames` (opcional, int) determina la longitud máxima de un video que se considerará para entrenamiento.
  - `is_i2v` (opcional, bool) determina si se hará entrenamiento i2v en un dataset.
    - Esto se establece en True por defecto para LTX. Puedes desactivarlo, sin embargo.
  - `bucket_strategy` (opcional, string) determina cómo se agrupan los videos en buckets:
    - `aspect_ratio` (predeterminado): Agrupar solo por relación de aspecto espacial (p. ej., `1.78`, `0.75`).
    - `resolution_frames`: Agrupar por resolución y recuento de frames en formato `WxH@F` (p. ej., `768x512@125`). Útil para datasets de resolución/duración mixta.
  - `frame_interval` (opcional, int) al usar `resolution_frames`, redondea recuentos de frames a este intervalo. Configúralo al factor de recuento de frames requerido por tu modelo.

Luego, crea un directorio `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset sayakpaul/video-dataset-disney-organized --local-dir=disney-black-and-white
popd
```

Esto descargará todas las muestras de video de Disney a tu directorio `datasets/disney-black-and-white`, que se creará automáticamente.

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

Como otros modelos, es posible que el uso de VRAM más bajo se pueda lograr con:

- OS: Ubuntu Linux 24
- GPU: Un solo dispositivo NVIDIA CUDA (10G, 12G)
- Memoria del sistema: alrededor de 11G de memoria del sistema
- Precisión del modelo base: `nf4-bnb`
- Optimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolución: 480px
- Tamaño de lote: 1, cero pasos de acumulación de gradiente
- DeepSpeed: deshabilitado / sin configurar
- PyTorch: 2.6
- Asegúrate de habilitar `--gradient_checkpointing` o nada evitará que haga OOM

**NOTA**: El pre-caché de embeddings del VAE y salidas del codificador de texto puede usar más memoria y aun así producir OOM. Si sucede, se pueden habilitar cuantización del codificador de texto y tiling del VAE. Más allá de estas opciones, `--offload_during_startup=true` ayudará a evitar competencia entre el VAE y el uso de memoria del codificador de texto.

La velocidad fue de aproximadamente 0.8 iteraciones por segundo en un Macbook Pro M3 Max.

### SageAttention

Al usar `--attention_mechanism=sageattention`, la inferencia puede acelerarse en tiempo de validación.

**Nota**: Esto no es compatible con _todas_ las configuraciones de modelo, pero vale la pena probarlo.

### Entrenamiento cuantizado con NF4

En términos simples, NF4 es una representación de 4 bits _aprox_ del modelo, lo que significa que el entrenamiento tiene serias preocupaciones de estabilidad que abordar.

En pruebas tempranas, se cumple lo siguiente:
- El optimizador Lion provoca colapso del modelo pero usa menos VRAM; las variantes de AdamW ayudan a mantenerlo estable; bnb-adamw8bit, adamw_bf16 son excelentes opciones
  - AdEMAMix no fue bien, pero los ajustes no se exploraron
- `--max_grad_norm=0.01` ayuda aún más a reducir la ruptura del modelo al evitar cambios enormes en muy poco tiempo
- NF4, AdamW8bit y un tamaño de lote más alto ayudan a superar los problemas de estabilidad, a costa de más tiempo de entrenamiento o VRAM usada
- Subir la resolución ralentiza MUCHO el entrenamiento y podría perjudicar el modelo
- Incrementar la longitud de los videos consume mucha más memoria. Reduce `num_frames` para resolver esto.
- Todo lo que es difícil de entrenar en int8 o bf16 se vuelve más difícil en NF4
- Es menos compatible con opciones como SageAttention

NF4 no funciona con torch.compile, así que cualquier velocidad que obtengas es la que hay.

Si la VRAM no es un problema, entonces int8 con torch.compile es tu opción mejor y más rápida.

### Pérdida con máscara

No uses esto con LTX Video.


### Cuantización
- La cuantización no es necesaria para entrenar este modelo

### Artefactos de imagen
+Como otros modelos DiT, si haces estas cosas (entre otras) pueden comenzar a aparecer artefactos de cuadrícula cuadrada en las muestras:
- Sobreentrenar con datos de baja calidad
- Usar una tasa de aprendizaje demasiado alta
- Sobreentrenamiento (en general) de una red de baja capacidad con demasiadas imágenes
- Subentrenamiento (también) de una red de alta capacidad con muy pocas imágenes
- Usar relaciones de aspecto extrañas o tamaños de datos de entrenamiento

### Aspect bucketing
- Los videos se agrupan en buckets como las imágenes.
- Entrenar demasiado tiempo en recortes cuadrados probablemente no dañará demasiado este modelo. Dale con todo, es excelente y confiable.
- Por otro lado, usar los buckets de aspecto natural de tu dataset podría sesgar demasiado esas formas en inferencia.
+  - Esto podría ser una cualidad deseable, ya que evita que estilos dependientes de aspecto, como lo cinematográfico, se filtren a otras resoluciones demasiado.
+  - Sin embargo, si buscas mejorar resultados por igual en muchos buckets de aspecto, quizá tengas que experimentar con `crop_aspect=random`, lo cual tiene sus propias desventajas.
- Mezclar configuraciones de dataset definiendo tu dataset de imágenes múltiples veces ha dado resultados muy buenos y un modelo bien generalizado.

### Entrenar modelos LTX ajustados personalizados

Algunos modelos ajustados en Hugging Face Hub carecen de la estructura completa de directorios, lo que requiere configurar opciones específicas.

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "model_family": "ltxvideo",
    "pretrained_model_name_or_path": "Lightricks/LTX-Video",
    "pretrained_transformer_model_name_or_path": "path/to-the-other-model",
    "pretrained_vae_model_name_or_path": "Lightricks/LTX-Video",
    "pretrained_transformer_subfolder": "none",
}
```
</details>

## Créditos

El proyecto [finetrainers](https://github.com/a-r-r-o-w/finetrainers) y el equipo de Diffusers.
- Originalmente usó algunos conceptos de diseño de SimpleTuner
- Ahora aporta conocimiento y código para que el entrenamiento de video sea fácil de implementar
