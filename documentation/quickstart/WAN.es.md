## Guía rápida de Wan 2.1

En este ejemplo, entrenaremos un LoRA de Wan 2.1 usando el [dataset Disney de dominio público](https://hf.co/datasets/sayakpaul/video-dataset-disney-organized) de Sayak Paul.



https://github.com/user-attachments/assets/51e6cbfd-5c46-407c-9398-5932fa5fa561


### Requisitos de hardware

Wan 2.1 **1.3B** no requiere mucha memoria del sistema **ni** de GPU. El modelo **14B**, también soportado, es bastante más exigente.

Actualmente, el entrenamiento imagen‑a‑video no está soportado para Wan, pero T2V LoRA y Lycoris correrán en los modelos I2V.

#### Texto a video

1.3B - https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- Resolución: 832x480
- LoRA de rango 16 usa un poco más de 12G (batch size 4)

14B - https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers
- Resolución: 832x480
- Cabe en 24G, pero tendrás que ajustar varios settings.

<!--
#### Image to Video
14B (720p) - https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
- Resolution: 1280x720
-->

#### Image to Video (Wan 2.2)

Los checkpoints I2V recientes de Wan 2.2 funcionan con el mismo flujo de entrenamiento:

- High stage: https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-Diffusers/tree/main/high_noise_model
- Low stage: https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-Diffusers/tree/main/low_noise_model

Puedes apuntar al stage deseado con `model_flavour` y `wan_validation_load_other_stage` como se describe más adelante en esta guía.

Necesitarás:
- **un mínimo realista**: 16GB o una sola GPU 3090 o V100
- **idealmente** varias 4090, A6000, L40S o mejor

Si encuentras desajustes de forma en las capas de time embedding al usar checkpoints Wan 2.2, habilita el nuevo
flag `wan_force_2_1_time_embedding`. Esto fuerza al transformer a volver a los time embeddings estilo Wan 2.1 y
resuelve el problema de compatibilidad.

#### Presets de stage y validación

- `model_flavour=i2v-14b-2.2-high` apunta al high-noise stage de Wan 2.2.
- `model_flavour=i2v-14b-2.2-low` apunta al low-noise stage (mismos checkpoints, distinto subfolder).
- Activa `wan_validation_load_other_stage=true` para cargar el stage opuesto junto al que entrenas para renders de validación.
- Deja el flavour sin definir (o usa `t2v-480p-1.3b-2.1`) para la ejecución estándar de texto‑a‑video de Wan 2.1.

Los sistemas Apple silicon no funcionan demasiado bien con Wan 2.1 por ahora; se pueden esperar tiempos de unos 10 minutos por paso de entrenamiento.

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
pip install simpletuner[cuda]
```

Para instalación manual o entorno de desarrollo, consulta la [documentación de instalación](../INSTALL.md).
#### SageAttention 2

Si deseas usar SageAttention 2, deben seguirse algunos pasos.

> Nota: SageAttention ofrece una aceleración mínima, no muy efectiva; no está claro por qué. Probado en 4090.

Ejecuta lo siguiente dentro de tu venv de python:
```bash
git clone https://github.com/thu-ml/SageAttention
pushd SageAttention
  pip install . --no-build-isolation
popd
```

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

### Offloading de memoria (opcional)

Wan es uno de los modelos más pesados que soporta SimpleTuner. Habilita offloading agrupado si estás cerca del límite de VRAM:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Solo dispositivos CUDA respetan `--group_offload_use_stream`; ROCm/MPS hacen fallback automáticamente.
- Deja el staging a disco comentado a menos que la memoria CPU sea el cuello de botella.
- `--enable_model_cpu_offload` es mutuamente excluyente con group offload.

### Chunking de feed-forward (opcional)

Si los checkpoints 14B aún hacen OOM durante gradient checkpointing, divide las capas feed‑forward de Wan:

```bash
--enable_chunked_feed_forward \
--feed_forward_chunk_size 2 \
```

Esto coincide con el nuevo toggle del asistente de configuración (`Training → Memory Optimisation`). Tamaños de chunk más pequeños ahorran más
memoria pero ralentizan cada paso. También puedes configurar `WAN_FEED_FORWARD_CHUNK_SIZE=2` en tu entorno para pruebas rápidas.


Si prefieres configurarlo manualmente:

Copia `config/config.json.example` a `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Usuarios multi-GPU pueden consultar [este documento](../OPTIONS.md#environment-configuration-variables) para información sobre cómo configurar la cantidad de GPUs a usar.

Tu config al final se verá como la mía:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "attention_mechanism": "sageattention",
  "data_backend_config": "config/wan/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "offload_during_startup": true,
  "disable_benchmark": false,
  "output_dir": "output/wan",
  "lora_type": "standard",
  "lycoris_config": "config/wan/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "wan-disney",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "wan-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "model_family": "wan",
  "train_batch_size": 2,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "832x480",
  "validation_prompt": "两只拟人化的猫咪身穿舒适的拳击装备，戴着鲜艳的手套，在聚光灯照射的舞台上激烈对战",
  "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
  "validation_guidance": 5.2,
  "validation_num_inference_steps": 40,
  "validation_num_video_frames": 81,
  "mixed_precision": "bf16",
  "optimizer": "optimi-lion",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.01,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "no_change",
  "vae_batch_size": 1,
  "webhook_config": "config/wan/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "validation_guidance_skip_layers": [9],
  "validation_guidance_skip_layers_start": 0.0,
  "validation_guidance_skip_layers_stop": 1.0,
  "lora_rank": 16,
  "flow_schedule_shift": 3,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

De particular importancia en esta configuración son los ajustes de validación. Sin ellos, las salidas no se ven tan bien.

### Opcional: regularizador temporal CREPA

Para movimiento más suave y menos drift de identidad en Wan:
- En **Training → Loss functions**, habilita **CREPA**.
- Comienza con **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- El codificador por defecto (`dinov2_vitg14`, tamaño `518`) funciona bien; cambia a `dinov2_vits14` + `224` solo si necesitas recortar VRAM.
- La primera ejecución descarga DINOv2 vía torch hub; cachea o precarga si entrenas offline.
- Solo habilita **Drop VAE Encoder** cuando entrenes completamente desde latentes en caché; si no, mantenlo desactivado para que las codificaciones de píxeles sigan funcionando.

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

</details>

### Entrenamiento TREAD

> ⚠️ **Experimental**: TREAD es una función recién implementada. Aunque es funcional, las configuraciones óptimas aún se están explorando.

[TREAD](../TREAD.md) (paper) significa **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion. Es un método que puede acelerar el entrenamiento de Flux al enrutar tokens de forma inteligente a través de capas del transformer. La aceleración es proporcional a cuántos tokens eliminas.

#### Configuración rápida

Agrega esto a tu `config.json` para un enfoque simple y conservador con el que llegar a unos 5 segundos por paso con bs=2 y 480p (reduciendo desde 10 segundos por paso a velocidad normal):

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.1,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

Esta configuración:
- Mantiene solo el 50% de los tokens de imagen durante las capas 2 hasta la penúltima
- Los tokens de texto nunca se eliminan
- Acelera el entrenamiento en ~25% con impacto mínimo en la calidad
- Potencialmente mejora la calidad de entrenamiento y la convergencia

Para Wan 1.3B podemos mejorar este enfoque usando una configuración progresiva de rutas en las 29 capas y lograr ~7.7 segundos por paso con bs=2 y 480p:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "tread_config": {
      "routes": [
          { "selection_ratio": 0.1, "start_layer_idx": 2, "end_layer_idx": 8 },
          { "selection_ratio": 0.25, "start_layer_idx": 9, "end_layer_idx": 11 },
          { "selection_ratio": 0.35, "start_layer_idx": 12, "end_layer_idx": 15 },
          { "selection_ratio": 0.25, "start_layer_idx": 16, "end_layer_idx": 23 },
          { "selection_ratio": 0.1, "start_layer_idx": 24, "end_layer_idx": -2 }
      ]
  }
}
```
</details>

Esta configuración intentará usar dropout de tokens más agresivo en las capas internas del modelo donde el conocimiento semántico no es tan importante.

Para algunos datasets, un dropout más agresivo puede ser tolerable, pero un valor de 0.5 es considerablemente alto para Wan 2.1.

#### Puntos clave

- **Soporte de arquitectura limitado** - TREAD solo está implementado para modelos Flux y Wan
- **Mejor a altas resoluciones** - Mayores aceleraciones a 1024x1024+ debido a la complejidad O(n²) de la atención
- **Compatible con pérdida enmascarada** - Las regiones enmascaradas se conservan automáticamente (pero esto reduce la aceleración)
- **Funciona con cuantización** - Puede combinarse con entrenamiento int8/int4/NF4
- **Espera un pico de pérdida inicial** - Al iniciar entrenamiento LoRA/LoKr, la pérdida será más alta al principio pero se corrige rápido

#### Consejos de ajuste

- **Conservador (enfocado en calidad)**: Usa `selection_ratio` de 0.1-0.3
- **Agresivo (enfocado en velocidad)**: Usa `selection_ratio` de 0.3-0.5 y acepta el impacto en calidad
- **Evita capas tempranas/tardías**: No enrutes en las capas 0-1 ni en la capa final
- **Para entrenamiento LoRA**: Podrías ver ligeras desaceleraciones - experimenta con distintas configs
- **Mayor resolución = mayor aceleración**: Más beneficioso en 1024px y arriba

#### Comportamiento conocido

- Cuantos más tokens se eliminen (mayor `selection_ratio`), más rápido el entrenamiento pero mayor pérdida inicial
- El entrenamiento LoRA/LoKr muestra un pico de pérdida inicial que se corrige rápidamente a medida que la red se adapta
  - Usar una configuración menos agresiva o múltiples rutas con capas internas más altas aliviará esto
- Algunas configuraciones LoRA pueden entrenar un poco más lento - las configs óptimas aún se exploran
- La implementación de RoPE (rotary position embedding) es funcional pero puede no ser 100% correcta

Para opciones de configuración detalladas y solución de problemas, consulta la [documentación completa de TREAD](../TREAD.md).


#### Prompts de validación

Dentro de `config/config.json` está el "prompt de validación principal", que suele ser el instance_prompt principal en el que estás entrenando para tu único sujeto o estilo. Además, se puede crear un archivo JSON que contiene prompts adicionales para ejecutar durante las validaciones.

El archivo de ejemplo `config/user_prompt_library.json.example` contiene el siguiente formato:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

Los apodos son el nombre de archivo de la validación, así que mantenlos cortos y compatibles con tu sistema de archivos.

Para indicar al entrenador esta librería de prompts, añádela a TRAINER_EXTRA_ARGS agregando una nueva línea al final de `config.json`:
<details>
<summary>Ver ejemplo de config</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

Un conjunto de prompts diverso ayudará a determinar si el modelo colapsa a medida que entrena. En este ejemplo, la palabra `<token>` debe reemplazarse por el nombre de tu sujeto (instance_prompt).

<details>
<summary>Ver ejemplo de config</summary>

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
</details>

> ℹ️ Wan 2.1 usa solo el codificador de texto UMT5, que tiene mucha información local en sus embeddings, lo que significa que los prompts más cortos podrían no tener suficiente información para que el modelo haga un buen trabajo. Asegúrate de usar prompts más largos y descriptivos.

#### Seguimiento de puntuaciones CLIP

Esto no debería habilitarse para entrenamiento de modelos de video por ahora.

# Pérdida de evaluación estable

Si deseas usar pérdida MSE estable para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/EVAL_LOSS.md) para información sobre cómo configurar e interpretar la pérdida de evaluación.

#### Vistas previas de validación

SimpleTuner admite la transmisión de vistas previas de validación intermedias durante la generación usando modelos Tiny AutoEncoder. Esto te permite ver imágenes de validación generándose paso a paso en tiempo real mediante callbacks de webhook.

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

Los modelos de flow-matching como Flux, Sana, SD3, LTX Video y Wan 2.1 tienen una propiedad llamada `shift` que permite desplazar la parte entrenada del calendario de timesteps usando un valor decimal simple.

##### Valores predeterminados
Por defecto, no se aplica shift de calendario, lo que da una forma de campana sigmoide a la distribución de muestreo de timesteps, también conocida como `logit_norm`.

##### Auto-shift
Un enfoque comúnmente recomendado es seguir varios trabajos recientes y habilitar el shift de timesteps dependiente de la resolución, `--flow_schedule_auto_shift`, que usa valores de shift más altos para imágenes grandes y valores más bajos para imágenes pequeñas. Esto da resultados de entrenamiento estables pero potencialmente mediocres.

##### Especificación manual
_Gracias a General Awareness de Discord por los siguientes ejemplos_

> ℹ️ Estos ejemplos muestran cómo funciona el valor usando Flux Dev, aunque Wan 2.1 debería ser muy similar.

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

#### Ajustes de validación

Durante la exploración inicial para agregar Wan 2.1 a SimpleTuner, salían resultados horribles, y esto se reducía a varias razones:

- No suficientes pasos de inferencia
  - A menos que uses UniPC, probablemente necesites al menos 40 pasos. UniPC puede bajar el número un poco, pero tendrás que experimentar.
- Configuración incorrecta del scheduler
  - Estaba usando un flujo de Euler normal, pero la distribución Betas parece funcionar mejor
  - Si no tocaste este ajuste, debería estar bien ahora
- Resolución incorrecta
  - Wan 2.1 solo funciona correctamente en las resoluciones en las que se entrenó; a veces tienes suerte, pero es común obtener malos resultados
- Valor CFG malo
  - Wan 2.1 1.3B en particular parece sensible a valores CFG, pero un valor alrededor de 4.0-5.0 parece seguro
- Prompts malos
  - Por supuesto, los modelos de video parecen requerir un equipo de místicos que pasen meses en las montañas en un retiro zen para aprender el arte sagrado del prompting, porque sus datasets y estilo de captions se guardan como el Santo Grial.
  - tl;dr prueba prompts diferentes.

A pesar de todo esto, a menos que tu batch size sea demasiado bajo y/o tu learning rate demasiado alto, el modelo funcionará correctamente en tu herramienta de inferencia favorita (suponiendo que ya tengas una que dé buenos resultados).

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
    "cache_dir_vae": "cache/vae/wan/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

- Las ejecuciones I2V de Wan 2.2 crean cachés de conditioning CLIP. En la entrada de dataset **video**, apunta a un backend dedicado y (opcionalmente) sobrescribe la ruta del caché:

<details>
<summary>Ver ejemplo de config</summary>

```json
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "conditioning_image_embeds": "disney-conditioning",
    "cache_dir_conditioning_image_embeds": "cache/conditioning_image_embeds/disney-black-and-white"
  }
```
</details>

- Define el backend de conditioning una sola vez y reutilízalo entre datasets si es necesario (objeto completo mostrado para claridad):

<details>
<summary>Ver ejemplo de config</summary>

```json
  {
    "id": "disney-conditioning",
    "type": "local",
    "dataset_type": "conditioning_image_embeds",
    "cache_dir": "cache/conditioning_image_embeds/disney-conditioning",
    "disabled": false
  }
```
</details>

- En la subsección `video`, tenemos las siguientes claves:
  - `num_frames` (opcional, int) es cuántos frames de datos entrenaremos.
    - A 15 fps, 75 frames son 5 segundos de video, salida estándar. Este debería ser tu objetivo.
  - `min_frames` (opcional, int) determina la longitud mínima de un video que se considerará para entrenamiento.
    - Esto debería ser al menos igual a `num_frames`. Si no se configura, será igual.
  - `max_frames` (opcional, int) determina la longitud máxima de un video que se considerará para entrenamiento.
  - `bucket_strategy` (opcional, string) determina cómo se agrupan los videos en buckets:
    - `aspect_ratio` (predeterminado): Agrupar solo por relación de aspecto espacial (p. ej., `1.78`, `0.75`).
    - `resolution_frames`: Agrupar por resolución y recuento de frames en formato `WxH@F` (p. ej., `832x480@75`). Útil para datasets de resolución/duración mixta.
  - `frame_interval` (opcional, int) al usar `resolution_frames`, redondea recuentos de frames a este intervalo.
<!--  - `is_i2v` (optional, bool) determines whether i2v training will be done on a dataset.
    - This is set to True by default for Wan 2.1. You can disable it, however.
-->

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
pip install simpletuner[cuda]
simpletuner train
```

**Opción 2 (Método de git clone):**
```bash
simpletuner train
```

> ℹ️ Agrega `--model_flavour i2v-14b-2.2-high` (o `low`) y, si lo deseas, `--wan_validation_load_other_stage` dentro de `TRAINER_EXTRA_ARGS` o tu invocación CLI cuando entrenes Wan 2.2. Agrega `--wan_force_2_1_time_embedding` solo cuando el checkpoint informe un desajuste de forma en time-embedding.

**Opción 3 (Método heredado - aún funciona):**
```bash
./train.sh
```

Esto iniciará el caché a disco de text embeds y salidas del VAE.

Para más información, consulta los documentos de [dataloader](../DATALOADER.md) y [tutorial](../TUTORIAL.md).

## Notas y consejos de solución de problemas

### Configuración de VRAM más baja

Wan 2.1 es sensible a la cuantización, y no puede usarse con NF4 o INT4 actualmente.

- OS: Ubuntu Linux 24
- GPU: Un solo dispositivo NVIDIA CUDA (10G, 12G)
- Memoria del sistema: alrededor de 12G de memoria del sistema
- Precisión del modelo base: `int8-quanto`
- Optimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolución: 480px
- Tamaño de lote: 1, cero pasos de acumulación de gradiente
- DeepSpeed: deshabilitado / sin configurar
- PyTorch: 2.6
- Asegúrate de habilitar `--gradient_checkpointing` o nada evitará que haga OOM
- Entrena solo con imágenes, o configura `num_frames` en 1 para tu dataset de video

**NOTA**: El pre-caché de embeddings del VAE y salidas del codificador de texto puede usar más memoria y aun así producir OOM. Como resultado, `--offload_during_startup=true` es básicamente requerido. Si sucede, se pueden habilitar cuantización del codificador de texto y tiling del VAE. (Wan no soporta actualmente VAE tiling/slicing)

Velocidades:
- 665.8 sec/iter en un Macbook Pro M3 Max
- 2 sec/iter en una NVIDIA 4090 con batch size 1
- 11 sec/iter en NVIDIA 4090 con batch size 4

### SageAttention

Al usar `--attention_mechanism=sageattention`, la inferencia puede acelerarse en tiempo de validación.

**Nota**: Esto no es compatible con el paso final de decodificación del VAE y no acelerará esa parte.

### Pérdida con máscara

No uses esto con Wan 2.1.

### Cuantización
- La cuantización no es necesaria para entrenar este modelo en 24G

### Artefactos de imagen
Wan requiere usar el scheduler de flow-matching Euler Betas o (por defecto) el solver multistep UniPC, un scheduler de orden superior que hará predicciones más fuertes.

Como otros modelos DiT, si haces estas cosas (entre otras) pueden comenzar a aparecer artefactos de cuadrícula cuadrada en las muestras:
- Sobreentrenar con datos de baja calidad
- Usar una tasa de aprendizaje demasiado alta
- Sobreentrenamiento (en general) de una red de baja capacidad con demasiadas imágenes
- Subentrenamiento (también) de una red de alta capacidad con muy pocas imágenes
- Usar relaciones de aspecto extrañas o tamaños de datos de entrenamiento

### Aspect bucketing
- Los videos se agrupan en buckets como las imágenes.
- Entrenar demasiado tiempo en recortes cuadrados probablemente no dañará demasiado este modelo. Dale con todo, es excelente y confiable.
- Por otro lado, usar los buckets de aspecto natural de tu dataset podría sesgar demasiado esas formas en inferencia.
  - Esto podría ser una cualidad deseable, ya que evita que estilos dependientes de aspecto, como lo cinematográfico, se filtren a otras resoluciones demasiado.
  - Sin embargo, si buscas mejorar resultados por igual en muchos buckets de aspecto, quizá tengas que experimentar con `crop_aspect=random`, lo cual tiene sus propias desventajas.
- Mezclar configuraciones de dataset definiendo tu dataset de imágenes múltiples veces ha dado resultados muy buenos y un modelo bien generalizado.

### Entrenar modelos Wan 2.1 ajustados personalizados

Algunos modelos ajustados en Hugging Face Hub carecen de la estructura completa de directorios, lo que requiere configurar opciones específicas.

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "model_family": "wan",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_transformer_model_name_or_path": "path/to-the-other-model",
    "pretrained_vae_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_transformer_subfolder": "none",
}
```
</details>

> Nota: Puedes proporcionar una ruta a un archivo `.safetensors` de un solo archivo para `pretrained_transformer_name_or_path`
