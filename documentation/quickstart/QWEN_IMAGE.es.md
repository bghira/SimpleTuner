## Qwen Image 2.1

Qwen Image 2.1 es la variante predeterminada (`model_flavour: "v2.1"`) y utiliza `Qwen/Qwen-Image-2.1`. Tiene un transformer de 32 bloques, un codificador de texto Qwen3-VL y un VAE de 64 canales con compresión espacial de 16×.

El ejemplo `qwen_image.peft-lora` entrena con `RareConcepts/Domokun` a 512px y usa el activador `🟫`. Empieza con BF16 (`base_model_precision: "no_change"`) y activa los checkpoints de gradientes cuando falte memoria. El ejemplo utiliza cachés de latentes y texto exclusivos de 2.1; no reutilices cachés de variantes anteriores.

```bash
simpletuner train example=qwen_image.peft-lora
```

Para la validación, usa `validation_guidance: 1.0`, `validation_guidance_real: 1.0` y `validation_num_inference_steps: 40`. Mantén el activador en los prompts de validación para comprobar si se ha aprendido el concepto.

Qwen Image 2.1 decodifica imágenes individuales sin conservar cachés temporales de características que no se utilizan. Al decodificar una imagen de 2048×2048 de forma aislada en H200 con BF16, esto redujo el pico de memoria asignada de 26.87 a 15.28 GiB con una salida idéntica. La decodificación por bloques ahorra más memoria, pero puede introducir líneas de color al limitar el contexto espacial; eliminar los cachés sin uso no corrige esas líneas.

Las variantes anteriores siguen disponibles: `v1.0` selecciona Qwen-Image, `v2.0` selecciona Qwen-Image-2512 y las variantes `edit-*` conservan sus checkpoints. Sus adaptadores y cachés de latentes no son intercambiables con los de 2.1.

La receta Domokun de 250 steps sirve para medir throughput, no como receta de convergencia fiable. Un checkpoint anterior generó Domokun reconocible tras recargarlo, pero nuevas ejecuciones de 250 steps no reprodujeron el resultado. Los controles conservando máscaras de padding, desactivando compilación y restaurando la expresión RoPE anterior también fallaron. Los latentes en caché decodifican el personaje correcto. La causa del deterioro sigue sin resolverse; las tablas de tiempos no demuestran calidad equivalente entre backends de atención.

### Configuraciones por VRAM

Los ejemplos usan BF16, LoRA de rango 32, Optimi Lion y compilación regional a 512px, sin checkpoint de gradientes. La primera ejecución incluye compilación; compare los pasos tras el calentamiento. Los límites de 24 GB y 32 GB se verificaron en L40S, no en tarjetas separadas de esas capacidades.

| VRAM | Ejemplo | Lote del dataset | Pico de VRAM (GiB) | Paso tras calentamiento (s) |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 1 | 20.6 | 0.238 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 2 | 26.5 | 0.390 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 2 | 26.5 | 0.390 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 10 | 71.8 | 0.639 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 20 | 128.4 | 1.223 |

Medido en L40S (configuraciones de 24/32/48 GB), H100 (80 GB) y H200 (144 GB), con 20 pasos y los cinco primeros excluidos del tiempo. El pico de VRAM incluye la preparación. Son resultados a 512px para cada lote, no garantías para imágenes mayores o prompts más largos.

La configuración de 48 GB también usa lote 2: en L40S logró mejor rendimiento por imagen que los lotes 3, 4 y 5. El lote 5 cabía en 43.3 GiB pero tardó 0.991 s/paso, frente a 0.390 s/paso con lote 2.

```bash
simpletuner train example=qwen_image-2.1-48g.peft-lora
```

Use el archivo de dataset incluido con cada ejemplo: su tamaño de lote es explícito. Inicie un entrenamiento nuevo al cambiar el lote o la configuración del dataset; no reutilice un checkpoint de estado incompatible.

Para reducir la VRAM, activa `gradient_checkpointing: true` y `gradient_checkpointing_interval: 2`. Ahora esto aplica checkpoint a grupos de dos bloques contiguos. Consulta las [mediciones de checkpoint y atención de Qwen Image 2.1](../experimental/SEGMENTED_CHECKPOINTING.es.md#qwen-image-21); el resultado anterior con bloques alternos ha quedado sustituido. BF16 cabe en estos presets sin un checkpoint int8.

La ruta de texto a imagen evita construir secuencias con control dependiente de valores de tensores, permitiendo captura sin cortes de grafo. RoPE con aritmética real permite que Inductor fusione normalización y rotación; también se compilan los epílogos de modulación, residual y MLP. Estos ejemplos de entrenamiento no usan los kernels existentes de GEMM CuTe ConvRot para Hopper ni de RoPE de LTX solo para inferencia.


### Piloto experimental de AnyFlow

Los tres ejemplos `qwen_image-2.1-anyflow-stage*.peft-lora` prueban si la destilación de intervalos puede conservar el comportamiento de la base al introducir `🟫`. Son experimentales; la ficha de Qwen no confirma que la destilación de guidance causara el deterioro anterior.

Todas las etapas usan 1024px, BF16, AdamW, rank 32 y batch 4. La etapa 1 desactiva el checkpointing de gradientes en H200; las etapas 2 y 3 usan grupos contiguos de dos bloques. La carga difiere de los presets de rendimiento a 512px anteriores. Instala `webshart` antes de ejecutarlos.

Estos ejemplos de AnyFlow usan explícitamente `grad_clip_method: "value"` con `max_grad_norm: 0.01`: cada elemento del gradiente se limita a ±0.01. No es un límite de la norma global. Para probar el recorte por norma a 1.0, configura tanto `grad_clip_method: "norm"` como `max_grad_norm: 1.0`.

1. **Etapa 1:** 10.000 actualizaciones forward de AnyFlow a `1e-5`. CC12M usa `webshart/cc12m-structured-captions` con `caption_key: "long_caption"`; e621 usa `webshart/e621-2024-webp-4Mpixel-webshart-indices`. Cada conjunto de conocimientos previos se limita a 4.096 imágenes aceptadas, con peso 0.49 cada uno. `RareConcepts/Domokun` usa el activador `🟫`, peso 0.02 y `repeats: 0`.
2. **Etapa 2:** 2.000 actualizaciones DMD on-policy a `2e-6`, coentrenando el objetivo forward con la misma mezcla. Es DMD de AnyFlow, no DPO con pares de preferencias. Incluir el personaje en ambas etapas aporta ejemplos reales; el profesor congelado por sí solo no puede enseñarlo.
3. **Etapa 3:** 100 actualizaciones a `5e-7`, con peso 0.5 para Domokun y 0.25 para cada conjunto de regularización, limitado a 64 imágenes cada uno. Todos los intervalos usan `r=t` y el objetivo flow bruto, conservando el embedder de intervalos durante un ajuste supervisado breve. Los batches de regularización usan la predicción de la base con el adapter desactivado.

Revisa los checkpoints de 2.000, 5.000 y 10.000 actualizaciones antes de ampliar la etapa 1 hasta 20.000. La etapa 2 tiene un presupuesto de 2.000 actualizaciones; detenla antes si las imágenes de validación comparables empeoran. El número de pasos por sí solo no determina un modelo final.

La compilación dinámica está habilitada para captions de longitud variable. `schedule_shift: 2.000802574061872` corresponde al scheduler publicado a 1024px (4.096 tokens latentes); vuelve a calcularlo al cambiar la resolución.

```bash
simpletuner train example=qwen_image-2.1-anyflow-stage1.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage2.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage3.peft-lora
```

Las etapas 2 y 3 cargan el adapter final anterior mediante `init_lora` e inician estados nuevos del optimizador y dataloader. Los pesos controlan la selección entre conjuntos todavía disponibles, sin garantizar porcentajes finales. El piloto limitado no cubre los corpus completos. El campo textual `long_caption` funciona con el selector Webshart existente y no requiere captions como objetos JSON nativos.

Las etapas 1 y 2 usan `diffusion_target: "base_prediction"` y `fuse_guidance_scale: 1.0`: la rama diffusion conserva el campo condicional congelado mientras las otras ramas aprenden de los datos. Es una hipótesis de conservación, no una garantía de aprender el personaje. Compara los prompts de personaje y de conocimientos previos a 4, 16 y 40 pasos de inferencia con una referencia de la base a 40 pasos; la validación automática usa 4. Consulta [AnyFlow](../experimental/ANYFLOW.es.md) para los objetivos y requisitos de checkpoints.

Piloto completado: la etapa 1 llegó a 10.000 actualizaciones y la etapa 2 a 2.000. Tras recargar los adaptadores, las imágenes de 40 pasos del zorro y de los prompts del personaje conservaron coherencia, pero estos últimos produjeron personas en vez de Domokun. Las imágenes de cuatro pasos siguieron borrosas o ruidosas. Otra comparación en L40S probó ambos checkpoints a 1024px, semilla 42 y CFG 1/2/4/6, con prompt negativo vacío. Las aserciones de llamadas verificaron dos pasadas por paso para CFG mayor que 1. Las muestras revisadas del zorro y la playa no mejoraron: aumentar CFG añadió saturación y artefactos. Estos resultados no identifican la causa; la etapa 3 sigue sin validarse.

Dos continuaciones de la etapa 1 desde el mismo checkpoint añadieron 1.000 actualizaciones cada una, comparando umbrales de clipping por valor de 0.01 y 1.0. Tras recargar, las imágenes del zorro a cuatro pasos siguieron ruidosas en ambas; las imágenes de playa a 40 pasos siguieron mostrando personas. El clipping se activó en 65/1.000 actualizaciones con 0.01 y en 0/1.000 con 1.0. Ambas ramas usaron el optimizador sin corregir, y sus gradientes iniciales diferían antes de activarse el clipping; las diferencias pequeñas no pueden atribuirse únicamente al umbral.

<a id="qwen21-optimizer-correction"></a>

Los pilotos de la etapa 1 y del asistente descritos aquí se ejecutaron antes de corregir el helper de suma estocástica de AdamW BF16: calculaba `other + alpha * input` en lugar de `input + alpha * other`. Con β₁ = 0,9, el primer momento seguía `m = 0.09 * m + g` en vez de `m = 0.9 * m + 0.1 * g`. Las pruebas de regresión con aritmética exacta cubren CPU, MPS y CUDA. Las observaciones de las imágenes siguen siendo válidas para esos checkpoints, pero no constituyen una prueba limpia de la arquitectura ni del objetivo de destilación; hay que repetir la comprobación con el optimizador corregido.

Repetir la continuación de 1.000 actualizaciones con el optimizador corregido, el mismo checkpoint inicial y clipping por valor de 1.0 no recuperó las imágenes revisadas del zorro ni de la playa a cuatro pasos. A 40 pasos el zorro conservó coherencia, mientras que el prompt de playa siguió produciendo una persona. Esto evalúa la recuperación del checkpoint existente, no el entrenamiento desde cero con el optimizador corregido; la causa del fallo general sigue sin resolverse.

### Configuración de Qwen Image anterior (v1.0 / v2.0)

> 🆕 ¿Buscas los checkpoints de edición? Consulta la [guía rápida de Qwen Image Edit](./QWEN_EDIT.md) para instrucciones de entrenamiento con referencia emparejada.

En este ejemplo, entrenaremos un LoRA para Qwen Image, un modelo visión‑lenguaje de 20B parámetros. Debido a su tamaño, necesitaremos técnicas agresivas de optimización de memoria.

Una GPU de 24GB es el mínimo absoluto, y aun así necesitarás cuantización extensa y configuración cuidadosa. Se recomiendan encarecidamente 40GB+ para una experiencia más fluida.

Al entrenar en 24G, las validaciones se quedarán sin memoria a menos que uses menor resolución o un nivel de cuantización agresivo más allá de int8.

### Requisitos de hardware

Qwen Image es un modelo de 20B parámetros con un codificador de texto sofisticado que por sí solo consume ~16GB de VRAM antes de la cuantización. El modelo usa un VAE personalizado con 16 canales latentes.

**Limitaciones importantes:**
- **No está soportado en AMD ROCm ni MacOS** por falta de flash attention eficiente
- Tamaño de lote > 1 no funciona correctamente por ahora; usa acumulación de gradiente en su lugar
- TREAD (Text-Representation Enhanced Adversarial Diffusion) aún no está soportado

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
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
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
- `lora_type` - Configúralo en `standard` para PEFT LoRA o `lycoris` para LoKr.
- `model_family` - Configúralo en `qwen_image`.
- `model_flavour` - Configúralo en `v1.0`.
- `output_dir` - Configúralo al directorio donde quieres guardar tus checkpoints y las imágenes de validación. Se recomienda usar una ruta completa aquí.
- `train_batch_size` - Ajústalo según la VRAM disponible. Los overrides actuales de Qwen en SimpleTuner admiten tamaños de lote mayores que 1.
- `gradient_accumulation_steps` - Configúralo en 2-8 si quieres un batch efectivo mayor sin subir la VRAM por paso.
- `validation_resolution` - Debes configurarlo en `1024x1024` o menor por restricciones de memoria.
  - 24G no puede manejar validaciones 1024x1024 actualmente - tendrás que reducir el tamaño
  - Se pueden especificar otras resoluciones separándolas con comas: `1024x1024,768x768,512x512`
- `validation_guidance` - Usa un valor alrededor de 3.0-4.0 para buenos resultados.
- `validation_num_inference_steps` - Usa alrededor de 30.
- `use_ema` - Configurar esto en `true` ayudará a obtener resultados más suaves pero usa más memoria.

- `optimizer` - Usa `optimi-lion` para buenos resultados, o `adamw-bf16` si tienes memoria de sobra.
- `mixed_precision` - Debe configurarse en `bf16` para Qwen Image.
- `gradient_checkpointing` - **Obligatorio** habilitar (`true`) para un uso razonable de memoria.
- `base_model_precision` - **Muy recomendado** configurar en `int8-quanto` o `nf4-bnb` para tarjetas de 24GB.
- `quantize_via` - Configúralo en `cpu` para evitar OOM durante la cuantización en GPUs más pequeñas.
- `quantize_activations` - Mantén esto en `false` para conservar la calidad de entrenamiento.

Ajustes de optimización de memoria para GPUs de 24GB:
- `lora_rank` - Usa 8 o menos.
- `lora_alpha` - Iguala esto a tu valor de lora_rank.
- `flow_schedule_shift` - Configura en 1.73 (o experimenta entre 1.0-3.0).

Tu config.json se verá algo así para un setup mínimo:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "model_type": "lora",
    "model_family": "qwen_image",
    "model_flavour": "v1.0",
    "lora_type": "standard",
    "lora_rank": 8,
    "lora_alpha": 8,
    "output_dir": "output/models-qwen_image",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "validation_resolution": "1024x1024",
    "validation_guidance": 4.0,
    "validation_num_inference_steps": 30,
    "validation_seed": 42,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_step_interval": 100,
    "vae_batch_size": 1,
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 1000,
    "max_grad_norm": 0.01,
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "learning_rate": "1e-4",
    "gradient_checkpointing": "true",
    "base_model_precision": "int2-quanto",
    "quantize_via": "cpu",
    "quantize_activations": false,
    "flow_schedule_shift": 1.73,
    "disable_benchmark": false,
    "data_backend_config": "config/qwen_image/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ Usuarios multi-GPU pueden consultar [este documento](../OPTIONS.md#environment-configuration-variables) para información sobre cómo configurar la cantidad de GPUs a usar.

> ⚠️ **Crítico para GPUs de 24GB**: El codificador de texto por sí solo usa ~16GB de VRAM. Con cuantización `int2-quanto` o `nf4-bnb`, esto se puede reducir significativamente.

Para una comprobación rápida con una configuración conocida que funciona:

**Opción 1 (Recomendada - instalación con pip):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train example=qwen_image.peft-lora
```

**Opción 2 (Método de git clone):**
```bash
simpletuner train env=examples/qwen_image.peft-lora
```

**Opción 3 (Método heredado - aún funciona):**
```bash
ENV=examples/qwen_image.peft-lora ./train.sh
```

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

Para indicar al entrenador esta librería de prompts, añádela a tu config.json:
```json
  "validation_prompt_library": "config/user_prompt_library.json",
```

Un conjunto de prompts diverso ayudará a determinar si el modelo está aprendiendo correctamente:

```json
{
    "anime_style": "a breathtaking anime-style portrait with vibrant colors and expressive features",
    "chef_cooking": "a high-quality, detailed photograph of a sous-chef immersed in culinary creation",
    "portrait": "a lifelike and intimate portrait showcasing unique personality and charm",
    "cinematic": "a cinematic, visually stunning photo with dramatic and captivating presence",
    "elegant": "an elegant and timeless portrait exuding grace and sophistication",
    "adventurous": "a dynamic and adventurous photo captured in an exciting moment",
    "mysterious": "a mysterious and enigmatic portrait shrouded in shadows and intrigue",
    "vintage": "a vintage-style portrait evoking the charm and nostalgia of a bygone era",
    "artistic": "an artistic and abstract representation blending creativity with visual storytelling",
    "futuristic": "a futuristic and cutting-edge portrayal set against advanced technology"
}
```

#### Seguimiento de puntuaciones CLIP

Si deseas habilitar evaluaciones para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/CLIP_SCORES.md) para información sobre cómo configurar e interpretar las puntuaciones CLIP.

#### Pérdida de evaluación estable

Si deseas usar pérdida MSE estable para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/EVAL_LOSS.md) para información sobre cómo configurar e interpretar la pérdida de evaluación.

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

Qwen Image, como modelo de flow-matching, admite el desplazamiento del calendario de timesteps para controlar qué partes del proceso de generación se entrenan.

El parámetro `flow_schedule_shift` controla esto:
- Valores bajos (0.1-1.0): Enfoque en detalles finos
- Valores medios (1.0-3.0): Entrenamiento equilibrado (recomendado)
- Valores altos (3.0-6.0): Enfoque en grandes rasgos compositivos

##### Auto-shift
Puedes habilitar el shift de timesteps dependiente de la resolución con `--flow_schedule_auto_shift`, que usa valores de shift más altos para imágenes grandes y valores más bajos para imágenes pequeñas. Esto puede dar resultados de entrenamiento estables pero potencialmente mediocres.

##### Especificación manual
Se recomienda un valor `--flow_schedule_shift` de 1.73 como punto de partida para Qwen Image, aunque quizá necesites experimentar según tu dataset y objetivos.

#### Consideraciones del dataset

Es crucial contar con un dataset sustancial para entrenar tu modelo. Hay limitaciones en el tamaño del dataset, y debes asegurarte de que sea lo suficientemente grande para entrenar tu modelo de forma efectiva.

> ℹ️ Con pocas imágenes, podrías ver el mensaje **no images detected in dataset** - aumentar el valor de `repeats` superará esta limitación.

> ⚠️ **Importante**: Debido a limitaciones actuales, mantén `train_batch_size` en 1 y usa `gradient_accumulation_steps` para simular tamaños de lote mayores.

Crea un documento `--data_backend_config` (`config/multidatabackend.json`) que contenga esto:

```json
[
  {
    "id": "pseudo-camera-10k-qwen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/pseudo-camera-10k",
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
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/dreambooth-subject",
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
    "cache_dir": "cache/text/qwen_image",
    "disabled": false,
    "write_batch_size": 16
  }
]
```

> ℹ️ Usa `caption_strategy=textfile` si tienes archivos `.txt` que contienen captions.
> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).
> ℹ️ Nota el `write_batch_size` reducido para text embeds para evitar OOM.

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

Desde el directorio de SimpleTuner, solo hay que ejecutar:

```bash
./train.sh
```

Esto iniciará el caché a disco de text embeds y salidas del VAE.

Para más información, consulta los documentos de [dataloader](../DATALOADER.md) y [tutorial](../TUTORIAL.md).

### Consejos de optimización de memoria

#### Configuración de VRAM más baja (mínimo 24GB)

La configuración de VRAM más baja de Qwen Image requiere aproximadamente 24GB:

- OS: Ubuntu Linux 24
- GPU: Un solo dispositivo NVIDIA CUDA (mínimo 24GB)
- Memoria del sistema: Se recomienda 64GB+
- Precisión del modelo base:
  - Para sistemas NVIDIA: `int2-quanto` o `nf4-bnb` (requerido para tarjetas de 24GB)
  - `int4-quanto` puede funcionar pero con menor calidad
- Optimizador: `optimi-lion` o `bnb-lion8bit-paged` para eficiencia de memoria
- Resolución: Comienza con 512px o 768px, sube a 1024px si la memoria lo permite
- Tamaño de lote: 1 (obligatorio por limitaciones actuales)
- Pasos de acumulación de gradiente: 2-8 para simular lotes más grandes
- Habilitar `--gradient_checkpointing` (obligatorio)
- Usa `--quantize_via=cpu` para evitar OOM durante el arranque
- Usa un rango LoRA pequeño (1-8)
- Configurar la variable de entorno `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ayuda a minimizar el uso de VRAM

**NOTA**: El pre-caché de embeddings del VAE y salidas del codificador de texto usará memoria significativa. Habilita `offload_during_startup=true` si encuentras problemas de OOM.

### Ejecutar inferencia en el LoRA después

Como Qwen Image es un modelo más nuevo, aquí hay un ejemplo funcional de inferencia:

<details>
<summary>Mostrar ejemplo de inferencia en Python</summary>

```python
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

model_id = 'Qwen/Qwen-Image'
adapter_id = 'your-username/your-lora-name'

# Load the pipeline
pipeline = QwenImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
pipeline.load_lora_weights(adapter_id)

# Optional: quantize the model to save VRAM
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

# Move to device
pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

# Generate an image
prompt = "Your test prompt here"
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=4.0,
    generator=torch.Generator(device='cuda').manual_seed(42),
    width=1024,
    height=1024,
).images[0]

image.save("output.png", format="PNG")
```
</details>

### Notas y consejos de solución de problemas

#### Limitaciones de tamaño de lote

Las compilaciones antiguas de diffusers para Qwen tenían problemas con batch size > 1 por el padding de embeddings de texto y el enmascarado de atención. Los overrides actuales de Qwen en SimpleTuner corrigen ambos puntos, así que los lotes mayores funcionan si tu VRAM lo permite.
- Aumenta `train_batch_size` solo después de confirmar que tienes memoria suficiente.
- Si todavía ves artefactos en una instalación antigua, actualiza y regenera cualquier embedding de texto obsoleto.

#### Cuantización

- `int2-quanto` brinda el mayor ahorro de memoria pero puede afectar la calidad
- `nf4-bnb` ofrece un buen equilibrio entre memoria y calidad
- `int4-quanto` es una opción intermedia
- Evita `int8` a menos que tengas 40GB+ de VRAM

#### Tasas de aprendizaje

Para entrenamiento LoRA:
- LoRAs pequeñas (rango 1-8): usa tasas alrededor de 1e-4
- LoRAs grandes (rango 16-32): usa tasas alrededor de 5e-5
- Con optimizador Prodigy: empieza con 1.0 y deja que se adapte

#### Artefactos de imagen

Si encuentras artefactos:
- Baja la tasa de aprendizaje
- Aumenta los pasos de acumulación de gradiente
- Asegúrate de que tus imágenes sean de alta calidad y estén bien preprocesadas
- Considera usar resoluciones más bajas al inicio

#### Entrenamiento de múltiples resoluciones

Comienza el entrenamiento en resoluciones más bajas (512px o 768px) para acelerar el aprendizaje inicial, luego ajusta fino en 1024px. Habilita `--flow_schedule_auto_shift` al entrenar en diferentes resoluciones.

### Limitaciones de plataforma

**No soportado en:**
- AMD ROCm (no tiene implementación eficiente de flash attention)
- Apple Silicon/MacOS (limitaciones de memoria y atención)
- GPUs de consumo con menos de 24GB de VRAM

### Problemas conocidos actuales

1. Tamaño de lote > 1 no funciona correctamente (usa acumulación de gradiente)
2. TREAD aún no está soportado
3. Alto uso de memoria del codificador de texto (~16GB antes de cuantización)
4. Problemas de manejo de longitud de secuencia ([issue upstream](https://github.com/huggingface/diffusers/issues/12075))

Para ayuda adicional y solución de problemas, consulta la [documentación de SimpleTuner](/documentation) o únete al Discord de la comunidad.
