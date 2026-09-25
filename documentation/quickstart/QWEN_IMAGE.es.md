## Qwen Image 2.1

Qwen Image 2.1 es la variante predeterminada (`model_flavour: "v2.1"`) y utiliza `Qwen/Qwen-Image-2.1`. Tiene un transformer de 32 bloques, un codificador de texto Qwen3-VL y un VAE de 64 canales con compresión espacial de 16×.

Qwen Image 2.1 usa el [VAE con corrección de textura de Ollin](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) de forma predeterminada para la validación y otras operaciones de decodificación VAE. Solo se ajustó el decodificador; el codificador no cambió, por lo que los latentes de entrenamiento existentes de la versión 2.1 siguen siendo compatibles. La revisión del VAE se fija independientemente del modelo base. Para reproducir resultados con el VAE original, configura `pretrained_vae_model_name_or_path: "Qwen/Qwen-Image-2.1"`. Las sustituciones explícitas del VAE y las variantes anteriores mantienen su comportamiento.

El ejemplo `qwen_image.peft-lora` entrena con `RareConcepts/Domokun` a 512px y usa el activador `🟫`. Empieza con BF16 (`base_model_precision: "no_change"`) y activa los checkpoints de gradientes cuando falte memoria. El ejemplo utiliza cachés de latentes y texto exclusivos de 2.1; no reutilices cachés de variantes anteriores.

```bash
simpletuner train example=qwen_image.peft-lora
```

Para la validación, usa `validation_guidance: 1.0`, `validation_guidance_real: 1.0` y `validation_num_inference_steps: 40`. Mantén el activador en los prompts de validación para comprobar si se ha aprendido el concepto.

Qwen Image 2.1 decodifica imágenes individuales sin conservar cachés temporales de características que no se utilizan. Al decodificar una imagen de 2048×2048 de forma aislada en H200 con BF16, esto redujo el pico de memoria asignada de 26.87 a 15.28 GiB con una salida idéntica. La decodificación por bloques ahorra más memoria, pero puede introducir líneas de color al limitar el contexto espacial; eliminar los cachés sin uso no corrige esas líneas.

Las variantes anteriores siguen disponibles: `v1.0` selecciona Qwen-Image, `v2.0` selecciona Qwen-Image-2512 y las variantes `edit-*` conservan sus checkpoints. Sus adaptadores y cachés de latentes no son intercambiables con los de 2.1.

<a id="vram-presets"></a>

### Configuraciones con asistente y REPA

Los ejemplos estándar combinan [el asistente v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2), regularización sintética, REPA y desplazamiento automático del calendario según la resolución. Sustituyen la receta de rendimiento de 250 pasos; consulta las imágenes en la [colección de experimentos](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-experiments). Los tiempos anteriores no representan esta receta.

Todos usan BF16, rango/alfa 32, lote 1, AdamW BF16 a `1e-4`, 25 pasos de calentamiento y recorte de norma a 1.0. REPA usa `dinov2_vitg14`, bloque 8, peso 0.5, tamaño 518, alineación espacial y distancia temporal 0. El desplazamiento automático está activado y el estático vale 0. El asistente permanece congelado y se desactiva en validación. La regularización toma como objetivo la predicción del modelo base con ambos adaptadores desactivados.

| VRAM | Ejemplo | Resoluciones base | Pasos | Intervalo de checkpointing |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 512px | 2000 | 1 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 512px | 2000 | 2 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 512px + 1024px | 4000 | 2 |

Las pruebas de memoria en L40S usaron 16 pasos, cuatro imágenes por backend, validación y guardado. La receta 512px con intervalo 1 pasó con un límite de 24 GiB: pico de 20.06 GiB asignados / 21.10 GiB reservados por PyTorch. La receta multiescala con intervalo 2 pasó en L40S con 32.49 / 41.21 GiB, pero agotó la memoria bajo un límite de 32 GiB. Son pruebas de memoria con subconjuntos pequeños, no de rendimiento ni convergencia; los límites menores se simularon en L40S. El preset final de 32 GB, con 512px e intervalo 2, también pasó: 22.12 GiB asignados / 23.68 GiB reservados.

El muestreador probabilístico habitual asigna la mitad del peso a `RareConcepts/Domokun` con el activador `🟫` y la otra mitad a `webshart/qwen-image-2.1-generated-images` con `is_regularisation_data: true`. En multiescala, cada mitad se divide por igual entre 512px y 1024px. Los buckets por área conservan las proporciones: 0.262144 y 1.048576 megapíxeles. Los subconjuntos sintéticos cuadrados, verticales y horizontales comparten por igual el peso de su resolución. No hay alternancia estricta. Se valida y guarda cada 250 pasos.

Instala `webshart`. Cada subconjunto sintético admite hasta 1.024 imágenes y tiene una caché de latentes separada. El VAE no usa mosaicos; la validación usa el VAE predeterminado con corrección de textura. Inicia un entrenamiento nuevo al cambiar datos, resolución o lote.

Los presets de 24/32 GB omiten 1024px para reservar memoria. `qwen_image.peft-lora` usa multiescala. Las mediciones antiguas no verifican la memoria de esta combinación; la compilación y la longitud de los textos también influyen. AnyFlow y la creación del asistente siguen siendo experimentos separados.

### Piloto experimental de AnyFlow

Los tres ejemplos `qwen_image-2.1-anyflow-stage*.peft-lora` prueban si la destilación de intervalos puede conservar el comportamiento de la base al introducir `🟫`. Son experimentales; la ficha de Qwen no confirma que la destilación de guidance causara el deterioro anterior.

Todas las etapas usan 1024px, BF16, AdamW, rank 32 y batch 4. La etapa 1 desactiva el checkpointing de gradientes en H200; las etapas 2 y 3 usan grupos contiguos de dos bloques. Es un entrenamiento de destilación distinto de los presets con REPA anteriores. Instala `webshart` antes de ejecutarlos.

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

<a id="assistant-lora"></a>

### Entrenar un LoRA auxiliar a partir de descripciones

`qwen_image-2.1-assistant-lora.peft-lora` es un punto de partida experimental para L40S: BF16, lote 1, checkpointing con intervalo 2, rango 32 y AdamW BF16 a `1e-4`. Presupuesta 1.000 actualizaciones y valida/guarda cada 50. Sustituye las doce descripciones de prueba por textos diversos antes de entrenar en serio; la convergencia no está validada.

`grad_clip_method: "norm"`, `max_grad_norm: 1.0`.

Selecciona `distillation_method: assistant_lora`. El backend precalcula los embeddings de texto. Cada lote genera latentes nuevos del modelo base con el adaptador desactivado, 40 pasos nativos y CFG 1. La canalización privada reutiliza el transformer sin cargar VAE, procesador ni codificador de texto. Después restaura el adaptador y ejecuta el entrenamiento de eliminación de ruido habitual. No almacena latentes finales. Actualmente solo admite texto a imagen con Qwen Image 2.1.

`distillation_config.assistant_lora` acepta `num_inference_steps` (40 por defecto), `resolutions` (lista no vacía de `[ancho, alto]`, por defecto `[[1024, 1024]]`) y `seed` (42). Las dimensiones deben ser múltiplos de 32. Las resoluciones rotan por lote y las semillas avanzan por muestra, incluidos los lotes incompletos. Los checkpoints guardan estos contadores. Reanuda sin cambiar datos, lote, acumulación ni topología distribuida. La caché de texto bajo demanda no está admitida.

Para comprobar el funcionamiento, usa 8 actualizaciones y 2 pasos del profesor; vuelve a 40 antes de evaluar imágenes. Revisa a las 100, 250, 500 y 1.000 actualizaciones con los mismos prompts y semillas del modelo base. La utilidad del auxiliar requiere otra prueba de entrenamiento de conceptos.

El entrenamiento LoRA de Qwen Image 2.1 carga por defecto el [asistente v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2), congelado durante el entrenamiento y desactivado durante la validación. Usa `disable_assistant_lora: true` para desactivarlo, o `assistant_lora_path` para elegir otro adaptador. Las versiones anteriores conservan su comportamiento. Al entrenar un nuevo asistente, mantén `disable_assistant_lora: true`, como en los ejemplos correspondientes.

Usa este como único método de destilación; no se admite combinarlo con otros destiladores.

Los datasets de descripciones requieren `dataloader_prefetch: false` para que el cursor del checkpoint corresponda a las descripciones consumidas. La reanudación rechaza cambios en identidades/textos, lote, repeticiones, mezcla, semilla, acumulación o distribución. Los checkpoints del auxiliar también rechazan cambios en la semilla de generación, la lista de resoluciones o los pasos de inferencia del profesor.

<a id="assistant-lora-multires"></a>

#### Experimento de asistente con cuatro resoluciones base y buckets de aspecto

`qwen_image-2.1-assistant-lora-multires.peft-lora` recorre 12 buckets de relación de aspecto repartidos entre las resoluciones base 512, 1024, 1536 y 2048. Cada base incluye un bucket cuadrado, uno vertical 4:7 y otro horizontal 7:4; cada lote usa un bucket, con la misma exposición por base y aspecto durante un ciclo completo. Conserva los 40 pasos del profesor, BF16 con lote 1, checkpointing cada 2 bloques, rango 32, AdamW BF16 y 1.000 actualizaciones del ejemplo base; comparte su backend de captions y sus prompts de validación. Sustituye los captions de prueba por el mismo conjunto diverso usado en la referencia. Empieza con un adaptador y un optimizador nuevos en el nuevo directorio de salida; `resume_from_checkpoint: ""` desactiva la reanudación. Conserva los pesos de la primera ejecución para compararlos.

Esto prueba si exponer al asistente a varios tamaños mejora el resultado; no demuestra un requisito de resolución nativa ni una mejora de calidad. La ejecución de 1.000 actualizaciones en L40S completó los 12 buckets, con validación a 1024×1024 y 2048×2048. El zorro y el retrato finales conservaron coherencia, con las conocidas líneas de color del VAE con tiling. El profesor genera objetivos latentes sin decodificar con el VAE. La mejora en el entrenamiento posterior de conceptos sigue sin verificarse.

Assistant LoRA usa el mismo mínimo de 32 entradas de caché Dynamo, limitado a la ejecución, que AnyFlow para las variantes del profesor, alumno y validación. Respeta límites mayores definidos por el usuario y restaura el original al salir. Vigila las recompilaciones al añadir resoluciones o captions más largos.

Una prueba posterior de Domokun entrenó dos adaptadores nuevos durante 250 actualizaciones cada uno a 2048px, batch 1, LR `1e-5` y clipping por valor de 1.0. La intensidad del asistente durante el entrenamiento fue 0 para el control y 1 para la otra ejecución; ambas lo desactivaron durante la inferencia. Los pesos iniciales y las imágenes de validación iniciales coincidieron exactamente. A 40 pasos de inferencia y 1024px, ambos resultados finales siguieron generando personas para los prompts del personaje y conservaron zorros/retratos coherentes; no se demostró una ventaja del asistente. Ambas ejecuciones y la preparación del asistente usaron el optimizador sin corregir descrito [arriba](#qwen21-optimizer-correction).

<a id="assistant-lora-offline"></a>

#### LoRA auxiliar con imágenes generadas reutilizables

`qwen_image-2.1-assistant-lora-offline.peft-lora` entrena con [10.000 imágenes generadas](https://huggingface.co/datasets/webshart/qwen-image-2.1-generated-images) mediante Webshart. El dataset usa prompts `long_caption` de CC12M, 40 pasos nativos del profesor, CFG 1 y decodificación VAE de imagen completa. Doce backends cubren buckets cuadrados, verticales y horizontales con resoluciones base de 512, 1024, 1536 y 2048, pesos de muestreo iguales y sin repeticiones.

La receta inicia un entrenamiento nuevo con `adamw_bf16` corregido, LR `1e-4`, `grad_clip_method: "norm"`, `max_grad_norm: 1.0`, BF16 con lote 1, rango 32 y checkpointing cada 2 bloques. Presupuesta 1.000 actualizaciones, guarda cada 50 y valida cada 100 a 1024. Genera vistas previas adicionales a 2048px antes de publicar. Desactiva el tiling del VAE y codifica con lote 1. `vae_cache_ondemand: true` codifica y almacena cada imagen al muestrearla, evitando codificar las 10.000 imágenes antes de 1.000 actualizaciones. El entrenamiento habitual con imágenes sustituye la generación del profesor en línea; omite `distillation_method` y conserva `disable_assistant_lora: true` al crear el auxiliar.

Puedes reutilizar el dataset en otros experimentos. Los PNG requieren recodificación VAE y no equivalen exactamente a los latentes finales del profesor. Revisa las imágenes de validación antes de publicar y evalúa la utilidad del auxiliar en otro entrenamiento de conceptos. Al cambiar desde la receta de captions, empieza de cero sin reanudar el estado del optimizador ni del dataset.

El [asistente v1 de reemplazo](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v1) completó esta receta de 1.000 actualizaciones en L40S, con revisión final de imágenes a 1024 y 2048. Una comparación equivalente de Domokun durante 1.000 actualizaciones a 2048, LR `1e-4` y clipping de norma 1.0 mantuvo el asistente congelado durante el entrenamiento y desactivado en validación. Tanto el control como la ejecución asistida siguieron generando personas para los dos prompts del personaje. El control introdujo rasgos marcados de Domokun en un prompt de zorro no relacionado; la ejecución asistida conservó un zorro reconocible. Ambas conservaron un retrato coherente. Es evidencia limitada de menor contaminación entre conceptos, no de aprendizaje exitoso del personaje ni de una mejora general de calidad; la evaluación usa una semilla de entrenamiento y cuatro prompts.

```bash
simpletuner train example=qwen_image-2.1-assistant-lora-offline.peft-lora
```
