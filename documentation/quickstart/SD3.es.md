## Stable Diffusion 3

En este ejemplo, entrenaremos un modelo Stable Diffusion 3 usando el toolkit SimpleTuner y usaremos el tipo de modelo `lora`.

### Requisitos previos

Asegúrate de tener Python instalado; SimpleTuner funciona bien con 3.10 a 3.12.

Puedes comprobarlo ejecutando:

```bash
python --version
```

Si no tienes Python 3.12 instalado en Ubuntu, puedes intentar lo siguiente:

```bash
apt -y install python3.13 python3.13-venv
```

#### Dependencias de imagen de contenedor

Para Vast, RunPod y TensorDock (entre otros), lo siguiente funcionará en una imagen CUDA 12.2-12.8 para permitir compilar extensiones CUDA:

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

Para instalación manual o setup de desarrollo, consulta la [documentación de instalación](../INSTALL.md).

#### Pasos adicionales para AMD ROCm

Lo siguiente debe ejecutarse para que un AMD MI300X sea utilizable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

### Configuración del entorno

Para ejecutar SimpleTuner, necesitarás configurar un archivo de configuración, los directorios de dataset y modelo, y un archivo de configuración del dataloader.

#### Archivo de configuración

Un script experimental, `configure.py`, puede permitirte omitir por completo esta sección mediante una configuración interactiva paso a paso. Incluye funciones de seguridad que ayudan a evitar errores comunes.

**Nota:** Esto no configura tu dataloader. Aún tendrás que hacerlo manualmente, más adelante.

Para ejecutarlo:

```bash
simpletuner configure
```

> ⚠️ Para usuarios ubicados en países donde Hugging Face Hub no es fácilmente accesible, deberías añadir `HF_ENDPOINT=https://hf-mirror.com` a tu `~/.bashrc` o `~/.zshrc` dependiendo de qué `$SHELL` use tu sistema.

Si prefieres configurar manualmente:

Copia `config/config.json.example` a `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Ahí, necesitarás modificar las siguientes variables:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "model_type": "lora",
  "model_family": "sd3",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
  "output_dir": "/home/user/outputs/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.0,
  "validation_prompt": "your main test prompt here",
  "user_prompt_library": "config/user_prompt_library.json"
}
```
</details>


- `pretrained_model_name_or_path` - Configura esto en `stabilityai/stable-diffusion-3.5-large`. Ten en cuenta que necesitarás iniciar sesión en Huggingface y que se te otorgue acceso para descargar este modelo. Más adelante en este tutorial veremos cómo iniciar sesión en Huggingface.
  - Si prefieres entrenar el SD3.0 Medium (2B) más antiguo, usa `stabilityai/stable-diffusion-3-medium-diffusers` en su lugar.
- `MODEL_TYPE` - Configura esto en `lora`.
- `MODEL_FAMILY` - Configura esto en `sd3`.
- `OUTPUT_DIR` - Configura esto en el directorio donde quieres guardar tus checkpoints e imágenes de validación. Se recomienda usar una ruta completa.
- `VALIDATION_RESOLUTION` - Como SD3 es un modelo de 1024px, puedes establecerlo en `1024x1024`.
  - Además, SD3 fue afinado con buckets multi-aspect, y se pueden especificar otras resoluciones separándolas con comas: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - SD3 se beneficia de un valor muy bajo. Configura esto en `3.0`.

Hay algunas más si usas una máquina Mac M-series:

- `mixed_precision` debería configurarse en `no`.

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de salida dejando que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

#### Entrenamiento de modelos cuantizados

Probado en sistemas Apple y NVIDIA, Hugging Face Optimum-Quanto puede usarse para reducir la precisión y los requisitos de VRAM muy por debajo de los requisitos del entrenamiento base de SDXL.



> ⚠️ Si usas un archivo de configuración JSON, asegúrate de usar este formato en `config.json` en lugar de `config.env`:

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "text_encoder_3_precision": "no_change",
  "optimizer": "adamw_bf16"
}
```

Para usuarios de `config.env` (obsoleto):

```bash
</details>

# choices: int8-quanto, int4-quanto, int2-quanto, fp8-quanto
# int8-quanto was tested with a single subject dreambooth LoRA.
# fp8-quanto does not work on Apple systems. you must use int levels.
# int2-quanto is pretty extreme and gets the whole rank-1 LoRA down to about 13.9GB VRAM.
# may the gods have mercy on your soul, should you push things Too Far.
export TRAINER_EXTRA_ARGS="--base_model_precision=int8-quanto"

# Maybe you want the text encoders to remain full precision so your text embeds are cake.
# We unload the text encoders before training, so, that's not an issue during training time - only during pre-caching.
# Alternatively, you can go ham on quantisation here and run them in int4 or int8 mode, because no one can stop you.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --text_encoder_1_precision=no_change --text_encoder_2_precision=no_change"

# When you're quantising the model, --base_model_default_dtype is set to bf16 by default. This setup requires adamw_bf16, but saves the most memory.
# adamw_bf16 only supports bf16 training, but any other optimiser will support both bf16 or fp32 training precision.
export OPTIMIZER="adamw_bf16"
```

#### Consideraciones de dataset

Es crucial contar con un dataset sustancial para entrenar tu modelo. Hay limitaciones en el tamaño del dataset y debes asegurarte de que tu dataset sea lo suficientemente grande para entrenar el modelo de manera efectiva. Ten en cuenta que el tamaño mínimo del dataset es `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS` así como más que `VAE_BATCH_SIZE`. El dataset no será usable si es demasiado pequeño.

Dependiendo del dataset que tengas, necesitarás configurar el directorio de dataset y el archivo de configuración del dataloader de forma diferente. En este ejemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

En tu directorio `/home/user/simpletuner/config`, crea un multidatabackend.json:

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sd3",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 0,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/home/user/simpletuner/output/cache/vae/sd3/pseudo-camera-10k",
    "instance_data_dir": "/home/user/simpletuner/datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sd3/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

Luego, crea un directorio `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
popd
```

Esto descargará alrededor de 10k muestras de fotografías a tu directorio `datasets/pseudo-camera-10k`, que se creará automáticamente.

#### Inicia sesión en WandB y Huggingface Hub

Querrás iniciar sesión en WandB y en HF Hub antes de comenzar el entrenamiento, especialmente si estás usando `push_to_hub: true` y `--report_to=wandb`.

Si vas a subir elementos manualmente a un repositorio Git LFS, también deberías ejecutar `git config --global credential.helper store`

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


Desde el directorio SimpleTuner, solo tienes que ejecutar:

```bash
bash train.sh
```

Esto iniciará el caché en disco de los embeddings de texto y las salidas VAE.

Para más información, consulta los documentos [dataloader](../DATALOADER.md) y [tutorial](../TUTORIAL.md).

## Notas y consejos de solución de problemas

### Skip-layer guidance (SD3.5 Medium)

StabilityAI recomienda habilitar SLG (Skip-layer guidance) en inferencia de SD 3.5 Medium. Esto no impacta los resultados de entrenamiento, solo la calidad de las muestras de validación.

Se recomiendan los siguientes valores para `config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "--validation_guidance_skip_layers": [7, 8, 9],
  "--validation_guidance_skip_layers_start": 0.01,
  "--validation_guidance_skip_layers_stop": 0.2,
  "--validation_guidance_skip_scale": 2.8,
  "--validation_guidance": 4.0,
  "--flow_use_uniform_schedule": true,
  "--flow_schedule_auto_shift": true
}
```
</details>

- `..skip_scale` determina cuánto escalar la predicción del prompt positivo durante skip-layer guidance. El valor predeterminado de 2.8 es seguro para el skip base del modelo de `7, 8, 9`, pero deberá aumentarse si se omiten más capas, duplicándolo por cada capa adicional.
- `..skip_layers` indica qué capas omitir durante la predicción del prompt negativo.
- `..skip_layers_start` determina la fracción de la canalización de inferencia durante la cual debe comenzar a aplicarse skip-layer guidance.
- `..skip_layers_stop` configurará la fracción del total de pasos de inferencia después de la cual SLG ya no se aplicará.

SLG puede aplicarse por menos pasos para un efecto más débil o una menor reducción de la velocidad de inferencia.

Parece que el entrenamiento extensivo de un modelo LoRA o LyCORIS requerirá modificar estos valores, aunque no está claro exactamente cómo cambia.

**Debe usarse CFG más bajo durante la inferencia.**

### Inestabilidad del modelo

El modelo SD 3.5 Large 8B tiene posibles inestabilidades durante el entrenamiento:

- Valores altos de `--max_grad_norm` permitirán que el modelo explore actualizaciones de peso potencialmente peligrosas
- Las tasas de aprendizaje pueden ser extremadamente sensibles; `1e-5` funciona con StableAdamW pero `4e-5` podría explotar
- Batch sizes más altos ayudan **mucho**
- La estabilidad no se ve afectada al desactivar la cuantización o entrenar en fp32 puro

El código oficial de entrenamiento no se publicó junto con SD3.5, dejando a los desarrolladores adivinar cómo implementar el bucle de entrenamiento basado en el [contenido del repositorio SD3.5](https://github.com/stabilityai/sd3.5).

Se hicieron algunos cambios en el soporte SD3.5 de SimpleTuner:
- Excluir más capas de la cuantización
- Ya no poner en cero el padding de T5 por defecto (`--t5_padding`)
- Ofrecer un switch (`--sd3_clip_uncond_behaviour` y `--sd3_t5_uncond_behaviour`) para usar captions vacías codificadas para predicciones incondicionales (`empty_string`, **predeterminado**) o ceros (`zero`), no es un ajuste recomendado.
- La función de pérdida de entrenamiento de SD3.5 se actualizó para coincidir con la encontrada en el repositorio upstream StabilityAI/SD3.5
- Se actualizó el valor predeterminado de `--flow_schedule_shift` a 3 para coincidir con el valor estático 1024px de SD3
  - StabilityAI publicó documentación para usar `--flow_schedule_shift=1` con `--flow_use_uniform_schedule`
  - Miembros de la comunidad han reportado que `--flow_schedule_auto_shift` funciona mejor cuando se usa entrenamiento multi-aspect o multi-resolución
- Se actualizó el límite fijo de longitud de secuencia del tokenizador a **154** con la opción de revertirlo a **77** tokens para ahorrar espacio en disco o cómputo a costa de degradación de calidad de salida


#### Valores de configuración estables

Estas opciones han sido conocidas por mantener SD3.5 intacto el mayor tiempo posible:
- optimizer=adamw_bf16
- flow_schedule_shift=1
- learning_rate=1e-4
- batch_size=4 * 3 GPUs
- max_grad_norm=0.1
- base_model_precision=int8-quanto
- Sin enmascaramiento de pérdida ni regularización de dataset, ya que su contribución a esta inestabilidad es desconocida
- `validation_guidance_skip_layers=[7,8,9]`

### Configuración de VRAM más baja

- SO: Ubuntu Linux 24
- GPU: Un solo dispositivo NVIDIA CUDA (10G, 12G)
- Memoria del sistema: ~50G de memoria del sistema aproximadamente
- Precisión del modelo base: `nf4-bnb`
- Optimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolución: 512px
- Batch size: 1, cero pasos de acumulación de gradiente
- DeepSpeed: deshabilitado / sin configurar
- PyTorch: 2.5

### SageAttention

Al usar `--attention_mechanism=sageattention`, la inferencia puede acelerarse en tiempo de validación.

**Nota**: Esto no es compatible con _todas_ las configuraciones de modelo, pero vale la pena probarlo.

### Pérdida enmascarada

Si estás entrenando un sujeto o estilo y quieres enmascarar uno u otro, consulta la sección [entrenamiento con pérdida enmascarada](../DREAMBOOTH.md#masked-loss) de la guía Dreambooth.

### Datos de regularización

Para más información sobre datasets de regularización, consulta [esta sección](../DREAMBOOTH.md#prior-preservation-loss) y [esta sección](../DREAMBOOTH.md#regularisation-dataset-considerations) de la guía Dreambooth.

### Entrenamiento cuantizado

Consulta [esta sección](../DREAMBOOTH.md#quantised-model-training-loralycoris-only) de la guía Dreambooth para información sobre cómo configurar la cuantización para SD3 y otros modelos.

### Seguimiento de puntuación CLIP

Si deseas habilitar evaluaciones para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/CLIP_SCORES.md) para información sobre cómo configurar e interpretar las puntuaciones CLIP.

# Pérdida de evaluación estable

Si deseas usar pérdida MSE estable para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/EVAL_LOSS.md) para información sobre cómo configurar e interpretar la pérdida de evaluación.

#### Vistas previas de validación

SimpleTuner soporta streaming de vistas previas intermedias de validación durante la generación usando modelos Tiny AutoEncoder. Esto te permite ver imágenes de validación generándose paso a paso en tiempo real vía callbacks de webhook.

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

Configura `validation_preview_steps` a un valor más alto (p. ej., 3 o 5) para reducir el overhead del Tiny AutoEncoder. Con `validation_num_inference_steps=20` y `validation_preview_steps=5`, recibirás vistas previas en los pasos 5, 10, 15 y 20.
