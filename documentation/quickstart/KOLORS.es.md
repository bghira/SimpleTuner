## Guía rápida de Kwai Kolors

En este ejemplo, entrenaremos un modelo Kwai Kolors usando el toolkit SimpleTuner y el tipo de modelo `lora`.

Kolors es aproximadamente del tamaño de SDXL, así que puedes intentar entrenamiento `full`, pero los cambios para eso no se describen en esta guía rápida.

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

#### Pasos adicionales para AMD ROCm

Lo siguiente debe ejecutarse para que una AMD MI300X sea utilizable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

Allí, necesitarás modificar las siguientes variables:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "model_type": "lora",
  "model_family": "kolors",
  "pretrained_model_name_or_path": "Kwai-Kolors/Kolors-diffusers",
  "output_dir": "/home/user/output/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.4,
  "use_gradient_checkpointing": true,
  "learning_rate": 1e-4
}
```
</details>

- `pretrained_model_name_or_path` - Configúralo en `Kwai-Kolors/Kolors-diffusers`.
- `MODEL_TYPE` - Configúralo en `lora`.
- `USE_DORA` - Configúralo en `true` si deseas entrenar DoRA.
- `MODEL_FAMILY` - Configúralo en `kolors`.
- `OUTPUT_DIR` - Configúralo al directorio donde quieres guardar tus checkpoints y las imágenes de validación. Se recomienda usar una ruta completa aquí.
- `VALIDATION_RESOLUTION` - Configúralo en `1024x1024` para este ejemplo.
  - Además, Kolors fue ajustado en buckets multi-aspecto, y otras resoluciones pueden especificarse separándolas con comas: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - Usa el valor con el que te sientas cómodo para probar en inferencia. Configúralo entre `4.2` y `6.4`.
- `USE_GRADIENT_CHECKPOINTING` - Probablemente debería ser `true` a menos que tengas MUCHA VRAM y quieras sacrificar algo para hacerlo más rápido.
- `LEARNING_RATE` - `1e-4` es bastante común para redes de bajo rango, aunque `1e-5` podría ser una opción más conservadora si notas "burning" o sobreentrenamiento temprano.

Hay algunas configuraciones adicionales si usas una máquina Mac M-series:

- `mixed_precision` debería configurarse en `no`.
- `attention_mechanism` debería configurarse en `diffusers`, ya que `xformers` y otros valores probablemente no funcionen.

#### Entrenamiento con modelo cuantizado

Probado en sistemas Apple y NVIDIA, Hugging Face Optimum-Quanto puede usarse para reducir la precisión y los requisitos de VRAM, especialmente para ChatGLM 6B (el codificador de texto).

Para `config.json`:
<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "optimizer": "adamw_bf16"
}
```
</details>

Para usuarios de `config.env` (deprecated):

```bash
# choices: int8-quanto, int4-quanto, int2-quanto, fp8-quanto
# int8-quanto was tested with a single subject dreambooth LoRA.
# fp8-quanto does not work on Apple systems. you must use int levels.
# int2-quanto is pretty extreme and gets the whole rank-1 LoRA down to about 13.9GB VRAM.
# may the gods have mercy on your soul, should you push things Too Far.
export TRAINER_EXTRA_ARGS="--base_model_precision=int8-quanto"

# Maybe you want the text encoders to remain full precision so your text embeds are cake.
# We unload the text encoders before training, so, that's not an issue during training time - only during pre-caching.
# Alternatively, you can go ham on quantisation here and run them in int4 or int8 mode, because no one can stop you.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --text_encoder_1_precision=no_change"

# When you're quantising the model, --base_model_default_dtype is set to bf16 by default. This setup requires adamw_bf16, but saves the most memory.
# adamw_bf16 only supports bf16 training, but any other optimiser will support both bf16 or fp32 training precision.
export OPTIMIZER="adamw_bf16"
```

#### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.
*   **[Diff2Flow](../experimental/DIFF2FLOW.md):** permite entrenar Kolors con un objetivo Flow Matching, lo que puede mejorar la linealidad y la calidad de la generación.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

</details>

#### Consideraciones del dataset

Es crucial contar con un dataset sustancial para entrenar tu modelo. Hay limitaciones en el tamaño del dataset, y debes asegurarte de que sea lo suficientemente grande para entrenar tu modelo de forma efectiva. Ten en cuenta que el tamaño mínimo del dataset es `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`. El dataset no será descubrible por el entrenador si es demasiado pequeño.

Dependiendo del dataset que tengas, necesitarás configurar el directorio del dataset y el archivo de configuración del dataloader de manera diferente. En este ejemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

En tu directorio `OUTPUT_DIR`, crea un multidatabackend.json:

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-kolors",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 1.0,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/kolors/pseudo-camera-10k",
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
    "cache_dir": "cache/text/kolors/pseudo-camera-10k",
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
huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=datasets/pseudo-camera-10k
```

Esto descargará alrededor de 10k muestras de fotografías a tu directorio `datasets/pseudo-camera-10k`, que se creará automáticamente.

#### Iniciar sesión en WandB y Huggingface Hub

Querrás iniciar sesión en WandB y HF Hub antes de empezar el entrenamiento, especialmente si usas `push_to_hub: true` y `--report_to=wandb`.

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

Desde el directorio de SimpleTuner, solo hay que ejecutar:

```bash
bash train.sh
```

Esto iniciará el caché a disco de text embeds y salidas del VAE.

Para más información, consulta los documentos de [dataloader](../DATALOADER.md) y [tutorial](../TUTORIAL.md).

### Seguimiento de puntuaciones CLIP

Si deseas habilitar evaluaciones para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/CLIP_SCORES.md) para información sobre cómo configurar e interpretar las puntuaciones CLIP.

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
