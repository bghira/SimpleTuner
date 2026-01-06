## Inicio rápido de PixArt Sigma

En este ejemplo, entrenaremos un modelo PixArt Sigma usando el toolkit SimpleTuner y usaremos el tipo de modelo `full`, ya que al ser un modelo más pequeño probablemente quepa en VRAM.

### Requisitos previos

Asegúrate de tener Python instalado; SimpleTuner funciona bien con 3.10 a 3.12.

Puedes comprobarlo ejecutando:

```bash
python --version
```

Si no tienes Python 3.12 instalado en Ubuntu, puedes intentar lo siguiente:

```bash
apt -y install python3.12 python3.12-venv
```

#### Dependencias de imagen de contenedor

Para Vast, RunPod y TensorDock (entre otros), lo siguiente funcionará en una imagen CUDA 12.2-12.8 para permitir compilar extensiones CUDA:

```bash
apt -y install nvidia-cuda-toolkit
```

### Instalación

Instala SimpleTuner vía pip:

```bash
pip install simpletuner[cuda]
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
  "model_type": "full",
  "use_bitfit": false,
  "pretrained_model_name_or_path": "pixart-alpha/pixart-sigma-xl-2-1024-ms",
  "model_family": "pixart_sigma",
  "output_dir": "/home/user/output/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.5
}
```
</details>

- `pretrained_model_name_or_path` - Configura esto en `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`.
- `MODEL_TYPE` - Configura esto en `full`.
- `USE_BITFIT` - Configura esto en `false`.
- `MODEL_FAMILY` - Configura esto en `pixart_sigma`.
- `OUTPUT_DIR` - Configura esto en el directorio donde quieres guardar tus checkpoints e imágenes de validación. Se recomienda usar una ruta completa.
- `VALIDATION_RESOLUTION` - Como PixArt Sigma viene en formato 1024px o 2048xp, deberías configurar esto con cuidado en `1024x1024` para este ejemplo.
  - Además, PixArt fue afinado con buckets multi-aspect, y se pueden especificar otras resoluciones separándolas con comas: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - PixArt se beneficia de un valor muy bajo. Configura esto entre `3.6` y `4.4`.

Hay algunas más si usas una máquina Mac M-series:

- `mixed_precision` debería configurarse en `no`.

> 💡 **Consejo:** Para datasets grandes donde el espacio en disco es una preocupación, puedes usar `--vae_cache_disable` para realizar la codificación VAE en línea sin cachear los resultados a disco.

#### Consideraciones de dataset

Es crucial contar con un dataset sustancial para entrenar tu modelo. Hay limitaciones en el tamaño del dataset y debes asegurarte de que sea lo suficientemente grande para entrenar el modelo de manera efectiva. Ten en cuenta que el tamaño mínimo del dataset es `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`. El dataset no será detectable por el trainer si es demasiado pequeño.

Dependiendo del dataset que tengas, necesitarás configurar el directorio de dataset y el archivo de configuración del dataloader de forma diferente. En este ejemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

En tu directorio `/home/user/simpletuner/config`, crea un multidatabackend.json:

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-pixart",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 1.0,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/pixart/pseudo-camera-10k",
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
    "cache_dir": "cache/text/pixart/pseudo-camera-10k",
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

### SageAttention

Al usar `--attention_mechanism=sageattention`, la inferencia puede acelerarse en tiempo de validación.

**Nota**: Esto no es compatible con _todas_ las configuraciones de modelo, pero vale la pena probarlo.

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de salida dejando que el modelo genere sus propias entradas durante el entrenamiento.
*   **[Diff2Flow](../experimental/DIFF2FLOW.md):** permite entrenar con un objetivo Flow Matching, potencialmente mejorando la rectitud y la calidad de generación.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.
</details>
