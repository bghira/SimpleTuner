# Guía rápida de Kandinsky 5.0 Image

En este ejemplo, entrenaremos un LoRA de Kandinsky 5.0 Image.

## Requisitos de hardware

Kandinsky 5.0 emplea un **enorme codificador de texto Qwen2.5-VL de 7B parámetros** además de un codificador CLIP estándar y el VAE de Flux. Esto exige mucho tanto de VRAM como de RAM del sistema.

Solo cargar el codificador Qwen requiere alrededor de **14GB** de memoria por sí solo. Al entrenar un LoRA de rango 16 con gradient checkpointing completo:

- **24GB de VRAM** es el mínimo cómodo (RTX 3090/4090).
- **16GB de VRAM** es posible pero requiere offloading agresivo y probablemente cuantización `int8` del modelo base.

Necesitarás:

- **RAM del sistema**: Al menos 32GB, idealmente 64GB, para manejar la carga inicial del modelo sin fallos.
- **GPU**: NVIDIA RTX 3090 / 4090 o tarjetas profesionales (A6000, A100, etc.).

### Offloading de memoria (recomendado)

Dado el tamaño del codificador de texto, casi seguro deberías usar offloading agrupado si estás en hardware de consumo. Esto descarga los bloques del transformer a memoria CPU cuando no se están computando activamente.

Agrega lo siguiente a tu `config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "group_offload_blocks_per_group": 1,
  "group_offload_use_stream": true
}
```
</details>

- `--group_offload_use_stream`: Solo funciona en dispositivos CUDA.
- **No** combines esto con `--enable_model_cpu_offload`.

Además, configura `"offload_during_startup": true` en tu `config.json` para reducir el uso de VRAM durante la fase de inicialización y caché. Esto asegura que el codificador de texto y el VAE no se carguen simultáneamente.

## Requisitos previos

Asegúrate de tener python instalado; SimpleTuner funciona bien con 3.10 a 3.12.

Puedes comprobarlo ejecutando:

```bash
python --version
```

Si no tienes python 3.12 instalado en Ubuntu, puedes intentar lo siguiente:

```bash
apt -y install python3.13 python3.13-venv
```

## Instalación

Instala SimpleTuner vía pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Para instalación manual o entorno de desarrollo, consulta la [documentación de instalación](../INSTALL.md).

## Configuración del entorno

### Método de interfaz web

La WebUI de SimpleTuner hace que la configuración sea bastante sencilla. Para ejecutar el servidor:

```bash
simpletuner server
```

Accede en http://localhost:8001.

### Método manual / línea de comandos

Para ejecutar SimpleTuner mediante herramientas de línea de comandos, necesitarás configurar un archivo de configuración, los directorios del dataset y del modelo, y un archivo de configuración del dataloader.

#### Archivo de configuración

Un script experimental, `configure.py`, puede ayudarte a omitir esta sección:

```bash
simpletuner configure
```

Si prefieres configurarlo manualmente:

Copia `config/config.json.example` a `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Necesitarás modificar las siguientes variables:

- `model_type`: `lora`
- `model_family`: `kandinsky5-image`
- `model_flavour`:
  - `t2i-lite-sft`: (Predeterminado) El checkpoint SFT estándar. Mejor para ajustar estilos/personajes.
  - `t2i-lite-pretrain`: El checkpoint pretrain. Mejor para enseñar conceptos totalmente nuevos desde cero.
  - `i2i-lite-sft` / `i2i-lite-pretrain`: Para entrenamiento de imagen a imagen. Requiere imágenes de condicionamiento en tu dataset.
- `output_dir`: Dónde guardar tus checkpoints.
- `train_batch_size`: Comienza con `1`.
- `gradient_accumulation_steps`: Usa `1` o más para simular lotes más grandes.
- `validation_resolution`: `1024x1024` es estándar para este modelo.
- `validation_guidance`: `5.0` es el valor recomendado por defecto para Kandinsky 5.
- `flow_schedule_shift`: `1.0` es el valor predeterminado. Ajustarlo cambia cómo el modelo prioriza detalles vs composición (ver abajo).

#### Prompts de validación

Dentro de `config/config.json` está el "prompt de validación principal". También puedes crear una biblioteca de prompts en `config/user_prompt_library.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "portrait": "A high quality portrait of a woman, cinematic lighting, 8k",
  "landscape": "A beautiful mountain landscape at sunset, oil painting style"
}
```
</details>

Habilítalo agregando esto a tu `config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "user_prompt_library": "config/user_prompt_library.json"
}
```
</details>

#### Desplazamiento del calendario de flujo

Kandinsky 5 es un modelo de flow-matching. El parámetro `shift` controla la distribución de ruido durante entrenamiento e inferencia.

- **Shift 1.0 (Predeterminado)**: Entrenamiento equilibrado.
- **Shift bajo (< 1.0)**: Enfoca el entrenamiento más en detalles de alta frecuencia (textura, ruido).
- **Shift alto (> 1.0)**: Enfoca el entrenamiento más en detalles de baja frecuencia (composición, color, estructura).

Si tu modelo aprende estilos bien pero falla en composición, intenta aumentar el shift. Si aprende composición pero le falta textura, intenta disminuirlo.

#### Entrenamiento con modelo cuantizado

Puedes reducir significativamente el uso de VRAM cuantizando el transformer a 8-bit.

En `config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "base_model_default_dtype": "bf16"
```
</details>

> **Nota**: No recomendamos cuantizar los codificadores de texto (`no_change`) ya que Qwen2.5-VL es sensible a los efectos de cuantización y ya es la parte más pesada del pipeline.

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

#### Consideraciones del dataset

Necesitarás un archivo de configuración del dataset, p. ej., `config/multidatabackend.json`.

```json
[
  {
    "id": "my-image-dataset",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/my_images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "crop": true,
    "crop_aspect": "square",
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/kandinsky5",
    "disabled": false
  }
]
```

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

Luego crea el directorio del dataset:

```bash
mkdir -p datasets/my_images
</details>

# Copy your images and .txt caption files here
```

#### Iniciar sesión en WandB y Huggingface Hub

```bash
wandb login
huggingface-cli login
```

### Ejecutar el entrenamiento

**Opción 1 (Recomendada):**

```bash
simpletuner train
```

**Opción 2 (Heredada):**

```bash
./train.sh
```

## Notas y consejos de solución de problemas

### Configuración de VRAM más baja

Para ejecutar en setups de 16GB o 24GB limitados:

1.  **Habilita Group Offload**: `--enable_group_offload`.
2.  **Cuantiza el modelo base**: Configura `"base_model_precision": "int8-quanto"`.
3.  **Tamaño de lote**: Mantenlo en `1`.

### Artefactos e imágenes "quemadas"

Si las imágenes de validación se ven sobresaturadas o ruidosas ("quemadas"):

- **Revisa Guidance**: Asegúrate de que `validation_guidance` esté alrededor de `5.0`. Valores más altos (como 7.0+) suelen estropear la imagen en este modelo.
- **Revisa Flow Shift**: Valores extremos de `flow_schedule_shift` pueden causar inestabilidad. Mantén `1.0` al inicio.
- **Tasa de aprendizaje**: 1e-4 es estándar para LoRA, pero si ves artefactos, intenta bajar a 5e-5.

### Entrenamiento TREAD

Kandinsky 5 admite [TREAD](../TREAD.md) para entrenar más rápido eliminando tokens.

Agrega a `config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

Esto elimina el 50% de los tokens en las capas centrales, acelerando el paso del transformer.
