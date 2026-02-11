# Guía rápida de FLUX.2

Esta guía cubre el entrenamiento de LoRAs en FLUX.2, la familia de modelos de generación de imágenes más reciente de Black Forest Labs.

> **Nota**: El flavour predeterminado es `klein-9b`, pero esta guía se centra en `dev` (el transformer completo de 12B con codificador de texto Mistral-3 de 24B) ya que tiene los mayores requisitos de recursos. Los modelos Klein son más fáciles de ejecutar - consulta [Variantes del modelo](#variantes-del-modelo) a continuación.

## Variantes del modelo

FLUX.2 viene en tres variantes:

| Variante | Transformer | Codificador de texto | Bloques totales | Predeterminado |
|---------|-------------|----------------------|-----------------|----------------|
| `dev` | 12B parámetros | Mistral-3 (24B) | 56 (8+48) | |
| `klein-9b` | 9B parámetros | Qwen3 (incluido) | 32 (8+24) | ✓ |
| `klein-4b` | 4B parámetros | Qwen3 (incluido) | 25 (5+20) | |

**Diferencias clave:**
- **dev**: Usa codificador de texto Mistral-Small-3.1-24B independiente, tiene embeddings de guía
- **modelos klein**: Usan codificador de texto Qwen3 incluido en el repositorio del modelo, **sin embeddings de guía** (las opciones de entrenamiento de guía se ignoran)

Para seleccionar una variante, configura `model_flavour` en tu configuración:
```json
{
  "model_flavour": "dev"
}
```

## Resumen del modelo

FLUX.2-dev introduce cambios arquitectónicos significativos respecto a FLUX.1:

- **Codificador de texto**: Mistral-Small-3.1-24B (dev) o Qwen3 (klein)
- **Arquitectura**: 8 DoubleStreamBlocks + 48 SingleStreamBlocks (dev)
- **Canales latentes**: 32 canales VAE → 128 después de pixel shuffle (vs 16 en FLUX.1)
- **VAE**: VAE personalizado con batch normalization y pixel shuffling
- **Dimensión de embedding**: 15,360 para dev (3×5,120), 12,288 para klein-9b (3×4,096), 7,680 para klein-4b (3×2,560)

## Requisitos de hardware

Los requisitos de hardware varían significativamente según la variante del modelo.

### Modelos Klein (Recomendado para la mayoría de usuarios)

Los modelos Klein son mucho más accesibles:

| Variante | VRAM bf16 | VRAM int8 | RAM del sistema |
|---------|-----------|-----------|-----------------|
| `klein-4b` | ~12GB | ~8GB | 32GB+ |
| `klein-9b` | ~22GB | ~14GB | 64GB+ |

**Recomendado para klein-9b**: GPU única de 24GB (RTX 3090/4090, A5000)
**Recomendado para klein-4b**: GPU única de 16GB (RTX 4080, A4000)

### FLUX.2-dev (Avanzado)

FLUX.2-dev tiene requisitos de recursos significativos debido al codificador de texto Mistral-3:

#### Requisitos de VRAM

El codificador de texto Mistral 24B por sí solo requiere VRAM significativa:

| Componente | bf16 | int8 | int4 |
|-----------|------|------|------|
| Mistral-3 (24B) | ~48GB | ~24GB | ~12GB |
| Transformer FLUX.2 | ~24GB | ~12GB | ~6GB |
| VAE + overhead | ~4GB | ~4GB | ~4GB |

| Configuración | VRAM total aproximada |
|--------------|------------------------|
| todo en bf16 | ~76GB+ |
| codificador int8 + transformer bf16 | ~52GB |
| todo en int8 | ~40GB |
| codificador int4 + transformer int8 | ~22GB |

#### RAM del sistema

- **Mínimo**: 96GB de RAM del sistema (cargar el codificador de texto de 24B requiere mucha memoria)
- **Recomendado**: 128GB+ para un funcionamiento cómodo

#### Hardware recomendado

- **Mínimo**: 2x GPUs de 48GB (A6000, L40S) con FSDP2 o DeepSpeed
- **Recomendado**: 4x H100 80GB con fp8-torchao
- **Con cuantización agresiva (int4)**: 2x GPUs de 24GB pueden funcionar pero es experimental

El entrenamiento distribuido multi-GPU (FSDP2 o DeepSpeed) es esencialmente requerido para FLUX.2-dev debido al tamaño combinado del codificador de texto Mistral-3 y el transformer.

## Requisitos previos

### Versión de Python

FLUX.2 requiere Python 3.10 o posterior con transformers recientes:

```bash
python --version  # Debe ser 3.10+
pip install transformers>=4.45.0
```

### Acceso al modelo

Los modelos FLUX.2 requieren aprobación de acceso en Hugging Face:

**Para dev:**
1. Visita [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)
2. Acepta el acuerdo de licencia

**Para modelos klein:**
1. Visita [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) o [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
2. Acepta el acuerdo de licencia

Asegúrate de haber iniciado sesión en Hugging Face CLI: `huggingface-cli login`

## Instalación

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Para configuración de desarrollo:
```bash
git clone https://github.com/bghira/SimpleTuner
cd SimpleTuner
pip install -e ".[cuda]"
```

## Configuración

### Interfaz web

```bash
simpletuner server
```

Accede a http://localhost:8001 y selecciona FLUX.2 como familia de modelos.

### Configuración manual

Crea `config/config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "model_type": "lora",
  "model_family": "flux2",
  "model_flavour": "dev",
  "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-dev",
  "output_dir": "/path/to/output",
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "gradient_checkpointing": true,
  "mixed_precision": "bf16",
  "learning_rate": 1e-4,
  "lr_scheduler": "constant",
  "max_train_steps": 10000,
  "validation_resolution": "1024x1024",
  "validation_num_inference_steps": 20,
  "flux_guidance_mode": "constant",
  "flux_guidance_value": 1.0,
  "lora_rank": 16
}
```
</details>

### Opciones clave de configuración

#### Configuración de guidance

> **Nota**: Los modelos Klein (`klein-4b`, `klein-9b`) no tienen embeddings de guía. Las siguientes opciones de guidance solo aplican a `dev`.

FLUX.2-dev usa guidance embedding similar a FLUX.1:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "flux_guidance_mode": "constant",
  "flux_guidance_value": 1.0
}
```
</details>

O para guidance aleatorio durante el entrenamiento:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "flux_guidance_mode": "random-range",
  "flux_guidance_min": 1.0,
  "flux_guidance_max": 5.0
}
```
</details>

#### Cuantización (optimización de memoria)

Para reducir el uso de VRAM:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "int8-quanto",
  "base_model_default_dtype": "bf16"
}
```
</details>

#### TREAD (aceleración de entrenamiento)

FLUX.2 admite TREAD para entrenar más rápido:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.5, "start_layer_idx": 2, "end_layer_idx": -2}
    ]
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

</details>

### Configuración del dataset

Crea `config/multidatabackend.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "my-dataset",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux2/my-dataset",
    "instance_data_dir": "datasets/my-dataset",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux2",
    "write_batch_size": 64
  }
]
```
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

### Condicionamiento opcional de edición / referencia

FLUX.2 puede entrenar **texto-a-imagen simple** (sin condicionamiento) o con **imágenes de referencia/edición emparejadas**. Para añadir condicionamiento, empareja tu dataset principal con uno o más datasets de `conditioning` usando [`conditioning_data`](../DATALOADER.md#conditioning_data) y elige un [`conditioning_type`](../DATALOADER.md#conditioning_type):

<details>
<summary>Ver ejemplo de config</summary>

```jsonc
[
  {
    "id": "flux2-edits",
    "type": "local",
    "instance_data_dir": "/datasets/flux2/edits",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": ["flux2-references"],
    "cache_dir_vae": "cache/vae/flux2/edits"
  },
  {
    "id": "flux2-references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/flux2/references",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/flux2/references"
  }
]
```
</details>

- Usa `conditioning_type=reference_strict` cuando necesites recortes alineados 1:1 con la imagen de edición. `reference_loose` permite relaciones de aspecto no coincidentes.
- Los nombres de archivo deben coincidir entre los datasets de edición y referencia; cada imagen de edición debe tener un archivo de referencia correspondiente.
- Al proporcionar múltiples datasets de condicionamiento, configura `conditioning_multidataset_sampling` (`combined` vs `random`) según sea necesario; consulta [OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling).
- Sin `conditioning_data`, FLUX.2 vuelve al entrenamiento estándar de texto-a-imagen.

### Targets de LoRA

Presets disponibles de LoRA target:

- `all` (predeterminado): Todas las capas de atención y MLP
- `attention`: Solo capas de atención (qkv, proj)
- `mlp`: Solo capas MLP/feed-forward
- `tiny`: Entrenamiento mínimo (solo capas qkv)

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "--flux_lora_target": "all"
}
```
</details>

## Entrenamiento

### Iniciar sesión en servicios

```bash
huggingface-cli login
wandb login  # opcional
```

### Iniciar entrenamiento

```bash
simpletuner train
```

O vía script:

```bash
./train.sh
```

### Offloading de memoria

Para entornos con memoria limitada, FLUX.2 admite offloading en grupo para el transformer y opcionalmente el codificador de texto Mistral-3:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
--group_offload_text_encoder
```

El flag `--group_offload_text_encoder` se recomienda para FLUX.2 ya que el codificador de texto Mistral 24B se beneficia significativamente del offloading durante el caché de embeddings de texto. También puedes agregar `--group_offload_vae` para incluir el VAE en el offloading durante el caché de latentes.

## Prompts de validación

Crea `config/user_prompt_library.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "portrait_subject": "a professional portrait photograph of <subject>, studio lighting, high detail",
  "artistic_subject": "an artistic interpretation of <subject> in the style of renaissance painting",
  "cinematic_subject": "a cinematic shot of <subject>, dramatic lighting, film grain"
}
```
</details>

## Inferencia

### Usar LoRA entrenado

Los LoRAs de FLUX.2 pueden cargarse con el pipeline de inferencia de SimpleTuner o herramientas compatibles una vez que la comunidad desarrolle soporte.

### Guidance scale

- Entrenar con `flux_guidance_value=1.0` funciona bien para la mayoría de casos
- En inferencia, usa valores normales de guidance (3.0-5.0)

## Diferencias con FLUX.1

| Aspecto | FLUX.1 | FLUX.2-dev | FLUX.2-klein-9b | FLUX.2-klein-4b |
|--------|--------|------------|-----------------|-----------------|
| Codificador de texto | CLIP-L/14 + T5-XXL | Mistral-3 (24B) | Qwen3 (incluido) | Qwen3 (incluido) |
| Dimensión de embedding | CLIP: 768, T5: 4096 | 15,360 | 12,288 | 7,680 |
| Canales latentes | 16 | 32 (→128) | 32 (→128) | 32 (→128) |
| VAE | AutoencoderKL | Personalizado (BatchNorm) | Personalizado (BatchNorm) | Personalizado (BatchNorm) |
| Bloques del transformer | 19 joint + 38 single | 8 double + 48 single | 8 double + 24 single | 5 double + 20 single |
| Guidance embeds | Sí | Sí | No | No |

## Solución de problemas

### Falta de memoria durante el inicio

- Habilita `--offload_during_startup=true`
- Usa `--quantize_via=cpu` para cuantización del codificador de texto
- Reduce `--vae_batch_size`

### Text embedding lento

Mistral-3 es grande; considera:
- Pre-cachar todos los embeddings de texto antes del entrenamiento
- Usar cuantización del codificador de texto
- Procesamiento en lotes con `write_batch_size` más alto

### Inestabilidad de entrenamiento

- Baja la tasa de aprendizaje (prueba 5e-5)
- Aumenta pasos de acumulación de gradiente
- Habilita gradient checkpointing
- Usa `--max_grad_norm=1.0`

### CUDA Out of Memory

- Habilita cuantización (`int8-quanto` o `int4-quanto`)
- Habilita gradient checkpointing
- Reduce el tamaño del lote
- Habilita group offloading
- Usa TREAD para eficiencia de enrutamiento de tokens

## Avanzado: configuración de TREAD

TREAD (Token Routing for Efficient Architecture-agnostic Diffusion) acelera el entrenamiento procesando selectivamente tokens:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 4,
        "end_layer_idx": -4
      }
    ]
  }
}
```
</details>

- `selection_ratio`: Fracción de tokens a conservar (0.5 = 50%)
- `start_layer_idx`: Primera capa donde aplicar routing
- `end_layer_idx`: Última capa (negativo = desde el final)

Aceleración esperada: 20-40% dependiendo de la configuración.

## Ver también

- [FLUX.1 Guía rápida](FLUX.md) - Para entrenamiento de FLUX.1
- [Documentación de TREAD](../TREAD.md) - Configuración detallada de TREAD
- [Guía de entrenamiento LyCORIS](../LYCORIS.md) - Métodos de entrenamiento LoRA y LyCORIS
- [Configuración del dataloader](../DATALOADER.md) - Configuración del dataset
