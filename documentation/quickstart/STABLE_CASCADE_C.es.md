# Guía rápida de Stable Cascade Stage C

Esta guía explica cómo configurar SimpleTuner para ajustar el **prior de Stable Cascade Stage C**. Stage C aprende el prior texto‑a‑imagen que alimenta el stack de decodificadores Stage B/C, así que una buena higiene de entrenamiento aquí mejora directamente las salidas del decodificador downstream. Nos enfocaremos en entrenamiento LoRA, pero los mismos pasos aplican a ajustes finos completos si tienes VRAM de sobra.

> **Aviso:** Stage C usa el codificador de texto CLIP‑G/14 de más de 1B parámetros y un autoencoder basado en EfficientNet. Asegúrate de tener torchvision instalado y espera cachés de text‑embed grandes (aprox. 5–6× más grandes por prompt que SDXL).

## Requisitos de hardware

- **Entrenamiento LoRA:** 20–24 GB de VRAM (RTX 3090/4090, A6000, etc.)
- **Entrenamiento de modelo completo:** 48 GB+ de VRAM recomendado (A6000, A100, H100). Offload con DeepSpeed/FSDP2 puede bajar el requisito pero introduce complejidad.
- **RAM del sistema:** 32 GB recomendado para que el codificador de texto CLIP‑G y los hilos de caché no se queden sin recursos.
- **Disco:** Reserva al menos ~50 GB para archivos de caché de prompts. Los embeddings CLIP‑G de Stage C son ~4–6 MB cada uno.

## Requisitos previos

1. Python 3.12 (que coincide con el `.venv` del proyecto).
2. CUDA 12.1+ o ROCm 5.7+ para aceleración GPU (o Apple Metal para Macs M‑series, aunque Stage C se prueba principalmente en CUDA).
3. `torchvision` (requerido para el autoencoder de Stable Cascade) y `accelerate` para lanzar entrenamiento.

Verifica tu versión de Python:

```bash
python --version
```

Instala paquetes faltantes (ejemplo Ubuntu):

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
```

## Instalación

Sigue la instalación estándar de SimpleTuner (pip o fuente). Para una workstation CUDA típica:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install simpletuner[cuda]
```

Para contribuidores o quienes trabajen directamente en el repo, instala desde fuente y luego ejecuta `pip install -e .[cuda,dev]`.

## Configuración del entorno

### 1. Copia la config base

```bash
cp config/config.json.example config/config.json
```

Configura las siguientes claves (los valores mostrados son una buena base para Stage C):

| Clave | Recomendación | Notas |
| --- | -------------- | ----- |
| `model_family` | `"stable_cascade"` | Requerido para cargar componentes Stage C |
| `model_flavour` | `"stage-c"` (o `"stage-c-lite"`) | El flavour lite recorta parámetros si solo tienes ~18 GB de VRAM |
| `model_type` | `"lora"` | El ajuste completo funciona pero requiere mucha más memoria |
| `mixed_precision` | `"no"` | Stage C se rehúsa a correr en precisión mixta a menos que configures `i_know_what_i_am_doing=true`; fp32 es la opción segura |
| `gradient_checkpointing` | `true` | Ahorra 3–4 GB de VRAM |
| `vae_batch_size` | `1` | El autoencoder de Stage C es pesado; mantenlo bajo |
| `validation_resolution` | `"1024x1024"` | Coincide con las expectativas del decodificador downstream |
| `stable_cascade_use_decoder_for_validation` | `true` | Asegura que la validación use el pipeline combinado prior+decoder |
| `stable_cascade_decoder_model_name_or_path` | `"stabilityai/stable-cascade"` | Cambia a una ruta local si tienes un decodificador Stage B/C personalizado |
| `stable_cascade_validation_prior_num_inference_steps` | `20` | Pasos de denoising del prior |
| `stable_cascade_validation_prior_guidance_scale` | `3.0–4.0` | CFG en el prior |
| `stable_cascade_validation_decoder_guidance_scale` | `0.0–0.5` | CFG del decoder (0.0 es fotorrealista, >0.0 añade más adherencia al prompt) |

#### Ejemplo `config/config.json`

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "base_model_precision": "int8-torchao",
  "checkpoint_step_interval": 100,
  "data_backend_config": "config/stable_cascade/multidatabackend.json",
  "gradient_accumulation_steps": 2,
  "gradient_checkpointing": true,
  "hub_model_id": "stable-cascade-stage-c-lora",
  "learning_rate": 1e-4,
  "lora_alpha": 16,
  "lora_rank": 16,
  "lora_type": "standard",
  "lr_scheduler": "cosine",
  "max_train_steps": 30000,
  "mixed_precision": "no",
  "model_family": "stable_cascade",
  "model_flavour": "stage-c",
  "model_type": "lora",
  "optimizer": "adamw_bf16",
  "output_dir": "output/stable_cascade_stage_c",
  "report_to": "wandb",
  "seed": 42,
  "stable_cascade_decoder_model_name_or_path": "stabilityai/stable-cascade",
  "stable_cascade_decoder_subfolder": "decoder_lite",
  "stable_cascade_use_decoder_for_validation": true,
  "stable_cascade_validation_decoder_guidance_scale": 0.0,
  "stable_cascade_validation_prior_guidance_scale": 3.5,
  "stable_cascade_validation_prior_num_inference_steps": 20,
  "train_batch_size": 4,
  "use_ema": true,
  "vae_batch_size": 1,
  "validation_guidance": 4.0,
  "validation_negative_prompt": "ugly, blurry, low-res",
  "validation_num_inference_steps": 30,
  "validation_prompt": "a cinematic photo of a shiba inu astronaut",
  "validation_resolution": "1024x1024"
}
```
</details>

Puntos clave:

- `model_flavour` acepta `stage-c` y `stage-c-lite`. Usa lite si te falta VRAM o prefieres el prior destilado.
- Mantén `mixed_precision` en `"no"`. Si lo cambias, configura `i_know_what_i_am_doing=true` y prepárate para NaNs.
- Habilitar `stable_cascade_use_decoder_for_validation` conecta la salida del prior con el decodificador Stage B/C para que la galería de validación muestre imágenes reales en lugar de latentes del prior.

### 2. Configura el data backend

Crea `config/stable_cascade/multidatabackend.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "primary",
    "type": "local",
    "dataset_type": "images",
    "instance_data_dir": "/data/stable-cascade",
    "resolution": "1024x1024",
    "bucket_resolutions": ["1024x1024", "896x1152", "1152x896"],
    "crop": true,
    "crop_style": "random",
    "minimum_image_size": 768,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "caption_strategy": "filename",
    "prepend_instance_prompt": false,
    "repeats": 1
  },
  {
    "id": "stable-cascade-text-cache",
    "type": "local",
    "dataset_type": "text_embeds",
    "cache_dir": "/data/cache/stable-cascade/text",
    "default": true
  }
]
```
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

Consejos:

- Los latentes de Stage C se derivan de un autoencoder, así que quédate en 1024×1024 (o un rango ajustado de buckets vertical/horizontal). El decodificador espera grillas latentes de ~24×24 desde una entrada de 1024px.
- Mantén `target_downsample_size` en 1024 para que los recortes estrechos no exploten las relaciones de aspecto más allá de ~2:1.
- Siempre configura un caché dedicado de text‑embeds. Sin él, cada ejecución gastará 30–60 minutos re‑embedding captions con CLIP‑G.

### 3. Biblioteca de prompts (opcional)

Crea `config/stable_cascade/prompt_library.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "portrait": "a cinematic portrait photograph lit by studio strobes",
  "landscape": "a sweeping ultra wide landscape with volumetric lighting",
  "product": "a product render on a seamless background, dramatic reflections",
  "stylized": "digital illustration in the style of a retro sci-fi book cover"
}
```
</details>

Habilítala en tu config agregando `"validation_prompt_library": "config/stable_cascade/prompt_library.json"`.

## Entrenamiento

1. Activa tu entorno y lanza la configuración de Accelerate si aún no lo has hecho:

```bash
source .venv/bin/activate
accelerate config
```

2. Inicia el entrenamiento:

```bash
accelerate launch simpletuner/train.py \
  --config_file config/config.json \
  --data_backend_config config/stable_cascade/multidatabackend.json
```

Durante la primera época, monitorea:

- **Throughput del caché de texto** – Stage C registrará el progreso del caché. Espera ~8–12 prompts/seg en GPUs de gama alta.
- **Uso de VRAM** – Apunta a <95% de utilización para evitar OOM cuando corran las validaciones.
- **Salidas de validación** – El pipeline combinado debería emitir PNGs a resolución completa en `output/<run>/validation/`.

## Notas de validación e inferencia

- El prior de Stage C por sí solo solo produce embeddings de imagen. El wrapper de validación de SimpleTuner los pasa automáticamente por el decodificador cuando `stable_cascade_use_decoder_for_validation=true`.
- Para cambiar el flavour del decodificador, configura `stable_cascade_decoder_subfolder` en `"decoder"`, `"decoder_lite"`, o una carpeta personalizada que contenga los pesos de Stage B o Stage C.
- Para previews más rápidos, baja `stable_cascade_validation_prior_num_inference_steps` a ~12 y `validation_num_inference_steps` a 20. Una vez satisfecho, súbelos para mayor calidad.

## Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.
*   **[Diff2Flow](../experimental/DIFF2FLOW.md):** permite entrenar Stable Cascade con un objetivo Flow Matching.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

</details>

## Solución de problemas

| Síntoma | Arreglo |
| --- | --- |
| "Stable Cascade Stage C requires --mixed_precision=no" | Configura `"mixed_precision": "no"` o agrega `"i_know_what_i_am_doing": true` (no recomendado) |
| La validación solo muestra priors (ruido verde) | Asegúrate de que `stable_cascade_use_decoder_for_validation` sea `true` y que los pesos del decodificador estén descargados |
| El caché de text embeds tarda horas | Usa SSD/NVMe para el directorio de caché y evita montajes de red. Considera podar prompts o pre‑computar con el CLI `simpletuner-text-cache` |
| Error al importar autoencoder | Instala torchvision dentro de tu `.venv` (`pip install torchvision --extra-index-url https://download.pytorch.org/whl/cu124`). Stage C necesita pesos de EfficientNet |

## Próximos pasos

- Experimenta con `lora_rank` (8–32) y `learning_rate` (5e-5 a 2e-4) según la complejidad del sujeto.
- Conecta ControlNet/adaptadores de condicionamiento a Stage B después de entrenar el prior.
- Si necesitas iterar más rápido, entrena el flavour `stage-c-lite` y mantén los pesos `decoder_lite` para validación.

¡Buen ajuste!
