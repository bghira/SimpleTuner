# Guía rápida de Qwen Image Edit

Esta guía cubre los flavours de **edición** de Qwen Image que SimpleTuner soporta:

- `edit-v1` – una sola imagen de referencia por ejemplo de entrenamiento. La imagen de referencia se codifica con el codificador de texto Qwen2.5-VL y se cachea como **conditioning image embeds**.
- `edit-v2` (“edit plus”) – hasta tres imágenes de referencia por muestra, codificadas en latentes VAE sobre la marcha.

Ambas variantes heredan la mayor parte de la guía base de [Qwen Image](./QWEN_IMAGE.md); las secciones de abajo se enfocan en lo que es *diferente* al ajustar los checkpoints de edición.

---

## 1. Lista de hardware

El modelo base sigue siendo de **20 B parámetros**:

| Requisito | Recomendación |
|-------------|----------------|
| VRAM GPU    | mínimo 24 G (con cuantización int8/nf4) • 40 G+ muy recomendado |
| Precisión   | `mixed_precision=bf16`, `base_model_precision=int8-quanto` (o `nf4-bnb`) |
| Tamaño de lote  | Debe permanecer en `train_batch_size=1`; usa acumulación de gradiente para el lote efectivo |

Se aplican todos los demás requisitos de entrenamiento de la [guía de Qwen Image](./QWEN_IMAGE.md) (Python ≥ 3.10, imagen CUDA 12.x, etc.).

---

## 2. Puntos destacados de configuración

Dentro de `config/config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```jsonc
{
  "model_type": "lora",
  "model_family": "qwen_image",
  "model_flavour": "edit-v1",      // o "edit-v2"
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.0,
  "validation_num_inference_steps": 30,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "base_model_precision": "int8-quanto",
  "quantize_via": "cpu",
  "quantize_activations": false,
  "flow_schedule_shift": 1.73,
  "data_backend_config": "config/qwen_edit/multidatabackend.json"
}
```
</details>

- EMA corre en CPU por defecto y es seguro dejarlo habilitado a menos que necesites checkpoints más rápidos.
- `validation_resolution` debe reducirse (p. ej., `768x768`) en tarjetas de 24 G.
- `match_target_res` puede agregarse bajo `model_kwargs` para `edit-v2` si quieres que las imágenes de control hereden la resolución objetivo en lugar del empaquetado por defecto de 1 MP:

<details>
<summary>Ver ejemplo de config</summary>

```jsonc
"model_kwargs": {
  "match_target_res": true
}
```
</details>

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

---

</details>

## 3. Diseño del dataloader

Ambos flavours esperan **datasets emparejados**: una imagen de edición, caption de edición opcional y una o más imágenes de control/referencia que compartan **exactamente los mismos nombres de archivo**.

Para detalles de campos, consulta [`conditioning_type`](../DATALOADER.md#conditioning_type) y [`conditioning_data`](../DATALOADER.md#conditioning_data). Si proporcionas múltiples datasets de condicionamiento, elige cómo se muestrean con `conditioning_multidataset_sampling` en [OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling).

### 3.1 edit‑v1 (una sola imagen de control)

El dataset principal debe referenciar un dataset de condicionamiento **y** un caché de conditioning-image-embeds:

<details>
<summary>Ver ejemplo de config</summary>

```jsonc
[
  {
    "id": "my-edited-images",
    "type": "local",
    "instance_data_dir": "/datasets/edited-images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": ["my-reference-images"],
    "conditioning_image_embeds": "my-reference-embeds",
    "cache_dir_vae": "cache/vae/edited-images"
  },
  {
    "id": "my-reference-images",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/reference-images",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/reference-images"
  },
  {
    "id": "my-reference-embeds",
    "type": "local",
    "dataset_type": "conditioning_image_embeds",
    "cache_dir": "cache/conditioning_image_embeds/reference"
  }
]
```
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

- `conditioning_type=reference_strict` garantiza que los recortes coincidan con la imagen de edición. Usa `reference_loose` solo si la referencia puede tener relaciones de aspecto distintas.
- La entrada `conditioning_image_embeds` almacena los tokens visuales de Qwen2.5-VL producidos para cada referencia. Si se omite, SimpleTuner creará un caché por defecto en `cache/conditioning_image_embeds/<dataset_id>`.

### 3.2 edit‑v2 (multi‑control)

Para `edit-v2`, lista cada dataset de control en `conditioning_data`. Cada entrada proporciona un frame de control adicional. **No** necesitas un caché de conditioning-image-embeds porque los latentes se calculan al vuelo.

<details>
<summary>Ver ejemplo de config</summary>

```jsonc
[
  {
    "id": "my-edited-images",
    "type": "local",
    "instance_data_dir": "/datasets/edited-images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": [
      "my-reference-images-a",
      "my-reference-images-b",
      "my-reference-images-c"
    ],
    "cache_dir_vae": "cache/vae/edited-images"
  },
  {
    "id": "my-reference-images-a",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/reference-images-a",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/reference-images-a"
  },
  {
    "id": "my-reference-images-b",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/reference-images-b",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/reference-images-b"
  },
  {
    "id": "my-reference-images-c",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/reference-images-c",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/reference-images-c"
  }
]
```
</details>

Usa tantos datasets de control como imágenes de referencia tengas (1–3). SimpleTuner los mantiene alineados por muestra haciendo coincidir los nombres de archivo.

---

## 4. Ejecutar el entrenamiento

La prueba rápida más simple es ejecutar uno de los presets de ejemplo:

```bash
simpletuner train example=qwen_image.edit-v1-lora
# o
simpletuner train example=qwen_image.edit-v2-lora
```

Para lanzar manualmente:

```bash
simpletuner train \
  --config config/config.json \
  --data config/qwen_edit/multidatabackend.json
```

### Consejos

- Mantén `caption_dropout_probability` en `0.0` a menos que tengas un motivo para entrenar sin la instrucción de edición.
- Para trabajos largos, reduce la cadencia de validación (`validation_step_interval`) para que las validaciones de edición (costosas) no dominen el tiempo de ejecución.
- Los checkpoints de edición de Qwen no traen un guidance head; `validation_guidance` suele estar en el rango **3.5–4.5**.

---

## 5. Vistas previas de validación

Si quieres previsualizar la imagen de referencia junto con la salida de validación, guarda tus pares de edición/referencia de validación en un dataset dedicado (misma estructura que el split de entrenamiento) y configura:

<details>
<summary>Ver ejemplo de config</summary>

```jsonc
{
  "eval_dataset_id": "qwen-edit-val"
}
```
</details>

SimpleTuner reutilizará las imágenes de condicionamiento de ese dataset durante la validación.

---

### Solución de problemas

- **`ValueError: Control tensor list length does not match batch size`** – asegúrate de que cada dataset de condicionamiento contenga archivos para *todas* las imágenes de edición. Carpetas vacías o nombres de archivo que no coinciden provocan este error.
- **Out of memory durante validación** – baja `validation_resolution`, `validation_num_inference_steps`, o cuantiza más (`base_model_precision=int2-quanto`) antes de reintentar.
- **Errores de caché no encontrado** al usar `edit-v1` – verifica que el campo `conditioning_image_embeds` del dataset principal coincida con una entrada de dataset de caché existente.

---

Ya estás listo para adaptar la guía base de Qwen Image al entrenamiento de edición. Para opciones de configuración completas (caché del codificador de texto, muestreo multi-backend, etc.), reutiliza la guía de [FLUX_KONTEXT.md](./FLUX_KONTEXT.md) – el flujo de emparejamiento de datasets es el mismo, solo cambia la familia del modelo a `qwen_image`.
