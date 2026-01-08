# Guía rápida de LongCat‑Image Edit

Esta es la variante de edición/img2img de LongCat‑Image. Lee primero [LONGCAT_IMAGE.md](../quickstart/LONGCAT_IMAGE.md); este archivo solo enumera lo que cambia para el flavour de edición.

---

## 1) Diferencias del modelo vs LongCat‑Image base

|                               | Base (text2img) | Edit |
| ----------------------------- | --------------- | ---- |
| Flavour                       | `final` / `dev` | `edit` |
| Condicionamiento              | ninguno         | **requiere latentes de condicionamiento (imagen de referencia)** |
| Codificador de texto          | Qwen‑2.5‑VL     | Qwen‑2.5‑VL **con contexto de visión** (la codificación del prompt necesita imagen de referencia) |
| Pipeline                      | TEXT2IMG        | IMG2IMG/EDIT |
| Entradas de validación        | solo prompt     | prompt **y** referencia |

---

## 2) Cambios de config (CLI/WebUI)

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_image",
  "model_flavour": "edit",
  "base_model_precision": "int8-quanto",      // fp8-torchao también sirve; ayuda a encajar 16–24 GB
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "learning_rate": 5e-5,
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 40,
  "validation_resolution": "768x768"
}
```

Mantén `aspect_bucket_alignment` en 64. No desactives los latentes de condicionamiento; el pipeline de edición los espera.

Creación rápida de config:
```bash
cp config/config.json.example config/config.json
```
Luego configura `model_family`, `model_flavour`, rutas del dataset y output_dir.

---

## 3) Dataloader: edición + referencia emparejadas

Usa dos datasets alineados: **imágenes de edición** (caption = instrucción de edición) e **imágenes de referencia**. El `conditioning_data` del dataset de edición debe apuntar al ID del dataset de referencia. Los nombres de archivo deben coincidir 1‑a‑1.

```jsonc
[
  {
    "id": "edit-images",
    "type": "local",
    "instance_data_dir": "/data/edits",
    "caption_strategy": "textfile",
    "resolution": 768,
    "cache_dir_vae": "/cache/vae/longcat/edit",
    "conditioning_data": ["ref-images"]
  },
  {
    "id": "ref-images",
    "type": "local",
    "instance_data_dir": "/data/refs",
    "caption_strategy": null,
    "resolution": 768,
    "cache_dir_vae": "/cache/vae/longcat/ref"
  }
]
```

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

Notas:
- Buckets de aspecto: mantenlos en la cuadrícula de 64 px.
- Los captions de referencia son opcionales; si están presentes reemplazan los captions de edición (normalmente no deseado).
- Los cachés del VAE para edición y referencia deben ser rutas separadas.
- Si ves cache misses o errores de forma, limpia los cachés del VAE de ambos datasets y regenera.

---

## 4) Específicos de validación

- La validación necesita imágenes de referencia para producir latentes de condicionamiento. Apunta la división de validación de `edit-images` a `ref-images` vía `conditioning_data`.
- Guidance: 4–6 funciona bien; deja el prompt negativo vacío.
- Los callbacks de preview están soportados; los latentes se desempacan automáticamente para los decodificadores.
- Si la validación falla por latentes de condicionamiento faltantes, revisa que el dataloader de validación incluya entradas de edición y referencia con nombres de archivo coincidentes.

---

## 5) Comandos de inferencia / validación

Validación rápida por CLI:
```bash
simpletuner validate \
  --model_family longcat_image \
  --model_flavour edit \
  --validation_resolution 768x768 \
  --validation_guidance 4.5 \
  --validation_num_inference_steps 40
```

WebUI: elige el pipeline **Edit**, proporciona la imagen fuente y la instrucción de edición.

---

## 6) Inicio de entrenamiento (CLI)

Después de configurar config y dataloader:
```bash
simpletuner train --config config/config.json
```
Asegúrate de que el dataset de referencia esté presente durante el entrenamiento para que los latentes de condicionamiento se puedan calcular o cargar desde caché.

---

## 7) Solución de problemas

- **Latentes de condicionamiento faltantes**: asegúrate de que el dataset de referencia esté conectado vía `conditioning_data` y que los nombres de archivo coincidan.
- **Errores de dtype en MPS**: el pipeline baja automáticamente pos‑ids a float32 en MPS; mantén el resto en float32/bf16.
- **Desajuste de canales en previews**: los previews des‑patchify los latentes antes de decodificar (mantén esta versión de SimpleTuner).
- **OOM durante edición**: baja la resolución/pasos de validación, reduce `lora_rank`, habilita group offload y prefiere `int8-quanto`/`fp8-torchao`.
