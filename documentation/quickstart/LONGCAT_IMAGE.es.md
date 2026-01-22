# Guía rápida de LongCat‑Image

LongCat‑Image es un modelo texto‑a‑imagen bilingüe (zh/en) de 6B que usa flow matching y el codificador de texto Qwen‑2.5‑VL. Esta guía te lleva por la configuración, la preparación de datos y la primera ejecución de entrenamiento/validación con SimpleTuner.

---

## 1) Requisitos de hardware (qué esperar)

- VRAM: 16–24 GB cubren LoRA a 1024px con `int8-quanto` o `fp8-torchao`. Las ejecuciones completas en bf16 pueden necesitar ~24 GB.
- RAM del sistema: ~32 GB suele ser suficiente.
- Apple MPS: compatible para inferencia/preview; ya hacemos downcast de pos‑ids a float32 en MPS para evitar problemas de dtype.

---

## 2) Requisitos previos (paso a paso)

1. Python 3.10–3.13 verificado:
   ```bash
   python --version
   ```
2. (Linux/CUDA) En imágenes nuevas, instala el toolchain habitual:
   ```bash
   apt -y update
   apt -y install build-essential nvidia-cuda-toolkit
   ```
3. Instala SimpleTuner con los extras correctos para tu backend:
   ```bash
   pip install "simpletuner[cuda]"   # CUDA
   pip install "simpletuner[cuda13]" # CUDA 13 / Blackwell (NVIDIA B-series GPUs)
   pip install "simpletuner[mps]"    # Apple Silicon
   pip install "simpletuner[cpu]"    # Solo CPU
   ```
4. La cuantización está integrada (`int8-quanto`, `int4-quanto`, `fp8-torchao`) y no requiere instalaciones manuales extra en setups normales.

---

## 3) Configuración del entorno

### Web UI (más guiado)
```bash
simpletuner server
```
Visita http://localhost:8001 y elige la familia de modelo `longcat_image`.

### Base CLI (config/config.json)

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_image",
  "model_flavour": "final",                // opciones: final, dev
  "pretrained_model_name_or_path": null,   // se selecciona automáticamente según flavour; reemplaza con una ruta local si es necesario
  "base_model_precision": "int8-quanto",   // buen valor por defecto; fp8-torchao también funciona
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 16,
  "learning_rate": 1e-4,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 30
}
```

**Valores predeterminados clave a mantener**
- El scheduler de flow matching es automático; no se necesitan flags de schedule especiales.
- Los buckets de aspecto permanecen alineados a 64 px; no reduzcas `aspect_bucket_alignment`.
- Longitud máxima de tokens 512 (Qwen‑2.5‑VL).

Ahorros de memoria opcionales (elige lo que se ajuste a tu hardware):
- `--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`
- Baja `lora_rank` (4–8) y/o usa precisión base `int8-quanto`.
- Si la validación hace OOM, reduce primero `validation_resolution` o los pasos.

### Creación rápida de config (una vez)
```bash
cp config/config.json.example config/config.json
```
Edita los campos anteriores (model_family, flavour, precision, rutas). Apunta `output_dir` y rutas del dataset a tu almacenamiento.

### Iniciar entrenamiento (CLI)
```bash
simpletuner train --config config/config.json
```
O inicia la WebUI y lanza una ejecución desde la página Jobs después de seleccionar la misma config.

---

## 4) Pistas del dataloader (qué suministrar)

- Carpetas de imágenes con captions estándar (textfile/JSON/CSV) funcionan. Incluye zh/en si quieres mantener la fuerza bilingüe.
- Mantén los bordes de buckets en la cuadrícula de 64 px. Si entrenas multi‑aspecto, lista varias resoluciones (p. ej., `1024x1024,1344x768`).
- El VAE es KL con shift+scale; los cachés usan el factor de escala integrado automáticamente.

---

## 5) Validación e inferencia

- Guidance: 4–6 es un buen inicio; deja el prompt negativo vacío.
- Pasos: ~30 para comprobaciones rápidas; 40–50 para mejor calidad.
- El preview de validación funciona de fábrica; los latentes se desempacan antes de decodificar para evitar desajustes de canales.

Ejemplo (validación CLI):
```bash
simpletuner validate \
  --model_family longcat_image \
  --model_flavour final \
  --validation_resolution 1024x1024 \
  --validation_num_inference_steps 30 \
  --validation_guidance 4.5
```

---

## 6) Solución de problemas

- **Errores float64 en MPS**: manejados internamente; mantén tu config en float32/bf16.
- **Desajuste de canales en previews**: solucionado al desempacar latentes antes de decodificar (incluido en el código de esta guía).
- **OOM**: baja `validation_resolution`, reduce `lora_rank`, habilita group offload o cambia a `int8-quanto` / `fp8-torchao`.
- **Tokenización lenta**: Qwen‑2.5‑VL se limita a 512 tokens; evita prompts muy largos.

---

## 7) Selección de flavour
- `final`: versión principal (mejor calidad).
- `dev`: checkpoint de entrenamiento intermedio para experimentos/ajustes finos.
