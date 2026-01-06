# Guía rápida de LongCat‑Video

LongCat‑Video es un modelo bilingüe (zh/en) de 13.6B para texto‑a‑video e imagen‑a‑video que usa flow matching, el codificador de texto Qwen‑2.5‑VL y el VAE de Wan. Esta guía te lleva por la configuración, preparación de datos y una primera ejecución de entrenamiento/validación con SimpleTuner.

---

## 1) Requisitos de hardware (qué esperar)

- Transformer 13.6B + VAE de Wan: espera más VRAM que en modelos de imagen; empieza con `train_batch_size=1`, gradient checkpointing y rangos LoRA bajos.
- RAM del sistema: más de 32 GB es útil para clips multi‑frame; mantén los datasets en almacenamiento rápido.
- Apple MPS: compatible para previews; los encodings posicionales se bajan a float32 automáticamente.

---

## 2) Requisitos previos

1. Verifica Python 3.12 (SimpleTuner incluye `.venv` por defecto):
   ```bash
   python --version
   ```
2. Instala SimpleTuner con el backend que coincide con tu hardware:
   ```bash
   pip install "simpletuner[cuda]"   # NVIDIA
   pip install "simpletuner[mps]"    # Apple Silicon
   pip install "simpletuner[cpu]"    # Solo CPU
   ```
3. La cuantización está integrada (`int8-quanto`, `int4-quanto`, `fp8-torchao`) y no requiere instalaciones manuales extra en setups normales.

---

## 3) Configuración del entorno

### Web UI
```bash
simpletuner server
```
Abre http://localhost:8001 y elige la familia de modelo `longcat_video`.

### Base CLI (config/config.json)

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_video",
  "model_flavour": "final",
  "pretrained_model_name_or_path": null,      // se selecciona automáticamente según flavour
  "base_model_precision": "bf16",             // int8-quanto/fp8-torchao también funcionan para LoRA
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "learning_rate": 1e-4,
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0
}
```

**Valores predeterminados clave a mantener**
- El scheduler de flow‑matching con shift `12.0` es automático; no se necesitan flags de ruido personalizados.
- Los buckets de aspecto permanecen alineados a 64 px; `aspect_bucket_alignment` se fuerza a 64.
- Longitud máxima de tokens 512 (Qwen‑2.5‑VL); el pipeline agrega negativos vacíos automáticamente cuando CFG está activo y no se proporciona prompt negativo.
- Los frames deben cumplir `(num_frames - 1)` divisible por el stride temporal del VAE (por defecto 4). Los 93 frames por defecto ya cumplen esto.

Ahorros de VRAM opcionales:
- Reduce `lora_rank` (4–8) y usa precisión base `int8-quanto`.
- Habilita group offload: `--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`.
- Baja primero `validation_resolution`, frames o pasos si los previews hacen OOM.
- Valores de atención por defecto: en CUDA, LongCat‑Video usa automáticamente el kernel Triton block‑sparse incluido cuando está disponible y hace fallback al dispatcher estándar en caso contrario. No hace falta toggle. Si quieres xFormers específicamente, configura `attention_implementation: "xformers"` en tu config/CLI.

### Iniciar entrenamiento (CLI)
```bash
simpletuner train --config config/config.json
```
O lanza la Web UI y envía un job con la misma config.

---

## 4) Guía del dataloader

- Usa datasets de video con captions; cada muestra debe proporcionar frames (o un clip corto) más un caption de texto. `dataset_type: video` se maneja automáticamente mediante `VideoToTensor`.
- Mantén las dimensiones de frames en la cuadrícula de 64 px (p. ej., 480x832, buckets 720p). Alto/ancho deben ser divisibles por el stride del VAE de Wan (16px con la configuración integrada) y por 64 para el bucketing.
- Para ejecuciones imagen‑a‑video, incluye una imagen de condicionamiento por muestra; se coloca en el primer frame latente y se mantiene fija durante el muestreo.
- LongCat‑Video está diseñado para 30 fps. Los 93 frames por defecto son ~3.1 s; si cambias recuentos de frames, mantén `(frames - 1) % 4 == 0` y recuerda que la duración escala con fps.

### Estrategia de buckets de video

En la sección `video` de tu dataset, puedes configurar cómo se agrupan los videos:
- `bucket_strategy`: `aspect_ratio` (predeterminado) agrupa por relación de aspecto espacial. `resolution_frames` agrupa por formato `WxH@F` (p. ej., `480x832@93`) para datasets de resolución/duración mixta.
- `frame_interval`: Al usar `resolution_frames`, redondea recuentos de frames a este intervalo (p. ej., configúralo en 4 para coincidir con el stride temporal del VAE).

---

## 5) Validación e inferencia

- Guidance: 3.5–5.0 funciona bien; los prompts negativos vacíos se generan automáticamente cuando CFG está habilitado.
- Pasos: 35–45 para comprobar calidad; menos para previews rápidos.
- Frames: 93 por defecto (se alinea con el stride temporal 4 del VAE).
- ¿Necesitas más margen para previews o entrenamiento? Configura `musubi_blocks_to_swap` (prueba 4–8) y opcionalmente `musubi_block_swap_device` para hacer streaming de los últimos bloques del transformer desde CPU mientras ejecutas forward/backward. Espera sobrecarga de transferencia pero picos de VRAM más bajos.

- Las validaciones se ejecutan desde los campos `validation_*` en tu config o vía la pestaña de preview de la WebUI después de iniciar `simpletuner server`. Usa esas rutas para comprobaciones rápidas en lugar de un subcomando CLI independiente.
- Para validación basada en datasets (incluyendo I2V), configura `validation_using_datasets: true` y apunta `eval_dataset_id` a tu split de validación. Si ese split está marcado `is_i2v` y tiene frames de condicionamiento vinculados, el pipeline mantiene fijo el primer frame automáticamente.
- Los previews latentes se desempacan antes de decodificar para evitar desajustes de canales.

---

## 6) Solución de problemas

- **Errores de alto/ancho**: asegúrate de que ambos sean divisibles por 16 y se mantengan en la cuadrícula de 64 px.
- **Advertencias float64 en MPS**: se manejan internamente; mantén precisión en bf16/float32.
- **OOM**: baja primero la resolución o los frames de validación, reduce el rango LoRA, habilita group offload o cambia a `int8-quanto`/`fp8-torchao`.
- **Negativos en blanco con CFG**: si omites el prompt negativo, el pipeline inserta uno vacío automáticamente.

---

## 7) Flavours

- `final`: versión principal de LongCat‑Video (texto‑a‑video + imagen‑a‑video en un solo checkpoint).
