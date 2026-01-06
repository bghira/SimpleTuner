# Guía rápida de LongCat‑Video Edit (Image‑to‑Video)

Esta guía te lleva por el entrenamiento y la validación del flujo imagen‑a‑video para LongCat‑Video. No necesitas cambiar flavours; el mismo checkpoint `final` cubre texto‑a‑video e imagen‑a‑video. La diferencia viene de tus datasets y ajustes de validación.

---

## 1) Diferencias del modelo vs LongCat‑Video base

|                               | Base (text2video) | Edit / I2V |
| ----------------------------- | ----------------- | ---------- |
| Flavour                       | `final`           | `final` (mismos pesos) |
| Condicionamiento              | ninguno           | **requiere frame de condicionamiento** (el primer latente se mantiene fijo) |
| Codificador de texto          | Qwen‑2.5‑VL       | Qwen‑2.5‑VL (igual) |
| Pipeline                      | TEXT2IMG          | IMG2VIDEO |
| Entradas de validación        | solo prompt       | prompt **y** imagen de condicionamiento |
| Buckets / stride              | buckets de 64px, `(frames-1)%4==0` | igual |

**Valores predeterminados base que heredas**
- Flow matching con shift `12.0`.
- Buckets de aspecto forzados a 64px.
- Codificador de texto Qwen‑2.5‑VL; negativos vacíos auto‑agregados cuando CFG está activo.
- Frames por defecto: 93 (cumple `(frames-1)%4==0`).

---

## 2) Cambios de config (CLI/WebUI)

```jsonc
{
  "model_family": "longcat_video",
  "model_flavour": "final",
  "model_type": "lora",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "learning_rate": 1e-4,
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0,
  "validation_using_datasets": true,
  "eval_dataset_id": "longcat-video-val"
}
```

Mantén `aspect_bucket_alignment` en 64. El primer frame latente conserva la imagen inicial; déjalo intacto. Quédate con 93 frames (ya cumple la regla de stride del VAE `(frames - 1) % 4 == 0`) a menos que tengas un motivo fuerte para cambiarlo.

Configuración rápida:
```bash
cp config/config.json.example config/config.json
```
Completa `model_family`, `model_flavour`, `output_dir`, `data_backend_config` y `eval_dataset_id`. Deja los valores predeterminados anteriores a menos que sepas que necesitas otros.

Opciones de atención en CUDA:
- En CUDA, LongCat‑Video prefiere automáticamente el kernel Triton block‑sparse incluido cuando está presente y hace fallback al dispatcher estándar de lo contrario. No se requiere toggle manual.
- Para forzar xFormers, configura `attention_implementation: "xformers"` en tu config/CLI.

---

## 3) Dataloader: emparejar clips con frames iniciales

- Crea dos datasets:
  - **Clips**: los videos objetivo + captions (instrucciones de edición). Márcalos `is_i2v: true` y configura `conditioning_data` al ID del dataset de frames iniciales.
  - **Frames iniciales**: una imagen por clip, mismos nombres de archivo, sin captions.
- Mantén ambos en la cuadrícula de 64px (p. ej., 480x832). Alto/ancho deben ser divisibles por 16. Los recuentos de frames deben cumplir `(frames - 1) % 4 == 0`; 93 ya es válido.
- Usa cachés VAE separados para clips vs frames iniciales.

Ejemplo de `multidatabackend.json`:
```jsonc
[
  {
    "id": "longcat-video-train",
    "type": "local",
    "dataset_type": "video",
    "is_i2v": true,
    "instance_data_dir": "/data/video-clips",
    "caption_strategy": "textfile",
    "resolution": 480,
    "cache_dir_vae": "/cache/vae/longcat/video",
    "conditioning_data": ["longcat-video-cond"]
  },
  {
    "id": "longcat-video-cond",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/data/video-start-frames",
    "caption_strategy": null,
    "resolution": 480,
    "cache_dir_vae": "/cache/vae/longcat/video-cond"
  }
]
```

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

---

## 4) Específicos de validación

- Agrega un split de validación pequeño con la misma estructura emparejada que el entrenamiento. Configura `validation_using_datasets: true` y apunta `eval_dataset_id` a ese split (p. ej., `longcat-video-val`) para que la validación tome el frame inicial automáticamente.
- Previews en WebUI: inicia `simpletuner server`, elige edición de LongCat‑Video y sube el frame inicial + prompt.
- Guidance: 3.5–5.0 funciona; los negativos vacíos se rellenan automáticamente cuando CFG está activo.
- Para previews o entrenamiento con poca VRAM, configura `musubi_blocks_to_swap` (empieza con 4–8) y opcionalmente `musubi_block_swap_device` para hacer streaming de los últimos bloques del transformer desde CPU durante forward/backward. Cambia algo de throughput por menor pico de VRAM.
- El frame de condicionamiento se mantiene fijo durante el muestreo; solo los frames posteriores se de-noise.

---

## 5) Inicio de entrenamiento (CLI)

Después de configurar config y dataloader:
```bash
simpletuner train --config config/config.json
```
Asegúrate de que los frames de condicionamiento estén presentes en los datos de entrenamiento para que el pipeline pueda construir latentes de condicionamiento.

---

## 6) Solución de problemas

- **Imagen de condicionamiento faltante**: proporciona un dataset de condicionamiento vía `conditioning_data` con nombres de archivo coincidentes; configura `eval_dataset_id` al ID de tu split de validación.
- **Errores de alto/ancho**: mantén dimensiones divisibles por 16 y en la cuadrícula de 64px.
- **Deriva del primer frame**: baja guidance (3.5–4.0) o reduce pasos.
- **OOM**: baja resolución/frames de validación, reduce `lora_rank`, habilita group offload o usa `int8-quanto`/`fp8-torchao`.
