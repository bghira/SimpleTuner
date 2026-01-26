# Guía de inicio rápido

**Nota**: Para configuraciones más avanzadas, consulta el [tutorial](TUTORIAL.md) y la [referencia de opciones](OPTIONS.md).

## Compatibilidad de funciones

Para la matriz de funciones completa y más precisa, consulta el [README principal](https://github.com/bghira/SimpleTuner#model-architecture-support).

## Guías de inicio rápido por modelo

| Modelo | Parámetros | PEFT LoRA | Lycoris | Rango completo | Cuantización | Precisión mixta | Checkpointing de gradiente | Flow Shift | TwinFlow | LayerSync | ControlNet | Sliders† | Guía |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | ✓ | int8 opcional | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | [SIGMA.md](quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | ✓ | int8 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [SANA.md](quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | ✓ | no recomendado | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | [KOLORS.md](quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | ✓ | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | [SD3.md](quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ | [FLUX.md](quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [FLUX2.md](quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ | [FLUX_KONTEXT.md](quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✗ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [ZIMAGE.md](quickstart/ZIMAGE.md) |
| ACE-Step | 3.5B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [ACE_STEP.md](quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [CHROMA.md](quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | [AURAFLOW.md](quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [HIDREAM.md](quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | ✓ | int8/fp8 opcional | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | [OMNIGEN.md](quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | ✓ | no recomendado | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | [SDXL.md](quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | ✓ | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [LUMINA2.md](quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | ✓ | no recomendado | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [COSMOS2IMAGE.md](quickstart/COSMOS2IMAGE.md) |
| LTX Video | ~2.5B | ✓ | ✓ | ✓ | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LTXVIDEO.md](quickstart/LTXVIDEO.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [HUNYUANVIDEO.md](quickstart/HUNYUANVIDEO.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [WAN.md](quickstart/WAN.md) |
| Qwen Image | 20B | ✓ | ✓ | ✓* | **requerido** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [QWEN_IMAGE.md](quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓ | ✓* | **requerido** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [QWEN_EDIT.md](quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, prior 3.6B | ✓ | ✓ | ✓* | no soportado | fp32 (requerido) | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | [STABLE_CASCADE_C.md](quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [KANDINSKY5_IMAGE.md](quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [KANDINSKY5_VIDEO.md](quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_VIDEO.md](quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_VIDEO_EDIT.md](quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_IMAGE.md](quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_EDIT.md](quickstart/LONGCAT_EDIT.md) |

*✓ = soportado, ✓* = requiere DeepSpeed/FSDP2 para rango completo, ✗ = no soportado, `✓+` indica que se recomienda checkpointing por presión de VRAM. TwinFlow ✓ significa soporte nativo cuando `twinflow_enabled=true` (los modelos de difusión necesitan `diff2flow_enabled+twinflow_allow_diff2flow`). LayerSync ✓ significa que el backbone expone estados ocultos del transformer para autoalineación; ✗ marca backbones tipo UNet sin ese buffer. †Sliders aplican a LoRA y LyCORIS (incluido LyCORIS de rango completo “full”).*

> ℹ️ El inicio rápido de Wan incluye presets de las etapas 2.1 y 2.2 y el toggle de time-embedding. Flux Kontext cubre flujos de edición construidos sobre Flux.1.

> ⚠️ Estos quickstarts son documentos vivos. Espera actualizaciones ocasionales a medida que llegan nuevos modelos o se mejoran recetas de entrenamiento.

### Rutas rápidas: Z-Image Turbo y Flux Schnell

- **Z-Image Turbo**: LoRA totalmente soportado con TREAD; funciona rápido en NVIDIA y macOS incluso sin quant (int8 también sirve). A menudo el cuello de botella es solo la configuración del trainer.
- **Flux Schnell**: La configuración de quickstart maneja automáticamente el fast noise schedule y la pila de assistant LoRA; no se requieren flags extra para entrenar LoRAs Schnell.

### Funciones experimentales avanzadas

- **Diff2Flow**: Permite entrenar modelos estándar epsilon/v-prediction (SD1.5, SDXL, DeepFloyd, etc.) usando un objetivo de pérdida de Flow Matching. Esto reduce la brecha entre arquitecturas antiguas y el entrenamiento moderno basado en flujo.
- **Scheduled Sampling**: Reduce el sesgo de exposición permitiendo que el modelo genere sus propios latentes ruidosos intermedios durante el entrenamiento ("rollout"). Esto ayuda a que el modelo aprenda a recuperarse de sus propios errores de generación.

## Problemas Comunes

### El dataset tiene menos muestras de lo esperado

Si tu dataset termina con menos muestras utilizables de lo esperado, los archivos pueden haber sido filtrados durante el procesamiento. Razones comunes incluyen:

- **Archivos demasiado pequeños**: Las imágenes por debajo de `minimum_image_size` son filtradas
- **Relación de aspecto fuera de rango**: Las imágenes fuera de los límites de `minimum_aspect_ratio`/`maximum_aspect_ratio` son excluidas
- **Límites de duración**: Los archivos de audio/video que exceden los límites de duración son omitidos

**Ver estadísticas de filtrado:**
- En la WebUI, navega al directorio de tu dataset y selecciónalo para ver estadísticas de filtrado
- Revisa los logs durante el procesamiento del dataset para estadísticas como: `Sample processing statistics: {'total_processed': 100, 'skipped': {'too_small': 15, ...}}`

Para solución de problemas detallada, consulta [Solución de problemas de datasets filtrados](DATALOADER.es.md#solución-de-problemas-de-datasets-filtrados) en la documentación del dataloader.
