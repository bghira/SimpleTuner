# Guías de modelos

Guías paso a paso para entrenar cada arquitectura de modelo compatible.

## Modelos de imagen

### Flow Matching

| Modelo | Parámetros | Guía |
|-------|------------|-------|
| **Flux.1** | 12B | [Guía de Flux.1](FLUX.md) |
| **Flux.2** | 32B | [Guía de Flux.2](FLUX2.md) |
| **Flux Kontext** | 12B | [Guía de Kontext](FLUX_KONTEXT.md) |
| **Chroma** | 8.9B | [Guía de Chroma](CHROMA.md) |
| **Stable Diffusion 3** | 2-8B | [Guía de SD3](SD3.md) |
| **Auraflow** | 6.8B | [Guía de Auraflow](AURAFLOW.md) |
| **Sana** | 0.6-4.8B | [Guía de Sana](SANA.md) |
| **Lumina2** | 2B | [Guía de Lumina2](LUMINA2.md) |
| **HiDream** | 17B MoE | [Guía de HiDream](HIDREAM.md) |
| **Z-Image** | - | [Guía de Z-Image](ZIMAGE.md) |

### DiT / Transformer

| Modelo | Parámetros | Guía |
|-------|------------|-------|
| **PixArt Sigma** | 0.6-0.9B | [Guía de Sigma](SIGMA.md) |
| **Cosmos2** | 2-14B | [Guía de Cosmos2](COSMOS2IMAGE.md) |
| **OmniGen** | 3.8B | [Guía de OmniGen](OMNIGEN.md) |
| **Qwen Image** | 20B | [Guía de Qwen](QWEN_IMAGE.md) |
| **LongCat Image** | 6B | [Guía de LongCat](LONGCAT_IMAGE.md) |
| **Kandinsky 5** | - | [Guía de Kandinsky](KANDINSKY5_IMAGE.md) |

### U-Net

| Modelo | Parámetros | Guía |
|-------|------------|-------|
| **Stable Diffusion XL** | 3.5B | [Guía de SDXL](SDXL.md) |
| **Kolors** | 5B | [Guía de Kolors](KOLORS.md) |
| **Stable Cascade** | - | [Guía de Cascade](STABLE_CASCADE_C.md) |

### Edición de imágenes

| Modelo | Guía |
|-------|-------|
| **Qwen Edit** | [Guía de Qwen Edit](QWEN_EDIT.md) |
| **LongCat Edit** | [Guía de LongCat Edit](LONGCAT_EDIT.md) |

## Modelos de video

| Modelo | Parámetros | Guía |
|-------|------------|-------|
| **Wan Video** | 1.3-14B | [Guía de Wan](WAN.md) |
| **LTX Video** | 5B | [Guía de LTX](LTXVIDEO.md) |
| **Hunyuan Video** | 8.3B | [Guía de Hunyuan](HUNYUANVIDEO.md) |
| **Sana Video** | - | [Guía de Sana Video](SANAVIDEO.md) |
| **Kandinsky 5 Video** | - | [Guía de Kandinsky Video](KANDINSKY5_VIDEO.md) |
| **LongCat Video** | - | [Guía de LongCat Video](LONGCAT_VIDEO.md) |
| **LongCat Video Edit** | - | [Guía de LongCat Video Edit](LONGCAT_VIDEO_EDIT.md) |

## Modelos de audio

| Modelo | Parámetros | Guía |
|-------|------------|-------|
| **ACE-Step** | 3.5B | [Guía de ACE-Step](ACE_STEP.md) |

## Elegir un modelo

**Para principiantes:**

- Empieza con **Flux.1** para generación de imágenes de alta calidad
- Usa entrenamiento **LoRA** para reducir requisitos de memoria

**Para producción:**

- **SD3** o **SDXL** para compatibilidad amplia
- **Flux.2** para máxima calidad (requiere más VRAM)

**Para video:**

- **Wan Video** para el mejor equilibrio calidad/recursos
- **Hunyuan Video** para I2V con super-resolución

**Para casos de uso específicos:**

- **Flux Kontext** para edición/condicionamiento de imágenes
- **ACE-Step** para texto a música
