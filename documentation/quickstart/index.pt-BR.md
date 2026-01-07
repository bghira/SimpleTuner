# Guias de Modelos

Guias passo a passo para treinar cada arquitetura de modelo suportada.

## Modelos de Imagem

### Flow Matching

| Modelo | Parâmetros | Guia |
|-------|------------|------|
| **Flux.1** | 12B | [Guia Flux.1](FLUX.md) |
| **Flux.2** | 32B | [Guia Flux.2](FLUX2.md) |
| **Flux Kontext** | 12B | [Guia Kontext](FLUX_KONTEXT.md) |
| **Chroma** | 8.9B | [Guia Chroma](CHROMA.md) |
| **Stable Diffusion 3** | 2-8B | [Guia SD3](SD3.md) |
| **Auraflow** | 6.8B | [Guia Auraflow](AURAFLOW.md) |
| **Sana** | 0.6-4.8B | [Guia Sana](SANA.md) |
| **Lumina2** | 2B | [Guia Lumina2](LUMINA2.md) |
| **HiDream** | 17B MoE | [Guia HiDream](HIDREAM.md) |
| **Z-Image** | - | [Guia Z-Image](ZIMAGE.md) |

### DiT / Transformador

| Modelo | Parâmetros | Guia |
|-------|------------|------|
| **PixArt Sigma** | 0.6-0.9B | [Guia Sigma](SIGMA.md) |
| **Cosmos2** | 2-14B | [Guia Cosmos2](COSMOS2IMAGE.md) |
| **OmniGen** | 3.8B | [Guia OmniGen](OMNIGEN.md) |
| **Qwen Image** | 20B | [Guia Qwen](QWEN_IMAGE.md) |
| **LongCat Image** | 6B | [Guia LongCat](LONGCAT_IMAGE.md) |
| **Kandinsky 5** | - | [Guia Kandinsky](KANDINSKY5_IMAGE.md) |

### U-Net

| Modelo | Parâmetros | Guia |
|-------|------------|------|
| **Stable Diffusion XL** | 3.5B | [Guia SDXL](SDXL.md) |
| **Kolors** | 5B | [Guia Kolors](KOLORS.md) |
| **Stable Cascade** | - | [Guia Cascade](STABLE_CASCADE_C.md) |

### Edição de Imagem

| Modelo | Guia |
|-------|------|
| **Qwen Edit** | [Guia Qwen Edit](QWEN_EDIT.md) |
| **LongCat Edit** | [Guia LongCat Edit](LONGCAT_EDIT.md) |

## Modelos de Vídeo

| Modelo | Parâmetros | Guia |
|-------|------------|------|
| **Wan Video** | 1.3-14B | [Guia Wan](WAN.md) |
| **LTX Video** | 5B | [Guia LTX](LTXVIDEO.md) |
| **LTX Video 2** | 19B | [Guia LTX Video 2](LTXVIDEO2.md) |
| **Hunyuan Video** | 8.3B | [Guia Hunyuan](HUNYUANVIDEO.md) |
| **Sana Video** | - | [Guia Sana Video](SANAVIDEO.md) |
| **Kandinsky 5 Video** | - | [Guia Kandinsky Video](KANDINSKY5_VIDEO.md) |
| **LongCat Video** | - | [Guia LongCat Video](LONGCAT_VIDEO.md) |
| **LongCat Video Edit** | - | [Guia LongCat Video Edit](LONGCAT_VIDEO_EDIT.md) |

## Modelos de Áudio

| Modelo | Parâmetros | Guia |
|-------|------------|------|
| **ACE-Step** | 3.5B | [Guia ACE-Step](ACE_STEP.md) |

## Como escolher um modelo

**Para iniciantes:**

- Comece com **Flux.1** para geração de imagens de alta qualidade
- Use treinamento **LoRA** para reduzir requisitos de memória

**Para produção:**

- **SD3** ou **SDXL** para ampla compatibilidade
- **Flux.2** para máxima qualidade (requer mais VRAM)

**Para vídeo:**

- **Wan Video** para melhor equilíbrio entre qualidade e recursos
- **Hunyuan Video** para I2V com super-resolução

**Para casos específicos:**

- **Flux Kontext** para edição/condicionamento de imagens
- **ACE-Step** para texto-para-música
