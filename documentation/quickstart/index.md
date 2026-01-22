# Model Guides

Step-by-step guides for training each supported model architecture.

## Image Models

### Flow Matching

| Model | Parameters | Guide |
|-------|------------|-------|
| **Flux.1** | 12B | [Flux.1 Guide](FLUX.md) |
| **Flux.2** | 32B | [Flux.2 Guide](FLUX2.md) |
| **Flux Kontext** | 12B | [Kontext Guide](FLUX_KONTEXT.md) |
| **Chroma** | 8.9B | [Chroma Guide](CHROMA.md) |
| **Stable Diffusion 3** | 2-8B | [SD3 Guide](SD3.md) |
| **Auraflow** | 6.8B | [Auraflow Guide](AURAFLOW.md) |
| **Sana** | 0.6-4.8B | [Sana Guide](SANA.md) |
| **Lumina2** | 2B | [Lumina2 Guide](LUMINA2.md) |
| **HiDream** | 17B MoE | [HiDream Guide](HIDREAM.md) |
| **Z-Image** | - | [Z-Image Guide](ZIMAGE.md) |

### DiT / Transformer

| Model | Parameters | Guide |
|-------|------------|-------|
| **PixArt Sigma** | 0.6-0.9B | [Sigma Guide](SIGMA.md) |
| **Cosmos2** | 2-14B | [Cosmos2 Guide](COSMOS2IMAGE.md) |
| **OmniGen** | 3.8B | [OmniGen Guide](OMNIGEN.md) |
| **Qwen Image** | 20B | [Qwen Guide](QWEN_IMAGE.md) |
| **LongCat Image** | 6B | [LongCat Guide](LONGCAT_IMAGE.md) |
| **Kandinsky 5** | - | [Kandinsky Guide](KANDINSKY5_IMAGE.md) |

### U-Net

| Model | Parameters | Guide |
|-------|------------|-------|
| **Stable Diffusion XL** | 3.5B | [SDXL Guide](SDXL.md) |
| **Kolors** | 5B | [Kolors Guide](KOLORS.md) |
| **Stable Cascade** | - | [Cascade Guide](STABLE_CASCADE_C.md) |

### Image Editing

| Model | Guide |
|-------|-------|
| **Qwen Edit** | [Qwen Edit Guide](QWEN_EDIT.md) |
| **LongCat Edit** | [LongCat Edit Guide](LONGCAT_EDIT.md) |

## Video Models

| Model | Parameters | Guide |
|-------|------------|-------|
| **Wan Video** | 1.3-14B | [Wan Guide](WAN.md) |
| **LTX Video** | 5B | [LTX Guide](LTXVIDEO.md) |
| **LTX Video 2** | 19B | [LTX Video 2 Guide](LTXVIDEO2.md) |
| **Hunyuan Video** | 8.3B | [Hunyuan Guide](HUNYUANVIDEO.md) |
| **Sana Video** | - | [Sana Video Guide](SANAVIDEO.md) |
| **Kandinsky 5 Video** | - | [Kandinsky Video Guide](KANDINSKY5_VIDEO.md) |
| **LongCat Video** | - | [LongCat Video Guide](LONGCAT_VIDEO.md) |
| **LongCat Video Edit** | - | [LongCat Video Edit Guide](LONGCAT_VIDEO_EDIT.md) |

## Audio Models

| Model | Parameters | Guide |
|-------|------------|-------|
| **ACE-Step** | 3.5B | [ACE-Step Guide](ACE_STEP.md) |
| **HeartMuLa** | 3B | [HeartMuLa Guide](HEARTMULA.md) |

## Choosing a Model

**For beginners:**

- Start with **Flux.1** for high-quality image generation
- Use **LoRA** training to reduce memory requirements

**For production:**

- **SD3** or **SDXL** for broad compatibility
- **Flux.2** for maximum quality (requires more VRAM)

**For video:**

- **Wan Video** for best quality/resource balance
- **Hunyuan Video** for I2V with super-resolution

**For specific use cases:**

- **Flux Kontext** for image editing/conditioning
- **ACE-Step** for text-to-music
- **HeartMuLa** for autoregressive text-to-audio
