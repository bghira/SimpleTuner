# 模型指南

面向每个支持的模型架构的逐步训练指南。

## 图像模型

### 流匹配

| 模型 | 参数 | 指南 |
|-------|------------|-------|
| **Flux.1** | 12B | [Flux.1 指南](FLUX.md) |
| **Flux.2** | 32B | [Flux.2 指南](FLUX2.md) |
| **Flux Kontext** | 12B | [Kontext 指南](FLUX_KONTEXT.md) |
| **Chroma** | 8.9B | [Chroma 指南](CHROMA.md) |
| **Stable Diffusion 3** | 2-8B | [SD3 指南](SD3.md) |
| **Auraflow** | 6.8B | [Auraflow 指南](AURAFLOW.md) |
| **Sana** | 0.6-4.8B | [Sana 指南](SANA.md) |
| **Lumina2** | 2B | [Lumina2 指南](LUMINA2.md) |
| **HiDream** | 17B MoE | [HiDream 指南](HIDREAM.md) |
| **Z-Image** | - | [Z-Image 指南](ZIMAGE.md) |

### DiT / Transformer

| 模型 | 参数 | 指南 |
|-------|------------|-------|
| **PixArt Sigma** | 0.6-0.9B | [Sigma 指南](SIGMA.md) |
| **Cosmos2** | 2-14B | [Cosmos2 指南](COSMOS2IMAGE.md) |
| **OmniGen** | 3.8B | [OmniGen 指南](OMNIGEN.md) |
| **Qwen Image** | 20B | [Qwen 指南](QWEN_IMAGE.md) |
| **LongCat Image** | 6B | [LongCat 指南](LONGCAT_IMAGE.md) |
| **Kandinsky 5** | - | [Kandinsky 指南](KANDINSKY5_IMAGE.md) |

### U-Net

| 模型 | 参数 | 指南 |
|-------|------------|-------|
| **Stable Diffusion XL** | 3.5B | [SDXL 指南](SDXL.md) |
| **Kolors** | 5B | [Kolors 指南](KOLORS.md) |
| **Stable Cascade** | - | [Cascade 指南](STABLE_CASCADE_C.md) |

### 图像编辑

| 模型 | 指南 |
|-------|-------|
| **Qwen Edit** | [Qwen Edit 指南](QWEN_EDIT.md) |
| **LongCat Edit** | [LongCat Edit 指南](LONGCAT_EDIT.md) |

## 视频模型

| 模型 | 参数 | 指南 |
|-------|------------|-------|
| **Wan Video** | 1.3-14B | [Wan 指南](WAN.md) |
| **LTX Video** | 5B | [LTX 指南](LTXVIDEO.md) |
| **LTX Video 2** | 19B | [LTX Video 2 指南](LTXVIDEO2.md) |
| **Hunyuan Video** | 8.3B | [Hunyuan 指南](HUNYUANVIDEO.md) |
| **Sana Video** | - | [Sana Video 指南](SANAVIDEO.md) |
| **Kandinsky 5 Video** | - | [Kandinsky Video 指南](KANDINSKY5_VIDEO.md) |
| **LongCat Video** | - | [LongCat Video 指南](LONGCAT_VIDEO.md) |
| **LongCat Video Edit** | - | [LongCat Video Edit 指南](LONGCAT_VIDEO_EDIT.md) |

## 音频模型

| 模型 | 参数 | 指南 |
|-------|------------|-------|
| **ACE-Step** | 3.5B | [ACE-Step 指南](ACE_STEP.md) |
| **HeartMuLa** | 3B | [HeartMuLa 指南](HEARTMULA.md) |

## 选择模型

**新手:**

- 获取高质量图像生成请从 **Flux.1** 开始
- 使用 **LoRA** 训练降低显存需求

**生产:**

- 需要更广兼容性可选 **SD3** 或 **SDXL**
- 追求最高质量可选 **Flux.2**（需要更多显存）

**视频:**

- 质量与资源平衡可选 **Wan Video**
- I2V 与超分辨率可选 **Hunyuan Video**

**特定场景:**

- 图像编辑/条件控制可选 **Flux Kontext**
- 文本到音乐可选 **ACE-Step**
- 自回归文本到音频可选 **HeartMuLa**
