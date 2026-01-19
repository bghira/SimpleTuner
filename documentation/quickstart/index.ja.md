# モデルガイド

対応する各モデルアーキテクチャのトレーニング手順ガイドです。

## 画像モデル

### Flow Matching

| モデル | パラメータ | ガイド |
|-------|------------|-------|
| **Flux.1** | 12B | [Flux.1 ガイド](FLUX.md) |
| **Flux.2** | 32B | [Flux.2 ガイド](FLUX2.md) |
| **Flux Kontext** | 12B | [Kontext ガイド](FLUX_KONTEXT.md) |
| **Chroma** | 8.9B | [Chroma ガイド](CHROMA.md) |
| **Stable Diffusion 3** | 2-8B | [SD3 ガイド](SD3.md) |
| **Auraflow** | 6.8B | [Auraflow ガイド](AURAFLOW.md) |
| **Sana** | 0.6-4.8B | [Sana ガイド](SANA.md) |
| **Lumina2** | 2B | [Lumina2 ガイド](LUMINA2.md) |
| **HiDream** | 17B MoE | [HiDream ガイド](HIDREAM.md) |
| **Z-Image** | - | [Z-Image ガイド](ZIMAGE.md) |

### DiT / Transformer

| モデル | パラメータ | ガイド |
|-------|------------|-------|
| **PixArt Sigma** | 0.6-0.9B | [Sigma ガイド](SIGMA.md) |
| **Cosmos2** | 2-14B | [Cosmos2 ガイド](COSMOS2IMAGE.md) |
| **OmniGen** | 3.8B | [OmniGen ガイド](OMNIGEN.md) |
| **Qwen Image** | 20B | [Qwen ガイド](QWEN_IMAGE.md) |
| **LongCat Image** | 6B | [LongCat ガイド](LONGCAT_IMAGE.md) |
| **Kandinsky 5** | - | [Kandinsky ガイド](KANDINSKY5_IMAGE.md) |

### U-Net

| モデル | パラメータ | ガイド |
|-------|------------|-------|
| **Stable Diffusion XL** | 3.5B | [SDXL ガイド](SDXL.md) |
| **Kolors** | 5B | [Kolors ガイド](KOLORS.md) |
| **Stable Cascade** | - | [Cascade ガイド](STABLE_CASCADE_C.md) |

### 画像編集

| モデル | ガイド |
|-------|-------|
| **Qwen Edit** | [Qwen Edit ガイド](QWEN_EDIT.md) |
| **LongCat Edit** | [LongCat Edit ガイド](LONGCAT_EDIT.md) |

## 動画モデル

| モデル | パラメータ | ガイド |
|-------|------------|-------|
| **Wan Video** | 1.3-14B | [Wan ガイド](WAN.md) |
| **LTX Video** | 5B | [LTX ガイド](LTXVIDEO.md) |
| **LTX Video 2** | 19B | [LTX Video 2 ガイド](LTXVIDEO2.md) |
| **Hunyuan Video** | 8.3B | [Hunyuan ガイド](HUNYUANVIDEO.md) |
| **Sana Video** | - | [Sana Video ガイド](SANAVIDEO.md) |
| **Kandinsky 5 Video** | - | [Kandinsky Video ガイド](KANDINSKY5_VIDEO.md) |
| **LongCat Video** | - | [LongCat Video ガイド](LONGCAT_VIDEO.md) |
| **LongCat Video Edit** | - | [LongCat Video Edit ガイド](LONGCAT_VIDEO_EDIT.md) |

## 音声モデル

| モデル | パラメータ | ガイド |
|-------|------------|-------|
| **ACE-Step** | 3.5B | [ACE-Step ガイド](ACE_STEP.md) |
| **HeartMuLa** | 3B | [HeartMuLa ガイド](HEARTMULA.md) |

## モデルの選び方

**初心者向け:**

- 高品質な画像生成には **Flux.1** から始めてください
- VRAM 要件を下げるには **LoRA** トレーニングを使ってください

**プロダクション向け:**

- 幅広い互換性には **SD3** または **SDXL**
- 最高品質には **Flux.2**（より多くの VRAM が必要）

**動画向け:**

- 品質とリソースのバランスには **Wan Video**
- I2V と超解像には **Hunyuan Video**

**用途別:**

- 画像編集/コンディショニングには **Flux Kontext**
- テキストから音楽には **ACE-Step**
- 自己回帰のテキストから音声には **HeartMuLa**
