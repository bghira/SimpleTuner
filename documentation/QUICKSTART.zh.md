# 快速开始指南

**注意**：如需更高级的配置，请参阅[教程](TUTORIAL.md)和[选项参考](OPTIONS.md)。

## 模型快速开始指南

| 模型 | 参数量 | 指南 |
| --- | --- | --- |
| ACE-Step | 3.5B | [ACE_STEP.zh.md](quickstart/ACE_STEP.zh.md) |
| Anima | Not specified | 暂无专用指南 |
| Auraflow | 6B | [AURAFLOW.zh.md](quickstart/AURAFLOW.zh.md) |
| Boogu-Image | Not specified | [BOOGU_IMAGE.zh.md](quickstart/BOOGU_IMAGE.zh.md) |
| Chroma 1 | 8.9B | [CHROMA.zh.md](quickstart/CHROMA.zh.md) |
| Cosmos2 | 2B-14B | [COSMOS2IMAGE.zh.md](quickstart/COSMOS2IMAGE.zh.md) |
| Cosmos3 | 16B-65B | [COSMOS3.zh.md](quickstart/COSMOS3.zh.md) |
| DeepFloyd IF | 0.4B-4.3B stages | 暂无专用指南 |
| ERNIE-Image | Not specified | [ERNIE.zh.md](quickstart/ERNIE.zh.md) |
| Flux.1 | 8B-12B | [FLUX.zh.md](quickstart/FLUX.zh.md)<br>[FLUX_KONTEXT.zh.md](quickstart/FLUX_KONTEXT.zh.md) |
| Flux.2 | 4B-32B | [FLUX2.zh.md](quickstart/FLUX2.zh.md) |
| HeartMuLa | 3B | [HEARTMULA.zh.md](quickstart/HEARTMULA.zh.md) |
| HiDream | 17B (8.5B MoE) | [HIDREAM.zh.md](quickstart/HIDREAM.zh.md) |
| Hunyuan Video | 8.3B | [HUNYUANVIDEO.zh.md](quickstart/HUNYUANVIDEO.zh.md) |
| Ideogram 4 | 9B | [IDEOGRAM4.zh.md](quickstart/IDEOGRAM4.zh.md) |
| InfiniteTalk | 14B | [INFINITETALK.zh.md](quickstart/INFINITETALK.zh.md) |
| Kandinsky 5.0 Image | 6B (lite) | [KANDINSKY5_IMAGE.zh.md](quickstart/KANDINSKY5_IMAGE.zh.md) |
| Kandinsky 5.0 Video | 2B lite, 19B pro | [KANDINSKY5_VIDEO.zh.md](quickstart/KANDINSKY5_VIDEO.zh.md) |
| Kwai Kolors | 2.7B | [KOLORS.zh.md](quickstart/KOLORS.zh.md) |
| Krea2 | Not specified | [KREA2.zh.md](quickstart/KREA2.zh.md) |
| LongCat Image | 6B | [LONGCAT_IMAGE.zh.md](quickstart/LONGCAT_IMAGE.zh.md)<br>[LONGCAT_EDIT.zh.md](quickstart/LONGCAT_EDIT.zh.md) |
| LongCat Video | 13.6B | [LONGCAT_VIDEO.zh.md](quickstart/LONGCAT_VIDEO.zh.md)<br>[LONGCAT_VIDEO_EDIT.zh.md](quickstart/LONGCAT_VIDEO_EDIT.zh.md) |
| LTX Video | ~2.5B | [LTXVIDEO.zh.md](quickstart/LTXVIDEO.zh.md) |
| LTX Video 2 | 19B | [LTXVIDEO2.zh.md](quickstart/LTXVIDEO2.zh.md) |
| Lumina2 | 2B | [LUMINA2.zh.md](quickstart/LUMINA2.zh.md) |
| Mage-Flow | 4B | [MAGEFLOW.zh.md](quickstart/MAGEFLOW.zh.md) |
| MiniMax H3 | 33B | [MINIMAX_H3.zh.md](/documentation/quickstart/MINIMAX_H3.zh.md) |
| MiniMax Music 3 | 2.4B transformer + 8B AR | [MINIMAX_MUSIC.zh.md](/documentation/quickstart/MINIMAX_MUSIC.zh.md) |
| OmniGen | 3.8B | [OMNIGEN.zh.md](quickstart/OMNIGEN.zh.md) |
| PixArt Sigma | 0.6B-0.9B | [SIGMA.zh.md](quickstart/SIGMA.zh.md) |
| Qwen Image | 20B | [QWEN_IMAGE.zh.md](quickstart/QWEN_IMAGE.zh.md)<br>[QWEN_EDIT.zh.md](quickstart/QWEN_EDIT.zh.md) |
| Sana | 0.6B-4.8B | [SANA.zh.md](quickstart/SANA.zh.md) |
| Sana Video | 2B | [SANAVIDEO.zh.md](quickstart/SANAVIDEO.zh.md) |
| SD 1.x/2.x (Legacy) | 0.9B | 暂无专用指南 |
| Stable Diffusion 3 | 2B-8B | [SD3.zh.md](quickstart/SD3.zh.md) |
| Stable Diffusion XL | 3.5B | [SDXL.zh.md](quickstart/SDXL.zh.md) |
| Stable Cascade (Stage C) | 1B, 3.6B prior | [STABLE_CASCADE_C.zh.md](quickstart/STABLE_CASCADE_C.zh.md) |
| Wan Video | 1.3B-14B | [WAN.zh.md](quickstart/WAN.zh.md) |
| Wan S2V | 14B | [WAN_S2V.zh.md](quickstart/WAN_S2V.zh.md) |
| Z-Image | 6B | [ZIMAGE.zh.md](quickstart/ZIMAGE.zh.md) |
| Z-Image Omni | 6B | [ZIMAGE.zh.md](quickstart/ZIMAGE.zh.md) |
| ZLab I1 | 3B | [ZLAB_i1.zh.md](quickstart/ZLAB_i1.zh.md) |

## 功能兼容性

完整兼容性矩阵按功能领域拆分，以保持每张表易读。

<details>
<summary>训练支持</summary>

| 模型 | PEFT LoRA | LyCORIS | 全秩 | ControlNet | Ref Inputs |
| --- | :---: | :---: | :---: | :---: | :---: |
| ACE-Step | ✓ | ✓ | ✓* | ✗ | ✗ |
| Anima | ✓ | ✓ | ✓* | ✗ | ✗ |
| Auraflow | ✓ | ✓ | ✓* | ✓ | ✗ |
| Boogu-Image | ✓ | ✓ | ✓* | ✗ | ✓ edit |
| Chroma 1 | ✓ | ✓ | ✓* | ✗ | ✗ |
| Cosmos2 | ✓ | ✓ | ✓ | ✗ | ✗ |
| Cosmos3 | ✓ | ✓ | ✓* | ✗ | audio opt |
| DeepFloyd IF | ✓ | ✓ | ✓ | ✗ | ✗ |
| ERNIE-Image | ✓ | ✓ | ✓* | ✗ | ✗ |
| Flux.1 | ✓ | ✓ | ✓* | ✓ | ✓ opt (Kontext) |
| Flux.2 | ✓ | ✓ | ✓* | ✗ | ✓ opt |
| HeartMuLa | ✓ | ✓ | ✓* | ✗ | ✗ |
| HiDream | ✓ | ✓ | ✓* | ✓ | ✗ |
| Hunyuan Video | ✓ | ✓ | ✓* | ✗ | ✓ I2V |
| Ideogram 4 | ✓ | ✓ | ✓* | ✗ | ✗ |
| InfiniteTalk | ✓ | ✓ | ✓* | ✗ | audio + I2V req |
| Kandinsky 5.0 Image | ✓ | ✓ | ✓* | ✗ | ✓ I2I |
| Kandinsky 5.0 Video | ✓ | ✓ | ✓* | ✗ | ✓ I2V |
| Kwai Kolors | ✓ | ✓ | ✓ | ✗ | ✗ |
| Krea2 | ✓ | ✓ | ✓* | ✗ | ✓ opt |
| LongCat Image | ✓ | ✓ | ✓* | ✗ | ✓ req (Edit) |
| LongCat Video | ✓ | ✓ | ✓* | ✗ | ✓ opt/edit |
| LTX Video | ✓ | ✓ | ✓ | ✗ | ✓ I2V |
| LTX Video 2 | ✓ | ✓ | ✓* | ✗ | ✓ opt |
| Lumina2 | ✓ | ✓ | ✓ | ✗ | ✗ |
| Mage-Flow | ✓ | ✓ | ✓* | ✗ | ✓ edit |
| MiniMax H3 | ✓ | ✓ | ✓* | ✗ | ✓ opt (FL2VA/Ref2VA) |
| MiniMax Music 3 | ✓ | ✓ | ✓* | ✗ | lyrics |
| OmniGen | ✓ | ✓ | ✓ | ✗ | ✗ |
| PixArt Sigma | ✗ | ✓ | ✓ | ✓ | ✗ |
| Qwen Image | ✓ | ✓ | ✓* | ✗ | ✓ req (Edit) |
| Sana | ✗ | ✓ | ✓ | ✗ | ✗ |
| Sana Video | ✓ | ✓ | ✓ | ✗ | ✗ |
| SD 1.x/2.x (Legacy) | ✓ | ✓ | ✓ | ✓ | ✗ |
| Stable Diffusion 3 | ✓ | ✓ | ✓* | ✓ | ✗ |
| Stable Diffusion XL | ✓ | ✓ | ✓ | ✓ | ✗ |
| Stable Cascade (Stage C) | ✓ | ✓ | ✓* | ✗ | ✗ |
| Wan Video | ✓ | ✓ | ✓* | ✗ | ✓ I2V/VACE |
| Wan S2V | ✓ | ✓ | ✓* | ✗ | ✗ |
| Z-Image | ✓ | ✓ | ✓* | ✗ | ✗ |
| Z-Image Omni | ✓ | ✓ | ✓* | ✗ | ✓ opt (Edit) |
| ZLab I1 | ✓ | ✓ | ✓ | ✗ | ✗ |

</details>

<details>
<summary>精度级别支持</summary>

| 模型 | 量化 | 混合精度 |
| --- | --- | --- |
| ACE-Step | int8 optional | bf16 |
| Anima | not specified | bf16 |
| Auraflow | int8/fp8/nf4 optional | bf16 |
| Boogu-Image | fp8 optional | bf16 |
| Chroma 1 | int8/fp8/nf4 optional | bf16 |
| Cosmos2 | int8 optional | bf16 |
| Cosmos3 | no_change first; int8 optional | bf16 |
| DeepFloyd IF | not recommended | bf16 |
| ERNIE-Image | int8 optional | bf16 |
| Flux.1 | int8/fp8/nf4 optional | bf16 |
| Flux.2 | int8/fp8/nf4 optional | bf16 |
| HeartMuLa | int8 optional | bf16 |
| HiDream | int8/fp8/nf4 optional | bf16 |
| Hunyuan Video | int8 optional | bf16 |
| Ideogram 4 | fp8 default, nf4 optional | bf16 |
| InfiniteTalk | int8 optional | bf16 |
| Kandinsky 5.0 Image | int8 optional | bf16 |
| Kandinsky 5.0 Video | int8 optional | bf16 |
| Kwai Kolors | not recommended | bf16 |
| Krea2 | int8 optional | bf16 |
| LongCat Image | int8/fp8 optional | bf16 |
| LongCat Video | int8/fp8 optional | bf16 |
| LTX Video | int8/fp8 optional | bf16 |
| LTX Video 2 | int8/fp8 optional | bf16 |
| Lumina2 | int8 optional | bf16 |
| Mage-Flow | fp8 optional | bf16 |
| MiniMax H3 | int8/fp8 optional; convrot-int8 | bf16 |
| MiniMax Music 3 | int8 optional | bf16 |
| OmniGen | int8/fp8 optional | bf16 |
| PixArt Sigma | int8 optional | bf16 |
| Qwen Image | required (int8/nf4) | bf16 |
| Sana | int8 optional | bf16 |
| Sana Video | not recommended for full | bf16 |
| SD 1.x/2.x (Legacy) | int8/nf4 optional | bf16 |
| Stable Diffusion 3 | int8/fp8/nf4 optional | bf16 |
| Stable Diffusion XL | int8/nf4 optional | bf16 |
| Stable Cascade (Stage C) | not supported | fp32 required |
| Wan Video | int8 optional | bf16 |
| Wan S2V | int8 optional | bf16 |
| Z-Image | int8 optional | bf16 |
| Z-Image Omni | int8 optional | bf16 |
| ZLab I1 | int8 optional | bf16 |

</details>

<details>
<summary>检查点粒度</summary>

| 模型 | Gradient Checkpoint | Interval | Segment Stride | Attention Offload |
| --- | :---: | :---: | :---: | :---: |
| ACE-Step | ✓ | ✓ | ✓ | ✗ |
| Anima | ✓ | ✗ | ✗ | ✗ |
| Auraflow | ✓ | ✓ | ✓ | ✗ |
| Boogu-Image | ✓ | ✓ | ✓ | ✗ |
| Chroma 1 | ✓ | ✓ | ✓ | ✓ |
| Cosmos2 | ✓ | ✓ | ✓ | ✗ |
| Cosmos3 | ✓ | ✓ | ✓ | ✗ |
| DeepFloyd IF | ✓ | ✗ | ✗ | ✗ |
| ERNIE-Image | ✓ | ✓ | ✓ | ✗ |
| Flux.1 | ✓ | ✓ | ✓ | ✓ |
| Flux.2 | ✓ | ✓ | ✓ | ✓ |
| HeartMuLa | ✓ | ✗ | ✗ | ✗ |
| HiDream | ✓ | ✓ | ✓ | ✗ |
| Hunyuan Video | ✓ | ✓ | ✓ | ✓ |
| Ideogram 4 | ✓ | ✓ | ✓ | ✗ |
| InfiniteTalk | ✓ | ✓ | ✓ | ✓ |
| Kandinsky 5.0 Image | ✓ | ✓ | ✓ | ✓ |
| Kandinsky 5.0 Video | ✓ | ✓ | ✓ | ✓ |
| Kwai Kolors | ✓ | ✗ | ✗ | ✗ |
| Krea2 | ✓ | ✓ | ✓ | ✓ |
| LongCat Image | ✓ | ✓ | ✓ | ✓ |
| LongCat Video | ✓ | ✓ | ✓ | ✓ |
| LTX Video | ✓ | ✓ | ✓ | ✗ |
| LTX Video 2 | ✓ | ✓ | ✓ | ✓ |
| Lumina2 | ✓ | ✓ | ✓ | ✗ |
| Mage-Flow | ✓ | ✓ | ✓ | ✓ |
| MiniMax H3 | ✓ | ✓ | ✓ | ✓ |
| MiniMax Music 3 | ✓ | ✓ | ✓ | ✗ |
| OmniGen | ✓ | ✗ | ✗ | ✗ |
| PixArt Sigma | ✓ | ✓ | ✓ | ✗ |
| Qwen Image | ✓ | ✓ | ✓ | ✗ |
| Sana | ✓ | ✓ | ✓ | ✗ |
| Sana Video | ✓ | ✓ | ✓ | ✗ |
| SD 1.x/2.x (Legacy) | ✓ | ✗ | ✗ | ✗ |
| Stable Diffusion 3 | ✓ | ✓ | ✓ | ✓ |
| Stable Diffusion XL | ✓ | ✗ | ✗ | ✗ |
| Stable Cascade (Stage C) | ✓ | ✓ | ✓ | ✗ |
| Wan Video | ✓ | ✓ | ✓ | ✓ |
| Wan S2V | ✓ | ✓ | ✓ | ✗ |
| Z-Image | ✓ | ✓ | ✓ | ✓ |
| Z-Image Omni | ✓ | ✗ | ✗ | ✗ |
| ZLab I1 | ✓ | ✓ | ✓ | ✗ |

</details>

<details>
<summary>Flow、蒸馏与对齐</summary>

| 模型 | Prediction | Flow Shift | TwinFlow | Self-Flow | LayerSync | Sliders |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| ACE-Step | flow matching | ✓ | ✓ | ✗ | ✓ | ✓ |
| Anima | flow matching | ✓ | ✗ | ✓ | ✓ | ✓ |
| Auraflow | flow matching | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ |
| Boogu-Image | flow matching | ✓ | ✗ | ✗ | ✗ | ✓ |
| Chroma 1 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| Cosmos2 | sample | ✗ | ✗ | ✓ | ✓ | ✓ |
| Cosmos3 | flow matching | ✓ | ✗ | ✗ | ✗ | ✓ |
| DeepFloyd IF | epsilon | ✗ | ✗ | ✗ | ✗ | ✓ |
| ERNIE-Image | flow matching | ✓ | ✓ | ✗ | ✓ | ✓ |
| Flux.1 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| Flux.2 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| HeartMuLa | autoregressive next-token | ✗ | ✗ | ✗ | ✗ | ✗ |
| HiDream | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| Hunyuan Video | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ideogram 4 | flow matching | ✓ | ✗ | ✗ | ✗ | ✓ |
| InfiniteTalk | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| Kandinsky 5.0 Image | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| Kandinsky 5.0 Video | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| Kwai Kolors | epsilon | ✗ | ✗ | ✗ | ✗ | ✓ |
| Krea2 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| LongCat Image | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| LongCat Video | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| LTX Video | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| LTX Video 2 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| Lumina2 | flow matching | ✓ | ✗ | ✓ | ✓ | ✓ |
| Mage-Flow | flow matching | ✓ | ✓ | ✗ | ✓ | ✓ |
| MiniMax H3 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| MiniMax Music 3 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| OmniGen | flow matching | ✓ | ✗ | ✓ | ✓ | ✓ |
| PixArt Sigma | epsilon | ✗ | ✗ | ✓ | ✓ | ✓ |
| Qwen Image | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| Sana | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| Sana Video | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| SD 1.x/2.x (Legacy) | epsilon / v-pred | ✗ | ✗ | ✗ | ✗ | ✓ |
| Stable Diffusion 3 | flow matching | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ |
| Stable Diffusion XL | epsilon | ✗ | ✗ | ✗ | ✗ | ✓ |
| Stable Cascade (Stage C) | epsilon | ✗ | ✗ | ✗ | ✗ | ✓ |
| Wan Video | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| Wan S2V | flow matching | ✓ | ✗ | ✓ | ✓ | ✓ |
| Z-Image | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| Z-Image Omni | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| ZLab I1 | flow matching | ✓ | ✗ | ✓ | ✓ | ✓ |

</details>

<details>
<summary>文本编码器与 VAE 类型</summary>

| 模型 | Text Encoders | Text Encoder Params | VAE |
| --- | --- | --- | --- |
| ACE-Step | UMT5 Encoder | 0.6B | Music DCAE |
| Anima | Qwen3 0.6B | 0.6B | Qwen Image VAE |
| Auraflow | Pile T5 | not specified | AutoencoderKL |
| Boogu-Image | Qwen3-VL | not specified | AutoencoderKL |
| Chroma 1 | T5 XXL v1.1 | 11B | AutoencoderKL |
| Cosmos2 | T5 11B | 11B | Wan VAE |
| Cosmos3 | Cosmos3 reasoner | not specified | Wan/Cosmos VAE |
| DeepFloyd IF | T5 XXL v1.1 | 11B | None |
| ERNIE-Image | ERNIE text encoder | not specified | Flux.2 VAE |
| Flux.1 | CLIP-L/14 + T5 XXL v1.1 | 123M + 11B | AutoencoderKL |
| Flux.2 | Mistral-Small-3.1-24B | 24B | Flux.2 VAE |
| HeartMuLa | None | N/A | HeartCodec tokens |
| HiDream | CLIP-L/14 + CLIP-G/14 + T5 XXL v1.1 + Llama | 123M + 694M + 11B + not specified | AutoencoderKL |
| Hunyuan Video | Hunyuan LLM | not specified | Hunyuan Video 3D VAE |
| Ideogram 4 | Qwen3-VL-8B-Instruct | 8B | Ideogram AutoEncoder |
| InfiniteTalk | UMT5 + Wav2Vec2 | 未指定 + 95M | Wan VAE |
| Kandinsky 5.0 Image | Qwen2.5-VL + CLIP-L/14 | 7B + 123M | Flux VAE (AutoencoderKL) |
| Kandinsky 5.0 Video | Qwen2.5-VL + CLIP-L/14 | 7B + 123M | Hunyuan Video VAE |
| Kwai Kolors | ChatGLM-6B | 6B | AutoencoderKL |
| Krea2 | Qwen3VL | not specified | Qwen Image VAE |
| LongCat Image | Qwen2.5-VL | 7B | AutoencoderKL |
| LongCat Video | Qwen2.5-VL | 7B | Wan VAE |
| LTX Video | T5 XXL v1.1 | 11B | LTX Video VAE |
| LTX Video 2 | Gemma3 / Gemma4 | not specified | LTX Video 2 VAE |
| Lumina2 | Gemma2 | 2B | AutoencoderKL |
| Mage-Flow | Qwen3-VL | not specified | Mage-VAE |
| MiniMax H3 | Qwen3-VL | not specified | MiniMax H3 Video VAE + Audio VAE |
| MiniMax Music 3 | Qwen3 AR | 8B | DAV audio autoencoder + vocoder decoder |
| OmniGen | Integrated OmniGen encoder | not specified | AutoencoderKL |
| PixArt Sigma | T5 XXL v1.1 | 11B | AutoencoderKL |
| Qwen Image | Qwen2.5-VL | 7B | Qwen Image VAE |
| Sana | Gemma2 2B-IT | 2B | Sana AutoencoderDC |
| Sana Video | Gemma 2 | 2B | Wan VAE |
| SD 1.x/2.x (Legacy) | CLIP-L/14 | 123M | AutoencoderKL |
| Stable Diffusion 3 | CLIP-L/14 + CLIP-G/14 + T5 XXL v1.1 | 123M + 694M + 11B | AutoencoderKL |
| Stable Diffusion XL | CLIP-L/14 + CLIP-G/14 | 123M + 694M | AutoencoderKL |
| Stable Cascade (Stage C) | CLIP-ViT-bigG-14 | 694M | Stable Cascade Stage C VAE |
| Wan Video | UMT5 | not specified | Wan VAE |
| Wan S2V | UMT5 | not specified | Wan VAE |
| Z-Image | Qwen3 4B | 4B | AutoencoderKL |
| Z-Image Omni | Qwen3 4B | 4B | AutoencoderKL |
| ZLab I1 | T5Gemma 2B | 2B | AutoencoderKL |

</details>

*✓ = 支持，✓* = 支持但 full-rank training 通常需要 DeepSpeed/FSDP2，✗ = 不支持。Ref Inputs 标记现有 reference/edit/I2V conditioning paths；`opt` 表示可选，`req` 表示 edit/I2V flavour 必需。*
*TwinFlow 在 `twinflow_enabled=true` 时为原生支持；diffusion models 仍需要 `diff2flow_enabled=true` 和 `twinflow_allow_diff2flow=true`。Self-Flow 指 CREPA self-flow support。LayerSync 标记公开 hidden states 用于 alignment 的 backbones。*

### 快速通道：Z-Image Turbo 和 Flux Schnell

- **Z-Image Turbo**：完全支持带 TREAD 的 LoRA；即使不使用量化（int8 也可以），在 NVIDIA 和 macOS 上运行速度也很快。通常瓶颈只是训练器设置。
- **Flux Schnell**：快速开始配置会自动处理快速噪声调度和辅助 LoRA 堆栈；训练 Schnell LoRA 不需要额外的标志。

### 高级实验功能

- **Diff2Flow**：允许使用 Flow Matching 损失目标训练标准的 epsilon/v-prediction 模型（SD1.5、SDXL、DeepFloyd 等）。这弥合了旧架构和现代基于流的训练之间的差距。
- **Scheduled Sampling**：通过让模型在训练期间生成自己的中间噪声潜变量（"rollout"）来减少曝光偏差。这有助于模型学习从自身的生成错误中恢复。

## 常见问题

### 数据集样本少于预期

如果数据集的可用样本少于预期，文件可能在处理过程中被过滤掉了。常见原因包括：

- **文件太小**：低于 `minimum_image_size` 的图像会被过滤
- **纵横比超出范围**：超出 `minimum_aspect_ratio`/`maximum_aspect_ratio` 边界的图像会被排除
- **时长限制**：超出时长限制的音频/视频文件会被跳过

**查看过滤统计：**
- 在 WebUI 中，浏览到您的数据集目录并选择它以查看过滤统计
- 在数据集处理期间检查日志中的统计信息，如：`Sample processing statistics: {'total_processed': 100, 'skipped': {'too_small': 15, ...}}`

详细故障排除请参阅数据加载器文档中的[故障排除-过滤后的数据集](DATALOADER.zh.md)。
