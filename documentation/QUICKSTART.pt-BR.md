# Guia de Início Rápido

**Nota**: Para configurações mais avançadas, veja o [tutorial](TUTORIAL.md) e a [referência de opções](OPTIONS.md).

## Guias de início rápido por modelo

| Modelo | Parâmetros | Guia |
| --- | --- | --- |
| ACE-Step | 3.5B | [ACE_STEP.pt-BR.md](quickstart/ACE_STEP.pt-BR.md) |
| Anima | Not specified | Sem guia dedicada |
| Auraflow | 6B | [AURAFLOW.pt-BR.md](quickstart/AURAFLOW.pt-BR.md) |
| Boogu-Image | Not specified | [BOOGU_IMAGE.pt-BR.md](quickstart/BOOGU_IMAGE.pt-BR.md) |
| Chroma 1 | 8.9B | [CHROMA.pt-BR.md](quickstart/CHROMA.pt-BR.md) |
| Cosmos2 | 2B-14B | [COSMOS2IMAGE.pt-BR.md](quickstart/COSMOS2IMAGE.pt-BR.md) |
| Cosmos3 | 16B-65B | [COSMOS3.pt-BR.md](quickstart/COSMOS3.pt-BR.md) |
| DeepFloyd IF | 0.4B-4.3B stages | Sem guia dedicada |
| ERNIE-Image | Not specified | [ERNIE.pt-BR.md](quickstart/ERNIE.pt-BR.md) |
| Flux.1 | 8B-12B | [FLUX.pt-BR.md](quickstart/FLUX.pt-BR.md)<br>[FLUX_KONTEXT.pt-BR.md](quickstart/FLUX_KONTEXT.pt-BR.md) |
| Flux.2 | 4B-32B | [FLUX2.pt-BR.md](quickstart/FLUX2.pt-BR.md) |
| HeartMuLa | 3B | [HEARTMULA.pt-BR.md](quickstart/HEARTMULA.pt-BR.md) |
| HiDream | 17B (8.5B MoE) | [HIDREAM.pt-BR.md](quickstart/HIDREAM.pt-BR.md) |
| Hunyuan Video | 8.3B | [HUNYUANVIDEO.pt-BR.md](quickstart/HUNYUANVIDEO.pt-BR.md) |
| Ideogram 4 | 9B | [IDEOGRAM4.pt-BR.md](quickstart/IDEOGRAM4.pt-BR.md) |
| Kandinsky 5.0 Image | 6B (lite) | [KANDINSKY5_IMAGE.pt-BR.md](quickstart/KANDINSKY5_IMAGE.pt-BR.md) |
| Kandinsky 5.0 Video | 2B lite, 19B pro | [KANDINSKY5_VIDEO.pt-BR.md](quickstart/KANDINSKY5_VIDEO.pt-BR.md) |
| Kwai Kolors | 2.7B | [KOLORS.pt-BR.md](quickstart/KOLORS.pt-BR.md) |
| Krea2 | Not specified | [KREA2.pt-BR.md](quickstart/KREA2.pt-BR.md) |
| LongCat Image | 6B | [LONGCAT_IMAGE.pt-BR.md](quickstart/LONGCAT_IMAGE.pt-BR.md)<br>[LONGCAT_EDIT.pt-BR.md](quickstart/LONGCAT_EDIT.pt-BR.md) |
| LongCat Video | 13.6B | [LONGCAT_VIDEO.pt-BR.md](quickstart/LONGCAT_VIDEO.pt-BR.md)<br>[LONGCAT_VIDEO_EDIT.pt-BR.md](quickstart/LONGCAT_VIDEO_EDIT.pt-BR.md) |
| LTX Video | ~2.5B | [LTXVIDEO.pt-BR.md](quickstart/LTXVIDEO.pt-BR.md) |
| LTX Video 2 | 19B | [LTXVIDEO2.pt-BR.md](quickstart/LTXVIDEO2.pt-BR.md) |
| Lumina2 | 2B | [LUMINA2.pt-BR.md](quickstart/LUMINA2.pt-BR.md) |
| Mage-Flow | 4B | [MAGEFLOW.pt-BR.md](quickstart/MAGEFLOW.pt-BR.md) |
| MiniMax H3 | 33B | [MINIMAX_H3.pt-BR.md](/documentation/quickstart/MINIMAX_H3.pt-BR.md) |
| OmniGen | 3.8B | [OMNIGEN.pt-BR.md](quickstart/OMNIGEN.pt-BR.md) |
| PixArt Sigma | 0.6B-0.9B | [SIGMA.pt-BR.md](quickstart/SIGMA.pt-BR.md) |
| Qwen Image | 20B | [QWEN_IMAGE.pt-BR.md](quickstart/QWEN_IMAGE.pt-BR.md)<br>[QWEN_EDIT.pt-BR.md](quickstart/QWEN_EDIT.pt-BR.md) |
| Sana | 0.6B-4.8B | [SANA.pt-BR.md](quickstart/SANA.pt-BR.md) |
| Sana Video | 2B | [SANAVIDEO.pt-BR.md](quickstart/SANAVIDEO.pt-BR.md) |
| SD 1.x/2.x (Legacy) | 0.9B | Sem guia dedicada |
| Stable Diffusion 3 | 2B-8B | [SD3.pt-BR.md](quickstart/SD3.pt-BR.md) |
| Stable Diffusion XL | 3.5B | [SDXL.pt-BR.md](quickstart/SDXL.pt-BR.md) |
| Stable Cascade (Stage C) | 1B, 3.6B prior | [STABLE_CASCADE_C.pt-BR.md](quickstart/STABLE_CASCADE_C.pt-BR.md) |
| Wan Video | 1.3B-14B | [WAN.pt-BR.md](quickstart/WAN.pt-BR.md) |
| Wan S2V | 14B | [WAN_S2V.pt-BR.md](quickstart/WAN_S2V.pt-BR.md) |
| Z-Image | 6B | [ZIMAGE.pt-BR.md](quickstart/ZIMAGE.pt-BR.md) |
| Z-Image Omni | 6B | [ZIMAGE.pt-BR.md](quickstart/ZIMAGE.pt-BR.md) |
| ZLab I1 | 3B | [ZLAB_i1.pt-BR.md](quickstart/ZLAB_i1.pt-BR.md) |

## Compatibilidade de recursos

A matriz completa de compatibilidade é dividida por área de recurso para manter cada tabela legível.

<details>
<summary>Suporte de treinamento</summary>

| Modelo | PEFT LoRA | LyCORIS | Full-Rank | ControlNet | Ref Inputs |
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
<summary>Suporte a níveis de precisão</summary>

| Modelo | Quantização | Precisão mista |
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
<summary>Granularidade de checkpointing</summary>

| Modelo | Gradient Checkpoint | Interval | Segment Stride | Attention Offload |
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
<summary>Flow, destilação e alinhamento</summary>

| Modelo | Prediction | Flow Shift | TwinFlow | Self-Flow | LayerSync | Sliders |
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
<summary>Text encoders e tipos de VAE</summary>

| Modelo | Text Encoders | Text Encoder Params | VAE |
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
| Kandinsky 5.0 Image | Qwen2.5-VL + CLIP-L/14 | 7B + 123M | Flux VAE (AutoencoderKL) |
| Kandinsky 5.0 Video | Qwen2.5-VL + CLIP-L/14 | 7B + 123M | Hunyuan Video VAE |
| Kwai Kolors | ChatGLM-6B | 6B | AutoencoderKL |
| Krea2 | Qwen3VL | not specified | Qwen Image VAE |
| LongCat Image | Qwen2.5-VL | 7B | AutoencoderKL |
| LongCat Video | Qwen2.5-VL | 7B | Wan VAE |
| LTX Video | T5 XXL v1.1 | 11B | LTX Video VAE |
| LTX Video 2 | Gemma3 | not specified | LTX Video 2 VAE |
| Lumina2 | Gemma2 | 2B | AutoencoderKL |
| Mage-Flow | Qwen3-VL | not specified | Mage-VAE |
| MiniMax H3 | Qwen3-VL | not specified | MiniMax H3 Video VAE + Audio VAE |
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

*✓ = suportado, ✓* = suportado mas normalmente requer DeepSpeed/FSDP2 para treinamento full-rank, ✗ = não suportado. Ref Inputs marca rotas existentes de condicionamento por referência/edição/I2V; `opt` significa opcional e `req` significa obrigatório para o flavour de edição/I2V.*
*TwinFlow é nativo quando `twinflow_enabled=true`; modelos de difusão ainda exigem `diff2flow_enabled=true` e `twinflow_allow_diff2flow=true`. Self-Flow se refere ao suporte CREPA self-flow. LayerSync marca backbones que expõem hidden states para alinhamento.*

### Caminhos rápidos: Z-Image Turbo e Flux Schnell

- **Z-Image Turbo**: LoRA totalmente suportado com TREAD; roda rápido em NVIDIA e macOS mesmo sem quantização (int8 também funciona). Muitas vezes o gargalo é apenas a configuração do trainer.
- **Flux Schnell**: A configuração do quickstart lida automaticamente com o agendamento rápido de ruído e o stack de LoRA assistente; não são necessários flags extras para treinar LoRAs Schnell.

### Recursos experimentais avançados

- **Diff2Flow**: Permite treinar modelos padrão de epsilon/v-prediction (SD1.5, SDXL, DeepFloyd etc.) usando uma loss de Flow Matching. Isso reduz a lacuna entre arquiteturas antigas e treinamento moderno baseado em fluxo.
- **Scheduled Sampling**: Reduz o viés de exposição ao permitir que o modelo gere seus próprios latentes ruidosos intermediários durante o treinamento ("rollout"). Isso ajuda o modelo a aprender a se recuperar de seus próprios erros de geração.

## Problemas Comuns

### Dataset tem menos amostras do que esperado

Se seu dataset acaba com menos amostras utilizáveis do que você esperava, arquivos podem ter sido filtrados durante o processamento. Razões comuns incluem:

- **Arquivos muito pequenos**: Imagens abaixo de `minimum_image_size` são filtradas
- **Proporção fora do intervalo**: Imagens fora dos limites de `minimum_aspect_ratio`/`maximum_aspect_ratio` são excluídas
- **Limites de duração**: Arquivos de áudio/vídeo que excedem limites de duração são ignorados

**Visualizando estatísticas de filtragem:**
- Na WebUI, navegue até o diretório do seu dataset e selecione-o para ver estatísticas de filtragem
- Verifique os logs durante o processamento do dataset por estatísticas como: `Sample processing statistics: {'total_processed': 100, 'skipped': {'too_small': 15, ...}}`

Para solução de problemas detalhada, consulte [Solucionando problemas de datasets filtrados](DATALOADER.pt-BR.md) na documentação do dataloader.
