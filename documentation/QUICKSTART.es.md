# Guía de inicio rápido

**Nota**: Para configuraciones más avanzadas, consulta el [tutorial](TUTORIAL.md) y la [referencia de opciones](OPTIONS.md).

## Guías de inicio rápido por modelo

| Modelo | Parámetros | Guía |
| --- | --- | --- |
| ACE-Step | 3.5B | [ACE_STEP.es.md](quickstart/ACE_STEP.es.md) |
| Anima | Not specified | Sin guía dedicada |
| Auraflow | 6B | [AURAFLOW.es.md](quickstart/AURAFLOW.es.md) |
| Boogu-Image | Not specified | [BOOGU_IMAGE.es.md](quickstart/BOOGU_IMAGE.es.md) |
| Chroma 1 | 8.9B | [CHROMA.es.md](quickstart/CHROMA.es.md) |
| Cosmos2 | 2B-14B | [COSMOS2IMAGE.es.md](quickstart/COSMOS2IMAGE.es.md) |
| Cosmos3 | 16B-65B | [COSMOS3.es.md](quickstart/COSMOS3.es.md) |
| DeepFloyd IF | 0.4B-4.3B stages | Sin guía dedicada |
| ERNIE-Image | Not specified | [ERNIE.es.md](quickstart/ERNIE.es.md) |
| Flux.1 | 8B-12B | [FLUX.es.md](quickstart/FLUX.es.md)<br>[FLUX_KONTEXT.es.md](quickstart/FLUX_KONTEXT.es.md) |
| Flux.2 | 4B-32B | [FLUX2.es.md](quickstart/FLUX2.es.md) |
| HeartMuLa | 3B | [HEARTMULA.es.md](quickstart/HEARTMULA.es.md) |
| HiDream | 17B (8.5B MoE) | [HIDREAM.es.md](quickstart/HIDREAM.es.md) |
| Hunyuan Video | 8.3B | [HUNYUANVIDEO.es.md](quickstart/HUNYUANVIDEO.es.md) |
| Ideogram 4 | 9B | [IDEOGRAM4.es.md](quickstart/IDEOGRAM4.es.md) |
| InfiniteTalk | 14B | [INFINITETALK.es.md](quickstart/INFINITETALK.es.md) |
| Kandinsky 5.0 Image | 6B (lite) | [KANDINSKY5_IMAGE.es.md](quickstart/KANDINSKY5_IMAGE.es.md) |
| Kandinsky 5.0 Video | 2B lite, 19B pro | [KANDINSKY5_VIDEO.es.md](quickstart/KANDINSKY5_VIDEO.es.md) |
| Kwai Kolors | 2.7B | [KOLORS.es.md](quickstart/KOLORS.es.md) |
| Krea2 | Not specified | [KREA2.es.md](quickstart/KREA2.es.md) |
| LongCat Image | 6B | [LONGCAT_IMAGE.es.md](quickstart/LONGCAT_IMAGE.es.md)<br>[LONGCAT_EDIT.es.md](quickstart/LONGCAT_EDIT.es.md) |
| LongCat Video | 13.6B | [LONGCAT_VIDEO.es.md](quickstart/LONGCAT_VIDEO.es.md)<br>[LONGCAT_VIDEO_EDIT.es.md](quickstart/LONGCAT_VIDEO_EDIT.es.md) |
| LTX Video | ~2.5B | [LTXVIDEO.es.md](quickstart/LTXVIDEO.es.md) |
| LTX Video 2 | 19B | [LTXVIDEO2.es.md](quickstart/LTXVIDEO2.es.md) |
| Lumina2 | 2B | [LUMINA2.es.md](quickstart/LUMINA2.es.md) |
| Mage-Flow | 4B | [MAGEFLOW.es.md](quickstart/MAGEFLOW.es.md) |
| MiniMax H3 | 33B | [MINIMAX_H3.es.md](/documentation/quickstart/MINIMAX_H3.es.md) |
| MiniMax Music 3 | 2.4B transformer + 8B AR | [MINIMAX_MUSIC.es.md](/documentation/quickstart/MINIMAX_MUSIC.es.md) |
| OmniGen | 3.8B | [OMNIGEN.es.md](quickstart/OMNIGEN.es.md) |
| PixArt Sigma | 0.6B-0.9B | [SIGMA.es.md](quickstart/SIGMA.es.md) |
| Qwen Image | 20B | [QWEN_IMAGE.es.md](quickstart/QWEN_IMAGE.es.md)<br>[QWEN_EDIT.es.md](quickstart/QWEN_EDIT.es.md) |
| Sana | 0.6B-4.8B | [SANA.es.md](quickstart/SANA.es.md) |
| Sana Video | 2B | [SANAVIDEO.es.md](quickstart/SANAVIDEO.es.md) |
| SD 1.x/2.x (Legacy) | 0.9B | Sin guía dedicada |
| Stable Diffusion 3 | 2B-8B | [SD3.es.md](quickstart/SD3.es.md) |
| Stable Diffusion XL | 3.5B | [SDXL.es.md](quickstart/SDXL.es.md) |
| Stable Cascade (Stage C) | 1B, 3.6B prior | [STABLE_CASCADE_C.es.md](quickstart/STABLE_CASCADE_C.es.md) |
| Wan Video | 1.3B-14B | [WAN.es.md](quickstart/WAN.es.md) |
| Wan S2V | 14B | [WAN_S2V.es.md](quickstart/WAN_S2V.es.md) |
| Z-Image | 6B | [ZIMAGE.es.md](quickstart/ZIMAGE.es.md) |
| Z-Image Omni | 6B | [ZIMAGE.es.md](quickstart/ZIMAGE.es.md) |
| ZLab I1 | 3B | [ZLAB_i1.es.md](quickstart/ZLAB_i1.es.md) |

## Compatibilidad de funciones

La matriz completa de compatibilidad se divide por área de función para que cada tabla sea legible.

<details>
<summary>Soporte de entrenamiento</summary>

| Modelo | PEFT LoRA | LyCORIS | Rango completo | ControlNet | Ref Inputs |
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
<summary>Compatibilidad de precisión</summary>

| Modelo | Cuantización | Precisión mixta |
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
<summary>Granularidad de checkpointing</summary>

| Modelo | Checkpointing de gradiente | Intervalo | Segment stride | Offload de atención |
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
<summary>Flujo, destilación y alineación</summary>

El post-entrenamiento MixFlow está disponible para todas las filas `flow matching`. Consulta la [guía MixFlow](experimental/MIXFLOW.md).

| Modelo | Predicción | Flow Shift | TwinFlow | Self-Flow | LayerSync | Internal Guidance | Sliders |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: |
| ACE-Step | flow matching | ✓ | ✓ | ✗ | ✓ | ✓ v1 | ✓ |
| Anima | flow matching | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Auraflow | flow matching | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Boogu-Image | flow matching | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Chroma 1 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Cosmos2 | sample | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Cosmos3 | flow matching | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| DeepFloyd IF | epsilon | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| ERNIE-Image | flow matching | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Flux.1 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Flux.2 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HeartMuLa | autoregressive next-token | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| HiDream | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Hunyuan Video | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ideogram 4 | flow matching | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| InfiniteTalk | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Kandinsky 5.0 Image | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Kandinsky 5.0 Video | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Kwai Kolors | epsilon | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Krea2 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| LongCat Image | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| LongCat Video | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| LTX Video | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| LTX Video 2 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Lumina2 | flow matching | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Mage-Flow | flow matching | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| MiniMax H3 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| MiniMax Music 3 | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| OmniGen | flow matching | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| PixArt Sigma | epsilon | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Qwen Image | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Sana | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Sana Video | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SD 1.x/2.x (Legacy) | epsilon / v-pred | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Stable Diffusion 3 | flow matching | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Stable Diffusion XL | epsilon | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Stable Cascade (Stage C) | epsilon | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Wan Video | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Wan S2V | flow matching | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Z-Image | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Z-Image Omni | flow matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ZLab I1 | flow matching | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |

</details>

<details>
<summary>Encoders de texto y tipos de VAE</summary>

| Modelo | Encoders de texto | Parámetros de encoder | VAE |
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
| InfiniteTalk | UMT5 + Wav2Vec2 | no especificado + 95M | Wan VAE |
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

*✓ = soportado, ✓* = soportado pero normalmente requiere DeepSpeed/FSDP2 para entrenamiento full-rank, ✗ = no soportado. Ref Inputs marca rutas existentes de condicionamiento por referencia/edición/I2V; `opt` significa opcional y `req` significa requerido por el flavour de edición/I2V.*
*TwinFlow es nativo cuando `twinflow_enabled=true`; los modelos de difusión aún requieren `diff2flow_enabled=true` y `twinflow_allow_diff2flow=true`. Self-Flow se refiere al soporte CREPA self-flow. Internal Guidance usa una cabeza auxiliar en diffusion transformers. LayerSync marca backbones que exponen estados ocultos para alineación.*

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

Para solución de problemas detallada, consulta [Solución de problemas de datasets filtrados](DATALOADER.es.md) en la documentación del dataloader.
