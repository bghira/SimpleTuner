# クイックスタートガイド

**注意**: より高度な設定については、[チュートリアル](TUTORIAL.md)および[オプションリファレンス](OPTIONS.md)を参照してください。

## モデル別クイックスタートガイド

| モデル | パラメータ | ガイド |
| --- | --- | --- |
| ACE-Step | 3.5B | [ACE_STEP.ja.md](quickstart/ACE_STEP.ja.md) |
| Anima | Not specified | 専用ガイドなし |
| Auraflow | 6B | [AURAFLOW.ja.md](quickstart/AURAFLOW.ja.md) |
| Boogu-Image | Not specified | [BOOGU_IMAGE.ja.md](quickstart/BOOGU_IMAGE.ja.md) |
| Chroma 1 | 8.9B | [CHROMA.ja.md](quickstart/CHROMA.ja.md) |
| Cosmos2 | 2B-14B | [COSMOS2IMAGE.ja.md](quickstart/COSMOS2IMAGE.ja.md) |
| Cosmos3 | 16B-65B | [COSMOS3.ja.md](quickstart/COSMOS3.ja.md) |
| DeepFloyd IF | 0.4B-4.3B stages | 専用ガイドなし |
| ERNIE-Image | Not specified | [ERNIE.ja.md](quickstart/ERNIE.ja.md) |
| Flux.1 | 8B-12B | [FLUX.ja.md](quickstart/FLUX.ja.md)<br>[FLUX_KONTEXT.ja.md](quickstart/FLUX_KONTEXT.ja.md) |
| Flux.2 | 4B-32B | [FLUX2.ja.md](quickstart/FLUX2.ja.md) |
| HeartMuLa | 3B | [HEARTMULA.ja.md](quickstart/HEARTMULA.ja.md) |
| HiDream | 17B (8.5B MoE) | [HIDREAM.ja.md](quickstart/HIDREAM.ja.md) |
| Hunyuan Video | 8.3B | [HUNYUANVIDEO.ja.md](quickstart/HUNYUANVIDEO.ja.md) |
| Ideogram 4 | 9B | [IDEOGRAM4.ja.md](quickstart/IDEOGRAM4.ja.md) |
| Kandinsky 5.0 Image | 6B (lite) | [KANDINSKY5_IMAGE.ja.md](quickstart/KANDINSKY5_IMAGE.ja.md) |
| Kandinsky 5.0 Video | 2B lite, 19B pro | [KANDINSKY5_VIDEO.ja.md](quickstart/KANDINSKY5_VIDEO.ja.md) |
| Kwai Kolors | 2.7B | [KOLORS.ja.md](quickstart/KOLORS.ja.md) |
| Krea2 | Not specified | [KREA2.ja.md](quickstart/KREA2.ja.md) |
| LongCat Image | 6B | [LONGCAT_IMAGE.ja.md](quickstart/LONGCAT_IMAGE.ja.md)<br>[LONGCAT_EDIT.ja.md](quickstart/LONGCAT_EDIT.ja.md) |
| LongCat Video | 13.6B | [LONGCAT_VIDEO.ja.md](quickstart/LONGCAT_VIDEO.ja.md)<br>[LONGCAT_VIDEO_EDIT.ja.md](quickstart/LONGCAT_VIDEO_EDIT.ja.md) |
| LTX Video | ~2.5B | [LTXVIDEO.ja.md](quickstart/LTXVIDEO.ja.md) |
| LTX Video 2 | 19B | [LTXVIDEO2.ja.md](quickstart/LTXVIDEO2.ja.md) |
| Lumina2 | 2B | [LUMINA2.ja.md](quickstart/LUMINA2.ja.md) |
| Mage-Flow | 4B | [MAGEFLOW.ja.md](quickstart/MAGEFLOW.ja.md) |
| MiniMax H3 | 33B | [MINIMAX_H3.ja.md](/documentation/quickstart/MINIMAX_H3.ja.md) |
| OmniGen | 3.8B | [OMNIGEN.ja.md](quickstart/OMNIGEN.ja.md) |
| PixArt Sigma | 0.6B-0.9B | [SIGMA.ja.md](quickstart/SIGMA.ja.md) |
| Qwen Image | 20B | [QWEN_IMAGE.ja.md](quickstart/QWEN_IMAGE.ja.md)<br>[QWEN_EDIT.ja.md](quickstart/QWEN_EDIT.ja.md) |
| Sana | 0.6B-4.8B | [SANA.ja.md](quickstart/SANA.ja.md) |
| Sana Video | 2B | [SANAVIDEO.ja.md](quickstart/SANAVIDEO.ja.md) |
| SD 1.x/2.x (Legacy) | 0.9B | 専用ガイドなし |
| Stable Diffusion 3 | 2B-8B | [SD3.ja.md](quickstart/SD3.ja.md) |
| Stable Diffusion XL | 3.5B | [SDXL.ja.md](quickstart/SDXL.ja.md) |
| Stable Cascade (Stage C) | 1B, 3.6B prior | [STABLE_CASCADE_C.ja.md](quickstart/STABLE_CASCADE_C.ja.md) |
| Wan Video | 1.3B-14B | [WAN.ja.md](quickstart/WAN.ja.md) |
| Wan S2V | 14B | [WAN_S2V.ja.md](quickstart/WAN_S2V.ja.md) |
| Z-Image | 6B | [ZIMAGE.ja.md](quickstart/ZIMAGE.ja.md) |
| Z-Image Omni | 6B | [ZIMAGE.ja.md](quickstart/ZIMAGE.ja.md) |
| ZLab I1 | 3B | [ZLAB_i1.ja.md](quickstart/ZLAB_i1.ja.md) |

## 機能互換性

完全な互換性マトリクスは、読みやすいように機能領域ごとに分割しています。

<details>
<summary>トレーニングサポート</summary>

| モデル | PEFT LoRA | LyCORIS | フルランク | ControlNet | Ref Inputs |
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
<summary>精度レベルのサポート</summary>

| モデル | 量子化 | 混合精度 |
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
<summary>チェックポイント粒度</summary>

| モデル | Gradient Checkpoint | Interval | Segment Stride | Attention Offload |
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
<summary>Flow・蒸留・アラインメント</summary>

| モデル | Prediction | Flow Shift | TwinFlow | Self-Flow | LayerSync | Sliders |
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
<summary>テキストエンコーダーと VAE タイプ</summary>

| モデル | Text Encoders | Text Encoder Params | VAE |
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
| LTX Video 2 | Gemma3 / Gemma4 | not specified | LTX Video 2 VAE |
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

*✓ = サポート、✓* = サポートされていますが、full-rank training では通常 DeepSpeed/FSDP2 が必要、✗ = 未サポート。Ref Inputs は既存の reference/edit/I2V conditioning path を示します。`opt` は任意、`req` は edit/I2V flavour で必須です。*
*TwinFlow は `twinflow_enabled=true` のとき native support です。diffusion models では `diff2flow_enabled=true` と `twinflow_allow_diff2flow=true` も必要です。Self-Flow は CREPA self-flow support です。LayerSync は alignment 用 hidden states を公開する backbone を示します。*

### 高速パス: Z-Image TurboとFlux Schnell

- **Z-Image Turbo**: TREADを使用した完全サポートのLoRA。量子化なし（int8も動作）でもNVIDIAとmacOSで高速に動作します。多くの場合、ボトルネックはトレーナーのセットアップだけです。
- **Flux Schnell**: クイックスタート設定が高速ノイズスケジュールとアシスタントLoRAスタックを自動的に処理します。Schnell LoRAをトレーニングするための追加フラグは不要です。

### 高度な実験的機能

- **Diff2Flow**: 標準的なepsilon/v-predictionモデル（SD1.5、SDXL、DeepFloydなど）をFlow Matching損失目的関数を使用してトレーニングできます。これにより、古いアーキテクチャと最新のフローベースのトレーニング間のギャップを埋めます。
- **Scheduled Sampling**: トレーニング中にモデル自身に中間ノイズ潜在変数を生成させる（「ロールアウト」）ことで露出バイアスを軽減します。これにより、モデルが自身の生成エラーから回復する方法を学習できます。

## よくある問題

### データセットのサンプル数が予想より少ない

データセットの使用可能なサンプル数が予想より少ない場合、処理中にファイルがフィルタされた可能性があります。一般的な理由：

- **ファイルが小さすぎる**: `minimum_image_size` 未満の画像はフィルタされます
- **アスペクト比が範囲外**: `minimum_aspect_ratio`/`maximum_aspect_ratio` の範囲外の画像は除外されます
- **時間制限**: 時間制限を超えるオーディオ/ビデオファイルはスキップされます

**フィルタリング統計の確認:**
- WebUI でデータセットディレクトリを参照し、選択するとフィルタリング統計が表示されます
- データセット処理中のログで次のような統計を確認: `Sample processing statistics: {'total_processed': 100, 'skipped': {'too_small': 15, ...}}`

詳細なトラブルシューティングについては、データローダードキュメントの[フィルタされたデータセットのトラブルシューティング](DATALOADER.ja.md)を参照してください。
