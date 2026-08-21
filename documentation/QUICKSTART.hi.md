# क्विकस्टार्ट गाइड

**नोट**: अधिक उन्नत कॉन्फ़िगरेशनों के लिए, [ट्यूटोरियल](TUTORIAL.md) और [options reference](OPTIONS.md) देखें।

## मॉडल क्विकस्टार्ट गाइड

| मॉडल | पैरामीटर | गाइड |
| --- | --- | --- |
| ACE-Step | 3.5B | [ACE_STEP.hi.md](quickstart/ACE_STEP.hi.md) |
| Anima | Not specified | Dedicated guide नहीं है |
| Auraflow | 6B | [AURAFLOW.hi.md](quickstart/AURAFLOW.hi.md) |
| Boogu-Image | Not specified | [BOOGU_IMAGE.hi.md](quickstart/BOOGU_IMAGE.hi.md) |
| Chroma 1 | 8.9B | [CHROMA.hi.md](quickstart/CHROMA.hi.md) |
| Cosmos2 | 2B-14B | [COSMOS2IMAGE.hi.md](quickstart/COSMOS2IMAGE.hi.md) |
| Cosmos3 | 16B-65B | [COSMOS3.hi.md](quickstart/COSMOS3.hi.md) |
| DeepFloyd IF | 0.4B-4.3B stages | Dedicated guide नहीं है |
| ERNIE-Image | Not specified | [ERNIE.hi.md](quickstart/ERNIE.hi.md) |
| Flux.1 | 8B-12B | [FLUX.hi.md](quickstart/FLUX.hi.md)<br>[FLUX_KONTEXT.hi.md](quickstart/FLUX_KONTEXT.hi.md) |
| Flux.2 | 4B-32B | [FLUX2.hi.md](quickstart/FLUX2.hi.md) |
| HeartMuLa | 3B | [HEARTMULA.hi.md](quickstart/HEARTMULA.hi.md) |
| HiDream | 17B (8.5B MoE) | [HIDREAM.hi.md](quickstart/HIDREAM.hi.md) |
| Hunyuan Video | 8.3B | [HUNYUANVIDEO.hi.md](quickstart/HUNYUANVIDEO.hi.md) |
| Ideogram 4 | 9B | [IDEOGRAM4.hi.md](quickstart/IDEOGRAM4.hi.md) |
| Kandinsky 5.0 Image | 6B (lite) | [KANDINSKY5_IMAGE.hi.md](quickstart/KANDINSKY5_IMAGE.hi.md) |
| Kandinsky 5.0 Video | 2B lite, 19B pro | [KANDINSKY5_VIDEO.hi.md](quickstart/KANDINSKY5_VIDEO.hi.md) |
| Kwai Kolors | 2.7B | [KOLORS.hi.md](quickstart/KOLORS.hi.md) |
| Krea2 | Not specified | [KREA2.hi.md](quickstart/KREA2.hi.md) |
| LongCat Image | 6B | [LONGCAT_IMAGE.hi.md](quickstart/LONGCAT_IMAGE.hi.md)<br>[LONGCAT_EDIT.hi.md](quickstart/LONGCAT_EDIT.hi.md) |
| LongCat Video | 13.6B | [LONGCAT_VIDEO.hi.md](quickstart/LONGCAT_VIDEO.hi.md)<br>[LONGCAT_VIDEO_EDIT.hi.md](quickstart/LONGCAT_VIDEO_EDIT.hi.md) |
| LTX Video | ~2.5B | [LTXVIDEO.hi.md](quickstart/LTXVIDEO.hi.md) |
| LTX Video 2 | 19B | [LTXVIDEO2.hi.md](quickstart/LTXVIDEO2.hi.md) |
| Lumina2 | 2B | [LUMINA2.hi.md](quickstart/LUMINA2.hi.md) |
| Mage-Flow | 4B | [MAGEFLOW.hi.md](quickstart/MAGEFLOW.hi.md) |
| MiniMax H3 | 33B | [MINIMAX_H3.hi.md](/documentation/quickstart/MINIMAX_H3.hi.md) |
| MiniMax Music 3 | 2.4B transformer + 8B AR | [MINIMAX_MUSIC.hi.md](/documentation/quickstart/MINIMAX_MUSIC.hi.md) |
| OmniGen | 3.8B | [OMNIGEN.hi.md](quickstart/OMNIGEN.hi.md) |
| PixArt Sigma | 0.6B-0.9B | [SIGMA.hi.md](quickstart/SIGMA.hi.md) |
| Qwen Image | 20B | [QWEN_IMAGE.hi.md](quickstart/QWEN_IMAGE.hi.md)<br>[QWEN_EDIT.hi.md](quickstart/QWEN_EDIT.hi.md) |
| Sana | 0.6B-4.8B | [SANA.hi.md](quickstart/SANA.hi.md) |
| Sana Video | 2B | [SANAVIDEO.hi.md](quickstart/SANAVIDEO.hi.md) |
| SD 1.x/2.x (Legacy) | 0.9B | Dedicated guide नहीं है |
| Stable Diffusion 3 | 2B-8B | [SD3.hi.md](quickstart/SD3.hi.md) |
| Stable Diffusion XL | 3.5B | [SDXL.hi.md](quickstart/SDXL.hi.md) |
| Stable Cascade (Stage C) | 1B, 3.6B prior | [STABLE_CASCADE_C.hi.md](quickstart/STABLE_CASCADE_C.hi.md) |
| Wan Video | 1.3B-14B | [WAN.hi.md](quickstart/WAN.hi.md) |
| Wan S2V | 14B | [WAN_S2V.hi.md](quickstart/WAN_S2V.hi.md) |
| Z-Image | 6B | [ZIMAGE.hi.md](quickstart/ZIMAGE.hi.md) |
| Z-Image Omni | 6B | [ZIMAGE.hi.md](quickstart/ZIMAGE.hi.md) |
| ZLab I1 | 3B | [ZLAB_i1.hi.md](quickstart/ZLAB_i1.hi.md) |

## फीचर संगतता

पूरी compatibility matrix को feature area के अनुसार बांटा गया है ताकि हर table पढ़ने योग्य रहे।

<details>
<summary>Training support</summary>

| मॉडल | PEFT LoRA | LyCORIS | Full-Rank | ControlNet | Ref Inputs |
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
<summary>Precision level support</summary>

| मॉडल | Quantization | Mixed Precision |
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
<summary>Checkpointing granularity</summary>

| मॉडल | Gradient Checkpoint | Interval | Segment Stride | Attention Offload |
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
<summary>Flow, distillation, and alignment</summary>

| मॉडल | Prediction | Flow Shift | TwinFlow | Self-Flow | LayerSync | Internal Guidance | Sliders |
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
<summary>Text encoders and VAE types</summary>

| मॉडल | Text Encoders | Text Encoder Params | VAE |
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

*✓ = supported, ✓* = supported लेकिन full-rank training के लिए आम तौर पर DeepSpeed/FSDP2 चाहिए, ✗ = not supported. Ref Inputs मौजूदा reference/edit/I2V conditioning paths को दिखाता है; `opt` optional और `req` edit/I2V flavour में required है.*
*TwinFlow native है जब `twinflow_enabled=true`; diffusion models को अभी भी `diff2flow_enabled=true` और `twinflow_allow_diff2flow=true` चाहिए। Self-Flow CREPA self-flow support है। Internal Guidance diffusion transformers पर auxiliary head उपयोग करता है। LayerSync alignment के लिए hidden states expose करने वाले backbones को दिखाता है.*

### तेज़ रास्ते: Z-Image Turbo और Flux Schnell

- **Z-Image Turbo**: TREAD के साथ पूरी तरह समर्थित LoRA; NVIDIA और macOS पर quant के बिना भी तेज़ चलता है (int8 भी काम करता है)। अक्सर bottleneck केवल trainer setup होता है।
- **Flux Schnell**: क्विकस्टार्ट कॉन्फ़िग fast noise schedule और assistant LoRA stack को स्वतः संभालता है; Schnell LoRAs ट्रेन करने के लिए अतिरिक्त फ़्लैग्स की आवश्यकता नहीं है।

### उन्नत प्रायोगिक विशेषताएँ

- **Diff2Flow**: Flow Matching loss objective के साथ standard epsilon/v‑prediction मॉडल्स (SD1.5, SDXL, DeepFloyd, आदि) को ट्रेन करने की अनुमति देता है। यह पुराने आर्किटेक्चर और आधुनिक flow‑based प्रशिक्षण के बीच का अंतर भरता है।
- **Scheduled Sampling**: प्रशिक्षण के दौरान मॉडल को अपने ही intermediate noisy latents उत्पन्न करने देता है ("rollout"), जिससे exposure bias कम होता है। यह मॉडल को अपनी ही generation errors से उबरना सिखाता है।

## सामान्य समस्याएं

### Dataset में expected से कम samples हैं

यदि आपके dataset में expected से कम usable samples हैं, तो processing के दौरान files filter हो गई हो सकती हैं। सामान्य कारण:

- **Files बहुत छोटी हैं**: `minimum_image_size` से नीचे की images filter कर दी जाती हैं
- **Aspect ratio range से बाहर**: `minimum_aspect_ratio`/`maximum_aspect_ratio` bounds से बाहर की images exclude कर दी जाती हैं
- **Duration limits**: Duration limits से अधिक audio/video files skip कर दी जाती हैं

**Filtering statistics देखना:**
- WebUI में, अपने dataset directory पर browse करें और filtering statistics देखने के लिए इसे select करें
- Dataset processing के दौरान logs में इस तरह के statistics check करें: `Sample processing statistics: {'total_processed': 100, 'skipped': {'too_small': 15, ...}}`

विस्तृत troubleshooting के लिए, dataloader documentation में [Filtered datasets का Troubleshooting](DATALOADER.hi.md) देखें।
