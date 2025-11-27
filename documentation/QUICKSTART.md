# Quickstart Guide

**Note**: For more advanced configurations, see the [tutorial](/documentation/TUTORIAL.md) and [options reference](/documentation/OPTIONS.md).

## Feature Compatibility

For the complete and most accurate feature matrix, refer to the [main README](../README.md#model-architecture-support).

## Model Quickstart Guides

| Model | Params | PEFT LoRA | Lycoris | Full-Rank | Quantization | Mixed Precision | Grad Checkpoint | Flow Shift | ControlNet | Guide |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | ✓ | int8 optional | bf16 | ✓ | ✗ | ✓ | [SIGMA.md](/documentation/quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | ✓ | int8 optional | bf16 | ✓+ | ✓ | ✗ | [SANA.md](/documentation/quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | ✓ | not recommended | bf16 | ✓ | ✗ | ✗ | [KOLORS.md](/documentation/quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | ✓ | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ (SLG) | ✓ | [SD3.md](/documentation/quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | [FLUX.md](/documentation/quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✗ | [FLUX2.md](/documentation/quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | [FLUX_KONTEXT.md](/documentation/quickstart/FLUX_KONTEXT.md) |
| ACE-Step | 3.5B | ✓ | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✗ | [ACE_STEP.md](/documentation/quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✗ | [CHROMA.md](/documentation/quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ (SLG) | ✓ | [AURAFLOW.md](/documentation/quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓ | ✓ | ✓ | [HIDREAM.md](/documentation/quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | ✓ | int8/fp8 optional | bf16 | ✓ | ✓ | ✗ | [OMNIGEN.md](/documentation/quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | ✓ | not recommended | bf16 | ✓ | ✗ | ✓ | [SDXL.md](/documentation/quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | ✓ | int8 optional | bf16 | ✓ | ✓ | ✗ | [LUMINA2.md](/documentation/quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | ✓ | not recommended | bf16 | ✓ | ✓ | ✗ | [COSMOS2IMAGE.md](/documentation/quickstart/COSMOS2IMAGE.md) |
| LTX Video | ~2.5B | ✓ | ✓ | ✓ | int8/fp8 optional | bf16 | ✓ | ✓ | ✗ | [LTXVIDEO.md](/documentation/quickstart/LTXVIDEO.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✗ | [HUNYUANVIDEO.md](/documentation/quickstart/HUNYUANVIDEO.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✗ | [WAN.md](/documentation/quickstart/WAN.md) |
| Qwen Image | 20B | ✓ | ✓ | ✓* | **required** (int8/nf4) | bf16 | ✓ | ✓ | ✗ | [QWEN_IMAGE.md](/documentation/quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓ | ✓* | **required** (int8/nf4) | bf16 | ✓ | ✓ | ✗ | [QWEN_EDIT.md](/documentation/quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, 3.6B prior | ✓ | ✓ | ✓* | not supported | fp32 (required) | ✓ | ✗ | ✗ | [STABLE_CASCADE_C.md](/documentation/quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✗ | [KANDINSKY5_IMAGE.md](/documentation/quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✗ | [KANDINSKY5_VIDEO.md](/documentation/quickstart/KANDINSKY5_VIDEO.md) |

*✓ = supported, ✓* = requires DeepSpeed/FSDP2 for full-rank, ✗ = not supported, `✓+` indicates checkpointing is recommended due to VRAM pressure.*

> ℹ️ Wan quickstart includes 2.1 + 2.2 stage presets and the time-embedding toggle. Flux Kontext covers editing workflows built atop Flux.1.

> ⚠️ These quickstarts are living documents. Expect occasional updates as new models land or training recipes improve.
