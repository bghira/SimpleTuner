# Quickstart Guide

**Note**: For more advanced configurations, see the [tutorial](TUTORIAL.md) and [options reference](OPTIONS.md).

## Feature Compatibility

For the complete and most accurate feature matrix, refer to the [main README](https://github.com/bghira/SimpleTuner#model-architecture-support).

## Model Quickstart Guides

| Model | Params | PEFT LoRA | Lycoris | Full-Rank | Quantization | Mixed Precision | Grad Checkpoint | Flow Shift | TwinFlow | LayerSync | ControlNet | Sliders† | Guide |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | ✓ | int8 optional | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | [SIGMA.md](quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | ✓ | int8 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [SANA.md](quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | ✓ | not recommended | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | [KOLORS.md](quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | ✓ | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | [SD3.md](quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ | [FLUX.md](quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [FLUX2.md](quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ | [FLUX_KONTEXT.md](quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✗ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [ZIMAGE.md](quickstart/ZIMAGE.md) |
| ACE-Step | 3.5B | ✓ | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [ACE_STEP.md](quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [CHROMA.md](quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | [AURAFLOW.md](quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [HIDREAM.md](quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | ✓ | int8/fp8 optional | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | [OMNIGEN.md](quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | ✓ | not recommended | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | [SDXL.md](quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | ✓ | int8 optional | bf16 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [LUMINA2.md](quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | ✓ | not recommended | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [COSMOS2IMAGE.md](quickstart/COSMOS2IMAGE.md) |
| LTX Video | ~2.5B | ✓ | ✓ | ✓ | int8/fp8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LTXVIDEO.md](quickstart/LTXVIDEO.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [HUNYUANVIDEO.md](quickstart/HUNYUANVIDEO.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [WAN.md](quickstart/WAN.md) |
| Qwen Image | 20B | ✓ | ✓ | ✓* | **required** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [QWEN_IMAGE.md](quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓ | ✓* | **required** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [QWEN_EDIT.md](quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, 3.6B prior | ✓ | ✓ | ✓* | not supported | fp32 (required) | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | [STABLE_CASCADE_C.md](quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [KANDINSKY5_IMAGE.md](quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [KANDINSKY5_VIDEO.md](quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓ | ✓* | int8/fp8 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_VIDEO.md](quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓ | ✓* | int8/fp8 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_VIDEO_EDIT.md](quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓ | ✓* | int8/fp8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_IMAGE.md](quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓ | ✓* | int8/fp8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_EDIT.md](quickstart/LONGCAT_EDIT.md) |

*✓ = supported, ✓* = requires DeepSpeed/FSDP2 for full-rank, ✗ = not supported, `✓+` indicates checkpointing is recommended due to VRAM pressure. TwinFlow ✓ means native support when `twinflow_enabled=true` (diffusion models need `diff2flow_enabled+twinflow_allow_diff2flow`). LayerSync ✓ means the backbone exposes transformer hidden states for self-alignment; ✗ marks UNet-style backbones without that buffer. †Sliders apply to LoRA and LyCORIS (including full-rank LyCORIS “full”).*

> ℹ️ Wan quickstart includes 2.1 + 2.2 stage presets and the time-embedding toggle. Flux Kontext covers editing workflows built atop Flux.1.

> ⚠️ These quickstarts are living documents. Expect occasional updates as new models land or training recipes improve.

### Fast paths: Z-Image Turbo & Flux Schnell

- **Z-Image Turbo**: Fully supported LoRA with TREAD; runs fast on NVIDIA and macOS even without quant (int8 works too). Often the bottleneck is just trainer setup.
- **Flux Schnell**: The quickstart config handles the fast noise schedule and assistant LoRA stack automatically; no extra flags needed to train Schnell LoRAs.

### Advanced Experimental Features

- **Diff2Flow**: Allows training standard epsilon/v-prediction models (SD1.5, SDXL, DeepFloyd, etc.) using a Flow Matching loss objective. This bridges the gap between older architectures and modern flow-based training.
- **Scheduled Sampling**: Reduces exposure bias by letting the model generate its own intermediate noisy latents during training ("rollout"). This helps the model learn to recover from its own generation errors.
