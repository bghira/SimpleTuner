# Quickstart Guide

**Note**: For more advanced configurations, see the [tutorial](TUTORIAL.md) and [options reference](OPTIONS.md).

## Feature Compatibility

For the complete and most accurate feature matrix, refer to the [main README](https://github.com/bghira/SimpleTuner#model-architecture-support).

## Model Quickstart Guides

| Model | Params | PEFT LoRA | Full-Rank | Quantization | Mixed Precision | Grad Checkpoint | Flow Shift | TwinFlow | Self-Flow | LayerSync | Ref Inputs | ControlNet | Sliders† | Guide |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | int8 optional | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | [SIGMA.md](/documentation/quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | int8 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [SANA.md](/documentation/quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | not recommended | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [KOLORS.md](/documentation/quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [SD3.md](/documentation/quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [FLUX.md](/documentation/quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [FLUX2.md](/documentation/quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✓ | ✓ | [FLUX_KONTEXT.md](/documentation/quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [ZIMAGE.md](/documentation/quickstart/ZIMAGE.md) |
| Boogu-Image 0.1 | - | ✓ | ✓* | fp8 optional | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ edit | ✗ | ✓ | [BOOGU_IMAGE.md](/documentation/quickstart/BOOGU_IMAGE.md) |
| zlab i1 | 3B | ✓ | ✓ | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [ZLAB_i1.md](/documentation/quickstart/ZLAB_i1.md) |
| Ideogram 4 | 9B | ✓ | ✓* | fp8 default, nf4 optional | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [IDEOGRAM4.md](/documentation/quickstart/IDEOGRAM4.md) |
| ACE-Step | 3.5B | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [ACE_STEP.md](/documentation/quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [CHROMA.md](/documentation/quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [AURAFLOW.md](/documentation/quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [HIDREAM.md](/documentation/quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | int8/fp8 optional | bf16 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | [OMNIGEN.md](/documentation/quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | not recommended | bf16 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | [SDXL.md](/documentation/quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | [LUMINA2.md](/documentation/quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | not recommended | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [COSMOS2IMAGE.md](/documentation/quickstart/COSMOS2IMAGE.md) |
| LTX Video | ~2.5B | ✓ | ✓ | int8/fp8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [LTXVIDEO.md](/documentation/quickstart/LTXVIDEO.md) |
| LTX Video 2 | 19B | ✓ | ✓* | int8/fp8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [LTXVIDEO2.md](/documentation/quickstart/LTXVIDEO2.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [HUNYUANVIDEO.md](/documentation/quickstart/HUNYUANVIDEO.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [WAN.md](/documentation/quickstart/WAN.md) |
| Wan 2.2 S2V | 14B | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [WAN_S2V.md](/documentation/quickstart/WAN_S2V.md) |
| Qwen Image | 20B | ✓ | ✓* | **required** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [QWEN_IMAGE.md](/documentation/quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓* | **required** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [QWEN_EDIT.md](/documentation/quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, 3.6B prior | ✓ | ✓* | not supported | fp32 (required) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [STABLE_CASCADE_C.md](/documentation/quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ I2I | ✗ | ✓ | [KANDINSKY5_IMAGE.md](/documentation/quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [KANDINSKY5_VIDEO.md](/documentation/quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓* | int8/fp8 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [LONGCAT_VIDEO.md](/documentation/quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓* | int8/fp8 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [LONGCAT_VIDEO_EDIT.md](/documentation/quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓* | int8/fp8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [LONGCAT_IMAGE.md](/documentation/quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓* | int8/fp8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [LONGCAT_EDIT.md](/documentation/quickstart/LONGCAT_EDIT.md) |

*✓ = supported, ✓* = requires DeepSpeed/FSDP2 for full-rank, ✗ = not supported, `✓+` indicates checkpointing is recommended due to VRAM pressure. Ref Inputs marks existing reference/edit/I2V conditioning paths; `opt` means optional, `req` means the edit/I2V flavour requires it. TwinFlow ✓ means native support when `twinflow_enabled=true` (diffusion models need `diff2flow_enabled+twinflow_allow_diff2flow`). Self-Flow ✓ means native support for `crepa_enabled=true` with `crepa_feature_source=self_flow`, `use_ema=true`, and `crepa_teacher_block_index` set. LayerSync ✓ means the backbone exposes transformer hidden states for self-alignment; ✗ marks UNet-style backbones without that buffer. †Sliders apply to LoRA and LyCORIS (including full-rank LyCORIS "full"). All models support LyCORIS.*

> ℹ️ Wan quickstart includes 2.1 + 2.2 stage presets and the time-embedding toggle. Flux Kontext covers editing workflows built atop Flux.1.

> ⚠️ These quickstarts are living documents. Expect occasional updates as new models land or training recipes improve.

### Fast paths: Z-Image Turbo & Flux Schnell

- **Z-Image Turbo**: Fully supported LoRA with TREAD; runs fast on NVIDIA and macOS even without quant (int8 works too). Often the bottleneck is just trainer setup.
- **Flux Schnell**: The quickstart config handles the fast noise schedule and assistant LoRA stack automatically; no extra flags needed to train Schnell LoRAs.

### Advanced Experimental Features

- **Diff2Flow**: Allows training standard epsilon/v-prediction models (SD1.5, SDXL, DeepFloyd, etc.) using a Flow Matching loss objective. This bridges the gap between older architectures and modern flow-based training.
- **Scheduled Sampling**: Reduces exposure bias by letting the model generate its own intermediate noisy latents during training ("rollout"). This helps the model learn to recover from its own generation errors.

## Common Issues

### Dataset has fewer samples than expected

If your dataset ends up with fewer usable samples than you expected, files may have been filtered during processing. Common reasons include:

- **Files too small**: Images below `minimum_image_size` are filtered out
- **Aspect ratio out of range**: Images outside `minimum_aspect_ratio`/`maximum_aspect_ratio` bounds are excluded
- **Duration limits**: Audio/video files exceeding duration limits are skipped

**Viewing filtering statistics:**
- In the WebUI, browse to your dataset directory and select it to see filtering statistics
- Check the logs during dataset processing for statistics like: `Sample processing statistics: {'total_processed': 100, 'skipped': {'too_small': 15, ...}}`

For detailed troubleshooting, see [Troubleshooting filtered datasets](DATALOADER.md#troubleshooting-filtered-datasets) in the dataloader documentation.
