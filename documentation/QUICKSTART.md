# Quickstart Guide

**Note**: For more advanced configurations, see the [tutorial](TUTORIAL.md) and [options reference](OPTIONS.md).

## Feature Compatibility

For the complete and most accurate feature matrix, refer to the [main README](https://github.com/bghira/SimpleTuner#model-architecture-support).

## Model Quickstart Guides

| Model | Params | PEFT LoRA | Full-Rank | Quantization | Mixed Precision | Grad Checkpoint | Flow Shift | TwinFlow | Self-Flow | LayerSync | Ref Inputs | ControlNet | Sliders† | License | Allows commercial use | Guide |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | int8 optional | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | Conditions apply<sup>1</sup> | [SIGMA.md](quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | int8 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [SANA.md](quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | not recommended | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Kwai Kolors License](https://huggingface.co/terminusresearch/kwai-kolors-1.0/blob/main/MODEL_LICENSE) | Conditions apply<sup>7</sup> | [KOLORS.md](quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [Stability AI Community](https://stability.ai/license) | Conditions apply<sup>2</sup> | [SD3.md](quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Conditions apply<sup>3</sup> | [FLUX.md](quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Conditions apply<sup>4</sup> | [FLUX2.md](quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✓ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) | No<sup>5</sup> | [FLUX_KONTEXT.md](quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [ZIMAGE.md](quickstart/ZIMAGE.md) |
| Krea2 | - | ✓ | ✓* | int8 optional | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✓ opt | ✗ | ✓ | [Krea 2 Community](https://www.krea.ai/krea-2-licensing) | Conditions apply<sup>6</sup> | [KREA2.md](quickstart/KREA2.md) |
| Mage-Flow | 4B | ✓ | ✓* | int8/fp8 optional | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ edit | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Yes | [MAGEFLOW.md](quickstart/MAGEFLOW.md) |
| Boogu-Image 0.1 | - | ✓ | ✓* | fp8 optional | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ edit | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [BOOGU_IMAGE.md](quickstart/BOOGU_IMAGE.md) |
| zlab i1 | 3B | ✓ | ✓ | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Unspecified](https://huggingface.co/bghira/zlab-i1-diffusers) | Conditions apply<sup>12</sup> | [ZLAB_i1.md](quickstart/ZLAB_i1.md) |
| Ideogram 4 | 9B | ✓ | ✓* | fp8 default, nf4 optional | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Ideogram 4 Non-Commercial](https://huggingface.co/ideogram-ai/ideogram-4-nf4/blob/main/LICENSE.md) | No<sup>5</sup> | [IDEOGRAM4.md](quickstart/IDEOGRAM4.md) |
| ERNIE-Image | - | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [ERNIE.md](quickstart/ERNIE.md) |
| ACE-Step | 3.5B | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) / [MIT](https://huggingface.co/ACE-Step/Ace-Step1.5) | Yes | [ACE_STEP.md](quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [CHROMA.md](quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) / [Pony License](https://huggingface.co/purplesmartai/pony-v7-base/blob/main/LICENSE) | Conditions apply<sup>8</sup> | [AURAFLOW.md](quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓* | int8/fp8/nf4 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [MIT](https://opensource.org/license/mit) | Yes | [HIDREAM.md](quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | int8/fp8 optional | bf16 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Yes | [OMNIGEN.md](quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | not recommended | bf16 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | Conditions apply<sup>1</sup> | [SDXL.md](quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [LUMINA2.md](quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | not recommended | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Yes<sup>9</sup> | [COSMOS2IMAGE.md](quickstart/COSMOS2IMAGE.md) |
| Cosmos3 | 16B-65B | ✓ | ✓* | no_change first | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | audio opt | ✗ | ✓ | [OpenMDW 1.1](https://github.com/OpenMDW/openmdw/blob/main/1.1/LICENSE.OpenMDW-1.1) | Yes | [COSMOS3.md](quickstart/COSMOS3.md) |
| LTX Video | ~2.5B | ✓ | ✓ | int8/fp8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [LTX Video OpenRAIL-M](https://huggingface.co/Lightricks/LTX-Video-0.9.5/blob/main/ltx-video-2b-v0.9.5.license.txt) | Conditions apply<sup>10</sup> | [LTXVIDEO.md](quickstart/LTXVIDEO.md) |
| LTX Video 2 | 19B | ✓ | ✓* | int8/fp8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [LTX-2 Community](https://ltx.io/model/license) | Conditions apply<sup>10</sup> | [LTXVIDEO2.md](quickstart/LTXVIDEO2.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [Tencent Hunyuan Community](https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/LICENSE) | Conditions apply<sup>11</sup> | [HUNYUANVIDEO.md](quickstart/HUNYUANVIDEO.md) |
| SanaVideo | 2B | ✓ | ✓* | int8/fp8 optional | bf16 | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [SANAVIDEO.md](quickstart/SANAVIDEO.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [WAN.md](quickstart/WAN.md) |
| Wan 2.2 S2V | 14B | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [WAN_S2V.md](quickstart/WAN_S2V.md) |
| Qwen Image | 20B | ✓ | ✓* | **required** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [QWEN_IMAGE.md](quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓* | **required** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [QWEN_EDIT.md](quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, 3.6B prior | ✓ | ✓* | not supported | fp32 (required) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Stable Cascade NC Community](https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE) | No<sup>5</sup> | [STABLE_CASCADE_C.md](quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ I2I | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Yes | [KANDINSKY5_IMAGE.md](quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓* | int8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Yes | [KANDINSKY5_VIDEO.md](quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓* | int8/fp8 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Yes | [LONGCAT_VIDEO.md](quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓* | int8/fp8 optional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Yes | [LONGCAT_VIDEO_EDIT.md](quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓* | int8/fp8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [LONGCAT_IMAGE.md](quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓* | int8/fp8 optional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [LONGCAT_EDIT.md](quickstart/LONGCAT_EDIT.md) |

*✓ = supported, ✓* = requires DeepSpeed/FSDP2 for full-rank, ✗ = not supported, `✓+` indicates checkpointing is recommended due to VRAM pressure. Ref Inputs marks existing reference/edit/I2V conditioning paths; `opt` means optional, `req` means the edit/I2V flavour requires it. TwinFlow ✓ means native support when `twinflow_enabled=true` (diffusion models need `diff2flow_enabled+twinflow_allow_diff2flow`). Self-Flow ✓ means native support for `crepa_enabled=true` with `crepa_feature_source=self_flow`, `use_ema=true`, and `crepa_teacher_block_index` set. LayerSync ✓ means the backbone exposes transformer hidden states for self-alignment; ✗ marks UNet-style backbones without that buffer. †Sliders apply to LoRA and LyCORIS (including full-rank LyCORIS "full"). All models support LyCORIS.*

**License notes:** The commercial-use status covers model weights, derivative checkpoints, fine-tunes, and hosted model use. Generated-output rights can differ; read the linked license text before commercial deployment.

<sup>1</sup> OpenRAIL-style licenses generally permit commercial use with usage restrictions that remain attached to the model and derivatives.

<sup>2</sup> Stability AI Community License is available for qualifying users below the revenue threshold; larger commercial use needs Stability enterprise terms.

<sup>3</sup> Flux.1 varies by flavour: Schnell and LibreFlux are Apache-2.0, while Dev, Krea, and Kontext use BFL non-commercial terms; review FluxBooru upstream metadata before commercial use.

<sup>4</sup> Flux.2 varies by flavour: Klein 4B is Apache-2.0, while Dev and Klein 9B use BFL non-commercial terms.

<sup>5</sup> Public non-commercial model terms do not permit commercial use of weights, derivative checkpoints, or hosted model services without a separate license.

<sup>6</sup> Krea 2 Community License permits commercial use only under its revenue and safety/filtering requirements; otherwise an enterprise license is required.

<sup>7</sup> Kolors commercial model or derivative use requires applying for and receiving explicit permission from the licensor.

<sup>8</sup> AuraFlow supports Apache-2.0 upstream flavours and a Pony flavour with a separate custom license; check the selected flavour.

<sup>9</sup> NVIDIA Open Model License permits commercial use but includes agreement, acceptable-use, and export-control terms.

<sup>10</sup> LTX Video 0.9.5 uses OpenRAIL-M; LTX Video 2 uses LTX community terms with a revenue threshold for commercial use.

<sup>11</sup> Tencent Hunyuan Community License includes territorial exclusions and a commercial threshold for very large services.

<sup>12</sup> This mirror publishes `license: other` without a standard license text; review upstream terms before commercial use.

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
