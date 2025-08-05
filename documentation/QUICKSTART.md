# Quickstart Guide

**Note**: For more advanced configurations, see the [tutorial](/TUTORIAL.md), [dataloader configuration guide](/documentation/DATALOADER.md), and the [options breakdown](/OPTIONS.md) pages.

## Feature Compatibility Matrix

| Model                                             | Params       | PEFT LoRA | Lycoris LoKr | Full-Rank | Quantization              | Mixed Precision | Grad Checkpoint      | Flow Shift      | ControlNet LoRA|
|---------------------------------------------------|-------------:|:----:|:----:|:---------:|:------------------------:|:---------------:|:--------------------:|:---------------:|:---------:|
| [PixArt Sigma](/documentation/quickstart/SIGMA.md)| 0.6B-0.9B        |      |  ✓   |     ✓     | optional (int8)           | bf16            | ✓                    |                 | ✓          |
| [NVLabs Sana](/documentation/quickstart/SANA.md)  | 1.6B-4.8B       |      |  ✓   |     ✓     | optional (int8)           | bf16            | ✓+                   | ✓               |           |
| [Kwai Kolors](/documentation/quickstart/KOLORS.md)| 2.7 B         |  ✓   |  ✓   |     ✓     | not recommended           | bf16            | ✓                    |                 |          |
| [Stable Diffusion 3](/documentation/quickstart/SD3.md)| 2B–8B    |  ✓   |  ✓   |     ✓     | optional (int8, fp8, nf4)           | bf16            | ✓+                   | ✓ (SLG)         | ✓         |
| [Flux.1](/documentation/quickstart/FLUX.md)      | ~8B-12B         |  ✓   |  ✓   |     ✓*    | optional (int8,  fp8,  nf4)  | bf16            | ✓+                   | ✓               | ✓         |
| [Auraflow](/documentation/quickstart/AURAFLOW.md)| 6 B          |  ✓   |  ✓   |     ✓*    | optional (int8,  fp8,  nf4)  | bf16            | ✓+                   | ✓ (SLG)         | ✓         |
| [HiDream I1](/documentation/quickstart/HIDREAM.md)| 17 B (8.5B MoE)|  ✓   |  ✓   |     ✓*    | optional (int8,  fp8,  nf4)  | bf16            | ✓                    | ✓               | ✓          |
| [OmniGen](/documentation/quickstart/OMNIGEN.md)  | 3.8 B        |  ✓   |  ✓   |     ✓     | optional (int8,  fp8)       | bf16            | ✓                    | ✓               |           |
| [Stable Diffusion XL](/documentation/quickstart/SDXL.md)| 2.6 B      |  ✓   |  ✓   |     ✓     | not recommended           | bf16            | ✓                    |                 | ✓         |
| [Lumina2](/documentation/quickstart/LUMINA2.md)      | 2B   |  ✓   |  ✓   |     ✓    | optional (int8)           | bf16            | ✓                    | ✓               |           |
| [Cosmos2](/documentation/quickstart/COSMOS2IMAGE.md)      | 2B   |  ✓   |  ✓   |     ✓    | not recommended         | bf16            | ✓                    | ✓               |           |
| [LTX Video](/documentation/quickstart/LTXVIDEO.md)| ~2.5 B      |  ✓   |  ✓   |     ✓     | optional (int8,  fp8)       | bf16            | ✓                    | ✓               |           |
| [Wan 2.1](/documentation/quickstart/WAN.md)      | 1.3B-14B   |  ✓   |  ✓   |     ✓*    | optional (int8)           | bf16            | ✓                    | ✓               |           |
| [Qwen Image](/documentation/quickstart/QWEN_IMAGE.md) | 20B |  ✓   |  ✓   |     ✓*    | required (int8, nf4)      | bf16            | ✓ (required)         | ✓               |           |

* _✓ = Supported_
* _* Requires DeepSpeed for full-rank_
* _+ Allows gradient checkpointing with interval specified_
* _SLG = Skip-layer guidance_

> ⚠️ These tutorials are a work-in-progress. They contain full end-to-end instructions for a basic training session.