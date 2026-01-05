# 快速开始指南

**注意**：如需更高级的配置，请参阅[教程](TUTORIAL.md)和[选项参考](OPTIONS.md)。

## 功能兼容性

完整且最准确的功能矩阵，请参阅[主 README](https://github.com/bghira/SimpleTuner#model-architecture-support)。

## 模型快速开始指南

| 模型 | 参数量 | PEFT LoRA | Lycoris | 全秩 | 量化 | 混合精度 | 梯度检查点 | Flow Shift | TwinFlow | LayerSync | ControlNet | Sliders† | 指南 |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | ✓ | int8 可选 | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | [SIGMA.md](quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | ✓ | int8 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [SANA.md](quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | ✓ | 不推荐 | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | [KOLORS.md](quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | ✓ | int8/fp8/nf4 可选 | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | [SD3.md](quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ | [FLUX.md](quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓ | ✓* | int8/fp8/nf4 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [FLUX2.md](quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ | [FLUX_KONTEXT.md](quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✗ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [ZIMAGE.md](quickstart/ZIMAGE.md) |
| ACE-Step | 3.5B | ✓ | ✓ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [ACE_STEP.md](quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓ | ✓* | int8/fp8/nf4 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [CHROMA.md](quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓ | ✓* | int8/fp8/nf4 可选 | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | [AURAFLOW.md](quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓ | ✓* | int8/fp8/nf4 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [HIDREAM.md](quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | ✓ | int8/fp8 可选 | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | [OMNIGEN.md](quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | ✓ | 不推荐 | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | [SDXL.md](quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | ✓ | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [LUMINA2.md](quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | ✓ | 不推荐 | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [COSMOS2IMAGE.md](quickstart/COSMOS2IMAGE.md) |
| LTX Video | ~2.5B | ✓ | ✓ | ✓ | int8/fp8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LTXVIDEO.md](quickstart/LTXVIDEO.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [HUNYUANVIDEO.md](quickstart/HUNYUANVIDEO.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [WAN.md](quickstart/WAN.md) |
| Qwen Image | 20B | ✓ | ✓ | ✓* | **必需** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [QWEN_IMAGE.md](quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓ | ✓* | **必需** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [QWEN_EDIT.md](quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, 3.6B prior | ✓ | ✓ | ✓* | 不支持 | fp32 (必需) | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | [STABLE_CASCADE_C.md](quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [KANDINSKY5_IMAGE.md](quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [KANDINSKY5_VIDEO.md](quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓ | ✓* | int8/fp8 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_VIDEO.md](quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓ | ✓* | int8/fp8 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_VIDEO_EDIT.md](quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓ | ✓* | int8/fp8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_IMAGE.md](quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓ | ✓* | int8/fp8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_EDIT.md](quickstart/LONGCAT_EDIT.md) |

*✓ = 支持，✓* = 全秩训练需要 DeepSpeed/FSDP2，✗ = 不支持，`✓+` 表示由于 VRAM 压力建议启用检查点。TwinFlow ✓ 表示当 `twinflow_enabled=true` 时原生支持（扩散模型需要 `diff2flow_enabled+twinflow_allow_diff2flow`）。LayerSync ✓ 表示骨干网络暴露 transformer 隐藏状态用于自对齐；✗ 标记没有该缓冲区的 UNet 风格骨干网络。†Sliders 适用于 LoRA 和 LyCORIS（包括全秩 LyCORIS "full"）。*

> 注：Wan 快速开始包含 2.1 + 2.2 阶段预设和时间嵌入切换。Flux Kontext 涵盖基于 Flux.1 构建的编辑工作流程。

> 警告：这些快速开始指南是持续更新的文档。随着新模型的发布或训练方案的改进，预计会有不定期更新。

### 快速通道：Z-Image Turbo 和 Flux Schnell

- **Z-Image Turbo**：完全支持带 TREAD 的 LoRA；即使不使用量化（int8 也可以），在 NVIDIA 和 macOS 上运行速度也很快。通常瓶颈只是训练器设置。
- **Flux Schnell**：快速开始配置会自动处理快速噪声调度和辅助 LoRA 堆栈；训练 Schnell LoRA 不需要额外的标志。

### 高级实验功能

- **Diff2Flow**：允许使用 Flow Matching 损失目标训练标准的 epsilon/v-prediction 模型（SD1.5、SDXL、DeepFloyd 等）。这弥合了旧架构和现代基于流的训练之间的差距。
- **Scheduled Sampling**：通过让模型在训练期间生成自己的中间噪声潜变量（"rollout"）来减少曝光偏差。这有助于模型学习从自身的生成错误中恢复。
