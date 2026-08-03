# 快速开始指南

**注意**：如需更高级的配置，请参阅[教程](TUTORIAL.md)和[选项参考](OPTIONS.md)。

## 功能兼容性

完整且最准确的功能矩阵，请参阅[主 README](https://github.com/bghira/SimpleTuner#model-architecture-support)。

## 模型快速开始指南

| 模型 | 参数量 | PEFT LoRA | Lycoris | 全秩 | 量化 | 混合精度 | 梯度检查点 | Flow Shift | TwinFlow | Self-Flow | LayerSync | Ref Inputs | ControlNet | Sliders† | 许可证 | 允许商用 | 指南 |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | ✓ | int8 可选 | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | 有条件<sup>1</sup> | [SIGMA.md](quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | ✓ | int8 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [SANA.md](quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | ✓ | 不推荐 | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Kwai Kolors License](https://huggingface.co/terminusresearch/kwai-kolors-1.0/blob/main/MODEL_LICENSE) | 有条件<sup>7</sup> | [KOLORS.md](quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | ✓ | int8/fp8/nf4 可选 | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [Stability AI Community](https://stability.ai/license) | 有条件<sup>2</sup> | [SD3.md](quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 有条件<sup>3</sup> | [FLUX.md](quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓ | ✓* | int8/fp8/nf4 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 有条件<sup>4</sup> | [FLUX2.md](quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✓ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) | 否<sup>5</sup> | [FLUX_KONTEXT.md](quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✗ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [ZIMAGE.md](quickstart/ZIMAGE.md) |
| Krea2 | - | ✓ | ✗ | ✓* | int8 可选 | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✓ opt | ✗ | ✓ | [Krea 2 Community](https://www.krea.ai/krea-2-licensing) | 有条件<sup>6</sup> | [KREA2.md](quickstart/KREA2.zh.md) |
| Mage-Flow | 4B | ✓ | ✓ | ✓* | int8/fp8 可选 | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ edit | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | 是 | [MAGEFLOW.md](quickstart/MAGEFLOW.zh.md) |
| Boogu-Image 0.1 | - | ✓ | ✓ | ✓* | fp8 可选 | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ edit | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [BOOGU_IMAGE.md](quickstart/BOOGU_IMAGE.zh.md) |
| zlab i1 | 3B | ✓ | ✓ | ✓ | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Unspecified](https://huggingface.co/bghira/zlab-i1-diffusers) | 有条件<sup>12</sup> | [ZLAB_i1.md](quickstart/ZLAB_i1.zh.md) |
| Ideogram 4 | 9B | ✓ | ✓ | ✓* | fp8 默认，nf4 可选 | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Ideogram 4 Non-Commercial](https://huggingface.co/ideogram-ai/ideogram-4-nf4/blob/main/LICENSE.md) | 否<sup>5</sup> | [IDEOGRAM4.md](quickstart/IDEOGRAM4.zh.md) |
| ERNIE-Image | - | ✓ | ✓ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [ERNIE.md](quickstart/ERNIE.zh.md) |
| ACE-Step | 3.5B | ✓ | ✓ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) / [MIT](https://huggingface.co/ACE-Step/Ace-Step1.5) | 是 | [ACE_STEP.md](quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓ | ✓* | int8/fp8/nf4 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [CHROMA.md](quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓ | ✓* | int8/fp8/nf4 可选 | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) / [Pony License](https://huggingface.co/purplesmartai/pony-v7-base/blob/main/LICENSE) | 有条件<sup>8</sup> | [AURAFLOW.md](quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓ | ✓* | int8/fp8/nf4 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [MIT](https://opensource.org/license/mit) | 是 | [HIDREAM.md](quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | ✓ | int8/fp8 可选 | bf16 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | 是 | [OMNIGEN.md](quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | ✓ | 不推荐 | bf16 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | 有条件<sup>1</sup> | [SDXL.md](quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | ✓ | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [LUMINA2.md](quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | ✓ | 不推荐 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | 是<sup>9</sup> | [COSMOS2IMAGE.md](quickstart/COSMOS2IMAGE.md) |
| Cosmos3 | 16B-65B | ✓ | ✓ | ✓* | no_change first | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | audio opt | ✗ | ✓ | [OpenMDW 1.1](https://github.com/OpenMDW/openmdw/blob/main/1.1/LICENSE.OpenMDW-1.1) | 是 | [COSMOS3.md](quickstart/COSMOS3.zh.md) |
| LTX Video | ~2.5B | ✓ | ✓ | ✓ | int8/fp8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [LTX Video OpenRAIL-M](https://huggingface.co/Lightricks/LTX-Video-0.9.5/blob/main/ltx-video-2b-v0.9.5.license.txt) | 有条件<sup>10</sup> | [LTXVIDEO.md](quickstart/LTXVIDEO.md) |
| LTX Video 2 | 19B | ✓ | ✓ | ✓* | int8/fp8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [LTX-2 Community](https://ltx.io/model/license) | 有条件<sup>10</sup> | [LTXVIDEO2.md](quickstart/LTXVIDEO2.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [Tencent Hunyuan Community](https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/LICENSE) | 有条件<sup>11</sup> | [HUNYUANVIDEO.md](quickstart/HUNYUANVIDEO.md) |
| SanaVideo | 2B | ✓ | ✓ | ✓* | int8/fp8 可选 | bf16 | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [SANAVIDEO.md](quickstart/SANAVIDEO.zh.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [WAN.md](quickstart/WAN.md) |
| Wan 2.2 S2V | 14B | ✓ | ✓ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [WAN_S2V.md](quickstart/WAN_S2V.md) |
| Qwen Image | 20B | ✓ | ✓ | ✓* | **必需** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [QWEN_IMAGE.md](quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓ | ✓* | **必需** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [QWEN_EDIT.md](quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, 3.6B prior | ✓ | ✓ | ✓* | 不支持 | fp32 (必需) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Stable Cascade NC Community](https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE) | 否<sup>5</sup> | [STABLE_CASCADE_C.md](quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ I2I | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | 是 | [KANDINSKY5_IMAGE.md](quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓ | ✓* | int8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | 是 | [KANDINSKY5_VIDEO.md](quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓ | ✓* | int8/fp8 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | 是 | [LONGCAT_VIDEO.md](quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓ | ✓* | int8/fp8 可选 | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | 是 | [LONGCAT_VIDEO_EDIT.md](quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓ | ✓* | int8/fp8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [LONGCAT_IMAGE.md](quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓ | ✓* | int8/fp8 可选 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [LONGCAT_EDIT.md](quickstart/LONGCAT_EDIT.md) |

*✓ = 支持，✓* = 全秩训练需要 DeepSpeed/FSDP2，✗ = 不支持，`✓+` 表示由于 VRAM 压力建议启用检查点。Ref Inputs 仅标记现有参考/编辑/I2V 条件路径；`opt` 表示可选，`req` 表示该编辑/I2V flavour 需要它。TwinFlow ✓ 表示当 `twinflow_enabled=true` 时原生支持（扩散模型需要 `diff2flow_enabled+twinflow_allow_diff2flow`）。Self-Flow ✓ 表示原生支持 `crepa_enabled=true`、`crepa_feature_source=self_flow`、`use_ema=true` 且设置 `crepa_teacher_block_index`。LayerSync ✓ 表示骨干网络暴露 transformer 隐藏状态用于自对齐；✗ 标记没有该缓冲区的 UNet 风格骨干网络。†Sliders 适用于 LoRA 和 LyCORIS（包括全秩 LyCORIS "full"）。*

**许可证说明：** 商用状态覆盖模型权重、派生 checkpoint、fine-tune 和托管模型使用。生成输出的权利可能不同；商业部署前请以链接的许可证正文为准。

<sup>1</sup> OpenRAIL 风格许可证通常允许商用，但使用限制仍适用于模型和派生物。

<sup>2</sup> Stability AI Community License 适用于低于收入门槛且符合条件的用户；更大规模商用需要 Stability 企业条款。

<sup>3</sup> Flux.1 随 flavour 不同而不同：Schnell 和 LibreFlux 为 Apache-2.0，Dev、Krea 和 Kontext 使用 BFL 非商用条款；FluxBooru 商用前请检查 upstream metadata。

<sup>4</sup> Flux.2 随 flavour 不同而不同：Klein 4B 为 Apache-2.0，Dev 和 Klein 9B 使用 BFL 非商用条款。

<sup>5</sup> 公开的非商用模型条款不允许在没有单独许可证的情况下商用权重、派生 checkpoint 或托管模型服务。

<sup>6</sup> Krea 2 Community License 只在满足收入和安全/过滤要求时允许商用；否则需要企业许可证。

<sup>7</sup> Kolors 模型或派生物的商用需要向许可方申请并获得明确许可。

<sup>8</sup> AuraFlow 支持 Apache-2.0 upstream flavour，以及带有单独自定义许可证的 Pony flavour；请检查所选 flavour。

<sup>9</sup> NVIDIA Open Model License 允许商用，但包含协议、可接受使用和出口管制条款。

<sup>10</sup> LTX Video 0.9.5 使用 OpenRAIL-M；LTX Video 2 使用带商用收入门槛的 LTX community terms。

<sup>11</sup> Tencent Hunyuan Community License 包含地域排除，以及针对超大规模服务的商用门槛。

<sup>12</sup> 此 mirror 发布 `license: other`，但没有标准许可证正文；商用前请检查 upstream terms。

> 注：Wan 快速开始包含 2.1 + 2.2 阶段预设和时间嵌入切换。Flux Kontext 涵盖基于 Flux.1 构建的编辑工作流程。

> 警告：这些快速开始指南是持续更新的文档。随着新模型的发布或训练方案的改进，预计会有不定期更新。

### 快速通道：Z-Image Turbo 和 Flux Schnell

- **Z-Image Turbo**：完全支持带 TREAD 的 LoRA；即使不使用量化（int8 也可以），在 NVIDIA 和 macOS 上运行速度也很快。通常瓶颈只是训练器设置。
- **Flux Schnell**：快速开始配置会自动处理快速噪声调度和辅助 LoRA 堆栈；训练 Schnell LoRA 不需要额外的标志。

### 高级实验功能

- **Diff2Flow**：允许使用 Flow Matching 损失目标训练标准的 epsilon/v-prediction 模型（SD1.5、SDXL、DeepFloyd 等）。这弥合了旧架构和现代基于流的训练之间的差距。
- **Scheduled Sampling**：通过让模型在训练期间生成自己的中间噪声潜变量（"rollout"）来减少曝光偏差。这有助于模型学习从自身的生成错误中恢复。

## 常见问题

### 数据集样本少于预期

如果数据集的可用样本少于预期，文件可能在处理过程中被过滤掉了。常见原因包括：

- **文件太小**：低于 `minimum_image_size` 的图像会被过滤
- **纵横比超出范围**：超出 `minimum_aspect_ratio`/`maximum_aspect_ratio` 边界的图像会被排除
- **时长限制**：超出时长限制的音频/视频文件会被跳过

**查看过滤统计：**
- 在 WebUI 中，浏览到您的数据集目录并选择它以查看过滤统计
- 在数据集处理期间检查日志中的统计信息，如：`Sample processing statistics: {'total_processed': 100, 'skipped': {'too_small': 15, ...}}`

详细故障排除请参阅数据加载器文档中的[故障排除-过滤后的数据集](DATALOADER.zh.md)。
