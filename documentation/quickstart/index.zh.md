# 模型指南

面向每个支持的模型架构的逐步训练指南。

## 图像模型

### 流匹配

| 模型 | 参数 | 许可证 | 允许商用 | 指南 |
| ------- | ------------ | --- | :---: | ------- |
| **Flux.1** | 12B | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 有条件<sup>3</sup> | [Flux.1 指南](FLUX.md) |
| **Flux.2** | 32B | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 有条件<sup>4</sup> | [Flux.2 指南](FLUX2.md) |
| **Flux Kontext** | 12B | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) | 否<sup>5</sup> | [Kontext 指南](FLUX_KONTEXT.md) |
| **Chroma** | 8.9B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [Chroma 指南](CHROMA.md) |
| **Stable Diffusion 3** | 2-8B | [Stability AI Community](https://stability.ai/license) | 有条件<sup>2</sup> | [SD3 指南](SD3.md) |
| **Auraflow** | 6.8B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) / [Pony License](https://huggingface.co/purplesmartai/pony-v7-base/blob/main/LICENSE) | 有条件<sup>8</sup> | [Auraflow 指南](AURAFLOW.md) |
| **Sana** | 0.6-4.8B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [Sana 指南](SANA.md) |
| **Lumina2** | 2B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [Lumina2 指南](LUMINA2.md) |
| **HiDream** | 17B MoE | [MIT](https://opensource.org/license/mit) | 是 | [HiDream 指南](HIDREAM.md) |
| **Z-Image** | - | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [Z-Image 指南](ZIMAGE.md) |
| **Krea2** | - | [Krea 2 Community](https://www.krea.ai/krea-2-licensing) | 是<sup>6</sup> | [Krea2 指南](KREA2.zh.md) |
| **Mage-Flow** | 4B | [MIT](https://opensource.org/license/mit) | 是 | [Mage-Flow 指南](MAGEFLOW.zh.md) |
| **Boogu-Image** | - | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [Boogu-Image 指南](BOOGU_IMAGE.zh.md) |
| **zlab i1** | 3B | [MIT](https://opensource.org/license/mit) | 是 | [zlab i1 指南](ZLAB_i1.zh.md) |
| **Ideogram 4** | 9B | [Ideogram 4 Non-Commercial](https://huggingface.co/ideogram-ai/ideogram-4-nf4/blob/main/LICENSE.md) | 否<sup>5</sup> | [Ideogram 4 指南](IDEOGRAM4.zh.md) |
| **ERNIE-Image** | - | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [ERNIE 指南](ERNIE.md) |

### DiT / Transformer

| 模型 | 参数 | 许可证 | 允许商用 | 指南 |
| ------- | ------------ | --- | :---: | ------- |
| **PixArt Sigma** | 0.6-0.9B | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | 有条件<sup>1</sup> | [Sigma 指南](SIGMA.md) |
| **Cosmos2** | 2-14B | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | 是<sup>9</sup> | [Cosmos2 指南](COSMOS2IMAGE.md) |
| **Cosmos3** | 4-65B | [OpenMDW 1.1](https://github.com/OpenMDW/openmdw/blob/main/1.1/LICENSE.OpenMDW-1.1) | 是 | [Cosmos3 指南](COSMOS3.zh.md) |
| **OmniGen** | 3.8B | [MIT](https://opensource.org/license/mit) | 是 | [OmniGen 指南](OMNIGEN.md) |
| **Qwen Image** | 20B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [Qwen 指南](QWEN_IMAGE.md) |
| **LongCat Image** | 6B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [LongCat 指南](LONGCAT_IMAGE.md) |
| **Kandinsky 5** | - | [MIT](https://opensource.org/license/mit) | 是 | [Kandinsky 指南](KANDINSKY5_IMAGE.md) |

### U-Net

| 模型 | 参数 | 许可证 | 允许商用 | 指南 |
| ------- | ------------ | --- | :---: | ------- |
| **Stable Diffusion XL** | 3.5B | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | 有条件<sup>1</sup> | [SDXL 指南](SDXL.md) |
| **Kolors** | 5B | [Kwai Kolors License](https://huggingface.co/terminusresearch/kwai-kolors-1.0/blob/main/MODEL_LICENSE) | Abandonware<sup>7</sup> | [Kolors 指南](KOLORS.md) |
| **Stable Cascade** | - | [Stable Cascade NC Community](https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE) | Abandonware<sup>7</sup> | [Cascade 指南](STABLE_CASCADE_C.md) |

### 图像编辑

| 模型 | 许可证 | 允许商用 | 指南 |
| ------- | --- | :---: | ------- |
| **Qwen Edit** | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [Qwen Edit 指南](QWEN_EDIT.md) |
| **LongCat Edit** | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [LongCat Edit 指南](LONGCAT_EDIT.md) |

## 视频模型

| 模型 | 参数 | 许可证 | 允许商用 | 指南 |
| ------- | ------------ | --- | :---: | ------- |
| **Wan Video** | 1.3-14B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [Wan 指南](WAN.md) |
| **LTX Video** | 5B | [LTX Video OpenRAIL-M](https://huggingface.co/Lightricks/LTX-Video-0.9.5/blob/main/ltx-video-2b-v0.9.5.license.txt) | 有条件<sup>10</sup> | [LTX 指南](LTXVIDEO.md) |
| **LTX Video 2** | 19B | [LTX-2 Community](https://ltx.io/model/license) | 有条件<sup>10</sup> | [LTX Video 2 指南](LTXVIDEO2.md) |
| **Cosmos3** | 4-65B | [OpenMDW 1.1](https://github.com/OpenMDW/openmdw/blob/main/1.1/LICENSE.OpenMDW-1.1) | 是 | [Cosmos3 指南](COSMOS3.zh.md) |
| **Hunyuan Video** | 8.3B | [Tencent Hunyuan Community](https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/LICENSE) | 有条件<sup>11</sup> | [Hunyuan 指南](HUNYUANVIDEO.md) |
| **MiniMax H3** | 33B | [MiniMax H3 Community](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) | 有条件<sup>12</sup> | 无专用指南 |
| **Sana Video** | - | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 是 | [Sana Video 指南](SANAVIDEO.md) |
| **Kandinsky 5 Video** | - | [MIT](https://opensource.org/license/mit) | 是 | [Kandinsky Video 指南](KANDINSKY5_VIDEO.md) |
| **LongCat Video** | - | [MIT](https://opensource.org/license/mit) | 是 | [LongCat Video 指南](LONGCAT_VIDEO.md) |
| **LongCat Video Edit** | - | [MIT](https://opensource.org/license/mit) | 是 | [LongCat Video Edit 指南](LONGCAT_VIDEO_EDIT.md) |

**许可证说明：** 商用状态覆盖模型权重、派生 checkpoint、fine-tune 和托管模型使用。生成输出的权利可能不同；商业部署前请以链接的许可证正文为准。

<sup>1</sup> OpenRAIL 风格许可证通常允许商用，但使用限制仍适用于模型和派生物。

<sup>2</sup> Stability AI Community License 适用于低于收入门槛且符合条件的用户；更大规模商用需要 Stability 企业条款。

<sup>3</sup> Flux.1 随 flavour 不同而不同：Schnell 和 LibreFlux 为 Apache-2.0，Dev、Krea 和 Kontext 使用 BFL 非商用条款；FluxBooru 商用前请检查 upstream metadata。

<sup>4</sup> Flux.2 随 flavour 不同而不同：Klein 4B 为 Apache-2.0，Dev 和 Klein 9B 使用 BFL 非商用条款。

<sup>5</sup> 公开的非商用模型条款不允许在没有单独许可证的情况下商用权重、派生 checkpoint 或托管模型服务。

<sup>6</sup> Krea 2 Community License 在满足收入上限（年收入低于 $1M）和安全/过滤要求时允许商用；否则需要企业许可证。

<sup>7</sup> Abandonware 表示原供应商实际上已放弃维护该模型，且没有可靠的许可申请路径；最终用户需要自行判断是否接受该风险。

<sup>8</sup> AuraFlow 支持 Apache-2.0 upstream flavour，以及带有单独自定义许可证的 Pony flavour；请检查所选 flavour。

<sup>9</sup> NVIDIA Open Model License 允许商用，但包含协议、可接受使用和出口管制条款。

<sup>10</sup> LTX Video 0.9.5 使用 OpenRAIL-M；LTX Video 2 使用带商用收入门槛的 LTX community terms。

<sup>11</sup> Tencent Hunyuan Community License 包含地域排除，以及针对超大规模服务的商用门槛。

<sup>12</sup> MiniMax H3 Community License 将美国、欧盟、英国和韩国排除在标准适用地区之外；这些地区需要单独授权。


## 音频模型

| 模型 | 规模 / 版本 | 指南 |
|-------|--------------|-------|
| **ACE-Step** | 3.5B / 1.5 | [ACE-Step 指南](ACE_STEP.md) |
| **HeartMuLa** | 3B | [HeartMuLa 指南](HEARTMULA.md) |

## 选择模型

**新手:**

- 获取高质量图像生成请从 **Flux.1** 开始
- 使用 **LoRA** 训练降低显存需求

**生产:**

- 需要更广兼容性可选 **SD3** 或 **SDXL**
- 追求最高质量可选 **Flux.2**（需要更多显存）

**视频:**

- 质量与资源平衡可选 **Wan Video**
- I2V 与超分辨率可选 **Hunyuan Video**

**特定场景:**

- 图像编辑/条件控制可选 **Flux Kontext**
- 文生音乐 LoRA 训练可选 **ACE-Step**（v1 与 v1.5）
- 自回归文本到音频可选 **HeartMuLa**
