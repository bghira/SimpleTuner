# SimpleTuner

> 除非通过手动配置的 `report_to`、`push_to_hub` 或 webhooks 选项明确启用，否则不会向任何第三方发送数据。

**SimpleTuner** 专注于简洁性，致力于让代码易于理解。这个代码库是一个开放的学术实践项目，欢迎贡献。

如果您想加入我们的社区，可以通过 Terminus Research Group 在 [Discord](https://discord.gg/JGkSwEbjRb) 上找到我们。
如有任何问题，请随时在那里联系我们。

<img width="1944" height="1657" alt="image" src="https://github.com/user-attachments/assets/af3a24ec-7347-4ddf-8edf-99818a246de1" />


## 目录

- [设计理念](#设计理念)
- [教程](#教程)
- [功能特性](#功能特性)
  - [核心训练功能](#核心训练功能)
  - [模型架构支持](#模型架构支持)
  - [高级训练技术](#高级训练技术)
  - [模型特定功能](#模型特定功能)
  - [快速入门指南](#快速入门指南)
- [硬件要求](#硬件要求)
- [工具包](#工具包)
- [安装](#安装)
- [故障排除](#故障排除)

## 设计理念

- **简洁性**：力求为大多数用例提供良好的默认设置，减少调整需求。
- **多功能性**：设计用于处理从小型数据集到大规模数据集合的各种图像数量。
- **前沿功能**：仅整合经过验证有效的功能，避免添加未经测试的选项。

## 教程

在开始阅读[新版 Web UI 教程](/documentation/webui/TUTORIAL.md)或[传统命令行教程](/documentation/TUTORIAL.md)之前，请先完整阅读本 README，因为本文档包含您可能需要了解的重要信息。

如需快速上手而无需阅读完整文档或使用任何网页界面进行手动配置，可以使用[快速入门](/documentation/QUICKSTART.md)指南。

对于内存受限的系统，请参阅 [DeepSpeed 文档](/documentation/DEEPSPEED.md)，其中解释了如何使用 Accelerate 配置 Microsoft 的 DeepSpeed 进行优化器状态卸载。关于基于 DTensor 的分片和上下文并行，请阅读 [FSDP2 指南](/documentation/FSDP2.md)，该指南涵盖了 SimpleTuner 中新的 FullyShardedDataParallel v2 工作流程。

对于多节点分布式训练，[本指南](/documentation/DISTRIBUTED.md)将帮助您调整安装和快速入门指南中的配置，使其适用于多节点训练，并针对数十亿样本的图像数据集进行优化。

---

## 功能特性

SimpleTuner 为多种扩散模型架构提供全面的训练支持，并保持一致的功能可用性：

### 核心训练功能

- **用户友好的 Web UI** - 通过简洁的仪表板管理整个训练生命周期
- **多模态训练** - 统一的**图像、视频和音频**生成模型训练流程
- **多 GPU 训练** - 跨多个 GPU 的分布式训练，自动优化
- **高级缓存** - 图像、视频、音频和文本嵌入缓存到磁盘，加快训练速度
- **宽高比分桶** - 支持各种图像/视频尺寸和宽高比
- **概念滑块** - 适用于 LoRA/LyCORIS/全秩（通过 LyCORIS `full`）的滑块目标定位，支持正向/负向/中性采样和每提示词强度；参见 [Slider LoRA 指南](/documentation/SLIDER_LORA.md)
- **内存优化** - 大多数模型可在 24G GPU 上训练，通过优化许多模型可在 16G 上训练
- **DeepSpeed 和 FSDP2 集成** - 通过优化器/梯度/参数分片、上下文并行注意力、梯度检查点和优化器状态卸载，在较小的 GPU 上训练大型模型
- **S3 训练** - 直接从云存储训练（Cloudflare R2、Wasabi S3）
- **EMA 支持** - 指数移动平均权重，提高稳定性和质量
- **自定义实验追踪器** - 将 `accelerate.GeneralTracker` 放入 `simpletuner/custom-trackers` 并使用 `--report_to=custom-tracker --custom_tracker=<name>`

### 多用户和企业功能

SimpleTuner 包含完整的多用户训练平台，具有企业级功能——**永久免费开源**。

- **Worker 编排** - 注册分布式 GPU worker，自动连接到中央面板并通过 SSE 接收任务分发；支持临时（云端启动）和持久（始终在线）worker；参见 [Worker 编排指南](/documentation/experimental/server/WORKERS.md)
- **SSO 集成** - 支持 LDAP/Active Directory 或 OIDC 提供商（Okta、Azure AD、Keycloak、Google）认证；参见[外部认证指南](/documentation/experimental/server/EXTERNAL_AUTH.md)
- **基于角色的访问控制** - 四个默认角色（查看者、研究员、负责人、管理员），17+ 项细粒度权限；使用 glob 模式定义资源规则，按团队限制配置、硬件或提供商
- **组织和团队** - 分层多租户结构，基于上限的配额；组织限制强制执行绝对最大值，团队限制在组织范围内运作
- **配额和支出限制** - 在组织、团队或用户范围内强制执行成本上限（每日/每月）、作业并发限制和提交频率限制；操作包括阻止、警告或要求审批
- **带优先级的作业队列** - 五个优先级级别（低→紧急），跨团队公平调度，防止长时间等待作业的饥饿，管理员可覆盖优先级
- **审批工作流** - 可配置规则，在作业超过成本阈值、首次用户或特定硬件请求时触发审批；通过 UI、API 或邮件回复审批
- **邮件通知** - SMTP/IMAP 集成，用于作业状态、审批请求、配额警告和完成提醒
- **API 密钥和范围权限** - 为 CI/CD 流程生成带过期时间和限定范围的 API 密钥
- **审计日志** - 跟踪所有用户操作，带链式验证以满足合规要求；参见[审计指南](/documentation/experimental/server/AUDIT.md)

有关部署详情，请参阅[企业指南](/documentation/experimental/server/ENTERPRISE.md)。

### 模型架构支持

| 模型 | 参数量 | PEFT LoRA | Lycoris | 全秩 | ControlNet | 量化 | Flow Matching | 文本编码器 |
|-------|------------|-----------|---------|-----------|------------|--------------|---------------|---------------|
| **Stable Diffusion XL** | 3.5B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L/G |
| **Stable Diffusion 3** | 2B-8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L/G + T5-XXL |
| **Flux.1** | 12B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL |
| **Flux.2** | 32B | ✓ | ✓ | ✓* | ✗ | int8/fp8/nf4 | ✓ | Mistral-3 Small |
| **ACE-Step** | 3.5B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **Chroma 1** | 8.9B | ✓ | ✓ | ✓* | ✗ | int8/fp8/nf4 | ✓ | T5-XXL |
| **Auraflow** | 6.8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | UMT5-XXL |
| **PixArt Sigma** | 0.6B-0.9B | ✗ | ✓ | ✓ | ✓ | int8 | ✗ | T5-XXL |
| **Sana** | 0.6B-4.8B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2-2B |
| **Lumina2** | 2B | ✓ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2 |
| **Kwai Kolors** | 5B | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ChatGLM-6B |
| **LTX Video** | 5B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **LTX Video 2** | 19B | ✓ | ✓ | ✓* | ✗ | int8/fp8 | ✓ | Gemma3 |
| **Wan Video** | 1.3B-14B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **HiDream** | 17B (8.5B MoE) | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL + Llama |
| **Cosmos2** | 2B-14B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | T5-XXL |
| **OmniGen** | 3.8B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **Qwen Image** | 20B | ✓ | ✓ | ✓* | ✗ | int8/nf4（必需） | ✓ | T5-XXL |
| **SD 1.x/2.x（旧版）** | 0.9B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L |

*✓ = 支持，✗ = 不支持，* = 全秩训练需要 DeepSpeed*

### 高级训练技术

- **TREAD** - Transformer 模型的 Token 级 Dropout，包括 Kontext 训练
- **遮罩损失训练** - 通过分割/深度引导实现更好的收敛
- **先验正则化** - 增强训练稳定性，提高角色一致性
- **梯度检查点** - 可配置间隔，优化内存/速度
- **损失函数** - L2、Huber、Smooth L1，支持调度
- **SNR 加权** - Min-SNR gamma 加权，改善训练动态
- **分组卸载** - Diffusers v0.33+ 模块组 CPU/磁盘暂存，可选 CUDA 流
- **验证适配器扫描** - 在验证期间临时附加 LoRA 适配器（单个或 JSON 预设），以测量仅适配器或对比渲染，而不影响训练循环
- **外部验证钩子** - 将内置验证流程或上传后步骤替换为您自己的脚本，以便在另一个 GPU 上运行检查或将工件转发到您选择的任何云提供商（[详情](/documentation/OPTIONS.md#validation_method)）
- **CREPA 正则化** - 视频 DiT 的跨帧表示对齐（[指南](/documentation/experimental/VIDEO_CREPA.md)）
- **LoRA I/O 格式** - 以标准 Diffusers 布局或 ComfyUI 风格的 `diffusion_model.*` 键加载/保存 PEFT LoRA（Flux/Flux2/Lumina2/Z-Image 自动检测 ComfyUI 输入）

### 模型特定功能

- **Flux Kontext** - Flux 模型的编辑条件和图像到图像训练
- **PixArt 两阶段** - PixArt Sigma 的 eDiff 训练流程支持
- **Flow matching 模型** - 使用 beta/uniform 分布的高级调度
- **HiDream MoE** - 专家混合门控损失增强
- **T5 遮罩训练** - 增强 Flux 和兼容模型的精细细节
- **QKV 融合** - 内存和速度优化（Flux、Lumina2）
- **TREAD 集成** - 大多数模型的选择性 Token 路由
- **Wan 2.x I2V** - 高/低阶段预设加 2.1 时间嵌入回退（参见 Wan 快速入门）
- **无分类器引导** - 蒸馏模型的可选 CFG 重新引入

### 快速入门指南

所有支持的模型都有详细的快速入门指南：

- **[TwinFlow 少步数（RCGM）指南](/documentation/distillation/TWINFLOW.md)** - 启用 RCGM 辅助损失实现少步数/单步生成（flow 模型或通过 diff2flow 的扩散模型）
- **[Flux.1 指南](/documentation/quickstart/FLUX.md)** - 包括 Kontext 编辑支持和 QKV 融合
- **[Flux.2 指南](/documentation/quickstart/FLUX2.md)** - **新！** 带有 Mistral-3 文本编码器的最新超大 Flux 模型
- **[Z-Image 指南](/documentation/quickstart/ZIMAGE.md)** - 带助手适配器的 Base/Turbo LoRA + TREAD 加速
- **[ACE-Step 指南](/documentation/quickstart/ACE_STEP.md)** - **新！** 音频生成模型训练（文本到音乐）
- **[Chroma 指南](/documentation/quickstart/CHROMA.md)** - Lodestone 的 flow-matching transformer，带 Chroma 特定调度
- **[Stable Diffusion 3 指南](/documentation/quickstart/SD3.md)** - 全秩和 LoRA 训练，支持 ControlNet
- **[Stable Diffusion XL 指南](/documentation/quickstart/SDXL.md)** - 完整的 SDXL 训练流程
- **[Auraflow 指南](/documentation/quickstart/AURAFLOW.md)** - Flow-matching 模型训练
- **[PixArt Sigma 指南](/documentation/quickstart/SIGMA.md)** - 支持两阶段的 DiT 模型
- **[Sana 指南](/documentation/quickstart/SANA.md)** - 轻量级 flow-matching 模型
- **[Lumina2 指南](/documentation/quickstart/LUMINA2.md)** - 2B 参数 flow-matching 模型
- **[Kwai Kolors 指南](/documentation/quickstart/KOLORS.md)** - 基于 SDXL，使用 ChatGLM 编码器
- **[LongCat-Video 指南](/documentation/quickstart/LONGCAT_VIDEO.md)** - 使用 Qwen-2.5-VL 的 flow-matching 文本到视频和图像到视频
- **[LongCat-Video Edit 指南](/documentation/quickstart/LONGCAT_VIDEO_EDIT.md)** - 条件优先风格（图像到视频）
- **[LongCat-Image 指南](/documentation/quickstart/LONGCAT_IMAGE.md)** - 6B 双语 flow-matching 模型，使用 Qwen-2.5-VL 编码器
- **[LongCat-Image Edit 指南](/documentation/quickstart/LONGCAT_EDIT.md)** - 需要参考潜变量的图像编辑风格
- **[LTX Video 指南](/documentation/quickstart/LTXVIDEO.md)** - 视频扩散训练
- **[Hunyuan Video 1.5 指南](/documentation/quickstart/HUNYUANVIDEO.md)** - 8.3B flow-matching T2V/I2V，带 SR 阶段
- **[Wan Video 指南](/documentation/quickstart/WAN.md)** - 支持 TREAD 的视频 flow-matching
- **[HiDream 指南](/documentation/quickstart/HIDREAM.md)** - 带高级功能的 MoE 模型
- **[Cosmos2 指南](/documentation/quickstart/COSMOS2IMAGE.md)** - 多模态图像生成
- **[OmniGen 指南](/documentation/quickstart/OMNIGEN.md)** - 统一图像生成模型
- **[Qwen Image 指南](/documentation/quickstart/QWEN_IMAGE.md)** - 20B 参数大规模训练
- **[Stable Cascade Stage C 指南](/quickstart/STABLE_CASCADE_C.md)** - 带先验+解码器联合验证的先验 LoRA
- **[Kandinsky 5.0 Image 指南](/documentation/quickstart/KANDINSKY5_IMAGE.md)** - 使用 Qwen2.5-VL + Flux VAE 的图像生成
- **[Kandinsky 5.0 Video 指南](/documentation/quickstart/KANDINSKY5_VIDEO.md)** - 使用 HunyuanVideo VAE 的视频生成

---

## 硬件要求

### 通用要求

- **NVIDIA**：推荐 RTX 3080+（已测试至 H200）
- **AMD**：已验证 7900 XTX 24GB 和 MI300X（内存使用量高于 NVIDIA）
- **Apple**：M3 Max+ 配合 24GB+ 统一内存用于 LoRA 训练

### 按模型大小的内存指南

- **大型模型（12B+）**：全秩训练需要 A100-80G，LoRA/Lycoris 需要 24G+
- **中型模型（2B-8B）**：LoRA 需要 16G+，全秩训练需要 40G+
- **小型模型（<2B）**：大多数训练类型 12G+ 即可满足

**注意**：量化（int8/fp8/nf4）可显著降低内存需求。有关模型特定要求，请参阅各个[快速入门指南](#快速入门指南)。

## 安装

对于大多数用户，SimpleTuner 可以通过 pip 安装：

```bash
# 基础安装（仅 CPU PyTorch）
pip install simpletuner

# CUDA 用户（NVIDIA GPU）
pip install 'simpletuner[cuda]'

# ROCm 用户（AMD GPU）
pip install 'simpletuner[rocm]'

# Apple Silicon 用户（M1/M2/M3/M4 Mac）
pip install 'simpletuner[apple]'
```

有关手动安装或开发设置，请参阅[安装文档](/documentation/INSTALL.md)。

## 故障排除

通过在环境文件（`config/config.env`）中添加 `export SIMPLETUNER_LOG_LEVEL=DEBUG` 来启用调试日志，以获得更详细的信息。

对于训练循环的性能分析，设置 `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` 将显示带时间戳的信息，以突出显示配置中的任何问题。

有关可用选项的完整列表，请参阅[此文档](/documentation/OPTIONS.md)。
