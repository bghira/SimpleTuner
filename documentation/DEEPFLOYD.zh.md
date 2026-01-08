# DeepFloyd IF

> 训练 DeepFloyd 即使使用 LoRA 也至少需要 24G 显存。本指南聚焦 4 亿参数的基础模型，但 4.3B 的 XL 版本也可按相同指引训练。

## 背景

2023 年春，StabilityAI 发布了一个名为 DeepFloyd 的级联像素扩散模型。
![](https://tripleback.net/public/deepfloyd.png)

与 Stable Diffusion XL 简要对比：
- 文本编码器
  - SDXL 使用两个 CLIP 编码器：“OpenCLIP G/14” 和 “OpenAI CLIP-L/14”
  - DeepFloyd 使用一个自监督 Transformer，Google 的 T5 XXL
- 参数规模
  - DeepFloyd 有多个规模：400M、900M、4.3B。规模越大训练成本越高。
  - SDXL 只有一个规模，约 3B 参数。
  - DeepFloyd 的文本编码器本身有 11B 参数，最大配置约 15.3B 参数。
- 模型数量
  - DeepFloyd 有 **三** 个阶段：64px -> 256px -> 1024px
    - 每个阶段都完成其去噪目标
  - SDXL 有 **两** 个阶段（包含 refiner），1024px -> 1024px
    - 每个阶段只部分完成去噪目标
- 设计
  - DeepFloyd 的三个模型逐步提升分辨率与细节
  - SDXL 的两个模型负责细节与构图

两种模型的第一阶段都定义了大部分图像构图（大物体/阴影的位置）。

## 模型评估

以下是使用 DeepFloyd 进行训练或推理时的预期表现。

### 美学

与 SDXL 或 Stable Diffusion 1.x/2.x 相比，DeepFloyd 的美学介于 Stable Diffusion 2.x 与 SDXL 之间。


### 缺点

该模型不太流行，原因如下：

- 推理时所需显存高于其他模型
- 训练时所需显存远高于其他模型
  - 全量 U-Net 微调需要超过 48G 显存
  - rank-32、batch-4 的 LoRA 也需约 24G 显存
  - 文本嵌入缓存极其庞大（每个多 MB，而 SDXL 的双 CLIP 嵌入仅为数百 KB）
  - 文本嵌入缓存生成很慢（A6000 non-Ada 上当前约每秒 9–10 个）
- 默认美学效果弱于其他模型（类似训练原版 SD 1.5）
- 推理时需要微调或加载 **三个** 模型（加上文本编码器则为四个）
- StabilityAI 的承诺与实际使用体验不一致（期望被高估）
- DeepFloyd-IF 许可对商业用途有限制
  - 这并未影响 NovelAI 权重，但其确实是非法泄露的。考虑到其他更严重的问题，商业许可限制显得像是一个方便的理由。

### 优点

同时，DeepFloyd 也有一些容易被忽视的优势：

- 推理时 T5 文本编码器对世界有较强理解能力
- 可直接训练超长字幕
- 第一阶段约为 64x64 像素面积，可在多种纵横比分辨率上训练
  - 低分辨率意味着 DeepFloyd 曾是唯一能够在 LAION-A 上对**所有**图片进行训练的模型（LAION 中小于 64x64 的图像很少）
- 各阶段可独立调优，关注不同目标
  - 第一阶段可聚焦构图，后续阶段聚焦放大后的细节
- 尽管训练内存占用更大，但训练速度很快
  - 吞吐高，stage 1 调优时每小时样本数较高
  - 学习速度快于 CLIP 等价模型，习惯 CLIP 训练的人可能需要适应
    - 也就是说，你需要调整对学习率与训练计划的预期
- 无需 VAE，训练样本直接下采样到目标尺寸，U-Net 直接消费像素
- 支持 ControlNet LoRA 以及许多在典型线性 CLIP U-Net 上可用的技巧

## LoRA 微调

> ⚠️ 由于 DeepFloyd 最小 400M 模型的全量 U-Net 反向传播计算需求过高，尚未测试。本文将使用 LoRA，但全量 U-Net 微调理论上也应可用。

这些说明假设你对 SimpleTuner 基本使用有所了解。新手建议从支持更完善的模型开始，例如 [Kwai Kolors](quickstart/KOLORS.md)。

若你确实想训练 DeepFloyd，需要使用 `model_flavour` 配置选项来表明训练的模型。

### config.json

```bash
"model_family": "deepfloyd",

# Possible values:
# - i-medium-400m
# - i-large-900m
# - i-xlarge-4.3b
# - ii-medium-450m
# - ii-large-1.2b
"model_flavour": "i-medium-400m",

# DoRA isn't tested a whole lot yet. It's still new and experimental.
"use_dora": false,
# Bitfit hasn't been tested for efficacy on DeepFloyd.
# It will probably work, but no idea what the outcome is.
"use_bitfit": false,

# Highest learning rate to use.
"learning_rate": 4e-5,
# For schedules that decay or oscillate, this will be the end LR or the bottom of the valley.
"lr_end": 4e-6,
```

- `model_family` 为 deepfloyd
- `model_flavour` 指向 Stage I 或 II
- `resolution` 为 `64`，`resolution_type` 为 `pixel`
- `attention_mechanism` 可设为 `xformers`，但 AMD 与 Apple 用户无法设置，意味着需要更多显存。
  - **注记** ~~Apple MPS 当前存在阻止 DeepFloyd 调优的 bug。~~ 从 Pytorch 2.6 或更早版本起，Apple MPS 上的 stage I 与 II 均可训练。

为了更彻底的验证，`validation_resolution` 可设置为：

- `validation_resolution=64` 将生成 64x64 的正方形图像。
- `validation_resolution=96x64` 将生成 3:2 的宽屏图像。
- `validation_resolution=64,96,64x96,96x64` 将在每次验证中生成 4 张图像：
  - 64x64
  - 96x96
  - 64x96
  - 96x64

### multidatabackend_deepfloyd.json

接下来配置 DeepFloyd 训练的数据加载器。该配置与 SDXL 或传统模型的数据集配置几乎相同，重点在分辨率参数。

```json
[
    {
        "id": "primary-dataset",
        "type": "local",
        "instance_data_dir": "/training/data/primary-dataset",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "random",
        "resolution": 64,
        "resolution_type": "pixel",
        "minimum_image_size": 64,
        "maximum_image_size": 256,
        "target_downsample_size": 128,
        "prepend_instance_prompt": false,
        "instance_prompt": "Your Subject Trigger Phrase or Word",
        "caption_strategy": "instanceprompt",
        "repeats": 1
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "disable": false,
        "type": "local",
        "cache_dir": "/training/cache/deepfloyd/text/dreambooth"
    }
]
```

以上为 DeepFloyd 的基础 Dreambooth 配置：

- `resolution` 与 `resolution_type` 分别设为 `64` 与 `pixel`
- `minimum_image_size` 降至 64 像素，避免误将更小图像上采样
- `maximum_image_size` 设为 256 像素，确保大图不会以超过 4:1 的比例被裁剪，从而造成严重的场景信息丢失
- `target_downsample_size` 设为 128 像素，使大于 256 像素的图像在裁剪前先缩放到 128 像素

注记：图像每次会按 25% 逐步下采样，避免尺寸变化过大导致细节被不正确地平均。

## 运行推理

目前 SimpleTuner 工具包中没有 DeepFloyd 专用的推理脚本。

除内置验证流程外，你可以参考 Hugging Face 的 [此文档](https://huggingface.co/docs/diffusers/v0.23.1/en/training/dreambooth#if)，其中包含推理示例：

```py
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", use_safetensors=True)
pipe.load_lora_weights("<lora weights path>")
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
```

> ⚠️ `DiffusionPipeline.from_pretrained(...)` 的第一个值设为 `IF-I-M-v1.0`，但你必须将其更新为训练 LoRA 所用的基础模型路径。

> ⚠️ Hugging Face 的所有建议并不都适用于 SimpleTuner。例如，借助高效预缓存与纯 bf16 优化器状态，DeepFloyd stage I 的 LoRA 可在 22G 显存下调优，而 Diffusers 的 Dreambooth 示例需要 28G。

## 微调超分阶段 II 模型

DeepFloyd 的 stage II 模型接收约 64x64（或 96x64）输入，并使用 `VALIDATION_RESOLUTION` 设置生成相应的放大图像。

评估图像会从数据集中自动收集，`--num_eval_images` 指定从每个数据集中挑选的放大图像数量。目前图像是随机选择的，但每次会话保持一致。

为避免误用错误尺寸，还设置了额外检查。

要训练 stage II，只需按上述步骤操作，并将 `MODEL_TYPE` 从 `deepfloyd-lora` 改为 `deepfloyd-stage2-lora`：

```bash
export MODEL_TYPE="deepfloyd-stage2-lora"
```
