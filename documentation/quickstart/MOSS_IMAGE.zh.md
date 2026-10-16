# MOSS-Image 快速入门

使用自己的图片训练 MOSS-Image LoRA。LoRA 是一个小型适配器，可以学习主体或风格，同时保留原始检查点。

使用 [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) 时，设置 `model_family: "moss_image"`、`model_flavour: "base"` 和 `model_type: "lora"`。若使用经过精选数据微调的 [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)，则设置 `model_flavour: "sft"`。两者共享 5B 图像 Transformer、Qwen3.5-9B 文本编码器和 FLUX.2 VAE，采用 [Apache-2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/LICENSE) 许可证。训练期间冻结文本编码器。

## 选择 VRAM 预设

根据 GPU 显存选择配方。所有档位都使用 1024×1024 等效面积训练、保留宽高比，并启用 rank 32 BF16 LoRA 和 REPA。

| GPU 显存 | Base 示例 | SFT 示例 | 每批图片数 | `gradient_checkpointing_interval` |
| --- | --- | --- | ---: | ---: |
| 24 GB | `moss_image-24g.peft-lora` | `moss_image-sft-24g.peft-lora` | 1 | 1 |
| 32 GB | `moss_image-32g.peft-lora` | `moss_image-sft-32g.peft-lora` | 1 | 2 |
| 48 GB | `moss_image-48g.peft-lora` | `moss_image-sft-48g.peft-lora` | 2 | 2 |
| 80 GB | `moss_image-80g.peft-lora` | `moss_image-sft-80g.peft-lora` | 4 | 2 |
| 144 GB | `moss_image-144g.peft-lora` | `moss_image-sft-144g.peft-lora` | 8 | 2 |

默认的 `moss_image.peft-lora` 和 `moss_image-sft.peft-lora` 使用 48 GB 设置。批次大小只是起点：描述较长、启用额外训练功能，或宽高比桶中的图片数少于批次大小时，请减小 `train_batch_size`。144 GB 档位面向 H200 级别 GPU。

## 安装 SimpleTuner

以下 CUDA 配方使用 Python 3.13 和 NVIDIA GPU。其他平台和 CUDA 版本请参阅[安装指南](../INSTALL.md)。

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda,attention-cute]'
```

## 准备图片和描述

创建 `datasets/moss-image`，放入图片及同名 `.txt` 描述文件。使用细节足以支持约一百万像素训练的清晰图片。正方形对应 1024×1024，宽图和长图保留宽高比。

```text
datasets/moss-image/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

为主体选择 `TOK` 等标识词，并写入该主体的每条描述。描述可见主体、构图、光线、环境和风格。项目建议先使用英语。准确记录图中文字，包括换行。例如：`A photograph of TOK, a small blue ceramic owl on a wooden table beside a window.`

## 设置训练配置

复制 48 GB Base 配方及示例提示词，将 `48g` 替换为所选档位。使用 SFT 时，把命令中的 `moss_image-48g.peft-lora` 换成 `moss_image-sft-48g.peft-lora`。

```bash
mkdir -p config/examples datasets/moss-image
cp -R simpletuner/examples/moss_image-48g.peft-lora config/examples/
cp simpletuner/examples/moss_image-prompts.json config/examples/
cp config/examples/moss_image-48g.peft-lora/config.json config/config.json
```

配方使用 rank 32 BF16 LoRA、`1e-4` 学习率、梯度检查点、REPA、4,000 个优化器步骤以及 1024 级别分辨率。打开 `config/config.json`，在原有配置内修改以下项目：

```json
{
  "output_dir": "output/my-moss-lora",
  "validation_prompt": "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
  "user_prompt_library": null
}
```

复制的 `dataloader.json` 指向 `datasets/moss-image`，读取配套文本并缓存潜变量和文本嵌入。`resolution_type: "area"` 与 `resolution: 1.048576` 表示 1024×1024 等效面积。图片尺寸对齐至 16 的倍数。更多数据集和正则化图片设置请参阅[数据加载器指南](../DATALOADER.md)。

## 训练并比较预览

激活虚拟环境，在 SimpleTuner 目录运行 `simpletuner train`。首次运行会下载模型、REPA 编码器并建立缓存。配方每 250 步保存检查点和验证图片。比较主体学习、构图和细节，按数据集调整 `max_train_steps`。

```bash
simpletuner train
```

## 采样设置

从 **50 步**、**CFG 3** 和 **1024×1024** 验证开始。采样的 `moss_image_validation_shift` 默认为 **5**；训练使用原始未偏移时间网格。`moss_image_uncond_skip_layers` 默认为 **[22]**，在负向 CFG 分支跳过第 22 层。设为 `[]` 可使用标准 CFG。正向分支的额外跳层引导由 `validation_guidance_skip_layers`、`validation_guidance_skip_scale` 及起止设置控制，参见 [OPTIONS](../OPTIONS.md)。Python 生成时设置 `use_cfg_zero_star=True` 可启用 CFG-Zero*。

## 采样过程预览

添加以下设置，可在 WebUI 查看去噪进度图片。预览使用 TAEF2，最终验证图片使用完整 VAE。CLI 可通过已配置的 webhook 发送预览。

```json
{
  "validation_preview": true,
  "validation_preview_steps": 5
}
```

## 内存与训练选项

配方已按所选 VRAM 档位设置 `train_batch_size`。数据集较小，或描述和额外功能需要更多显存时，请减小该值。 `gradient_checkpointing_backend` 支持 `torch`、`unsloth`、`torch-ffn` 和 `unsloth-ffn`；interval 与 `gradient_checkpointing_segment_stride` 控制覆盖范围。Musubi 通过 `musubi_blocks_to_swap` 在 CPU 与 GPU 之间移动块。RamTorch 使用 `ramtorch: true`，可通过 `ramtorch_transformer_percent` 选择卸载比例。这些方案占用系统内存并增加传输开销，请一次配置一种。参见 [OPTIONS](../OPTIONS.md)。

CUDA 安装包含 CuTE 注意力依赖。训练时，默认的 `attention_mechanism: "kohaku-fa-auto"` 在 Ada 和 Hopper GPU 上选择 CuTE，在其他架构上使用原生 SDPA。启用 `gradient_checkpointing_offload_attention: true` 可卸载为反向传播保存的注意力张量；`gradient_checkpointing_offload_prefetch: true` 启用传输预取。其他注意力后端请参阅 [OPTIONS](../OPTIONS.md)。

配方启用 [REPA](../experimental/IMAGE_REPA.md)。如需使较后层的图像特征与前层对齐，添加 [LayerSync](../experimental/LAYERSYNC.md)：

```json
{
  "layersync_enabled": true,
  "layersync_student_block": 20,
  "layersync_teacher_block": 8,
  "layersync_lambda": 0.2
}
```

后训练流程请参考 [AnyFlow](../quickstart/ANYFLOW.md) 或 [TwinFlow](../distillation/TWINFLOW.md)。AnyFlow 请保持 `lora_dropout: 0.0`。在 Python 中加载其 LoRA 时，将训练设置传给 `load_lora_weights`，例如 `flowmap_deltatime_type="r", flowmap_gate_value=0.25`。

## 提示词优化与布局

MOSS-Image 接受详细描述、标签序列和 JSON 布局。项目的[提示词优化技能](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer)可帮助智能体编写描述和布局。将完成的文本用于图片描述或生成提示词。[项目指南](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)提供了描述建议。

布局描述场景、文字和物体位置。坐标使用像素单位的 `[y_min, x_min, y_max, x_max]`，以画布中心为原点。1024×1024 画布的范围为 −512 到 +512。用 `json.dumps(layout, ensure_ascii=False)` 序列化后作为提示词传入，生成尺寸应与布局一致。

```json
{
  "summary": {"caption": "A cream travel poster with the title EXPLORE above a blue mountain."},
  "items": [
    {"label": "text", "text": "EXPLORE", "caption": "Large navy title", "bbox": [-416, -384, -288, 384]},
    {"label": "object", "caption": "Blue mountain", "bbox": [-192, -288, 384, 288]}
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

## 使用 LoRA 生成图片

最终适配器为输出目录中的 `pytorch_lora_weights.safetensors`。使用训练时相同的 Base 或 SFT 检查点加载，在提示词中保留主体标识词并描述希望生成的场景：

```python
import torch
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline

pipe = MOSSImagePipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-Image-Base", torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("output/my-moss-lora")
pipe.to("cuda")
image = pipe(
    "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
    height=1024, width=1024, num_inference_steps=50, guidance_scale=3.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("my-moss-image.png")
```
