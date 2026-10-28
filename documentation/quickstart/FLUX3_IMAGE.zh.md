# FLUX.3 Image Klein 快速入门

本指南带你使用自己的图像训练 FLUX.3 Image Klein LoRA。LoRA 是一种小型适配器，可让模型学习特定主体或风格，同时将原始模型权重单独保留。

从 `model_family: "flux3_image"`、`model_flavour: "klein-base"` 和 `model_type: "lora"` 开始。这会加载 [black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base)。下载前请访问模型页面，接受访问条件，并阅读 [FLUX [Kom]munity 许可证](https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/LICENSE)。

## 安装

准备 Python 3.13、Git，以及用于模型、训练图像和缓存的磁盘空间。以下命令适用于 NVIDIA CUDA GPU。其他平台及 CUDA 13 安装方法见[安装指南](../INSTALL.zh.md)。

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda]'
hf auth login
```

后续命令均在 `SimpleTuner` 目录中执行，并保持虚拟环境激活。`hf auth login` 提示输入令牌时，使用具有模型读取权限的 Hugging Face 令牌。

## 选择适合显卡的配置

显存（VRAM）决定了每次能同时处理多少张图像。按显存容量选择示例。24 和 32 GB 的 Base 配置使用 512 分辨率；更大容量的配置还包含 1024 分辨率数据集。

| 显存 | Base 示例 | 512 / 1024 每批图像数 |
| --- | --- | --- |
| 24 GB | `flux3_image/klein-base/24g/peft-lora` | 3 / — |
| 32 GB | `flux3_image/klein-base/32g/peft-lora` | 4 / — |
| 48 GB | `flux3_image/klein-base/48g/peft-lora` | 9 / 2 |
| 80 GB | `flux3_image/klein-base/80g/peft-lora` | 18 / 5 |
| 144 GB | `flux3_image/klein-base/144g/peft-lora` | 36 / 10 |

蒸馏版 `klein` 使用以下配方。请在后面的配置复制命令中替换完整的示例名称。训练尺度表示正方形等效面积，图片保留原始宽高比。

| GPU 显存 | 蒸馏版示例 | 训练尺度 | 每批图片数 |
| --- | --- | --- | ---: |
| 24 GB | `flux3_image/klein/24g/peft-lora` | 768px | 1 |
| 80 GB | `flux3_image/klein/80g/peft-lora` | 1024px | 4 |

表中的批大小可作为起点。较长描述和参考图会增加显存消耗；显存不足时减小 `train_batch_size`。示例采用 rank 32 LoRA、`1e-4` 学习率和 REPA，通过比较图像特征辅助训练。随附 dataloader 将主体图像与正则化图像结合，帮助保留模型的通用图像知识。

## 准备图像与描述

在 SimpleTuner 目录中创建 `datasets/my-concept` 并放入训练图像。每张图像都配一个同名文本文件，写入对应描述：

```text
datasets/my-concept/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

训练主体 LoRA 时，选择 `TOK` 这样的独特词，并在描述中一致使用。描述可见的主体、场景和姿态。例如 `image01.txt` 可以写：

```text
一张 TOK 的照片：这是一只带蓝色羽毛的小陶瓷猫头鹰，放在木桌上。
```

## 设置训练配置

复制所选配方。以下命令使用 `flux3_image/klein-base/48g/peft-lora`，请将完整示例名称替换为所选的 Base 或蒸馏版配方：

```bash
mkdir -p config/examples/_shared
mkdir -p config/examples/flux3_image/klein-base/48g
cp -R simpletuner/examples/flux3_image/klein-base/48g/peft-lora config/examples/flux3_image/klein-base/48g/
mkdir -p config/examples/_shared
cp simpletuner/examples/_shared/flux3_image-prompts.json config/examples/_shared/
cp config/examples/flux3_image/klein-base/48g/peft-lora/config.json config/config.json
```

用文本编辑器打开 `config/config.json`。首次使用上述本地图像时，在现有配置中修改以下值：

```json
{
  "data_backend_config": "config/dataloader.json",
  "output_dir": "output/my-flux3-lora",
  "train_batch_size": 1
}
```

批大小为 1 时，少量图像也能开始训练。首次运行成功且图像和显存充足后再逐步增加。`max_train_steps` 控制训练时长。编辑 `config/examples/_shared/flux3_image-prompts.json`，加入包含主体词的提示词，用于生成训练进度图。

将以下内容保存为 `config/dataloader.json`。它告诉 SimpleTuner 从哪里读取图像、如何读取描述，以及缓存放在哪里：

```json
[
  {
    "id": "my-concept",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/my-concept",
    "caption_strategy": "textfile",
    "resolution": 0.589824,
    "resolution_type": "area",
    "crop": false,
    "cache_dir_vae": "cache/vae/flux3-image/my-concept",
    "text_embeds": "text"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux3-image/my-concept"
  }
]
```

此本地数据集使用 `resolution: 0.589824` 和 `resolution_type: "area"`，对应 768×768 等效面积。80 GB 蒸馏版配方的 1024 尺度数据集使用 `1.048576`。

`resolution` 控制训练尺寸；图像尺寸会对齐到 16 的倍数。缓存保存可重复使用的图像潜变量和文本嵌入，加快后续启动。不同模型家族使用独立缓存目录。添加更多数据集或正则化图像的方法见[数据加载指南](../DATALOADER.zh.md)。

## 可选：使用提示词扩写模型

在 `config/config.json` 中添加以下设置，通过 [FLUX.3 Image Klein 提示词扩写模型](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)扩写简短的训练描述和验证提示词：

```json
{
  "flux3_prompt_upsample": true
}
```

此选项默认关闭。启用后，SimpleTuner 会下载独立的 Qwen3.5-9B 模型（权重约 19 GB），在缓存文本嵌入之前扩写正向提示词。编辑任务会使用配对参考图。负向提示词、dropout 空描述和原始描述文件均保持不变。

请预留额外系统内存和预处理时间。扩写模型与文本编码器轮流使用 GPU，生成的嵌入会在训练中复用。SimpleTuner 将扩写嵌入与原始嵌入分别保存，再次选择相同参考图时可复用缓存结果。

## 开始训练

在 SimpleTuner 目录中运行：

```bash
simpletuner train
```

首次运行时，SimpleTuner 会先下载模型、读取图像并建立文本与图像缓存，然后开始训练。编译可能增加启动时间，请保持进程运行。进度图和检查点保存在 `output/my-flux3-lora`；可通过进度图观察模型对主体的学习效果。

`klein-base` 使用 50 个验证步和 3.5 引导强度。训练蒸馏版时选择 `model_flavour: "klein"`，它加载 `black-forest-labs/flux-3-image-klein`，使用 16 个验证步和 1 的引导强度。开始前先取得该仓库的访问权限。

## 编辑与参考图训练

两个变体均支持可选参考图。目标数据集保存期望的编辑结果和指令描述；条件数据集保存源图或参考图。将下列 JSON 单独保存，在 `data_backend_config` 中指定路径，并从 `train_batch_size: 1` 开始：

```json
[
  {
    "id": "edits",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/edits",
    "caption_strategy": "textfile",
    "resolution": 512,
    "resolution_type": "pixel",
    "conditioning_data": ["references"],
    "cache_dir_vae": "cache/vae/flux3-image/edits",
    "text_embeds": "text"
  },
  {
    "id": "references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "datasets/references",
    "conditioning_type": "reference_loose",
    "resolution": 512,
    "resolution_type": "pixel",
    "cache_dir_vae": "cache/vae/flux3-image/references"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux3-image-edit"
  }
]
```

目标和参考目录中的文件名必须对应，例如 `edits/sample.png`、`edits/sample.txt` 和 `references/sample.png`。`reference_loose` 允许不同尺寸和宽高比；`reference_strict` 用于对齐裁剪。在 `conditioning_data` 中添加更多数据集 ID，每个样本最多十张参考图；设置 `conditioning_multidataset_sampling: "combined"` 同时提供全部参考。省略 `conditioning_data` 即为文生图训练。参考、目标以及其他模型家族必须使用独立缓存。

使用 `conditioning_multidataset_sampling: "random"` 时，每个训练样本使用一个参考数据集及其描述。请为每个参考数据集设置 `caption_strategy: "textfile"`，并将编辑指令写入对应的 `.txt` 文件，例如 `references/sample.txt`。

## 用 LoRA 生成图像

找到包含 `pytorch_lora_weights.safetensors` 的检查点目录，用它替换下方的 `output/my-flux3-lora/checkpoint-1000`，并使用与训练相同的模型变体。此示例使用 base 和 CFG-Zero-Star 引导：

```python
import torch
from simpletuner.helpers.models.flux3_image.pipeline import Flux3ImagePipeline

pipe = Flux3ImagePipeline.from_pretrained(
    "black-forest-labs/flux-3-image-klein-base", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("output/my-flux3-lora/checkpoint-1000")
image = pipe(
    "A photo of TOK, a small ceramic owl, sitting on a wooden shelf",
    num_inference_steps=50,
    guidance_scale=3.5,
    use_cfg_zero_star=True,
).images[0]
image.save("my-flux3-image.png")
```

## 助手 LoRA

使用 `model_flavour: "klein"` 训练 LoRA 时，会自动加载[已发布的训练助手](https://huggingface.co/SimpleTuner/flux-3-klein-image-training-assistant)作为冻结的适配器。训练时 `assistant_lora_strength` 默认为 `1.0`，验证时 `assistant_lora_inference_strength` 默认为 `0.0`。设置 `disable_assistant_lora: true` 可关闭助手。使用其他适配器时，设置 `assistant_lora_path`，并按需指定 `assistant_lora_weight_name`。

如需训练自己的助手，请选择 `flux3_image/klein/default/assistant-lora.peft-lora`。此配方会关闭已有助手，并使用模型生成的样本学习。完成后，在蒸馏版 LoRA 配置中将 `assistant_lora_path` 设为保存的适配器目录。

更多训练设置见[选项参考](../OPTIONS.zh.md)。启用 TwinFlow 时同时设置 `twinflow_enabled: true` 和 `use_ema: true`；EMA 维护该训练方法使用的教师模型。
