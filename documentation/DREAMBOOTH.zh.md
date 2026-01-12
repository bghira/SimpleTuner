# Dreambooth（单主体训练）

## 背景

Dreambooth 指的是 Google 开发的一种技术，通过少量高质量图像对模型进行微调，将主体注入到模型中（[paper](https://dreambooth.github.io)）。

在微调语境下，Dreambooth 引入了新的技术以防止模型崩塌，例如过拟合或伪影。

### 正则化图像

正则化图像通常由你正在训练的模型生成，使用一个代表类别的 token。

它们不**一定**必须是模型生成的合成图像，但相比真实数据（例如真实人物照片），使用合成数据可能表现更好。

示例：如果你在训练男性主体，正则化数据可以是随机男性的照片或合成样本。

> 🟢 正则化图像可以作为单独的数据集配置，从而与训练数据均匀混合。

### 稀有 token 训练

原论文中有一个价值存疑的概念：反向搜索模型的 tokenizer 词表，找出训练关联极少的“稀有”字符串。

此后该想法经历了演进与争论，另一派选择用“足够相似”的名人名字进行训练，因为这样计算成本更低。

> 🟡 SimpleTuner 支持稀有 token 训练，但没有工具帮助你找到稀有 token。

### 先验保持损失

模型中存在所谓的“先验”，理论上可在 Dreambooth 训练中保留。但在 Stable Diffusion 的实验中似乎无效——模型会对自身知识过拟合。

> 🟢 （[#1031](https://github.com/bghira/SimpleTuner/issues/1031)）在训练 LyCORIS 适配器时，将 `is_regularisation_data` 设置到该数据集即可在 SimpleTuner 中使用先验保持损失。

### 掩码损失

图像掩码可以与图像数据成对定义。掩码的暗色区域会让损失计算忽略这些部分。

提供了一个 [脚本](/scripts/toolkit/datasets/masked_loss/generate_dataset_masks.py) 用于生成掩码，需提供 input_dir 与 output_dir：

```bash
python generate_dataset_masks.py --input_dir /images/input \
                      --output_dir /images/output \
                      --text_input "person"
```

不过该脚本没有掩码填充、模糊等高级功能。

在定义掩码数据集时：

- 每张图像都必须有掩码。不需要掩码时可使用全白图。
- 在掩码数据文件夹上设置 `dataset_type=conditioning`
- 在掩码数据集上设置 `conditioning_type=mask`
- 在图像数据集上设置 `conditioning_data=` 为掩码数据集的 `id`

```json
[
    {
        "id": "dreambooth-data",
        "type": "local",
        "dataset_type": "image",
        "conditioning_data": "dreambooth-conditioning",
        "instance_data_dir": "/training/datasets/test_datasets/dreambooth",
        "cache_dir_vae": "/training/cache/vae/sdxl/dreambooth-data",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an dreambooth",
        "metadata_backend": "discovery",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution_type": "pixel_area"
    },
    {
        "id": "dreambooth-conditioning",
        "type": "local",
        "dataset_type": "conditioning",
        "instance_data_dir": "/training/datasets/test_datasets/dreambooth_mask",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution_type": "pixel_area",
        "conditioning_type": "mask"
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "/training/cache/text/sdxl-base/masked_loss"
    }
]
```

## 设置

在继续 Dreambooth 专项配置前，必须先完成 [教程](TUTORIAL.md)。

如果进行 DeepFloyd 调优，建议访问 [此页面](DEEPFLOYD.md) 获取该模型的特定建议。

### 量化模型训练（仅限 LoRA/LyCORIS）

在 Apple 和 NVIDIA 系统上测试过，可使用 Hugging Face Optimum-Quanto 降低精度与显存需求。

在 SimpleTuner venv 中运行：

```bash
pip install optimum-quanto
```

可用精度等级取决于硬件与其能力。

- int2-quanto, int4-quanto, **int8-quanto**（推荐）
- fp8-quanto, fp8-torchao（仅 CUDA >= 8.9，如 4090 或 H100）
- nf4-bnb（低显存用户所需）

在 config.json 中修改或添加以下值：
```json
{
    "base_model_precision": "int8-quanto",
    "text_encoder_1_precision": "no_change",
    "text_encoder_2_precision": "no_change",
    "text_encoder_3_precision": "no_change"
}
```

数据加载器配置 `multidatabackend-dreambooth.json` 大致如下：

```json
[
    {
        "id": "subjectname-data-512px",
        "type": "local",
        "instance_data_dir": "/training/datasets/subjectname",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "subjectname",
        "cache_dir_vae": "/training/vae_cache/subjectname",
        "repeats": 100,
        "crop": false,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "minimum_image_size": 192
    },
    {
        "id": "subjectname-data-1024px",
        "type": "local",
        "instance_data_dir": "/training/datasets/subjectname",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "subjectname",
        "cache_dir_vae": "/training/vae_cache/subjectname-1024px",
        "repeats": 100,
        "crop": false,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "minimum_image_size": 768
    },
    {
        "id": "regularisation-data",
        "type": "local",
        "instance_data_dir": "/training/datasets/regularisation",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "a picture of a man",
        "cache_dir_vae": "/training/vae_cache/regularisation",
        "repeats": 0,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "minimum_image_size": 192,
        "is_regularisation_data": true
    },
    {
        "id": "regularisation-data-1024px",
        "type": "local",
        "instance_data_dir": "/training/datasets/regularisation",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "a picture of a man",
        "cache_dir_vae": "/training/vae_cache/regularisation-1024px",
        "repeats": 0,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "minimum_image_size": 768,
        "is_regularisation_data": true
    },
    {
        "id": "textembeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/training/text_cache/sdxl_base"
    }
]
```

为简化单主体训练，做了以下调整：

- 将两个数据集各配置两次，共四个数据集。正则化数据是可选的，训练可能在不使用正则化时效果更好。需要时可从列表中移除。
- 分辨率采用 512px 与 1024px 混合分桶，有助于提升训练速度与收敛
- 最小图像尺寸设为 192px 或 768px，以允许对部分低分辨率但重要的图像进行上采样
- `caption_strategy` 为 `instanceprompt`，因此每张图像的字幕都使用 `instance_prompt`
  - **注记:** 使用 instance prompt 是 Dreambooth 的传统方法，但更短的字幕可能效果更好。如果模型无法泛化，可考虑使用字幕。

### 正则化数据集注意事项

对于正则化数据集：

- 将 Dreambooth 主体数据的 `repeats` 设得很高，使其总样本数（`repeats` 倍）超过正则化集
  - 如果正则化集有 1000 张，而训练集只有 10 张，建议 repeats 至少 100
- 提高 `minimum_image_size` 以避免引入过多低质量伪影
- 使用更描述性的字幕可能有助于避免遗忘。从 `instanceprompt` 切换为 `textfile` 等策略需要为每张图像创建 `.txt` 文件。
- 设置 `is_regularisation_data`（或美式拼写 `is_regularization_data`）后，该数据集会输入到基础模型中，生成用于 LyCORIS 学生模型损失目标的预测。
  - 注意当前仅对 LyCORIS 适配器生效。

## 选择 instance prompt

如前所述，Dreambooth 最初的重点是选择稀有 token 进行训练。

另一种方式是使用主体真实姓名或“足够相似”的名人姓名。

多次训练实验表明，若用真实姓名生成的结果相似度不足，选择“足够相似”的名人姓名往往更有效。

# 计划采样（Rollout）

在 Dreambooth 等小数据集训练时，模型可能很快过拟合于训练中加入的“完美”噪声，导致**暴露偏差**：模型学会去噪完美输入，但在推理时面对自身略有偏差的输出就失败。

**计划采样（Rollout）** 通过在训练循环中偶尔让模型生成自身的噪声潜变量来解决该问题。模型不再只在纯高斯噪声 + 信号上训练，而会在包含自身先前误差的“rollout”样本上训练，从而学会自我纠错，使主体生成更稳定、更鲁棒。

> 🟢 此功能是实验性的，但对于小数据集常见的过拟合或“烤糊”问题强烈推荐。
> ⚠️ 启用 rollout 会增加计算成本，因为训练循环中需要额外的推理步数。

要启用，请在 `config.json` 中添加：

```json
{
  "scheduled_sampling_max_step_offset": 10,
  "scheduled_sampling_probability": 1.0,
  "scheduled_sampling_ramp_steps": 1000,
  "scheduled_sampling_sampler": "unipc"
}
```

*   `scheduled_sampling_max_step_offset`: 生成步数。小值（如 5-10）通常足够。
*   `scheduled_sampling_probability`: 应用该技术的频率（0.0 到 1.0）。
*   `scheduled_sampling_ramp_steps`: 在前 N 步逐步提高概率，避免早期训练不稳定。

# 指数移动平均（EMA）

可以并行训练第二个模型，几乎没有额外成本——仅占用系统内存（默认），不会额外消耗显存。

在配置文件中设置 `use_ema=true` 即可启用。

# CLIP 分数跟踪

若要通过评估对模型性能打分，请参阅 [此文档](evaluation/CLIP_SCORES.md) 了解 CLIP 分数的配置与解读。

# 稳定评估损失

若要使用稳定 MSE 损失评估模型性能，请参阅 [此文档](evaluation/EVAL_LOSS.md) 了解评估损失的配置与解读。

# 验证预览

SimpleTuner 支持在生成时使用 Tiny AutoEncoder 模型流式输出验证预览。该功能可通过 Webhook 回调实时展示验证图像生成过程，而无需等待最终生成完成。

## 启用验证预览

在 `config.json` 中加入以下内容：

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

## 要求

- 支持 Tiny AutoEncoder 的模型系列（Flux、SDXL、SD3 等）
- 可接收预览图像的 Webhook 配置
- 必须启用验证（`validation_disable` 不能设为 true）

## 配置选项

- `--validation_preview`: 启用/禁用预览功能（默认: false）
- `--validation_preview_steps`: 控制采样过程中预览解码频率（默认: 1）
  - 设为 1 可在每个采样步输出预览
  - 设为更高值（如 3 或 5）可减少 Tiny AutoEncoder 解码开销

## 示例

当 `validation_num_inference_steps=20` 且 `validation_preview_steps=5` 时，每次验证生成将在第 5、10、15、20 步收到预览图像。

# Refiner 调优

如果你喜欢 SDXL refiner，可能会发现它会“破坏”你的 Dreambooth 生成结果。

SimpleTuner 支持使用 LoRA 与全秩训练 SDXL refiner。

需要注意以下事项：
- 图像应尽可能高质量
- 文本嵌入不能与基础模型共享
- VAE 嵌入**可以**与基础模型共享

需要在 dataloader 配置 `multidatabackend.json` 中更新 `cache_dir`：

```json
[
    {
        "id": "textembeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/training/text_cache/sdxl_refiner"
    }
]
```

若想针对特定美学分数训练，可在 `config/config.json` 中添加：

```bash
"--data_aesthetic_score": 5.6,
```

将 **5.6** 替换为目标分数，默认值为 **7.0**。

> ⚠️ 训练 SDXL refiner 时，验证提示词会被忽略。将改为对数据集中随机图像进行精修。
