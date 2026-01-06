# ControlNet 训练指南

## 背景

ControlNet 模型能够完成多种任务，这取决于训练时提供的条件数据。

最初训练它们需要非常高的资源消耗，但现在可以使用 PEFT LoRA 或 Lycoris，在资源大幅减少的情况下完成相同的任务。

示例（来自 Diffusers 的 ControlNet 模型卡）：

![示例](https://tripleback.net/public/controlnet-example-1.png)

左侧是作为条件输入提供的「Canny 边缘图」。右侧是 ControlNet 从基础 SDXL 模型引导得到的输出。

以这种方式使用时，提示词几乎不负责构图，只是补充细节。

## ControlNet 训练过程

在训练开始时，ControlNet 几乎没有控制力：

![示例](https://tripleback.net/public/controlnet-example-2.png)
(_在 Stable Diffusion 2.1 模型上只训练 4 步的 ControlNet_)

羚羊的提示词仍然主导构图，ControlNet 的条件输入被忽略。

随着训练推进，条件输入应当逐渐被遵循：

![示例](https://tripleback.net/public/controlnet-example-3.png)
(_在 Stable Diffusion XL 模型上只训练 100 步的 ControlNet_)

此时开始出现少量 ControlNet 影响的迹象，但结果非常不稳定。

要达到可用效果需要远多于 100 步。

## 数据加载器配置示例

数据加载器配置与典型的文本到图像数据集配置非常接近：

- 主图像数据是 `antelope-data` 数据集
  - 现在需要设置 `conditioning_data`，并指向与该数据集配对的条件数据 `id`。
  - 基础数据集的 `dataset_type` 应设置为 `image`。
- 配置一个次要数据集 `antelope-conditioning`
  - 名称并不重要，这里只是为了示例说明才使用 `-data` 和 `-conditioning`。
  - 将 `dataset_type` 设置为 `conditioning`，告知训练器该数据用于评估和条件输入训练。
- 训练 SDXL 时，条件输入不会进行 VAE 编码，而是作为像素值直接送入模型训练。这意味着训练开始时不会额外花时间处理 VAE 嵌入。
- 训练 Flux、SD3、Auraflow、HiDream 或其他 MMDiT 模型时，条件输入会编码到潜变量中，并在训练过程中按需计算。
- 虽然这里都显式标注为 `-controlnet`，你仍可以复用常规全量/LoRA 调优所使用的文本嵌入。ControlNet 输入不会修改提示词嵌入。
- 使用宽高比分桶和随机裁剪时，条件样本会以与主图像样本相同的方式裁剪，无需额外处理。

```json
[
    {
        "id": "antelope-data",
        "type": "local",
        "dataset_type": "image",
        "conditioning_data": "antelope-conditioning",
        "instance_data_dir": "datasets/animals/antelope-data",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an antelope",
        "metadata_backend": "discovery",
        "minimum_image_size": 512,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "cache_dir_vae": "cache/vae/sdxl/antelope-data",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "cache_file_suffix": "controlnet-sdxl"
    },
    {
        "id": "antelope-conditioning",
        "type": "local",
        "dataset_type": "conditioning",
        "instance_data_dir": "datasets/animals/antelope-conditioning",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an antelope",
        "metadata_backend": "discovery",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "minimum_image_size": 512,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel_area",
        "cache_file_suffix": "controlnet-sdxl"
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/sdxl-base/controlnet"
    }
]
```

## 生成条件图像输入

SimpleTuner 对 ControlNet 的支持还很新，目前用于生成训练集的选项只有一个：

- [create_canny_edge.py](/scripts/toolkit/datasets/controlnet/create_canny_edge.py)
  - 这是一个非常基础的示例，用于生成 Canny 模型训练集。
  - 你需要修改脚本中的 `input_dir` 与 `output_dir`。

少于 100 张图片的小数据集大约需要 30 秒。

## 修改配置以训练 ControlNet 模型

仅配置数据加载器还不足以开始训练 ControlNet 模型。

在 `config/config.json` 中，需要设置以下值：

```bash
"model_type": 'lora',
"controlnet": true,

# You may have to reduce TRAIN_BATCH_SIZE and RESOLUTION more than usual
"train_batch_size": 1
```

最终配置会类似这样：

```json
{
    "aspect_bucket_rounding": 2,
    "caption_dropout_probability": 0.1,
    "checkpoint_step_interval": 100,
    "checkpoints_total_limit": 5,
    "controlnet": true,
    "data_backend_config": "config/controlnet-sdxl/multidatabackend.json",
    "disable_benchmark": false,
    "gradient_checkpointing": true,
    "hub_model_id": "simpletuner-controlnet-sdxl-lora-test",
    "learning_rate": 3e-5,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 100,
    "max_train_steps": 1000,
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "sdxl",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "bnb-lion8bit",
    "output_dir": "output/controlnet-sdxl/models",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 42,
    "train_batch_size": 1,
    "use_ema": false,
    "vae_cache_ondemand": true,
    "validation_guidance": 4.2,
    "validation_guidance_rescale": 0.0,
    "validation_num_inference_steps": 20,
    "validation_resolution": "1024x1024",
    "validation_seed": 42,
    "validation_step_interval": 10,
    "validation_torch_compile": false
}
```

## 对生成的 ControlNet 模型进行推理

下面是 **完整** ControlNet 模型（不是 ControlNet LoRA）的 SDXL 推理示例：

```py
# Update these values:
base_model = "stabilityai/stable-diffusion-xl-base-1.0"         # This is the model you used as `--pretrained_model_name_or_path`
controlnet_model_path = "diffusers/controlnet-canny-sdxl-1.0"   # This is the path to the resulting ControlNet checkpoint
# controlnet_model_path = "/path/to/controlnet/checkpoint-100"

# Leave the rest alone:
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'

image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    controlnet_model_path,
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

images = pipe(
    prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

images[0].save(f"hug_lab.png")
```
(_演示代码摘自 [Hugging Face SDXL ControlNet 示例](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)_)


## 自动数据增强与条件生成

SimpleTuner 可以在启动时自动生成条件数据集，从而无需手动预处理。它对以下场景尤其有用：
- 超分辨率训练
- JPEG 伪影去除
- 深度引导生成
- 边缘检测（Canny）

### 工作原理

无需手动创建条件数据集，你可以在主数据集配置中指定 `conditioning` 数组。SimpleTuner 会：
1. 在启动时生成条件图像
2. 创建带有适当元数据的独立数据集
3. 自动将它们与主数据集关联

### 性能注意事项

部分生成器在 CPU 受限时会更慢，系统 CPU 负载较高时尤其明显。另一些生成器需要 GPU 资源，因此会在主进程中运行，从而增加启动时间。

**基于 CPU 的生成器（较快）：**
- `superresolution` - 模糊与噪声操作
- `jpeg_artifacts` - 压缩模拟
- `random_masks` - 掩码生成
- `canny` - 边缘检测

**基于 GPU 的生成器（较慢）：**
- `depth` / `depth_midas` - 需要加载 Transformer 模型
- `segmentation` - 语义分割模型
- `optical_flow` - 运动估计

基于 GPU 的生成器在主进程中运行，对于大型数据集可能显著增加启动时间。

### 示例：多任务条件数据集

下面是一个从单一源数据集生成多种条件类型的完整示例：

```json
[
  {
    "id": "multitask-training",
    "type": "local",
    "instance_data_dir": "/datasets/high-quality-images",
    "caption_strategy": "filename",
    "resolution": 512,
    "conditioning": [
      {
        "type": "superresolution",
        "blur_radius": 2.0,
        "noise_level": 0.02,
        "captions": ["enhance image quality", "increase resolution", "sharpen"]
      },
      {
        "type": "jpeg_artifacts",
        "quality_range": [20, 40],
        "captions": ["remove compression", "fix jpeg artifacts"]
      },
      {
        "type": "canny",
        "low_threshold": 50,
        "high_threshold": 150
      }
    ]
  },
  {
    "id": "text-embed-cache",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/sdxl"
  }
]
```

该配置将会：
1. 从 `/datasets/high-quality-images` 加载高质量图像
2. 自动生成三种条件数据集
3. 为超分辨率与 JPEG 任务使用指定的提示词
4. 为 Canny 边缘数据集使用原始图像提示词

#### 生成数据集的提示词策略

为生成的条件数据设置提示词有两种方式：

1. **使用源提示词**（默认）：省略 `captions` 字段
2. **自定义提示词**：提供字符串或字符串数组

对于“enhance”或“remove artifacts”等任务特定训练，自定义提示词通常比原始图像描述效果更好。

### 启动时间优化

对于大型数据集，条件生成可能耗时较长。优化方法：

1. **只生成一次**：条件数据会被缓存，若已存在则不会重新生成
2. **使用 CPU 生成器**：可利用多进程加速生成
3. **禁用未使用类型**：只生成训练需要的类型
4. **预生成**：使用 `--skip_file_discovery=true` 跳过发现与条件生成
5. **避免磁盘扫描**：在大型数据集配置中使用 `preserve_data_backend_cache=True` 可避免重新扫描已有条件数据，大幅缩短启动时间

生成过程会显示进度条，且支持中断后恢复。
