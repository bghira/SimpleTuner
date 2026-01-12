# Mixture-of-Experts

SimpleTuner 允许将训练任务拆分为两个部分，使推理时的 self-attention 与 cross-attention 阶段可以由两套完全不同的权重承担。

在本示例中，我们将使用 SegMind 与 Hugging Face 合作的 [SSD-1B](https://huggingface.co/segmind/ssd-1b)，构建两个新模型，它们相比单模型训练更稳定，细节也更好。

由于 SSD-1B 体积较小，即使在更轻量的硬件上也能训练。我们使用其预训练权重，因此必须遵守 Apache 2.0 许可，但过程相对简单。甚至可以在商业场景中使用最终权重。

SDXL 0.9 和 1.0 都包含一个完整的基础模型与一个分段调度的 refiner。

- 基础模型训练步数为 999 到 0
  - 基础模型超过 3B 参数，完全可独立运行。
- refiner 模型训练步数为 199 到 0
  - refiner 也超过 3B 参数，资源开销看似不必要。它单独运行时会产生明显变形并偏向卡通风格。

让我们看看如何改进这一点。


## 基础模型（“Stage One”）

混合专家模型的第一部分称为基础模型。如 SDXL 所示，它可在 1000 个时间步上训练，但并非必须。以下配置仅训练 1000 步中的 650 步，节省时间并提升稳定性。

### 环境配置

在 `config/config.env` 中设置以下值：

```bash
# Ensure these aren't incorrectly set.
export USE_BITFIT=false
export USE_DORA=false
# lora could be used here instead, but the concept hasn't been explored.
export MODEL_TYPE="full"
export MODEL_FAMILY="sdxl"
export MODEL_NAME="segmind/SSD-1B"
# The original Segmind model used a learning rate of 1e-5, which is
# probably too high for whatever batch size most users can pull off.
export LEARNING_RATE=4e-7

# We really want this as high as you can tolerate.
# - If training is very slow, ensure your CHECKPOINT_STEPS and VALIDATION_STEPS
#   are set low enough that you'll get a checkpoint every couple hours.
# - The original Segmind models used a batch size of 32 with 4 accumulations.
export TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1

# If you are running on a beefy machine that doesn't fully utilise its VRAM during training, set this to "false" and your training will go faster.
export USE_GRADIENT_CHECKPOINTING=true

# Enable first stage model training
export TRAINER_EXTRA_ARGS="--refiner_training --refiner_training_strength=0.35 --refiner_training_invert_schedule"

# Optionally reparameterise it to v-prediction/zero-terminal SNR. 'sample' prediction_type can be used instead for x-prediction.
# This will start out looking pretty terrible until about 1500-2500 steps have passed, but it could be very worthwhile.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --prediction_type=v_prediction --rescale_betas_zero_snr --training_scheduler_timestep_spacing=trailing"
```

### 数据加载器配置

数据加载器无需特殊设置。参见 [dataloader 配置指南](DATALOADER.md)。

### 验证

当前 SimpleTuner 在 stage one 评估时不会启用第二阶段模型。

未来将支持此选项，以便当 stage two 已存在或同时训练时使用。

---

## Refiner 模型（“Stage Two”）

### 与 SDXL refiner 训练的对比

- 使用 Segmind SSD-1B 作为两阶段时，文本嵌入**可以**在两次训练之间共享
  - SDXL refiner 的文本嵌入布局与 SDXL base 不同。
- VAE 嵌入**可以**共享，与 SDXL refiner 一样。两模型使用相同输入布局。
- Segmind 模型不使用美学评分，而是使用与 SDXL 相同的微条件输入（例如裁剪坐标）
- 由于模型更小、可复用 stage one 的文本嵌入，训练速度明显更快

### 环境配置

在 `config/config.env` 中更新以下值，将训练切换到 stage two。建议保留一份 base 配置以便对照。

```bash
# Update your OUTPUT_DIR value, so that we don't overwrite the stage one model checkpoints.
export OUTPUT_DIR="/some/new/path"

# We'll swap --refiner_training_invert_schedule for --validation_using_datasets
# - Train the end of the model instead of the beginning
# - Validate using images as input for partial denoising to evaluate fine detail improvements
export TRAINER_EXTRA_ARGS="--refiner_training --refiner_training_strength=0.35 --validation_using_datasets"

# Don't update these values if you've set them on the stage one. Be sure to use the same parameterisation for both models!
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --prediction_type=v_prediction --rescale_betas_zero_snr --training_scheduler_timestep_spacing=trailing"
```

### 数据集格式

图像应为高质量，请移除含压缩伪影或其他问题的数据集。

除此之外，两次训练可以使用完全相同的数据加载器配置。

若需要示例数据集，许可宽松的 [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) 是不错的选择。

### 验证

stage two refiner 训练会自动从各训练集选择图像，用于验证时的部分去噪输入。

## CLIP 分数跟踪

若要通过评估对模型表现打分，请参见 [此文档](evaluation/CLIP_SCORES.md) 了解 CLIP 分数的配置与解读。

# 稳定评估损失

若要使用稳定 MSE 损失评估模型性能，请参见 [此文档](evaluation/EVAL_LOSS.md) 了解评估损失的配置与解读。

## 推理时组合两种模型

若要在简单脚本中组合两种模型进行实验，可参考以下示例：

```py
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, UniPCMultistepScheduler
from torch import float16, cuda
from torch.backends import mps

# For a training_refiner_strength of .35, you'll set the base model strength to 0.65.
# Formula: 1 - training_refiner_strength
training_refiner_strength = 0.35
base_model_power = 1 - training_refiner_strength
# Reduce this for lower quality but speed-up.
num_inference_steps = 40
# Update these to your local or hugging face hub paths.
stage_1_model_id = 'bghira/terminus-xl-velocity-v2'
stage_2_model_id = 'bghira/terminus-xl-refiner'
torch_device = 'cuda' if cuda.is_available() else 'mps' if mps.is_available() else 'cpu'

pipe = StableDiffusionXLPipeline.from_pretrained(stage_1_model_id, add_watermarker=False, torch_dtype=float16).to(torch_device)
pipe.scheduler = UniPCMultistepScheduler.from_pretrained(stage_1_model_id, subfolder="scheduler")
img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(stage_2_model_id).to(device=torch_device, dtype=float16)
img2img_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(stage_1_model_id, subfolder="scheduler")

prompt = "An astronaut riding a green horse"

# Important: update this to True if you reparameterised the models.
use_zsnr = True

image = pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_end=base_model_power,
    guidance_scale=9.2,
    guidance_rescale=0.7 if use_zsnr else 0.0,
    output_type="latent",
).images
image = img2img_pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_start=base_model_power,
    guidance_scale=9.2,
    guidance_rescale=0.7 if use_zsnr else 0.0,
    image=image,
).images[0]
image.save('demo.png', format="PNG")
```

可尝试的实验：
- 调整 `base_model_power` 或 `num_inference_steps`（两条 pipeline 必须一致）
- `guidance_scale`、`guidance_rescale` 可在两个阶段分别设置，会影响对比度与真实感
- 基础模型与 refiner 使用不同提示词以改变细节引导
