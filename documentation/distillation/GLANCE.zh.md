# Glance 风格单样本 LoRA

Glance 更像是“带分段日程的单图 LoRA”，而非真正的蒸馏流水线。你会在同一张图片/字幕上训练两个小 LoRA：一个用于早期步（“Slow”），一个用于后期步（“Fast”），推理时再串联。

## 你将获得

- 在单张图片/字幕上训练的两个 LoRA
- 使用自定义 flow 时间步而非 CDF 采样（`--flow_custom_timesteps`）
- 适用于 flow-matching 模型（Flux、SD3 系、Qwen-Image 等）

## 前置条件

- Python 3.10–3.13，已安装 SimpleTuner（`pip install 'simpletuner[cuda]'`）
  - CUDA 13 / Blackwell users (NVIDIA B-series GPUs): `pip install 'simpletuner[cuda13]'`
- 一张图像与对应字幕文件，文件名相同（如 `data/glance.png` + `data/glance.txt`）
- 一个 flow 模型检查点（示例使用 `black-forest-labs/FLUX.1-dev`）

## Step 1 – 用单样本配置 dataloader

在 `config/multidatabackend.json` 中引用你的单张图像/文本对。最小示例：

```json
[
  {
    "backend": "simple",
    "resolution": 1024,
    "shuffle": true,
    "image_dir": "data",
    "caption_extension": ".txt"
  }
]
```

## Step 2 – 训练 Slow LoRA（早期步）

创建 `config/glance-slow.json`，填入早期步列表：

```json
{
  "--model_type": "lora",
  "--model_family": "flux",
  "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
  "--data_backend_config": "config/multidatabackend.json",
  "--train_batch_size": 1,
  "--max_train_steps": 60,
  "--lora_rank": 32,
  "--output_dir": "output/glance-slow",
  "--flow_custom_timesteps": "1000,979.1915,957.5157,934.9171,911.3354"
}
```

运行：

```bash
simpletuner train --config config/glance-slow.json
```

## Step 3 – 训练 Fast LoRA（后期步）

将配置复制为 `config/glance-fast.json`，修改输出路径与时间步列表：

```json
{
  "--model_type": "lora",
  "--model_family": "flux",
  "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
  "--data_backend_config": "config/multidatabackend.json",
  "--train_batch_size": 1,
  "--max_train_steps": 60,
  "--lora_rank": 32,
  "--output_dir": "output/glance-fast",
  "--flow_custom_timesteps": "886.7053,745.0728,562.9505,320.0802,20.0"
}
```

运行：

```bash
simpletuner train --config config/glance-fast.json
```

注记：
- `--flow_custom_timesteps` 会覆盖 flow-matching 的常规 sigmoid/uniform/beta 采样，并在每步从列表中均匀抽取。
- 其他 flow schedule 相关标志保持默认即可，列表会绕过它们。
- 如果你更喜欢使用 sigma 而非 0–1000 时间步，请在 `[0,1]` 范围内提供数值。

## Step 4 – 一起使用两种 LoRA（使用正确的 sigma）

Diffusers 需要与你训练时相同的 sigma 调度。将 0–1000 时间步转换为 0–1 的 sigma 并显式传入；`num_inference_steps` 必须与每个列表长度一致。

```python
import torch
from diffusers import FluxPipeline

device = "cuda"
base_model = "black-forest-labs/FLUX.1-dev"

# Convert the training timesteps to sigmas in [0, 1]
slow_sigmas = [t / 1000.0 for t in (1000.0, 979.1915, 957.5157, 934.9171, 911.3354)]
fast_sigmas = [t / 1000.0 for t in (886.7053, 745.0728, 562.9505, 320.0802, 20.0)]
generator = torch.Generator(device=device).manual_seed(0)

# Early phase
pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
pipe.load_lora_weights("output/glance-slow")
latents = pipe(
    prompt="your prompt",
    num_inference_steps=len(slow_sigmas),
    sigmas=slow_sigmas,
    output_type="latent",
    generator=generator,
).images

# Late phase (continue the same schedule)
pipe_fast = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
pipe_fast.load_lora_weights("output/glance-fast")
image = pipe_fast(
    prompt="your prompt",
    num_inference_steps=len(fast_sigmas),
    sigmas=fast_sigmas,
    latents=latents,
    guidance_scale=1.0,
    generator=generator,
).images[0]
image.save("glance.png")
```

请复用相同的 prompt 与 generator/seed，确保 Fast LoRA 从 Slow LoRA 停止处无缝续接，并让 sigma 列表与训练时的 `--flow_custom_timesteps` 对齐。
