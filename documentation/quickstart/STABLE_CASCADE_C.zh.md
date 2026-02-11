# Stable Cascade Stage C 快速入门

本指南将介绍如何配置 SimpleTuner 微调 **Stable Cascade Stage C prior**。Stage C 学习的是文本到图像的 prior，它会向 Stage B/C 解码器堆栈提供输入，因此这里的训练质量会直接影响下游解码器输出。本文聚焦 LoRA 训练，但若显存足够，也可用于全参微调。

> **提示：**Stage C 使用 10 亿+ 参数的 CLIP-G/14 文本编码器与 EfficientNet 自动编码器。请确保安装 torchvision，并预期文本嵌入缓存非常大（每条提示约为 SDXL 的 5–6 倍）。

## 硬件要求

- **LoRA 训练：**20–24 GB VRAM（RTX 3090/4090、A6000 等）
- **全模型训练：**推荐 48 GB+ VRAM（A6000、A100、H100）。DeepSpeed/FSDP2 offload 可降低要求，但复杂度较高。
- **系统内存：**建议 32 GB，以避免 CLIP-G 编码器与缓存线程受限。
- **磁盘：**至少预留约 50 GB 用于 prompt 缓存。Stage C 的 CLIP-G 嵌入每条约 4–6 MB。

## 前提条件

1. Python 3.13（与项目 `.venv` 一致）。
2. CUDA 12.1+ 或 ROCm 5.7+ 用于 GPU 加速（或 Apple M 系列的 Metal，但 Stage C 主要在 CUDA 上测试）。
3. `torchvision`（Stable Cascade 自动编码器所需）以及 `accelerate` 用于启动训练。

检查 Python 版本：

```bash
python --version
```

安装缺失包（Ubuntu 示例）：

```bash
sudo apt update && sudo apt install -y python3.13 python3.13-venv
```

## 安装

按标准 SimpleTuner 安装（pip 或源码）。对于典型 CUDA 工作站：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

如果直接在仓库开发，请从源码安装并执行 `pip install -e .[cuda,dev]`。

## 环境设置

### 1. 复制基础配置

```bash
cp config/config.json.example config/config.json
```

设置以下关键项（示例为 Stage C 推荐基线）：

| 键 | 推荐值 | 说明 |
| --- | -------------- | ----- |
| `model_family` | `"stable_cascade"` | 用于加载 Stage C 组件 |
| `model_flavour` | `"stage-c"`（或 `"stage-c-lite"`） | lite 版本会裁剪参数，约 18 GB VRAM 可用 |
| `model_type` | `"lora"` | 全参微调可用但显存需求更高 |
| `mixed_precision` | `"no"` | Stage C 默认不允许混合精度，除非设置 `i_know_what_i_am_doing=true`；fp32 更稳妥 |
| `gradient_checkpointing` | `true` | 节省约 3–4 GB VRAM |
| `vae_batch_size` | `1` | Stage C 自动编码器很重，保持小 batch |
| `validation_resolution` | `"1024x1024"` | 与下游解码器期望一致 |
| `stable_cascade_use_decoder_for_validation` | `true` | 验证时使用 prior+decoder 管线 |
| `stable_cascade_decoder_model_name_or_path` | `"stabilityai/stable-cascade"` | 可替换为本地 Stage B/C 解码器路径 |
| `stable_cascade_validation_prior_num_inference_steps` | `20` | prior 去噪步数 |
| `stable_cascade_validation_prior_guidance_scale` | `3.0–4.0` | prior 的 CFG |
| `stable_cascade_validation_decoder_guidance_scale` | `0.0–0.5` | decoder CFG（0.0 写实，>0.0 更贴合提示词） |

#### 示例 `config/config.json`

<details>
<summary>查看示例配置</summary>

```json
{
  "base_model_precision": "int8-torchao",
  "checkpoint_step_interval": 100,
  "data_backend_config": "config/stable_cascade/multidatabackend.json",
  "gradient_accumulation_steps": 2,
  "gradient_checkpointing": true,
  "hub_model_id": "stable-cascade-stage-c-lora",
  "learning_rate": 1e-4,
  "lora_alpha": 16,
  "lora_rank": 16,
  "lora_type": "standard",
  "lr_scheduler": "cosine",
  "max_train_steps": 30000,
  "mixed_precision": "no",
  "model_family": "stable_cascade",
  "model_flavour": "stage-c",
  "model_type": "lora",
  "optimizer": "adamw_bf16",
  "output_dir": "output/stable_cascade_stage_c",
  "report_to": "wandb",
  "seed": 42,
  "stable_cascade_decoder_model_name_or_path": "stabilityai/stable-cascade",
  "stable_cascade_decoder_subfolder": "decoder_lite",
  "stable_cascade_use_decoder_for_validation": true,
  "stable_cascade_validation_decoder_guidance_scale": 0.0,
  "stable_cascade_validation_prior_guidance_scale": 3.5,
  "stable_cascade_validation_prior_num_inference_steps": 20,
  "train_batch_size": 4,
  "use_ema": true,
  "vae_batch_size": 1,
  "validation_guidance": 4.0,
  "validation_negative_prompt": "ugly, blurry, low-res",
  "validation_num_inference_steps": 30,
  "validation_prompt": "a cinematic photo of a shiba inu astronaut",
  "validation_resolution": "1024x1024"
}
```
</details>

要点：

- `model_flavour` 可用 `stage-c` 和 `stage-c-lite`。显存不足或希望使用蒸馏 prior 时选 lite。
- `mixed_precision` 保持为 `"no"`。若强行改动，需设置 `i_know_what_i_am_doing=true` 并准备应对 NaN。
- 启用 `stable_cascade_use_decoder_for_validation` 后，prior 输出会被送入 Stage B/C 解码器，验证画廊会显示真实图像而非 latent。

### 2. 配置数据后端

创建 `config/stable_cascade/multidatabackend.json`：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "primary",
    "type": "local",
    "dataset_type": "images",
    "instance_data_dir": "/data/stable-cascade",
    "resolution": "1024x1024",
    "bucket_resolutions": ["1024x1024", "896x1152", "1152x896"],
    "crop": true,
    "crop_style": "random",
    "minimum_image_size": 768,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "caption_strategy": "filename",
    "prepend_instance_prompt": false,
    "repeats": 1
  },
  {
    "id": "stable-cascade-text-cache",
    "type": "local",
    "dataset_type": "text_embeds",
    "cache_dir": "/data/cache/stable-cascade/text",
    "default": true
  }
]
```
</details>

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

提示：

- Stage C latent 来自自动编码器，建议保持 1024×1024（或在较窄的人像/横向桶中）。解码器期望 1024px 输入对应的约 24×24 latent 网格。
- 保持 `target_downsample_size` 为 1024，避免窄裁剪导致宽高比超过 ~2:1。
- 一定要配置专用文本嵌入缓存，否则每次运行都会花 30–60 分钟重新计算 CLIP-G 嵌入。

### 3. 提示词库（可选）

创建 `config/stable_cascade/prompt_library.json`：

<details>
<summary>查看示例配置</summary>

```json
{
  "portrait": "a cinematic portrait photograph lit by studio strobes",
  "landscape": "a sweeping ultra wide landscape with volumetric lighting",
  "product": "a product render on a seamless background, dramatic reflections",
  "stylized": "digital illustration in the style of a retro sci-fi book cover"
}
```
</details>

在配置中添加 `"validation_prompt_library": "config/stable_cascade/prompt_library.json"`。

## 训练

1. 激活环境并运行 Accelerate 配置（若尚未配置）：

```bash
source .venv/bin/activate
accelerate config
```

2. 启动训练：

```bash
accelerate launch simpletuner/train.py \
  --config_file config/config.json \
  --data_backend_config config/stable_cascade/multidatabackend.json
```

第一轮训练时重点关注：

- **文本缓存吞吐** – Stage C 会记录缓存进度。高端 GPU 期望约 8–12 prompts/sec。
- **VRAM 占用** – 目标 <95% 以避免验证时 OOM。
- **验证输出** – 组合管线会在 `output/<run>/validation/` 输出全分辨率 PNG。

## 验证与推理注意事项

- 单独的 Stage C prior 只生成图像嵌入。SimpleTuner 验证封装会在 `stable_cascade_use_decoder_for_validation=true` 时自动通过解码器生成图像。
- 更换解码器版本可设置 `stable_cascade_decoder_subfolder` 为 `"decoder"`、`"decoder_lite"` 或包含 Stage B/C 权重的自定义目录。
- 如需更快预览，可将 `stable_cascade_validation_prior_num_inference_steps` 降到 ~12，`validation_num_inference_steps` 降到 20。满意后再提高以获得更高质量。

## 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。
*   **[Diff2Flow](../experimental/DIFF2FLOW.md)：**允许以流匹配目标训练 Stable Cascade。

> ⚠️ 这些功能会增加训练的计算开销。

</details>

## 故障排除

| 症状 | 解决方案 |
| --- | --- |
| “Stable Cascade Stage C requires --mixed_precision=no” | 设置 `"mixed_precision": "no"` 或添加 `"i_know_what_i_am_doing": true`（不推荐） |
| 验证只显示 prior（绿色噪声） | 确保 `stable_cascade_use_decoder_for_validation` 为 `true` 且已下载解码器权重 |
| 文本嵌入缓存耗时过久 | 使用 SSD/NVMe 缓存目录，避免网络挂载；可裁剪提示词或使用 `simpletuner-text-cache` CLI 预计算 |
| 自动编码器导入错误 | 在 `.venv` 内安装 torchvision（`pip install torchvision --extra-index-url https://download.pytorch.org/whl/cu124`）。Stage C 需要 EfficientNet 权重 |

## 下一步

- 根据主体复杂度尝试 `lora_rank`（8–32）与 `learning_rate`（5e-5 到 2e-4）。
- 训练 prior 后，可为 Stage B 加载 ControlNet/conditioning 适配器。
