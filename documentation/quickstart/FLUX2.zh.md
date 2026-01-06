# FLUX.2 快速入门

本指南介绍如何在 FLUX.2-dev 上训练 LoRA，这是 Black Forest Labs 最新的图像生成模型，采用 Mistral-3 文本编码器。

## 模型概述

FLUX.2-dev 相比 FLUX.1 引入了重大架构变化：

- **文本编码器**：使用 Mistral-Small-3.1-24B 替代 CLIP+T5
- **架构**：8 个 DoubleStreamBlock + 48 个 SingleStreamBlock
- **潜在通道**：32 个 VAE 通道 → 像素重排后为 128（FLUX.1 为 16）
- **VAE**：自定义 VAE，带有批归一化和像素重排
- **嵌入维度**：15,360（从 Mistral 的第 10、20、30 层堆叠而来）

## 硬件要求

由于 Mistral-3 文本编码器的存在，FLUX.2 对资源有较高要求：

### 显存要求

仅 24B 的 Mistral 文本编码器就需要大量显存：

| 组件 | bf16 | int8 | int4 |
|-----------|------|------|------|
| Mistral-3 (24B) | ~48GB | ~24GB | ~12GB |
| FLUX.2 Transformer | ~24GB | ~12GB | ~6GB |
| VAE + 开销 | ~4GB | ~4GB | ~4GB |

| 配置 | 大致总显存 |
|--------------|------------------------|
| 全部使用 bf16 | ~76GB+ |
| int8 文本编码器 + bf16 transformer | ~52GB |
| 全部使用 int8 | ~40GB |
| int4 文本编码器 + int8 transformer | ~22GB |

### 系统内存

- **最低**：96GB 系统内存（加载 24B 文本编码器需要大量内存）
- **推荐**：128GB+ 以确保流畅运行

### 推荐硬件

- **最低配置**：2 张 48GB GPU（A6000、L40S）配合 FSDP2 或 DeepSpeed
- **推荐配置**：4 张 H100 80GB 配合 fp8-torchao
- **重度量化（int4）**：2 张 24GB GPU 可能可行，但属于实验性配置

由于 Mistral-3 文本编码器和 transformer 的总体大小，FLUX.2 基本上需要多 GPU 分布式训练（FSDP2 或 DeepSpeed）。

## 前提条件

### Python 版本

FLUX.2 需要 Python 3.10 或更高版本，以及较新版本的 transformers：

```bash
python --version  # 应为 3.10+
pip install transformers>=4.45.0
```

### 模型访问

FLUX.2-dev 需要在 Hugging Face 上获得访问批准：

1. 访问 [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)
2. 接受许可协议
3. 确保已登录 Hugging Face CLI

## 安装

```bash
pip install simpletuner[cuda]
```

如需开发环境设置：
```bash
git clone https://github.com/bghira/SimpleTuner
cd SimpleTuner
pip install -e ".[cuda]"
```

## 配置

### 网页界面方式

```bash
simpletuner server
```

访问 http://localhost:8001 并选择 FLUX.2 作为模型系列。

### 手动配置

创建 `config/config.json`：

<details>
<summary>查看示例配置</summary>

```json
{
  "model_type": "lora",
  "model_family": "flux2",
  "model_flavour": "dev",
  "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-dev",
  "output_dir": "/path/to/output",
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "gradient_checkpointing": true,
  "mixed_precision": "bf16",
  "learning_rate": 1e-4,
  "lr_scheduler": "constant",
  "max_train_steps": 10000,
  "validation_resolution": "1024x1024",
  "validation_num_inference_steps": 20,
  "flux_guidance_mode": "constant",
  "flux_guidance_value": 1.0,
  "lora_rank": 16
}
```
</details>

### 关键配置选项

#### 引导配置

FLUX.2 使用与 FLUX.1 类似的引导嵌入：

<details>
<summary>查看示例配置</summary>

```json
{
  "flux_guidance_mode": "constant",
  "flux_guidance_value": 1.0
}
```
</details>

或者在训练时使用随机引导：

<details>
<summary>查看示例配置</summary>

```json
{
  "flux_guidance_mode": "random-range",
  "flux_guidance_min": 1.0,
  "flux_guidance_max": 5.0
}
```
</details>

#### 量化（内存优化）

如需减少显存使用：

<details>
<summary>查看示例配置</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "int8-quanto",
  "base_model_default_dtype": "bf16"
}
```
</details>

#### TREAD（训练加速）

FLUX.2 支持 TREAD 以加速训练：

<details>
<summary>查看示例配置</summary>

```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.5, "start_layer_idx": 2, "end_layer_idx": -2}
    ]
  }
}
```
</details>

### 高级实验性功能

<details>
<summary>显示高级实验性详情</summary>


SimpleTuner 包含可以显著改善训练稳定性和性能的实验性功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：** 通过让模型在训练期间生成自己的输入，减少暴露偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

</details>

### 数据集配置

创建 `config/multidatabackend.json`：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "my-dataset",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux2/my-dataset",
    "instance_data_dir": "datasets/my-dataset",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux2",
    "write_batch_size": 64
  }
]
```
</details>

> 请参阅 [DATALOADER.md](../DATALOADER.md#caption_strategy) 中的 caption_strategy 选项和要求。

### 可选的编辑/参考条件

FLUX.2 可以训练**纯文生图**（无条件）或使用**配对的参考/编辑图像**。要添加条件，使用 [`conditioning_data`](../DATALOADER.md#conditioning_data) 将主数据集与一个或多个 `conditioning` 数据集配对，并选择 [`conditioning_type`](../DATALOADER.md#conditioning_type)：

<details>
<summary>查看示例配置</summary>

```jsonc
[
  {
    "id": "flux2-edits",
    "type": "local",
    "instance_data_dir": "/datasets/flux2/edits",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": ["flux2-references"],
    "cache_dir_vae": "cache/vae/flux2/edits"
  },
  {
    "id": "flux2-references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/flux2/references",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/flux2/references"
  }
]
```
</details>

- 当您需要裁剪与编辑图像 1:1 对齐时，使用 `conditioning_type=reference_strict`。`reference_loose` 允许不匹配的宽高比。
- 编辑数据集和参考数据集之间的文件名必须匹配；每个编辑图像都应有相应的参考文件。
- 当提供多个条件数据集时，根据需要设置 `conditioning_multidataset_sampling`（`combined` 或 `random`）；详见 [OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling)。
- 如果没有 `conditioning_data`，FLUX.2 将回退到标准文生图训练。

### LoRA 目标

可用的 LoRA 目标预设：

- `all`（默认）：所有注意力层和 MLP 层
- `attention`：仅注意力层（qkv、proj）
- `mlp`：仅 MLP/前馈层
- `tiny`：最小化训练（仅 qkv 层）

<details>
<summary>查看示例配置</summary>

```json
{
  "--flux_lora_target": "all"
}
```
</details>

## 训练

### 登录服务

```bash
huggingface-cli login
wandb login  # 可选
```

### 开始训练

```bash
simpletuner train
```

或通过脚本：

```bash
./train.sh
```

### 内存卸载

对于内存受限的配置，FLUX.2 支持对 transformer 和（可选的）Mistral-3 文本编码器进行分组卸载：

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
--group_offload_text_encoder
```

推荐为 FLUX.2 使用 `--group_offload_text_encoder` 标志，因为 24B 的 Mistral 文本编码器在文本嵌入缓存期间可以显著受益于卸载。您还可以添加 `--group_offload_vae` 在潜在缓存期间包含 VAE 的卸载。

## 验证提示词

创建 `config/user_prompt_library.json`：

<details>
<summary>查看示例配置</summary>

```json
{
  "portrait_subject": "a professional portrait photograph of <subject>, studio lighting, high detail",
  "artistic_subject": "an artistic interpretation of <subject> in the style of renaissance painting",
  "cinematic_subject": "a cinematic shot of <subject>, dramatic lighting, film grain"
}
```
</details>

## 推理

### 使用训练好的 LoRA

FLUX.2 LoRA 可以使用 SimpleTuner 推理管道或兼容的工具加载（待社区支持开发完成后）。

### 引导值

- 使用 `flux_guidance_value=1.0` 进行训练适用于大多数用例
- 在推理时，使用正常的引导值（3.0-5.0）

## 与 FLUX.1 的区别

| 方面 | FLUX.1 | FLUX.2 |
|--------|--------|--------|
| 文本编码器 | CLIP-L/14 + T5-XXL | Mistral-Small-3.1-24B |
| 嵌入维度 | CLIP: 768, T5: 4096 | 15,360 (3×5,120) |
| 潜在通道 | 16 | 32 (→像素重排后为 128) |
| VAE | AutoencoderKL | 自定义（BatchNorm） |
| VAE 缩放因子 | 8 | 16 (8×2 像素重排) |
| Transformer 块 | 19 个联合 + 38 个单独 | 8 个双流 + 48 个单流 |

## 故障排除

### 启动时内存不足

- 启用 `--offload_during_startup=true`
- 使用 `--quantize_via=cpu` 进行文本编码器量化
- 减少 `--vae_batch_size`

### 文本嵌入速度慢

Mistral-3 体积很大；可以考虑：
- 在训练前预缓存所有文本嵌入
- 使用文本编码器量化
- 使用更大的 `write_batch_size` 进行批处理

### 训练不稳定

- 降低学习率（尝试 5e-5）
- 增加梯度累积步数
- 启用梯度检查点
- 使用 `--max_grad_norm=1.0`

### CUDA 内存不足

- 启用量化（`int8-quanto` 或 `int4-quanto`）
- 启用梯度检查点
- 减少批次大小
- 启用分组卸载
- 使用 TREAD 进行令牌路由效率优化

## 高级：TREAD 配置

TREAD（Token Routing for Efficient Architecture-agnostic Diffusion）通过选择性处理令牌来加速训练：

<details>
<summary>查看示例配置</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 4,
        "end_layer_idx": -4
      }
    ]
  }
}
```
</details>

- `selection_ratio`：要保留的令牌比例（0.5 = 50%）
- `start_layer_idx`：开始应用路由的第一层
- `end_layer_idx`：最后一层（负数表示从末尾计算）

预期加速：根据配置不同，可达 20-40%。

## 另请参阅

- [FLUX.1 快速入门](FLUX.zh.md) - FLUX.1 训练
- [TREAD 文档](../TREAD.md) - 详细的 TREAD 配置
- [LyCORIS 训练指南](../LYCORIS.md) - LoRA 和 LyCORIS 训练方法
- [数据加载器配置](../DATALOADER.md) - 数据集设置
