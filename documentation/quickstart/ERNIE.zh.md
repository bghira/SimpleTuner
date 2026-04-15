# ERNIE-Image [base / turbo] 快速开始

本指南介绍如何训练 ERNIE-Image LoRA。ERNIE-Image 是百度的单流 flow-matching transformer 系列，SimpleTuner 当前支持 `base` 与 `turbo` 两种 flavour。

## 硬件要求

ERNIE 不是轻量模型，建议按大型单流 transformer 的硬件标准准备：

- 使用 int8 量化 + bf16 LoRA 时，较现实的目标是 24G 以上显存
- 16G 也可能运行，但需要激进的 offload、RamTorch，并且速度会较慢
- 多卡、FSDP2、CPU/RAM offload 都有帮助

不建议使用 Apple GPU 训练。

### 可选的显存卸载

ERNIE 很适合启用 RamTorch，因为文本编码器比较大。如果还需要更多显存空间，可以再加 grouped module offload：

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

## 安装

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

更多内容见 [安装文档](../INSTALL.md)。

## 环境配置

### WebUI

```bash
simpletuner server
```

然后在训练向导中选择 ERNIE 模型族。

### 命令行

可以直接从仓库自带示例开始：

- 示例配置：`simpletuner/examples/ernie.peft-lora/config.json`
- 本地环境：`config/ernie-example/config.json`

运行：

```bash
simpletuner train --env ernie-example
```

如果手动配置，核心选项如下：

- `model_type`: `lora`
- `model_family`: `ernie`
- `model_flavour`: `base` 或 `turbo`
- `pretrained_model_name_or_path`:
  - `base`: `baidu/ERNIE-Image`
  - `turbo`: `baidu/ERNIE-Image-Turbo`
- `resolution`: 建议先从 `512` 开始
- `train_batch_size`: 先用 `1`
- `ramtorch`: `true`
- `ramtorch_text_encoder`: `true`
- `gradient_checkpointing`: `true`

示例配置使用：

- `max_train_steps: 100`
- `optimizer: optimi-lion`
- `learning_rate: 1e-4`
- `validation_guidance: 4.0`
- `validation_num_inference_steps: 20`

### Turbo 的 Assistant LoRA

ERNIE Turbo 已经接入 assistant LoRA 支持，但目前没有默认的适配器路径。

- 支持 flavour：`turbo`
- 默认权重名：`pytorch_lora_weights.safetensors`
- 需要用户自己提供：`assistant_lora_path`

如果你有自己的 assistant adapter：

```json
{
  "assistant_lora_path": "your-org/your-ernie-turbo-assistant-lora",
  "assistant_lora_weight_name": "pytorch_lora_weights.safetensors"
}
```

如果不想使用：

```json
{
  "disable_assistant_lora": true
}
```

### 数据集与 caption

示例环境使用：

- `dataset_name`: `RareConcepts/Domokun`
- `caption_strategy`: `instanceprompt`
- `instance_prompt`: `🟫`

这适合做 smoke test，但 ERNIE 对真实文本通常比单个触发词更敏感。正式训练时，建议使用更完整的 caption，或者至少使用类似 `a studio photo of <token>` 的描述性实例提示词。

### 其他说明

ERNIE 也支持这些单流模型功能：

- TREAD
- LayerSync
- REPA / CREPA 风格隐藏层采集
- Turbo 的 assistant LoRA 加载

建议先让基础训练稳定运行，再叠加这些高级功能。
