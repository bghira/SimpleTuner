# Chroma 1 快速入门

![image](https://github.com/user-attachments/assets/3c8a12c6-9d45-4dd4-9fc8-6b7cd3ed51dd)

Chroma 1 是 Lodestone Labs 发布的 Flux.1 Schnell 的 8.9B 参数精简变体。本指南说明如何在 SimpleTuner 中配置 LoRA 训练。

## 硬件要求

尽管参数量更小，但内存占用接近 Flux Schnell：

- 量化基础 Transformer 仍可能使用 **≈40–50 GB** 系统内存。
- Rank-16 LoRA 训练通常消耗：
  - 基础不量化约 ~28 GB VRAM
  - int8 + bf16 约 ~16 GB VRAM
  - int4 + bf16 约 ~11 GB VRAM
  - NF4 + bf16 约 ~8 GB VRAM
- 现实的 GPU 最低要求：**RTX 3090 / RTX 4090 / L40S** 级别或更高。
- 在 **Apple M 系列（MPS）** 上进行 LoRA 训练表现良好，也支持 AMD ROCm。
- 全量微调建议使用 80 GB 级别加速卡或多 GPU 设置。

## 前提条件

Chroma 与 Flux 指南具有相同的运行环境要求：

- Python **3.10 – 3.12**
- 支持的加速后端（CUDA、ROCm 或 MPS）

检查 Python 版本：

```bash
python3 --version
```

安装 SimpleTuner（CUDA 示例）：

```bash
pip install 'simpletuner[cuda]'
```

关于后端特定的安装细节（CUDA、ROCm、Apple），请参考[安装指南](../INSTALL.md)。

## 启动 Web UI

```bash
simpletuner server
```

UI 将在 http://localhost:8001 提供。

## 通过 CLI 配置

`simpletuner configure` 会引导你完成核心设置。Chroma 的关键值是：

- `model_type`: `lora`
- `model_family`: `chroma`
- `model_flavour`: 以下之一
  - `base`（默认，质量均衡）
  - `hd`（更高保真、更耗算力）
  - `flash`（快但不稳定 — 不推荐用于生产）
- `pretrained_model_name_or_path`: 留空以使用上述风味映射
- `model_precision`: 保持默认 `bf16`
- `flux_fast_schedule`: 保持 **禁用**；Chroma 有自己的自适应采样

### 手动配置示例

<details>
<summary>查看示例配置</summary>

```jsonc
{
  "model_type": "lora",
  "model_family": "chroma",
  "model_flavour": "base",
  "output_dir": "/workspace/chroma-output",
  "network_rank": 16,
  "learning_rate": 2.0e-4,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "pretrained_model_name_or_path": null
}
```
</details>

> ⚠️ 若所在地区访问 Hugging Face 较慢，请在启动前设置 `HF_ENDPOINT=https://hf-mirror.com`。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

</details>

## 数据集与数据加载器

Chroma 使用与 Flux 相同的数据加载器格式。数据集准备与提示词库可参考[通用教程](../TUTORIAL.md)或[Web UI 教程](../webui/TUTORIAL.md)。

## Chroma 专属训练选项

- `flux_lora_target`：控制哪些 Transformer 模块接收 LoRA 适配器（`all`、`all+ffs`、`context`、`tiny` 等）。默认值与 Flux 一致，适合大多数情况。
- `flux_guidance_mode`：`constant` 表现良好；Chroma 不暴露 guidance 范围。
- 注意力掩码始终启用 — 请确保文本嵌入缓存生成时带有 padding mask（当前 SimpleTuner 版本的默认行为）。
- 调度偏移选项（`flow_schedule_shift` / `flow_schedule_auto_shift`）对 Chroma 不需要 — helper 已自动提升尾部时间步。
- `flux_t5_padding`：如果希望在掩码前将 padding token 置零，可设为 `zero`。

## 自动尾部时间步采样

Flux 使用 log-normal 调度，导致高噪/低噪极端值采样不足。Chroma 的训练辅助会对采样的 sigma 应用二次映射（`σ ↦ σ²` / `1-(1-σ)²`），使尾部区域被更频繁访问。**无需额外配置** — 该逻辑内置在 `chroma` 模型家族中。

## 验证与采样提示

- `validation_guidance_real` 会直接映射到管线的 `guidance_scale`。单次采样保持 `1.0`，如需验证中使用 CFG，可提高到 `2.0`–`3.0`。
- 快速预览用 20 步；更高质量用 28–32 步。
- 负向提示词可选；基础模型已去蒸馏。
- 当前仅支持 text‑to‑image；img2img 支持会在后续更新中加入。

## 故障排查

- **启动时 OOM**：启用 `offload_during_startup` 或对基础模型量化（`base_model_precision: int8-quanto`）。
- **训练早期发散**：确保梯度检查点开启，将 `learning_rate` 降至 `1e-4`，并确保 caption 多样。
- **验证重复同一姿势**：拉长提示词；流匹配模型在提示词多样性不足时会崩塌。

关于 DeepSpeed、FSDP2、评估指标等高级主题，请参考 README 中链接的共享指南。
