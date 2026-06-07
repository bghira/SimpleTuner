# AnyFlow 继续训练快速入门

本指南用于在下游 Wan 数据集上继续 AnyFlow training objective。实现概览见 [AnyFlow](/documentation/experimental/ANYFLOW.zh.md)。

NVIDIA 公开的 AnyFlow checkpoints 是带完整 transformer 权重的 Diffusers pipelines，不是 LoRA adapters。不要把这些 repositories 填到 `init_lora`。只有在你拥有 SimpleTuner-compatible LoRA 文件或 repository 时才使用 `init_lora`。

## 使用哪个 checkpoint

把 bidirectional T2V AnyFlow checkpoints 作为 pretrained transformer：

- `nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers`
- `nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers`

继续使用原始 Wan checkpoint 提供 text encoder、tokenizer、VAE 和 scheduler：

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- `Wan-AI/Wan2.1-T2V-14B-Diffusers`

FAR checkpoints (`nvidia/AnyFlow-FAR-*`) 使用 causal AnyFlow transformer architecture，不是本 SimpleTuner quickstart 的目标。

## 示例配置

从普通 Wan quickstart config 开始，然后修改 model 和 distillation 字段：

```json
{
  "model_family": "wan",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_vae_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_transformer_model_name_or_path": "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_transformer_subfolder": "transformer",
  "data_backend_config": "config/wan/multidatabackend.json",
  "output_dir": "output/wan-anyflow-lora",
  "lora_rank": 32,
  "lora_alpha": 32,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "learning_rate": 0.0001,
  "max_train_steps": 1000,
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "target_mode": "online_teacher",
      "teacher_rollout_steps": 1,
      "r_timestep_sampler": "uniform",
      "min_interval_ratio": 0.02,
      "gate_value": 0.25,
      "deltatime_type": "r",
      "loss_weight": 1.0
    }
  }
}
```

在 SimpleTuner 目录运行训练：

```bash
simpletuner train
```

生成的 LoRA 会从蒸馏后的 AnyFlow transformer 继续训练，并在 downstream fine-tuning 中保持 AnyFlow objective。

## 如果你已有 AnyFlow LoRA

如果已经单独发布了 extracted AnyFlow LoRA，就使用原始 Wan base checkpoint，并用 `init_lora` 加载 adapter：

```json
{
  "model_family": "wan",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_vae_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "init_lora": "your-org/anyflow-wan21-1.3b-lora",
  "lora_rank": 32,
  "lora_alpha": 32,
  "distillation_method": "anyflow"
}
```

LoRA rank 和 target modules 必须与发布的 adapter 匹配。完整 transformer checkpoint 不是有效的 `init_lora` 值。

## 关于提取 LoRA

原则上可以从完整 AnyFlow transformer 提取 LoRA，但这是转换项目，不是训练选项。SimpleTuner 包含实验性脚本：

```bash
python scripts/extract_peft_lora.py \
  Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers \
  output/anyflow-wan21-1.3b-r32.safetensors \
  --rank 32
```

LyCORIS/LoCon 可使用 `scripts/extract_lycoris_adapter.py`，参数相同并加上 `--algo locon`。

转换会加载匹配的 Wan base transformer 和 AnyFlow transformer，计算匹配 linear-layer weights 的 delta，把 delta factorize 成 low-rank LoRA matrices，保存 compatible adapter，并验证结果。

这是 rank-dependent 的近似转换。默认 target 与 SimpleTuner 的 Wan PEFT defaults (`to_q,to_k,to_v,to_out.0`) 匹配。只有 downstream config 也 target 相同 modules 时才使用 `--target-modules all-linear`。

## 当前限制

- NVIDIA AnyFlow 公开模型 license 是 noncommercial；发布 derived adapters 前请检查 upstream model card。
- AnyFlow validation 已通过 distiller scheduler hook 为已注册的 FlowMap-capable pipeline 接入。自定义或 external validation path 仍需把 `r_timestep` 或 `timestep_r` 传入 model component。
- full-rank online-teacher continuation 仍需要单独的 student/teacher wiring。目前支持的路径是 LoRA continuation。
