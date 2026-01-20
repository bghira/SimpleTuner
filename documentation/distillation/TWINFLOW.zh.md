# TwinFlow（RCGM）少步训练

TwinFlow 是围绕 **递归一致性梯度匹配（RCGM）** 的轻量少步方案。它 **不在 `distillation_method` 主选项内**——需要通过 `twinflow_*` 标志直接启用。加载器会将从 Hub 获取的配置默认设置为 `twinflow_enabled=false`，确保普通 Transformer 配置不受影响。

SimpleTuner 中的 TwinFlow：
* 默认仅支持 flow-matching，除非明确启用 `diff2flow_enabled` + `twinflow_allow_diff2flow` 来桥接扩散模型。
* 默认使用 EMA 教师；围绕教师/CFG 路径的 RNG 捕获/恢复 **始终开启**，以匹配参考 TwinFlow 行为。
* 负时间语义的符号嵌入已经接入 Transformer，但仅在 `twinflow_enabled=true` 时启用；HF 配置未启用不会改变行为。
* 默认损失使用 RCGM + real-velocity；可通过 `twinflow_adversarial_enabled: true` 启用完整的自对抗训练（L_adv 与 L_rectify 损失）。期望在 guidance `0.0` 下进行 1–4 步生成。
* W&B 日志可输出实验性的 TwinFlow 轨迹散点图（理论未验证）用于调试。

---

## 快速配置（flow-matching 模型）

在常规配置中加入 TwinFlow 项（`distillation_method` 保持未设置/null）：

```json
{
  "model_family": "sd3",
  "model_type": "lora",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
  "output_dir": "output/sd3-twinflow",

  "distillation_method": null,
  "use_ema": true,

  "twinflow_enabled": true,
  "twinflow_target_step_count": 2,
  "twinflow_estimate_order": 2,
  "twinflow_enhanced_ratio": 0.5,
  "twinflow_delta_t": 0.01,
  "twinflow_target_clamp": 1.0,

  "learning_rate": 1e-4,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "mixed_precision": "bf16",
  "validation_guidance": 0.0,
  "validation_num_inference_steps": 2
}
```

对于扩散模型（epsilon/v prediction），需要明确启用：

```json
{
  "prediction_type": "epsilon",
  "diff2flow_enabled": true,
  "twinflow_allow_diff2flow": true
}
```

> 默认情况下，TwinFlow 使用 RCGM + real-velocity 损失。启用 `twinflow_adversarial_enabled: true` 可获得完整的自对抗训练（L_adv 与 L_rectify 损失），无需外部判别器。

---

## 预期表现（论文数据）

来自 arXiv:2512.05150（PDF 文本）：
* 推理基准在 **单张 A100（BF16）** 上测得，吞吐（batch=10）与延迟（batch=1）均在 1024×1024 下测试。文中未给具体数值，只说明硬件条件。
* **GPU 内存对比**（1024×1024）显示 Qwen-Image-20B（LoRA）与 SANA-1.6B 在 TwinFlow 下可运行，而 DMD2 / SANA-Sprint 可能 OOM。
* 训练配置（表 6）列出 **batch size 128/64/32/24** 与 **训练步数 30k–60k（或 7k–10k 短周期）**；使用恒定学习率，EMA 衰减常为 0.99。
* PDF **未报告** 总 GPU 数、节点布局或墙钟时间。

这些仅作方向性参考，不是保证。精确硬件/运行时间需作者确认。

---

## 关键选项

* `twinflow_enabled`: 开启 RCGM 辅助损失；保持 `distillation_method` 为空并禁用 scheduled sampling。配置缺失时默认 `false`。
* `twinflow_target_step_count`（推荐 1–4）：用于训练并复用于验证/推理。由于 CFG 已内置，guidance 被强制为 `0.0`。
* `twinflow_estimate_order`: RCGM rollout 的积分阶数（默认 2）。更高值会增加教师 passes。
* `twinflow_enhanced_ratio`: 使用教师 cond/uncond 预测进行 CFG 风格目标细化（默认 0.5；设为 0.0 禁用）。使用捕获的 RNG 保证 cond/uncond 对齐。
* `twinflow_delta_t` / `twinflow_target_clamp`: 控制递归目标的形状；默认值与论文稳定设置一致。
* `use_ema` + `twinflow_require_ema`（默认 true）：使用 EMA 权重作为教师。仅在接受学生当教师时设置 `twinflow_allow_no_ema_teacher: true`。
* `twinflow_allow_diff2flow`: 当 `diff2flow_enabled` 为 true 时启用 epsilon/v-prediction 桥接。
* RNG 捕获/恢复：始终开启以匹配参考 TwinFlow 实现，没有关闭开关。
* 符号嵌入：`twinflow_enabled` 为 true 时，向支持 `timestep_sign` 的 Transformer 传递 `twinflow_time_sign`；否则不使用额外嵌入。

### 对抗分支（完整 TwinFlow）

启用原始论文中的自对抗训练以提升质量：

* `twinflow_adversarial_enabled`（默认 false）：启用 L_adv 与 L_rectify 损失。使用负时间训练"假"轨迹，实现无需外部判别器的分布匹配。
* `twinflow_adversarial_weight`（默认 1.0）：对抗损失（L_adv）的权重乘数。
* `twinflow_rectify_weight`（默认 1.0）：校正损失（L_rectify）的权重乘数。

启用后，训练会通过单步生成创建假样本，然后训练两个损失：
- **L_adv**：带负时间的假速度损失——教模型将假样本映射回噪声。
- **L_rectify**：分布匹配损失——对齐真假轨迹预测以获得更直的路径。

---

## 训练与验证流程

1. 以正常的 flow-matching 训练方式运行（无需 distiller）。除非明确关闭，EMA 必须存在；RNG 对齐自动处理。
2. 验证自动切换到 **TwinFlow/UCGM 调度器**，使用 `twinflow_target_step_count` 步并设置 `guidance_scale=0.0`。
3. 导出管线需手动设置调度器：

```python
from simpletuner.helpers.training.custom_schedule import TwinFlowScheduler

pipe = ...  # your loaded diffusers pipeline
pipe.scheduler = TwinFlowScheduler(num_train_timesteps=1000, prediction_type="flow_matching", shift=1.0)
pipe.scheduler.set_timesteps(num_inference_steps=2, device=pipe.device)
result = pipe(prompt="A cinematic portrait, 35mm", guidance_scale=0.0, num_inference_steps=2).images
```

---

## 日志

* 当 `report_to=wandb` 且 `twinflow_enabled=true` 时，训练器可记录实验性的 TwinFlow 轨迹散点图（σ vs tt vs sign）。该图仅用于调试，在 UI 中会标注 “experimental/theory unverified”。

---

## 排障

* **flow-matching 错误**：TwinFlow 需要 `prediction_type=flow_matching`，除非启用 `diff2flow_enabled` + `twinflow_allow_diff2flow`。
* **需要 EMA**：启用 `use_ema`，或在接受学生当教师时设置 `twinflow_allow_no_ema_teacher: true` / `twinflow_require_ema: false`。
* **1 步质量停滞**：尝试 `twinflow_target_step_count: 2`–`4`，保持 guidance `0.0`，若过拟合则降低 `twinflow_enhanced_ratio`。
* **Teacher/Student 漂移**：RNG 对齐始终开启，漂移应来自模型不匹配而非随机差异。如果 Transformer 不支持 `timestep_sign`，请关闭 `twinflow_enabled`，或先让模型消费该信号再启用 TwinFlow。
