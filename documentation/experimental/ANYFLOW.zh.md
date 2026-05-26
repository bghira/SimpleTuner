# AnyFlow

AnyFlow 是面向 flow-matching 模型的实验性蒸馏模式。它让模型同时条件化在普通训练 timestep `t` 和更低的参考 timestep `r` 上，使网络学习一个跨 interval 的 flow map，而不是只学习单点 rectified-flow velocity。

在 SimpleTuner 中：

- `--distillation_method=anyflow` 启用 `AnyFlowDistiller`。
- distiller 在启动时调用训练组件的 `enable_flowmap_time_conditioning()`。
- 每个 prepared batch 会加入 `flowmap_r_timesteps`。
- 常规 target 会在计算 model loss 之前替换为 AnyFlow target。

SimpleTuner 的 AnyFlow 是 online 的，不需要预先计算 ODE cache。

关于使用 NVIDIA 发布的 AnyFlow checkpoints 继续训练 Wan 的示例，见 [AnyFlow 继续训练快速入门](/documentation/quickstart/ANYFLOW.zh.md)。

## 快速配置

```json
{
  "model_type": "lora",
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

SimpleTuner 的所有 distillation methods 都会禁止 text encoder training，AnyFlow 也一样。

## 工作方式

对每个 flow-matching batch，SimpleTuner 会：

1. 使用正常的 `prepare_batch()` 采样 `sigmas`、`timesteps`、`noisy_latents` 和 base flow target。
2. 从当前 interval 中采样 `r < t`。
3. 将 `flowmap_r_timesteps` 写入 batch，让 model wrapper 作为 `r_timestep` 传入。
4. 构建训练 target。
5. 使用正常 model loss 比较 prediction 和 target。

在 `target_mode=online_teacher` 下，target 是从 `t` 处 noisy latent 指向 `r` 的平均 velocity。LoRA 和 LyCORIS 训练时，distiller 会在 teacher rollout 期间临时禁用 adapter，然后再启用。

在 `target_mode=linear` 下，不使用 teacher rollout。target 是 straight flow target `noise - latents`。它适合 smoke test 和 ablation，但不是完整的 AnyFlow teacher-map objective。

## 选项

- `target_mode`：`online_teacher` 或 `linear`。默认：`online_teacher`。
- `teacher_rollout_steps`：`t` 到 `r` 之间的 online teacher Euler steps。默认：`1`。
- `r_timestep_sampler`：`uniform` 或 `zero`。默认：`uniform`。
- `min_interval_ratio`：`t` 和 `r` 之间保留的最小 normalized interval。默认：`0.02`。
- `gate_value`：FlowMap delta timestep embedding 的混合权重。默认：`0.25`。
- `deltatime_type`：`r` 或 `t-r`。默认：`r`。
- `loss_weight`：已计算 training loss 的乘数。默认：`1.0`。
- `timestep_scale`：用于自定义 timestep scale 的模型。通常保持未设置。

## 限制

- 需要 flow-matching 模型。
- 需要每个样本一个 scalar timestep。Tokenwise AnyFlow intervals 还没有接入。
- 需要 `r_timestep < timestep`；timestep zero 会被拒绝。
- 当前 online teacher 模式面向 LoRA/LyCORIS。Full-rank online teacher 需要单独的 student/teacher wiring。
- 标准 validation 仍可不传 `r_timestep` 运行，但 AnyFlow 式 few-step sampling 需要 sampler 或 pipeline 支持，把 interval endpoint 作为 `r_timestep` 传入。这个 generation-time integration 仍是后续工作。
