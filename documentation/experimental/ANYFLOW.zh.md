# AnyFlow

SimpleTuner 将 NVIDIA AnyFlow 实现为面向 flow-matching 模型的两个显式训练阶段。这两个阶段都会训练一个模型，使其同时接收当前 flow 时间 `t` 和区间端点 `r`。

- `stage=forward` 实现 NVIDIA 的 forward MeanFlow 目标。
- `stage=onpolicy` 在共同训练 forward 目标的同时，实现 Flow Map Backward Simulation 和 on-policy DMD。

已移除的 `online_teacher` 和 `linear` target mode 是 SimpleTuner 过去的自定义目标，现在不再接受。

使用 NVIDIA 已发布 checkpoint 继续训练 Wan 的示例，请参阅
[AnyFlow Continuation Quickstart](/documentation/quickstart/ANYFLOW.zh.md)。


Qwen Image 2.1 支持 FlowMap 区间条件。[三阶段 Qwen 试验](../quickstart/QWEN_IMAGE.zh.md) 混合 CC12M 长描述、e621 和少量 Domokun 样本，随后进行带正则化的简短微调。基础预测锚点用于测试能力保留，并不能证明 Qwen 经过了引导蒸馏。

编译后的 AnyFlow 训练会临时为每个 Dynamo frame 允许至少 32 个变体，以涵盖教师、生成器、判别器以及不同梯度模式、padding 和 batch size。用户设置的更大限制会保留，运行结束后恢复原限制。在 Torch 2.11 中，用尽默认的 8 个变体可能使检查点重计算转为 eager 执行，并导致保存张量的元数据检查失败。描述长度可变时请使用 `dynamo_dynamic: true`；Qwen 试验已启用该设置。

AnyFlow 检查点为每个进程保存 `anyflow_rng_state_<rank>.pt`，以保留区间采样和 on-policy rollout 使用的独立随机数生成器状态。恢复训练时，阶段、种子、数据集设置和分布式拓扑必须保持一致。缺少该状态的旧检查点无法精确续训，因此会被拒绝；可通过 `init_lora` 加载其适配器，并使用新的优化器开始新训练。

`init_lora` 默认读取适配器保存的训练步数。开始新阶段时设置 `init_lora_step: 0`，使训练步数预算和学习率调度器从零开始。

导出适配器时会从张量名称中移除区域编译包装层，并保留区间嵌入张量。普通模型和编译模型均可重新加载。当 `diffusion_ratio: 1.0` 时，所有区间都满足 `r=t`，目标准备会跳过贡献为零的两次有限差分预测。

## Forward 阶段

```json
{
  "model_type": "lora",
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "stage": "forward",
      "diffusion_ratio": 0.5,
      "consistency_ratio": 0.25,
      "central_difference_epsilon": 0.005,
      "meanflow_weight_type": "beta08",
      "meanflow_adaptive_weighting": true,
      "gate_value": 0.25,
      "deltatime_type": "r",
      "loss_weight": 1.0
    }
  }
}
```

对于每个全局 batch，forward 阶段会：

1. 采样两个均匀 flow 时间，并排序为 `t >= r`。
2. 将 50% 样本分配到 diffusion 区间 (`r=t`)，25% 分配到 endpoint 区间 (`r=0`)，其余分配到任意区间。
3. 对两个端点应用模型 scheduler 的 flow shift。
4. 沿直线 latent flow 路径计算中心差分。
5. 构造 MeanFlow tangent target，并应用 NVIDIA 归一化的 `beta08` timestep weighting。
6. 将每个非 diffusion 样本与全局 diffusion 分支 loss 均值进行平衡。

## On-Policy 阶段

通过 `init_lora` 加载 forward 阶段 AnyFlow adapter，并使用新的优化器状态开始此阶段。仅在阶段、数据集配置和分布式拓扑均不变时恢复训练状态 checkpoint：

```json
{
  "model_type": "lora",
  "lora_type": "standard",
  "init_lora": "path-or-repo-to-forward-anyflow-adapter",
  "init_lora_step": 0,
  "learning_rate": 0.000002,
  "optimizer_beta1": 0.0,
  "optimizer_beta2": 0.999,
  "optimizer_config": "weight_decay=0.0",
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "stage": "onpolicy",
      "cotrain_forward": true,
      "rollout_step_counts": [2, 4, 8, 16, 50],
      "dmd_weight": 1.0,
      "dmd_batch_size": 1,
      "real_score_guidance_scale": 0.0,
      "discriminator_lr": 0.000002,
      "discriminator_betas": [0.0, 0.999],
      "discriminator_weight_decay": 0.0,
      "discriminator_grad_clip": 1.0
    }
  }
}
```

on-policy 阶段使用三个 score 角色。标准 LoRA 训练会在它们之间共享一个冻结的 base transformer：

- 已加载的 AnyFlow adapter 是 generator。
- 禁用 adapter 的 base model 是冻结的 real score。
- 单独优化的 `anyflow_discriminator` adapter 是 fake score。

每次 generator 更新都会从 `rollout_step_counts` 中选择 rollout 预算，执行可微 FlowMap rollout，在一个 shifted uniform
时间点给生成 latent 加噪，并应用 NVIDIA 的归一化 DMD 梯度。每次 discriminator 更新都会执行无梯度 student rollout，采样
logit-normal shifted 时间，并用普通 flow target 训练 fake score。discriminator adapter 和 optimizer 会随每个 SimpleTuner
checkpoint 保存为 `anyflow_discriminator.safetensors` 和 `anyflow_discriminator_optim.pt`。

MiniMax-H3 已经包含 CFG distillation，因此它的 on-policy 运行通常应保持 `real_score_guidance_scale=0`。需要外部 real-score
CFG pass 的模型必须缓存 negative text embeddings，并可以显式设置该 scale。

设置 `--seed` 时，AnyFlow 会从按设备隔离的 Torch generator 中采样 MeanFlow 区间、rollout schedule、rollout latent、DMD
noise 和 DMD sigma。这样即使无关训练代码消耗了全局 Torch RNG，AnyFlow 样本也会保持稳定。这不会让 CUDA attention
backward 达到 bit-stable。

## 共享配置

- `stage`: `forward` 或 `onpolicy`。默认：`forward`。
- `diffusion_ratio`: 使用 `r=t` 的全局 batch 比例。默认：`0.5`。
- `consistency_ratio`: 使用 `r=0` 的全局 batch 比例。默认：`0.25`。
- `central_difference_epsilon`: 归一化 shifted-time offset。默认：`0.005`，对应 NVIDIA 的 `5/1000`。
- `meanflow_weight_type`: `beta08` 或 `uniform`。默认：`beta08`。
- `meanflow_adaptive_weighting`: 将非 diffusion 样本与 diffusion 分支平衡。默认：`true`。
- `gate_value`: FlowMap delta-timestep embedding 混合权重。默认：`0.25`。
- `deltatime_type`: `r` 或 `t-r`。默认：`r`。
- `loss_weight`: forward MeanFlow loss 乘数。默认：`1.0`。

## 限制

- AnyFlow 需要 flow-matching 模型，并且该模型要有模型专用的 FlowMap 区间 conditioning。
- on-policy 训练目前需要标准 PEFT LoRA。共享 base 可以避免在每个 DDP rank 上分配 generator、real-score 和 discriminator 三份大型 transformer。
- 目前拒绝 MiniMax-H3 audio-video 联合训练。video 使用 schedule shift 12，audio 使用 shift 3；需要先实现原生双 schedule MeanFlow target 和 rollout，AV 训练才有效。
- 所有 SimpleTuner distillation 方法都禁用 text encoder training。
- 验证使用 `AnyFlowValidationScheduler`，它会把下一个区间端点传给已注册的 FlowMap 模型组件。

## Logs

Forward 训练会添加 `anyflow_forward_loss`、timestep 和 interval 值，以及全局分支比例。On-policy 训练还会添加
`anyflow_dmd_loss`、`anyflow_dmd_gradient_norm`、`anyflow_dmd_sigma` 和 `anyflow_rollout_steps`。
