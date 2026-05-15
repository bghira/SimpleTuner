# Flow-DPO 与 Masked Flow-DPO

Flow-DPO 是面向 flow-matching 模型的实验性蒸馏方法，用 preferred/rejected 成对样本训练低秩 adapter。SimpleTuner 只支持 LoRA/LyCORIS，不支持 full-model Flow-DPO；所有蒸馏方法都会禁止 text encoder 训练。

SimpleTuner 复用现有 reference dataset 系统：普通 `image` 或 `video` 数据集提供 preferred 样本，配对的 `conditioning` 数据集用 `conditioning_type=reference_strict` 提供 rejected 样本。参考 [`conditioning_type`](../DATALOADER.zh.md#conditioning_type) 和 [`conditioning_data`](../DATALOADER.zh.md#conditioning_data)。

## 工作方式

每个 batch 中，SimpleTuner 会：

1. 用启用 adapter 的模型预测 preferred latents。
2. 用相同 prompt、noise、timestep 预测 rejected latents。
3. 关闭 LoRA/LyCORIS adapter，作为冻结 reference 再预测 preferred 和 rejected。
4. 使用 Flow-DPO margin loss：

```text
win_adv  = L(reference_win, target_win) - L(policy_win, target_win)
lose_adv = L(policy_lose, target_lose) - L(reference_lose, target_lose)
loss     = -logsigmoid(beta / 2 * (win_adv + lose_adv))
```

flow-matching target 为 `noise - latents`。

## Masked Flow-DPO

如果 batch 同时包含 `conditioning_type=mask` 或 `conditioning_type=segmentation` 的 conditioning 数据集，SimpleTuner 会在 reduction 前把 mask 应用到 DPO prediction error，使偏好信号集中在变化区域。

`anchor_alpha` 可在 preferred 与 rejected 样本上同时添加全局 MSE regularizer，对比 adapter-enabled 与 adapter-disabled 预测。该 anchor 不使用 mask，因此约束的是整帧 drift。

## 配置

最小配置：

```bash
--model_type=lora
--distillation_method=flow_dpo
--flow_custom_timesteps=801,694,548,338
--flow_timesteps_mode=round-robin
```

常用 `distillation_config`：

```json
{
  "flow_dpo": {
    "beta": 1.0,
    "auto_beta": true,
    "auto_beta_target_gf": 0.2,
    "auto_beta_decay": 0.99,
    "norm_type": "sum",
    "mask_dilate": 1,
    "anchor_alpha": 0.0,
    "sft_loss_weight": 0.0
  }
}
```

- `norm_type=sum` 对应常见 Flow-DPO 公式。`mean` 会平均所有 latent 元素，`masked_mean` 会在存在 mask 时平均 active mask 元素。
- `auto_beta=true` 会根据 margin 运行均值调整 beta，适合小型配对数据集。
- `flow_timesteps_mode=fixed-list` 从 `flow_custom_timesteps` 随机采样。
- `flow_timesteps_mode=round-robin` 循环使用 `flow_custom_timesteps`，让 timestep 覆盖更均匀。分布式 rank 会使用不同 offset，恢复训练时会从 `global_step` 初始化 cursor。
- `sft_loss_weight` 默认为 `0.0`，不会混入普通 diffusion loss。

SimpleTuner 会记录核心 Flow-DPO 健康指标：beta、margin、win/lose advantage、policy/reference error、negative-margin percentage 和 gradient factor。原 demo model card 中的扩展 reward-hacking detector 指标属于该发布的分析工具，SimpleTuner 目前还不会全部输出。

## 数据集形状

rejected 数据集必须以 `reference_strict` 与 preferred 数据集配对：

```json
[
  {
    "id": "preferred",
    "dataset_type": "image",
    "type": "local",
    "instance_data_dir": "/data/win",
    "conditioning_data": ["rejected"]
  },
  {
    "id": "rejected",
    "dataset_type": "conditioning",
    "conditioning_type": "reference_strict",
    "type": "local",
    "instance_data_dir": "/data/lose",
    "source_dataset_id": "preferred"
  }
]
```

使用 masked Flow-DPO 时，把 mask conditioning 数据集也加入同一个 `conditioning_data` 列表。

## 限制

Flow-DPO 当前要求：

- flow-matching 模型。
- `model_type=lora`。
- 配对的 `reference_strict` conditioning 数据集。
- 不训练 text encoder。

它不会加载第二份完整模型权重。reference pass 会关闭训练中的 adapter（包括 LyCORIS multiplier），然后为 policy 路径重新开启。
