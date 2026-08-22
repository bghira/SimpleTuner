# Explorative Modeling (XM)

Explorative Modeling，在 SimpleTuner 中简称 XM，是一种训练时方法：对同一个监督样本尝试多个隐藏候选，然后只从最能解释目标的候选学习。

原始 Explorative Modeling 工作把 exploration 看作生成模型的第三个扩展轴：除了更多数据和更多参数，也可以把更多训练算力用于探索更多候选。在 SimpleTuner 中，XM 是面向受支持的图像、视频、音频和自回归模型族的实验性训练目标。

推理不变。XM 只改变训练 batch 的构造、评分和 loss reduction。

## ELI5

可以把它想成画图练习：不是只画一次就评分，而是允许先画四个草稿，然后只评价最接近目标的那个草稿。

XM 的核心流程：

1. 为同一个 sample 创建多个候选。
2. 让模型处理所有候选。
3. 将每个候选与真实 target 比较。
4. 按 sample 或 token block 选择最好的候选。
5. 只对选中的 loss 做 backprop。

当目标可以有多种合理解释时，单一路径容易让模型学到平均化结果；多个探索路径可以让模型选择一个合理 mode。

## XM 改变什么

XM 不添加新的推理 sampler、不改变 checkpoint 格式，也不需要第二个 teacher model。它改变训练时的候选选择：

- 标准训练只采样一个候选并从它学习。
- XM 采样 `K` 个候选，并从 loss 最低的候选学习。
- 更大的 `K` 提供更多 exploration，但训练 compute 更高。

对 diffusion 和 flow models，候选通常是在采样 timestep 上构造 noised latent 的 noise。

对 autoregressive token models，例如 RVQ/audio planners，候选是 learned route embedding，让模型对同一个监督 token sequence 拥有多个内部路径。

## SimpleTuner 中的行为

### Diffusion 和 Flow Models

受支持的 diffusion 或 flow matching family 使用 `xm_training_target=noise`。

SimpleTuner 会：

1. 采样常规训练 timestep 或 sigma。
2. 将 batch 重复 `xm_candidate_count` 次。
3. 为每个候选生成不同 noise tensor。
4. 用每个候选 noise 构造 noised latents。
5. 在扩展后的 candidate batch 上运行模型。
6. 为每个候选计算正常训练 loss。
7. 对每个原始 sample 选择 lowest-loss candidate。
8. 对选中的 loss 做 backprop。

模型仍学习自己的正常 prediction type：flow velocity、epsilon、v-prediction 或 sample prediction。

### Autoregressive 和 RVQ Models

受支持的 autoregressive planner 使用 `xm_training_target=route`。

SimpleTuner 会：

1. 添加一个小型 learned route embedding table。
2. 将每个监督 token sequence 展开到多个 route candidates。
3. 把 route signal 插入模型输入。
4. 为每条 route 计算 token losses。
5. 对整个 sample 或配置的 token blocks 选择最佳 route。
6. 只对选中的 route loss 做 backprop。

这适合预测 RVQ audio codes 或其他离散 token streams 的 global LM 风格 planner。Route embedding 在不改变推理解码的情况下，为同一个 target sequence 提供多个内部解释。

## 伪代码

```text
for each batch:
    candidates = []

    for candidate_id in 1..K:
        candidate_input = make_candidate(batch, candidate_id)
        prediction = model(candidate_input)
        loss = compare(prediction, target)
        candidates.append(loss)

    selected_loss = minimum_loss_per_sample_or_block(candidates)
    train_on(selected_loss)
```

Diffusion:

```text
candidate_input = add_noise(clean_latent, random_noise_candidate, timestep)
loss = diffusion_or_flow_loss(model(candidate_input), training_target)
```

Autoregressive route selection:

```text
candidate_input = add_route_embedding(token_sequence, route_candidate)
loss = token_loss(model(candidate_input), target_tokens)
```

## 快速设置

### WebUI

1. 打开 **Training → Loss functions**。
2. 启用 **XM**。
3. 将 **XM Candidates** 设为 `2` 或 `4`。
4. 选择 **XM Training Target**：
   - diffusion 或 flow 使用 `noise`。
   - autoregressive/RVQ planner 使用 `route`。
5. 除非模型指南建议，否则保持 **XM Selection Scope** 为 `sample`。
6. 除非使用 route block selection，否则保持 **XM Block Size** 为 `0`。

### Config JSON / CLI

```json
{
  "xm_enabled": true,
  "xm_candidate_count": 4,
  "xm_training_target": "noise",
  "xm_selection_scope": "sample",
  "xm_block_size": 0
}
```

AR/RVQ route training:

```json
{
  "xm_enabled": true,
  "xm_candidate_count": 4,
  "xm_training_target": "route",
  "xm_selection_scope": "block",
  "xm_block_size": 16
}
```

## 设置项

- `xm_enabled`：启用 XM。
- `xm_candidate_count`：每个 sample 的候选数；启用 XM 时至少为 `2`。
- `xm_training_target`：候选类型，diffusion/flow 用 `noise`，token planner 用 `route`。
- `xm_selection_scope`：winner 选择粒度。`sample` 按整个 sample 选；`block` 在支持的 family 中按 token/frame block 选。
- `xm_block_size`：block-level selection 的 token 或 frame span。`0` 表示完整监督序列。

## 如何选择值

| 场景 | 建议起点 |
| --- | --- |
| 图像或视频 diffusion LoRA | `xm_candidate_count=2`, `xm_training_target=noise`, `xm_selection_scope=sample` |
| 数据集歧义更强或 batch 更大 | 尝试 `xm_candidate_count=4` |
| RVQ/audio planner | `xm_training_target=route`, `xm_selection_scope=block`，block size 按模型指南 |
| 新 family 第一次尝试 | 保持 block size `0`，与非 XM baseline 比较验证结果 |

候选数增加时，成本通常近似线性增长。

## Logs

XM 可能记录：

- `xm_loss`：选择后的 loss。
- `xm_candidate_loss_mean`：选择前候选平均 loss。
- `xm_candidate_0_wins`, `xm_candidate_1_wins`：各候选胜出的次数。
- `xm_route_usage`：AR/RVQ route 使用情况。

好的信号：多个候选都有胜出，validation 改善，route usage 不长时间 collapse。

需要注意：从一开始只有一个候选总是胜出，training loss 下降但 validation 变差，或者显存/step time 成本过高。

## 兼容性

当前 family-level support 见 [Quick Start](../QUICKSTART.zh.md) 的功能表。

一般规则：

- Diffusion/flow XM 使用 noise candidates 和 sample-level selection。
- AR/RVQ XM 使用 route candidates，并可能支持 block-level selection。
- 不支持的 family 会明确报错。

对 diffusion noise-candidate XM，除非 family 明确支持，否则 SimpleTuner 目前认为 TwinFlow、Scheduled Sampling、`input_perturbation`、CREPA self-flow 和 stochastic segmentation masked loss 不兼容。

## 与其他功能的关系

- **MixFlow** 改变 flow model 的训练轨迹；XM 改变候选选择。
- **Diff2Flow** 改变 legacy diffusion model 的 target。
- **NextLat** 正则化 hidden-state dynamics；XM 选择 route 或 noise candidate。
- **LayerSync 和 CREPA** 对齐 representations；XM 选择最能解释 target 的候选。

## 实用建议

- 比较 XM 和 baseline 时固定 validation seeds。
- 如果 `xm_candidate_count` 带来显存压力，降低 batch size。
- 不要只看 training loss；更应看 validation 和 sample diversity。
- AR/RVQ 中除非指南建议，否则避免 block size `1`。
- 先做短 ablation：同模型、同数据、同 seed，只切换 XM。

## 参考

- [Explorative Modeling 项目页](https://explorative-modeling.github.io/)
- [Explorative Modeling paper](https://arxiv.org/abs/2607.27372)
