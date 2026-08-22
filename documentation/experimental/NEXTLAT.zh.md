# NextLat

NextLat 是一个辅助训练目标，用来教 transformer 让当前 hidden state 能预测下一个 hidden state。

原始 Next-Latent Prediction paper 研究 language-style transformers，并指出标准 next-token prediction 不一定会迫使模型把历史压缩成稳定、紧凑的内部状态。NextLat 添加一个 latent-space self-supervised transition objective：从当前 hidden state 预测下一个 hidden state。在 SimpleTuner 中，它被适配为受支持 transformer family 的实验性 regularizer。

推理不变。NextLat 只添加训练 loss 和一个小型 predictor，不添加新的 sampler。

## ELI5

标准训练说：“根据已经看到的内容，预测下一个 output。”

NextLat 额外说：“也要让你的内部笔记能预测下一条内部笔记。”

对图像、视频和音频模型来说，这些内部笔记就是 transformer 中的 hidden tokens。如果模型学会更连贯的 hidden-state transitions，它可能在 tokens、frames、patches 或 RVQ code positions 之间形成更稳定的计划。

## NextLat 改变什么

训练期间：

1. SimpleTuner 从一个 transformer block 捕获 hidden states。
2. Predictor 接收除最后一个以外的每个 hidden token。
3. Predictor 预测下一个 hidden token。
4. 真实的下一个 hidden token 作为 detached target。
5. 辅助 loss 加到正常训练 loss 上。

基础模型仍按主目标训练。NextLat 是一个 side objective，让内部状态更有预测性。

## 伪代码

```text
for each batch:
    prediction = model(batch)
    main_loss = normal_training_loss(prediction, target)

    hidden = captured_hidden_states
    current_hidden = hidden tokens 0..N-2
    next_hidden = hidden tokens 1..N-1

    predicted_next_hidden = nextlat_predictor(current_hidden)
    nextlat_loss = distance(predicted_next_hidden, stop_gradient(next_hidden))

    total_loss = main_loss + nextlat_weight * nextlat_loss
    train_on(total_loss)
```

如果模型 family 提供兼容 logits head，也可以启用可选 KL：

```text
predicted_logits = logits_head(predicted_next_hidden)
target_logits = logits_head(stop_gradient(next_hidden))
total_loss += nextlat_kl_weight * agreement_loss(predicted_logits, target_logits)
```

大多数用户应保持 `nextlat_kl_weight=0`。

## SimpleTuner 中的行为

- 适用于暴露 hidden states 的 transformer families。
- 通过 `nextlat_block_index` 选择一个 block。
- `-1` 表示最后一个受支持的 block。
- 将 image、video、audio 或 token hidden states 展平为序列。
- 按 hidden-token order 预测一步。
- Target hidden state 会 detach。
- 当训练模式支持保存时，predictor 会作为额外可训练模块保存。

除非模型指南说明其他 adapter mode 受支持，建议使用标准 PEFT LoRA 或 full-model training。

## 快速设置

### WebUI

1. 打开 **Training → Loss functions**。
2. 启用 **NextLat**。
3. 首次运行保持 **NextLat Block Index** 为 `-1`。
4. 将 **NextLat Weight** 设为较小正值。
5. 保持 **NextLat State Loss** 为 `smooth_l1`。
6. 除非模型指南建议，否则保持 **NextLat KL Weight** 为 `0`。

### Config JSON / CLI

```json
{
  "nextlat_enabled": true,
  "nextlat_block_index": -1,
  "nextlat_weight": 0.05,
  "nextlat_state_loss": "smooth_l1",
  "nextlat_kl_weight": 0.0
}
```

## 设置项

- `nextlat_enabled`：启用 NextLat。
- `nextlat_block_index`：zero-based transformer block；`-1` 使用最后一个受支持 block。
- `nextlat_weight`：辅助 hidden-state prediction loss 的 multiplier；启用时必须大于 0。
- `nextlat_state_loss`：`smooth_l1` 默认，也可用 `mse`。
- `nextlat_kl_weight`：当 family 提供兼容 logits head 时的可选 KL weight。

## 如何选择值

| 场景 | 建议起点 |
| --- | --- |
| 第一次 transformer LoRA | `nextlat_block_index=-1`, `nextlat_weight=0.02` 到 `0.05` |
| AR/RVQ planner | late block、`smooth_l1`、小权重 |
| Video transformer | 如果最后 block 约束太强，试 middle-to-late block |
| 辅助 loss 不稳定 | 先降低 `nextlat_weight`，再考虑换 block |
| 模型指南推荐 KL | 只使用文档给出的值 |

## Logs

- `nextlat_loss`：加入训练目标的加权辅助 loss。
- `nextlat_state_loss`：原始 hidden-state prediction loss。
- `nextlat_kl_loss`：可选 KL term。

原始 state loss 主要用于看趋势，不需要和主 loss 在同一尺度。

## 兼容性

当前支持情况见 [Quick Start](../QUICKSTART.zh.md) 的功能表。

要求：

- 模型必须暴露 transformer hidden states。
- 选择的 block 必须存在且可捕获。
- 捕获序列至少要有两个 hidden tokens。
- 训练模式必须能保存 NextLat predictor。

NextLat 可以自然地与 LayerSync、Internal Guidance、CREPA 等 hidden-state 功能配合，但会增加显存，因为 hidden states 要保留到辅助 loss 计算完成。

## 预期效果

NextLat 更可能帮助需要连贯内部转移的任务：RVQ/audio code planners、有时间结构的 video transformers、token order 含有空间结构的 image transformers，以及需要稳定内部计划的 multimodal models。

在非常小的实验、辅助权重压过主 loss、或 family 没有暴露有用 hidden states 时，效果可能较弱。

## 实用建议

- 先做短 ablation run。
- `nextlat_weight` 从小值开始。
- 除非有特别理由，使用 `smooth_l1`。
- 先用 `-1`，必要时试 middle-to-late block。
- 没有模型指南时不要启用 KL。
- 如果 VRAM 增长过多，降低 batch size 或关闭其他 hidden-state regularizers。

## 参考

- [Next-Latent Prediction paper](https://arxiv.org/abs/2511.05963)
- [NextLat reference code](https://github.com/JaydenTeoh/NextLat)
