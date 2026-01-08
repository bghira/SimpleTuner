# Scheduled Sampling（Rollout）

## 背景

标准扩散训练依赖“Teacher Forcing”。我们取一张干净图像，加入精确量级的噪声，并让模型预测该噪声（或速度/原图）。输入始终是“完美”的噪声——严格落在理论噪声日程上。

然而在推理（生成）阶段，模型会以自身输出作为下一步输入。如果在 $t$ 步出现小误差，该误差会传到 $t-1$，逐步累积，导致生成偏离有效图像流形。训练（完美输入）与推理（不完美输入）的差异称为 **Exposure Bias**。

**Scheduled Sampling**（此处也称“Rollout”）通过在训练中让模型使用自身输出来缓解这一问题。

## 工作原理

训练循环会偶尔执行一次小型推理，而不是直接对干净图像加噪：

1.  选择 **目标时间步** $t$（我们想训练的步）。
2.  选择 **源时间步** $t+k$（更靠后的、更噪的步）。
3.  使用模型 *当前* 权重，从 $t+k$ 实际生成（去噪）到 $t$。
4.  将该自生成的、略有瑕疵的 $t$ 步 latent 作为训练输入。

这样模型会看到自己生成的真实误差与伪影，并学会“纠正自身错误”，把生成拉回到有效路径。

## 配置

该功能为实验性，会增加计算开销，但能显著提升提示词遵循度与结构稳定性，尤其在小数据集（Dreambooth）上。

启用它需要设置非零的 `max_step_offset`。

### 基础设置

在 `config.json` 中加入：

```json
{
  "scheduled_sampling_max_step_offset": 10,
  "scheduled_sampling_probability": 1.0,
  "scheduled_sampling_sampler": "unipc"
}
```

### 选项参考

#### `scheduled_sampling_max_step_offset`（Integer）
**默认：** `0`（禁用）
最大 rollout 步数。设置为 `10` 时，每个样本会在 0–10 之间随机选择 rollout 长度。
> 🟢 **建议：** 从较小值开始（如 `5` 到 `10`）。短 rollout 也能帮助模型学习纠错且不会显著拖慢训练。

#### `scheduled_sampling_probability`（Float）
**默认：** `0.0`
任意 batch 样本进行 rollout 的概率（0.0–1.0）。
*   `1.0`：每个样本都 rollout（计算最重）。
*   `0.5`：50% 样本为标准训练，50% 为 rollout。

#### `scheduled_sampling_ramp_steps`（Integer）
**默认：** `0`
若设置，将在指定步数内把概率从 `scheduled_sampling_prob_start`（默认 0.0）线性提升到 `scheduled_sampling_prob_end`（默认 0.5）。
> 🟢 **提示：** 这相当于“热身”。让模型先学会基本去噪，再引入纠错任务。

#### `scheduled_sampling_sampler`（String）
**默认：** `unipc`
rollout 生成步骤使用的求解器。
*   **可选项：** `unipc`（推荐，快且准）、`euler`、`dpm`、`rk4`。
*   `unipc` 通常在这些短采样中速度/精度平衡最好。

### Flow Matching + ReflexFlow

对于 flow-matching 模型（`--prediction_type flow_matching`），Scheduled Sampling 支持 ReflexFlow 风格的曝光偏差缓解：

*   `scheduled_sampling_reflexflow`：在 rollout 中启用 ReflexFlow 增强（当 scheduled sampling 激活且为 flow-matching 模型时自动启用；传 `--scheduled_sampling_reflexflow=false` 可关闭）。
*   `scheduled_sampling_reflexflow_alpha`：基于曝光偏差的损失权重缩放（频率补偿）。
*   `scheduled_sampling_reflexflow_beta1`：方向性抗漂移正则项缩放（默认 10.0，与论文一致）。
*   `scheduled_sampling_reflexflow_beta2`：频率补偿损失缩放（默认 1.0）。

这些会复用已计算的 rollout 预测/latent，无需额外梯度传递，并帮助偏差 rollout 与干净轨迹对齐，同时在去噪早期强调缺失的低频分量。

### 性能影响

> ⚠️ **警告：** 启用 rollout 需要在训练循环中运行推理。
>
> 若设置 `max_step_offset=10`，每个训练步可能额外执行最多 10 次前向。这会降低 `it/s`（每秒迭代数）。请通过 `scheduled_sampling_probability` 平衡训练速度与质量收益。
