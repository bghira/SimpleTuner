# MiniMax H3 漂移蒸馏

MiniMax H3 是已经蒸馏过的 flow-matching 视频/音频模型。普通 LoRA 或 LyCORIS 训练会让 adapter 学习数据集目标，但也可能让基础 checkpoint 的蒸馏行为发生漂移：guidance 行为、模态平衡以及打包的视频/音频序列布局都会被过度改变。

`h3_drift` 会把 adapter 启用时的 prediction，与同一个模型在 adapter 关闭时的 frozen-base prediction 对齐。它不加载第二个 teacher，也不使用 distillation cache。每个 batch 中 SimpleTuner 会：

1. 用 adapter 启用路径计算正常 MiniMax H3 SFT loss；
2. 暂时关闭 adapter；
3. 用相同 prepared batch 在 `torch.no_grad()` 下运行冻结 base；
4. 对 video/audio prediction 计算 MSE；
5. 重新启用 adapter，并反传组合 loss。

```text
total = sft_loss_weight * normal_h3_loss + loss_weight * frozen_base_prediction_mse
```

启用内部 distiller 时：

```text
total = inner_distiller_loss + sft_loss_weight * normal_h3_loss + loss_weight * frozen_base_prediction_mse
```

## 何时使用

除非你明确想移除或替换原始蒸馏行为，否则 MiniMax H3 LoRA / LyCORIS 训练建议启用它。它适合 style/concept LoRA、FL2VA/Ref2VA、联合音视频训练，以及 `convrot-int8` / `convrot-int4` 等量化 flavour。

Full-rank 不支持 H3 drift。更新整个 transformer 时，不再存在可比较的冻结 base 路径。

## 快速配置

```json
{
  "model_family": "minimaxh3",
  "model_flavour": "convrot-int8",
  "model_type": "lora",
  "distillation_method": "h3_drift",
  "distillation_config": {
    "h3_drift": {
      "loss_weight": 0.5,
      "sft_loss_weight": 1.0,
      "balance": "token",
      "video_weight": 1.0,
      "audio_weight": 1.0
    }
  }
}
```

随仓库提供的 H3 examples 默认启用该 distiller。`loss_weight: 0.5` 会让数据集目标保持主导，同时让 base reference 有足够权重来限制漂移。

## 配置项

- `loss_weight`：冻结 base prediction loss 的倍率。窄域 LoRA 可从 `0.25` 到 `0.5` 开始；如果验证显示 base 行为被破坏，可用 `1.0`。
- `sft_loss_weight`：正常 MiniMax H3 training loss 的倍率。普通 fine-tuning 保持 `1.0`。
- `balance`：`token` 按有效元素数平均；`modality` 按 video/audio 模态均值加权平均。
- `video_weight`：video drift term 权重。
- `audio_weight`：audio drift term 权重。
- `inner_distillation_method`：可选的内部 distiller，会在 `h3_drift` 内执行，例如 `anyflow`、`dmd`、`perflow`、`flow_dpo` 或 `self_forcing`。
- `inner_distillation_config`：传给内部 distiller 的配置。

## 组合另一个 distiller

`h3_drift` 可以包裹另一个 distiller，在使用 step distillation 或偏好目标的同时继续保持 MiniMax H3 的冻结 base 行为：

```json
{
  "distillation_method": "h3_drift",
  "distillation_config": {
    "h3_drift": {
      "loss_weight": 0.5,
      "sft_loss_weight": 1.0,
      "inner_distillation_method": "anyflow",
      "inner_distillation_config": {
        "stage": "forward",
        "diffusion_ratio": 0.5,
        "consistency_ratio": 0.25,
        "central_difference_epsilon": 0.005,
        "loss_weight": 1.0
      }
    }
  }
}
```

该 wrapper 会把 batch preparation、validation scheduler hook、distillation cache、caption batch 支持以及 generator/discriminator 生命周期 hook 委托给内部 distiller。内部 distiller 仍会执行自己的兼容性检查。

即使内部 distiller 重写了 `target`，`sft_loss_weight` 仍表示正常 MiniMax H3 目标。如果内部 distiller 添加 FlowMap/AnyFlow timestep conditioning，H3 drift 会先移除内部 timestep key，再额外运行一次 adapter-enabled forward 来计算 SFT 项。这样 step-distilled 路径可以训练，同时标准 30-step H3 推理路径仍被锚定。

## 视频与音频

`minimax_h3_target_mode: "auto"` 会解析为 video-only。使用 `"video"` 跳过音频目标行，使用 `"av"` 训练联合音视频目标行。也可以在 data backend 中设置 `h3_target_mode` 或 `minimax_h3_target_mode`。

Distiller 跟随 prepared batch：video-only batch 只比较 `model_prediction`；`av` batch 同时比较 video 和 `audio_prediction`；并尊重 `audio_latent_mask`、`sample_weight` 和视觉 mask。

## 保持 CFG 蒸馏行为

MiniMax H3 是 CFG-distilled。基础 checkpoint 通常使用 `validation_guidance: 1.0`、`validation_guidance_real: 1.0`、`validation_disable_unconditional: true` 验证。Negative prompting 不属于基础训练契约。

SimpleTuner 支持 real CFG 和 negative prompt encoding，是因为社区可能会把 H3 重新训练到不再保持原始蒸馏。`h3_drift` 是相反的约束：它让 adapter 靠近 base conditional prediction。如果目标是训练 negative prompt 行为或 de-distill H3，请降低 `loss_weight` 或关闭该 distiller。

## 日志与成本

主要日志包括 `h3_drift_loss`、`h3_drift_video_loss`、`h3_drift_audio_loss`、元素计数、`h3_drift_weighted_loss`、`h3_drift_sft_loss`、启用内部 distiller 时的 `h3_drift_inner_total` 和 `total`。

每个 step 会增加一次 forward pass，但不会在显存中保存第二个 transformer。若包裹 FlowMap/AnyFlow 内部 distiller 且启用 `sft_loss_weight`，还会为 SFT 锚点运行一次正常 adapter forward。它可与 ConvRot、RamTorch、musubi block swap、gradient checkpointing 和 attention offload 一起使用；不过额外 forward 可能改变最快 backend，因此每个 preset 都应重新 benchmark。

## 排错

- low-rank 错误：使用 `model_type: "lora"`。
- audio loss 始终为零：batch 是 video-only、target mode 不是 `av`，或 `audio_latent_mask` 排除了所有音频行。
- adapter 学不到概念：降低 `loss_weight`、提高 rank 或延长训练。
- 音频发生漂移：尝试 `balance: "modality"` 或提高 `audio_weight`。
