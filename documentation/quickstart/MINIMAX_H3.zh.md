# MiniMax H3 快速开始

MiniMax H3 是 33B flow-matching 视频/音频模型。SimpleTuner 通过 `minimaxh3` model family 支持 adapter 训练，包括 FL2VA 首/末帧 conditioning 和量化 ConvRot flavour。

## 起始配置

建议从这些 examples 开始：

- `simpletuner/examples/minimaxh3-fl2va-convrot-int8.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-24g.peft-lora+ramtorch`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-32g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-48g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-80g.peft-lora`

先选择最接近显存的 preset，完成 smoke test 后再调整分辨率、帧数、attention backend 和 checkpointing。

## 核心设置

```json
{
  "model_family": "minimaxh3",
  "model_flavour": "convrot-int8",
  "model_type": "lora",
  "mixed_precision": "bf16",
  "base_model_precision": "no_change",
  "text_encoder_1_precision": "int8-quanto",
  "validation_disable_unconditional": true,
  "validation_guidance": 1.0,
  "validation_guidance_real": 1.0
}
```

Examples 默认使用 `convrot-int8`。如果需要更低 precision checkpoint，可以在同一 model family 中使用 `convrot-int4`。

## 维持蒸馏

MiniMax H3 是 CFG-distilled。基础 checkpoint 预期不使用 unconditional branch，因此 examples 使用 guidance `1.0` 并设置 `validation_disable_unconditional: true`。

Adapter 训练仍可能偏离基础蒸馏行为。因此 examples 默认启用 `h3_drift`：

```json
{
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

该 distiller 会在 adapter 关闭时运行 no-grad frozen-base reference pass，并惩罚 video/audio prediction 的漂移。普通 H3 LoRA 建议保持启用。adapter 学不到概念时降低 `loss_weight`；验证开始丢失基础行为时提高它。完整说明见 [MiniMax H3 Drift Distillation](../distillation/MINIMAX_H3_DRIFT.zh.md)。

Negative prompting 不属于基础 H3 契约。SimpleTuner 为 de-distilled checkpoints 保留 real CFG 和 negative prompt 支持，但 `h3_drift` 会保留原始蒸馏条件行为。

## 音频目标模式

`minimax_h3_target_mode: "auto"` 会解析为 video-only，并避免 audio VAE 工作：

```json
{
  "minimax_h3_target_mode": "video"
}
```

只有当 dataset 有 target audio latents 并需要联合音视频训练时才使用 `"av"`。也可以在 data backend 中设置 `h3_target_mode` 或 `minimax_h3_target_mode`。

## 显存选项

- 显存紧张时使用 24G RamTorch example。
- 在加重 checkpointing 前测试 `musubi_blocks_to_swap`。
- 保持 VAE tiling、slicing 和 temporal roll 启用。
- 在目标 GPU 上 benchmark `attention_mechanism`。
- 修改 `torch.compile` 后重新 smoke test，因为 compile cache 可能改变峰值显存。

## 运行

```bash
simpletuner train example=minimaxh3-fl2va-convrot-int8.peft-lora
```

长训练前先跑短 smoke test，确认 `h3_drift_loss`、正常 loss 和验证样本走势一致。
