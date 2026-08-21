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
  "flow_schedule_shift": 12.0,
  "audio_flow_schedule_shift": 3.0,
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

仅音频训练只需设置 `dataset_type: "audio"`。H3 声明支持 fake video，因此 SimpleTuner 会在规范化后的 backend
配置中记录 `audio.audio_only: true`，构建占位视频流，并屏蔽视频 loss。仍可显式设置 `audio_only`，但这不是必需的。

## 上下文并行

H3 context parallelism 使用 Ulysses 和 `context_parallel_strategy: "alltoall"`。packed sequence 可能会 padding 到
CP degree，因此本地 attention backend 必须支持 mask。`native` 和 `cudnn` 受支持；启用 CP 时，SimpleTuner 会把
其他 backend 替换为 `native`。

在约 8k audio tokens 时，CP 主要用通信开销换取更低的 activation memory 和更轻的 checkpointing。CP 本身不
shard weights，因此除非 sequence 更长或与 FSDP 组合使用，否则应与 DDP 做 benchmark。

## 实验性 Sparse Attention

MiniMax 表示 H3 在最终训练阶段对视频 token 使用了 MoBA 风格的 3D sparse attention。初始公开版本使用 dense attention，MiniMax 还没有发布准确的 block shape、retention budget、layer schedule 或生产 kernel。因此 SimpleTuner 默认关闭这个实验性近似实现。

```json
{
  "minimax_h3_sparse_attention": "moba3d",
  "minimax_h3_sparse_block_shape": "1,8,16",
  "minimax_h3_sparse_video_kv_fraction": 0.25,
  "minimax_h3_sparse_share_heads": false,
  "minimax_h3_sparse_start_layer": 0
}
```

该实现会对 3D query/key 视频块做 mean-pooling，然后进行无参数 top-k routing。目标视频 queries 仍然对文本、音频和 reference context 保持 dense access；非目标 queries 仍然是 dense。block dimensions 的乘积必须为 128。video KV fraction 为 `1.0` 时，是通过 FlexAttention 的 dense-connectivity 数值对照。

该模式当前需要 CUDA，并会在 FlexAttention 周围引入 Dynamo graph boundary。Ulysses context parallelism 支持 `context_parallel_strategy=alltoall`；ring context parallelism 和 TREAD 不支持。在 480px 下，由于目标 lattice 和 packed context 必须 padding 和 reorder，sparse routing 可能比 FlashAttention 使用更多显存。在 MiniMax 发布参考实现前，请把它视为 fine-tuning ablation，而不是确定的加速选项。

## 显存选项

- 显存紧张时使用 24G RamTorch example。
- 在加重 checkpointing 前测试 `musubi_blocks_to_swap`。
- video `flow_schedule_shift` 保持 `12.0`，`audio_flow_schedule_shift` 保持 `3.0`。H3 helper 会修正继承来的全局 video 默认值 `3.0`，因为它不匹配 MiniMax H3 schedule。
- SimpleTuner 会强制启用 H3 video VAE tiling 和 temporal roll/chunking。tiling 几何与 upstream 一致，使用 `256` tile size 和 `64` overlap；把这些选项设为 false 会被忽略，因为未 tiling 的 decode 可能产生严重偏色和 halftone artifact。
- 在目标 GPU 上 benchmark `attention_mechanism`。
- 修改 `torch.compile` 后重新 smoke test，因为 compile cache 可能改变峰值显存。

## 运行

```bash
simpletuner train example=minimaxh3-fl2va-convrot-int8.peft-lora
```

长训练前先跑短 smoke test，确认 `h3_drift_loss`、正常 loss 和验证样本走势一致。
