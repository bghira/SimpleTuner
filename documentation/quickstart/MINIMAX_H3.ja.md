# MiniMax H3 Quickstart

MiniMax H3 は 33B の flow-matching video/audio モデルです。SimpleTuner は `minimaxh3` family で adapter training をサポートし、FL2VA first/last-frame conditioning と quantized ConvRot flavours を扱えます。

## Starting Configs

次の examples から始めます。

- `simpletuner/examples/minimaxh3-fl2va-convrot-int8.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-24g.peft-lora+ramtorch`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-32g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-48g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-80g.peft-lora`

VRAM に近い preset を選び、smoke test 後に resolution、frame count、attention backend、checkpointing を調整します。

## Core Settings

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

Examples は `convrot-int8` を使います。より低い precision の checkpoint を使う場合は同じ family で `convrot-int4` を指定できます。

## Distillation を保つ

MiniMax H3 は CFG-distilled です。base checkpoint は unconditional branch なしで動く想定なので、examples は guidance `1.0` と `validation_disable_unconditional: true` を使います。

Adapter training は distilled behavior から drift することがあります。そのため examples は default で `h3_drift` を有効化します。

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

これは adapter を無効化した frozen-base reference pass を実行し、video/audio prediction drift を penalize します。通常の H3 LoRA では有効のままにしてください。concept を学びにくい場合は `loss_weight` を下げ、validation が base behavior を失う場合は上げます。詳細は [MiniMax H3 Drift Distillation](../distillation/MINIMAX_H3_DRIFT.ja.md) を参照してください。

Negative prompting は base H3 contract の一部ではありません。SimpleTuner は de-distilled checkpoint 向けに real CFG と negative prompts を残していますが、`h3_drift` は original distilled conditional behavior を保つためのものです。

## Audio Target Mode

`minimax_h3_target_mode: "auto"` は video-only になり、audio VAE work を避けます。

```json
{
  "minimax_h3_target_mode": "video"
}
```

dataset に target audio latents があり joint audio/video training したい場合だけ `"av"` を使います。data backend ごとに `h3_target_mode` または `minimax_h3_target_mode` でも設定できます。

## Memory Knobs

- VRAM が厳しい場合は 24G RamTorch example を使います。
- heavy checkpointing の前に `musubi_blocks_to_swap` を試します。
- VAE tiling、slicing、temporal roll は有効のままにします。
- 実際の GPU で `attention_mechanism` を benchmark します。
- `torch.compile` を変えたら smoke test をやり直します。compile cache が VRAM を変えるためです。

## Run

```bash
simpletuner train example=minimaxh3-fl2va-convrot-int8.peft-lora
```

long run の前に短い smoke test を行い、`h3_drift_loss`、normal loss、validation sample が一貫して動くことを確認します。
