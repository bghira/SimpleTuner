# MiniMax H3 Drift Distillation

MiniMax H3 は distilled flow-matching の video/audio モデルです。通常の LoRA / LyCORIS training では adapter が dataset target を学習しますが、base checkpoint の distilled behavior が drift することがあります。guidance behavior、modality balance、packed video/audio sequence layout が必要以上に変わる場合があります。

`h3_drift` は、adapter 有効時の prediction を、adapter を無効化した同じ model の frozen-base prediction と比較します。別の teacher checkpoint は読み込まず、distillation cache も使いません。各 batch で SimpleTuner は次を行います。

1. adapter 有効の通常 path で MiniMax H3 SFT loss を計算する。
2. adapter を一時的に無効化する。
3. 同じ prepared batch を `torch.no_grad()` の frozen base で実行する。
4. video/audio prediction の MSE を計算する。
5. adapter を戻し、combined loss を backpropagate する。

```text
total = sft_loss_weight * normal_h3_loss + loss_weight * frozen_base_prediction_mse
```

## 使う場面

MiniMax H3 の LoRA / LyCORIS training では、元の distillation behavior を意図的に外す場合を除き有効化してください。style/concept LoRA、FL2VA/Ref2VA、joint audio/video training、`convrot-int8` / `convrot-int4` などの quantized flavour に向いています。

Full-rank では未対応です。transformer 全体を更新する場合、比較対象となる frozen base path が成立しません。

## Quick Config

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

同梱 H3 examples はこれを default で有効化します。`loss_weight: 0.5` は dataset target を主にしつつ、base reference drift を抑える実用的な開始値です。

## Config Keys

- `loss_weight`: frozen-base prediction loss の倍率。狭い LoRA は `0.25` から `0.5`、base behavior が崩れる場合は `1.0`。
- `sft_loss_weight`: 通常の MiniMax H3 training loss の倍率。通常は `1.0`。
- `balance`: `token` は valid element 数で平均、`modality` は video/audio の modality mean を重み付きで平均。
- `video_weight`: video drift term の倍率。
- `audio_weight`: audio drift term の倍率。

## Video と Audio

`minimax_h3_target_mode: "auto"` は video-only に解決されます。`"video"` は audio target rows を使わず、`"av"` は joint audio/video rows を学習します。global config または data backend の `h3_target_mode` / `minimax_h3_target_mode` で指定できます。

Distiller は prepared batch に従います。video-only batch では `model_prediction` のみ、`av` batch では video と `audio_prediction` を比較します。`audio_latent_mask`、`sample_weight`、visual mask も反映されます。

## CFG Distillation を保つ

MiniMax H3 は CFG-distilled です。base checkpoint は通常 `validation_guidance: 1.0`、`validation_guidance_real: 1.0`、`validation_disable_unconditional: true` で検証します。Negative prompting は base contract に含まれません。

SimpleTuner は real CFG と negative prompt encoding に対応しています。community が H3 を de-distill する可能性があるためです。`h3_drift` は逆方向の制約で、adapter を base conditional prediction に近づけます。negative prompt behavior を教えたい、または de-distillation したい場合は `loss_weight` を下げるか無効化してください。

## Logs と Cost

主な logs は `h3_drift_loss`、`h3_drift_video_loss`、`h3_drift_audio_loss`、element counts、`h3_drift_weighted_loss`、`h3_drift_sft_loss`、`total` です。

各 step に extra forward pass が 1 回追加されますが、2 つ目の transformer は保持しません。ConvRot、RamTorch、musubi block swap、gradient checkpointing、attention offload と併用できます。ただし extra forward により fastest backend が変わるため、preset ごとに benchmark してください。

## Troubleshooting

- low-rank error: `model_type: "lora"` を使います。
- audio loss が zero: batch が video-only、target mode が `av` ではない、または `audio_latent_mask` が全行を除外しています。
- adapter が concept を学びにくい: `loss_weight` を下げる、rank を上げる、または longer training。
- audio が drift する: `balance: "modality"` または高い `audio_weight` を試します。
