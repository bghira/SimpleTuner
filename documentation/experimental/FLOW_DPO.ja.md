# Flow-DPO と Masked Flow-DPO

Flow-DPO は flow-matching モデル向けの実験的な蒸留手法で、preferred/rejected のペアから低ランク adapter を学習します。SimpleTuner では LoRA/LyCORIS のみ対応し、full-model Flow-DPO は対応しません。蒸留では text encoder 学習も無効です。

SimpleTuner は既存の reference dataset システムを使います。通常の `image` または `video` データセットが preferred サンプルを提供し、`conditioning_type=reference_strict` の `conditioning` データセットが rejected サンプルを提供します。詳しくは [`conditioning_type`](../DATALOADER.ja.md#conditioning_type) と [`conditioning_data`](../DATALOADER.ja.md#conditioning_data) を参照してください。

## 動作

各 batch で SimpleTuner は次を実行します。

1. adapter を有効にして preferred latents を予測します。
2. 同じ prompt、noise、timestep で rejected latents を予測します。
3. LoRA/LyCORIS adapter を無効にし、凍結 reference として preferred/rejected を再予測します。
4. Flow-DPO margin loss を適用します。

```text
win_adv  = L(reference_win, target_win) - L(policy_win, target_win)
lose_adv = L(policy_lose, target_lose) - L(reference_lose, target_lose)
loss     = -logsigmoid(beta / 2 * (win_adv + lose_adv))
```

flow-matching の target は `noise - latents` です。

## Masked Flow-DPO

batch に `conditioning_type=mask` または `conditioning_type=segmentation` の conditioning データセットがある場合、SimpleTuner は DPO prediction error に mask を適用してから reduction します。これにより、差分のある領域に preference signal を集中できます。

`anchor_alpha` は preferred 側のみの global MSE regularizer です。adapter 有効時と無効時の preferred prediction を比較し、mask は使いません。

## 設定

最小構成：

```bash
--model_type=lora
--distillation_method=flow_dpo
--flow_custom_timesteps=801,694,548,338
--flow_timesteps_mode=round-robin
```

よく使う `distillation_config`：

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

- `norm_type=sum` は一般的な Flow-DPO formulation に対応します。
- `auto_beta=true` は margin の running value から beta を調整します。小さなペアデータセットで特に有用です。
- `flow_timesteps_mode=fixed-list` は `flow_custom_timesteps` からランダムに選びます。
- `flow_timesteps_mode=round-robin` は `flow_custom_timesteps` を順に循環します。
- `sft_loss_weight` の既定値は `0.0` で、通常の diffusion loss は混ぜません。

## データセット形状

rejected データセットは `reference_strict` として preferred データセットにペアリングします。

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

Masked Flow-DPO を使う場合は、mask conditioning データセットも同じ `conditioning_data` に追加します。

## 制限

Flow-DPO の現在の条件：

- flow-matching モデル。
- `model_type=lora`。
- ペアリングされた `reference_strict` conditioning データセット。
- text encoder 学習なし。

2つ目の完全なモデルコピーは読み込みません。reference pass では adapter（LyCORIS multiplier を含む）を無効にし、policy path では再度有効にします。
