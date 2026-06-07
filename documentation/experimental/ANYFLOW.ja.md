# AnyFlow

AnyFlow は flow-matching モデル向けの実験的な distillation モードです。通常の training timestep `t` と、それより小さい reference timestep `r` の 2 つで条件付けし、単一の rectified-flow velocity ではなく interval 上の flow map を学習します。

SimpleTuner では次のように動作します。

- `--distillation_method=anyflow` で `AnyFlowDistiller` を有効化します。
- distiller は起動時に trained component の `enable_flowmap_time_conditioning()` を呼びます。
- 各 prepared batch に `flowmap_r_timesteps` を追加します。
- 通常の target を AnyFlow target に置き換えてから model loss を計算します。

SimpleTuner の AnyFlow は online です。事前計算された ODE cache は不要です。

NVIDIA が公開した AnyFlow checkpoints を使う Wan continuation example は [AnyFlow 継続学習クイックスタート](/documentation/quickstart/ANYFLOW.ja.md) を参照してください。

## Quick Setup

```json
{
  "model_type": "lora",
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "target_mode": "online_teacher",
      "teacher_rollout_steps": 1,
      "r_timestep_sampler": "uniform",
      "min_interval_ratio": 0.02,
      "gate_value": 0.25,
      "deltatime_type": "r",
      "loss_weight": 1.0
    }
  }
}
```

AnyFlow を含む SimpleTuner の distillation methods では text encoder training は使えません。

## 仕組み

flow-matching batch ごとに SimpleTuner は次を行います。

1. 通常の `prepare_batch()` で `sigmas`, `timesteps`, `noisy_latents`, base flow target を作ります。
2. 現在の interval から `r < t` をサンプルします。
3. model wrapper が `r_timestep` として渡せるように `flowmap_r_timesteps` を batch に書き込みます。
4. training target を構築します。
5. 通常の model loss で prediction と target を比較します。

`target_mode=online_teacher` では、target は `t` の noisy latent から `r` へ向かう平均 velocity です。LoRA / LyCORIS では teacher rollout 中だけ adapter を無効化し、その後で再度有効化します。

`target_mode=linear` では teacher rollout を使いません。target は straight flow target の `noise - latents` です。smoke test や ablation には便利ですが、完全な AnyFlow teacher-map objective ではありません。

## Options

- `target_mode`: `online_teacher` または `linear`。既定値: `online_teacher`。
- `teacher_rollout_steps`: `t` から `r` までの online teacher Euler step 数。既定値: `1`。
- `r_timestep_sampler`: `uniform` または `zero`。既定値: `uniform`。
- `min_interval_ratio`: `t` と `r` の最小 normalized interval。既定値: `0.02`。
- `gate_value`: FlowMap delta timestep embedding の blend weight。既定値: `0.25`。
- `deltatime_type`: `r` または `t-r`。既定値: `r`。
- `loss_weight`: 計算済み training loss への倍率。既定値: `1.0`。
- `timestep_scale`: custom timestep scale 用の override。通常は未設定にします。

## 制限

- flow-matching model が必要です。
- sample ごとの scalar timestep が必要です。Tokenwise AnyFlow interval はまだ未対応です。
- `r_timestep < timestep` が必要です。timestep zero は拒否されます。
- 現在の online teacher mode は LoRA / LyCORIS 向けです。Full-rank online teacher には別の student/teacher wiring が必要です。
- validation は AnyFlow distiller の scheduler hook で接続されています。active pipeline scheduler は proxy され、validation transformer/UNet には次の interval endpoint が `r_timestep` または `timestep_r` として渡されます。registered FlowMap-capable validation pipelines はこの経路で動作します。custom/external validation path では FlowMap timestep kwarg を自前で渡す必要があります。
