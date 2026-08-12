# AnyFlow

SimpleTuner は NVIDIA AnyFlow を、flow-matching モデル向けの 2 つの明示的な training stage として実装します。どちらの
stage でも、現在の flow time `t` と interval endpoint `r` を受け取るモデルを学習します。

- `stage=forward` は NVIDIA の forward MeanFlow objective を実装します。
- `stage=onpolicy` は forward objective を co-train しながら、Flow Map Backward Simulation と on-policy DMD を実装します。

削除された `online_teacher` と `linear` target mode は SimpleTuner 固有の objective で、現在は受け付けません。

NVIDIA が公開した checkpoint を使った Wan continuation の例は
[AnyFlow Continuation Quickstart](/documentation/quickstart/ANYFLOW.ja.md) を参照してください。

## Forward Stage

```json
{
  "model_type": "lora",
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "stage": "forward",
      "diffusion_ratio": 0.5,
      "consistency_ratio": 0.25,
      "central_difference_epsilon": 0.005,
      "meanflow_weight_type": "beta08",
      "meanflow_adaptive_weighting": true,
      "gate_value": 0.25,
      "deltatime_type": "r",
      "loss_weight": 1.0
    }
  }
}
```

各 global batch で、forward stage は次を行います。

1. 2 つの一様な flow time をサンプリングし、`t >= r` になるように並べます。
2. サンプルの 50% を diffusion interval (`r=t`)、25% を endpoint interval (`r=0`)、残りを arbitrary interval に割り当てます。
3. 両方の endpoint にモデル scheduler の flow shift を適用します。
4. 直線 latent flow path に沿って central difference を評価します。
5. MeanFlow tangent target を構築し、NVIDIA の正規化された `beta08` timestep weighting を適用します。
6. 非 diffusion サンプルを global diffusion-branch loss mean に対してバランスします。

## On-Policy Stage

`init_lora` を設定するか checkpoint から resume して、forward-stage AnyFlow adapter からこの stage を開始します。

```json
{
  "model_type": "lora",
  "lora_type": "standard",
  "init_lora": "path-or-repo-to-forward-anyflow-adapter",
  "learning_rate": 0.000002,
  "optimizer_beta1": 0.0,
  "optimizer_beta2": 0.999,
  "optimizer_weight_decay": 0.0,
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "stage": "onpolicy",
      "cotrain_forward": true,
      "rollout_step_counts": [2, 4, 8, 16, 50],
      "dmd_weight": 1.0,
      "dmd_batch_size": 1,
      "real_score_guidance_scale": 0.0,
      "discriminator_lr": 0.000002,
      "discriminator_betas": [0.0, 0.999],
      "discriminator_weight_decay": 0.0,
      "discriminator_grad_clip": 1.0
    }
  }
}
```

on-policy stage は 3 つの score role を使います。標準 LoRA training では、凍結された base transformer をそれらの間で共有します。

- 読み込まれた AnyFlow adapter が generator です。
- adapter を無効にした base model が凍結 real score です。
- 別途最適化される `anyflow_discriminator` adapter が fake score です。

各 generator update は `rollout_step_counts` から rollout budget を選び、微分可能な FlowMap rollout を実行し、shifted uniform
time で生成 latent に noise を加え、NVIDIA の正規化 DMD gradient を適用します。各 discriminator update は no-grad student
rollout を実行し、logit-normal shifted time をサンプリングして、通常の flow target で fake score を学習します。discriminator
adapter と optimizer は、各 SimpleTuner checkpoint の横に `anyflow_discriminator.safetensors` と
`anyflow_discriminator_optim.pt` として保存されます。

MiniMax-H3 はすでに CFG distillation を含むため、on-policy run では通常 `real_score_guidance_scale=0` のままにします。
外部 real-score CFG pass が必要なモデルでは negative text embeddings を cache し、scale を明示的に設定できます。

`--seed` が設定されている場合、AnyFlow は MeanFlow interval、rollout schedule、rollout latent、DMD noise、DMD
sigma を device ごとに隔離された Torch generator から sample します。これにより、無関係な training code が global
Torch RNG を消費しても AnyFlow sample は安定します。CUDA attention backward を bit-stable にするものではありません。

## 共通設定

- `stage`: `forward` または `onpolicy`。デフォルト: `forward`。
- `diffusion_ratio`: `r=t` を使う global batch fraction。デフォルト: `0.5`。
- `consistency_ratio`: `r=0` を使う global batch fraction。デフォルト: `0.25`。
- `central_difference_epsilon`: 正規化された shifted-time offset。デフォルト: NVIDIA の `5/1000` に対応する `0.005`。
- `meanflow_weight_type`: `beta08` または `uniform`。デフォルト: `beta08`。
- `meanflow_adaptive_weighting`: 非 diffusion サンプルを diffusion branch に対してバランスします。デフォルト: `true`。
- `gate_value`: FlowMap delta-timestep embedding の blend。デフォルト: `0.25`。
- `deltatime_type`: `r` または `t-r`。デフォルト: `r`。
- `loss_weight`: forward MeanFlow loss multiplier。デフォルト: `1.0`。

## 制限

- AnyFlow には、モデル固有の FlowMap interval conditioning を持つ flow-matching model が必要です。
- on-policy training は現在、標準 PEFT LoRA が必要です。base を共有することで、各 DDP rank に generator、real-score、discriminator の大型 transformer コピーを割り当てずに済みます。
- MiniMax-H3 の joint audio-video training は拒否されます。video は schedule shift 12、audio は shift 3 を使うため、AV training を有効にするには native dual-schedule MeanFlow target と rollout の実装が必要です。
- text encoder training は SimpleTuner のすべての distillation method で無効です。
- validation は `AnyFlowValidationScheduler` を使い、登録済み FlowMap model component に次の interval endpoint を渡します。

## Logs

Forward training は `anyflow_forward_loss`、timestep と interval の値、global branch fraction を追加します。On-policy training
ではさらに `anyflow_dmd_loss`、`anyflow_dmd_gradient_norm`、`anyflow_dmd_sigma`、`anyflow_rollout_steps` が追加されます。
