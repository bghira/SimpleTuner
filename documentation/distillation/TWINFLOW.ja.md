# TwinFlow（RCGM）少ステップ学習

TwinFlow は **recursive consistency gradient matching（RCGM）** を中心にした軽量な少ステップレシピです。**`distillation_method` のメイン選択肢ではありません**。`twinflow_*` フラグで明示的に有効化します。Hub から取得した設定では `twinflow_enabled` は既定で `false` にされ、通常の Transformer 設定を変更しません。

SimpleTuner における TwinFlow:
* `diff2flow_enabled` + `twinflow_allow_diff2flow` を明示しない限り、flow-matching 専用です。
* 既定では EMA 教師。教師/CFG パスの前後で RNG のキャプチャ/復元が **常に有効** で、参照 TwinFlow 実行に合わせます。
* 負時間の意味を扱う符号埋め込みは Transformer に配線済みですが、`twinflow_enabled` が true のときのみ使用されます。フラグなしの HF 設定は挙動が変わりません。
* 既定の損失は RCGM + real-velocity。`twinflow_adversarial_enabled: true` で完全な自己敵対訓練（L_adv と L_rectify 損失）を有効化できます。ガイダンス `0.0` で 1–4 ステップ生成を想定しています。
* W&B ログでは TwinFlow の軌跡散布図（理論は未検証）をデバッグ目的で出力できます。

---

## クイック設定（flow-matching モデル）

通常の設定に TwinFlow の項目を追加します（`distillation_method` は未設定/null）:

```json
{
  "model_family": "sd3",
  "model_type": "lora",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
  "output_dir": "output/sd3-twinflow",

  "distillation_method": null,
  "use_ema": true,

  "twinflow_enabled": true,
  "twinflow_target_step_count": 2,
  "twinflow_estimate_order": 2,
  "twinflow_enhanced_ratio": 0.5,
  "twinflow_delta_t": 0.01,
  "twinflow_target_clamp": 1.0,

  "learning_rate": 1e-4,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "mixed_precision": "bf16",
  "validation_guidance": 0.0,
  "validation_num_inference_steps": 2
}
```

拡散モデル（epsilon/v prediction）の場合は明示的に有効化します:

```json
{
  "prediction_type": "epsilon",
  "diff2flow_enabled": true,
  "twinflow_allow_diff2flow": true
}
```

> 既定では TwinFlow は RCGM + real-velocity 損失を使用します。`twinflow_adversarial_enabled: true` で完全な自己敵対訓練（L_adv と L_rectify 損失）を有効化でき、外部 discriminator は不要です。

---

## 想定される結果（論文データ）

arXiv:2512.05150（PDF テキスト）より:
* 推論ベンチマークは **単一の A100（BF16）** で、スループット（batch=10）とレイテンシ（batch=1）を 1024×1024 で測定。具体的な数値は記載されておらず、ハードウェア条件のみ。
* **GPU メモリ比較**（1024×1024）では、Qwen-Image-20B（LoRA）と SANA-1.6B で TwinFlow が DMD2 / SANA-Sprint より OOM しにくいことが示されています。
* 学習設定（表 6）には **バッチサイズ 128/64/32/24** と **学習ステップ 30k〜60k（または 7k〜10k の短期）** が記載。学習率は一定、EMA 減衰は 0.99 が多い。
* PDF には総 GPU 数、ノード構成、実行時間は **記載されていません**。

これらは方向性の参考であり、保証ではありません。正確なハードウェア/ランタイムは著者確認が必要です。

---

## 主なオプション

* `twinflow_enabled`: RCGM 補助損失を有効化。`distillation_method` は空にし、scheduled sampling は無効にします。設定にない場合は既定で `false`。
* `twinflow_target_step_count`（推奨 1〜4）: 学習の指標で、検証/推論にも再利用。CFG は組み込み済みのためガイダンスは `0.0` に固定されます。
* `twinflow_estimate_order`: RCGM ロールアウトの積分次数（既定 2）。値を上げると教師パスが増えます。
* `twinflow_enhanced_ratio`: 教師の cond/uncond 予測からの CFG 風ターゲット補正（既定 0.5、0.0 で無効）。RNG をキャプチャして cond/uncond を揃えます。
* `twinflow_delta_t` / `twinflow_target_clamp`: 再帰ターゲットの形状。既定は論文の安定設定に合わせています。
* `use_ema` + `twinflow_require_ema`（既定 true）: EMA 重みを教師として使用。学生を教師にする品質低下を許容する場合のみ `twinflow_allow_no_ema_teacher: true` を設定します。
* `twinflow_allow_diff2flow`: `diff2flow_enabled` と併用時に epsilon/v-prediction をブリッジします。
* RNG キャプチャ/復元: 参照 TwinFlow 実装に合わせるため常時有効。無効化スイッチはありません。
* 符号埋め込み: `twinflow_enabled` が true のとき、`timestep_sign` をサポートする Transformer に `twinflow_time_sign` を渡します。false の場合は追加埋め込みを使いません。

### 敵対ブランチ（完全 TwinFlow）

オリジナル論文の自己敵対訓練を有効にして品質を向上させます：

* `twinflow_adversarial_enabled`（既定 false）: L_adv と L_rectify 損失を有効化。負時間を使用して「偽」軌道を訓練し、外部 discriminator なしで分布マッチングを実現します。
* `twinflow_adversarial_weight`（既定 1.0）: 敵対損失（L_adv）の重み係数。
* `twinflow_rectify_weight`（既定 1.0）: 修正損失（L_rectify）の重み係数。

有効化すると、訓練は 1 ステップ生成で偽サンプルを生成し、以下の損失を訓練します:
- **L_adv**: 負時間での偽速度損失 — モデルに偽サンプルからノイズへのマッピングを学習させます。
- **L_rectify**: 分布マッチング損失 — 真偽の軌道予測を揃えてより直線的なパスを得ます。

---

## 学習 & 検証フロー

1. 通常の flow-matching 学習として実行（distiller 不要）。EMA は必須（明示的にオプトアウトしない限り）。RNG 揃えは自動。
2. 検証では **TwinFlow/UCGM スケジューラ** に自動で切り替わり、`twinflow_target_step_count` ステップ、`guidance_scale=0.0` を使用します。
3. エクスポートしたパイプラインでは、スケジューラを手動で付与します:

```python
from simpletuner.helpers.training.custom_schedule import TwinFlowScheduler

pipe = ...  # your loaded diffusers pipeline
pipe.scheduler = TwinFlowScheduler(num_train_timesteps=1000, prediction_type="flow_matching", shift=1.0)
pipe.scheduler.set_timesteps(num_inference_steps=2, device=pipe.device)
result = pipe(prompt="A cinematic portrait, 35mm", guidance_scale=0.0, num_inference_steps=2).images
```

---

## ログ

* `report_to=wandb` かつ `twinflow_enabled=true` の場合、TwinFlow の軌跡散布図（σ vs tt vs sign）を実験的にログ出力できます。デバッグ用途のみで、UI 上は “experimental/theory unverified” と表示されます。

---

## トラブルシューティング

* **flow-matching に関するエラー**: `prediction_type=flow_matching` が必要です。`diff2flow_enabled` + `twinflow_allow_diff2flow` を有効にしない限り必須です。
* **EMA 必須**: `use_ema` を有効化するか、学生を教師にすることを許容する場合のみ `twinflow_allow_no_ema_teacher: true` / `twinflow_require_ema: false` を設定します。
* **1 ステップで品質が頭打ち**: `twinflow_target_step_count: 2`〜`4` を試し、ガイダンスは `0.0` を維持し、過学習なら `twinflow_enhanced_ratio` を下げます。
* **Teacher/Student の乖離**: RNG は常に揃えているため、乖離は確率差ではなくモデル不一致によるものです。Transformer が `timestep_sign` を持たない場合は `twinflow_enabled` をオフにするか、モデル側の対応を追加してから有効化してください。
