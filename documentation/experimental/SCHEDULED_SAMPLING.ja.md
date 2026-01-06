# Scheduled Sampling（Rollout）

## 背景

標準的な拡散学習は「Teacher Forcing」に依存します。クリーンな画像に正確な量のノイズを加え、モデルにそのノイズ（または速度/元画像）を予測させます。モデルへの入力は常に「完全に」ノイズ化され、理論的なノイズスケジュール上にあります。

しかし推論（生成）では、モデルは自分の出力を次の入力として使います。ステップ $t$ で小さな誤差が生じると、その誤差がステップ $t-1$ に伝播し、誤差が蓄積して有効画像の多様体から逸脱します。学習（完全入力）と推論（不完全入力）のこの不一致は **Exposure Bias** と呼ばれます。

**Scheduled Sampling**（この文脈では「Rollout」とも呼ばれます）は、モデル自身の生成出力で学習することでこれを緩和します。

## 仕組み

クリーン画像に単純にノイズを加える代わりに、学習ループがときどきミニ推論セッションを実行します:

1.  **ターゲットタイムステップ** $t$ を選ぶ（学習対象のステップ）。
2.  **ソースタイムステップ** $t+k$ を選ぶ（よりノイズの多いステップ）。
3.  モデルの *現在の* 重みを使い、$t+k$ から $t$ まで実際に生成（デノイズ）する。
4.  自己生成された、わずかに不完全なステップ $t$ の潜在表現を学習入力として使う。

これにより、モデルは自分が生み出しているアーティファクトや誤差を含む入力を見ることになります。「この誤りをしているから、こう直す」と学習し、生成を正しい経路に引き戻します。

## 設定

この機能は実験的で計算負荷が増えますが、特に小規模データセット（Dreambooth）でプロンプト追従性や構造の安定性を大きく改善できます。

有効化するには、`max_step_offset` を 0 以外に設定する必要があります。

### 基本設定

`config.json` に以下を追加します:

```json
{
  "scheduled_sampling_max_step_offset": 10,
  "scheduled_sampling_probability": 1.0,
  "scheduled_sampling_sampler": "unipc"
}
```

### オプション参照

#### `scheduled_sampling_max_step_offset`（Integer）
**既定:** `0`（無効）
ロールアウトの最大ステップ数。`10` にすると、各サンプルで 0〜10 のランダムな長さが選ばれます。
> 🟢 **推奨:** 小さく始める（例: `5`〜`10`）。短いロールアウトでも、学習の極端な遅延なしに誤差補正を学べます。

#### `scheduled_sampling_probability`（Float）
**既定:** `0.0`
Scheduled Sampling（Rollout）の対象になる確率（0.0〜1.0）。
*   `1.0`: すべてのサンプルでロールアウト（最も重い）。
*   `0.5`: 50% は通常学習、50% がロールアウト。

#### `scheduled_sampling_ramp_steps`（Integer）
**既定:** `0`
設定した場合、確率は `scheduled_sampling_prob_start`（既定 0.0）から `scheduled_sampling_prob_end`（既定 0.5）へ、指定ステップ数で線形に増加します。
> 🟢 **Tip:** これは「ウォームアップ」として機能します。まず基本的なデノイズを学ばせ、徐々に自分の誤りを修正する難しいタスクを導入できます。

#### `scheduled_sampling_sampler`（String）
**既定:** `unipc`
ロールアウト生成に使うソルバ。
*   **選択肢:** `unipc`（推奨・高速・高精度）、`euler`、`dpm`、`rk4`。
*   `unipc` は短いサンプリングにおける速度と精度のバランスが最も良い傾向です。

### Flow Matching + ReflexFlow

flow-matching モデル（`--prediction_type flow_matching`）では、Scheduled Sampling が ReflexFlow 風の Exposure Bias 緩和に対応します:

*   `scheduled_sampling_reflexflow`: rollout 中の ReflexFlow 強化を有効化（scheduled sampling が有効な flow-matching モデルでは自動的に有効。無効化するには `--scheduled_sampling_reflexflow=false`）。
*   `scheduled_sampling_reflexflow_alpha`: Exposure Bias に基づく損失重み（周波数補償）のスケール。
*   `scheduled_sampling_reflexflow_beta1`: 方向性 anti-drift 正則化のスケール（既定 10.0、論文に合わせる）。
*   `scheduled_sampling_reflexflow_beta2`: 周波数補償損失のスケール（既定 1.0）。

これらは既に計算しているロールアウト予測/潜在を再利用するため追加の勾配パスは不要で、偏ったロールアウトがクリーントラジェクトリに沿うように保ちつつ、デノイズ初期の低周波成分不足を強調します。

### 性能への影響

> ⚠️ **警告:** ロールアウトを有効化すると、学習ループ内でモデルを推論モードで実行する必要があります。
>
> `max_step_offset=10` にすると、学習ステップごとに最大 10 回の追加フォワードが走る可能性があります。`it/s`（秒間イテレーション）は低下します。`scheduled_sampling_probability` を調整して、速度と品質のトレードオフを調整してください。
