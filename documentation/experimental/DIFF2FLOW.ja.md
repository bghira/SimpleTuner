# Diff2Flow（Diffusion-to-Flow ブリッジ）

## 背景

これまで拡散モデルは、予測対象で分類されてきました:
*   **Epsilon（$\epsilon$）:** 画像に加えたノイズを予測（SD 1.5、SDXL）
*   **V-Prediction（$v$）:** ノイズとデータを組み合わせた速度を予測（SD 2.0、SDXL Refiner）

**Flux**、**Stable Diffusion 3**、**AuraFlow** のような最新モデルは **Flow Matching**（特に Rectified Flow）を使用します。Flow Matching は生成過程を、ノイズ分布からデータ分布へ粒子を直線で移動させる常微分方程式（ODE）として扱います。

この直線的な軌道は一般にソルバが追いやすく、より少ないステップと安定した生成を可能にします。

## ブリッジ

**Diff2Flow** は、「レガシー」モデル（Epsilon または V-pred）が、基盤アーキテクチャを変えずに Flow Matching 目的で学習できるようにする軽量アダプタです。

モデルの出力（例: epsilon 予測）を数学的に flow ベクトル場 $u_t(x|1)$ に変換し、flow 目標（$x_1 - x_0$、または `noise - latents`）に対して損失を計算します。

> 🟡 **実験的ステータス:** この機能はモデルが見る損失地形を実質的に変更します。理論的には正しいものの、学習ダイナミクスを大きく変えます。主に研究・実験用途を想定しています。

## 設定

Diff2Flow を使うには、ブリッジを有効にし、必要に応じて損失関数を切り替えます。

### 基本設定

`config.json` に以下を追加します:

```json
{
  "diff2flow_enabled": true,
  "diff2flow_loss": true
}
```

### オプション参照

#### `--diff2flow_enabled`（Boolean）
**既定:** `false`
数学的ブリッジを初期化します。タイムステップ計算用の小さなバッファを確保しますが、`diff2flow_loss` も設定されない限り学習挙動は変わりません。
*   **必須:** `diff2flow_loss` を使う場合
*   **対応モデル:** `epsilon` または `v_prediction` を使うモデル（SD1.5、SD2.x、SDXL、DeepFloyd IF、PixArt Alpha）

#### `--diff2flow_loss`（Boolean）
**既定:** `false`
学習目的を切り替えます。
*   **False:** 予測と標準ターゲットの誤差を最小化（例: `MSE(pred_noise, real_noise)`）。
*   **True:** *flow 変換後* の予測と flow ターゲット（`noise - latents`）の誤差を最小化。

### シナジー

Diff2Flow は **Scheduled Sampling** と非常に相性が良いです。

次を組み合わせると:
1.  **Diff2Flow**（軌道の直線化）
2.  **Scheduled Sampling**（自己生成ロールアウトで学習）

**Reflow** や **Rectified Flow** モデルで使われる学習レシピに近づき、SDXL のような旧アーキテクチャにも最新の安定性と品質特性をもたらす可能性があります。
