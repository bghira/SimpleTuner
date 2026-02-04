# REPA & U-REPA（画像正則化）

表現アラインメント（REPA）は、拡散モデルの隠れ状態を凍結された視覚エンコーダの特徴（通常はDINOv2）と整列させる正則化技術です。事前学習された視覚表現を活用することで、生成品質とトレーニング効率を向上させます。

SimpleTunerは2つのバリアントをサポートしています：

- **REPA** - DiTベースの画像モデル用（Flux、SD3、Chroma、Sana、PixArtなど）- PR #2562
- **U-REPA** - UNetベースの画像モデル用（SDXL、SD1.5、Kolors）- PR #2563

> **動画モデルをお探しですか？** 時間的アラインメント付きの動画モデルCREPAサポートについては [VIDEO_CREPA.ja.md](VIDEO_CREPA.ja.md) をご覧ください。

## 使用するタイミング

### REPA（DiTモデル）
- DiTベースの画像モデルをトレーニングしていて、より速い収束を望む場合
- 品質の問題に気づいた場合や、より強いセマンティックグラウンディングが必要な場合
- サポートされているモデルファミリー：`flux`、`flux2`、`sd3`、`chroma`、`sana`、`pixart`、`hidream`、`auraflow`、`lumina2`など

### U-REPA（UNetモデル）
- UNetベースの画像モデル（SDXL、SD1.5、Kolors）をトレーニングしている場合
- UNetアーキテクチャ向けに最適化された表現アラインメントを活用したい場合
- U-REPAは**ミッドブロック**アラインメント（初期レイヤーではない）を使用し、より良い相対類似性構造のための**多様体損失**を追加します

## クイックセットアップ（WebUI）

### DiTモデル（REPA）

1. **トレーニング → 損失関数** を開きます。
2. **CREPA** を有効にします（同じオプションで画像モデルのREPAが有効になります）。
3. **CREPA Block Index** を初期エンコーダレイヤーに設定します：
   - Flux / Flux2：`8`
   - SD3：`8`
   - Chroma：`8`
   - Sana / PixArt：`10`
4. **Weight** を `0.5` に設定して開始します。
5. 視覚エンコーダのデフォルト値（`dinov2_vitg14`、解像度 `518`）を維持します。

### UNetモデル（U-REPA）

1. **トレーニング → 損失関数** を開きます。
2. **U-REPA** を有効にします。
3. **U-REPA Weight** を `0.5`（論文のデフォルト）に設定します。
4. **U-REPA Manifold Weight** を `3.0`（論文のデフォルト）に設定します。
5. 視覚エンコーダのデフォルト値を維持します。

## クイックセットアップ（設定JSON / CLI）

### DiTモデル（REPA）

```json
{
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 0.5,
  "crepa_encoder": "dinov2_vitg14",
  "crepa_encoder_image_size": 518
}
```

### UNetモデル（U-REPA）

```json
{
  "urepa_enabled": true,
  "urepa_lambda": 0.5,
  "urepa_manifold_weight": 3.0,
  "urepa_model": "dinov2_vitg14",
  "urepa_encoder_image_size": 518
}
```

## 主な違い：REPA vs U-REPA

| 側面 | REPA（DiT） | U-REPA（UNet） |
|------|-----------|---------------|
| アーキテクチャ | Transformerブロック | ミッドブロック付きUNet |
| アラインメントポイント | 初期transformerレイヤー | ミッドブロック（ボトルネック） |
| 隠れ状態の形状 | `(B, S, D)` シーケンス | `(B, C, H, W)` 畳み込み |
| 損失コンポーネント | コサインアラインメント | コサイン + 多様体損失 |
| デフォルト重み | 0.5 | 0.5 |
| 設定プレフィックス | `crepa_*` | `urepa_*` |

## U-REPAの詳細

U-REPAは2つの主要な革新でUNetアーキテクチャ向けにREPAを適応させます：

### ミッドブロックアラインメント
初期transformerレイヤーを使用するDiTベースのREPAとは異なり、U-REPAはUNetの**ミッドブロック**（ボトルネック）から特徴を抽出します。これはUNetが最も多くのセマンティック情報を圧縮している場所です。

- **SDXL/Kolors**：1024x1024画像の場合、ミッドブロックは `(B, 1280, 16, 16)` を出力
- **SD1.5**：512x512画像の場合、ミッドブロックは `(B, 1280, 8, 8)` を出力

### 多様体損失
コサインアラインメントに加えて、U-REPAは相対類似性構造を整列させる**多様体損失**を追加します：

```
L_manifold = ||sim(y[i],y[j]) - sim(h[i],h[j])||^2_F
```

これにより、2つのエンコーダパッチが類似している場合、対応する投影パッチも類似することが保証されます。`urepa_manifold_weight` パラメータ（デフォルト3.0）は、直接アラインメントと多様体アラインメントのバランスを制御します。

## チューニングパラメータ

### REPA（DiTモデル）
- `crepa_lambda`：アラインメント損失の重み（デフォルト0.5）
- `crepa_block_index`：タップするtransformerブロック（0インデックス）
- `crepa_spatial_align`：トークンを補間して一致させる（デフォルトtrue）
- `crepa_encoder`：視覚エンコーダモデル（デフォルト`dinov2_vitg14`）
- `crepa_encoder_image_size`：入力解像度（デフォルト518）

### U-REPA（UNetモデル）
- `urepa_lambda`：アラインメント損失の重み（デフォルト0.5）
- `urepa_manifold_weight`：多様体損失の重み（デフォルト3.0）
- `urepa_model`：視覚エンコーダモデル（デフォルト`dinov2_vitg14`）
- `urepa_encoder_image_size`：入力解像度（デフォルト518）
- `urepa_use_tae`：より高速なデコードのためにTiny AutoEncoderを使用

## 係数スケジューリング

REPAとU-REPAの両方が、トレーニング中に正則化を減衰させるスケジューリングをサポートしています：

```json
{
  "crepa_scheduler": "cosine",
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 5000,
  "crepa_lambda_end": 0.0
}
```

U-REPAの場合は、`urepa_` プレフィックスを使用します：

```json
{
  "urepa_scheduler": "cosine",
  "urepa_warmup_steps": 100,
  "urepa_cutoff_step": 5000
}
```

<details>
<summary>動作原理（実践者向け）</summary>

### REPA（DiT）
- 選択したtransformerブロックから隠れ状態をキャプチャ
- LayerNorm + Linearを通じてエンコーダ次元に投影
- 凍結されたDINOv2特徴とのコサイン類似度を計算
- カウントが異なる場合は空間トークンを補間して一致させる

### U-REPA（UNet）
- UNet mid_blockにフォワードフックを登録
- 畳み込み特徴 `(B, C, H, W)` をキャプチャ
- シーケンス `(B, H*W, C)` に変形してエンコーダ次元に投影
- コサインアラインメントと多様体損失の両方を計算
- 多様体損失はペアワイズ類似性構造を整列

</details>

<details>
<summary>技術詳細（SimpleTuner内部）</summary>

### REPA
- 実装：`simpletuner/helpers/training/crepa.py`（`CrepaRegularizer` クラス）
- モード検出：画像モデルには `CrepaMode.IMAGE`、`crepa_mode` プロパティで自動設定
- 隠れ状態はモデル出力の `crepa_hidden_states` キーに保存

### U-REPA
- 実装：`simpletuner/helpers/training/crepa.py`（`UrepaRegularizer` クラス）
- ミッドブロックキャプチャ：`simpletuner/helpers/utils/hidden_state_buffer.py`（`UNetMidBlockCapture`）
- 隠れサイズは `block_out_channels[-1]` から推論（SDXL/SD1.5/Kolorsは1280）
- `MODEL_TYPE == ModelTypes.UNET` の場合のみ有効
- 隠れ状態はモデル出力の `urepa_hidden_states` キーに保存

</details>

## よくある問題

- **モデルタイプの誤り**：REPA（`crepa_*`）はDiTモデル用、U-REPA（`urepa_*`）はUNetモデル用です。間違ったものを使用しても効果はありません。
- **ブロックインデックスが高すぎる**（REPA）：「hidden states not returned」エラーが出た場合はインデックスを下げてください。
- **VRAMスパイク**：より小さいエンコーダ（`dinov2_vits14` + 画像サイズ `224`）を試すか、デコード用に `use_tae` を有効にしてください。
- **多様体重みが高すぎる**（U-REPA）：トレーニングが不安定になった場合、`urepa_manifold_weight` を3.0から1.0に下げてください。

## 参考文献

- [REPA論文](https://arxiv.org/abs/2402.17750) - 生成のための表現アラインメント
- [U-REPA論文](https://arxiv.org/abs/2410.xxxxx) - UNetアーキテクチャ向けユニバーサルREPA（NeurIPS 2025）
- [DINOv2](https://github.com/facebookresearch/dinov2) - 自己教師あり視覚エンコーダ
