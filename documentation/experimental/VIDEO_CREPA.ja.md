# CREPA（動画正則化）

Cross-frame Representation Alignment（CREPA）は動画モデル向けの軽量な正則化です。各フレームの隠れ状態を、凍結済みビジョンエンコーダの特徴（当該フレーム**および近傍フレーム**）へ寄せることで、主損失を変えずに時間的一貫性を改善します。

> **画像モデルをお探しですか？** 画像DiTモデル（Flux、SD3など）のREPAサポートとUNetモデル（SDXL、SD1.5、Kolors）のU-REPAサポートについては [IMAGE_REPA.ja.md](IMAGE_REPA.ja.md) をご覧ください。

## 使いどき

- 複雑な動き、シーン変化、オクルージョンを含む動画を学習している。
- 動画 DiT（LoRA またはフル）を微調整していてフリッカー/アイデンティティドリフトが見える。
- 対応モデルファミリー: `kandinsky5_video`, `ltxvideo`, `sanavideo`, `wan`（他は CREPA フックがありません）。
- 追加の VRAM（設定により約 1〜2GB）があり、DINO エンコーダと VAE を訓練中にメモリ保持できる。

## クイック設定（WebUI）

1. **Training → Loss functions** を開く。
2. **CREPA** を有効化。
3. **CREPA Block Index** をエンコーダ側のレイヤーに設定。開始値:
   - Kandinsky5 Video: `8`
   - LTXVideo / Wan: `8`
   - SanaVideo: `10`
4. **Weight** は `0.5` のまま開始。
5. **Adjacent Distance** は `1`、**Temporal Decay** は `1.0` にして、CREPA 論文に近い設定にする。
6. ビジョンエンコーダは既定（`dinov2_vitg14`、解像度 `518`）を使用。VRAM を節約したい場合のみ小さいエンコーダ（例: `dinov2_vits14` + 画像サイズ `224`）に変更。
7. 通常通り学習。CREPA は補助損失を追加し、`crepa_loss` / `crepa_similarity` をログします。

## クイック設定（config JSON / CLI）

`config.json` もしくは CLI に以下を追加します:

```json
{
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 0.5,
  "crepa_adjacent_distance": 1,
  "crepa_adjacent_tau": 1.0,
  "crepa_encoder": "dinov2_vitg14",
  "crepa_encoder_image_size": 518
}
```

## 調整項目

- `crepa_spatial_align`: パッチレベルの構造を維持（既定）。メモリが厳しい場合は `false` でプーリング。
- `crepa_normalize_by_frames`: クリップ長に対して損失スケールを安定化（既定）。長いクリップにより大きく寄与させたい場合は無効化。
- `crepa_drop_vae_encoder`: 潜在の**デコードのみ**を行う場合のメモリ削減（ピクセルをエンコードする場合は危険）。
- `crepa_adjacent_distance=0`: フレームごとの REPA* のように動作（隣接補助なし）。距離減衰には `crepa_adjacent_tau` を併用。
- `crepa_cumulative_neighbors=true`（設定のみ）: 近傍だけでなく `1..d` の全オフセットを使用。
- `crepa_use_backbone_features=true`: 外部エンコーダを使わず、より深い Transformer ブロックへの整合に切り替え。教師は `crepa_teacher_block_index` で指定。
- エンコーダサイズ: VRAM が厳しければ `dinov2_vits14` + `224` にダウン。品質重視なら `dinov2_vitg14` + `518`。

## 係数スケジューリング

CREPA は訓練中に係数（`crepa_lambda`）をスケジュールする機能をサポートしています。ウォームアップ、減衰、類似度閾値に基づく自動カットオフが利用可能です。これは特に text2video 訓練で有用で、CREPA を強く長く適用しすぎると水平/垂直の縞模様や色褪せた感じが出る場合があります。

### 基本スケジューリング

```json
{
  "crepa_enabled": true,
  "crepa_lambda": 0.5,
  "crepa_scheduler": "cosine",
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 5000,
  "crepa_lambda_end": 0.0
}
```

この設定では:
1. 最初の 100 ステップで CREPA 重みを 0 から 0.5 にウォームアップ
2. 5000 ステップかけてコサインスケジュールで 0.5 から 0.0 に減衰
3. ステップ 5100 以降、CREPA は実質的に無効

### スケジューラタイプ

- `constant`: 減衰なし、重みは `crepa_lambda` のまま（既定）
- `linear`: `crepa_lambda` から `crepa_lambda_end` への線形補間
- `cosine`: 滑らかなコサインアニーリング（ほとんどのケースで推奨）
- `polynomial`: `crepa_power` で設定可能な累乗による多項式減衰

### ステップベースのカットオフ

特定のステップ以降でハードカットオフする場合:

```json
{
  "crepa_cutoff_step": 3000
}
```

ステップ 3000 以降、CREPA は完全に無効化されます。

### 類似度ベースのカットオフ

最も柔軟なアプローチです。類似度メトリクスがプラトーに達したときに自動的に CREPA を無効化し、モデルが十分な時間的整合を学習したことを示します:

```json
{
  "crepa_similarity_threshold": 0.9,
  "crepa_similarity_ema_decay": 0.99,
  "crepa_threshold_mode": "permanent"
}
```

- `crepa_similarity_threshold`: 類似度の指数移動平均がこの値に達するとカットオフ
- `crepa_similarity_ema_decay`: 平滑化係数（0.99 ≈ 100 ステップのウィンドウ）
- `crepa_threshold_mode`: `permanent`（オフのまま）または `recoverable`（類似度が下がると再有効化）

### 推奨設定

**image2video (i2v) 向け**:
```json
{
  "crepa_scheduler": "constant",
  "crepa_lambda": 0.5
}
```
参照フレームが一貫性のアンカーになるため、i2v では標準の CREPA でうまく機能します。

**text2video (t2v) 向け**:
```json
{
  "crepa_scheduler": "cosine",
  "crepa_lambda": 0.5,
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 0,
  "crepa_lambda_end": 0.1,
  "crepa_similarity_threshold": 0.85,
  "crepa_threshold_mode": "permanent"
}
```
訓練中に CREPA を減衰させ、類似度が飽和するとカットオフしてアーティファクトを防ぎます。

**単色背景向け (t2v)**:
```json
{
  "crepa_cutoff_step": 2000
}
```
早期カットオフにより均一な背景での縞模様アーティファクトを防ぎます。

<details>
<summary>仕組み（実務者向け）</summary>

- 指定した DiT ブロックの隠れ状態を取り出し、LayerNorm+Linear ヘッドで投影して凍結済みのビジョン特徴へ整合します。
- 既定では DINOv2 でピクセルフレームをエンコード。バックボーンモードでは深い Transformer ブロックを再利用します。
- 近傍フレームへ指数減衰（`crepa_adjacent_tau`）で整合。累積モードでは `d` までの全オフセットを合算可能。
- 空間/時間整合でトークンを再サンプリングし、DiT パッチとエンコーダパッチを揃えてからコサイン類似度を計算。損失はパッチとフレームで平均化されます。

</details>

<details>
<summary>技術詳細（SimpleTuner 内部）</summary>

- 実装: `simpletuner/helpers/training/crepa.py`。`ModelFoundation._init_crepa_regularizer` から登録され、学習対象モデルに付与（プロジェクタは最適化対象になるようモデル側に保持）。
- 隠れ状態キャプチャ: `crepa_enabled` が true のとき、動画 Transformer が `crepa_hidden_states`（必要なら `crepa_frame_features`）を保持。バックボーンモードでは共有バッファの `layer_{idx}` も使用。
- 損失経路: `crepa_use_backbone_features` が有効でない限り、VAE で潜在をピクセルにデコード。投影された隠れ状態とエンコーダ特徴を正規化し、距離重み付きコサイン類似度を適用、`crepa_loss` / `crepa_similarity` をログしてスケール損失を加算。
- 相互作用: LayerSync より前に実行し、同じ隠れ状態バッファを再利用。終了後にバッファをクリア。有効なブロックインデックスと Transformer 設定からの隠れサイズが必要。

</details>

## よくある落とし穴

- 未対応ファミリーで CREPA を有効化すると隠れ状態が取れません。`kandinsky5_video`, `ltxvideo`, `sanavideo`, `wan` に限定してください。
- **ブロックインデックスが高すぎる** → “hidden states not returned”。インデックスを下げてください。Transformer ブロックは 0 始まりです。
- **VRAM スパイク** → `crepa_spatial_align=false`、小さいエンコーダ（`dinov2_vits14` + `224`）、またはブロックインデックスを下げる。
- **バックボーンモードのエラー** → `crepa_block_index`（学生）と `crepa_teacher_block_index`（教師）を実在するレイヤーに設定。
- **メモリ不足** → RamTorch が効かない場合、大きな GPU が唯一の解決策かもしれません。H200 や B200 でも不足する場合は issue を報告してください。
