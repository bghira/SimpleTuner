# Self-Flow（内部整合）

Self-Flow は、外部ビジョンエンコーダを同一モデルのよりクリーンな EMA 教師ビューで置き換える CREPA モードです。Black Forest Labs の論文にかなり近い形で、学生には混合トークン単位ノイズ、教師にはよりクリーンなビューを与え、通常の生成 loss を保ったまま内部 hidden state を整合させます。

> **外部エンコーダ整合が欲しい場合**: REPA / U-REPA は [IMAGE_REPA.ja.md](IMAGE_REPA.ja.md)、時間整合付き CREPA は [VIDEO_CREPA.ja.md](VIDEO_CREPA.ja.md) を参照してください。

## 使う場面

- 外部エンコーダではなく BFL スタイルの自己教師あり正則化を使いたい。
- SimpleTuner で Self-Flow フックを持つ transformer 系ファミリを学習している。
- 通常生成、編集、マルチモーダル学習に同じ正則化を使いたい。
- EMA を既に使っている、または有効化できる。Self-Flow は EMA 教師が必須です。

現在の対応ファミリ:

- 画像 / 編集: `flux`, `flux2`, `sd3`, `pixart`, `sana`, `qwen_image`, `chroma`, `hidream`, `auraflow`, `lumina2`, `z_image`, `z_image_omni`, `kandinsky5_image`, `longcat_image`, `omnigen`, `ace_step`
- 動画 / マルチモーダル: `wan`, `wan_s2v`, `ltxvideo`, `ltxvideo2`, `sanavideo`, `kandinsky5_video`, `hunyuanvideo`, `longcat_video`, `cosmos`, `anima`

## クイック設定（WebUI）

1. **Training → Loss functions** を開く。
2. **CREPA** を有効化する。
3. **CREPA Feature Source** を `self_flow` にする。
4. **CREPA Block Index** は早めの学生ブロックにする。24 層 DiT なら `8`、より深いスタックなら `10` から開始。
5. **CREPA Teacher Block Index** はより深い教師ブロックにする。`16` か `20` が開始点として無難。
6. **Weight** は `0.5` から開始。
7. **Self-Flow Mask Ratio** は以下が目安:
   - 画像: `0.25`
   - 動画: `0.10`
   - `ace_step` のような音声系: `0.50`
8. **EMA** を有効にする。
9. TwinFlow とは併用しない。

## クイック設定（config JSON / CLI）

```json
{
  "use_ema": true,
  "crepa_enabled": true,
  "crepa_feature_source": "self_flow",
  "crepa_block_index": 8,
  "crepa_teacher_block_index": 16,
  "crepa_lambda": 0.5,
  "crepa_self_flow_mask_ratio": 0.25
}
```

旧エイリアスの `crepa_self_flow=true` も使えますが、新規設定では `crepa_feature_source=self_flow` を推奨します。

## 主な調整項目

- `crepa_block_index`: 学生ブロック
- `crepa_teacher_block_index`: EMA 教師ブロック。必須
- `crepa_lambda`: 整合の強さ。`0.5` から開始
- `crepa_self_flow_mask_ratio`: 別 timestep を受けるトークン比率。`[0.0, 0.5]`
- `crepa_scheduler`, `crepa_warmup_steps`, `crepa_decay_steps`, `crepa_lambda_end`, `crepa_cutoff_step`: CREPA と同じ係数スケジューリング
- `crepa_use_backbone_features`: 別モードなので Self-Flow と併用しない

## サンプリング / 検証

Self-Flow が変えるのは学習であり、基本的な推論アルゴリズムではありません。

- 学習では学生に混合トークンノイズ、教師によりクリーンな EMA ビューを使います。
- 検証 loss は要求された均一 timestep スケジュールをそのまま評価します。
- 通常のサンプリングは変わりません。推論時に dual-timestep masking は使いません。

<details>
<summary>どう動くか（実運用）</summary>

- 2 つの timestep をサンプルし、ランダムマスクでトークンごとに割り当てます。
- 学生用には混合破損ビュー、教師用にはよりクリーンな timestep のビューを作ります。
- 学生は通常どおり、EMA 教師は `no_grad` で実行します。
- 早い学生層をより深い教師層に cosine similarity で整合しつつ、通常の生成 loss も学習します。

</details>

<details>
<summary>技術メモ（SimpleTuner 内部）</summary>

- モード選択は `simpletuner/helpers/training/crepa.py` の `CrepaFeatureSource.SELF_FLOW`
- 共有バッチビルダは `_prepare_image_crepa_self_flow_batch` と `_prepare_video_crepa_self_flow_batch`
- EMA 教師 forward は `auxiliary_loss` から `_run_crepa_teacher_forward` を通して実行
- `custom_timesteps` を使う検証では均一評価バッチを再構築し、学習時の混合バッチが eval loss に混ざらないようにしています

</details>

## よくある落とし穴

- **EMA 無効**: `use_ema=true` が必要
- **教師ブロック未設定**: `crepa_teacher_block_index` を設定する
- **TwinFlow 有効**: 併用不可
- **未対応ファミリ**: `supports_crepa_self_flow()` を実装したモデルのみ
- **mask ratio が高すぎる**: `0.5` 以下にする
- **特別な sampler が必要だと思っている**: 推論は通常どおり

## 参考

- [Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](https://bfl.ai/research/self-flow)
