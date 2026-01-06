# LongCat‑Video クイックスタート

LongCat‑Video は、フローマッチング、Qwen‑2.5‑VL テキストエンコーダ、Wan VAE を使用する 13.6B のバイリンガル（zh/en）text‑to‑video / image‑to‑video モデルです。このガイドでは、SimpleTuner でのセットアップ、データ準備、初回の学習/検証の実行を説明します。

---

## 1) ハードウェア要件（目安）

- 13.6B トランスフォーマー + Wan VAE: 画像モデルより VRAM を多く使います。`train_batch_size=1`、勾配チェックポイント、低 LoRA rank から開始してください。
- システム RAM: マルチフレームのクリップでは 32GB 超が有用です。データセットは高速ストレージに置いてください。
- Apple MPS: プレビュー対応。位置エンコーディングは自動で float32 にダウンキャストされます。

---

## 2) 前提条件

1. Python 3.12 を確認（SimpleTuner はデフォルトで `.venv` を用意します）:
   ```bash
   python --version
   ```
2. ハードウェアに合った extras で SimpleTuner をインストール:
   ```bash
   pip install "simpletuner[cuda]"   # NVIDIA
   pip install "simpletuner[mps]"    # Apple Silicon
   pip install "simpletuner[cpu]"    # CPU-only
   ```
3. 量子化は組み込み（`int8-quanto`, `int4-quanto`, `fp8-torchao`）で、通常は追加インストールは不要です。

---

## 3) 環境セットアップ

### Web UI
```bash
simpletuner server
```
http://localhost:8001 を開き、モデルファミリ `longcat_video` を選択します。

### CLI 基本設定（config/config.json）

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_video",
  "model_flavour": "final",
  "pretrained_model_name_or_path": null,      // auto-selected from flavour
  "base_model_precision": "bf16",             // int8-quanto/fp8-torchao also work for LoRA
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "learning_rate": 1e-4,
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0
}
```

**保持すべき既定値**
- shift `12.0` のフローマッチングスケジューラは自動。特別なノイズフラグは不要。
- アスペクトバケットは 64px 整列のまま。`aspect_bucket_alignment` は 64 に固定。
- 最大トークン長は 512（Qwen‑2.5‑VL）。CFG 有効時にネガティブプロンプトが未指定なら、パイプラインが空のネガティブを自動追加します。
- フレーム数は `(num_frames - 1)` が VAE の時間ストライド（デフォルト 4）で割り切れる必要があります。既定の 93 フレームは条件を満たします。

VRAM 節約オプション:
- `lora_rank` を下げる（4〜8）+ `int8-quanto` のベース精度を使う
- group offload を有効化: `--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`
- プレビューが OOM する場合は、`validation_resolution` / フレーム数 / ステップ数を先に下げる
- Attention 既定: CUDA では、LongCat‑Video は利用可能な場合にバンドルされた block‑sparse Triton カーネルを自動使用し、なければ標準ディスパッチャへフォールバックします。切り替えは不要です。xFormers を使いたい場合は、設定/CLI に `attention_implementation: "xformers"` を指定してください。

### トレーニング開始（CLI）
```bash
simpletuner train --config config/config.json
```
または Web UI を起動し、同じ設定でジョブを送信します。

---

## 4) データローダーのガイダンス

- キャプション付き動画データセットを使用します。各サンプルはフレーム（または短いクリップ）とテキストキャプションを提供します。`dataset_type: video` は `VideoToTensor` で自動処理されます。
- フレームサイズは 64px グリッド（例: 480x832、720p バケット）に揃えます。高さ/幅は Wan VAE のストライド（組み込み設定では 16px）および 64 で割り切れる必要があります。
- image‑to‑video では、各サンプルに条件画像を含めます。最初の latent フレームに配置され、サンプリング中は固定されます。
- LongCat‑Video は 30 fps 前提です。既定の 93 フレームは約 3.1 秒。フレーム数を変更する場合は `(frames - 1) % 4 == 0` を維持し、fps に応じて尺が変わる点に注意してください。

### 動画バケット戦略

データセットの `video` セクションで、グルーピング方法を設定できます:
- `bucket_strategy`: `aspect_ratio`（デフォルト）は空間アスペクト比でグルーピング。`resolution_frames` は `WxH@F` 形式（例: `480x832@93`）で解像度/フレーム数をまとめ、混在解像度/尺のデータセットに便利です。
- `frame_interval`: `resolution_frames` 使用時、フレーム数をこの間隔に丸めます（例: VAE 時間ストライドに合わせて 4）。

---

## 5) 検証と推論

- ガイダンス: 3.5〜5.0 が良好。CFG が有効でネガティブ未指定なら空のネガティブが自動生成されます。
- ステップ数: 品質確認は 35〜45。クイックプレビューなら低め。
- フレーム数: 既定 93（VAE 時間ストライド 4 に整合）。
- プレビュー/学習で余裕が必要なら、`musubi_blocks_to_swap`（4〜8 から試す）と必要に応じて `musubi_block_swap_device` を設定し、最後の Transformer ブロックを CPU からストリーミングします。転送オーバーヘッドは増えますがピーク VRAM は下がります。

- 検証は config の `validation_*` フィールド、または WebUI のプレビュータブから実行できます。単独 CLI サブコマンドよりまずこちらを使うと手早いです。
- データセット駆動の検証（I2V を含む）では `validation_using_datasets: true` を設定し、`eval_dataset_id` を検証スプリットに指定します。`is_i2v` かつ条件フレームが紐づいていれば、パイプラインが最初のフレームを自動固定します。
- latent プレビューはデコード前に unpack してチャネル不一致を回避します。

---

## 6) トラブルシューティング

- **高さ/幅エラー**: 両方が 16 で割り切れ、64px グリッドに揃っていることを確認してください。
- **MPS float64 警告**: 内部処理済み。精度は bf16/float32 のままにしてください。
- **OOM**: まず検証解像度/フレーム数を下げ、LoRA rank を減らす、group offload を有効化する、`int8-quanto`/`fp8-torchao` に切り替える。
- **CFG で負のプロンプトが空**: 未指定ならパイプラインが自動で空を挿入します。

---

## 7) フレーバー

- `final`: LongCat‑Video のメインリリース（text‑to‑video + image‑to‑video を単一チェックポイントで提供）。
