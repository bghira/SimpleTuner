# LongCat‑Image クイックスタート

LongCat‑Image は、フローマッチングと Qwen‑2.5‑VL テキストエンコーダを使用する 6B のバイリンガル（zh/en）text‑to‑image モデルです。このガイドでは、SimpleTuner を使ったセットアップ、データ準備、初回の学習/検証の実行方法を説明します。

---

## 1) ハードウェア要件（目安）

- VRAM: 16〜24GB で `int8-quanto` または `fp8-torchao` の 1024px LoRA が可能です。bf16 フル実行では約 24GB が必要になる場合があります。
- システム RAM: 通常は約 32GB で十分です。
- Apple MPS: 推論/プレビュー対応。dtype 問題を避けるため、MPS では pos‑ids を float32 にダウンキャスト済みです。

---

## 2) 前提条件（手順）

1. Python 3.10〜3.13 を確認:
   ```bash
   python --version
   ```
2. (Linux/CUDA) 新規イメージでは一般的なビルド/ツールチェーンをインストール:
   ```bash
   apt -y update
   apt -y install build-essential nvidia-cuda-toolkit
   ```
3. ハードウェアに合った extras で SimpleTuner をインストール:
   ```bash
   pip install "simpletuner[cuda]"   # CUDA
   pip install "simpletuner[cuda13]" # CUDA 13 / Blackwell (NVIDIA B-series GPUs)
   pip install "simpletuner[mps]"    # Apple Silicon
   pip install "simpletuner[cpu]"    # CPU-only
   ```
4. 量子化は組み込み（`int8-quanto`, `int4-quanto`, `fp8-torchao`）で、通常は追加インストールは不要です。

---

## 3) 環境セットアップ

### Web UI（ガイド付き）
```bash
simpletuner server
```
http://localhost:8001 を開き、モデルファミリ `longcat_image` を選択します。

### CLI 基本設定（config/config.json）

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_image",
  "model_flavour": "final",                // options: final, dev
  "pretrained_model_name_or_path": null,   // auto-selected from flavour; override with a local path if needed
  "base_model_precision": "int8-quanto",   // good default; fp8-torchao also works
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 16,
  "learning_rate": 1e-4,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 30
}
```

**保持すべき既定値**
- Flow matching スケジューラは自動。特別なスケジュールフラグは不要。
- アスペクトバケットは 64px 整列のまま。`aspect_bucket_alignment` を下げない。
- 最大トークン長は 512（Qwen‑2.5‑VL）。

メモリ節約オプション（環境に合わせて選択）:
- `--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`
- `lora_rank` を下げる（4〜8）/ `int8-quanto` ベース精度を使う
- 検証で OOM する場合は、まず `validation_resolution` やステップ数を下げる

### 迅速な設定作成（初回のみ）
```bash
cp config/config.json.example config/config.json
```
上記の項目（model_family、flavour、precision、paths）を編集します。`output_dir` とデータセットパスはストレージ先に合わせて設定してください。

### トレーニング開始（CLI）
```bash
simpletuner train --config config/config.json
```
または WebUI を起動し、同じ設定を選んだうえで Jobs ページから実行します。

---

## 4) データローダーの要点（用意するもの）

- 標準的なキャプション付き画像フォルダ（textfile/JSON/CSV）でOK。バイリンガルの強度を維持したい場合は zh/en の両方を含めます。
- バケット境界は 64px グリッドを維持。マルチアスペクトで学習する場合は複数の解像度を列挙（例: `1024x1024,1344x768`）。
- VAE は shift+scale の KL。キャッシュは組み込みスケーリング係数を自動で使用します。

---

## 5) 検証と推論

- ガイダンス: 4〜6 が出発点。ネガティブプロンプトは空のままでOK。
- ステップ数: 速度重視なら約 30、最高品質なら 40〜50。
- 検証プレビューはそのまま動作します。チャネル不一致を避けるため、デコード前に latent を unpack します。

例（CLI 検証）:
```bash
simpletuner validate \
  --model_family longcat_image \
  --model_flavour final \
  --validation_resolution 1024x1024 \
  --validation_num_inference_steps 30 \
  --validation_guidance 4.5
```

---

## 6) トラブルシューティング

- **MPS float64 エラー**: 内部で処理済み。設定は float32/bf16 のままにしてください。
- **プレビューのチャネル不一致**: デコード前に latent を unpack することで修正済み（本ガイドのコードに含まれます）。
- **OOM**: `validation_resolution` を下げる、`lora_rank` を減らす、group offload を有効にする、`int8-quanto` / `fp8-torchao` に切り替える。
- **トークナイズが遅い**: Qwen‑2.5‑VL は 512 トークン上限。極端に長いプロンプトは避ける。

---

## 7) フレーバー選択
- `final`: 本番リリース（最良品質）。
- `dev`: 実験/微調整向けの中間チェックポイント。
