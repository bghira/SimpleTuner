# LongCat‑Video Edit（Image‑to‑Video）クイックスタート

このガイドでは、LongCat‑Video の image‑to‑video ワークフローの学習と検証を説明します。フレーバーを切り替える必要はありません。同じ `final` チェックポイントが text‑to‑video と image‑to‑video の両方をカバーします。違いはデータセットと検証設定にあります。

---

## 1) ベース LongCat‑Video との違い

|                               | Base (text2video) | Edit / I2V |
| ----------------------------- | ----------------- | ---------- |
| Flavour                       | `final`           | `final`（同一ウェイト） |
| Conditioning                  | なし              | **条件フレームが必須**（最初の latent を固定） |
| Text encoder                  | Qwen‑2.5‑VL       | Qwen‑2.5‑VL（同じ） |
| Pipeline                      | TEXT2IMG          | IMG2VIDEO |
| Validation inputs             | プロンプトのみ    | プロンプト **と** 条件画像 |
| Buckets / stride              | 64px バケット、`(frames-1)%4==0` | 同じ |

**引き継ぐコア既定値**
- shift `12.0` のフローマッチング。
- アスペクトバケットは 64px 固定。
- Qwen‑2.5‑VL テキストエンコーダ。CFG 有効時は空のネガティブが自動追加されます。
- 既定フレーム数: 93（`(frames-1)%4==0` を満たす）。

---

## 2) 設定変更（CLI/WebUI）

```jsonc
{
  "model_family": "longcat_video",
  "model_flavour": "final",
  "model_type": "lora",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "learning_rate": 1e-4,
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0,
  "validation_using_datasets": true,
  "eval_dataset_id": "longcat-video-val"
}
```

`aspect_bucket_alignment` は 64 のままにしてください。最初の latent フレームは開始画像として固定されます。VAE ストライドルール `(frames - 1) % 4 == 0` をすでに満たす 93 フレームを基本とし、強い理由がない限り変更しないでください。

クイックセットアップ:
```bash
cp config/config.json.example config/config.json
```
`model_family`、`model_flavour`、`output_dir`、`data_backend_config`、`eval_dataset_id` を設定します。上記の既定値は基本そのままでOKです。

CUDA の attention オプション:
- CUDA では、LongCat‑Video はバンドルされた block‑sparse Triton カーネルがあれば自動優先し、なければ標準ディスパッチャへフォールバックします。手動トグルは不要です。
- xFormers を強制する場合は、設定/CLI に `attention_implementation: "xformers"` を指定します。

---

## 3) データローダー: クリップと開始フレームのペア

- 2 つのデータセットを用意します:
  - **クリップ**: 目標動画 + キャプション（編集指示）。`is_i2v: true` を付け、`conditioning_data` に開始フレームのデータセット ID を指定します。
  - **開始フレーム**: 1 クリップにつき 1 枚の画像。ファイル名は一致させ、キャプションは不要です。
- どちらも 64px グリッド（例: 480x832）に揃えます。高さ/幅は 16 で割り切れる必要があります。フレーム数は `(frames - 1) % 4 == 0` を満たしてください。93 は有効値です。
- クリップと開始フレームで VAE キャッシュは分けてください。

例 `multidatabackend.json`:
```jsonc
[
  {
    "id": "longcat-video-train",
    "type": "local",
    "dataset_type": "video",
    "is_i2v": true,
    "instance_data_dir": "/data/video-clips",
    "caption_strategy": "textfile",
    "resolution": 480,
    "cache_dir_vae": "/cache/vae/longcat/video",
    "conditioning_data": ["longcat-video-cond"]
  },
  {
    "id": "longcat-video-cond",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/data/video-start-frames",
    "caption_strategy": null,
    "resolution": 480,
    "cache_dir_vae": "/cache/vae/longcat/video-cond"
  }
]
```

> caption_strategy のオプションと要件については [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

---

## 4) 検証の注意点

- 学習と同じペア構造の小さな検証スプリットを用意します。`validation_using_datasets: true` を設定し、`eval_dataset_id` をそのスプリット（例: `longcat-video-val`）に向けて、検証時に開始フレームを自動取得します。
- WebUI プレビュー: `simpletuner server` を起動し、LongCat‑Video edit を選んで開始フレーム + プロンプトを入力します。
- ガイダンス: 3.5〜5.0 が良好。CFG 有効時は空のネガティブが自動補完されます。
- 低 VRAM のプレビュー/学習では、`musubi_blocks_to_swap`（4〜8 から）と必要に応じて `musubi_block_swap_device` を設定し、最後の Transformer ブロックを CPU からストリーミングします。スループットは落ちますがピーク VRAM は下がります。
- 条件フレームはサンプリング中に固定され、後続フレームのみがデノイズされます。

---

## 5) トレーニング開始（CLI）

設定とデータローダーの準備後:
```bash
simpletuner train --config config/config.json
```
条件付き latent を生成できるよう、学習データに開始フレームが含まれていることを確認してください。

---

## 6) トラブルシューティング

- **条件画像がない**: `conditioning_data` で条件データセットを指定し、ファイル名が一致していることを確認してください。検証では `eval_dataset_id` を検証スプリット ID に設定します。
- **高さ/幅エラー**: 16 で割り切れ、64px グリッドに揃っていることを確認してください。
- **最初のフレームがドリフト**: ガイダンスを下げる（3.5〜4.0）か、ステップ数を減らしてください。
- **OOM**: 検証解像度/フレーム数を下げ、`lora_rank` を減らす、group offload を有効化する、`int8-quanto`/`fp8-torchao` を使う。
