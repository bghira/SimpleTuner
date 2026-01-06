# LongCat‑Image Edit クイックスタート

これは LongCat‑Image の edit/img2img 版です。まず [LONGCAT_IMAGE.md](../quickstart/LONGCAT_IMAGE.md) を読んでください。本ファイルは edit フレーバーで変更される点のみをまとめています。

---

## 1) ベース LongCat‑Image との違い

|                               | Base (text2img) | Edit |
| ----------------------------- | --------------- | ---- |
| Flavour                       | `final` / `dev` | `edit` |
| Conditioning                  | なし            | **条件付き latent（参照画像）が必須** |
| Text encoder                  | Qwen‑2.5‑VL     | 参照画像が必要な **vision context** 付き Qwen‑2.5‑VL |
| Pipeline                      | TEXT2IMG        | IMG2IMG/EDIT |
| Validation inputs             | プロンプトのみ  | プロンプト **と** 参照画像 |

---

## 2) 設定変更（CLI/WebUI）

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_image",
  "model_flavour": "edit",
  "base_model_precision": "int8-quanto",      // fp8-torchao also fine; helps fit 16–24 GB
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "learning_rate": 5e-5,
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 40,
  "validation_resolution": "768x768"
}
```

`aspect_bucket_alignment` は 64 のままにしてください。条件付き latent を無効にしないでください。edit パイプラインはこれを前提とします。

迅速な設定作成:
```bash
cp config/config.json.example config/config.json
```
`model_family`、`model_flavour`、データセットパス、output_dir を設定します。

---

## 3) データローダー: edit + 参照のペア

2 つの整合したデータセットを使用します: **edit 画像**（キャプション = 編集指示）と **参照画像**。edit データセットの `conditioning_data` は参照データセット ID を指す必要があります。ファイル名は 1 対 1 で一致させてください。

```jsonc
[
  {
    "id": "edit-images",
    "type": "local",
    "instance_data_dir": "/data/edits",
    "caption_strategy": "textfile",
    "resolution": 768,
    "cache_dir_vae": "/cache/vae/longcat/edit",
    "conditioning_data": ["ref-images"]
  },
  {
    "id": "ref-images",
    "type": "local",
    "instance_data_dir": "/data/refs",
    "caption_strategy": null,
    "resolution": 768,
    "cache_dir_vae": "/cache/vae/longcat/ref"
  }
]
```

> caption_strategy のオプションと要件については [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

注意点:
- アスペクトバケットは 64px グリッドを維持。
- 参照キャプションは任意です。存在すると edit キャプションを置き換えるため（通常は望ましくない）、基本は空にします。
- edit と参照で VAE キャッシュのパスは分けてください。
- キャッシュミスや形状エラーが出た場合は、両方の VAE キャッシュを削除して再生成してください。

---

## 4) 検証の注意点

- 検証では参照画像が必要です。`edit-images` の検証スプリットを `conditioning_data` で `ref-images` に紐づけます。
- ガイダンス: 4〜6 が良好。ネガティブプロンプトは空のままでOK。
- プレビューコールバック対応。latent は自動で unpack されます。
- 条件付き latent が見つからず検証に失敗する場合は、検証データローダーに edit/参照の両方があり、ファイル名が一致していることを確認してください。

---

## 5) 推論 / 検証コマンド

クイック CLI 検証:
```bash
simpletuner validate \
  --model_family longcat_image \
  --model_flavour edit \
  --validation_resolution 768x768 \
  --validation_guidance 4.5 \
  --validation_num_inference_steps 40
```

WebUI: **Edit** パイプラインを選択し、元画像と編集指示の両方を入力します。

---

## 6) トレーニング開始（CLI）

設定とデータローダーの準備後:
```bash
simpletuner train --config config/config.json
```
条件付き latent を計算/キャッシュできるよう、参照データセットが学習時に存在していることを確認してください。

---

## 7) トラブルシューティング

- **条件付き latent がない**: `conditioning_data` で参照データセットを紐づけ、ファイル名が一致していることを確認してください。
- **MPS dtype エラー**: パイプラインが MPS で pos‑ids を float32 に自動ダウンキャストします。残りは float32/bf16 のままでOK。
- **プレビューのチャネル不一致**: デコード前に latent を un‑patchify（この SimpleTuner バージョンを維持）。
- **edit 時の OOM**: 検証解像度/ステップ数を下げ、`lora_rank` を減らす、group offload を有効化し、`int8-quanto`/`fp8-torchao` を優先します。
