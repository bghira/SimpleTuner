# Qwen Image Edit クイックスタート

このガイドでは、SimpleTuner がサポートする Qwen Image の **edit** フレーバーを扱います:

- `edit-v1` – 1 サンプルあたり参照画像 1 枚。参照画像は Qwen2.5‑VL テキストエンコーダでエンコードされ、**conditioning image embeds** としてキャッシュされます。
- `edit-v2`（“edit plus”）– 1 サンプルあたり最大 3 枚の参照画像を使用し、VAE latent をオンザフライで生成します。

どちらも [Qwen Image quickstart](./QWEN_IMAGE.md) の大部分を継承します。以下は edit チェックポイントを微調整する際に *異なる点* に焦点を当てています。

---

## 1. ハードウェアチェックリスト

ベースモデルは **20B パラメータ** のままです:

| 要件 | 推奨 |
|-------------|----------------|
| GPU VRAM    | 最低 24 G（int8/nf4 量子化あり）• 40 G+ を強く推奨 |
| 精度        | `mixed_precision=bf16`, `base_model_precision=int8-quanto`（または `nf4-bnb`） |
| バッチサイズ | `train_batch_size=1` 固定。実効バッチは gradient accumulation で稼ぐ |

その他の前提条件は [Qwen Image ガイド](./QWEN_IMAGE.md) に従ってください（Python ≥ 3.10、CUDA 12.x イメージなど）。

---

## 2. 設定の要点

`config/config.json` 内:

<details>
<summary>設定例を表示</summary>

```jsonc
{
  "model_type": "lora",
  "model_family": "qwen_image",
  "model_flavour": "edit-v1",      // or "edit-v2"
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.0,
  "validation_num_inference_steps": 30,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "base_model_precision": "int8-quanto",
  "quantize_via": "cpu",
  "quantize_activations": false,
  "flow_schedule_shift": 1.73,
  "data_backend_config": "config/qwen_edit/multidatabackend.json"
}
```
</details>

- EMA は既定で CPU 上で動作し、チェックポイント速度が不要なら有効のままで安全です。
- 24 G カードでは `validation_resolution` を下げる必要があります（例: `768x768`）。
- `edit-v2` で制御画像にターゲット解像度を継承させたい場合は、`model_kwargs` に `match_target_res` を追加します（既定は 1 MP パッキング）:

<details>
<summary>設定例を表示</summary>

```jsonc
"model_kwargs": {
  "match_target_res": true
}
```
</details>

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTuner には、トレーニングの安定性とパフォーマンスを大幅に向上させる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが自身の入力を生成することで露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

---

</details>

## 3. データローダー構成

両フレーバーとも **ペアになったデータセット** が必要です: edit 画像、任意の edit キャプション、同一ファイル名を持つ 1 枚以上の参照画像。

フィールドの詳細は [`conditioning_type`](../DATALOADER.md#conditioning_type) と [`conditioning_data`](../DATALOADER.md#conditioning_data) を参照してください。複数の conditioning データセットを使う場合は、[OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling) の `conditioning_multidataset_sampling` でサンプリング方法を指定します。

### 3.1 edit‑v1（単一の参照画像）

メインデータセットは **1 つの conditioning データセット** と conditioning-image-embed キャッシュを参照します:

<details>
<summary>設定例を表示</summary>

```jsonc
[
  {
    "id": "my-edited-images",
    "type": "local",
    "instance_data_dir": "/datasets/edited-images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": ["my-reference-images"],
    "conditioning_image_embeds": "my-reference-embeds",
    "cache_dir_vae": "cache/vae/edited-images"
  },
  {
    "id": "my-reference-images",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/reference-images",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/reference-images"
  },
  {
    "id": "my-reference-embeds",
    "type": "local",
    "dataset_type": "conditioning_image_embeds",
    "cache_dir": "cache/conditioning_image_embeds/reference"
  }
]
```
</details>

> caption_strategy のオプションと要件については [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

- `conditioning_type=reference_strict` は edit 画像と同じクロップになることを保証します。参照の縦横比が異なる場合のみ `reference_loose` を使用してください。
- `conditioning_image_embeds` は参照ごとに生成される Qwen2.5‑VL の視覚トークンを保存します。省略した場合、SimpleTuner は `cache/conditioning_image_embeds/<dataset_id>` に既定キャッシュを作成します。

### 3.2 edit‑v2（マルチ参照）

`edit-v2` では、`conditioning_data` にすべての参照データセットを列挙します。各エントリが追加の参照フレームになります。conditioning-image-embed キャッシュは不要です（latents はオンザフライで計算されます）。

<details>
<summary>設定例を表示</summary>

```jsonc
[
  {
    "id": "my-edited-images",
    "type": "local",
    "instance_data_dir": "/datasets/edited-images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": [
      "my-reference-images-a",
      "my-reference-images-b",
      "my-reference-images-c"
    ],
    "cache_dir_vae": "cache/vae/edited-images"
  },
  {
    "id": "my-reference-images-a",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/reference-images-a",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/reference-images-a"
  },
  {
    "id": "my-reference-images-b",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/reference-images-b",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/reference-images-b"
  },
  {
    "id": "my-reference-images-c",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/reference-images-c",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/reference-images-c"
  }
]
```
</details>

参照画像の数（1〜3）に応じて conditioning データセットを用意してください。SimpleTuner はファイル名一致でサンプルを揃えます。

---

## 4. トレーナーの実行

最短のスモークテストは、サンプル preset を使うことです:

```bash
simpletuner train example=qwen_image.edit-v1-lora
# or
simpletuner train example=qwen_image.edit-v2-lora
```

手動起動の場合:

```bash
simpletuner train \
  --config config/config.json \
  --data config/qwen_edit/multidatabackend.json
```

### ヒント

- `caption_dropout_probability` は、edit 指示なしで学習させる理由がない限り `0.0` を維持してください。
- 長時間の学習では `validation_step_interval` を下げ、重い edit 検証が実行時間を支配しないようにします。
- Qwen edit チェックポイントには guidance head がないため、`validation_guidance` は通常 **3.5〜4.5** が目安です。

---

## 5. 検証プレビュー

参照画像と検証出力を一緒に見たい場合は、検証用の edit/参照ペアを専用データセットに保存し（学習分割と同じ構成）、以下を設定します:

<details>
<summary>設定例を表示</summary>

```jsonc
{
  "eval_dataset_id": "qwen-edit-val"
}
```
</details>

SimpleTuner は検証中にそのデータセットの conditioning 画像を再利用します。

---

### トラブルシューティング

- **`ValueError: Control tensor list length does not match batch size`** – すべての edit 画像に対して conditioning データセットに対応ファイルがあることを確認してください。空フォルダやファイル名不一致で発生します。
- **検証時の OOM** – `validation_resolution`、`validation_num_inference_steps` を下げるか、さらに量子化（`base_model_precision=int2-quanto`）してから再試行してください。
- **`edit-v1` でキャッシュ未検出** – メインデータセットの `conditioning_image_embeds` が既存のキャッシュデータセットと一致しているか確認してください。

---

これで base Qwen Image quickstart を edit 学習に適用できます。全設定（テキストエンコーダのキャッシュ、マルチバックエンドサンプリング等）については [FLUX_KONTEXT.md](./FLUX_KONTEXT.md) のガイダンスを再利用してください。データセットのペアリング手順は同じで、`model_family` が `qwen_image` に変わるだけです。
