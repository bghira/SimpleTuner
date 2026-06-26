# Krea2 クイックスタート

このガイドでは、SimpleTuner で Krea2 の LoRA をトレーニングする方法を扱います。Krea2 は Qwen 系のテキスト条件付けと Qwen Image VAE を使う、大規模な flow matching 画像 transformer です。高メモリの NVIDIA GPU で使うのが現実的です。

スターター例:

```bash
simpletuner/examples/krea2.peft-lora/config.json
```

## 推奨スタート設定

最初の実行では、例の設定をベースにして保守的に始めてください。

```json
{
  "model_family": "krea2",
  "model_flavour": "raw",
  "model_type": "lora",
  "pretrained_model_name_or_path": "krea/Krea-2-Raw",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "fuse_qkv_projections": true,
  "train_batch_size": 1,
  "base_model_precision": "no_change"
}
```

Krea2 は 1024px ネイティブの画像モデルですが、512px や 768px は高速な確認に便利です。実行が安定してから 1024px の dataloader に移行してください。

## ハードウェアメモ

Krea2 は 80GB H100 で、1024px・batch 1 の bf16 トレーニングが可能でした。compile なしなら大きめの batch も入りましたが、compile は graph/cudagraph 用のメモリを大きく増やすため、多くの大きな batch 設定で OOM になります。

TorchAO int8 weight-only は VRAM を大きく下げますが、今回テストした SimpleTuner のトレーニング経路では bf16 より高速ではありませんでした。速度よりメモリ余裕が重要な場合に使ってください。

推奨:

- 入るなら `bf16` を使う。
- メモリが足りない場合は `int8-torchao` を使う。
- `gradient_checkpointing=true` を維持する。
- `fuse_qkv_projections=true` を維持する。
- `dynamo_backend=inductor`、`dynamo_mode=reduce-overhead`、`dynamo_use_regional_compilation=true` は、batch/resolution が入ることを確認してから使う。

## 主要設定

```json
{
  "model_family": "krea2",
  "model_flavour": "raw",
  "model_type": "lora",
  "pretrained_model_name_or_path": "krea/Krea-2-Raw",
  "base_model_precision": "no_change",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "fuse_qkv_projections": true,
  "optimizer": "optimi-lion",
  "learning_rate": 1e-4,
  "lora_rank": 64,
  "train_batch_size": 1,
  "resolution": 1024,
  "validation_resolution": "1024x1024"
}
```

TorchAO int8:

```json
{
  "base_model_precision": "int8-torchao",
  "quantize_via": "cpu"
}
```

reduce-overhead compile:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_mode": "reduce-overhead",
  "dynamo_use_regional_compilation": true
}
```

## 参照画像トレーニング

Krea2 は編集系 dataset 向けに、任意の参照 latent 条件付けをサポートします。ペアの参照画像またはキャッシュ済み参照 latent を dataloader が提供する場合に有効化します。

```json
{
  "krea2_reference_latents": true
}
```

参照 latent はターゲット latent と同じ shape である必要があります。

## Dataloader 設定

Krea2 は他の画像 transformer と同じ基本 dataloader 形式を使います。実際の学習解像度はトップレベルの `resolution` だけではなく、dataloader JSON の `resolution`、`maximum_image_size`、`target_downsample_size` で決まります。1024px で学習する場合は、dataloader 側も 1024 にしてください。

512px dataset は、caption、crop、learning rate の問題を素早く見つけるのに便利です。最終品質を確認するには 1024px の run がより信頼できます。

ローカル dataset では `type: local`、`instance_data_dir`、caption strategy を設定します。小さな subject LoRA は `caption_strategy=instanceprompt` から始められます。style LoRA では filenames や通常 caption の方が向くことがあります。

## 検証

Krea2 の validation は重いので、調整中は prompt を少なくしてください。1つの prompt だけでは overfit や暗記を見逃すことがあります。安定したら小さな prompt library を追加します。

```json
{
  "validation_prompt": "a studio portrait of <token>, soft directional light, detailed fabric texture",
  "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
  "validation_num_inference_steps": 28,
  "validation_guidance": 4.5,
  "validation_resolution": "1024x1024"
}
```

## 量子化メモ

`int8-torchao` は transformer のベース重みを int8 に保存し、その上で bf16 LoRA 重みを学習します。H100 では VRAM を大きく削減しましたが、この training path では bf16 より高速ではありませんでした。速度ではなく容量のための選択肢として考えてください。

## ベンチマーク結果

以下は、単一の NVIDIA H100 80GB、SimpleTuner の実トレーナー、Krea2 LoRA、QKV fusion、gradient checkpointing、小さな Domokun dataset で測定した結果です。VRAM は `nvidia-smi` で外部サンプリングしました。比較用の目安として扱ってください。PyTorch、CUDA、driver、dataset、LoRA rank、optimizer、attention backend、GPU が変わると結果も変わります。

### QKV fusion + checkpointing、compile オフ

| 精度 | 解像度 | Batch | 安定時 s/step | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 512 | 1 | 0.353 | 31.10 GiB |
| bf16 | 512 | 4 | 1.230 | 39.31 GiB |
| bf16 | 512 | 8 | 2.430 | 50.32 GiB |
| bf16 | 1024 | 1 | 0.990 | 33.28 GiB |
| bf16 | 1024 | 4 | 3.850 | 48.35 GiB |
| bf16 | 1024 | 8 | 7.690 | 67.88 GiB |
| int8-torchao | 512 | 1 | 0.535 | 18.10 GiB |
| int8-torchao | 512 | 4 | 1.690 | 27.46 GiB |
| int8-torchao | 512 | 8 | 3.220 | 40.52 GiB |
| int8-torchao | 1024 | 1 | 1.330 | 20.35 GiB |
| int8-torchao | 1024 | 4 | 4.850 | 36.99 GiB |
| int8-torchao | 1024 | 8 | 9.520 | 58.84 GiB |

### QKV fusion + checkpointing + reduce-overhead compile

| 精度 | 解像度 | Batch | 状態 | 安定時 s/step | Peak VRAM |
| --- | ---: | ---: | --- | ---: | ---: |
| bf16 | 512 | 1 | ok | 0.260 | 41.20 GiB |
| bf16 | 512 | 4 | OOM | - | 79.07 GiB |
| bf16 | 512 | 8 | OOM | - | 79.10 GiB |
| bf16 | 1024 | 1 | ok | 0.704 | 63.71 GiB |
| bf16 | 1024 | 4 | OOM | - | 79.11 GiB |
| bf16 | 1024 | 8 | OOM | - | 78.40 GiB |
| int8-torchao | 512 | 1 | ok | 0.410 | 30.93 GiB |
| int8-torchao | 512 | 4 | ok | 1.300 | 78.60 GiB |
| int8-torchao | 512 | 8 | OOM | - | 79.12 GiB |
| int8-torchao | 1024 | 1 | ok | 0.990 | 58.68 GiB |
| int8-torchao | 1024 | 4 | OOM | - | 78.92 GiB |
| int8-torchao | 1024 | 8 | OOM | - | 78.09 GiB |

## 実用上の指針

- H100 単一 GPU で最速に試すなら、bf16、QKV fusion、checkpointing、compile オン、batch 1。
- 大きな effective batch が必要なら、compile なしの bf16 で `train_batch_size` を上げる。
- メモリ制約が強い場合は `int8-torchao` を使う。ただし step は遅くなる可能性がある。
- compile は batch 1 では有効ですが、VRAM を大きく増やし、大きな batch を失敗させることがあります。

## よくある問題

- 1024px のつもりで log が 512px の場合は、dataloader JSON を確認してください。
- compile で OOM し、compile なしで入る場合は、batch size を下げるか compile を無効にしてください。
- int8 が低 VRAM でも遅い場合、それは今回の H100 測定と一致します。
- 参照画像が validation に効かない場合は、`krea2_reference_latents=true` と paired reference dataset を確認してください。
- すぐ overfit する場合は、learning rate、step 数、dataset の多様性を見直してください。
