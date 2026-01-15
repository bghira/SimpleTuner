# FLUX.2 クイックスタート

このガイドでは、Black Forest Labsの最新画像生成モデルファミリーであるFLUX.2でLoRAをトレーニングする方法を説明します。

> **注意**: デフォルトのモデルフレーバーは`klein-9b`ですが、このガイドは`dev`(24B Mistral-3テキストエンコーダーを搭載した完全な12Bトランスフォーマー)に焦点を当てています。これはリソース要件が最も高いためです。Kleinモデルはより簡単に実行できます - 下記の[モデルバリアント](#モデルバリアント)を参照してください。

## モデルバリアント

FLUX.2は3つのバリアントがあります:

| バリアント | トランスフォーマー | テキストエンコーダー | 総ブロック数 | デフォルト |
|---------|-------------|--------------|--------------|---------|
| `dev` | 12Bパラメータ | Mistral-3 (24B) | 56 (8+48) | |
| `klein-9b` | 9Bパラメータ | Qwen3 (同梱) | 32 (8+24) | ✓ |
| `klein-4b` | 4Bパラメータ | Qwen3 (同梱) | 25 (5+20) | |

**主な違い:**
- **dev**: スタンドアロンのMistral-Small-3.1-24Bテキストエンコーダーを使用、ガイダンスエンベッディングあり
- **kleinモデル**: モデルリポジトリに同梱されたQwen3テキストエンコーダーを使用、**ガイダンスエンベッディングなし**(ガイダンストレーニングオプションは無視されます)

バリアントを選択するには、設定で`model_flavour`を設定します:
```json
{
  "model_flavour": "dev"
}
```

## モデル概要

FLUX.2-devはFLUX.1から大幅なアーキテクチャ変更を導入しています:

- **テキストエンコーダー**: Mistral-Small-3.1-24B(dev)またはQwen3(klein)
- **アーキテクチャ**: 8 DoubleStreamBlocks + 48 SingleStreamBlocks(dev)
- **潜在チャネル**: 32 VAEチャネル → ピクセルシャッフル後128(FLUX.1の16に対して)
- **VAE**: バッチ正規化とピクセルシャッフリングを備えたカスタムVAE
- **エンベッディング次元**: devは15,360(3×5,120)、klein-9bは12,288(3×4,096)、klein-4bは7,680(3×2,560)

## ハードウェア要件

ハードウェア要件はモデルバリアントによって大きく異なります。

### Kleinモデル(ほとんどのユーザーに推奨)

Kleinモデルはより手軽に利用できます:

| バリアント | bf16 VRAM | int8 VRAM | システムRAM |
|---------|-----------|-----------|------------|
| `klein-4b` | ~12GB | ~8GB | 32GB以上 |
| `klein-9b` | ~22GB | ~14GB | 64GB以上 |

**klein-9b推奨**: シングル24GB GPU(RTX 3090/4090、A5000)
**klein-4b推奨**: シングル16GB GPU(RTX 4080、A4000)

### FLUX.2-dev(上級者向け)

FLUX.2-devはMistral-3テキストエンコーダーのため、かなりのリソースを必要とします:

#### VRAM要件

24BのMistralテキストエンコーダー単体でかなりのVRAMを必要とします:

| コンポーネント | bf16 | int8 | int4 |
|-----------|------|------|------|
| Mistral-3 (24B) | ~48GB | ~24GB | ~12GB |
| FLUX.2 Transformer | ~24GB | ~12GB | ~6GB |
| VAE + オーバーヘッド | ~4GB | ~4GB | ~4GB |

| 設定 | 概算総VRAM |
|--------------|------------------------|
| bf16すべて | ~76GB+ |
| int8テキストエンコーダー + bf16トランスフォーマー | ~52GB |
| int8すべて | ~40GB |
| int4テキストエンコーダー + int8トランスフォーマー | ~22GB |

#### システムRAM

- **最小**: 96GBシステムRAM(24Bテキストエンコーダーのロードにかなりのメモリが必要)
- **推奨**: 快適な動作には128GB以上

#### 推奨ハードウェア

- **最小**: 2x 48GB GPU(A6000、L40S)でFSDP2またはDeepSpeed
- **推奨**: 4x H100 80GBでfp8-torchao
- **重い量子化(int4)**: 2x 24GB GPUでも動作する可能性がありますが、実験的です

Mistral-3テキストエンコーダーとトランスフォーマーの合計サイズのため、FLUX.2-devにはマルチGPU分散トレーニング(FSDP2またはDeepSpeed)が本質的に必要です。

## 前提条件

### Pythonバージョン

FLUX.2にはPython 3.10以降と最新のtransformersが必要です:

```bash
python --version  # 3.10以上である必要があります
pip install transformers>=4.45.0
```

### モデルアクセス

FLUX.2モデルはHugging Faceでのアクセス承認が必要です:

**devの場合:**
1. [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)にアクセス
2. ライセンス契約に同意

**kleinモデルの場合:**
1. [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)または[black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)にアクセス
2. ライセンス契約に同意

Hugging Face CLIにログインしていることを確認: `huggingface-cli login`

## インストール

```bash
pip install 'simpletuner[cuda]'
```

開発セットアップの場合:
```bash
git clone https://github.com/bghira/SimpleTuner
cd SimpleTuner
pip install -e ".[cuda]"
```

## 設定

### Webインターフェース

```bash
simpletuner server
```

http://localhost:8001にアクセスし、モデルファミリーとしてFLUX.2を選択します。

### 手動設定

`config/config.json`を作成:

<details>
<summary>設定例を表示</summary>

```json
{
  "model_type": "lora",
  "model_family": "flux2",
  "model_flavour": "dev",
  "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-dev",
  "output_dir": "/path/to/output",
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "gradient_checkpointing": true,
  "mixed_precision": "bf16",
  "learning_rate": 1e-4,
  "lr_scheduler": "constant",
  "max_train_steps": 10000,
  "validation_resolution": "1024x1024",
  "validation_num_inference_steps": 20,
  "flux_guidance_mode": "constant",
  "flux_guidance_value": 1.0,
  "lora_rank": 16
}
```
</details>

### 主要な設定オプション

#### ガイダンス設定

> **注意**: Kleinモデル(`klein-4b`、`klein-9b`)にはガイダンスエンベッディングがありません。以下のガイダンスオプションは`dev`にのみ適用されます。

FLUX.2-devはFLUX.1と同様にガイダンスエンベッディングを使用します:

<details>
<summary>設定例を表示</summary>

```json
{
  "flux_guidance_mode": "constant",
  "flux_guidance_value": 1.0
}
```
</details>

または、トレーニング中にランダムガイダンスを使用する場合:

<details>
<summary>設定例を表示</summary>

```json
{
  "flux_guidance_mode": "random-range",
  "flux_guidance_min": 1.0,
  "flux_guidance_max": 5.0
}
```
</details>

#### 量子化(メモリ最適化)

VRAM使用量を削減する場合:

<details>
<summary>設定例を表示</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "int8-quanto",
  "base_model_default_dtype": "bf16"
}
```
</details>

#### TREAD(トレーニング加速)

FLUX.2はより高速なトレーニングのためにTREADをサポートしています:

<details>
<summary>設定例を表示</summary>

```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.5, "start_layer_idx": 2, "end_layer_idx": -2}
    ]
  }
}
```
</details>

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTunerには、トレーニングの安定性とパフォーマンスを大幅に向上させることができる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが独自の入力を生成できるようにすることで、露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

</details>

### データセット設定

`config/multidatabackend.json`を作成:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "my-dataset",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux2/my-dataset",
    "instance_data_dir": "datasets/my-dataset",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux2",
    "write_batch_size": 64
  }
]
```
</details>

> caption_strategyのオプションと要件については、[DATALOADER.md](../DATALOADER.md#caption_strategy)を参照してください。

### オプションの編集/参照コンディショニング

FLUX.2は**プレーンなtext-to-image**(コンディショニングなし)または**ペア参照/編集画像**でトレーニングできます。コンディショニングを追加するには、[`conditioning_data`](../DATALOADER.md#conditioning_data)を使用してメインデータセットを1つ以上の`conditioning`データセットにペアリングし、[`conditioning_type`](../DATALOADER.md#conditioning_type)を選択します:

<details>
<summary>設定例を表示</summary>

```jsonc
[
  {
    "id": "flux2-edits",
    "type": "local",
    "instance_data_dir": "/datasets/flux2/edits",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": ["flux2-references"],
    "cache_dir_vae": "cache/vae/flux2/edits"
  },
  {
    "id": "flux2-references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/flux2/references",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/flux2/references"
  }
]
```
</details>

- 編集画像と1:1で揃えたクロップが必要な場合は`conditioning_type=reference_strict`を使用します。`reference_loose`はアスペクト比の不一致を許可します。
- ファイル名は編集データセットと参照データセット間で一致する必要があります。各編集画像には対応する参照ファイルが必要です。
- 複数のコンディショニングデータセットを提供する場合は、必要に応じて`conditioning_multidataset_sampling`(`combined`または`random`)を設定します。[OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling)を参照してください。
- `conditioning_data`がない場合、FLUX.2は標準のtext-to-imageトレーニングにフォールバックします。

### LoRAターゲット

利用可能なLoRAターゲットプリセット:

- `all`(デフォルト): すべてのアテンションとMLPレイヤー
- `attention`: アテンションレイヤーのみ(qkv、proj)
- `mlp`: MLP/フィードフォワードレイヤーのみ
- `tiny`: 最小限のトレーニング(qkvレイヤーのみ)

<details>
<summary>設定例を表示</summary>

```json
{
  "--flux_lora_target": "all"
}
```
</details>

## トレーニング

### サービスへのログイン

```bash
huggingface-cli login
wandb login  # オプション
```

### トレーニング開始

```bash
simpletuner train
```

またはスクリプト経由:

```bash
./train.sh
```

### メモリオフロード

メモリ制約のあるセットアップの場合、FLUX.2はトランスフォーマーとオプションでMistral-3テキストエンコーダーの両方についてグループオフロードをサポートしています:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
--group_offload_text_encoder
```

`--group_offload_text_encoder`フラグは、24BのMistralテキストエンコーダーがテキストエンベッディングキャッシング中のオフロードから大きな恩恵を受けるため、FLUX.2に推奨されます。潜在キャッシング中にVAEをオフロードに含めるには`--group_offload_vae`も追加できます。

## 検証プロンプト

`config/user_prompt_library.json`を作成:

<details>
<summary>設定例を表示</summary>

```json
{
  "portrait_subject": "a professional portrait photograph of <subject>, studio lighting, high detail",
  "artistic_subject": "an artistic interpretation of <subject> in the style of renaissance painting",
  "cinematic_subject": "a cinematic shot of <subject>, dramatic lighting, film grain"
}
```
</details>

## 推論

### トレーニング済みLoRAの使用

FLUX.2 LoRAは、SimpleTuner推論パイプラインまたはコミュニティサポートが開発されれば互換性のあるツールでロードできます。

### ガイダンススケール

- `flux_guidance_value=1.0`でのトレーニングはほとんどのユースケースでうまく機能します
- 推論時には通常のガイダンス値(3.0-5.0)を使用します

## FLUX.1との違い

| 側面 | FLUX.1 | FLUX.2-dev | FLUX.2-klein-9b | FLUX.2-klein-4b |
|--------|--------|------------|-----------------|-----------------|
| テキストエンコーダー | CLIP-L/14 + T5-XXL | Mistral-3 (24B) | Qwen3 (同梱) | Qwen3 (同梱) |
| エンベッディング次元 | CLIP: 768、T5: 4096 | 15,360 | 12,288 | 7,680 |
| 潜在チャネル | 16 | 32(→128) | 32(→128) | 32(→128) |
| VAE | AutoencoderKL | カスタム(BatchNorm) | カスタム(BatchNorm) | カスタム(BatchNorm) |
| トランスフォーマーブロック | 19 joint + 38 single | 8 double + 48 single | 8 double + 24 single | 5 double + 20 single |
| ガイダンスエンベッド | あり | あり | なし | なし |

## トラブルシューティング

### 起動時のメモリ不足

- `--offload_during_startup=true`を有効化
- テキストエンコーダー量子化に`--quantize_via=cpu`を使用
- `--vae_batch_size`を減らす

### テキストエンベッディングが遅い

Mistral-3は大きいため、以下を検討:
- トレーニング前にすべてのテキストエンベッディングを事前キャッシュ
- テキストエンコーダー量子化を使用
- より大きな`write_batch_size`でバッチ処理

### トレーニングの不安定性

- 学習率を下げる(5e-5を試す)
- 勾配累積ステップを増やす
- 勾配チェックポイントを有効化
- `--max_grad_norm=1.0`を使用

### CUDA メモリ不足

- 量子化を有効化(`int8-quanto`または`int4-quanto`)
- 勾配チェックポイントを有効化
- バッチサイズを減らす
- グループオフロードを有効化
- トークンルーティング効率のためにTREADを使用

## 高度: TREAD設定

TREAD(Token Routing for Efficient Architecture-agnostic Diffusion)は、選択的にトークンを処理することでトレーニングを高速化します:

<details>
<summary>設定例を表示</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 4,
        "end_layer_idx": -4
      }
    ]
  }
}
```
</details>

- `selection_ratio`: 保持するトークンの割合(0.5 = 50%)
- `start_layer_idx`: ルーティングを適用する最初のレイヤー
- `end_layer_idx`: 最後のレイヤー(負の値 = 末尾から)

期待される高速化: 設定に応じて20-40%。

## 関連項目

- [FLUX.1クイックスタート](FLUX.md) - FLUX.1トレーニング
- [TREADドキュメント](../TREAD.md) - 詳細なTREAD設定
- [LyCORISトレーニングガイド](../LYCORIS.md) - LoRAとLyCORISトレーニング方法
- [データローダー設定](../DATALOADER.md) - データセットセットアップ
