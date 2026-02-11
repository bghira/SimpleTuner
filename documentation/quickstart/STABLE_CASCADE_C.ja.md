# Stable Cascade Stage C クイックスタート

このガイドでは、**Stable Cascade Stage C prior** を SimpleTuner で微調整するための設定を説明します。Stage C は Stage B/C デコーダスタックへ渡す text-to-image prior を学習するため、ここでの学習品質が下流のデコーダ出力に直接影響します。ここでは LoRA トレーニングに焦点を当てますが、十分な VRAM がある場合はフルチューニングでも同じ手順が使えます。

> **注意:** Stage C は 1B+ パラメータの CLIP-G/14 テキストエンコーダと EfficientNet ベースのオートエンコーダを使用します。torchvision が必要で、テキスト埋め込みキャッシュは SDXL のプロンプトより 5〜6 倍大きくなると想定してください。

## ハードウェア要件

- **LoRA トレーニング:** 20〜24 GB VRAM（RTX 3090/4090、A6000 など）
- **フルモデル トレーニング:** 48 GB+ VRAM 推奨（A6000、A100、H100）。DeepSpeed/FSDP2 のオフロードで要件は下げられますが複雑になります。
- **システム RAM:** 32 GB 推奨。CLIP-G テキストエンコーダとキャッシュ処理が詰まらないようにします。
- **ディスク:** プロンプトキャッシュ用に最低 ~50 GB を確保。Stage C の CLIP-G 埋め込みは 1 本あたり ~4–6 MB 程度です。

## 前提条件

1. Python 3.13（プロジェクトの `.venv` と一致）。
2. GPU アクセラレーション用に CUDA 12.1+ または ROCm 5.7+（Apple M シリーズは Metal でも動作しますが Stage C は主に CUDA で検証）。
3. `torchvision`（Stable Cascade オートエンコーダに必須）と `accelerate`（トレーニング起動用）。

Python バージョンを確認:

```bash
python --version
```

不足パッケージのインストール（Ubuntu 例）:

```bash
sudo apt update && sudo apt install -y python3.13 python3.13-venv
```

## インストール

標準の SimpleTuner インストール（pip もしくはソース）に従ってください。一般的な CUDA ワークステーションの例:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

コントリビューターやリポジトリ直編集の場合は、ソースからインストールし `pip install -e .[cuda,dev]` を実行します。

## 環境設定

### 1. ベース設定のコピー

```bash
cp config/config.json.example config/config.json
```

以下のキーを設定します（Stage C に適したベースライン値）:

| キー | 推奨値 | 備考 |
| --- | -------------- | ----- |
| `model_family` | `"stable_cascade"` | Stage C コンポーネントを読み込むため必須 |
| `model_flavour` | `"stage-c"`（または `"stage-c-lite"`） | Lite は ~18 GB VRAM 程度向けにパラメータを削減 |
| `model_type` | `"lora"` | フルチューニングは可能だが大幅にメモリが必要 |
| `mixed_precision` | `"no"` | `i_know_what_i_am_doing=true` を設定しない限り Stage C は混合精度を拒否します。fp32 が安全 |
| `gradient_checkpointing` | `true` | VRAM を 3〜4 GB 節約 |
| `vae_batch_size` | `1` | Stage C のオートエンコーダは重いので小さく |
| `validation_resolution` | `"1024x1024"` | 下流デコーダの期待値に合わせる |
| `stable_cascade_use_decoder_for_validation` | `true` | 検証で prior+decoder を結合して実行 |
| `stable_cascade_decoder_model_name_or_path` | `"stabilityai/stable-cascade"` | カスタム Stage B/C デコーダを使う場合はローカルパスに変更 |
| `stable_cascade_validation_prior_num_inference_steps` | `20` | Prior のデノイズステップ |
| `stable_cascade_validation_prior_guidance_scale` | `3.0–4.0` | Prior の CFG |
| `stable_cascade_validation_decoder_guidance_scale` | `0.0–0.5` | デコーダ CFG（0.0 はフォトリアル、>0.0 でプロンプト順守が増える） |

#### `config/config.json` の例

<details>
<summary>設定例を表示</summary>

```json
{
  "base_model_precision": "int8-torchao",
  "checkpoint_step_interval": 100,
  "data_backend_config": "config/stable_cascade/multidatabackend.json",
  "gradient_accumulation_steps": 2,
  "gradient_checkpointing": true,
  "hub_model_id": "stable-cascade-stage-c-lora",
  "learning_rate": 1e-4,
  "lora_alpha": 16,
  "lora_rank": 16,
  "lora_type": "standard",
  "lr_scheduler": "cosine",
  "max_train_steps": 30000,
  "mixed_precision": "no",
  "model_family": "stable_cascade",
  "model_flavour": "stage-c",
  "model_type": "lora",
  "optimizer": "adamw_bf16",
  "output_dir": "output/stable_cascade_stage_c",
  "report_to": "wandb",
  "seed": 42,
  "stable_cascade_decoder_model_name_or_path": "stabilityai/stable-cascade",
  "stable_cascade_decoder_subfolder": "decoder_lite",
  "stable_cascade_use_decoder_for_validation": true,
  "stable_cascade_validation_decoder_guidance_scale": 0.0,
  "stable_cascade_validation_prior_guidance_scale": 3.5,
  "stable_cascade_validation_prior_num_inference_steps": 20,
  "train_batch_size": 4,
  "use_ema": true,
  "vae_batch_size": 1,
  "validation_guidance": 4.0,
  "validation_negative_prompt": "ugly, blurry, low-res",
  "validation_num_inference_steps": 30,
  "validation_prompt": "a cinematic photo of a shiba inu astronaut",
  "validation_resolution": "1024x1024"
}
```
</details>

重要ポイント:

- `model_flavour` は `stage-c` と `stage-c-lite` を受け付けます。VRAM が厳しい場合や蒸留 prior を使いたい場合は lite を選択してください。
- `mixed_precision` は `"no"` のままにしてください。上書きするなら `i_know_what_i_am_doing=true` を設定し、NaN への備えが必要です。
- `stable_cascade_use_decoder_for_validation` を有効にすると、prior 出力を Stage B/C デコーダに接続し、検証ギャラリーに潜在ではなく実画像を表示できます。

### 2. データバックエンドの設定

`config/stable_cascade/multidatabackend.json` を作成します:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "primary",
    "type": "local",
    "dataset_type": "images",
    "instance_data_dir": "/data/stable-cascade",
    "resolution": "1024x1024",
    "bucket_resolutions": ["1024x1024", "896x1152", "1152x896"],
    "crop": true,
    "crop_style": "random",
    "minimum_image_size": 768,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "caption_strategy": "filename",
    "prepend_instance_prompt": false,
    "repeats": 1
  },
  {
    "id": "stable-cascade-text-cache",
    "type": "local",
    "dataset_type": "text_embeds",
    "cache_dir": "/data/cache/stable-cascade/text",
    "default": true
  }
]
```
</details>

> caption_strategy のオプションと要件については [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

ヒント:

- Stage C の latent はオートエンコーダ由来なので、1024×1024（または縦横比の幅が狭い portrait/landscape バケット）に絞ってください。デコーダは 1024px 入力から ~24×24 の latent グリッドを期待します。
- `target_downsample_size` は 1024 のままにして、細いクロップでアスペクト比が ~2:1 を超えないようにします。
- 専用のテキスト埋め込みキャッシュは必須です。無いと毎回 30〜60 分 CLIP-G で再埋め込みすることになります。

### 3. プロンプトライブラリ（任意）

`config/stable_cascade/prompt_library.json` を作成します:

<details>
<summary>設定例を表示</summary>

```json
{
  "portrait": "a cinematic portrait photograph lit by studio strobes",
  "landscape": "a sweeping ultra wide landscape with volumetric lighting",
  "product": "a product render on a seamless background, dramatic reflections",
  "stylized": "digital illustration in the style of a retro sci-fi book cover"
}
```
</details>

`"validation_prompt_library": "config/stable_cascade/prompt_library.json"` を設定に追加してください。

## トレーニング

1. 環境を有効化し、未設定なら Accelerate の設定を行います:

```bash
source .venv/bin/activate
accelerate config
```

2. トレーニング開始:

```bash
accelerate launch simpletuner/train.py \
  --config_file config/config.json \
  --data_backend_config config/stable_cascade/multidatabackend.json
```

最初のエポックでは以下を監視してください:

- **テキストキャッシュのスループット** – Stage C はキャッシュ進捗をログに出します。ハイエンド GPU で ~8–12 プロンプト/秒が目安です。
- **VRAM 使用率** – 検証時の OOM を避けるため <95% を目指します。
- **検証出力** – 結合パイプラインは `output/<run>/validation/` にフル解像度 PNG を出力します。

## 検証・推論の注意

- Stage C prior 単体では画像埋め込みのみを生成します。SimpleTuner の検証ラッパは `stable_cascade_use_decoder_for_validation=true` の場合、自動でデコーダに通します。
- デコーダのフレーバーを切り替えるには `stable_cascade_decoder_subfolder` を `"decoder"`、`"decoder_lite"`、または Stage B/C 重みを含むカスタムフォルダに設定します。
- 速いプレビューのためには `stable_cascade_validation_prior_num_inference_steps` を ~12、`validation_num_inference_steps` を 20 に下げます。満足したら高品質のために戻してください。

## 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTuner には、トレーニングの安定性とパフォーマンスを大幅に向上させる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが自身の入力を生成することで露出バイアスを減らし、出力品質を向上させます。
*   **[Diff2Flow](../experimental/DIFF2FLOW.md):** Flow Matching 目的で Stable Cascade をトレーニングできます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

</details>

## トラブルシューティング

| 症状 | 対処 |
| --- | --- |
| "Stable Cascade Stage C requires --mixed_precision=no" | `"mixed_precision": "no"` を設定するか `"i_know_what_i_am_doing": true` を追加（非推奨） |
| 検証が prior（緑ノイズ）だけになる | `stable_cascade_use_decoder_for_validation` が `true` であり、デコーダ重みがダウンロード済みか確認 |
| テキスト埋め込みキャッシュに数時間かかる | SSD/NVMe をキャッシュ先に使い、ネットワークマウントは避ける。`simpletuner-text-cache` CLI で事前計算も検討 |
| オートエンコーダの import エラー | `.venv` に torchvision をインストール（`pip install torchvision --extra-index-url https://download.pytorch.org/whl/cu124`）。Stage C は EfficientNet 重みが必要 |

## 次のステップ

- `lora_rank`（8〜32）や `learning_rate`（5e-5〜2e-4）を題材の複雑さに応じて調整。
- prior 学習後に Stage B へ ControlNet/conditioning アダプタを付与。
- 反復を速くしたい場合は `stage-c-lite` で学習し、検証には `decoder_lite` を使う。

Happy tuning!
