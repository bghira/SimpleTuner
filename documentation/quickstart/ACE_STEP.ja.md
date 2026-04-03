# ACE-Step クイックスタート

この例では、ACE-Step の音声生成モデルをトレーニングします。SimpleTuner は現在、元の ACE-Step v1 3.5B の学習経路と、ACE-Step v1.5 バンドル向けの forward-compatible な LoRA 学習をサポートしています。

## 概要

ACE-Step は高品質な音声合成向けの transformer ベースのフローマッチング音声モデルです。SimpleTuner では次の flavour を使えます。

- `base`: 元の ACE-Step v1 3.5B の学習経路
- `v15-turbo`、`v15-base`、`v15-sft`: `ACE-Step/Ace-Step1.5` から読み込む ACE-Step v1.5 バンドルの各バリアント

## ハードウェア要件

ACE-Step は 3.5B パラメータのモデルで、Flux のような大型画像生成モデルと比べると比較的軽量です。

- **最小:** 12GB 以上の VRAM を持つ NVIDIA GPU（例: 3060、4070）。
- **推奨:** 24GB 以上の VRAM を持つ NVIDIA GPU（例: 3090、4090、A10G）で大きなバッチサイズが可能。
- **Mac:** Apple Silicon の MPS で対応（統合メモリ ~36GB 以上が必要）。

### ストレージ要件

> ⚠️ **ディスク使用量の注意:** 音声モデルの VAE キャッシュは大きくなりがちです。例として、60 秒の音声クリップ 1 本で ~89MB のキャッシュ latent が生成されます。これは学習時の VRAM 要件を大きく下げるための戦略です。データセットのキャッシュに十分なディスク容量を確保してください。

> 💡 **ヒント:** 大規模データセットでは `--vae_cache_disable` を使うことでディスクへの埋め込み書き込みを無効化できます。これによりオンデマンドキャッシュが暗黙に有効になり、ディスク使用量は抑えられますが、学習ループ中にエンコードが走るため時間とメモリ使用量が増えます。

> 💡 **ヒント:** `int8-quanto` 量子化を使うと、VRAM の少ない GPU（例: 12GB〜16GB）でも品質低下を最小限に抑えて学習できます。

## 前提条件

Python 3.10+ の環境を用意してください。

```bash
pip install simpletuner
```

## 設定

設定ファイルは整理しておくことを推奨します。このデモ用に専用フォルダを作成します。

```bash
mkdir -p config/acestep-training-demo
```

### 重要な設定

SimpleTuner で使える ACE-Step flavour は次のとおりです。

- `base`: 元の ACE-Step v1 3.5B
- `v15-turbo`、`v15-base`、`v15-sft`: ACE-Step v1.5 バンドルの各バリアント

目的のバリアントに合わせて設定を選んでください。

すぐに使える example preset は次の場所にあります。

- `simpletuner/examples/ace_step-v1-0.peft-lora`
- `simpletuner/examples/ace_step-v1-5.peft-lora`

`simpletuner train example=ace_step-v1-0.peft-lora` または `simpletuner train example=ace_step-v1-5.peft-lora` で直接起動できます。

#### ACE-Step v1 の例

以下の内容で `config/acestep-training-demo/config.json` を作成します:

<details>
<summary>設定例を表示</summary>

```json
{
  "model_family": "ace_step",
  "model_type": "lora",
  "model_flavour": "base",
  "pretrained_model_name_or_path": "ACE-Step/ACE-Step-v1-3.5B",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/acestep-training-demo/multidatabackend.json"
}
```
</details>

#### ACE-Step v1.5 の例

ACE-Step v1.5 では `model_family: "ace_step"` を維持したまま、v1.5 flavour を選び、チェックポイントのルートを共有 v1.5 バンドルに向けます。

<details>
<summary>設定例を表示</summary>

```json
{
  "model_family": "ace_step",
  "model_type": "lora",
  "model_flavour": "v15-base",
  "pretrained_model_name_or_path": "ACE-Step/Ace-Step1.5",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/acestep-training-demo/multidatabackend.json"
}
```
</details>

### 検証設定

進捗を確認するため、以下を `config.json` に追加します:

- **`validation_prompt`**: 生成したい音声のテキスト説明（例: "A catchy pop song with upbeat drums"）。
- **`validation_lyrics`**: （任意）モデルに歌わせたい歌詞。
- **`validation_audio_duration`**: 検証クリップの秒数（デフォルト: 30.0）。
- **`validation_guidance`**: ガイダンススケール（デフォルト: 約 3.0〜5.0）。
- **`validation_step_interval`**: サンプル生成の間隔（例: 100 ステップごと）。

> ⚠️ **ACE-Step v1.5 の制限:** 現在の SimpleTuner 統合は v1.5 の学習をサポートしていますが、組み込みの ACE-Step 検証/推論パイプラインはまだ v1.0 専用です。v1.5 実行では学習中の検証を無効化するか、上流/外部の推論ツールで検証してください。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTuner には、トレーニングの安定性とパフォーマンスを大幅に向上させる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが自身の入力を生成することで露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

</details>

## データセット設定

ACE-Step では **音声専用** のデータセット設定が必要です。

### オプション 1: デモデータセット（Hugging Face）

クイックスタートには、準備済みの [ACEStep-Songs プリセット](../data_presets/preset_audio_dataset_with_lyrics.md) を使用できます。

`config/acestep-training-demo/multidatabackend.json` を作成します:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "acestep-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "cache_dir_vae": "cache/vae/{model_family}/acestep-demo-data"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```
</details>

> caption_strategy のオプションと要件については [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

### オプション 2: ローカル音声ファイル

`config/acestep-training-demo/multidatabackend.json` を作成します:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "my-audio-dataset",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/my_audio_files",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "disabled": false
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```
</details>

### データ構成

音声ファイルは `datasets/my_audio_files` に配置します。SimpleTuner は以下の形式をサポートします:

- **ロスレス:** `.wav`, `.flac`, `.aiff`, `.alac`
- **ロッシー:** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **注:** MP3、AAC、WMA などを扱うにはシステムに **FFmpeg** をインストールしておく必要があります。

キャプションと歌詞は音声ファイルの隣にテキストファイルとして配置します:

- **音声:** `track_01.wav`
- **キャプション（プロンプト）:** `track_01.txt`（例: "A slow jazz ballad"）
- **歌詞（任意）:** `track_01.lyrics`（歌詞テキスト）

<details>
<summary>データセット構成例</summary>

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```
</details>

> 💡 **高度:** 異なる命名規則（例: `_lyrics.txt`）を使う場合は、データセット設定でカスタマイズできます。

<details>
<summary>歌詞ファイル名のカスタム例を表示</summary>

```json
"audio": {
  "lyrics_filename_format": "{filename}_lyrics.txt"
}
```
</details>

> ⚠️ **歌詞に関する注意:** サンプルに `.lyrics` ファイルが見つからない場合、歌詞埋め込みはゼロ化されます。ACE-Step は歌詞条件付けを想定しているため、歌詞なし（インスト）データで大きく学習すると、ゼロ歌詞入力でも高品質なインスト音声を生成できるようになるまでに追加の学習ステップが必要になる場合があります。

## トレーニング

環境を指定して学習を開始します:

```bash
simpletuner train env=acestep-training-demo
```

このコマンドは `config/acestep-training-demo/` 内の `config.json` を参照します。

> 💡 **ヒント（学習の継続）:** 既存の LoRA（公式 ACE-Step チェックポイントやコミュニティアダプタなど）から継続微調整する場合は `--init_lora` を使用します:
> ```bash
> simpletuner train env=acestep-training-demo --init_lora=/path/to/existing_lora.safetensors
> ```

### Lyrics Embedder のトレーニング（上流準拠）

> ℹ️ **バージョン注意:** `lyrics_embedder_train` は現在 ACE-Step v1 の学習経路にのみ適用されます。SimpleTuner の v1.5 forward-compatible LoRA 経路は decoder-only です。

上流の ACE-Step トレーナーは、デノイザと一緒に歌詞埋め込み器を微調整します。SimpleTuner で同様の挙動にするには（full または standard LoRA のみ）:

- 有効化: `lyrics_embedder_train: true`
- 追加設定（指定しない場合はメインの optimizer/scheduler を再利用）:
  - `lyrics_embedder_lr`
  - `lyrics_embedder_optimizer`
  - `lyrics_embedder_lr_scheduler`

例:

<details>
<summary>設定例を表示</summary>

```json
{
  "lyrics_embedder_train": true,
  "lyrics_embedder_lr": 5e-5,
  "lyrics_embedder_optimizer": "torch-adamw",
  "lyrics_embedder_lr_scheduler": "cosine_with_restarts"
}
```
</details>
埋め込み器の重みは LoRA 保存時に一緒にチェックポイントされ、再開時に復元されます。

## トラブルシューティング

- **検証エラー:** `num_validation_images` > 1（音声ではバッチサイズに相当）や画像ベースのメトリクス（CLIP スコア）など、画像向けの検証機能を使っていないか確認してください。
- **メモリ不足:** OOM の場合は `train_batch_size` を下げるか、`gradient_checkpointing` を有効にしてください。

## 上流トレーナーからの移行

元の ACE-Step トレーニングスクリプトから移行する場合の対応表です:

| Upstream Parameter | SimpleTuner `config.json` | Default / Notes |
| :--- | :--- | :--- |
| `--learning_rate` | `learning_rate` | `1e-4` |
| `--num_workers` | `dataloader_num_workers` | `8` |
| `--max_steps` | `max_train_steps` | `2000000` |
| `--every_n_train_steps` | `checkpointing_steps` | `2000` |
| `--precision` | `mixed_precision` | `"fp16"` または `"bf16"`（fp32 は `"no"`） |
| `--accumulate_grad_batches` | `gradient_accumulation_steps` | `1` |
| `--gradient_clip_val` | `max_grad_norm` | `0.5` |
| `--shift` | `flow_schedule_shift` | `3.0`（ACE-Step 固有） |

### 生データの変換

生の音声/テキスト/歌詞ファイルがあり、Hugging Face 形式（上流の `convert2hf_dataset.py` ツールで使われる）に変換したい場合、生成されたデータセットはそのまま SimpleTuner で利用できます。

上流コンバータは `tags` と `norm_lyrics` カラムを持つデータセットを作成します。これらを使う場合は以下のように設定します:

<details>
<summary>設定例を表示</summary>

```json
{
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "path/to/converted/dataset",
    "config": {
        "audio_caption_fields": ["tags"],
        "lyrics_column": "norm_lyrics"
    }
}
```
</details>
