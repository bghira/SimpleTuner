# HeartMuLa クイックスタート

この例では、HeartMuLa oss 3B の音声生成モデルをトレーニングします。

## 概要

HeartMuLa は 3B パラメータの自己回帰トランスフォーマーで、タグと歌詞から離散的な音声トークンを予測します。トークンは HeartCodec でデコードして波形を生成します。

## ハードウェア要件

HeartMuLa は 3B パラメータのモデルで、Flux のような大型画像生成モデルと比べると比較的軽量です。

- **最小:** 12GB 以上の VRAM を持つ NVIDIA GPU（例: 3060、4070）。
- **推奨:** 24GB 以上の VRAM を持つ NVIDIA GPU（例: 3090、4090、A10G）で大きなバッチサイズが可能。
- **Mac:** Apple Silicon の MPS で対応（統合メモリ ~36GB 以上が必要）。

### ストレージ要件

> ⚠️ **トークンデータセットの注意:** HeartMuLa は事前計算された音声トークンで学習します。SimpleTuner は学習中にトークンを生成しないため、データセット側で `audio_tokens` または `audio_tokens_path` のメタデータを用意してください。トークンファイルは大きくなる可能性があるため、ディスク容量に注意してください。

> 💡 **ヒント:** `int8-quanto` 量子化を使うと、VRAM の少ない GPU（例: 12GB〜16GB）でも品質低下を最小限に抑えて学習できます。

## 前提条件

Python 3.10+ の環境を用意してください。

```bash
pip install simpletuner
```

## 設定

設定ファイルは整理しておくことを推奨します。このデモ用に専用フォルダを作成します。

```bash
mkdir -p config/heartmula-training-demo
```

### 重要な設定

`config/heartmula-training-demo/config.json` を以下の内容で作成します:

<details>
<summary>設定例を表示</summary>

```json
{
  "model_family": "heartmula",
  "model_type": "lora",
  "model_flavour": "3b",
  "pretrained_model_name_or_path": "HeartMuLa/HeartMuLa-oss-3B",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/heartmula-training-demo/multidatabackend.json"
}
```
</details>

### バリデーション設定

進捗確認のために `config.json` に以下を追加します:

- **`validation_prompt`**: タグまたは音声の説明文（例: "明るいシンセのアップビートなポップ"）。
- **`validation_lyrics`**: (任意) モデルに歌わせる歌詞。インストゥルメンタルは空文字を使用。
- **`validation_prompt_library`**: 組み込みの caption + lyrics 検証ライブラリには `"audio"` を使います。
- **`validation_audio_duration`**: バリデーションクリップの長さ（秒、デフォルト: 30.0）。
- **`validation_guidance`**: ガイダンススケール（1.5 - 3.0 付近から開始）。
- **`validation_step_interval`**: サンプル生成の頻度（例: 100 ステップごと）。

### 高度な実験機能

<details>
<summary>高度な実験内容を表示</summary>


SimpleTuner には学習の安定性と性能を大きく改善できる実験機能があります。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** 露出バイアスを減らし、学習中にモデル自身の入力生成を使うことで出力品質を改善します。

> ⚠️ これらの機能は学習の計算負荷を増やします。

</details>

## データセット設定

HeartMuLa は事前計算済みトークンを含む **音声専用** データセットが必要です。

各サンプルに必要な項目:

- `tags`（文字列）
- `lyrics`（文字列、空でも可）
- `audio_tokens` または `audio_tokens_path`

トークン配列は 2D で、形状は `[frames, num_codebooks]` または `[num_codebooks, frames]` です。

> 💡 **注意:** HeartMuLa は独立したテキストエンコーダを使わないため、text-embeds バックエンドは不要です。

### オプション 1: Hugging Face データセット（列にトークン）

`config/heartmula-training-demo/multidatabackend.json` を作成します:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "heartmula-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "your-org/heartmula-audio-tokens",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "config": {
      "audio_caption_fields": ["tags"],
      "lyrics_column": "lyrics"
    }
  }
]
```
</details>

テキスト列に加えて `audio_tokens` または `audio_tokens_path` 列が含まれることを確認してください。

### オプション 2: ローカル音声ファイル + トークンメタデータ

`config/heartmula-training-demo/multidatabackend.json` を作成します:

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
  }
]
```
</details>

各サンプルに `audio_tokens` または `audio_tokens_path` を提供できるメタデータバックエンドを用意してください。

### データ構成

音声ファイルを `datasets/my_audio_files` に配置します。SimpleTuner は以下の形式に対応しています:

- **ロスレス:** `.wav`, `.flac`, `.aiff`, `.alac`
- **ロッシー:** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **注意:** MP3、AAC、WMA などを使うには **FFmpeg** が必要です。

`caption_strategy: textfile` を使う場合は、タグと歌詞のテキストファイルを音声ファイルと同じ場所に置いてください:

- **音声:** `track_01.wav`
- **タグ (プロンプト):** `track_01.txt`（例: "ゆったりしたジャズバラード"）
- **歌詞 (任意):** `track_01.lyrics`

トークン配列はメタデータから渡します（例: `.npy` / `.npz` を指す `audio_tokens_path`）。

<details>
<summary>データセット構成例</summary>

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
├── track_01.lyrics
└── track_01.tokens.npy
```
</details>

> ⚠️ **歌詞の注意:** HeartMuLa は各サンプルに歌詞文字列を要求します。インストゥルメンタルは空文字を設定してください。

## トレーニング

環境を指定してトレーニングを開始します:

```bash
simpletuner train env=heartmula-training-demo
```

このコマンドは `config/heartmula-training-demo/` 内の `config.json` を読み込みます。

> 💡 **ヒント (学習の継続):** 既存の LoRA から再開する場合は `--init_lora` を使います:
> ```bash
> simpletuner train env=heartmula-training-demo --init_lora=/path/to/existing_lora.safetensors
> ```

## トラブルシューティング

- **バリデーションエラー:** `num_validation_images` > 1（音声ではバッチサイズに相当）や、CLIP スコアなどの画像向けメトリクスは使用しないでください。
- **メモリ不足:** OOM の場合は `train_batch_size` を減らすか `gradient_checkpointing` を有効にしてください。
