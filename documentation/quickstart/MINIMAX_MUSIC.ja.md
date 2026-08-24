# MiniMax Music 3 クイックスタート

このガイドでは、SimpleTuner で MiniMax Music 3 の LoRA 学習を設定します。

## 概要

MiniMax Music 3 は、キャプションと歌詞で条件付けする音楽生成モデルです。Diffusers レイアウトでは、Qwen3 の自己回帰言語モデルでテキスト/音声条件を作り、128 チャンネルの DAV latent を flow-matching transformer で学習し、decoder/vocoder で検証音声を生成します。

SimpleTuner は以下をサポートします。

- transformer の LoRA、LyCORIS、full-rank 学習
- 元の `dav.pth` autoencoder を使った raw audio からの VAECache エンコード
- 音声データセット metadata からの caption、lyrics、duration 条件
- `validation_prompt`、`validation_lyrics`、`validation_audio_duration`、プロンプトライブラリによる検証音声
- `lora_format: "comfyui"` による ComfyUI MiniMax Music LoRA の import/export
- AnyFlow、TwinFlow、CREPA self-flow、LayerSync

## ハードウェア要件

MiniMax Music 3 は 2.4B の flow transformer と 8B の Qwen3 AR 条件モデルを使います。

- **最低:** 控えめな LoRA 学習には 24GB 以上の VRAM を持つ NVIDIA GPU。
- **推奨:** 高い rank、長いクリップ、頻繁な検証には 48GB 以上の VRAM、または CPU/RAM offload。
- **Mac:** 一部は MPS で動く可能性がありますが、学習と検証の実用的な対象は CUDA です。

まず `base_model_precision: "int8-quanto"`、`text_encoder_1_precision: "int8-quanto"`、`gradient_checkpointing: true` から始めてください。text encoder がボトルネックの場合は、LoRA rank を上げる前に text encoder offload を使います。

## 前提条件

SimpleTuner と、音声読み込み用の FFmpeg をインストールします。

```bash
pip install simpletuner
```

手動インストールや開発環境については、[インストール手順](../INSTALL.md)を参照してください。

## 設定

専用の設定ディレクトリを作成します。

```bash
mkdir -p config/minimaxmusic-training-demo
```

`config/minimaxmusic-training-demo/config.json` を作成します。

<details>
<summary>設定例を見る</summary>

```json
{
  "model_family": "minimaxmusic",
  "model_type": "lora",
  "model_flavour": "music3",
  "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-Music3",
  "pretrained_vae_model_name_or_path": "SimpleTuner/MiniMax-Music-3-Encoder",
  "resolution": 512,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "int8-quanto",
  "gradient_checkpointing": true,
  "lora_rank": 64,
  "lora_format": "comfyui",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "train_batch_size": 1,
  "vae_batch_size": 1,
  "data_backend_config": "config/minimaxmusic-training-demo/multidatabackend.json",
  "validation_prompt": "bright synth pop with clean vocal melody and crisp percussion",
  "validation_lyrics": "[verse]\nturning sparks into a skyline\n[chorus]\nwe keep singing through the night",
  "validation_audio_duration": 30,
  "validation_guidance": 1.7,
  "validation_num_inference_steps": 30,
  "validation_steps": 50,
  "validation_disable_unconditional": true
}
```
</details>

テンプレートは以下にあります。

- `simpletuner/examples/minimaxmusic-music3.peft-lora`
- `simpletuner/examples/minimaxmusic-audio.json`
- `simpletuner/examples/minimaxmusic-prompts.json`

例を実行します。

```bash
simpletuner train example=minimaxmusic-music3.peft-lora
```

## VAECache

MiniMax Music 3 の raw audio cache は DAV audio autoencoder を使います。推奨する SimpleTuner VAE repository は `SimpleTuner/MiniMax-Music-3-Encoder` で、変換済み component は Diffusers-style loading 用に `audio_vae/` に保存されています。

上流の `MiniMaxAI/MiniMax-Music3` repository にも元の `dav.pth` が含まれており、SimpleTuner はそれも直接読み込めます。ローカルに変換した Diffusers ディレクトリを使う場合は、checkpoint のルートに `dav.pth` を置くか、`pretrained_vae_model_name_or_path` を `dav.pth` または `audio_vae/` を含む場所に向けてください。`vocoder/` だけでも検証 decode はできますが、raw audio の VAE caching には不足します。

## データセット設定

MiniMax Music 3 には **audio** dataset と **text embeds** cache backend が必要です。

```json
[
  {
    "id": "minimaxmusic-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "audio": {
      "bucket_strategy": "duration",
      "duration_interval": 3.0,
      "max_duration_seconds": 30
    },
    "cache_dir_vae": "cache/vae/{model_family}/minimaxmusic-demo-data"
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

ローカル音声では、`.txt` に説明文、`.lyrics` に歌詞を置きます。

```text
datasets/minimaxmusic-audio/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```

## 検証設定

- **`validation_prompt`**: 音楽の説明や tags。
- **`validation_lyrics`**: 歌唱用の歌詞。インストゥルメンタル検証では空文字を使います。
- **`validation_audio_duration`**: 生成するクリップの秒数。
- **`validation_guidance`**: CFG scale。`1.5` から `2.0` 付近で始めます。
- **`validation_num_inference_steps`**: 検証 sampling steps。まず `30` 前後にします。
- **`validation_steps`**: 検証音声を生成する間隔。
- **`validation_prompt_library`**: 組み込みの音楽 caption + lyrics ライブラリには `"audio"` を使います。
- **`user_prompt_library`**: JSON ライブラリへのパス。エントリは `prompt` または `caption` と、任意の複数行 `lyrics` を使えます。

## 学習

```bash
simpletuner train env=minimaxmusic-training-demo
```

既存の MiniMax Music 3 LoRA から始める場合:

```bash
simpletuner train env=minimaxmusic-training-demo --init_lora=/path/to/adapter.safetensors --init_lora_step=0
```

adapter が native ComfyUI 形式なら、設定に `lora_format: "comfyui"` を残してください。SimpleTuner は学習時に変換し、同じ形式で export します。

## 高度な機能

MiniMax Music 3 は SimpleTuner の flow-matching 学習パスを使うため、AnyFlow、TwinFlow、CREPA self-flow、LayerSync を利用できます。まず標準 LoRA で始め、必要な機能を 1 つずつ追加してください。

## 言語モデル（AR ステージ）のトレーニング

MiniMax Music 3 のセマンティックコードを計画する Qwen3 言語モデルを、音楽 DiT の代わりにトレーニングできます — 音楽スタイルをキーワードに結びつける dreambooth 式トリガーワードに便利です。

[fiona crapple](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple) は、このモードで作成した完全な LM LoRA トレーニング例で、設定、チェックポイント、音声比較が含まれています。

```json
{
  "minimax_music_train_component": "language_model",
  "minimax_music_lm_max_frames": 0,
  "minimax_music_lm_window_mode": "prefix"
}
```

要件と DiT トレーニングとの違い:

- 各データセットサンプルは `prompt`（または `tags`）、`lyrics`、および `[frames, codebooks]` 形状の生のコードブック別 RVQ コードの `.pt` ファイルを指す `audio_tokens_path` メタデータを提供する必要があります（セマンティックコード `< 16384`、残差コード `< audio_vocab_size`、語彙オフセットなし）。専用の `minimax-music3-latent-replanner` リポジトリにある `precompute_rvq_codes.py --raw-codes` でエクスポートしてください。
- 損失はセマンティックコードブックの次トークン交差エントロピーで、オーディオ位置にマスクされます。RVQ depth decoder は凍結されたまま、残差コード入力エンベディングを供給します。
- 標準 PEFT LoRA のみ対応で、`lora_format: "comfyui"` は拒否されます。チェックポイントは `language_model.` プレフィックス付きキーで `pytorch_lora_weights.safetensors` を保存します。
- このモードではトレーナー内の検証オーディオは無効です。保存されたチェックポイントから標準生成スタックでレンダリングしてください。
- このモードでは VAE やテキストエンベッドのキャッシュは行われません — トレーニングはトークンを直接読み取るため、`cache_dir_vae` やテキストエンベッドバックエンドは使用されません。
- トリガーキーワード（例: `"fiona crapple"`）を各サンプルの caption/`prompt` フィールドに入れ、歌詞はそのまま保持してください。
- 短いフレーム上限での実行では、常にイントロだけを学習しないように `minimax_music_lm_window_mode: "random"` を設定して、位置付き RVQ 窓をサンプリングできます。ランダム窓は開始/終了/長さをプロンプトへ追加し、サンプルが `lyrics_window` を持つ場合を除いてフルトラック歌詞を省きます。
- 曲構成のトレーニングには `minimax_music_lm_window_mode: "continuation"` を使います。末尾の `minimax_music_lm_target_frames` に損失を適用し、それ以前の可視フレームはマスクされた因果コンテキストになります。`full` は常に曲頭から始まり、`random` は最低 1 個のネイティブ 128 フレームコンテキストを保ちながら曲中を移動できます。時間はネイティブ 128 フレーム/5.12 秒間隔に揃えられ、最大値 `0` は利用可能な曲長を使います。

メモリ上限付き full-prefix continuation の例:

```json
{
  "minimax_music_lm_window_mode": "continuation",
  "minimax_music_lm_target_frames": 128,
  "minimax_music_lm_continuation_crop_mode": "full",
  "minimax_music_lm_min_duration_seconds": 5.12,
  "minimax_music_lm_max_duration_seconds": 30.72
}
```

同じ上限で位置付き continuation を学習するには crop mode を `random` に変更します。位置付き区間は時間範囲をプロンプトへ追加し、`lyrics_window` がなければフルトラック歌詞を省きます。終端区間と非終端区間の両方が可能な場合、固定 25% のサンプルが実際の曲末へ達するため、EOS 教師信号は曲長に依存しません。サンプリングは完全なキャッシュ済み RVQ シーケンスに対する LM collate 時に行われ、データセットの音声やキャッシュは変更しません。
- **プライア保存**: `is_regularisation_data: true` を付けた無関係な楽曲の第二のオーディオバックエンドを追加します（空の歌詞も許可）。それらのバッチでは、損失は正解コードではなく凍結されたベースモデル自身の次トークン分布を対象とするため、LoRA は外科的に保たれます。無関係なキャプションはベースモデルと全く同じように予測し続け、スタイルの漏れが大幅に減ります。

## トラブルシューティング

- **`VAE caching requires the original dav.pth checkpoint`**: `SimpleTuner/MiniMax-Music-3-Encoder` または `MiniMaxAI/MiniMax-Music3` を使うか、ローカル checkpoint root に `dav.pth` を置くか、`pretrained_vae_model_name_or_path` をそれを含む場所に向けます。
- **歌詞が使われない**: backend metadata に `lyrics` があることを確認するか、`caption_strategy: "textfile"` の場合は音声の横に `.lyrics` sidecar を置きます。
- **Text embedding または validation OOM**: validation duration を短くし、int8 text encoder precision または text encoder offload を使います。

## 関連する MiniMax Music 3 実験

- [Open RVQ encoder](https://huggingface.co/SimpleTuner/open-rvq-encoder-minimax-music3)
- [RVQ 参照音声統合](https://github.com/bghira/minimax-music3-rvq-reference-audio)
- [Fiona Crapple LM LoRA](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple)
- [Latent refiner](https://github.com/bghira/minimax-music3-latent-refiner) と [v0.10 weights](https://huggingface.co/terminusresearch/minimax-music3-latent-refiner-v0.10)
- [Latent replanner](https://github.com/bghira/minimax-music3-latent-replanner) と [experiment log](https://huggingface.co/terminusresearch/minimax-music3-replanner-experiment)
