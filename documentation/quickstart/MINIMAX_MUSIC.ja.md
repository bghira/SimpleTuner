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

## トラブルシューティング

- **`VAE caching requires the original dav.pth checkpoint`**: `SimpleTuner/MiniMax-Music-3-Encoder` または `MiniMaxAI/MiniMax-Music3` を使うか、ローカル checkpoint root に `dav.pth` を置くか、`pretrained_vae_model_name_or_path` をそれを含む場所に向けます。
- **歌詞が使われない**: backend metadata に `lyrics` があることを確認するか、`caption_strategy: "textfile"` の場合は音声の横に `.lyrics` sidecar を置きます。
- **Text embedding または validation OOM**: validation duration を短くし、int8 text encoder precision または text encoder offload を使います。
