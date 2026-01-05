# Kandinsky 5.0 Image クイックスタート

この例では、Kandinsky 5.0 Image の LoRA をトレーニングします。

## ハードウェア要件

Kandinsky 5.0 は、標準の CLIP エンコーダと Flux VAE に加え、**巨大な 7B パラメータの Qwen2.5-VL テキストエンコーダ**を使用します。これにより VRAM とシステム RAM の両方に大きな負荷がかかります。

Qwen エンコーダを読み込むだけで単体で約 **14GB** のメモリが必要です。フルの gradient checkpointing で rank-16 の LoRA をトレーニングする場合:

- **24GB VRAM** が快適な最低ライン (RTX 3090/4090)。
- **16GB VRAM** も可能ですが、積極的なオフロードとベースモデルの `int8` 量子化が必要になる可能性が高いです。

必要な構成:

- **システム RAM**: 初期ロードで落ちないよう最低 32GB、理想は 64GB。
- **GPU**: NVIDIA RTX 3090 / 4090 または業務用カード (A6000, A100 など)。

### メモリオフロード（推奨）

テキストエンコーダが巨大なので、コンシューマー向けハードウェアではほぼ確実にグループオフロードを使うべきです。計算されていないトランスフォーマーブロックを CPU メモリへオフロードします。

`config.json` に以下を追加してください:

<details>
<summary>設定例を表示</summary>

```json
{
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "group_offload_blocks_per_group": 1,
  "group_offload_use_stream": true
}
```
</details>

- `--group_offload_use_stream`: CUDA デバイスでのみ動作します。
- **`--enable_model_cpu_offload` とは併用しないでください。**

さらに、初期化とキャッシュ作成フェーズの VRAM 使用量を減らすために `config.json` で `"offload_during_startup": true` を設定してください。これによりテキストエンコーダと VAE が同時に読み込まれません。

## 前提条件

Python がインストールされていることを確認してください。SimpleTuner は 3.10 から 3.12 でうまく動作します。

以下を実行して確認できます:

```bash
python --version
```

Ubuntu に Python 3.12 がインストールされていない場合は、以下を試してください:

```bash
apt -y install python3.12 python3.12-venv
```

## インストール

pip で SimpleTuner をインストールします:

```bash
pip install simpletuner[cuda]
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](../INSTALL.md)を参照してください。

## 環境のセットアップ

### Web インターフェース方式

SimpleTuner WebUI を使うと設定が簡単です。サーバーの起動:

```bash
simpletuner server
```

http://localhost:8001 にアクセスします。

### 手動 / コマンドライン方式

コマンドラインで SimpleTuner を実行するには、設定ファイル、データセットとモデルのディレクトリ、データローダー設定ファイルを用意する必要があります。

#### 設定ファイル

実験的なスクリプト `configure.py` でこのセクションをスキップできる場合があります:

```bash
simpletuner configure
```

手動で設定する場合:

`config/config.json.example` を `config/config.json` にコピー:

```bash
cp config/config.json.example config/config.json
```

以下の変数を変更する必要があります:

- `model_type`: `lora`
- `model_family`: `kandinsky5-image`
- `model_flavour`:
  - `t2i-lite-sft`: (デフォルト) 標準の SFT チェックポイント。スタイル/キャラクターのファインチューニングに最適。
  - `t2i-lite-pretrain`: プリトレイン チェックポイント。新しい概念をゼロから教えるのに適しています。
  - `i2i-lite-sft` / `i2i-lite-pretrain`: 画像から画像の学習用。データセットにコンディショニング画像が必要です。
- `output_dir`: チェックポイントの保存先。
- `train_batch_size`: まずは `1` から。
- `gradient_accumulation_steps`: `1` 以上にして疑似的に大きなバッチを作ります。
- `validation_resolution`: このモデルの標準は `1024x1024`。
- `validation_guidance`: Kandinsky 5 の推奨デフォルトは `5.0`。
- `flow_schedule_shift`: 既定値は `1.0`。調整すると、ディテールと構図の優先度が変わります (後述)。

#### 検証プロンプト

`config/config.json` に「主」検証プロンプトがあります。`config/user_prompt_library.json` にプロンプトのライブラリを作成することもできます:

<details>
<summary>設定例を表示</summary>

```json
{
  "portrait": "A high quality portrait of a woman, cinematic lighting, 8k",
  "landscape": "A beautiful mountain landscape at sunset, oil painting style"
}
```
</details>

`config.json` に次を追加して有効化します:

<details>
<summary>設定例を表示</summary>

```json
{
  "user_prompt_library": "config/user_prompt_library.json"
}
```
</details>

#### Flow schedule shifting

Kandinsky 5 はフローマッチングモデルです。`shift` パラメータはトレーニング/推論時のノイズ分布を制御します。

- **Shift 1.0 (デフォルト)**: バランスの取れた学習。
- **低い Shift (< 1.0)**: 高周波のディテール (テクスチャ、ノイズ) を重視。
- **高い Shift (> 1.0)**: 低周波のディテール (構図、色、構造) を重視。

スタイルは学ぶが構図が崩れる場合は shift を上げ、構図は学ぶが質感が弱い場合は下げてみてください。

#### 量子化モデルトレーニング

トランスフォーマーを 8-bit に量子化して VRAM 使用量を大幅に削減できます。

`config.json` では:

<details>
<summary>設定例を表示</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "base_model_default_dtype": "bf16"
```
</details>

> **注意**: Qwen2.5-VL は量子化に敏感で、すでに最も重い部分でもあるため、テキストエンコーダ (`no_change`) の量子化は推奨しません。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTuner には、トレーニングの安定性とパフォーマンスを大幅に向上させることができる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが独自の入力を生成できるようにすることで、露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

#### データセットの考慮事項

データセット設定ファイルが必要です。例: `config/multidatabackend.json`。

```json
[
  {
    "id": "my-image-dataset",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/my_images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "crop": true,
    "crop_aspect": "square",
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/kandinsky5",
    "disabled": false
  }
]
```

> caption_strategy のオプションと要件は [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

次にデータセットディレクトリを作成します:

```bash
mkdir -p datasets/my_images
</details>

# ここに画像と .txt キャプションファイルを置いてください
```

#### WandB と Huggingface Hub へのログイン

```bash
wandb login
huggingface-cli login
```

### トレーニングの実行

**オプション 1 (推奨):**

```bash
simpletuner train
```

**オプション 2 (レガシー):**

```bash
./train.sh
```

## メモとトラブルシューティングのヒント

### 最低 VRAM 構成

16GB または制約のある 24GB 環境で動かす場合:

1.  **Group Offload を有効化**: `--enable_group_offload`。
2.  **ベースモデルを量子化**: `"base_model_precision": "int8-quanto"` を設定。
3.  **バッチサイズ**: `1` を維持。

### アーティファクトと「焼けた」画像

検証画像が過飽和やノイズだらけ ("焼けた" 状態) に見える場合:

- **ガイダンスを確認**: `validation_guidance` が 5.0 付近になっているか確認。7.0+ の高い値はこのモデルでは画像を破綻させがちです。
- **Flow Shift を確認**: `flow_schedule_shift` の極端な値は不安定さを招きます。まずは 1.0 から。
- **学習率**: LoRA の標準は 1e-4 ですが、アーティファクトが出る場合は 5e-5 に下げてください。

### TREAD トレーニング

Kandinsky 5 はトークンを落として高速化する [TREAD](../TREAD.md) をサポートします。

`config.json` に追加:

<details>
<summary>設定例を表示</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

これにより、中間層のトークンを 50% 落とし、トランスフォーマー処理が高速化されます。
