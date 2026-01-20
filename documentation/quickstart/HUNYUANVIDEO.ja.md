# Hunyuan Video 1.5 クイックスタート

このガイドでは、SimpleTunerを使用してTencentの8.3B **Hunyuan Video 1.5**リリース(`tencent/HunyuanVideo-1.5`)でLoRAをトレーニングする方法を説明します。

## ハードウェア要件

Hunyuan Video 1.5は大規模モデル(8.3Bパラメータ)です。

- **最小**: 480pでフル勾配チェックポイントを使用したRank-16 LoRAには**24GB-32GB VRAM**が快適です。
- **推奨**: 720pトレーニングやより大きなバッチサイズにはA6000 / A100(48GB-80GB)。
- **システムRAM**: モデルロードを処理するために**64GB以上**を推奨。

### メモリオフロード(オプション)

`config.json`に以下を追加:

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

- `--group_offload_use_stream`: CUDAデバイスでのみ機能します。
- `--enable_model_cpu_offload`と**組み合わせないでください**。

## 前提条件

Pythonがインストールされていることを確認してください。SimpleTunerは3.10から3.12で動作します。

以下のコマンドで確認できます:

```bash
python --version
```

Ubuntuにpython 3.12がインストールされていない場合は、以下を試してください:

```bash
apt -y install python3.13 python3.13-venv
```

### コンテナイメージの依存関係

Vast、RunPod、TensorDock(など)の場合、CUDA 12.2-12.8イメージでCUDA拡張のコンパイルを有効にするには以下が機能します:

```bash
apt -y install nvidia-cuda-toolkit
```

### AMD ROCmフォローアップ手順

AMD MI300Xを使用可能にするには、以下を実行する必要があります:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## インストール

pip経由でSimpleTunerをインストール:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](../INSTALL.md)を参照してください。

### 必要なチェックポイント

メインの`tencent/HunyuanVideo-1.5`リポジトリにはtransformer/vae/schedulerが含まれていますが、**テキストエンコーダー**(`text_encoder/llm`)と**ビジョンエンコーダー**(`vision_encoder/siglip`)は別のダウンロードにあります。起動前にSimpleTunerをローカルコピーに向けてください:

```bash
export HUNYUANVIDEO_TEXT_ENCODER_PATH=/path/to/text_encoder_root
export HUNYUANVIDEO_VISION_ENCODER_PATH=/path/to/vision_encoder_root
```

これらが設定されていない場合、SimpleTunerはモデルリポジトリからプルしようとします。ほとんどのミラーはそれらをバンドルしていないため、起動エラーを回避するために明示的にパスを設定してください。

## 環境のセットアップ

### Webインターフェース方式

SimpleTuner WebUIにより、セットアップがかなり簡単になります。サーバーを実行するには:

```bash
simpletuner server
```

これにより、デフォルトでポート8001にWebサーバーが作成され、http://localhost:8001にアクセスすることで利用できます。

### 手動/コマンドライン方式

SimpleTunerをコマンドラインツール経由で実行するには、設定ファイル、データセットとモデルのディレクトリ、およびデータローダー設定ファイルをセットアップする必要があります。

#### 設定ファイル

実験的なスクリプト`configure.py`により、インタラクティブなステップバイステップの設定を通じて、このセクション全体をスキップできる可能性があります。

**注意:** これはデータローダーを設定しません。後で手動で設定する必要があります。

実行するには:

```bash
simpletuner configure
```

手動で設定する場合:

`config/config.json.example`を`config/config.json`にコピー:

```bash
cp config/config.json.example config/config.json
```

HunyuanVideoの主要な設定オーバーライド:

<details>
<summary>設定例を表示</summary>

```json
{
  "model_type": "lora",
  "model_family": "hunyuanvideo",
  "pretrained_model_name_or_path": "tencent/HunyuanVideo-1.5",
  "model_flavour": "t2v-480p",
  "output_dir": "output/hunyuan-video",
  "validation_resolution": "854x480",
  "validation_num_video_frames": 61,
  "validation_guidance": 6.0,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "learning_rate": 1e-4,
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "lora_rank": 16,
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "dataset_backend_config": "config/multidatabackend.json"
}
```
</details>

- `model_flavour`オプション:
  - `t2v-480p`(デフォルト)
  - `t2v-720p`
  - `i2v-480p`(Image-to-Video)
  - `i2v-720p`(Image-to-Video)
- `validation_num_video_frames`: `(frames - 1) % 4 == 0`である必要があります。例: 61、129。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTunerには、トレーニングの安定性とパフォーマンスを大幅に向上させることができる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが独自の入力を生成できるようにすることで、露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

#### データセットの考慮事項

以下を含む`--data_backend_config`(`config/multidatabackend.json`)ドキュメントを作成します:

```json
[
  {
    "id": "my-video-dataset",
    "type": "local",
    "dataset_type": "video",
    "instance_data_dir": "datasets/videos",
    "caption_strategy": "textfile",
    "resolution": 480,
    "video": {
        "num_frames": 61,
        "min_frames": 61,
        "frame_rate": 24,
        "bucket_strategy": "aspect_ratio"
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/hunyuan",
    "disabled": false
  }
]
```

`video`サブセクションでは:
- `num_frames`: トレーニングのターゲットフレーム数。`(frames - 1) % 4 == 0`を満たす必要があります。
- `min_frames`: 最小動画長(短い動画は破棄されます)。
- `max_frames`: 最大動画長フィルター。
- `bucket_strategy`: 動画がバケットにグループ化される方法:
  - `aspect_ratio`(デフォルト): 空間的アスペクト比のみでグループ化。
  - `resolution_frames`: 混合解像度/期間データセット用に`WxH@F`形式(例: `854x480@61`)でグループ化。
- `frame_interval`: `resolution_frames`を使用する場合、フレーム数をこの間隔に丸めます。

> caption_strategyのオプションと要件については、[DATALOADER.md](../DATALOADER.md#caption_strategy)を参照してください。

- **テキストエンベッドキャッシング**: 強く推奨。Hunyuanは大規模なLLMテキストエンコーダーを使用します。キャッシングによりトレーニング中のVRAMを大幅に節約できます。

#### WandBとHuggingface Hubへのログイン

```bash
wandb login
huggingface-cli login
```

</details>

### トレーニング実行の実行

SimpleTunerディレクトリから:

```bash
simpletuner train
```

## 注意事項とトラブルシューティングのヒント

### VRAM最適化

- **グループオフロード**: コンシューマーGPUには必須。`enable_group_offload`がtrueであることを確認してください。
- **解像度**: VRAMが限られている場合は480p(`854x480`または類似)に固定してください。720p(`1280x720`)はメモリ使用量が大幅に増加します。
- **量子化**: `base_model_precision`を使用(`bf16`がデフォルト)。速度を犠牲にしてさらに節約するには`int8-torchao`が機能します。
- **VAEパッチコンボリューション**: HunyuanVideo VAE OOMの場合、`--vae_enable_patch_conv=true`を設定(またはUIでトグル)。これにより3D conv/attentionワークがスライスされ、ピークVRAMが低下します。若干のスループット低下を予想してください。

### Image-to-Video(I2V)

- `model_flavour="i2v-480p"`を使用。
- SimpleTunerは動画データセットサンプルの最初のフレームをコンディショニング画像として自動的に使用します。
- 検証セットアップがコンディショニング入力を含むか、自動抽出された最初のフレームに依存していることを確認してください。

### テキストエンコーダー

Hunyuanはデュアルテキストエンコーダーセットアップ(LLM + CLIP)を使用します。システムRAMがキャッシングフェーズ中にこれらのロードを処理できることを確認してください。
