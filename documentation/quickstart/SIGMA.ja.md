## PixArt Sigma クイックスタート

この例では、SimpleTuner ツールキットを使用して PixArt Sigma モデルをトレーニングし、サイズが小さく VRAM に収まりやすいため `full` モデルタイプを使用します。

### 前提条件

Python がインストールされていることを確認してください。SimpleTuner は 3.10 から 3.12 でうまく動作します。

以下を実行して確認できます:

```bash
python --version
```

Ubuntu に Python 3.12 がインストールされていない場合は、以下を試してください:

```bash
apt -y install python3.12 python3.12-venv
```

#### コンテナイメージの依存関係

Vast、RunPod、TensorDock（など）の場合、CUDA 12.2-12.8 イメージで CUDA 拡張のコンパイルを有効にするには以下が機能します:

```bash
apt -y install nvidia-cuda-toolkit
```

### インストール

pip で SimpleTuner をインストールします:

```bash
pip install simpletuner[cuda]
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](../INSTALL.md)を参照してください。

#### AMD ROCm フォローアップ手順

AMD MI300X を使用可能にするには、以下を実行する必要があります:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

### 環境のセットアップ

SimpleTuner を実行するには、設定ファイル、データセットとモデルのディレクトリ、およびデータローダー設定ファイルをセットアップする必要があります。

#### 設定ファイル

実験的なスクリプト `configure.py` を使用すると、インタラクティブなステップバイステップの設定でこのセクションを完全にスキップできる可能性があります。一般的な落とし穴を避けるための安全機能が含まれています。

**注意:** これはデータローダーを設定しません。後で手動で設定する必要があります。

実行するには:

```bash
simpletuner configure
```
> ⚠️ Hugging Face Hub にアクセスしにくい国にいるユーザーは、システムが使用する `$SHELL` に応じて `~/.bashrc` または `~/.zshrc` に `HF_ENDPOINT=https://hf-mirror.com` を追加してください。

手動で設定したい場合:

`config/config.json.example` を `config/config.json` にコピーします:

```bash
cp config/config.json.example config/config.json
```

そこで、以下の変数を変更する必要があります:

<details>
<summary>設定例を表示</summary>

```json
{
  "model_type": "full",
  "use_bitfit": false,
  "pretrained_model_name_or_path": "pixart-alpha/pixart-sigma-xl-2-1024-ms",
  "model_family": "pixart_sigma",
  "output_dir": "/home/user/output/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.5
}
```
</details>

- `pretrained_model_name_or_path` - `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` に設定します。
- `MODEL_TYPE` - `full` に設定します。
- `USE_BITFIT` - `false` に設定します。
- `MODEL_FAMILY` - `pixart_sigma` に設定します。
- `OUTPUT_DIR` - チェックポイントと検証画像を保存するディレクトリに設定します。フルパスの使用を推奨します。
- `VALIDATION_RESOLUTION` - PixArt Sigma は 1024px または 2048px のモデル形式なので、この例では `1024x1024` に慎重に設定してください。
  - PixArt はマルチアスペクトバケットでファインチューニングされているため、カンマ区切りで他の解像度を指定できます: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - PixArt は非常に低い値が有効です。`3.6`〜`4.4` の範囲に設定してください。

Mac M-series を使用している場合の追加設定:

- `mixed_precision` を `no` に設定します。

> 💡 **ヒント:** ディスク容量が問題になる大規模データセットの場合、`--vae_cache_disable` を使ってオンライン VAE エンコードを行い、結果をディスクにキャッシュしないようにできます。

#### データセットの考慮事項

モデルをトレーニングするには十分なデータセットが不可欠です。データセットサイズには制限があり、モデルを効果的にトレーニングできる十分な大きさのデータセットであることを確認する必要があります。最小限のデータセットサイズは `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS` であり、データセットが小さすぎるとトレーナーに検出されません。

手元のデータセットに応じて、データセットディレクトリとデータローダー設定ファイルを異なる方法でセットアップする必要があります。この例では [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) を使用します。

`/home/user/simpletuner/config` ディレクトリに multidatabackend.json を作成します:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "pseudo-camera-10k-pixart",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 1.0,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/pixart/pseudo-camera-10k",
    "instance_data_dir": "/home/user/simpletuner/datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/pixart/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> caption_strategy のオプションと要件については [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

次に `datasets` ディレクトリを作成します:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
popd
```

これにより、約 10k の写真サンプルが `datasets/pseudo-camera-10k` ディレクトリにダウンロードされ、自動的に作成されます。

#### WandB と Huggingface Hub へのログイン

特に `push_to_hub: true` と `--report_to=wandb` を使う場合は、トレーニング開始前に WandB と HF Hub にログインしておく必要があります。

Git LFS リポジトリに手動でアイテムをプッシュする場合は、`git config --global credential.helper store` も実行してください。

以下のコマンドを実行します:

```bash
wandb login
```

および

```bash
huggingface-cli login
```

指示に従って両方のサービスにログインしてください。

### トレーニングの実行

SimpleTuner ディレクトリから、以下を実行するだけです:

```bash
bash train.sh
```

これにより、テキスト埋め込みと VAE 出力のディスクへのキャッシングが開始されます。

詳細については、[データローダー](../DATALOADER.md) と [チュートリアル](../TUTORIAL.md) のドキュメントを参照してください。

### CLIP スコアトラッキング

モデルのパフォーマンスをスコアリングするための評価を有効にしたい場合は、CLIP スコアの設定と解釈に関する情報について [このドキュメント](../evaluation/CLIP_SCORES.md) を参照してください。

# 安定した評価損失

モデルのパフォーマンスをスコアリングするために安定した MSE 損失を使用したい場合は、評価損失の設定と解釈に関する情報について [このドキュメント](../evaluation/EVAL_LOSS.md) を参照してください。

#### 検証プレビュー

SimpleTuner は Tiny AutoEncoder モデルを使用して生成中の中間検証プレビューのストリーミングをサポートしています。これにより、webhook コールバックを介してリアルタイムで検証画像が生成されるのを段階的に確認できます。

有効にするには:
<details>
<summary>設定例を表示</summary>

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```
</details>

**要件:**
- Webhook 設定
- 検証が有効

Tiny AutoEncoder のオーバーヘッドを削減するには、`validation_preview_steps` をより高い値（例: 3 または 5）に設定してください。`validation_num_inference_steps=20` と `validation_preview_steps=5` の場合、ステップ 5、10、15、20 でプレビュー画像を受け取ります。

### SageAttention

`--attention_mechanism=sageattention` を使用すると、検証時の推論を高速化できます。

**注**: これは _すべて_ のモデル構成と互換ではありませんが、試す価値があります。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTuner には、トレーニングの安定性とパフォーマンスを大幅に向上させる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが自身の入力を生成することで露出バイアスを減らし、出力品質を向上させます。
*   **[Diff2Flow](../experimental/DIFF2FLOW.md):** Flow Matching 目的での学習を可能にし、生成の直線性と品質を改善する可能性があります。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。
</details>
