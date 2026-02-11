## Stable Diffusion XL クイックスタート

この例では、SimpleTunerツールキットを使用してStable Diffusion XLモデルをトレーニングし、`lora`モデルタイプを使用します。

現代の大規模モデルと比較して、SDXLはかなり控えめなサイズなので、`full`トレーニングを使用することも可能かもしれませんが、LoRAトレーニングと比較してより多くのVRAMと他のハイパーパラメータの調整が必要になります。

### 前提条件

Pythonがインストールされていることを確認してください。SimpleTunerは3.10から3.12で正常に動作します（AMD ROCmマシンでは3.12が必要です）。

以下のコマンドを実行して確認できます:

```bash
python --version
```

Ubuntuでpython 3.12がインストールされていない場合は、以下を試してください:

```bash
apt -y install python3.13 python3.13-venv
```

#### コンテナイメージの依存関係

Vast、RunPod、TensorDock（など）の場合、CUDA 12.2-12.8イメージでCUDA拡張のコンパイルを有効にするには以下が機能します:

```bash
apt -y install nvidia-cuda-toolkit
```

### インストール

SimpleTunerをpip経由でインストール:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](../INSTALL.md)を参照してください。

### 環境のセットアップ

SimpleTunerを実行するには、設定ファイル、データセットとモデルのディレクトリ、およびデータローダー設定ファイルをセットアップする必要があります。

#### 設定ファイル

実験的なスクリプト`configure.py`を使用すると、インタラクティブなステップバイステップの設定を通じて、このセクションを完全にスキップできる可能性があります。これには、一般的な落とし穴を回避するのに役立つセーフティ機能が含まれています。

**注意:** これはデータローダーを**完全には**設定しません。後で手動で設定する必要があります。

実行するには:

```bash
simpletuner configure
```
> ⚠️ Hugging Face Hubに容易にアクセスできない国にいるユーザーは、使用している`$SHELL`に応じて、`~/.bashrc`または`~/.zshrc`に`HF_ENDPOINT=https://hf-mirror.com`を追加する必要があります。

手動で設定したい場合:

`config/config.json.example`を`config/config.json`にコピー:

```bash
cp config/config.json.example config/config.json
```

#### AMD ROCmフォローアップ手順

AMD MI300Xを使用可能にするには、以下を実行する必要があります:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

そこで、以下の変数を変更する必要があります:

<details>
<summary>設定例を表示</summary>

```json
{
  "model_type": "lora",
  "model_family": "sdxl",
  "model_flavour": "base-1.0",
  "output_dir": "/home/user/output/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.4,
  "use_gradient_checkpointing": true,
  "learning_rate": 1e-4
}
```
</details>

- `model_family` - `sdxl`に設定します。
- `model_flavour` - `base-1.0`に設定するか、`pretrained_model_name_or_path`を使用して別のモデルを指定します。
- `model_type` - `lora`に設定します。
- `use_dora` - DoRAをトレーニングしたい場合は`true`に設定します。
- `output_dir` - チェックポイントと検証画像を保存するディレクトリに設定します。ここではフルパスを使用することをお勧めします。
- `validation_resolution` - この例では`1024x1024`に設定します。
  - さらに、Stable Diffusion XLはマルチアスペクトバケットでファインチューニングされており、カンマで区切って他の解像度を指定できます: `1024x1024,1280x768`
- `validation_guidance` - 推論時にテストするための快適な値を使用します。`4.2`から`6.4`の間に設定します。
- `use_gradient_checkpointing` - 大量のVRAMがあり、速度を上げるために一部を犠牲にしたい場合を除き、おそらく`true`にする必要があります。
- `learning_rate` - `1e-4`は低ランクネットワークでかなり一般的ですが、「バーニング」や早期の過学習に気づいた場合は`1e-5`がより保守的な選択かもしれません。

Mac M-seriesマシンを使用している場合は、さらにいくつかあります:

- `mixed_precision`は`no`に設定する必要があります。
  - これはpytorch 2.4では当てはまりましたが、2.6以降ではbf16を使用できるかもしれません
- `attention_mechanism`は`xformers`に設定できますが、やや時代遅れになっています。

#### 量子化モデルトレーニング

AppleおよびNVIDIAシステムでテスト済み、Hugging Face Optimum-QuantoはUnetの精度とVRAM要件を削減するために使用できますが、SD3/Fluxのような拡散トランスフォーマーモデルほどうまく機能しないため、推奨されません。

ただし、リソースに制約がある場合は、使用できます。

`config.json`の場合:
<details>
<summary>設定例を表示</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "optimizer": "optimi-lion"
}
```
</details>

#### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTunerには、トレーニングの安定性とパフォーマンスを大幅に向上させることができる実験的機能が含まれています。特に小規模なデータセットやSDXLのような古いアーキテクチャに有効です。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが独自の入力を生成できるようにすることで、露出バイアスを減らし、出力品質を向上させます。
*   **[Diff2Flow](../experimental/DIFF2FLOW.md):** フローマッチング目的でSDXLをトレーニングし、生成の直線性と品質を向上させる可能性があります。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

</details>

#### データセットの考慮事項

モデルをトレーニングするには、かなりのデータセットが必要です。データセットサイズには制限があり、モデルを効果的にトレーニングするためにデータセットが十分に大きいことを確認する必要があります。最小限のデータセットサイズは`TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`であることに注意してください。データセットが小さすぎると、トレーナーに発見されません。

> 💡 **ヒント:** ディスク容量が問題になる大規模データセットの場合、`--vae_cache_disable`を使用して、結果をディスクにキャッシュせずにオンラインVAEエンコーディングを実行できます。これは`--vae_cache_ondemand`を使用する場合に暗黙的に有効になりますが、`--vae_cache_disable`を追加すると、ディスクに何も書き込まれないことが保証されます。

持っているデータセットに応じて、データセットディレクトリとデータローダー設定ファイルを異なる方法でセットアップする必要があります。この例では、データセットとして[pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k)を使用します。

`OUTPUT_DIR`ディレクトリに、multidatabackend.jsonを作成します:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sdxl",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 1.0,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/sdxl/pseudo-camera-10k",
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
    "cache_dir": "cache/text/sdxl/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> caption_strategyオプションと要件については、[DATALOADER.md](../DATALOADER.md#caption_strategy)を参照してください。

次に、`datasets`ディレクトリを作成します:

```bash
mkdir -p datasets
huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=datasets/pseudo-camera-10k
```

これにより、約10kの写真サンプルが`datasets/pseudo-camera-10k`ディレクトリにダウンロードされ、自動的に作成されます。

#### WandBとHuggingface Hubにログイン

特に`push_to_hub: true`と`--report_to=wandb`を使用している場合は、トレーニングを開始する前にWandBとHF Hubにログインする必要があります。

Git LFSリポジトリに手動でアイテムをプッシュする場合は、`git config --global credential.helper store`も実行する必要があります

以下のコマンドを実行します:

```bash
wandb login
```

および

```bash
huggingface-cli login
```

指示に従って両方のサービスにログインします。

### トレーニング実行の実行

SimpleTunerディレクトリから、以下を実行するだけです:

```bash
bash train.sh
```

これにより、テキスト埋め込みとVAE出力のディスクへのキャッシングが開始されます。

詳細については、[データローダー](../DATALOADER.md)および[チュートリアル](../TUTORIAL.md)ドキュメントを参照してください。

### CLIPスコアトラッキング

モデルのパフォーマンスをスコアリングするための評価を有効にしたい場合は、CLIPスコアの設定と解釈に関する情報について[このドキュメント](../evaluation/CLIP_SCORES.md)を参照してください。

# 安定した評価損失

モデルのパフォーマンスをスコアリングするために安定したMSE損失を使用したい場合は、評価損失の設定と解釈に関する情報について[このドキュメント](../evaluation/EVAL_LOSS.md)を参照してください。

#### 検証プレビュー

SimpleTunerは、Tiny AutoEncoderモデルを使用した生成中の中間検証プレビューのストリーミングをサポートしています。これにより、Webhookコールバックを介してリアルタイムで段階的に生成される検証画像を確認できます。

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
- Webhook設定
- 検証が有効

Tiny AutoEncoderのオーバーヘッドを削減するには、`validation_preview_steps`をより高い値（3または5など）に設定します。`validation_num_inference_steps=20`および`validation_preview_steps=5`の場合、ステップ5、10、15、20でプレビュー画像を受け取ります。
