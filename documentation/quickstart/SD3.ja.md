## Stable Diffusion 3

この例では、SimpleTunerツールキットを使用してStable Diffusion 3モデルをトレーニングし、`lora`モデルタイプを使用します。

### 前提条件

Pythonがインストールされていることを確認してください。SimpleTunerは3.10から3.12で正常に動作します。

以下のコマンドを実行して確認できます:

```bash
python --version
```

Ubuntuでpython 3.12がインストールされていない場合は、以下を試してください:

```bash
apt -y install python3.12 python3.12-venv
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
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](../INSTALL.md)を参照してください。

#### AMD ROCmフォローアップ手順

AMD MI300Xを使用可能にするには、以下を実行する必要があります:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

### 環境のセットアップ

SimpleTunerを実行するには、設定ファイル、データセットとモデルのディレクトリ、およびデータローダー設定ファイルをセットアップする必要があります。

#### 設定ファイル

実験的なスクリプト`configure.py`を使用すると、インタラクティブなステップバイステップの設定を通じて、このセクションを完全にスキップできる可能性があります。これには、一般的な落とし穴を回避するのに役立つセーフティ機能が含まれています。

**注意:** これはデータローダーを設定しません。後で手動で設定する必要があります。

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

そこで、以下の変数を変更する必要があります:

<details>
<summary>設定例を表示</summary>

```json
{
  "model_type": "lora",
  "model_family": "sd3",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
  "output_dir": "/home/user/outputs/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.0,
  "validation_prompt": "your main test prompt here",
  "user_prompt_library": "config/user_prompt_library.json"
}
```
</details>


- `pretrained_model_name_or_path` - `stabilityai/stable-diffusion-3.5-large`に設定します。このモデルをダウンロードするには、Huggingfaceにログインしてアクセス権を付与される必要があることに注意してください。Huggingfaceへのログインについては、このチュートリアルの後半で説明します。
  - 古いSD3.0 Medium（2B）をトレーニングしたい場合は、代わりに`stabilityai/stable-diffusion-3-medium-diffusers`を使用してください。
- `MODEL_TYPE` - `lora`に設定します。
- `MODEL_FAMILY` - `sd3`に設定します。
- `OUTPUT_DIR` - チェックポイントと検証画像を保存するディレクトリに設定します。ここではフルパスを使用することをお勧めします。
- `VALIDATION_RESOLUTION` - SD3は1024pxモデルなので、`1024x1024`に設定できます。
  - さらに、SD3はマルチアスペクトバケットでファインチューニングされており、カンマで区切って他の解像度を指定できます: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - SD3は非常に低い値から恩恵を受けます。`3.0`に設定します。

Mac M-seriesマシンを使用している場合は、さらにいくつかあります:

- `mixed_precision`は`no`に設定する必要があります。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTunerには、トレーニングの安定性とパフォーマンスを大幅に向上させることができる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが独自の入力を生成できるようにすることで、露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

#### 量子化モデルトレーニング

AppleおよびNVIDIAシステムでテスト済み、Hugging Face Optimum-Quantoを使用して精度とVRAM要件を削減し、ベースSDXLトレーニングの要件を大幅に下回ることができます。



> ⚠️ JSON設定ファイルを使用している場合は、`config.env`の代わりに`config.json`でこの形式を使用してください:

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "text_encoder_3_precision": "no_change",
  "optimizer": "adamw_bf16"
}
```

`config.env`ユーザー向け（非推奨）:

```bash
</details>

# 選択肢: int8-quanto, int4-quanto, int2-quanto, fp8-quanto
# int8-quantoは単一サブジェクトのdreambooth LoRAでテストされました。
# fp8-quantoはAppleシステムでは動作しません。intレベルを使用する必要があります。
# int2-quantoはかなり極端で、rank-1 LoRA全体を約13.9GB VRAMに収めます。
# 設定を大きく攻める場合は、安定性に注意してください。
export TRAINER_EXTRA_ARGS="--base_model_precision=int8-quanto"

# テキストエンコーダーをフル精度のままにして、テキスト埋め込みを完璧にしたいかもしれません。
# トレーニング前にテキストエンコーダーをアンロードするので、トレーニング時間中は問題ありません - 事前キャッシング中のみです。
# 必要に応じて、int4またはint8モードで量子化して実行することもできます。
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --text_encoder_1_precision=no_change --text_encoder_2_precision=no_change"

# モデルを量子化する場合、--base_model_default_dtypeはデフォルトでbf16に設定されます。このセットアップにはadamw_bf16が必要ですが、最もメモリを節約します。
# adamw_bf16はbf16トレーニングのみをサポートしますが、他のオプティマイザはbf16またはfp32トレーニング精度の両方をサポートします。
export OPTIMIZER="adamw_bf16"
```

#### データセットの考慮事項

モデルをトレーニングするには、かなりのデータセットが必要です。データセットサイズには制限があり、モデルを効果的にトレーニングするためにデータセットが十分に大きいことを確認する必要があります。最小限のデータセットサイズは`TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`であり、さらに`VAE_BATCH_SIZE`より多い必要があることに注意してください。データセットが小さすぎると使用できません。

持っているデータセットに応じて、データセットディレクトリとデータローダー設定ファイルを異なる方法でセットアップする必要があります。この例では、データセットとして[pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k)を使用します。

`/home/user/simpletuner/config`ディレクトリに、multidatabackend.jsonを作成します:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sd3",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 0,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/home/user/simpletuner/output/cache/vae/sd3/pseudo-camera-10k",
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
    "cache_dir": "cache/text/sd3/pseudo-camera-10k",
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
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
popd
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

## 注意事項とトラブルシューティングのヒント

### Skip-layer guidance（SD3.5 Medium）

StabilityAIはSD 3.5 Medium推論でSLG（Skip-layer guidance）を有効にすることを推奨しています。これはトレーニング結果に影響しませんが、検証サンプルの品質のみに影響します。

`config.json`には以下の値が推奨されます:

<details>
<summary>設定例を表示</summary>

```json
{
  "--validation_guidance_skip_layers": [7, 8, 9],
  "--validation_guidance_skip_layers_start": 0.01,
  "--validation_guidance_skip_layers_stop": 0.2,
  "--validation_guidance_skip_scale": 2.8,
  "--validation_guidance": 4.0,
  "--flow_use_uniform_schedule": true,
  "--flow_schedule_auto_shift": true
}
```
</details>

- `..skip_scale`は、skip-layer guidance中にポジティブプロンプト予測をどれだけスケーリングするかを決定します。デフォルト値の2.8は、ベースモデルの`7, 8, 9`のスキップ値には安全ですが、追加のレイヤーがスキップされる場合は、追加レイヤーごとに倍にする必要があります。
- `..skip_layers`は、ネガティブプロンプト予測中にスキップするレイヤーを指定します。
- `..skip_layers_start`は、skip-layer guidanceの適用を開始する推論パイプラインの割合を決定します。
- `..skip_layers_stop`は、SLGが適用されなくなる総推論ステップ数の割合を設定します。

SLGは、より少ないステップで適用することで、効果を弱めたり、推論速度の低下を抑えたりできます。

LoRAまたはLyCORISモデルの広範なトレーニングでは、これらの値の変更が必要になるようですが、正確にどのように変更されるかは明確ではありません。

**推論時にはより低いCFGを使用する必要があります。**

### モデルの不安定性

SD 3.5 Large 8Bモデルには、トレーニング中の潜在的な不安定性があります:

- 高い`--max_grad_norm`値は、モデルが潜在的に危険な重み更新を探索することを許可します
- 学習率は非常に敏感になる可能性があります。`1e-5`はStableAdamWで動作しますが、`4e-5`は爆発する可能性があります
- より高いバッチサイズは**大いに**役立ちます
- 安定性は量子化の無効化や純粋なfp32でのトレーニングによって影響を受けません

公式トレーニングコードはSD3.5と一緒にリリースされず、開発者は[SD3.5リポジトリの内容](https://github.com/stabilityai/sd3.5)に基づいてトレーニングループを実装する方法を推測することになりました。

SimpleTunerのSD3.5サポートにはいくつかの変更が行われました:
- 量子化からより多くのレイヤーを除外
- デフォルトでT5パディングスペースをゼロにしなくなりました（`--t5_padding`）
- 無条件予測に空のエンコードされた空白キャプション（`empty_string`、**デフォルト**）またはゼロ（`zero`）を使用するスイッチ（`--sd3_clip_uncond_behaviour`と`--sd3_t5_uncond_behaviour`）を提供、調整は推奨されない設定
- SD3.5トレーニング損失関数が上流のStabilityAI/SD3.5リポジトリにあるものと一致するように更新されました
- SD3の静的1024px値と一致するように、デフォルトの`--flow_schedule_shift`値を3に更新しました
  - StabilityAIは、`--flow_use_uniform_schedule`と一緒に`--flow_schedule_shift=1`を使用するドキュメントでフォローアップしました
  - コミュニティメンバーは、マルチアスペクトまたはマルチ解像度トレーニングを使用する場合、`--flow_schedule_auto_shift`がより良く機能すると報告しています
- ハードコードされたトークナイザーシーケンス長制限が**154**に更新され、ディスク容量または計算を節約するために出力品質の低下を犠牲にして**77**トークンに戻すオプションがあります


#### 安定した設定値

これらのオプションは、SD3.5をできるだけ長く無傷に保つことが知られています:
- optimizer=adamw_bf16
- flow_schedule_shift=1
- learning_rate=1e-4
- batch_size=4 * 3 GPUs
- max_grad_norm=0.1
- base_model_precision=int8-quanto
- 損失マスキングやデータセット正則化なし、この不安定性への寄与は不明なため
- `validation_guidance_skip_layers=[7,8,9]`

### 最低VRAM構成

- OS: Ubuntu Linux 24
- GPU: 単一のNVIDIA CUDAデバイス（10G、12G）
- システムメモリ: 約50Gのシステムメモリ
- ベースモデル精度: `nf4-bnb`
- オプティマイザ: Lion 8Bit Paged、`bnb-lion8bit-paged`
- 解像度: 512px
- バッチサイズ: 1、勾配累積ステップなし
- DeepSpeed: 無効/未設定
- PyTorch: 2.5

### SageAttention

`--attention_mechanism=sageattention`を使用すると、検証時の推論を高速化できます。

**注意**: これはすべてのモデル構成と互換性があるわけではありませんが、試してみる価値があります。

### マスク損失

被写体またはスタイルをトレーニングしていて、一方または他方をマスクしたい場合は、Dreamboothガイドの[マスク損失トレーニング](../DREAMBOOTH.md#masked-loss)セクションを参照してください。

### 正則化データ

正則化データセットの詳細については、Dreamboothガイドの[このセクション](../DREAMBOOTH.md#prior-preservation-loss)と[このセクション](../DREAMBOOTH.md#regularisation-dataset-considerations)を参照してください。

### 量子化トレーニング

SD3および他のモデルの量子化を設定する方法については、Dreamboothガイドの[このセクション](../DREAMBOOTH.md#quantised-model-training-loralycoris-only)を参照してください。

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
