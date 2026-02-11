# Flux[dev] / Flux[schnell] クイックスタート

![image](https://github.com/user-attachments/assets/6409d790-3bb4-457c-a4b4-a51a45fc91d1)

この例では、Flux.1 Krea LoRAをトレーニングします。

## ハードウェア要件

FluxはGPUメモリに加えて大量の**システムRAM**を必要とします。起動時にモデルを量子化するだけで約50GBのシステムメモリが必要です。起動に過度に時間がかかる場合は、ハードウェアの能力と変更が必要かどうかを評価する必要があるかもしれません。

rank-16 LoRAのすべてのコンポーネント(MLP、プロジェクション、マルチモーダルブロック)をトレーニングする場合、使用するメモリは以下のようになります:

- ベースモデルを量子化しない場合、30GB以上のVRAM
- int8 + bf16ベース/LoRA重みに量子化する場合、18GB以上のVRAM
- int4 + bf16ベース/LoRA重みに量子化する場合、13GB以上のVRAM
- NF4 + bf16ベース/LoRA重みに量子化する場合、9GB以上のVRAM
- int2 + bf16ベース/LoRA重みに量子化する場合、9GB以上のVRAM

必要な環境:

- **絶対最小要件**は単一の**3080 10G**
- **現実的な最小要件**は単一の3090またはV100 GPU
- **理想的には**複数の4090、A6000、L40S、またはそれ以上

幸いなことに、これらは[LambdaLabs](https://lambdalabs.com)などのプロバイダーを通じて容易に利用可能です。LambdaLabsは最低料金を提供し、マルチノードトレーニングのためのローカライズされたクラスターを提供しています。

**他のモデルとは異なり、Apple GPUは現在Fluxのトレーニングには対応していません。**


## 前提条件

Pythonがインストールされていることを確認してください。SimpleTunerは3.10から3.12で正常に動作します。

以下のコマンドを実行して確認できます:

```bash
python --version
```

Ubuntuでpython 3.12がインストールされていない場合は、以下を試してください:

```bash
apt -y install python3.13 python3.13-venv
```

### コンテナイメージの依存関係

Vast、RunPod、TensorDock(など)の場合、CUDA 12.2-12.8イメージでCUDA拡張のコンパイルを有効にするには以下が機能します:

```bash
apt -y install nvidia-cuda-toolkit
```

## インストール

SimpleTunerをpip経由でインストール:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](../INSTALL.md)を参照してください。

### AMD ROCmフォローアップ手順

AMD MI300Xを使用可能にするには、以下を実行する必要があります:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

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

実験的なスクリプト`configure.py`を使用すると、対話的なステップバイステップの設定を通じて、このセクション全体をスキップできる可能性があります。これには、一般的な落とし穴を回避するのに役立ついくつかの安全機能が含まれています。

**注意:** これはデータローダーを設定しません。後で手動で設定する必要があります。

実行するには:

```bash
simpletuner configure
```

> ⚠️ Hugging Face Hubに容易にアクセスできない国にいるユーザーは、使用している`$SHELL`に応じて、`~/.bashrc`または`~/.zshrc`に`HF_ENDPOINT=https://hf-mirror.com`を追加する必要があります。

手動で設定する場合:

`config/config.json.example`を`config/config.json`にコピー:

```bash
cp config/config.json.example config/config.json
```

そこで、以下の変数を変更する必要がある可能性があります:

- `model_type` - `lora`に設定します。
- `model_family` - `flux`に設定します。
- `model_flavour` - デフォルトは`krea`ですが、オリジナルのFLUX.1-Devリリースをトレーニングするには`dev`に設定できます。
  - `krea` - デフォルトのFLUX.1-Krea [dev]モデル、BFLとKrea.aiのプロプライエタリモデルコラボレーションであるKrea 1のオープンウェイトバリアント
  - `dev` - Devモデルフレーバー、以前のデフォルト
  - `schnell` - Schnellモデルフレーバー。クイックスタートは高速ノイズスケジュールとアシスタントLoRAスタックを自動的に設定します
  - `kontext` - Kontextトレーニング(具体的なガイダンスについては[このガイド](../quickstart/FLUX_KONTEXT.md)を参照)
  - `fluxbooru` - FLUX.1-Devベースのde-distilled(CFGが必要)モデル、[FluxBooru](https://hf.co/terminusresearch/fluxbooru-v0.3)と呼ばれ、terminus research groupによって作成
  - `libreflux` - FLUX.1-Schnellベースのde-distilledモデルで、T5テキストエンコーダー入力にアテンションマスキングが必要
- `offload_during_startup` - VAEエンコード中にメモリ不足になる場合は`true`に設定します。
- `pretrained_model_name_or_path` - `black-forest-labs/FLUX.1-dev`に設定します。
- `pretrained_vae_model_name_or_path` - `black-forest-labs/FLUX.1-dev`に設定します。
  - このモデルをダウンロードするには、Huggingfaceにログインしてアクセス権を付与される必要があることに注意してください。Huggingfaceへのログインについては、このチュートリアルの後半で説明します。
- `output_dir` - チェックポイントと検証画像を保存するディレクトリに設定します。ここではフルパスを使用することをお勧めします。
- `train_batch_size` - 特に非常に小さなデータセットがある場合は、これを1のままにしておく必要があります。
- `validation_resolution` - Fluxは1024pxモデルなので、`1024x1024`に設定できます。
  - さらに、Fluxはマルチアスペクトバケットでファインチューニングされており、カンマで区切って他の解像度を指定できます: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - Fluxの推論時に選択する通常の値を使用します。
- `validation_guidance_real` - Flux推論でCFGを使用するには>1.0を使用します。検証が遅くなりますが、より良い結果が得られます。空の`VALIDATION_NEGATIVE_PROMPT`で最良の結果が得られます。
- `validation_num_inference_steps` - 時間を節約しながらまともな品質を確認するには、20前後を使用します。Fluxはあまり多様性がなく、ステップ数を増やしても時間を浪費するだけかもしれません。
- `--lora_rank=4` トレーニングされるLoRAのサイズを大幅に削減したい場合。これはVRAM使用量を削減するのに役立ちます。
- Schnell LoRA実行では、クイックスタートのデフォルトにより高速スケジュールが自動的に使用されます。追加のフラグは不要です。

- `gradient_accumulation_steps` - 以前のガイダンスでは、bf16トレーニングではモデルが劣化するため、これらを避けるように推奨していました。さらなるテストにより、Fluxの場合は必ずしもそうではないことが示されました。
  - このオプションにより、更新ステップが複数のステップにわたって蓄積されます。これによりトレーニングランタイムが線形に増加し、値2ではトレーニングが半分の速度で実行され、2倍の時間がかかります。
- `optimizer` - 初心者にはadamw_bf16が推奨されますが、optimi-lionやoptimi-stableadamwも良い選択肢です。
- `mixed_precision` - 初心者は`bf16`のままにしておく必要があります
- `gradient_checkpointing` - ほぼすべての状況、すべてのデバイスでtrueに設定します
- `gradient_checkpointing_interval` - 大きなGPUでは、_n_ブロックごとにのみチェックポイントするように2以上の値に設定できます。値2ではブロックの半分がチェックポイントされ、3では3分の1になります。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTunerには、トレーニングの安定性とパフォーマンスを大幅に向上させることができる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが独自の入力を生成できるようにすることで、露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

</details>

### メモリオフロード(オプション)

Fluxはdiffusers v0.33+を介してグループ化されたモジュールオフロードをサポートしています。これにより、トランスフォーマーの重みでボトルネックになっている場合、VRAMの圧力が劇的に軽減されます。`TRAINER_EXTRA_ARGS`(またはWebUIのハードウェアページ)に以下のフラグを追加することで有効にできます:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- `--group_offload_use_stream`はCUDAデバイスでのみ有効です。SimpleTunerはROCm、MPS、CPUバックエンドでストリームを自動的に無効にします。
- `--enable_model_cpu_offload`と組み合わせ**ない**でください。2つの戦略は相互に排他的です。
- `--group_offload_to_disk_path`を使用する場合は、高速なローカルSSD/NVMeターゲットを優先してください。

#### 検証プロンプト

`config/config.json`内には「プライマリ検証プロンプト」があり、これは通常、単一の被写体またはスタイルに対してトレーニングしているメインのinstance_promptです。さらに、検証中に実行する追加のプロンプトを含むJSONファイルを作成できます。

サンプル設定ファイル`config/user_prompt_library.json.example`には以下の形式が含まれています:

<details>
<summary>サンプル設定を表示</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

ニックネームは検証のファイル名なので、短く、ファイルシステムと互換性のあるものにしてください。

トレーナーをこのプロンプトライブラリに指定するには、`config.json`の末尾に新しい行を追加してTRAINER_EXTRA_ARGSに追加します:

<details>
<summary>サンプル設定を表示</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

多様なプロンプトのセットは、モデルがトレーニング中に崩壊しているかどうかを判断するのに役立ちます。この例では、`<token>`という単語を被写体名(instance_prompt)に置き換える必要があります。

<details>
<summary>サンプル設定を表示</summary>

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing her essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing her unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing her dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```
</details>

> ℹ️ Fluxはフローマッチングモデルであり、強い類似性を持つ短いプロンプトは、モデルによってほぼ同じ画像が生成されます。必ずより長く、より説明的なプロンプトを使用してください。

#### CLIPスコアトラッキング

モデルのパフォーマンスをスコアリングするための評価を有効にしたい場合は、CLIPスコアの設定と解釈に関する情報について[このドキュメント](../evaluation/CLIP_SCORES.md)を参照してください。

# 安定した評価損失

モデルのパフォーマンスをスコアリングするために安定したMSE損失を使用したい場合は、評価損失の設定と解釈に関する情報について[このドキュメント](../evaluation/EVAL_LOSS.md)を参照してください。

#### 検証プレビュー

SimpleTunerは、Tiny AutoEncoderモデルを使用した生成中の中間検証プレビューのストリーミングをサポートしています。これにより、Webhookコールバックを介してリアルタイムで段階的に生成される検証画像を確認できます。

有効にするには:
<details>
<summary>サンプル設定を表示</summary>

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

Tiny AutoEncoderのオーバーヘッドを削減するには、`validation_preview_steps`をより高い値(3または5など)に設定します。`validation_num_inference_steps=20`および`validation_preview_steps=5`の場合、ステップ5、10、15、20でプレビュー画像を受け取ります。

#### Fluxタイムスケジュールシフト

FluxやSD3などのフローマッチングモデルには、単純な10進数値を使用してタイムステップスケジュールのトレーニング部分をシフトできる「シフト」と呼ばれるプロパティがあります。

##### デフォルト

デフォルトでは、fluxにスケジュールシフトは適用されず、タイムステップサンプリング分布にシグモイドベル形が生じます。これはFluxにとって理想的なアプローチではない可能性がありますが、自動シフトよりも短期間でより多くの学習が得られます。

##### 自動シフト

一般的に推奨されるアプローチは、最近のいくつかの研究に従って、解像度依存のタイムステップシフトを有効にすることです。`--flow_schedule_auto_shift`は、大きな画像にはより高いシフト値を使用し、小さな画像にはより低いシフト値を使用します。これにより安定したトレーニング結果が得られますが、潜在的に平凡な結果になる可能性があります。

##### 手動指定

(_Discordのジェネラル・アウェアネスによる以下の例に感謝_)

`--flow_schedule_shift`値0.1(非常に低い値)を使用すると、画像の細かいディテールのみが影響を受けます:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

`--flow_schedule_shift`値4.0(非常に高い値)を使用すると、大きな構成的特徴とモデルの色空間が潜在的に影響を受けます:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)

#### 量子化モデルトレーニング

AppleおよびNVIDIAシステムでテスト済み、Hugging Face Optimum-Quantoを使用して精度とVRAM要件を削減し、わずか16GBでFluxをトレーニングできます。

`config.json`ユーザー向け:

<details>
<summary>サンプル設定を表示</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

##### LoRA固有の設定(LyCORISではない)

```bash
# 'mmdit'をトレーニングすると、非常に安定したトレーニングが得られ、モデルの学習に時間がかかります。
# 'all'をトレーニングすると、モデル分布を簡単にシフトできますが、忘却しやすく、高品質データの恩恵を受けます。
# 'all+ffs'をトレーニングすると、すべてのアテンションレイヤーに加えてフィードフォワードがトレーニングされ、LoRAのモデル目的の適応に役立ちます。
# - このモードは移植性に欠けると報告されており、ComfyUIなどのプラットフォームではLoRAをロードできない可能性があります。
# 'context'ブロックのみをトレーニングするオプションも提供されていますが、その影響は不明であり、実験的な選択肢として提供されています。
# - このモードの拡張である'context+ffs'も利用可能で、`--init_lora`を介して継続的にファインチューニングする前に、新しいトークンをLoRAに事前トレーニングするのに役立ちます。
# その他のオプションには、1つまたは2つのレイヤーのみをトレーニングする'tiny'と'nano'が含まれます。
"--flux_lora_target": "all",

# LoftQ初期化を使用したい場合、Quantoを使用してベースモデルを量子化することはできません。
# これにより、より良い/より速い収束が得られる可能性がありますが、NVIDIAデバイスでのみ機能し、Bits n Bytesが必要で、Quantoとは互換性がありません。
# その他のオプションは'default'、'gaussian'(困難)、およびテストされていないオプション: 'olora'および'pissa'です。
"--lora_init_type": "loftq",
```

#### データセットの考慮事項

> ⚠️ トレーニング用の画像品質は、他のほとんどのモデルよりもFluxにとって重要です。画像内のアーティファクトを_最初に_吸収し、その後コンセプト/被写体を学習するためです。

モデルをトレーニングするには、かなりのデータセットが必要です。データセットサイズには制限があり、モデルを効果的にトレーニングするためにデータセットが十分に大きいことを確認する必要があります。最小限のデータセットサイズは`train_batch_size * gradient_accumulation_steps`であり、さらに`vae_batch_size`よりも多い必要があることに注意してください。データセットが小さすぎると使用できません。

> ℹ️ 画像が十分に少ない場合、**データセットに画像が検出されません**というメッセージが表示される可能性があります - `repeats`値を増やすことでこの制限を克服できます。

所有しているデータセットに応じて、データセットディレクトリとデータローダー設定ファイルを異なる方法でセットアップする必要があります。この例では、データセットとして[pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k)を使用します。

これを含む`--data_backend_config`(`config/multidatabackend.json`)ドキュメントを作成します:

<details>
<summary>サンプル設定を表示</summary>

```json
[
  {
    "id": "pseudo-camera-10k-flux",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "dreambooth-subject-512",
    "type": "local",
    "crop": false,
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/dreambooth-subject-512",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> caption_strategyのオプションと要件については、[DATALOADER.md](../DATALOADER.md#caption_strategy)を参照してください。

> ℹ️ 512pxと1024pxのデータセットを同時に実行することはサポートされており、Fluxのより良い収束をもたらす可能性があります。

次に、`datasets`ディレクトリを作成します:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

これにより、約10kの写真サンプルが`datasets/pseudo-camera-10k`ディレクトリにダウンロードされ、自動的に作成されます。

Dreambooth画像は`datasets/dreambooth-subject`ディレクトリに配置する必要があります。

#### WandBとHuggingface Hubにログイン

特に`--push_to_hub`と`--report_to=wandb`を使用している場合は、トレーニングを開始する前にWandBとHF Hubにログインする必要があります。

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

SimpleTunerディレクトリから、トレーニングを開始するためのいくつかのオプションがあります:

**オプション1(推奨 - pipインストール):**

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train
```

**オプション2(Gitクローン方式):**

```bash
simpletuner train
```

**オプション3(レガシー方式 - まだ機能します):**

```bash
./train.sh
```

これにより、テキスト埋め込みとVAE出力のディスクへのキャッシングが開始されます。

詳細については、[データローダー](../DATALOADER.md)および[チュートリアル](../TUTORIAL.md)ドキュメントを参照してください。

**注意:** 現時点では、マルチアスペクトバケットでのトレーニングがFluxで正しく機能するかどうかは不明です。`crop_style=random`および`crop_aspect=square`を使用することをお勧めします。

## マルチGPU構成

SimpleTunerには、WebUIを介した**自動GPU検出**が含まれています。オンボーディング中に次のように設定します:

- **自動モード**: 最適な設定で検出されたすべてのGPUを自動的に使用
- **手動モード**: 特定のGPUを選択するか、カスタムプロセス数を設定
- **無効モード**: シングルGPUトレーニング

WebUIはハードウェアを検出し、`--num_processes`と`CUDA_VISIBLE_DEVICES`を自動的に設定します。

手動設定または高度なセットアップについては、インストールガイドの[マルチGPUトレーニングセクション](../INSTALL.md#multiple-gpu-training)を参照してください。

## 推論のヒント

### CFGトレーニングされたLoRA(flux_guidance_value > 1)

ComfyUIでは、FluxをAdaptiveGuiderと呼ばれる別のノードを通す必要があります。コミュニティのメンバーの1人が、修正されたノードをここで提供しています:

(**外部リンク**) [IdiotSandwichTheThird/ComfyUI-Adaptive-Guidan...](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps) および彼らのサンプルワークフロー [こちら](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps/blob/master/ExampleWorkflow.json)

### CFG蒸留LoRA(flux_guidance_scale == 1)

CFG蒸留LoRAの推論は、トレーニングされた値の周りの低いguidance_scaleを使用するのと同じくらい簡単です。

## 注意事項とトラブルシューティングのヒント

### 最低VRAM構成

現在、最低VRAM使用率(9090M)は以下で達成できます:

- OS: Ubuntu Linux 24
- GPU: 単一のNVIDIA CUDAデバイス(10G、12G)
- システムメモリ: 約50Gのシステムメモリ
- ベースモデル精度: `nf4-bnb`
- オプティマイザ: Lion 8Bit Paged、`bnb-lion8bit-paged`
- 解像度: 512px
  - 1024pxには >= 12G VRAMが必要
- バッチサイズ: 1、勾配累積ステップなし
- DeepSpeed: 無効/未設定
- PyTorch: 2.6 Nightly(9月29日ビルド)
- `--quantize_via=cpu`を使用して、<=16Gカードでの起動時のoutOfMemoryエラーを回避
- `--attention_mechanism=sageattention`を使用してVRAMをさらに0.1GB削減し、トレーニング検証画像生成速度を向上
- `--gradient_checkpointing`を必ず有効にしてください。そうしないと、何をしてもOOMが止まりません

**注意**: VAE埋め込みとテキストエンコーダ出力の事前キャッシングはより多くのメモリを使用する可能性があり、OOMになる可能性があります。その場合、テキストエンコーダ量子化とVAEタイリングを`--vae_enable_tiling=true`経由で有効にできます。起動時のメモリをさらに節約するには、`--offload_during_startup=true`を使用できます。

4090での速度は1秒あたり約1.4イテレーションでした。

### SageAttention

`--attention_mechanism=sageattention`を使用すると、検証時の推論を高速化できます。

**注意**: これはすべてのモデル構成と互換性があるわけではありませんが、試してみる価値があります。

### NF4量子化トレーニング

簡単に言うと、NF4はモデルの4ビット的な表現であり、トレーニングには対処すべき深刻な安定性の懸念があります。

初期テストでは、以下が当てはまります:

- Lionオプティマイザはモデル崩壊を引き起こしますが、VRAMを最も使用しません。AdamWバリアントはそれを維持するのに役立ちます。bnb-adamw8bit、adamw_bf16は素晴らしい選択肢です
  - AdEMAMixはうまくいきませんでしたが、設定は探索されていません
- `--max_grad_norm=0.01`は、モデルが短時間で大きく変化するのを防ぐことでモデルの破損をさらに減らすのに役立ちます
- NF4、AdamW8bit、およびより高いバッチサイズはすべて、トレーニングに費やす時間やVRAMの使用を犠牲にして、安定性の問題を克服するのに役立ちます
- 解像度を512pxから1024pxに上げると、トレーニングが遅くなります。たとえば、1ステップあたり1.4秒から3.5秒になります(バッチサイズ1、4090)
- int8またはbf16でトレーニングするのが難しいものは、NF4ではさらに難しくなります
- SageAttentionなどのオプションとの互換性が低くなります

NF4はtorch.compileで動作しないため、速度に関して得られるものが得られるものです。

VRAMが問題でない場合(48G以上など)、torch.compileを使用したint8が最良で最速のオプションです。

### マスク損失

被写体またはスタイルをトレーニングしていて、一方または他方をマスクしたい場合は、Dreamboothガイドの[マスク損失トレーニング](../DREAMBOOTH.md#masked-loss)セクションを参照してください。

### TREADトレーニング {#tread-training}

> ⚠️ **実験的**: TREADは新しく実装された機能です。機能的ですが、最適な構成はまだ探索中です。

[TREAD](../TREAD.md)(論文)は、**T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusionの略です。これは、トランスフォーマーレイヤーを介してトークンをインテリジェントにルーティングすることで、Fluxトレーニングを加速できる方法です。スピードアップは、ドロップするトークンの数に比例します。

#### クイックセットアップ

これを`config.json`に追加します:

<details>
<summary>サンプル設定を表示</summary>

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

この構成により:

- レイヤー2から最後から2番目のレイヤーまでの間、画像トークンの50%のみを保持
- テキストトークンはドロップされません
- 品質への影響を最小限に抑えながら、トレーニングのスピードアップは約25%

#### キーポイント

- **限定的なアーキテクチャサポート** - TREADはFluxおよびWanモデルに対してのみ実装されています
- **高解像度で最適** - アテンションのO(n²)複雑性のため、1024x1024+で最大のスピードアップ
- **マスク損失と互換性あり** - マスクされた領域は自動的に保持されます(ただし、これによりスピードアップが減少します)
- **量子化と動作** - int8/int4/NF4トレーニングと組み合わせることができます
- **初期損失スパイクを予想** - LoRA/LoKrトレーニングを開始すると、損失は最初は高くなりますが、すぐに修正されます

#### チューニングのヒント

- **保守的(品質重視)**: `selection_ratio`を0.3-0.5に使用
- **積極的(速度重視)**: `selection_ratio`を0.6-0.8に使用
- **早期/後期レイヤーを避ける**: レイヤー0-1または最終レイヤーでルーティングしない
- **LoRAトレーニングの場合**: わずかな速度低下が見られる可能性があります - 異なる構成を試してください
- **より高い解像度 = より良いスピードアップ**: 1024px以上で最も有益

#### 既知の動作

- ドロップされるトークンが多いほど(より高い`selection_ratio`)、トレーニングは速くなりますが、初期損失は高くなります
- LoRA/LoKrトレーニングは初期損失スパイクを示しますが、ネットワークが適応するにつれて急速に修正されます
- 一部のLoRA構成はわずかに遅くトレーニングされる可能性があります - 最適な構成はまだ探索中です
- RoPE(回転位置埋め込み)実装は機能的ですが、100%正確ではない可能性があります

詳細な構成オプションとトラブルシューティングについては、[完全なTREADドキュメント](../TREAD.md)を参照してください。

### クラシファイアフリーガイダンス

#### 問題

Devモデルはガイダンス蒸留された状態で提供されます。つまり、教師モデルの出力への非常に直線的な軌道を取ります。これは、トレーニングおよび推論時にモデルに供給されるガイダンスベクトルを介して行われます - このベクトルの値は、最終的に得られるLoRAのタイプに大きく影響します:

#### 解決策

- 1.0の値(**デフォルト**)は、Devモデルに対して行われた初期蒸留を保持します
  - これは最も互換性のあるモードです
  - 推論は元のモデルと同じくらい高速です
  - フローマッチング蒸留により、元のFlux Devモデルと同様に、モデルの創造性と出力の変動性が低下します(すべてが同じ構成/外観を保ちます)
- より高い値(3.5-4.5前後でテスト済み)は、CFG目的をモデルに再導入します
  - これには、推論パイプラインにCFGのサポートが必要です
  - 推論は50%遅く、0%のVRAM増加**または**約20%遅く、バッチCFG推論による20%のVRAM増加
  - ただし、このスタイルのトレーニングは創造性とモデル出力の変動性を改善し、特定のトレーニングタスクに必要な場合があります

ベクトル値1.0を使用してモデルを継続的に調整することで、de-distilledモデルに部分的に蒸留を再導入できます。完全に回復することはありませんが、少なくともより使いやすくなります。

#### 注意事項

- これにより、最終的に**以下のいずれかの**影響があります:
  - 2つの別々のフォワードパスで無条件出力を順次計算する場合など、推論レイテンシが2倍になります
  - `num_images_per_prompt=2`を使用して推論時に2つの画像を受け取るのと同等のVRAM消費の増加、および同じパーセンテージの速度低下を伴います。
    - この方法は、順次計算よりも極端な速度低下が少ないことが多いですが、VRAM使用量はほとんどの消費者向けトレーニングハードウェアには多すぎる可能性があります。
    - この方法は現在SimpleTunerに統合されて_いません_が、作業は進行中です。
- ComfyUIまたは他のアプリケーション(AUTOMATIC1111など)の推論ワークフローは、「真の」CFGを有効にするように変更する必要があります。これは、すぐには可能ではない可能性があります。

### 量子化

- 16Gカードでこのモデルをトレーニングするには、最低8ビット量子化が必要です
  - bfloat16/float16では、rank-1 LoRAはメモリ使用量が30GBをわずかに超えます
- モデルを8ビットに量子化してもトレーニングに害はありません
  - より高いバッチサイズをプッシュでき、より良い結果が得られる可能性があります
  - フル精度トレーニングと同じように動作します - fp32はbf16+int8よりもモデルを改善しません。
- **int8**には、新しいNVIDIAハードウェア(3090以上)でハードウェアアクセラレーションと`torch.compile()`サポートがあります
- **nf4-bnb**はVRAM要件を9GBに引き下げ、10Gカード(bfloat16サポート付き)に収まります
- 後でComfyUIでLoRAをロードするときは、LoRAをトレーニングしたのと同じベースモデル精度を使用**する必要があります**。
- **int4**はカスタムbf16カーネルに依存しており、カードがbfloat16をサポートしていない場合は機能しません

### クラッシュ

- テキストエンコーダがアンロードされた後にSIGKILLが発生する場合、これはFluxを量子化するのに十分なシステムメモリがないことを意味します。
  - `--base_model_precision=bf16`をロードしてみてください。それでもうまくいかない場合は、より多くのメモリが必要になる可能性があります。
  - GPU代わりに使用するには、`--quantize_via=accelerator`を試してください

### Schnell

- DevでLyCORIS LoKrをトレーニングすると、わずか4ステップ後にSchnellで**通常**非常にうまく機能します。
  - 直接Schnellトレーニングには本当にもう少し時間が必要です - 現在、結果は良くありません

> ℹ️ SchnellをDevと何らかの方法でマージすると、Devのライセンスが引き継がれ、非商用になります。これはほとんどのユーザーにとって実際には問題にならないはずですが、注意する価値があります。

### 学習率

#### LoRA(--lora_type=standard)

- LoRAは、大規模データセットに対してLoKrよりも全体的にパフォーマンスが悪い
- Flux LoRAはSD 1.5 LoRAと同様にトレーニングされると報告されています
- ただし、12Bほど大きなモデルは、経験的に**より低い学習率**でより良いパフォーマンスを発揮しました。
  - 1e-3のLoRAは完全にローストする可能性があります。1e-5のLoRAはほとんど何もしません。
- 64から128ほど大きなランクは、ベースモデルのサイズに比例する一般的な困難性のため、12Bモデルでは望ましくない可能性があります。
  - より小さなネットワーク(rank-1、rank-4)を最初に試して、徐々に上げていってください - より速くトレーニングされ、必要なすべてを実行できる可能性があります。
  - コンセプトをモデルにトレーニングするのが過度に難しい場合は、より高いランクとより多くの正則化データが必要になる可能性があります。
- PixArtやSD3などの他の拡散トランスフォーマーモデルは、`--max_grad_norm`から大きな恩恵を受け、SimpleTunerはデフォルトでFluxでこれにかなり高い値を保持しています。
  - より低い値は、モデルがすぐに崩壊するのを防ぎますが、ベースモデルのデータ分布から遠く離れた新しいコンセプトを学習することを非常に困難にする可能性もあります。モデルが行き詰まり、改善されない可能性があります。

#### LoKr(--lora_type=lycoris)

- LoKrにはより高い学習率の方が良い(`1e-3`でAdamW、`2e-4`でLion)
- 他のアルゴはさらに探索が必要です。
- そのようなデータセットで`is_regularisation_data`を設定すると、ブリードを保持/防止し、最終的なモデルの品質を向上させるのに役立つ可能性があります。
  - これは、トレーニングバッチサイズを2倍にすることで知られているが、結果をあまり改善しない「事前損失保存」とは異なる動作をします
  - SimpleTunerの正則化データ実装は、ベースモデルを保存する効率的な方法を提供します

### 画像アーティファクト

Fluxは悪い画像アーティファクトを即座に吸収します。それがそのままです - 最後に高品質データのみでの最終トレーニング実行が、最後にそれを修正するために必要になる可能性があります。

これらのこと(その他)を行うと、サンプルに正方形グリッドアーティファクトが現れ**始める可能性があります**:

- 低品質データでオーバートレーニング
- 学習率が高すぎる使用
- オーバートレーニング(一般的に)、画像が多すぎる低容量ネットワーク
- アンダートレーニング(また)、画像が少なすぎる高容量ネットワーク
- 非標準のアスペクト比またはトレーニングデータサイズの使用

### アスペクトバケッティング

- 正方形クロップで長時間トレーニングしても、このモデルを過度に損傷することはおそらくありません。どんどん試してください。素晴らしくて信頼性があります。
- 一方、データセットの自然なアスペクトバケットを使用すると、推論時にこれらの形状を過度にバイアスする可能性があります。
  - これは望ましい品質である可能性があります。シネマティックなもののようなアスペクト依存のスタイルが他の解像度に過度にブリードするのを防ぐためです。
  - ただし、多くのアスペクトバケットで均等に結果を改善したい場合は、`crop_aspect=random`を試す必要があるかもしれません。これには独自の欠点があります。
- 画像ディレクトリデータセットを複数回定義することでデータセット構成を混合すると、本当に良い結果とうまく一般化されたモデルが生成されました。

### カスタムファインチューニングされたFluxモデルのトレーニング

Hugging Face Hub上の一部のファインチューニングされたFluxモデル(Dev2Proなど)には完全なディレクトリ構造がないため、これらの特定のオプションを設定する必要があります。

作成者が行ったように、これらのオプション`flux_guidance_value`、`validation_guidance_real`、`flux_attention_masked_training`もその情報が利用可能な場合は必ず設定してください。

<details>
<summary>サンプル設定を表示</summary>

```json
{
    "model_family": "flux",
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_transformer_model_name_or_path": "ashen0209/Flux-Dev2Pro",
    "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_transformer_subfolder": "none",
}
```
</details>
