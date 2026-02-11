## NVLabs Sana クイックスタート

この例では、NVLabs Sanaモデルをフルランクでトレーニングします。

### ハードウェア要件

Sanaは非常に軽量で、24GカードではフルGradient Checkpointingを有効にする必要すらないかもしれません。これは非常に高速にトレーニングできることを意味します！

- **絶対最小要件**は約12G VRAMですが、このガイドではそこまで到達できないかもしれません
- **現実的な最小要件**は単一の3090またはV100 GPU
- **理想的には**複数の4090、A6000、L40S、またはそれ以上

SanaはSimpleTunerでトレーニング可能な他のモデルと比較して独特なアーキテクチャです:

- 当初、他のモデルとは異なり、Sanaはfp16トレーニングを必要とし、bf16ではクラッシュしていました
  - NVIDIAのモデル作成者は、ファインチューニング用のbf16互換ウェイトでフォローアップしてくれました
- bf16/fp16の問題により、このモデルファミリでは量子化がより敏感になる可能性があります
- SageAttentionは、現在サポートされていないhead_dim形状のため、Sanaでは（まだ）動作しません
- Sanaをトレーニングする際の損失値は非常に高く、他のモデルよりもはるかに低い学習率（例：`1e-5`前後）が必要になる可能性があります
- トレーニングがNaN値に達することがあり、なぜこれが起こるかは明確ではありません

Gradient Checkpointingは VRAMを解放できますが、トレーニングを遅くします。5800X3D搭載4090でのテスト結果のチャート:

![image](https://github.com/user-attachments/assets/310bf099-a077-4378-acf4-f60b4b82fdc4)

SimpleTunerのSanaモデリングコードでは、`--gradient_checkpointing_interval`を指定して_n_ブロックごとにチェックポイントし、上記のチャートに示されている結果を達成できます。

### 前提条件

Pythonがインストールされていることを確認してください。SimpleTunerは3.10から3.12で正常に動作します。

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

そこで、以下の変数を変更する必要がある可能性があります:

- `model_type` - `full`に設定します。
- `model_family` - `sana`に設定します。
- `pretrained_model_name_or_path` - `terminusresearch/sana-1.6b-1024px`に設定します
- `output_dir` - チェックポイントと検証画像を保存するディレクトリに設定します。ここではフルパスを使用することをお勧めします。
- `train_batch_size` - フルGradient Checkpointingを使用した24Gカードでは、6まで上げることができます。
- `validation_resolution` - このSanaチェックポイントは1024pxモデルなので、`1024x1024`またはSanaの他のサポートされている解像度に設定する必要があります。
  - 他の解像度はカンマで区切って指定できます: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - Sanaの推論時に選択する通常の値を使用します。
- `validation_num_inference_steps` - 最高品質には約50を使用しますが、結果に満足していれば少なくすることもできます。
- `use_ema` - `true`に設定すると、メインのトレーニング済みチェックポイントと一緒により滑らかな結果を得るのに大いに役立ちます。

- `optimizer` - 慣れ親しんだオプティマイザを使用できますが、この例では`optimi-adamw`を使用します。
- `mixed_precision` - 最も効率的なトレーニング設定には`bf16`に設定することをお勧めします。または`no`（ただし、より多くのメモリを消費し、遅くなります）。
  - `fp16`の値はここでは推奨されませんが、特定のSanaファインチューンでは必要な場合があります（これを有効にすると他の新しい問題が発生します）
- `gradient_checkpointing` - これを無効にすると最速になりますが、バッチサイズが制限されます。最低VRAM使用量を得るにはこれを有効にする必要があります。
- `gradient_checkpointing_interval` - GPUで`gradient_checkpointing`が過剰に感じる場合は、_n_ブロックごとにのみチェックポイントするように2以上の値に設定できます。値2ではブロックの半分がチェックポイントされ、3では3分の1になります。

マルチGPUユーザーは、使用するGPUの数を設定する方法について[このドキュメント](../OPTIONS.md#environment-configuration-variables)を参照してください。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTunerには、トレーニングの安定性とパフォーマンスを大幅に向上させることができる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが独自の入力を生成できるようにすることで、露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

#### 検証プロンプト

`config/config.json`内には「プライマリ検証プロンプト」があり、これは通常、単一のサブジェクトまたはスタイルに対してトレーニングしているメインのinstance_promptです。さらに、検証中に実行する追加のプロンプトを含むJSONファイルを作成できます。

サンプル設定ファイル`config/user_prompt_library.json.example`には以下の形式が含まれています:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

ニックネームは検証のファイル名なので、短く、ファイルシステムと互換性のあるものにしてください。

トレーナーをこのプロンプトライブラリに指定するには、`config.json`の末尾に新しい行を追加してTRAINER_EXTRA_ARGSに追加します:
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

多様なプロンプトのセットは、モデルがトレーニング中に崩壊しているかどうかを判断するのに役立ちます。この例では、`<token>`という単語をサブジェクト名（instance_prompt）に置き換える必要があります。

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

> ℹ️ Sanaは独特なテキストエンコーダー構成を使用しているため、短いプロンプトは非常に悪く見える可能性があります。

#### CLIPスコアトラッキング

モデルのパフォーマンスをスコアリングするための評価を有効にしたい場合は、CLIPスコアの設定と解釈に関する情報について[このドキュメント](../evaluation/CLIP_SCORES.md)を参照してください。

</details>

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

#### Sanaタイムスケジュールシフト

Sana、Flux、SD3などのフローマッチングモデルには、単純な小数値を使用してタイムステップスケジュールのトレーニング部分をシフトできる「シフト」と呼ばれるプロパティがあります。

##### 自動シフト
一般的に推奨されるアプローチは、最近のいくつかの研究に従って、解像度依存のタイムステップシフト`--flow_schedule_auto_shift`を有効にすることです。これは、大きな画像にはより高いシフト値を使用し、小さな画像にはより低いシフト値を使用します。これにより安定したトレーニング結果が得られますが、潜在的に平凡な結果になる可能性があります。

##### 手動指定
_Discordのジェネラル・アウェアネスによる以下の例に感謝_

`--flow_schedule_shift`値0.1（非常に低い値）を使用すると、画像の細かいディテールのみが影響を受けます:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

`--flow_schedule_shift`値4.0（非常に高い値）を使用すると、大きな構成的特徴とモデルの色空間が潜在的に影響を受けます:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)

#### データセットの考慮事項

> ⚠️ トレーニング用の画像品質は、他のほとんどのモデルよりもSanaにとって重要です。画像内のアーティファクトを*最初に*吸収し、その後コンセプト/サブジェクトを学習するためです。

モデルをトレーニングするには、かなりのデータセットが必要です。データセットサイズには制限があり、モデルを効果的にトレーニングするためにデータセットが十分に大きいことを確認する必要があります。最小限のデータセットサイズは`train_batch_size * gradient_accumulation_steps`であり、さらに`vae_batch_size`より多い必要があることに注意してください。データセットが小さすぎると使用できません。

> ℹ️ 画像が十分に少ない場合、**no images detected in dataset**というメッセージが表示される可能性があります - `repeats`値を増やすとこの制限を克服できます。

持っているデータセットに応じて、データセットディレクトリとデータローダー設定ファイルを異なる方法でセットアップする必要があります。この例では、データセットとして[pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k)を使用します。

以下を含む`--data_backend_config`（`config/multidatabackend.json`）ドキュメントを作成します:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sana",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/sana/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/sana/dreambooth-subject",
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
    "cache_dir_vae": "cache/vae/sana/dreambooth-subject-512",
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
    "cache_dir": "cache/text/sana",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> caption_strategyオプションと要件については、[DATALOADER.md](../DATALOADER.md#caption_strategy)を参照してください。

> ℹ️ 512pxと1024pxのデータセットを同時に実行することはサポートされており、Sanaのより良い収束をもたらす可能性があります。

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

SimpleTunerディレクトリから、トレーニングを開始するいくつかのオプションがあります:

**オプション1（推奨 - pipインストール）:**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train
```

**オプション2（Gitクローン方式）:**
```bash
simpletuner train
```

**オプション3（レガシー方式 - まだ機能します）:**
```bash
./train.sh
```

これにより、テキスト埋め込みとVAE出力のディスクへのキャッシングが開始されます。

詳細については、[データローダー](../DATALOADER.md)および[チュートリアル](../TUTORIAL.md)ドキュメントを参照してください。

## 注意事項とトラブルシューティングのヒント

### 最低VRAM構成

現在、最低VRAM使用率は以下で達成できます:

- OS: Ubuntu Linux 24
- GPU: 単一のNVIDIA CUDAデバイス（10G、12G）
- システムメモリ: 約50Gのシステムメモリ
- ベースモデル精度: `nf4-bnb`
- オプティマイザ: Lion 8Bit Paged、`bnb-lion8bit-paged`
- 解像度: 1024px
- バッチサイズ: 1、勾配累積ステップなし
- DeepSpeed: 無効/未設定
- PyTorch: 2.5.1
- `--quantize_via=cpu`を使用して、<=16Gカードでの起動時のoutOfMemoryエラーを回避
- `--gradient_checkpointing`を有効化

**注意**: VAE埋め込みとテキストエンコーダ出力の事前キャッシングはより多くのメモリを使用する可能性があり、OOMになる可能性があります。その場合、テキストエンコーダ量子化を有効にできます。VAEタイリングは現時点でSanaでは動作しない可能性があります。ディスク容量が問題になる大規模データセットの場合は、`--vae_cache_disable`を使用してディスクにキャッシュせずにオンラインエンコーディングを実行できます。

4090での速度は1秒あたり約1.4イテレーションでした。

### マスク損失

被写体またはスタイルをトレーニングしていて、一方または他方をマスクしたい場合は、Dreamboothガイドの[マスク損失トレーニング](../DREAMBOOTH.md#masked-loss)セクションを参照してください。

### 量子化

まだ十分にテストされていません。

### 学習率

#### LoRA（--lora_type=standard）

*サポートされていません。*

#### LoKr（--lora_type=lycoris）
- LoKrにはより穏やかな学習率が良い（AdamWで`1e-4`、Lionで`2e-5`）
- 他のアルゴリズムはさらに探索が必要です。
- `is_regularisation_data`の設定はSanaへの影響/効果が不明です（テストされていません）

### 画像アーティファクト

Sanaは画像アーティファクトへの反応が不明です。

一般的なトレーニングアーティファクトが生成されるかどうか、またその原因が何であるかは現時点では不明です。

画像品質の問題が発生した場合は、Githubで問題を開いてください。

### アスペクトバケッティング

このモデルはアスペクトバケット化されたデータへの反応が不明です。実験が役立つでしょう。
