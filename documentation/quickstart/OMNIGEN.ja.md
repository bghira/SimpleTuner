## OmniGen クイックスタート

この例では、OmniGen 用の Lycoris LoKr をトレーニングします。現時点では一般的な T2I 性能改善に焦点を当てており、編集/指示学習ではありません。

### ハードウェア要件

OmniGen は約 3.8B パラメータの比較的コンパクトなモデルです。SDXL VAE を使用しますが、テキストエンコーダは使いません。代わりに OmniGen はネイティブトークン ID を入力として使用し、マルチモーダルモデルのように振る舞います。

トレーニング時のメモリ使用量はまだ不明ですが、バッチサイズ 2 または 3 で 24G カードに十分収まると予想されます。モデルは量子化でき、さらに VRAM を節約できます。

OmniGen は SimpleTuner で学習可能な他のモデルと比べて独特なアーキテクチャです。

- 現在サポートされているのは t2i (text-to-image) 学習のみで、モデル出力が学習プロンプトと入力画像に整合します。
- 画像から画像の学習モードはまだサポートされていませんが、将来対応する可能性があります。
  - このモードでは 2 枚目の画像を入力として提供し、モデルはそれを出力のコンディショニング/参照データとして使用します。
- OmniGen の学習中の損失値は非常に高く、その理由は不明です。


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

### 環境のセットアップ

SimpleTuner を実行するには、設定ファイル、データセットとモデルのディレクトリ、およびデータローダー設定ファイルをセットアップする必要があります。

#### 設定ファイル

実験的なスクリプト `configure.py` を使用すると、インタラクティブなステップバイステップの設定でこのセクションを完全にスキップできる可能性があります。一般的な落とし穴を避けるための安全機能が含まれています。

**注意:** これはデータローダーを設定しません。後で手動で設定する必要があります。

実行するには:

```bash
simpletuner configure
```

> ⚠️ Hugging Face Hub に容易にアクセスできない国にいるユーザーは、使用している `$SHELL` に応じて、`~/.bashrc` または `~/.zshrc` に `HF_ENDPOINT=https://hf-mirror.com` を追加する必要があります。


手動で設定したい場合:

`config/config.json.example` を `config/config.json` にコピー:

```bash
cp config/config.json.example config/config.json
```

そこで、以下の変数を変更する必要があります:

- `model_type` - `lora` に設定します。
- `lora_type` - `lycoris` に設定します。
- `model_family` - `omnigen` に設定します。
- `model_flavour` - `v1` に設定します。
- `output_dir` - チェックポイントと検証画像を保存するディレクトリに設定します。ここではフルパスを使用することをお勧めします。
- `train_batch_size` - 24G カードでフルの gradient checkpointing を使う場合、6 まで上げられます。
- `validation_resolution` - OmniGen のこのチェックポイントは 1024px モデルなので、`1024x1024` か OmniGen の他の対応解像度に設定します。
  - 他の解像度はカンマ区切りで指定できます: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - OmniGen の推論時に普段使う値を設定します。2.5-3.0 程度の低い値はよりリアルになります。
- `validation_num_inference_steps` - 30 前後を使用します。
- `use_ema` - `true` にすると、メインのチェックポイントに加えてより滑らかな結果が得られます。

- `optimizer` - 使い慣れた最適化手法なら何でも良いですが、この例では `adamw_bf16` を使用します。
- `mixed_precision` - 最も効率的な学習のために `bf16` を推奨します (または `no` ですがメモリ消費が増え遅くなります)。
- `gradient_checkpointing` - これを無効化すると最速ですがバッチサイズが制限されます。VRAM 最小化には有効化が必須です。

マルチ GPU ユーザーは、使用 GPU 数の設定に関する情報として [このドキュメント](../OPTIONS.md#environment-configuration-variables) を参照してください。

最終的な config.json は次のようになります:

<details>
<summary>設定例を表示</summary>

```json
{
    "validation_torch_compile": "false",
    "validation_step_interval": 200,
    "validation_seed": 42,
    "validation_resolution": "1024x1024",
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_num_inference_steps": "20",
    "validation_guidance": 2.0,
    "validation_guidance_rescale": "0.0",
    "vae_cache_ondemand": true,
    "vae_batch_size": 1,
    "train_batch_size": 1,
    "tracker_run_name": "eval_loss_test1",
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "output_dir": "output/models-omnigen",
    "optimizer": "adamw_bf16",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "omnigen",
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 10000,
    "max_grad_norm": 0.01,
    "lycoris_config": "config/lycoris_config.json",
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant",
    "lora_type": "lycoris",
    "learning_rate": "4e-5",
    "gradient_checkpointing": "true",
    "grad_clip_method": "value",
    "eval_steps_interval": 100,
    "disable_benchmark": false,
    "data_backend_config": "config/omnigen/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "no_change",
    "aspect_bucket_rounding": 2
}
```
</details>

そしてシンプルな `config/lycoris_config.json`。追加の学習安定性のため `FeedForward` を削除することもできます。

<details>
<summary>設定例を表示</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 16,
    "apply_preset": {
        "target_module": [
            "Attention",
            "FeedForward"
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 16
            },
            "FeedForward": {
                "factor": 8
            }
        }
    }
}
```
</details>

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTuner には、トレーニングの安定性とパフォーマンスを大幅に向上させることができる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが独自の入力を生成できるようにすることで、露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

#### 検証プロンプト

`config/config.json` の "primary validation prompt" (`--validation_prompt`) は、単一の被写体やスタイルに対して学習しているメインの instance_prompt です。加えて、検証時に回す追加プロンプトを含む JSON ファイルを作成できます。

`config/user_prompt_library.json.example` の形式は次の通りです:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

ニックネームは検証ファイル名になるため、短くしてファイルシステムと互換にしてください。

このプロンプトライブラリをトレーナーに指定するには、`config.json` の末尾に新しい行として TRAINER_EXTRA_ARGS へ追加します:
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

多様なプロンプトのセットは、モデルが崩壊していないかを判断するのに役立ちます。この例では `<token>` を被写体名 (instance_prompt) に置き換えてください。

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

> ℹ️ OmniGen はおそらく 122 トークン程度で理解が頭打ちになります。これ以上理解できるかは不明です。

#### CLIPスコアトラッキング

モデルの性能をスコアで評価したい場合は、CLIP スコアの設定と解釈について [このドキュメント](../evaluation/CLIP_SCORES.md) を参照してください。

</details>

# 安定した評価損失

安定した MSE 損失を使用してモデルのパフォーマンスをスコアリングしたい場合は、評価損失の設定と解釈に関する情報について [このドキュメント](../evaluation/EVAL_LOSS.md) を参照してください。

#### 検証プレビュー

SimpleTuner は、Tiny AutoEncoder モデルを使用した生成中の中間検証プレビューのストリーミングをサポートしています。これにより、Webhook コールバックを介してリアルタイムで段階的に生成される検証画像を確認できます。

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

Tiny AutoEncoder のオーバーヘッドを削減するには、`validation_preview_steps` をより高い値 (例: 3 または 5) に設定します。`validation_num_inference_steps=20` と `validation_preview_steps=5` の場合、ステップ 5、10、15、20 でプレビュー画像を受け取ります。

#### Flow schedule shifting

現在 OmniGen は独自のフローマッチング方式を使うようハードコードされており、スケジュールシフトは適用されません。

<!--
Flow-matching models such as OmniGen, Sana, Flux, and SD3 have a property called "shift" that allows us to shift the trained portion of the timestep schedule using a simple decimal value.

##### Auto-shift
A commonly-recommended approach is to follow several recent works and enable resolution-dependent timestep shift, `--flow_schedule_auto_shift` which uses higher shift values for larger images, and lower shift values for smaller images. This results in stable but potentially mediocre training results.

##### Manual specification
_Thanks to General Awareness from Discord for the following examples_

When using a `--flow_schedule_shift` value of 0.1 (a very low value), only the finer details of the image are affected:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

When using a `--flow_schedule_shift` value of 4.0 (a very high value), the large compositional features and potentially colour space of the model becomes impacted:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)
-->

#### データセットの考慮事項

モデルをトレーニングするには、かなりのデータセットが必要です。データセットサイズには制限があり、モデルを効果的にトレーニングするためにデータセットが十分に大きいことを確認する必要があります。最小限のデータセットサイズは `train_batch_size * gradient_accumulation_steps` に加えて `vae_batch_size` よりも大きい必要があります。データセットが小さすぎると使用できません。

> ℹ️ 画像数が少なすぎると **no images detected in dataset** と表示されることがあります。その場合は `repeats` を増やすと解決します。

持っているデータセットに応じて、データセットディレクトリとデータローダー設定ファイルを異なる方法でセットアップする必要があります。この例では、データセットとして [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) を使用します。

`--data_backend_config` (`config/multidatabackend.json`) を作成し、次を含めます:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "pseudo-camera-10k-omnigen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/omnigen/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/omnigen/dreambooth-subject",
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
    "cache_dir_vae": "cache/vae/omnigen/dreambooth-subject-512",
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
    "cache_dir": "cache/text/omnigen",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> caption_strategy のオプションと要件は [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

> ℹ️ 512px と 1024px のデータセットを同時に走らせることはサポートされており、収束の改善につながる可能性があります。

> ℹ️ OmniGen はテキストエンコーダ埋め込みを生成しませんが、現時点では定義が必要です。

私の OmniGen 設定は非常に基本的で、安定評価損失の学習セットを使用した場合は次のようになりました:

<details>
<summary>設定例を表示</summary>

```json
[
    {
        "id": "something-special-to-remember-by",
        "type": "local",
        "instance_data_dir": "/datasets/pseudo-camera-10k/train",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": "filename",
        "cache_dir_vae": "cache/vae/omnigen",
        "vae_cache_clear_each_epoch": false,
        "crop": true,
        "crop_aspect": "square"
    },
    {
        "id": "omnigen-eval",
        "type": "local",
        "dataset_type": "eval",
        "crop": true,
        "crop_aspect": "square",
        "instance_data_dir": "/datasets/test_datasets/squares",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/omnigen-eval",
        "caption_strategy": "filename"
    },
    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/omnigen"
    }
]
```
</details>


次に `datasets` ディレクトリを作成します:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

これにより、約 10k の写真サンプルが `datasets/pseudo-camera-10k` にダウンロードされ、自動的にディレクトリが作成されます。

Dreambooth 画像は `datasets/dreambooth-subject` に配置します。

#### WandB と Huggingface Hub へのログイン

特に `--push_to_hub` や `--report_to=wandb` を使う場合、トレーニング開始前に WandB と HF Hub にログインしておくと良いでしょう。

Git LFS リポジトリに手動でプッシュする場合は、`git config --global credential.helper store` も実行してください。

以下のコマンドを実行します:

```bash
wandb login
```

と

```bash
huggingface-cli login
```

指示に従って両方にログインしてください。

### トレーニングの実行

SimpleTuner ディレクトリから、以下のいずれかで開始できます:

**オプション 1 (推奨 - pip install):**
```bash
pip install simpletuner[cuda]
simpletuner train
```

**オプション 2 (Git clone 方式):**
```bash
simpletuner train
```

**オプション 3 (レガシー方式 - まだ動きます):**
```bash
./train.sh
```

これにより、テキスト埋め込みと VAE 出力のディスクキャッシュが開始されます。

詳細は [dataloader](../DATALOADER.md) と [tutorial](../TUTORIAL.md) を参照してください。

## メモとトラブルシューティングのヒント

### 最低 VRAM 構成

OmniGen の最小 VRAM 構成はまだ不明ですが、以下に近いと予想されます:

- OS: Ubuntu Linux 24
- GPU: 単一の NVIDIA CUDA デバイス (10G, 12G)
- システムメモリ: 約 50G
- ベースモデル精度: `int8-quanto` (または `fp8-torchao`, `int8-torchao` は同様のメモリ特性)
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- 解像度: 1024px
- バッチサイズ: 1、勾配蓄積なし
- DeepSpeed: 無効 / 未設定
- PyTorch: 2.7+
- 起動時の outOfMemory を避けるため `--quantize_via=cpu` を使用
- `--gradient_checkpointing` を有効化
- 小さな LoRA / Lycoris 設定 (例: LoRA rank 1 または Lokr factor 25)

**注意**: VAE 埋め込みとテキストエンコーダ出力の事前キャッシュはさらにメモリを使い OOM する可能性があります。その場合は VAE のタイル/スライスを有効にします。大規模データセットで VAE キャッシュのディスク使用を避けるには `--vae_cache_disable` を使用してください。

AMD 7900XTX + Pytorch 2.7 + ROCm 6.3 で速度は約 3.4 iter/sec でした。

### マスク付き損失

被写体やスタイルをマスクしたい場合は、Dreambooth ガイドの [masked loss training](../DREAMBOOTH.md#masked-loss) を参照してください。

### 量子化

まだ十分に検証されていません。

### 学習率

#### LoRA (--lora_type=standard)

*未対応。*

#### LoKr (--lora_type=lycoris)
- LoKr には穏やかな学習率が良い (`1e-4` with AdamW, `2e-5` with Lion)
- 他のアルゴリズムは要検証。
- `is_regularisation_data` の設定が OmniGen に与える影響は不明 (未検証)

### 画像アーティファクト

OmniGen のアーティファクトへの反応は不明ですが、SDXL VAE を使用するため、細部の制限は同じです。

画質の問題があれば、GitHub で issue を開いてください。

### アスペクトバケッティング

アスペクトバケットデータへの反応は不明です。試験的な検証が役立ちます。

### 高い損失値

OmniGen は損失値が非常に高く、その理由は不明です。損失値は無視し、生成画像の視覚品質に注目することを推奨します。
