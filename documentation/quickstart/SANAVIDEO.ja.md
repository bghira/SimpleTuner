## Sana Video クイックスタート

この例では、Sana Video 2B 480pモデルをトレーニングします。

### ハードウェア要件

Sana VideoはWanオートエンコーダを使用し、デフォルトで480pの81フレームシーケンスを処理します。メモリ使用量は他のビデオモデルと同等であることを予想してください。Gradient Checkpointingを早めに有効にし、VRAMの余裕を確認した後にのみ`train_batch_size`を増やしてください。

### メモリオフロード（オプション）

VRAM制限に近い場合は、設定でグループオフロードを有効にしてください:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# オプション: オフロードされたウェイトをRAMの代わりにディスクに書き出す
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- CUDAユーザーは`--group_offload_use_stream`の恩恵を受けます。他のバックエンドは自動的に無視します。
- システムRAMが限られている場合を除き、`--group_offload_to_disk_path`はスキップしてください。ディスクステージングは遅くなりますが、実行を安定させます。
- グループオフロードを使用する場合は`--enable_model_cpu_offload`を無効にしてください。

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
pip install simpletuner[cuda]
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
- `model_family` - `sanavideo`に設定します。
- `pretrained_model_name_or_path` - `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`に設定します。
- `pretrained_vae_model_name_or_path` - `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`に設定します。
- `output_dir` - チェックポイントと検証ビデオを保存するディレクトリに設定します。ここではフルパスを使用することをお勧めします。
- `train_batch_size` - ビデオトレーニングでは低く始め、VRAM使用量を確認した後にのみ増やしてください。
- `validation_resolution` - Sana Videoは480pモデルとして出荷されています。`832x480`または検証したいアスペクトバケットを使用してください。
- `validation_num_video_frames` - デフォルトのサンプラー長に合わせて`81`に設定します。
- `validation_guidance` - Sana Videoの推論時に選択する通常の値を使用します。
- `validation_num_inference_steps` - 安定した品質のために約50を使用します。
- `framerate` - 省略した場合、Sana Videoはデフォルトで16 fpsになります。データセットに合わせて設定してください。

- `optimizer` - 慣れ親しんだオプティマイザを使用できますが、この例では`optimi-adamw`を使用します。
- `mixed_precision` - 最も効率的なトレーニング設定には`bf16`に設定することをお勧めします。または`no`（ただし、より多くのメモリを消費し、遅くなります）。
- `gradient_checkpointing` - VRAM使用量を制御するためにこれを有効にします。
- `use_ema` - `true`に設定すると、メインのトレーニング済みチェックポイントと一緒により滑らかな結果を得るのに大いに役立ちます。

マルチGPUユーザーは、使用するGPUの数を設定する方法について[このドキュメント](../OPTIONS.md#environment-configuration-variables)を参照してください。

最終的な設定は以下のようになるはずです:

<details>
<summary>設定例を表示</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "data_backend_config": "config/sanavideo/multidatabackend.json",
  "seed": 42,
  "output_dir": "output/sanavideo",
  "max_train_steps": 400000,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "tracker_project_name": "video-training",
  "tracker_run_name": "sanavideo-2b-480p",
  "report_to": "wandb",
  "model_type": "full",
  "pretrained_model_name_or_path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
  "pretrained_vae_model_name_or_path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
  "model_family": "sanavideo",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 200,
  "validation_resolution": "832x480",
  "validation_negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "validation_guidance": 6.0,
  "validation_num_inference_steps": 50,
  "validation_num_video_frames": 81,
  "validation_prompt": "A short video of a small, fluffy animal exploring a sunny room with soft window light and gentle camera motion.",
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.0,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "bf16",
  "vae_batch_size": 1,
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "framerate": 16,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

### オプション: CREPA temporal regularizer

ビデオでちらつきやサブジェクトのドリフトが見られる場合は、CREPAを有効にしてください:
- **Training → Loss functions**で**CREPA**をオンにします。
- 推奨デフォルト: **Block Index = 10**、**Weight = 0.5**、**Adjacent Distance = 1**、**Temporal Decay = 1.0**。
- VRAMを節約するためにより小さいオプション（`dinov2_vits14` + `224`）が必要でない限り、デフォルトエンコーダー（`dinov2_vitg14`、サイズ`518`）を使用してください。
- 初回実行時にtorch hub経由でDINOv2をダウンロードします。オフラインの場合はキャッシュまたは事前取得してください。
- キャッシュされたlatentsから完全にトレーニングする場合にのみ**Drop VAE Encoder**を切り替えてください。まだピクセルをエンコードしている場合はオフのままにしてください。

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
    "anime_<token>": "a breathtaking anime-style video featuring <token>, capturing her essence with vibrant colors, dynamic motion, and expressive storytelling",
    "chef_<token>": "a high-quality, detailed video of <token> as a sous-chef, immersed in the art of culinary creation with captivating close-ups and engaging sequences",
    "just_<token>": "a lifelike and intimate video portrait of <token>, showcasing her unique personality and charm through nuanced movement and expression",
    "cinematic_<token>": "a cinematic, visually stunning video of <token>, emphasizing her dramatic and captivating presence through fluid camera movements and atmospheric effects",
    "elegant_<token>": "an elegant and timeless video portrait of <token>, exuding grace and sophistication with smooth transitions and refined visuals",
    "adventurous_<token>": "a dynamic and adventurous video featuring <token>, captured in an exciting, action-filled sequence that highlights her energy and spirit",
    "mysterious_<token>": "a mysterious and enigmatic video portrait of <token>, shrouded in shadows and intrigue with a narrative that unfolds in subtle, cinematic layers",
    "vintage_<token>": "a vintage-style video of <token>, evoking the charm and nostalgia of a bygone era through sepia tones and period-inspired visual storytelling",
    "artistic_<token>": "an artistic and abstract video representation of <token>, blending creativity with visual storytelling through experimental techniques and fluid visuals",
    "futuristic_<token>": "a futuristic and cutting-edge video portrayal of <token>, set against a backdrop of advanced technology with sleek, high-tech visuals",
    "woman": "a beautifully crafted video portrait of a woman, highlighting her natural beauty and unique features through elegant motion and storytelling",
    "man": "a powerful and striking video portrait of a man, capturing his strength and character with dynamic sequences and compelling visuals",
    "boy": "a playful and spirited video portrait of a boy, capturing youthful energy and innocence through lively scenes and engaging motion",
    "girl": "a charming and vibrant video portrait of a girl, emphasizing her bright personality and joy with colorful visuals and fluid movement",
    "family": "a heartwarming and cohesive family video, showcasing the bonds and connections between loved ones through intimate moments and shared experiences"
}
```

> ℹ️ Sana Videoはフローマッチングモデルです。短いプロンプトでは、モデルが良い仕事をするのに十分な情報がない可能性があります。できるだけ説明的なプロンプトを使用してください。

#### CLIPスコアトラッキング

現時点では、ビデオモデルトレーニングではこれを有効にすべきではありません。

</details>

# 安定した評価損失

モデルのパフォーマンスをスコアリングするために安定したMSE損失を使用したい場合は、評価損失の設定と解釈に関する情報について[このドキュメント](../evaluation/EVAL_LOSS.md)を参照してください。

#### 検証プレビュー

SimpleTunerは、Tiny AutoEncoderモデルを使用した生成中の中間検証プレビューのストリーミングをサポートしています。これにより、Webhookコールバックを介してリアルタイムで段階的に生成される検証ビデオを確認できます。

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

Tiny AutoEncoderのオーバーヘッドを削減するには、`validation_preview_steps`をより高い値（3または5など）に設定します。`validation_num_inference_steps=20`および`validation_preview_steps=5`の場合、ステップ5、10、15、20でプレビューフレームを受け取ります。

#### フローマッチングスケジュール

Sana Videoはチェックポイントからの正規フローマッチングスケジュールを使用します。ユーザー提供のシフトオーバーライドは無視されます。このモデルでは`flow_schedule_shift`と`flow_schedule_auto_shift`を未設定のままにしてください。

#### 量子化モデルトレーニング

精度オプション（bf16、int8、fp8）は設定で利用可能です。ハードウェアに合わせて調整し、不安定性が発生した場合はより高い精度にフォールバックしてください。

#### データセットの考慮事項

データセットサイズには、処理とトレーニングにかかる計算と時間以外の制限はほとんどありません。

データセットがモデルを効果的にトレーニングするのに十分な大きさであることを確認する必要がありますが、利用可能な計算量に対して大きすぎないようにしてください。

最小データセットサイズは`train_batch_size * gradient_accumulation_steps`であり、`vae_batch_size`より多い必要があることに注意してください。データセットが小さすぎると使用できません。

> ℹ️ サンプルが十分に少ない場合、**no samples detected in dataset**というメッセージが表示される可能性があります - `repeats`値を増やすとこの制限を克服できます。

持っているデータセットに応じて、データセットディレクトリとデータローダー設定ファイルを異なる方法でセットアップする必要があります。

この例では、データセットとして[video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)を使用します。

以下を含む`--data_backend_config`（`config/multidatabackend.json`）ドキュメントを作成します:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/sanavideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 81,
        "min_frames": 81,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sanavideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> caption_strategyオプションと要件については、[DATALOADER.md](../DATALOADER.md#caption_strategy)を参照してください。

- `video`サブセクションには、設定できる以下のキーがあります:
  - `num_frames`（オプション、int）は、トレーニングするデータのフレーム数です。
  - `min_frames`（オプション、int）は、トレーニングに考慮されるビデオの最小長を決定します。
  - `max_frames`（オプション、int）は、トレーニングに考慮されるビデオの最大長を決定します。
  - `is_i2v`（オプション、bool）は、データセットでi2vトレーニングを行うかどうかを決定します。
  - `bucket_strategy`（オプション、string）は、ビデオがバケットにグループ化される方法を決定します:
    - `aspect_ratio`（デフォルト）: 空間的アスペクト比のみでグループ化（例: `1.78`、`0.75`）。
    - `resolution_frames`: 解像度とフレーム数で`WxH@F`形式でグループ化（例: `832x480@81`）。混合解像度/期間データセットに便利です。
  - `frame_interval`（オプション、int）`resolution_frames`を使用する場合、フレーム数をこの間隔に丸めます。

次に、`datasets`ディレクトリを作成します:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset sayakpaul/video-dataset-disney-organized --local-dir=disney-black-and-white
popd
```

これにより、すべてのDisneyビデオサンプルが`datasets/disney-black-and-white`ディレクトリにダウンロードされ、自動的に作成されます。

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
pip install simpletuner[cuda]
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

### 検証デフォルト

- 検証設定が提供されていない場合、Sana Videoはデフォルトで81フレームと16 fpsになります。
- Wanオートエンコーダパスはベースモデルパスと一致する必要があります。ロード時エラーを回避するためにそれらを揃えておいてください。

### マスク損失

被写体またはスタイルをトレーニングしていて、一方または他方をマスクしたい場合は、Dreamboothガイドの[マスク損失トレーニング](../DREAMBOOTH.md#masked-loss)セクションを参照してください。
