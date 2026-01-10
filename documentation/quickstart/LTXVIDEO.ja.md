## LTX Video クイックスタート

この例では、Sayak Paulの[パブリックドメインDisneyデータセット](https://hf.co/datasets/sayakpaul/video-dataset-disney-organized)を使用してLTX-Video LoRAをトレーニングします。

### ハードウェア要件

LTXはシステムメモリ**または**GPUメモリをそれほど必要としません。

rank-16 LoRA(MLP、プロジェクション、マルチモーダルブロック)のすべてのコンポーネントをトレーニングする場合、M3 Macで12G強を使用します(バッチサイズ4)。

必要なもの:
- **現実的な最小要件**は16GB、または単一の3090またはV100 GPU
- **理想的には**複数の4090、A6000、L40S、またはそれ以上

Apple siliconシステムはMPSバックエンドの制限により解像度が低くなりますが、LTXで良好に動作します。

### メモリオフロード(オプション)

VRAM上限に近い場合は、設定でグループオフロードを有効にしてください:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# オプション: オフロードされたウェイトをRAMの代わりにディスクに書き出す
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- CUDAユーザーは`--group_offload_use_stream`の恩恵を受けます。他のバックエンドは自動的に無視します。
- システムRAMが64 GB未満でない限り、`--group_offload_to_disk_path`はスキップしてください。ディスクステージングは遅くなりますが、実行を安定させます。
- グループオフロードを使用する場合は、`--enable_model_cpu_offload`を無効にしてください。

### 前提条件

Pythonがインストールされていることを確認してください。SimpleTunerは3.10から3.12で動作します。

以下のコマンドで確認できます:

```bash
python --version
```

Ubuntuにpython 3.12がインストールされていない場合は、以下を試してください:

```bash
apt -y install python3.12 python3.12-venv
```

#### コンテナイメージの依存関係

Vast、RunPod、TensorDock(など)の場合、CUDA 12.2-12.8イメージでCUDA拡張のコンパイルを有効にするには以下が機能します:

```bash
apt -y install nvidia-cuda-toolkit
```

### インストール

pip経由でSimpleTunerをインストール:

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

実験的なスクリプト`configure.py`により、インタラクティブなステップバイステップの設定を通じて、このセクション全体をスキップできる可能性があります。一般的な落とし穴を回避するのに役立つセーフティ機能が含まれています。

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
- `model_family` - `ltxvideo`に設定します。
- `pretrained_model_name_or_path` - `Lightricks/LTX-Video-0.9.5`に設定します。
- `pretrained_vae_model_name_or_path` - `Lightricks/LTX-Video-0.9.5`に設定します。
- `output_dir` - チェックポイントと検証画像を保存するディレクトリに設定します。フルパスの使用を推奨します。
- `train_batch_size` - 安定性を高めるために増やすことができますが、開始時は4で問題なく動作します
- `validation_resolution` - LTXを使用して通常動画を生成する解像度(`768x512`)に設定します
  - カンマで区切って複数の解像度を指定できます: `1280x768,768x512`
- `validation_guidance` - LTXの推論時に選択する通常の値を使用します。
- `validation_num_inference_steps` - 時間を節約しながらまともな品質を確認するには、25前後を使用します。
- `--lora_rank=4` トレーニングされるLoRAのサイズを大幅に削減したい場合。VRAM使用量を削減しながら学習容量を減らすのに役立ちます。

- `gradient_accumulation_steps` - このオプションにより、更新ステップが複数のステップにわたって蓄積されます。
  - これによりトレーニングランタイムが線形に増加し、値2ではトレーニングが半分の速度で実行され、2倍の時間がかかります。
- `optimizer` - 初心者にはadamw_bf16が推奨されますが、optimi-lionやoptimi-stableadamwも良い選択肢です。
- `mixed_precision` - 初心者は`bf16`のままにしておく必要があります
- `gradient_checkpointing` - ほぼすべての状況、すべてのデバイスでtrueに設定します
- `gradient_checkpointing_interval` - LTX Videoではまだサポートされていないため、設定から削除してください。

マルチGPUユーザーは、使用するGPU数の設定について[このドキュメント](../OPTIONS.md#environment-configuration-variables)を参照してください。

最終的な設定は以下のようになります:

<details>
<summary>設定例を表示</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "data_backend_config": "config/ltxvideo/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "disable_benchmark": false,
  "offload_during_startup": true,
  "output_dir": "output/ltxvideo",
  "lora_type": "lycoris",
  "lycoris_config": "config/ltxvideo/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "ltxvideo-disney",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "ltxvideo-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5",
  "model_family": "ltxvideo",
  "train_batch_size": 8,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 800,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "768x512",
  "validation_negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "validation_guidance": 3.0,
  "validation_num_inference_steps": 40,
  "validation_prompt": "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day. And then the camera switch to a inding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region.",
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.0,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "fp8-torchao",
  "vae_batch_size": 1,
  "webhook_config": "config/ltxvideo/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "lora_rank": 128,
  "flow_schedule_shift": 1,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

### オプション: CREPAテンポラルレギュラライザー

LTX実行でフリッカーやアイデンティティドリフトが発生する場合は、CREPA(クロスフレームアライメント)を試してください:
- WebUIで、**Training → Loss functions**に移動し、**CREPA**を有効にします。
- **Block Index = 8**、**Weight = 0.5**、**Adjacent Distance = 1**、**Temporal Decay = 1.0**から始めます。
- デフォルトのビジョンエンコーダー(`dinov2_vitg14`、サイズ`518`)のままにしてください。VRAMを下げる必要がある場合のみ`dinov2_vits14` + `224`に切り替えてください。
- 初回はDINOv2ウェイトを取得するためにインターネット(またはキャッシュされたtorch hub)が必要です。
- オプション: キャッシュされた潜在変数のみからトレーニングする場合は、メモリを節約するために**Drop VAE Encoder**を有効にします。新しい動画をエンコードする必要がある場合はオフのままにしてください。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTunerには、トレーニングの安定性とパフォーマンスを大幅に向上させることができる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが独自の入力を生成できるようにすることで、露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

#### 検証プロンプト

`config/config.json`内には「プライマリ検証プロンプト」があり、これは通常、単一の被写体またはスタイルに対してトレーニングしているメインのinstance_promptです。さらに、検証中に実行する追加のプロンプトを含むJSONファイルを作成できます。

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

多様なプロンプトのセットは、モデルがトレーニング中に崩壊しているかどうかを判断するのに役立ちます。この例では、`<token>`という単語を被写体名(instance_prompt)に置き換える必要があります。

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

> ℹ️ LTX VideoはT5 XXLベースのフローマッチングモデルです。短いプロンプトにはモデルが良い仕事をするのに十分な情報がない可能性があります。より長く、より詳細なプロンプトを使用してください。

#### CLIPスコアトラッキング

現時点では動画モデルのトレーニングには有効にしないでください。

</details>

# 安定した評価損失

安定したMSE損失を使用してモデルのパフォーマンスをスコアリングしたい場合は、評価損失の設定と解釈に関する情報について[このドキュメント](../evaluation/EVAL_LOSS.md)を参照してください。

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

Tiny AutoEncoderのオーバーヘッドを削減するには、`validation_preview_steps`をより高い値(例: 3または5)に設定します。`validation_num_inference_steps=20`および`validation_preview_steps=5`の場合、ステップ5、10、15、20でプレビュー画像を受け取ります。

#### フローマッチングスケジュールシフト

Flux、Sana、SD3、LTX Videoなどのフローマッチングモデルには、単純な10進数値を使用してタイムステップスケジュールのトレーニング部分をシフトできる`shift`というプロパティがあります。

##### デフォルト
デフォルトでは、スケジュールシフトは適用されず、タイムステップサンプリング分布にシグモイドベル形が生じます。これは`logit_norm`として知られています。

##### 自動シフト
一般的に推奨されるアプローチは、最近のいくつかの研究に従って、解像度依存のタイムステップシフト`--flow_schedule_auto_shift`を有効にすることです。これは、大きな画像にはより高いシフト値を使用し、小さな画像にはより低いシフト値を使用します。これにより安定したトレーニング結果が得られますが、潜在的に平凡な結果になる可能性があります。

##### 手動指定
_DiscordのGeneral Awarenessによる以下の例に感謝_

> ℹ️ これらの例はFlux Devを使用して値がどのように機能するかを示していますが、LTX Videoは非常に類似しているはずです。

`--flow_schedule_shift`値0.1(非常に低い値)を使用すると、画像の細かいディテールのみが影響を受けます:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

`--flow_schedule_shift`値4.0(非常に高い値)を使用すると、大きな構成的特徴とモデルの色空間が潜在的に影響を受けます:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### 量子化モデルトレーニング

AppleおよびNVIDIAシステムでテスト済み、Hugging Face Optimum-Quantoを使用して精度とVRAM要件を削減し、わずか16GBでトレーニングできます。



`config.json`ユーザー向け:
<details>
<summary>設定例を表示</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

#### データセットの考慮事項

データセットサイズには、処理とトレーニングにかかる計算量と時間以外にほとんど制限はありません。

データセットがモデルを効果的にトレーニングするのに十分な大きさであることを確認する必要がありますが、利用可能な計算量に対して大きすぎないようにしてください。

最小データセットサイズは`train_batch_size * gradient_accumulation_steps`であり、`vae_batch_size`より大きい必要があることに注意してください。データセットが小さすぎると使用できません。

> ℹ️ サンプルが十分に少ない場合、**no samples detected in dataset**というメッセージが表示される可能性があります - `repeats`値を増やすことでこの制限を克服できます。

持っているデータセットに応じて、データセットディレクトリとデータローダー設定ファイルを異なる方法でセットアップする必要があります。

この例では、データセットとして[video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)を使用します。

以下を含む`--data_backend_config`(`config/multidatabackend.json`)ドキュメントを作成します:

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
    "cache_dir_vae": "cache/vae/ltxvideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 125,
        "min_frames": 125,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> caption_strategyのオプションと要件については、[DATALOADER.md](../DATALOADER.md#caption_strategy)を参照してください。

- `video`サブセクションには、設定できる以下のキーがあります:
  - `num_frames`(オプション、int)はトレーニングするフレームデータの数です。
    - 25 fpsでは、125フレームは5秒の動画で、標準出力です。これをターゲットにする必要があります。
  - `min_frames`(オプション、int)はトレーニングに考慮される動画の最小長を決定します。
    - これは少なくとも`num_frames`と等しくなければなりません。設定しない場合は等しくなります。
  - `max_frames`(オプション、int)はトレーニングに考慮される動画の最大長を決定します。
  - `is_i2v`(オプション、bool)はデータセットでi2vトレーニングを行うかどうかを決定します。
    - LTXではデフォルトでTrueに設定されています。ただし、無効にすることもできます。
  - `bucket_strategy`(オプション、string)は動画がバケットにグループ化される方法を決定します:
    - `aspect_ratio`(デフォルト): 空間的アスペクト比のみでグループ化(例: `1.78`、`0.75`)。
    - `resolution_frames`: 解像度とフレーム数で`WxH@F`形式でグループ化(例: `768x512@125`)。混合解像度/期間データセットに便利。
  - `frame_interval`(オプション、int)`resolution_frames`を使用する場合、フレーム数をこの間隔に丸めます。モデルの必要なフレーム数ファクターに設定してください。

次に、`datasets`ディレクトリを作成します:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset sayakpaul/video-dataset-disney-organized --local-dir=disney-black-and-white
popd
```

これにより、すべてのDisney動画サンプルが`datasets/disney-black-and-white`ディレクトリにダウンロードされ、自動的に作成されます。

#### WandBとHuggingface Hubへのログイン

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

これにより、テキストエンベッドとVAE出力のディスクへのキャッシングが開始されます。

詳細については、[データローダー](../DATALOADER.md)および[チュートリアル](../TUTORIAL.md)ドキュメントを参照してください。

## 注意事項とトラブルシューティングのヒント

### 最低VRAM設定

他のモデルと同様に、以下で最低VRAM使用率を達成できる可能性があります:

- OS: Ubuntu Linux 24
- GPU: 単一のNVIDIA CUDAデバイス(10G、12G)
- システムメモリ: 約11Gのシステムメモリ
- ベースモデル精度: `nf4-bnb`
- オプティマイザ: Lion 8Bit Paged、`bnb-lion8bit-paged`
- 解像度: 480px
- バッチサイズ: 1、勾配累積ステップなし
- DeepSpeed: 無効/未設定
- PyTorch: 2.6
- `--gradient_checkpointing`を必ず有効にしてください。そうしないと、何をしてもOOMが止まりません

**注意**: VAEエンベッドとテキストエンコーダー出力の事前キャッシングはより多くのメモリを使用する可能性があり、OOMになる可能性があります。その場合、テキストエンコーダー量子化とVAEタイリングを有効にできます。これらのオプションを超えて、`--offload_during_startup=true`がVAEとテキストエンコーダーのメモリ使用の競合を回避するのに役立ちます。

M3 Max Macbook Proでの速度は約0.8イテレーション/秒でした。

### SageAttention

`--attention_mechanism=sageattention`を使用すると、検証時の推論を高速化できます。

**注意**: これはすべてのモデル設定と互換性があるわけではありませんが、試してみる価値があります。

### NF4量子化トレーニング

簡単に言うと、NF4はモデルの4ビット的な表現であり、トレーニングには対処すべき深刻な安定性の懸念があります。

初期テストでは、以下が当てはまります:
- Lionオプティマイザはモデル崩壊を引き起こしますが、VRAMを最も使用しません。AdamWバリアントはそれを維持するのに役立ちます。bnb-adamw8bit、adamw_bf16は素晴らしい選択肢です
  - AdEMAMixはうまくいきませんでしたが、設定は探索されていません
- `--max_grad_norm=0.01`は、モデルが短時間で大きく変化するのを防ぐことでモデルの破損をさらに減らすのに役立ちます
- NF4、AdamW8bit、およびより高いバッチサイズはすべて、トレーニングに費やす時間やVRAMの使用を犠牲にして、安定性の問題を克服するのに役立ちます
- 解像度を上げるとトレーニングが非常に遅くなり、モデルを損なう可能性があります
- 動画の長さを増やすとメモリ消費も増加します。これを克服するには`num_frames`を減らしてください。
- int8またはbf16でトレーニングするのが難しいものは、NF4ではさらに難しくなります
- SageAttentionなどのオプションとの互換性が低くなります

NF4はtorch.compileで動作しないため、速度に関して得られるものが得られるものです。

VRAMが問題でない場合、torch.compileを使用したint8が最良で最速のオプションです。

### マスク損失

LTX Videoではこれを使用しないでください。


### 量子化
- このモデルをトレーニングするのに量子化は必要ありません

### 画像アーティファクト
他のDiTモデルと同様に、これらのこと(その他)を行うと、サンプルに正方形グリッドアーティファクトが**現れ始める可能性があります**:
- 低品質データでオーバートレーニング
- 学習率が高すぎる使用
- オーバートレーニング(一般的に)、画像が多すぎる低容量ネットワーク
- アンダートレーニング(また)、画像が少なすぎる高容量ネットワーク
- 非標準のアスペクト比またはトレーニングデータサイズの使用

### アスペクトバケッティング
- 動画は画像と同様にバケット化されます。
- 正方形クロップで長時間トレーニングしても、このモデルを過度に損傷することはおそらくありません。どんどん試してください。素晴らしくて信頼性があります。
- 一方、データセットの自然なアスペクトバケットを使用すると、推論時にこれらの形状を過度にバイアスする可能性があります。
  - これは望ましい品質である可能性があります。シネマティックなもののようなアスペクト依存のスタイルが他の解像度に過度にブリードするのを防ぐためです。
  - ただし、多くのアスペクトバケットで均等に結果を改善したい場合は、`crop_aspect=random`を試す必要があるかもしれません。これには独自の欠点があります。
- 画像ディレクトリデータセットを複数回定義することでデータセット構成を混合すると、本当に良い結果とうまく一般化されたモデルが生成されました。

### カスタムファインチューニングされたLTXモデルのトレーニング

Hugging Face Hub上の一部のファインチューニングされたモデルには完全なディレクトリ構造がないため、特定のオプションを設定する必要があります。

<details>
<summary>設定例を表示</summary>

```json
{
    "model_family": "ltxvideo",
    "pretrained_model_name_or_path": "Lightricks/LTX-Video",
    "pretrained_transformer_model_name_or_path": "path/to-the-other-model",
    "pretrained_vae_model_name_or_path": "Lightricks/LTX-Video",
    "pretrained_transformer_subfolder": "none",
}
```
</details>

## クレジット

[finetrainers](https://github.com/a-r-r-o-w/finetrainers)プロジェクトとDiffusersチーム。
- 元々SimpleTunerからいくつかの設計コンセプトを使用
- 現在は動画トレーニングを簡単に実装するための洞察とコードを提供
