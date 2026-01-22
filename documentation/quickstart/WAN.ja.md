## Wan 2.1 クイックスタート

この例では、Sayak Paulの[パブリックドメインDisneyデータセット](https://hf.co/datasets/sayakpaul/video-dataset-disney-organized)を使用してWan 2.1 LoRAをトレーニングします。



https://github.com/user-attachments/assets/51e6cbfd-5c46-407c-9398-5932fa5fa561


### ハードウェア要件

Wan 2.1 **1.3B**はシステムメモリ**または**GPUメモリをそれほど必要としません。**14B**モデルもサポートされていますが、より多くのリソースを必要とします。

現在、WanではImage-to-Video(I2V)トレーニングはサポートされていませんが、T2V LoRAとLycorisはI2Vモデルで実行できます。

#### Text to Video

1.3B - https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- 解像度: 832x480
- Rank-16 LoRAは12G以上を使用(バッチサイズ4)

14B - https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers
- 解像度: 832x480
- 24Gに収まりますが、設定を少し調整する必要があります。

<!--
#### Image to Video
14B (720p) - https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
- 解像度: 1280x720
-->

#### Image to Video (Wan 2.2)

最新のWan 2.2 I2Vチェックポイントは同じトレーニングフローで動作します:

- High stage: https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-Diffusers/tree/main/high_noise_model
- Low stage: https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-Diffusers/tree/main/low_noise_model

このガイドの後半で説明する`model_flavour`と`wan_validation_load_other_stage`設定でトレーニングしたいステージをターゲットできます。

必要なもの:
- **現実的な最低要件**は16GB、または単一の3090またはV100 GPU
- **理想的には**複数の4090、A6000、L40S、またはそれ以上

Wan 2.2チェックポイントを実行する際にtime embeddingレイヤーでshapeミスマッチが発生した場合は、新しい
`wan_force_2_1_time_embedding`フラグを有効にしてください。これにより、transformerがWan 2.1スタイルのtime embeddingsにフォールバックし、
互換性の問題が解決されます。

#### ステージプリセットと検証

- `model_flavour=i2v-14b-2.2-high`はWan 2.2 high-noiseステージをターゲットします。
- `model_flavour=i2v-14b-2.2-low`はlow-noiseステージをターゲットします(同じチェックポイント、異なるサブフォルダ)。
- `wan_validation_load_other_stage=true`を切り替えると、トレーニングしているステージとは反対のステージを検証レンダリング用に読み込みます。
- 標準のWan 2.1 text-to-video実行には、flavourを未設定のままにする(または`t2v-480p-1.3b-2.1`を使用)してください。

Apple siliconシステムではWan 2.1があまりうまく動作せず、単一のトレーニングステップに約10分かかることが予想されます。

### 前提条件

Pythonがインストールされていることを確認してください。SimpleTunerは3.10から3.12で動作します。

以下のコマンドで確認できます:

```bash
python --version
```

Ubuntuにpython 3.12がインストールされていない場合は、以下を試してください:

```bash
apt -y install python3.13 python3.13-venv
```

#### コンテナイメージの依存関係

Vast、RunPod、TensorDock(など)の場合、以下はCUDA 12.2-12.8イメージでCUDA拡張のコンパイルを有効にするために動作します:

```bash
apt -y install nvidia-cuda-toolkit
```

### インストール

pipを使用してSimpleTunerをインストールします:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](../INSTALL.md)を参照してください。
#### SageAttention 2

SageAttention 2を使用したい場合は、いくつかの手順に従う必要があります。

> 注意: SageAttentionは最小限の高速化を提供しますが、あまり効果的ではありません。理由は不明です。4090でテストしました。

Python venv内で以下を実行してください:
```bash
git clone https://github.com/thu-ml/SageAttention
pushd SageAttention
  pip install . --no-build-isolation
popd
```

#### AMD ROCmのフォローアップ手順

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

> ⚠️ Hugging Face Hubに容易にアクセスできない国に居住しているユーザーは、使用している`$SHELL`に応じて、`~/.bashrc`または`~/.zshrc`に`HF_ENDPOINT=https://hf-mirror.com`を追加する必要があります。

### メモリオフロード(オプション)

WanはSimpleTunerがサポートする最も重いモデルの1つです。VRAM上限に近い場合は、グループオフロードを有効にしてください:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# オプション: オフロードされたウェイトをRAMの代わりにディスクに書き出す
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- CUDAデバイスのみが`--group_offload_use_stream`を尊重します。ROCm/MPSは自動的にフォールバックします。
- CPUメモリがボトルネックでない限り、ディスクステージングはコメントアウトしたままにしてください。
- `--enable_model_cpu_offload`はグループオフロードと相互排他的です。

### フィードフォワードチャンキング(オプション)

勾配チェックポイント中に14BチェックポイントがまだOOMする場合は、Wanフィードフォワードレイヤーをチャンクしてください:

```bash
--enable_chunked_feed_forward \
--feed_forward_chunk_size 2 \
```

これは設定ウィザード(`Training → Memory Optimisation`)の新しいトグルと一致します。チャンクサイズが小さいほどメモリを節約できますが、各ステップが遅くなります。クイック実験のために環境に`WAN_FEED_FORWARD_CHUNK_SIZE=2`を設定することもできます。


手動で設定したい場合:

`config/config.json.example`を`config/config.json`にコピーします:

```bash
cp config/config.json.example config/config.json
```

マルチGPUユーザーは、使用するGPUの数を設定する方法について、[このドキュメント](../OPTIONS.md#environment-configuration-variables)を参照してください。

最終的な設定は私のものと同様になります:

<details>
<summary>設定例を表示</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "attention_mechanism": "sageattention",
  "data_backend_config": "config/wan/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "offload_during_startup": true,
  "disable_benchmark": false,
  "output_dir": "output/wan",
  "lora_type": "standard",
  "lycoris_config": "config/wan/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "wan-disney",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "wan-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "model_family": "wan",
  "train_batch_size": 2,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "832x480",
  "validation_prompt": "两只拟人化的猫咪身穿舒适的拳击装备，戴着鲜艳的手套，在聚光灯照射的舞台上激烈对战",
  "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
  "validation_guidance": 5.2,
  "validation_num_inference_steps": 40,
  "validation_num_video_frames": 81,
  "mixed_precision": "bf16",
  "optimizer": "optimi-lion",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.01,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "no_change",
  "vae_batch_size": 1,
  "webhook_config": "config/wan/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "validation_guidance_skip_layers": [9],
  "validation_guidance_skip_layers_start": 0.0,
  "validation_guidance_skip_layers_stop": 1.0,
  "lora_rank": 16,
  "flow_schedule_shift": 3,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

この設定で特に重要なのは検証設定です。これらがないと、出力はあまり良く見えません。

### オプション: CREPA temporal regularizer

Wanでより滑らかなモーションと少ないアイデンティティドリフトを実現するには:
- **Training → Loss functions**で**CREPA**を有効にします。
- **Block Index = 8**、**Weight = 0.5**、**Adjacent Distance = 1**、**Temporal Decay = 1.0**から始めます。
- デフォルトエンコーダー(`dinov2_vitg14`、サイズ`518`)は良好に動作します。VRAMを削減する必要がある場合にのみ`dinov2_vits14` + `224`に切り替えてください。
- 初回実行時にtorch hub経由でDINOv2をダウンロードします。オフラインでトレーニングする場合はキャッシュまたは事前取得してください。
- キャッシュされたlatentsから完全にトレーニングする場合にのみ**Drop VAE Encoder**を有効にしてください。それ以外の場合は、ピクセルエンコードが引き続き機能するようにオフにしておきます。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTunerには、トレーニングの安定性とパフォーマンスを大幅に向上させる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure biasを減らし、トレーニング中にモデルが独自の入力を生成できるようにすることで出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

</details>

### TREADトレーニング

> ⚠️ **実験的**: TREADは新しく実装された機能です。機能的ですが、最適な設定はまだ模索中です。

[TREAD](../TREAD.md)(論文)は、**T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusionの略です。これは、transformerレイヤーを通じてトークンをインテリジェントにルーティングすることで、Fluxトレーニングを加速できる手法です。高速化はドロップするトークン数に比例します。

#### クイックセットアップ

bs=2と480pで約5秒/ステップに到達するためのシンプルで保守的なアプローチとして、`config.json`に以下を追加します(バニラ速度の10秒/ステップから短縮):

<details>
<summary>設定例を表示</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.1,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

この設定は以下を行います:
- レイヤー2から最後から2番目までの間、画像トークンの50%のみを保持
- テキストトークンは決してドロップされません
- トレーニングの高速化は約25%で、品質への影響は最小限
- トレーニング品質と収束を向上させる可能性があります

Wan 1.3Bの場合、全29レイヤーにわたる段階的なルート設定を使用してこのアプローチを強化し、bs=2と480pで約7.7秒/ステップの速度を達成できます:

<details>
<summary>設定例を表示</summary>

```json
{
  "tread_config": {
      "routes": [
          { "selection_ratio": 0.1, "start_layer_idx": 2, "end_layer_idx": 8 },
          { "selection_ratio": 0.25, "start_layer_idx": 9, "end_layer_idx": 11 },
          { "selection_ratio": 0.35, "start_layer_idx": 12, "end_layer_idx": 15 },
          { "selection_ratio": 0.25, "start_layer_idx": 16, "end_layer_idx": 23 },
          { "selection_ratio": 0.1, "start_layer_idx": 24, "end_layer_idx": -2 }
      ]
  }
}
```
</details>

この設定は、意味的知識がそれほど重要でないモデルの内部レイヤーで、より積極的なトークンドロップアウトを使用しようとします。

一部のデータセットでは、より積極的なドロップアウトが許容される場合がありますが、0.5の値はWan 2.1にとってかなり高いです。

#### 重要なポイント

- **限定的なアーキテクチャサポート** - TREADはFluxとWanモデルにのみ実装されています
- **高解像度で最適** - 1024x1024以上でアテンションのO(n²)複雑性により最大の高速化
- **マスク損失と互換性あり** - マスク領域は自動的に保持されます(ただし、これにより高速化が減少します)
- **量子化と動作** - int8/int4/NF4トレーニングと組み合わせることができます
- **初期損失スパイクを予想** - LoRA/LoKrトレーニングを開始する際、損失は最初に高くなりますが、すぐに修正されます

#### チューニングのヒント

- **保守的(品質重視)**: `selection_ratio`を0.1-0.3に設定
- **積極的(速度重視)**: `selection_ratio`を0.3-0.5に設定し、品質への影響を受け入れる
- **初期/後期レイヤーを避ける**: レイヤー0-1または最終レイヤーでルーティングしない
- **LoRAトレーニングの場合**: わずかな速度低下が見られる可能性があります - 異なる設定で実験してください
- **高解像度 = より良い高速化**: 1024px以上で最も有益

#### 既知の動作

- ドロップされるトークンが多いほど(より高い`selection_ratio`)、トレーニングは速くなりますが、初期損失が高くなります
- LoRA/LoKrトレーニングでは、ネットワークが適応するにつれて急速に修正される初期損失スパイクが表示されます
  - より積極的でないトレーニング設定、または内部レイヤーがより高いレベルを持つ複数のルートを使用すると、これが軽減されます
- 一部のLoRA設定ではトレーニングがわずかに遅くなる可能性があります - 最適な設定はまだ模索中です
- RoPE(回転位置エンベッディング)実装は機能していますが、100%正確ではない可能性があります

詳細な設定オプションとトラブルシューティングについては、[完全なTREADドキュメント](../TREAD.md)を参照してください。


#### 検証プロンプト

`config/config.json`内には「プライマリ検証プロンプト」があり、これは通常、トレーニングしている単一のサブジェクトまたはスタイルのメインinstance_promptです。さらに、検証中に実行する追加のプロンプトを含むJSONファイルを作成できます。

例の設定ファイル`config/user_prompt_library.json.example`には以下の形式が含まれています:

<details>
<summary>設定例を表示</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

nicknamesは検証のファイル名なので、短くファイルシステムと互換性のあるものにしてください。

トレーナーにこのプロンプトライブラリを指定するには、`config.json`の末尾に新しい行を追加してTRAINER_EXTRA_ARGSに追加します:
<details>
<summary>設定例を表示</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

多様なプロンプトのセットは、トレーニング中にモデルが崩壊しているかどうかを判断するのに役立ちます。この例では、`<token>`という単語をサブジェクト名(instance_prompt)に置き換える必要があります。

<details>
<summary>設定例を表示</summary>

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
</details>

> ℹ️ Wan 2.1はUMT5テキストエンコーダーのみを使用し、エンベッディングに多くのローカル情報があるため、短いプロンプトではモデルが良い仕事をするのに十分な情報がない可能性があります。より長く、より詳細なプロンプトを使用してください。

#### CLIPスコア追跡

これは現時点では動画モデルのトレーニングには有効にすべきではありません。

# 安定した評価損失

安定したMSE損失を使用してモデルのパフォーマンスをスコアリングしたい場合は、評価損失の設定と解釈に関する情報について[このドキュメント](../evaluation/EVAL_LOSS.md)を参照してください。

#### 検証プレビュー

SimpleTunerは、Tiny AutoEncoderモデルを使用して生成中の中間検証プレビューのストリーミングをサポートしています。これにより、webhookコールバックを介してリアルタイムでステップバイステップで生成される検証画像を確認できます。

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

Tiny AutoEncoderのオーバーヘッドを減らすために、`validation_preview_steps`をより高い値(例: 3または5)に設定してください。`validation_num_inference_steps=20`と`validation_preview_steps=5`の場合、ステップ5、10、15、20でプレビュー画像を受け取ります。

#### フローマッチングスケジュールシフト

Flux、Sana、SD3、LTX Video、Wan 2.1などのフローマッチングモデルには、単純な小数値を使用してタイムステップスケジュールのトレーニング部分をシフトできる`shift`というプロパティがあります。

##### デフォルト
デフォルトでは、スケジュールシフトは適用されず、タイムステップサンプリング分布のシグモイドベル形状が生成されます。これは`logit_norm`として知られています。

##### 自動シフト
一般的に推奨されるアプローチは、最近のいくつかの研究に従い、解像度に依存したタイムステップシフト`--flow_schedule_auto_shift`を有効にすることです。これにより、より大きな画像に対してはより高いシフト値が使用され、より小さな画像に対してはより低いシフト値が使用されます。これにより、安定しているが潜在的に平凡なトレーニング結果が得られます。

##### 手動指定
_以下の例は、DiscordのGeneral Awarenessに感謝します_

> ℹ️ これらの例は、Flux Devを使用して値がどのように機能するかを示していますが、Wan 2.1は非常に似ているはずです。

`--flow_schedule_shift`値0.1(非常に低い値)を使用すると、画像の細かい詳細のみが影響を受けます:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

`--flow_schedule_shift`値4.0(非常に高い値)を使用すると、大きな構成特徴と潜在的にモデルの色空間が影響を受けます:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### 量子化モデルトレーニング

AppleおよびNVIDIAシステムでテストされ、Hugging Face Optimum-Quantoを使用して精度とVRAM要件を削減し、わずか16GBでトレーニングできます。



`config.json`ユーザーの場合:
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

#### 検証設定

SimpleTunerへのWan 2.1の追加を探索する初期段階では、Wan 2.1から恐ろしい悪夢のような出力が出てきましたが、これはいくつかの理由に帰着します:

- 推論のステップ数が不十分
  - UniPCを使用していない限り、おそらく少なくとも40ステップが必要です。UniPCは数を少し減らすことができますが、実験する必要があります。
- 不正なスケジューラー設定
  - 通常のEulerフローマッチングスケジュールを使用していましたが、Betas分布が最も機能するようです
  - この設定に触れていない場合は、今は問題ないはずです
- 不正な解像度
  - Wan 2.1はトレーニングされた解像度でのみ正しく動作し、うまくいけば運が良ければ機能しますが、悪い結果になることが一般的です
- 悪いCFG値
  - Wan 2.1 1.3Bは特にCFG値に敏感なようですが、約4.0-5.0の値は安全なようです
- 悪いプロンプティング
  - プロンプト設計はデータセットやキャプションのスタイルに強く影響されます。試行錯誤しながら調整してください。
  - まずは異なるプロンプトを試してみてください。

これらすべてにもかかわらず、バッチサイズが低すぎる、および/または学習率が高すぎる場合を除き、モデルはお気に入りの推論ツール(既に良い結果が得られるツールがあると仮定)で正しく実行されます。

#### データセットの考慮事項

データセットサイズには、処理とトレーニングにかかる計算と時間以外の制限はほとんどありません。

データセットがモデルを効果的にトレーニングするのに十分な大きさであることを確認する必要がありますが、利用可能な計算量に対して大きすぎないようにしてください。

最小データセットサイズは`train_batch_size * gradient_accumulation_steps`であり、`vae_batch_size`より大きい必要があることに注意してください。データセットが小さすぎると使用できません。

> ℹ️ サンプルが十分に少ない場合、**no samples detected in dataset**というメッセージが表示される可能性があります - `repeats`値を増やすと、この制限を克服できます。

持っているデータセットに応じて、データセットディレクトリとデータローダー設定ファイルを異なる方法でセットアップする必要があります。

この例では、データセットとして[video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)を使用します。

以下を含む`--data_backend_config` (`config/multidatabackend.json`)ドキュメントを作成します:

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
    "cache_dir_vae": "cache/vae/wan/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> caption_strategyオプションと要件については、[DATALOADER.md](../DATALOADER.md#caption_strategy)を参照してください。

- Wan 2.2 image-to-video実行はCLIPコンディショニングキャッシュを作成します。**video**データセットエントリで、専用のバックエンドを指定し、(オプションで)キャッシュパスをオーバーライドします:

<details>
<summary>設定例を表示</summary>

```json
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "conditioning_image_embeds": "disney-conditioning",
    "cache_dir_conditioning_image_embeds": "cache/conditioning_image_embeds/disney-black-and-white"
  }
```
</details>

- コンディショニングバックエンドを一度定義し、必要に応じてデータセット間で再利用します(明確にするために完全なオブジェクトをここに示します):

<details>
<summary>設定例を表示</summary>

```json
  {
    "id": "disney-conditioning",
    "type": "local",
    "dataset_type": "conditioning_image_embeds",
    "cache_dir": "cache/conditioning_image_embeds/disney-conditioning",
    "disabled": false
  }
```
</details>

- `video`サブセクションには、設定できる以下のキーがあります:
  - `num_frames`(オプション、int)は、トレーニングするフレームデータの数です。
    - 15 fpsでは、75フレームは5秒の動画で、標準出力です。これをターゲットにする必要があります。
  - `min_frames`(オプション、int)は、トレーニングに考慮される動画の最小長を決定します。
    - これは少なくとも`num_frames`と等しくなければなりません。設定しない場合は等しくなります。
  - `max_frames`(オプション、int)は、トレーニングに考慮される動画の最大長を決定します。
  - `bucket_strategy`(オプション、string)は、動画がバケットにグループ化される方法を決定します:
    - `aspect_ratio`(デフォルト): 空間的アスペクト比のみでグループ化します(例: `1.78`、`0.75`)。
    - `resolution_frames`: 解像度とフレーム数で`WxH@F`形式でグループ化します(例: `832x480@75`)。混合解像度/期間データセットに便利です。
  - `frame_interval`(オプション、int) `resolution_frames`を使用する場合、フレーム数をこの間隔に丸めます。
<!--  - `is_i2v`(オプション、bool)は、データセットでi2vトレーニングを行うかどうかを決定します。
    - これはWan 2.1ではデフォルトでTrueに設定されています。ただし、無効にできます。
-->

次に、`datasets`ディレクトリを作成します:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset sayakpaul/video-dataset-disney-organized --local-dir=disney-black-and-white
popd
```

これにより、すべてのDisney動画サンプルが自動的に作成される`datasets/disney-black-and-white`ディレクトリにダウンロードされます。

#### WandBとHuggingface Hubへのログイン

トレーニングを開始する前に、特に`--push_to_hub`と`--report_to=wandb`を使用している場合は、WandBとHF Hubにログインする必要があります。

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

**オプション1(推奨 - pipインストール):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
simpletuner train
```

**オプション2(Gitクローン方式):**
```bash
simpletuner train
```

> ℹ️ Wan 2.2をトレーニングする際は、`TRAINER_EXTRA_ARGS`内またはCLI呼び出しで`--model_flavour i2v-14b-2.2-high`(または`low`)を追加し、必要に応じて`--wan_validation_load_other_stage`を追加してください。チェックポイントがtime-embedding shapeミスマッチを報告する場合にのみ`--wan_force_2_1_time_embedding`を追加してください。

**オプション3(レガシー方式 - まだ動作します):**
```bash
./train.sh
```

これにより、テキストエンベッドとVAE出力のディスクへのキャッシングが開始されます。

詳細については、[dataloader](../DATALOADER.md)および[tutorial](../TUTORIAL.md)ドキュメントを参照してください。

## 注意事項とトラブルシューティングのヒント

### 最低VRAM設定

Wan 2.1は量子化に敏感であり、現在NF4またはINT4では使用できません。

- OS: Ubuntu Linux 24
- GPU: 単一のNVIDIA CUDAデバイス(10G、12G)
- システムメモリ: 約12Gのシステムメモリ
- ベースモデル精度: `int8-quanto`
- オプティマイザー: Lion 8Bit Paged、`bnb-lion8bit-paged`
- 解像度: 480px
- バッチサイズ: 1、ゼロ勾配累積ステップ
- DeepSpeed: 無効/未設定
- PyTorch: 2.6
- OOMを防ぐために`--gradient_checkpointing`を有効にしてください
- 画像のみでトレーニングするか、動画データセットの`num_frames`を1に設定してください

**注意**: VAEエンベッドとテキストエンコーダー出力の事前キャッシングはより多くのメモリを使用し、まだOOMする可能性があります。そのため、`--offload_during_startup=true`が基本的に必要です。その場合、テキストエンコーダー量子化とVAEタイリングを有効にできます。(WanはVAEタイリング/スライシングを現在サポートしていません)

速度:
- M3 Max Macbook Proで665.8秒/イテレーション
- NVIDIA 4090でバッチサイズ1で2秒/イテレーション
- NVIDIA 4090でバッチサイズ4で11秒/イテレーション

### SageAttention

`--attention_mechanism=sageattention`を使用する場合、検証時に推論を高速化できます。

**注意**: これは最終的なVAEデコードステップと互換性がなく、その部分は高速化されません。

### マスク損失

これはWan 2.1では使用しないでください。

### 量子化
- このモデルを24Gでトレーニングするために量子化は必要ありません

### 画像アーティファクト
Wanは、Euler Betasフローマッチングスケジュール、または(デフォルトで)UniPCマルチステップソルバー(より強い予測を行う高次スケジューラー)を使用する必要があります。

他のDiTモデルと同様に、以下のようなことを行うと(他にも)、サンプルに正方形のグリッドアーティファクトが現れ**始める可能性があります**:
- 低品質データで過剰トレーニング
- 学習率が高すぎる
- 過剰トレーニング(一般的に)、画像が多すぎる低容量ネットワーク
- 不十分なトレーニング(同様に)、画像が少なすぎる高容量ネットワーク
- 非標準のアスペクト比またはトレーニングデータサイズの使用

### アスペクトバケット化
- 動画は画像のようにバケット化されます。
- 正方形のクロップで長時間トレーニングしても、このモデルはそれほど損傷しません。自由に使ってください、素晴らしく信頼性があります。
- 一方、データセットの自然なアスペクトバケットを使用すると、推論時にこれらの形状が過度にバイアスされる可能性があります。
  - これは望ましい品質である可能性があります。シネマティックなものなどのアスペクト依存スタイルが他の解像度に過度に浸透するのを防ぐためです。
  - ただし、多くのアスペクトバケット全体で結果を均等に改善したい場合は、`crop_aspect=random`を試す必要がありますが、これには独自の欠点があります。
- 画像ディレクトリデータセットを複数回定義することでデータセット設定を混合すると、非常に良い結果と適切に一般化されたモデルが生成されました。

### カスタムファインチューンされたWan 2.1モデルのトレーニング

Hugging Face Hub上の一部のファインチューンされたモデルには完全なディレクトリ構造がなく、特定のオプションを設定する必要があります。

<details>
<summary>設定例を表示</summary>

```json
{
    "model_family": "wan",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_transformer_model_name_or_path": "path/to-the-other-model",
    "pretrained_vae_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_transformer_subfolder": "none",
}
```
</details>

> 注意: `pretrained_transformer_name_or_path`に単一ファイルの`.safetensors`へのパスを提供できます
