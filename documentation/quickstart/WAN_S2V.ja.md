## Wan 2.2 S2V クイックスタート

この例では、Wan 2.2 S2V (Speech-to-Video) LoRA をトレーニングします。S2V モデルは音声入力を条件として動画を生成し、音声駆動の動画生成を可能にします。

### ハードウェア要件

Wan 2.2 S2V **14B** は、大量の GPU メモリを必要とする要求の厳しいモデルです。

#### Speech to Video

14B - https://huggingface.co/tolgacangoz/Wan2.2-S2V-14B-Diffusers
- 解像度: 832x480
- 24G に収まりますが、設定を少し調整する必要があります。

必要なもの:
- **現実的な最小要件**は 24GB、つまり単一の 4090 または A6000 GPU
- **理想的には**複数の 4090、A6000、L40S、またはそれ以上

Apple シリコンシステムは現時点で Wan 2.2 とあまりうまく動作せず、単一のトレーニングステップに約 10 分かかることが予想されます。

### 前提条件

Python がインストールされていることを確認してください。SimpleTuner は 3.10 から 3.12 でうまく動作します。

以下を実行して確認できます:

```bash
python --version
```

Ubuntu に Python 3.12 がインストールされていない場合は、以下を試してください:

```bash
apt -y install python3.13 python3.13-venv
```

#### コンテナイメージの依存関係

Vast、RunPod、TensorDock (およびその他) では、CUDA 12.2-12.8 イメージで CUDA 拡張機能のコンパイルを有効にするために以下が機能します:

```bash
apt -y install nvidia-cuda-toolkit
```

### インストール

pip で SimpleTuner をインストールします:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](/documentation/INSTALL.md)を参照してください。
#### SageAttention 2

SageAttention 2 を使用したい場合は、いくつかの手順に従う必要があります。

> 注: SageAttention は最小限の高速化しか提供せず、あまり効果的ではありません。理由は不明です。4090 でテスト済み。

Python venv 内で以下を実行してください:
```bash
git clone https://github.com/thu-ml/SageAttention
pushd SageAttention
  pip install . --no-build-isolation
popd
```

#### AMD ROCm のフォローアップ手順

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

実験的なスクリプト `configure.py` を使用すると、インタラクティブなステップバイステップの設定によりこのセクション全体をスキップできる可能性があります。一般的な落とし穴を避けるためのいくつかの安全機能が含まれています。

**注:** これはデータローダーを設定しません。データローダーは後で手動で設定する必要があります。

実行するには:

```bash
simpletuner configure
```

> Hugging Face Hub にアクセスしにくい国にいるユーザーは、システムが使用する `$SHELL` に応じて `~/.bashrc` または `~/.zshrc` に `HF_ENDPOINT=https://hf-mirror.com` を追加してください。

### メモリオフロード (オプション)

Wan は SimpleTuner がサポートする中で最も重いモデルの一つです。VRAM の上限に近い場合は、グループオフロードを有効にしてください:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# オプション: オフロードされた重みを RAM ではなくディスクに退避
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- `--group_offload_use_stream` を尊重するのは CUDA デバイスのみです。ROCm/MPS は自動的にフォールバックします。
- CPU メモリがボトルネックでない限り、ディスクステージングはコメントアウトしたままにしてください。
- `--enable_model_cpu_offload` はグループオフロードと排他的です。

### Feed-forward チャンキング (オプション)

勾配チェックポイント中に 14B チェックポイントがまだ OOM になる場合は、Wan の feed-forward レイヤーをチャンク化してください:

```bash
--enable_chunked_feed_forward \
--feed_forward_chunk_size 2 \
```

これは設定ウィザード (`Training -> Memory Optimisation`) の新しいトグルに対応しています。チャンクサイズが小さいほどメモリを節約できますが、各ステップが遅くなります。クイック実験のために環境で `WAN_FEED_FORWARD_CHUNK_SIZE=2` を設定することもできます。


手動で設定する場合:

`config/config.json.example` を `config/config.json` にコピーします:

```bash
cp config/config.json.example config/config.json
```

マルチ GPU ユーザーは、使用する GPU 数の設定については[このドキュメント](/documentation/OPTIONS.md#environment-configuration-variables)を参照してください。

最終的な設定は以下のようになります:

<details>
<summary>設定例を表示</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "attention_mechanism": "sageattention",
  "data_backend_config": "config/wan_s2v/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "offload_during_startup": true,
  "disable_benchmark": false,
  "output_dir": "output/wan_s2v",
  "lora_type": "standard",
  "lycoris_config": "config/wan_s2v/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "wan-s2v-lora",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "wan-s2v-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
  "pretrained_t5_model_name_or_path": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
  "model_family": "wan_s2v",
  "model_flavour": "s2v-14b-2.2",
  "train_batch_size": 2,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "832x480",
  "validation_prompt": "A person speaking with natural gestures",
  "validation_negative_prompt": "blurry, low quality, distorted",
  "validation_guidance": 4.5,
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
  "webhook_config": "config/wan_s2v/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "lora_rank": 16,
  "flow_schedule_shift": 3,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

この設定で特に重要なのは検証設定です。これらがないと、出力の品質があまり良くなりません。

### オプション: CREPA 時間的正則化

Wan S2V でよりスムーズなモーションとアイデンティティドリフトの軽減のために:
- **Training -> Loss functions** で **CREPA** を有効にします。
- **Block Index = 8**、**Weight = 0.5**、**Adjacent Distance = 1**、**Temporal Decay = 1.0** から始めてください。
- デフォルトのエンコーダー (`dinov2_vitg14`、サイズ `518`) はうまく機能します。VRAM を削減する必要がある場合のみ `dinov2_vits14` + `224` に切り替えてください。
- 初回実行時に torch hub 経由で DINOv2 がダウンロードされます。オフラインでトレーニングする場合はキャッシュまたはプリフェッチしてください。
- **Drop VAE Encoder** は完全にキャッシュされた潜在変数からトレーニングする場合のみ有効にしてください。そうでなければピクセルエンコードが引き続き機能するようにオフのままにしてください。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTuner には、トレーニングの安定性とパフォーマンスを大幅に向上させる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが自身の入力を生成することで、露出バイアスを軽減し、出力品質を向上させます。

> これらの機能はトレーニングの計算オーバーヘッドを増加させます。

</details>

### TREAD トレーニング

> **実験的**: TREAD は新しく実装された機能です。機能しますが、最適な設定はまだ探索中です。

[TREAD](/documentation/TREAD.md) (論文) は **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion の略です。トランスフォーマーレイヤーを通じてトークンをインテリジェントにルーティングすることで、Wan S2V トレーニングを高速化できる手法です。高速化はドロップするトークン数に比例します。

#### クイックセットアップ

シンプルで控えめなアプローチのために、これを `config.json` に追加してください:

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

この設定は:
- レイヤー 2 から最後から 2 番目までの間、画像トークンの 50% のみを保持します
- テキストトークンはドロップされません
- 品質への影響を最小限に抑えながら約 25% のトレーニング高速化
- トレーニング品質と収束が改善する可能性があります

#### 重要なポイント

- **限定的なアーキテクチャサポート** - TREAD は Flux と Wan モデル (S2V を含む) にのみ実装されています
- **高解像度で最適** - アテンションの O(n^2) 複雑度により、1024x1024 以上で最大の高速化が得られます
- **マスク損失と互換性あり** - マスクされた領域は自動的に保持されます (ただし高速化は減少します)
- **量子化と連携** - int8/int4/NF4 トレーニングと組み合わせ可能
- **初期損失スパイクを予期** - LoRA/LoKr トレーニング開始時、損失は最初は高くなりますがすぐに修正されます

#### チューニングのヒント

- **控えめ (品質重視)**: `selection_ratio` を 0.1-0.3 で使用
- **積極的 (速度重視)**: `selection_ratio` を 0.3-0.5 で使用し、品質への影響を許容
- **初期/後期レイヤーを避ける**: レイヤー 0-1 または最終レイヤーでルーティングしない
- **LoRA トレーニングの場合**: わずかな速度低下が見られる場合があります - 異なる設定を試してください
- **高解像度 = より良い高速化**: 1024px 以上で最も効果的

詳細な設定オプションとトラブルシューティングについては、[完全な TREAD ドキュメント](/documentation/TREAD.md)を参照してください。


#### 検証プロンプト

`config/config.json` 内には「プライマリ検証プロンプト」があり、これは通常、単一の被写体やスタイルでトレーニングしているメインの instance_prompt です。さらに、検証中に実行する追加のプロンプトを含む JSON ファイルを作成できます。

設定ファイル例 `config/user_prompt_library.json.example` には以下の形式が含まれています:

<details>
<summary>設定例を表示</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

ニックネームは検証のファイル名になるため、短くファイルシステムと互換性のあるものにしてください。

トレーナーをこのプロンプトライブラリに向けるには、`config.json` の末尾に新しい行を追加して TRAINER_EXTRA_ARGS に追加します:
<details>
<summary>設定例を表示</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

> S2V は UMT5 テキストエンコーダーを使用しており、その埋め込みには多くのローカル情報が含まれているため、短いプロンプトではモデルが良い仕事をするのに十分な情報がない可能性があります。より長く、より説明的なプロンプトを使用してください。

#### CLIP スコアトラッキング

現時点では、動画モデルのトレーニングでは有効にしないでください。

# 安定した評価損失

モデルのパフォーマンスをスコアリングするために安定した MSE 損失を使用したい場合は、評価損失の設定と解釈について[このドキュメント](/documentation/evaluation/EVAL_LOSS.md)を参照してください。

#### 検証プレビュー

SimpleTuner は、Tiny AutoEncoder モデルを使用して生成中の中間検証プレビューのストリーミングをサポートしています。これにより、webhook コールバックを介してリアルタイムで検証画像が生成されるのを段階的に確認できます。

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

Tiny AutoEncoder のオーバーヘッドを削減するには、`validation_preview_steps` をより高い値 (例: 3 または 5) に設定してください。`validation_num_inference_steps=20` と `validation_preview_steps=5` の場合、ステップ 5、10、15、20 でプレビュー画像を受け取ります。

#### Flow-matching スケジュールシフト

Flux、Sana、SD3、LTX Video、Wan S2V などのフローマッチングモデルには、単純な小数値を使用してトレーニングされるタイムステップスケジュールの部分をシフトできる `shift` というプロパティがあります。

##### デフォルト
デフォルトでは、スケジュールシフトは適用されず、タイムステップサンプリング分布にシグモイドのベル形状が生成されます。これは `logit_norm` とも呼ばれます。

##### 自動シフト
一般的に推奨されるアプローチは、いくつかの最近の研究に従って解像度依存のタイムステップシフト `--flow_schedule_auto_shift` を有効にすることです。これは大きな画像には高いシフト値を、小さな画像には低いシフト値を使用します。これにより安定したが潜在的に平凡なトレーニング結果が得られます。

##### 手動指定
_以下の例は Discord の General Awareness 氏のご協力によるものです_

> これらの例は Flux Dev を使用して値の動作を示していますが、Wan S2V も非常に類似しているはずです。

`--flow_schedule_shift` 値を 0.1 (非常に低い値) で使用すると、画像の細部のみが影響を受けます:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

`--flow_schedule_shift` 値を 4.0 (非常に高い値) で使用すると、モデルの大きな構成的特徴と潜在的にカラースペースが影響を受けます:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### 量子化モデルトレーニング

Apple および NVIDIA システムでテスト済みで、Hugging Face Optimum-Quanto を使用して精度と VRAM 要件を削減し、わずか 16GB でトレーニングできます。



`config.json` ユーザーの場合:
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

初期の探索中、Wan S2V からの出力品質が悪い場合があり、これはいくつかの理由に帰着します:

- 推論のステップ数が不十分
  - UniPC を使用していない限り、おそらく少なくとも 40 ステップが必要です。UniPC を使用すると数を少し減らせますが、実験が必要です。
- 不正なスケジューラー設定
  - 通常の Euler フローマッチングスケジュールを使用していましたが、Betas 分布が最もうまく機能するようです
  - この設定に触れていなければ、現在は問題ないはずです
- 不正な解像度
  - Wan S2V はトレーニングされた解像度でのみ正しく動作し、うまくいけばラッキーですが、悪い結果になることが一般的です
- 悪い CFG 値
  - 4.0-5.0 前後の値が安全なようです
- 悪いプロンプティング
  - もちろん、動画モデルは神聖なプロンプティングの技術を学ぶために、神秘家チームが山で何ヶ月も禅の修行をする必要があるようです。なぜなら、彼らのデータセットとキャプションスタイルは聖杯のように守られているからです。
  - 要約: 異なるプロンプトを試してください。
- 欠落または不一致の音声
  - S2V は検証に音声入力が必要です - 検証サンプルに対応する音声ファイルがあることを確認してください

これらすべてにもかかわらず、バッチサイズが低すぎたり学習率が高すぎたりしない限り、モデルはお気に入りの推論ツール (すでに良い結果が得られるものがあると仮定して) で正しく実行されます。

#### データセットの考慮事項

S2V トレーニングには、ペアになった動画と音声データが必要です。デフォルトでは SimpleTuner が動画データセットから
音声を auto-split するため、別の音声データセットは必須ではありません。無効化する場合は `audio.auto_split: false` を
設定し、`s2v_datasets` を手動で指定します。

処理とトレーニングに必要な計算量と時間以外に、データセットサイズに関する制限はほとんどありません。

モデルを効果的にトレーニングできる十分な大きさのデータセットを確保する必要がありますが、利用可能な計算量に対して大きすぎないようにする必要があります。

最小データセットサイズは `train_batch_size * gradient_accumulation_steps` であり、`vae_batch_size` より大きい必要があることに注意してください。小さすぎるとデータセットは使用できません。

> サンプル数が少ない場合、**no samples detected in dataset** というメッセージが表示されることがあります - `repeats` 値を増やすことでこの制限を克服できます。

#### 音声データセットのセットアップ

##### 動画からの自動音声抽出 (推奨)

動画にすでに音声トラックが含まれている場合、SimpleTuner は別の音声データセットを必要とせずに自動的に音声を抽出して処理できます。これは最も簡単でデフォルトのアプローチです:

```json
[
  {
    "id": "s2v-videos",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan_s2v/videos",
    "instance_data_dir": "datasets/s2v-videos",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    },
    "audio": {
        "auto_split": true,
        "sample_rate": 16000,
        "channels": 1
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan_s2v",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

音声 auto-split が有効な場合 (デフォルト)、SimpleTuner は以下を行います:
1. 音声データセット設定を自動生成 (`s2v-videos_audio`)
2. メタデータ検出中に各動画から音声を抽出
3. 専用ディレクトリに音声 VAE 潜在変数をキャッシュ
4. `s2v_datasets` を介して音声データセットを自動的にリンク

**音声設定オプション:**
- `audio.auto_split` (bool): 動画からの自動音声抽出を有効化 (デフォルト: true)
- `audio.sample_rate` (int): ターゲットサンプルレート (Hz) (デフォルト: Wav2Vec2 用に 16000)
- `audio.channels` (int): 音声チャンネル数 (デフォルト: モノラル用に 1)
- `audio.allow_zero_audio` (bool): 音声ストリームのない動画に対してゼロ埋め音声を生成 (デフォルト: false)
- `audio.max_duration_seconds` (float): 最大音声長; これより長いファイルはスキップされます
- `audio.duration_interval` (float): バケットグループ化の秒単位での持続時間間隔 (デフォルト: 3.0)
- `audio.truncation_mode` (string): 長い音声の切り詰め方法: "beginning"、"end"、"random" (デフォルト: "beginning")

**注**: `audio.allow_zero_audio: true` が設定されていない限り、音声トラックのない動画は S2V トレーニングで自動的にスキップされます。

##### 手動音声データセット (代替)

別の音声ファイルを使用したい場合、カスタム音声処理が必要な場合、または auto-split を無効化する場合は、
S2V モデルはファイル名で動画ファイルと一致する事前抽出された音声ファイルも使用できます。例:
- `video_001.mp4` には対応する `video_001.wav` (または `.mp3`、`.flac`、`.ogg`、`.m4a`) が必要です

音声ファイルは、`s2v_datasets` バックエンドとして設定する別のディレクトリに配置する必要があります。

##### 動画からの音声抽出 (手動)

動画にすでに音声が含まれている場合は、提供されているスクリプトを使用して抽出します:

```bash
# 音声のみを抽出 (元の動画は変更なし)
python scripts/generate_s2v_audio.py \
    --input-dir datasets/s2v-videos \
    --output-dir datasets/s2v-audio

# 音声を抽出してソース動画から削除 (冗長なデータを避けるために推奨)
python scripts/generate_s2v_audio.py \
    --input-dir datasets/s2v-videos \
    --output-dir datasets/s2v-audio \
    --strip-audio
```

スクリプトは:
- 16kHz モノラル WAV で音声を抽出 (Wav2Vec2 のネイティブサンプルレート)
- ファイル名を自動的に一致させます (例: `video.mp4` -> `video.wav`)
- 音声ストリームのない動画をスキップ
- `ffmpeg` のインストールが必要

##### データセット設定 (手動)

以下を含む `--data_backend_config` (`config/multidatabackend.json`) ドキュメントを作成します:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "s2v-videos",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan_s2v/videos",
    "instance_data_dir": "datasets/s2v-videos",
    "s2v_datasets": ["s2v-audio"],
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
    "id": "s2v-audio",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/s2v-audio",
    "disabled": false
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan_s2v",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> caption_strategy のオプションと要件については [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

S2V データセット設定の重要なポイント:
- 動画データセットの `s2v_datasets` フィールドは音声バックエンドを指します
- 音声ファイルはファイル名のステム (拡張子を除いた部分) で一致します (例: `video_001.mp4` は `video_001.wav` と一致)
- 音声は Wav2Vec2 を使用してオンザフライでエンコードされます (約 600MB VRAM)、キャッシングは不要
- 音声データセットタイプは `audio`

- `video` サブセクションでは、以下のキーを設定できます:
  - `num_frames` (オプション、int) はトレーニングするフレームデータ数です。
    - 15 fps で 75 フレームは 5 秒の動画、標準出力です。これがターゲットであるべきです。
  - `min_frames` (オプション、int) はトレーニングで考慮される動画の最小長を決定します。
    - これは少なくとも `num_frames` と等しくする必要があります。設定しないと等しくなります。
  - `max_frames` (オプション、int) はトレーニングで考慮される動画の最大長を決定します。
  - `bucket_strategy` (オプション、string) は動画がバケットにグループ化される方法を決定します:
    - `aspect_ratio` (デフォルト): 空間アスペクト比のみでグループ化 (例: `1.78`、`0.75`)。
    - `resolution_frames`: `WxH@F` 形式で解像度とフレーム数でグループ化 (例: `832x480@75`)。混合解像度/持続時間データセットに便利。
  - `frame_interval` (オプション、int) `resolution_frames` を使用する場合、フレーム数をこの間隔に丸めます。

次に、動画と音声ファイルを含む `datasets` ディレクトリを作成します:

```bash
mkdir -p datasets/s2v-videos datasets/s2v-audio
# datasets/s2v-videos/ に動画ファイルを配置
# datasets/s2v-audio/ に音声ファイルを配置
```

各動画にファイル名のステムで一致する音声ファイルがあることを確認してください。

#### WandB と Huggingface Hub へのログイン

特に `--push_to_hub` と `--report_to=wandb` を使用している場合は、トレーニングを開始する前に WandB と HF Hub にログインする必要があります。

Git LFS リポジトリに手動でアイテムをプッシュする場合は、`git config --global credential.helper store` も実行する必要があります。

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

SimpleTuner ディレクトリから、トレーニングを開始するいくつかのオプションがあります:

**オプション 1 (推奨 - pip install):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train
```

**オプション 2 (Git clone 方法):**
```bash
simpletuner train
```

**オプション 3 (レガシー方法 - まだ動作します):**
```bash
./train.sh
```

これにより、テキスト埋め込みと VAE 出力のディスクへのキャッシングが開始されます。

詳細については、[データローダー](/documentation/DATALOADER.md)と[チュートリアル](/documentation/TUTORIAL.md)のドキュメントを参照してください。

## 注意とトラブルシューティングのヒント

### 最低 VRAM 設定

Wan S2V は量子化に敏感で、現在 NF4 または INT4 では使用できません。

- OS: Ubuntu Linux 24
- GPU: 単一の NVIDIA CUDA デバイス (24G 推奨)
- システムメモリ: 約 16G のシステムメモリ
- ベースモデル精度: `int8-quanto`
- オプティマイザ: Lion 8Bit Paged、`bnb-lion8bit-paged`
- 解像度: 480px
- バッチサイズ: 1、勾配蓄積ステップゼロ
- DeepSpeed: 無効 / 未設定
- PyTorch: 2.6
- `--gradient_checkpointing` を有効にしないと、何をしても OOM を止められません
- 画像のみでトレーニングするか、動画データセットの `num_frames` を 1 に設定してください

**注**: VAE 埋め込みとテキストエンコーダー出力の事前キャッシングは、より多くのメモリを使用して OOM になる可能性があります。そのため、`--offload_during_startup=true` が基本的に必要です。その場合、テキストエンコーダーの量子化と VAE タイリングを有効にできます。(Wan は現在 VAE タイリング/スライシングをサポートしていません)

### SageAttention

`--attention_mechanism=sageattention` を使用すると、検証時の推論を高速化できます。

**注**: これは最終 VAE デコードステップと互換性がなく、その部分は高速化されません。

### マスク損失

Wan S2V ではこれを使用しないでください。

### 量子化
- バッチサイズによっては、このモデルを 24G でトレーニングするために量子化が必要な場合があります

### 画像アーティファクト
Wan は Euler Betas フローマッチングスケジュールまたは (デフォルトで) UniPC マルチステップソルバー (より強力な予測を行う高次スケジューラー) の使用が必要です。

他の DiT モデルと同様に、以下のこと (など) を行うと、サンプルに四角いグリッドアーティファクトが現れ**始める可能性があります**:
- 低品質データでオーバートレーニング
- 高すぎる学習率を使用
- オーバートレーニング (一般的に)、画像が多すぎる低容量ネットワーク
- アンダートレーニング (も)、画像が少なすぎる高容量ネットワーク
- 変なアスペクト比やトレーニングデータサイズを使用

### アスペクトバケッティング
- 動画は画像と同様にバケット化されます。
- 正方形クロップで長時間トレーニングしても、このモデルにはあまりダメージを与えないでしょう。思い切ってやってください、素晴らしくて信頼性があります。
- 一方、データセットの自然なアスペクトバケットを使用すると、推論時にこれらの形状に過度にバイアスがかかる可能性があります。
  - これは望ましい品質かもしれません。シネマティックなものなどのアスペクト依存のスタイルが他の解像度に過度ににじみ出るのを防ぎます。
  - しかし、多くのアスペクトバケットで均等に結果を改善したい場合は、独自のデメリットがある `crop_aspect=random` を試す必要があるかもしれません。
- 画像ディレクトリデータセットを複数回定義してデータセット設定を混合すると、非常に良い結果と適切に一般化されたモデルが得られています。

### 音声同期

S2V で最良の結果を得るために:
- 音声の長さが動画の長さと一致することを確認
- 音声は内部で 16kHz にリサンプリングされます
- Wav2Vec2 エンコーダーは音声をオンザフライで処理します (約 600MB VRAM オーバーヘッド)
- 音声特徴は動画フレーム数に一致するように補間されます
