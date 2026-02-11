# Z-Image [base / turbo] クイックスタート

この例では Z-Image Turbo LoRA をトレーニングします。Z-Image は 6B のフローマッチング Transformer（Flux の約半分）で、base と turbo のフレーバーがあります。Turbo はアシスタントアダプタを要求しますが、SimpleTuner が自動的にロードできます。

## ハードウェア要件

Z-Image は Flux よりメモリが少ないものの、強力な GPU が有利です。rank-16 LoRA の全コンポーネント（MLP、projection、Transformer ブロック）を学習する場合、典型的な使用量は以下です:

- ベースモデル非量子化: ~32–40G VRAM
- int8 + bf16 で量子化: ~16–24G VRAM
- NF4 + bf16 で量子化: ~10–12G VRAM

さらに Ramtorch と group offload を使えば VRAM 使用をさらに下げられます。マルチ GPU ユーザーは FSDP2 で小さな GPU を多数使うことも可能です。

必要なもの:

- **絶対最低**: 単一の **3080 10G**（強い量子化/オフロードが必須）
- **現実的な最低**: 単一の 3090/4090 または V100/A6000
- **理想**: 複数の 4090、A6000、L40S 以上

Apple GPU は学習に推奨されません。

### メモリオフロード（オプション）

Transformer 重みがボトルネックの場合、グループオフロードで VRAM を大幅に削減できます。`TRAINER_EXTRA_ARGS`（または WebUI の Hardware ページ）に以下を追加します:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- ストリームは CUDA のみ有効で、ROCm/MPS/CPU では自動的に無効になります。
- 他の CPU オフロード戦略と **併用しない** でください。
- Group offload は Quanto 量子化と互換性がありません。
- ディスクへオフロードする場合は高速なローカル SSD/NVMe を推奨します。

## 前提条件

Python がインストールされていることを確認してください。SimpleTuner は 3.10 から 3.12 でうまく動作します。

以下を実行して確認できます:

```bash
python --version
```

Ubuntu に Python 3.12 がインストールされていない場合は、以下を試してください:

```bash
apt -y install python3.13 python3.13-venv
```

### コンテナイメージの依存関係

Vast、RunPod、TensorDock（など）の場合、CUDA 12.x イメージで CUDA 拡張のコンパイルを有効にするには以下が機能します:

```bash
apt -y install nvidia-cuda-toolkit-12-8
```

## インストール

pip で SimpleTuner をインストールします:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](../INSTALL.md)を参照してください。

### AMD ROCm フォローアップ手順

AMD MI300X を使用可能にするには、以下を実行する必要があります:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## 環境のセットアップ

### Web インターフェイス方式

SimpleTuner WebUI を使うと簡単に設定できます。サーバを起動するには:

```bash
simpletuner server
```

これによりデフォルトでポート 8001 に Web サーバが起動し、http://localhost:8001 でアクセスできます。

### 手動 / コマンドライン方式

コマンドラインで実行する場合は、設定ファイル、データセットとモデルのディレクトリ、データローダー設定ファイルが必要です。

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

- `model_type` - `lora` に設定します。
- `model_family` - `z-image` に設定します。
- `model_flavour` - `turbo` に設定（v2 アシスタントは `turbo-ostris-v2`）。base は現在未公開チェックポイントを指します。
- `pretrained_model_name_or_path` - `TONGYI-MAI/Z-Image-Turbo` に設定します。
- `output_dir` - チェックポイントと検証画像を保存するディレクトリに設定します。フルパスの使用を推奨します。
- `train_batch_size` - 特に小さなデータセットでは 1 を維持してください。
- `validation_resolution` - Z-Image は 1024px。`1024x1024` またはマルチアスペクトバケット: `1024x1024,1280x768,2048x2048`。
- `validation_guidance` - Turbo は 0–1 の低ガイダンスが典型。base は 4–6 が必要です。
- `validation_num_inference_steps` - Turbo は 8 で十分、Base は 50–100 程度。
- `--lora_rank=4` を使うと LoRA サイズを大幅に縮小でき VRAM に有利です。
- turbo ではアシスタントアダプタを指定するか、明示的に無効化します（後述）。

- `gradient_accumulation_steps` - 実行時間が線形に増えるため、VRAM が必要なときのみ使用。
- `optimizer` - 初心者は adamw_bf16 を推奨。他の adamw/lion も良好。
- `mixed_precision` - 最新 GPU は `bf16`、それ以外は `fp16`。
- `gradient_checkpointing` - ほぼ全ての状況で true 推奨。
- `gradient_checkpointing_interval` - 大きな GPU では 2 以上にして _n_ ブロックごとにチェックポイント可。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTuner には、トレーニングの安定性とパフォーマンスを大幅に向上させる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが自身の入力を生成することで露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

</details>

### アシスタント LoRA（Turbo）

Turbo はアシスタントアダプタを要求します:

- `assistant_lora_path`: `ostris/zimage_turbo_training_adapter`
- `assistant_lora_weight_name`:
  - `turbo`: `zimage_turbo_training_adapter_v1.safetensors`
  - `turbo-ostris-v2`: `zimage_turbo_training_adapter_v2.safetensors`

SimpleTuner は turbo フレーバーで自動入力します。上書きする場合は指定してください。品質低下を許容する場合は `--disable_assistant_lora` で無効化できます。

### 検証プロンプト

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

多様なプロンプトのセットは、モデルが学習中に崩壊していないかを判断する助けになります。この例では、`<token>` という単語を被写体名（instance_prompt）に置き換えてください。

<details>
<summary>設定例を表示</summary>

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

> ℹ️ Z-Image はフローマッチングモデルです。短く似たプロンプトはほぼ同一の画像になりやすいので、より長く具体的なプロンプトを使ってください。

### CLIP スコアトラッキング

モデルのパフォーマンスをスコアリングするための評価を有効にしたい場合は、CLIP スコアの設定と解釈に関する情報について [このドキュメント](../evaluation/CLIP_SCORES.md) を参照してください。

### 安定した評価損失

モデルのパフォーマンスをスコアリングするために安定した MSE 損失を使用したい場合は、評価損失の設定と解釈に関する情報について [このドキュメント](../evaluation/EVAL_LOSS.md) を参照してください。

### 検証プレビュー

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

Tiny AutoEncoder のオーバーヘッドを削減するには、`validation_preview_steps` をより高い値（例: 3 または 5）に設定してください。

### Flow スケジュールシフト（flow matching）

Z-Image のようなフローマッチングモデルには、学習されるタイムステップスケジュールの部分を動かす `shift` パラメータがあります。解像度に応じた auto-shift は安全な既定です。手動で増やすと粗い特徴に寄り、減らすと細部に寄ります。Turbo モデルでは値の変更で品質が落ちる可能性があります。

### 量子化モデルトレーニング

TorchAO などの量子化で精度と VRAM 要件を削減できます。Optimum Quanto は縮小傾向ですが、利用可能です。

`config.json` の例:

<details>
<summary>設定例を表示</summary>

```json
  "base_model_precision": "int8-torchao",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

### データセットの考慮事項

> ⚠️ 学習用の画像品質は重要です。Z-Image はアーティファクトを早期に吸収します。最終的に高品質データで再学習する必要がある場合があります。

データセットは十分に大きく保ちます（少なくとも `train_batch_size * gradient_accumulation_steps`、かつ `vae_batch_size` より大きく）。**no images detected in dataset** が出る場合は `repeats` を増やしてください。

マルチバックエンド設定例（`config/multidatabackend.json`）:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "pseudo-camera-10k-zimage",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zimage/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/zimage/dreambooth-subject",
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
    "cache_dir_vae": "cache/vae/zimage/dreambooth-subject-512",
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
    "cache_dir": "cache/text/zimage",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> caption_strategy のオプションと要件については [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

512px と 1024px のデータセットを同時に動かすことができ、収束に役立ちます。

データセットディレクトリを作成:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

### WandB と Huggingface Hub へのログイン

特に `--push_to_hub` と `--report_to=wandb` を使う場合は、トレーニング前にログインしてください:

```bash
wandb login
huggingface-cli login
```

### トレーニングの実行

SimpleTuner ディレクトリから、トレーニングを開始するいくつかのオプションがあります:

**オプション 1（推奨 - pip install）:**

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train
```

**オプション 2（Git clone 方法）:**

```bash
simpletuner train
```

**オプション 3（レガシー方法 - まだ動作します）:**

```bash
./train.sh
```

これにより、テキスト埋め込みと VAE 出力のディスクへのキャッシングが開始されます。

詳細については、[データローダー](../DATALOADER.md) と [チュートリアル](../TUTORIAL.md) のドキュメントを参照してください。

## マルチ GPU 構成

SimpleTuner は WebUI で **自動 GPU 検出** を提供します。オンボーディング時に以下を設定します:

- **Auto Mode**: 検出した GPU を最適設定で自動使用
- **Manual Mode**: 使用 GPU の選択やプロセス数のカスタム指定
- **Disabled Mode**: 単一 GPU 学習

WebUI はハードウェアを検出し、`--num_processes` と `CUDA_VISIBLE_DEVICES` を自動設定します。

手動設定や高度な構成については、インストールガイドの [Multi-GPU Training セクション](../INSTALL.md#multiple-gpu-training) を参照してください。

## 推論のヒント

### ガイダンス設定

Z-Image はフローマッチングです。低いガイダンス値（0〜1）が品質と多様性を保ちやすいです。高いガイダンスで学習した場合は、推論側で CFG をサポートしていることを確認し、バッチ CFG では生成が遅くなったり VRAM 使用が増える点に注意してください。

## 注意事項とトラブルシューティングのヒント

### 最低 VRAM 設定

- GPU: 単一の NVIDIA CUDA デバイス（10〜12G、強い量子化/オフロード必須）
- システムメモリ: ~32〜48G
- ベースモデル精度: `nf4-bnb` または `int8`
- オプティマイザ: Lion 8Bit Paged（`bnb-lion8bit-paged`）または adamw 系
- 解像度: 512px（1024px は VRAM 増）
- バッチサイズ: 1、勾配蓄積 0
- DeepSpeed: 無効 / 未設定
- `--quantize_via=cpu` を使用して <=16G 起動 OOM を回避
- `--gradient_checkpointing` を有効化
- Ramtorch または group offload を有効化

プリキャッシュ段階で OOM になる場合は、`--text_encoder_precision=int8-torchao` と `--vae_enable_tiling=true` によりテキストエンコーダ量子化と VAE タイリングを有効化できます。起動時のメモリをさらに下げるには `--offload_during_startup=true` を使用し、テキストエンコーダまたは VAE のどちらかだけをロードします。

### 量子化

- 16G カードで学習するには最低でも 8bit 量子化が必要になることが多いです。
- 8bit 量子化は学習に大きな悪影響を与えず、バッチサイズを上げられます。
- **int8** はハードウェア加速の恩恵が大きく、**nf4-bnb** は VRAM をさらに削減しますが敏感です。
- LoRA を後でロードする際は、学習時と同じベース精度を使うのが理想です。

### アスペクトバケッティング

- 正方形クロップのみでも動作しますが、マルチアスペクトバケットで汎化が向上します。
- データセットの自然なアスペクトを使うと形状に偏りが出る場合があるため、ランダムクロップで改善できます。
- 画像ディレクトリを複数回定義して構成を混ぜると、良い汎化が得られています。

### 学習率

#### LoRA (--lora_type=standard)

- 大型 Transformer では低い学習率が安定しやすいです。
- まず中程度の rank（4〜16）から試し、極端に大きい rank は後で。
- モデルが不安定なら `max_grad_norm` を下げ、学習が止まるなら上げます。

#### LoKr (--lora_type=lycoris)

- 高めの学習率（例: AdamW で `1e-3`、Lion で `2e-4`）が有効な場合があります。好みに合わせて調整してください。
- 正則化データに `is_regularisation_data` を設定するとベースモデルの保持に役立ちます。

### 画像アーティファクト

Z-Image は悪いアーティファクトを早期に吸収します。最終的に高品質データで再学習する必要があるかもしれません。学習率が高すぎる、データ品質が低い、アスペクト処理が不適切な場合にグリッドアーティファクトが出やすいです。

### カスタム微調整 Z-Image の学習

微調整済みチェックポイントの中にはディレクトリ構造が不完全なものがあります。その場合は以下を設定してください:

<details>
<summary>設定例を表示</summary>

```json
{
    "model_family": "z-image",
    "pretrained_model_name_or_path": "TONGYI-MAI/Z-Image-Base",
    "pretrained_transformer_model_name_or_path": "your-custom-transformer",
    "pretrained_vae_model_name_or_path": "TONGYI-MAI/Z-Image-Base",
    "pretrained_transformer_subfolder": "none"
}
```
</details>

## トラブルシューティング

- 起動時 OOM: group offload（Quanto とは併用不可）、LoRA rank を下げる、または量子化（`--base_model_precision int8`/`nf4`）。
- ぼやけた出力: `validation_num_inference_steps` を増やす（例: 24–28）かガイダンスを 1.0 付近まで上げる。
- アーティファクト/過学習: rank や学習率を下げ、プロンプトの多様性を増やすか学習を短くする。
- アシスタントアダプタの問題: turbo はアダプタが必須。品質低下を許容する場合のみ無効化。
- 検証が遅い: 検証解像度/ステップ数を下げる。フローマッチングは収束が速いです。
