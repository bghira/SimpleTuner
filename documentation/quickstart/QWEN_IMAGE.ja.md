## Qwen Image クイックスタート

> 🆕 edit チェックポイントを探していますか？ 参照ペア学習の手順は [Qwen Image Edit quickstart](./QWEN_EDIT.md) を参照してください。

この例では、20B パラメータの Vision-Language モデルである Qwen Image の LoRA をトレーニングします。サイズが大きいため、積極的なメモリ最適化が必要です。

24GB GPU は最低ラインで、さらに強い量子化と慎重な設定が必要です。スムーズな運用には 40GB+ を強く推奨します。

24G で学習する場合、検証はメモリ不足になりやすいため、低解像度や int8 を超える強い量子化が必要です。

### ハードウェア要件

Qwen Image は 20B パラメータのモデルで、洗練されたテキストエンコーダだけでも量子化前で ~16GB VRAM を消費します。16 チャンネルの独自 VAE を使用します。

**重要な制限:**
- **AMD ROCm と MacOS は未対応**（効率的な Flash Attention がないため）
- バッチサイズ > 1 は現在正しく動作しないため、gradient accumulation を使用してください
- TREAD（Text-Representation Enhanced Adversarial Diffusion）は未対応

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

Vast、RunPod、TensorDock（など）の場合、CUDA 12.2-12.8 イメージで CUDA 拡張のコンパイルを有効にするには以下が機能します:

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

> ⚠️ Hugging Face Hub にアクセスしにくい国にいるユーザーは、システムが使用する `$SHELL` に応じて `~/.bashrc` または `~/.zshrc` に `HF_ENDPOINT=https://hf-mirror.com` を追加してください。

手動で設定したい場合:

`config/config.json.example` を `config/config.json` にコピーします:

```bash
cp config/config.json.example config/config.json
```

そこで、以下の変数を変更する必要があります:

- `model_type` - `lora` に設定します。
- `lora_type` - PEFT LoRA は `standard`、LoKr は `lycoris` を使用します。
- `model_family` - `qwen_image` に設定します。
- `model_flavour` - `v1.0` に設定します。
- `output_dir` - チェックポイントと検証画像を保存するディレクトリに設定します。フルパスの使用を推奨します。
- `train_batch_size` - 1 に固定（バッチサイズ > 1 は現在動作しません）。
- `gradient_accumulation_steps` - 2〜8 を設定して実効バッチを大きくします。
- `validation_resolution` - `1024x1024` もしくはメモリ制約のためより低い値に設定します。
  - 24G は現状 1024x1024 検証に対応できません。サイズを下げてください。
  - 他の解像度はカンマ区切りで指定できます: `1024x1024,768x768,512x512`
- `validation_guidance` - 3.0〜4.0 前後が良好です。
- `validation_num_inference_steps` - 30 前後を使用します。
- `use_ema` - `true` に設定すると滑らかな結果が得られますがメモリを追加で消費します。

- `optimizer` - 良好な結果のため `optimi-lion` を使用するか、余裕があれば `adamw-bf16`。
- `mixed_precision` - Qwen Image は `bf16` 必須です。
- `gradient_checkpointing` - **必須**（`true`）。妥当なメモリ使用量のため必要です。
- `base_model_precision` - **強く推奨** `int8-quanto` または `nf4-bnb`（24GB では必須）。
- `quantize_via` - 小型 GPU の量子化 OOM を避けるため `cpu` に設定します。
- `quantize_activations` - 学習品質維持のため `false` にします。

24GB GPU 向けのメモリ最適化設定:
- `lora_rank` - 8 以下を使用。
- `lora_alpha` - `lora_rank` と同じ値にする。
- `flow_schedule_shift` - 1.73 に設定（1.0〜3.0 で調整）。

最小構成の `config.json` 例:

<details>
<summary>設定例を表示</summary>

```json
{
    "model_type": "lora",
    "model_family": "qwen_image",
    "model_flavour": "v1.0",
    "lora_type": "standard",
    "lora_rank": 8,
    "lora_alpha": 8,
    "output_dir": "output/models-qwen_image",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "validation_resolution": "1024x1024",
    "validation_guidance": 4.0,
    "validation_num_inference_steps": 30,
    "validation_seed": 42,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_step_interval": 100,
    "vae_batch_size": 1,
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 1000,
    "max_grad_norm": 0.01,
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "learning_rate": "1e-4",
    "gradient_checkpointing": "true",
    "base_model_precision": "int2-quanto",
    "quantize_via": "cpu",
    "quantize_activations": false,
    "flow_schedule_shift": 1.73,
    "disable_benchmark": false,
    "data_backend_config": "config/qwen_image/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ マルチ GPU ユーザーは、使用する GPU 数の設定については [このドキュメント](../OPTIONS.md#environment-configuration-variables) を参照してください。

> ⚠️ **24GB GPU で重要:** テキストエンコーダ単体で ~16GB VRAM を消費します。`int2-quanto` または `nf4-bnb` を使うことで大幅に削減できます。

動作確認用の既知構成:

**オプション 1（推奨 - pip install）:**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train example=qwen_image.peft-lora
```

**オプション 2（Git clone 方法）:**
```bash
simpletuner train env=examples/qwen_image.peft-lora
```

**オプション 3（レガシー方法 - まだ動作します）:**
```bash
ENV=examples/qwen_image.peft-lora ./train.sh
```

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTuner には、トレーニングの安定性とパフォーマンスを大幅に向上させる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが自身の入力を生成することで露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

#### 検証プロンプト

`config/config.json` 内には「プライマリ検証プロンプト」があり、これは通常、単一の被写体やスタイルでトレーニングしているメインの instance_prompt です。さらに、検証中に実行する追加のプロンプトを含む JSON ファイルを作成できます。

設定ファイル例 `config/user_prompt_library.json.example` には以下の形式が含まれています:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

ニックネームは検証のファイル名になるため、短くファイルシステムと互換性のあるものにしてください。

このプロンプトライブラリを使用するには、`config.json` に以下を追加します:
```json
  "validation_prompt_library": "config/user_prompt_library.json",
```

多様なプロンプトのセットは、モデルが正しく学習しているかを判断する助けになります:

```json
{
    "anime_style": "a breathtaking anime-style portrait with vibrant colors and expressive features",
    "chef_cooking": "a high-quality, detailed photograph of a sous-chef immersed in culinary creation",
    "portrait": "a lifelike and intimate portrait showcasing unique personality and charm",
    "cinematic": "a cinematic, visually stunning photo with dramatic and captivating presence",
    "elegant": "an elegant and timeless portrait exuding grace and sophistication",
    "adventurous": "a dynamic and adventurous photo captured in an exciting moment",
    "mysterious": "a mysterious and enigmatic portrait shrouded in shadows and intrigue",
    "vintage": "a vintage-style portrait evoking the charm and nostalgia of a bygone era",
    "artistic": "an artistic and abstract representation blending creativity with visual storytelling",
    "futuristic": "a futuristic and cutting-edge portrayal set against advanced technology"
}
```

#### CLIP スコアトラッキング

モデルのパフォーマンスをスコアリングするための評価を有効にしたい場合は、CLIP スコアの設定と解釈に関する情報について [このドキュメント](../evaluation/CLIP_SCORES.md) を参照してください。

#### 安定した評価損失

モデルのパフォーマンスをスコアリングするために安定した MSE 損失を使用したい場合は、評価損失の設定と解釈に関する情報について [このドキュメント](../evaluation/EVAL_LOSS.md) を参照してください。

#### 検証プレビュー

SimpleTuner は Tiny AutoEncoder モデルを使用して生成中の中間検証プレビューのストリーミングをサポートしています。これにより、webhook コールバックを介してリアルタイムで検証画像が生成されるのを段階的に確認できます。

有効にするには:
```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

**要件:**
- Webhook 設定
- 検証が有効

Tiny AutoEncoder のオーバーヘッドを削減するには、`validation_preview_steps` をより高い値（例: 3 または 5）に設定してください。`validation_num_inference_steps=20` と `validation_preview_steps=5` の場合、ステップ 5、10、15、20 でプレビュー画像を受け取ります。

#### Flow スケジュールシフト

Qwen Image はフローマッチングモデルとして、生成過程のどの部分を学習するかを制御するタイムステップシフトに対応しています。

`flow_schedule_shift` の目安:
- 低い値（0.1〜1.0）: 細部重視
- 中程度（1.0〜3.0）: バランス（推奨）
- 高い値（3.0〜6.0）: 大域的構図重視

##### 自動シフト
`--flow_schedule_auto_shift` を有効にすると、解像度依存のタイムステップシフトが適用されます。大きな画像には高いシフト値、小さな画像には低いシフト値が使用され、安定する一方で平凡になる可能性があります。

##### 手動指定
Qwen Image では `--flow_schedule_shift` を 1.73 にするのが出発点として推奨されます。データセットや目的に応じて調整してください。

#### データセットの考慮事項

モデルをトレーニングするには十分なデータセットが不可欠です。データセットサイズには制限があり、モデルを効果的にトレーニングできる十分な大きさのデータセットであることを確認する必要があります。

> ℹ️ 画像が少なすぎる場合、**no images detected in dataset** というメッセージが表示されることがあります。`repeats` 値を増やすことでこの制限を克服できます。

> ⚠️ **重要**: 現在の制約により `train_batch_size` は 1 に固定し、代わりに `gradient_accumulation_steps` で実効バッチを増やしてください。

以下を含む `--data_backend_config`（`config/multidatabackend.json`）ドキュメントを作成します:

```json
[
  {
    "id": "pseudo-camera-10k-qwen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/pseudo-camera-10k",
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
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/dreambooth-subject",
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
    "cache_dir": "cache/text/qwen_image",
    "disabled": false,
    "write_batch_size": 16
  }
]
```

> ℹ️ `.txt` のキャプションがある場合は `caption_strategy=textfile` を使用してください。
> caption_strategy のオプションと要件については [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。
> ℹ️ OOM を避けるため、テキスト埋め込みの `write_batch_size` は小さくしています。

次に、`datasets` ディレクトリを作成します:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

これにより、約 10k の写真サンプルが `datasets/pseudo-camera-10k` ディレクトリにダウンロードされ、自動的に作成されます。

Dreambooth の画像は `datasets/dreambooth-subject` ディレクトリに入れてください。

#### WandB と Huggingface Hub へのログイン

特に `--push_to_hub` と `--report_to=wandb` を使う場合は、トレーニング開始前に WandB と HF Hub にログインしておく必要があります。

Git LFS リポジトリに手動でアイテムをプッシュする場合は、`git config --global credential.helper store` も実行してください。

以下のコマンドを実行します:

```bash
wandb login
```

および

```bash
huggingface-cli login
```

指示に従って両方のサービスにログインしてください。

</details>

### トレーニングの実行

SimpleTuner ディレクトリから、以下を実行するだけです:

```bash
./train.sh
```

これにより、テキスト埋め込みと VAE 出力のディスクへのキャッシングが開始されます。

詳細については、[データローダー](../DATALOADER.md) と [チュートリアル](../TUTORIAL.md) のドキュメントを参照してください。

### メモリ最適化のヒント

#### 最低 VRAM 構成（24GB 最低）

Qwen Image の最低 VRAM 構成は約 24GB 必要です:

- OS: Ubuntu Linux 24
- GPU: 単一の NVIDIA CUDA デバイス（24GB 最低）
- システムメモリ: 64GB+ 推奨
- ベースモデル精度:
  - NVIDIA: `int2-quanto` または `nf4-bnb`（24GB 必須）
  - `int4-quanto` でも動作するが品質低下の可能性
- オプティマイザ: `optimi-lion` または `bnb-lion8bit-paged` でメモリ効率重視
- 解像度: まず 512px または 768px、余裕があれば 1024px
- バッチサイズ: 1（制約のため必須）
- 勾配蓄積: 2〜8 で実効バッチを稼ぐ
- `--gradient_checkpointing` を有効化（必須）
- `--quantize_via=cpu` を使用して起動時 OOM を回避
- 小さな LoRA rank（1〜8）を使用
- 環境変数 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定すると VRAM 使用を最小化できます

**注**: VAE 埋め込みやテキストエンコーダ出力の事前キャッシュはメモリを多く使います。OOM が出る場合は `offload_during_startup=true` を有効にしてください。

### LoRA の推論

Qwen Image は新しいモデルのため、以下に動作する推論例を示します:

<details>
<summary>Python 推論例を表示</summary>

```python
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

model_id = 'Qwen/Qwen-Image'
adapter_id = 'your-username/your-lora-name'

# Load the pipeline
pipeline = QwenImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
pipeline.load_lora_weights(adapter_id)

# Optional: quantize the model to save VRAM
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

# Move to device
pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

# Generate an image
prompt = "Your test prompt here"
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=4.0,
    generator=torch.Generator(device='cuda').manual_seed(42),
    width=1024,
    height=1024,
).images[0]

image.save("output.png", format="PNG")
```
</details>

### 注意事項とトラブルシューティングのヒント

#### バッチサイズの制限

現在、Qwen Image はテキストエンコーダのシーケンス長処理の関係でバッチサイズ > 1 に問題があります。必ず以下を使用してください:
- `train_batch_size: 1`
- 実効バッチを稼ぐため `gradient_accumulation_steps: 2-8`

#### 量子化

- `int2-quanto` は最も強い省メモリだが品質に影響する可能性
- `nf4-bnb` はメモリと品質のバランスが良い
- `int4-quanto` は中間的
- 40GB+ の VRAM がある場合を除き `int8` は避ける

#### 学習率

LoRA トレーニングの場合:
- 小さな LoRA（rank 1〜8）: 1e-4 前後
- 大きな LoRA（rank 16〜32）: 5e-5 前後
- Prodigy オプティマイザ: 1.0 から開始し自動適応

#### 画像アーティファクト

アーティファクトが出る場合:
- 学習率を下げる
- 勾配蓄積を増やす
- 画像品質と前処理を確認
- 初期は低解像度から開始する

#### 複数解像度トレーニング

まず 512px または 768px で学習し、その後 1024px で微調整します。異なる解像度で学習する場合は `--flow_schedule_auto_shift` を有効にしてください。

### プラットフォームの制限

**未対応:**
- AMD ROCm（効率的な Flash Attention がない）
- Apple Silicon/MacOS（メモリと注意機構の制限）
- 24GB 未満のコンシューマ GPU

### 既知の問題

1. バッチサイズ > 1 は正しく動作しない（勾配蓄積を使用）
2. TREAD は未対応
3. テキストエンコーダのメモリ消費が大きい（量子化前 ~16GB）
4. シーケンス長処理の問題（[上流 issue](https://github.com/huggingface/diffusers/issues/12075)）

追加のヘルプとトラブルシューティングは [SimpleTuner documentation](/documentation) を参照するか、コミュニティ Discord に参加してください。
