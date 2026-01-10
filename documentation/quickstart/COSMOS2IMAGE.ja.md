## Cosmos2 Predict (Image) クイックスタート

この例では、NVIDIA のフローマッチングモデルである Cosmos2 Predict (Image) の Lycoris LoKr をトレーニングします。

### ハードウェア要件

Cosmos2 Predict (Image) はフローマッチングを使用する Vision Transformer ベースのモデルです。

**注**: アーキテクチャ上の理由から学習中の量子化は推奨されません。つまり、bf16 精度をそのまま使える十分な VRAM が必要です。

大きな最適化なしで快適に学習するには 24GB GPU を最低ラインとして推奨します。

### メモリオフロード（オプション）

Cosmos2 を小さな GPU に収めるには、グループオフロードを有効にします:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- ストリームは CUDA のみで有効です。他のデバイスは自動的にフォールバックします。
- `--enable_model_cpu_offload` とは併用しないでください。
- ディスクステージングは任意で、システム RAM がボトルネックの場合に有効です。

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
pip install 'simpletuner[cuda]'
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
- `lora_type` - `lycoris` に設定します。
- `model_family` - `cosmos2image` に設定します。
- `base_model_precision` - **重要**: `no_change` に設定します。Cosmos2 は量子化しないでください。
- `output_dir` - チェックポイントと検証画像を保存するディレクトリに設定します。フルパスの使用を推奨します。
- `train_batch_size` - 1 から開始し、VRAM に余裕があれば増やします。
- `validation_resolution` - Cosmos2 のデフォルトは `1024x1024`。
  - 他の解像度はカンマ区切りで指定できます: `1024x1024,768x768`
- `validation_guidance` - Cosmos2 には 4.0 前後を使用します。
- `validation_num_inference_steps` - 20 ステップ前後を使用します。
- `use_ema` - `true` に設定すると、メインの学習済みチェックポイントに加えてより滑らかな結果が得られます。
- `optimizer` - 例では `adamw_bf16` を使用します。
- `mixed_precision` - 最も効率的な学習のため `bf16` を推奨します。
- `gradient_checkpointing` - 学習速度と引き換えに VRAM を削減するため有効化します。

最終的な config.json は以下のようになります:

<details>
<summary>設定例を表示</summary>

```json
{
    "base_model_precision": "no_change",
    "checkpoint_step_interval": 500,
    "data_backend_config": "config/cosmos2image/multidatabackend.json",
    "disable_bucket_pruning": true,
    "flow_schedule_shift": 0.0,
    "flow_schedule_auto_shift": true,
    "gradient_checkpointing": true,
    "hub_model_id": "cosmos2image-lora",
    "learning_rate": 6e-5,
    "lora_type": "lycoris",
    "lycoris_config": "config/cosmos2image/lycoris_config.json",
    "lr_scheduler": "constant",
    "lr_warmup_steps": 100,
    "max_train_steps": 10000,
    "model_family": "cosmos2image",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/cosmos2image",
    "push_checkpoints_to_hub": false,
    "push_to_hub": false,
    "quantize_via": "cpu",
    "report_to": "tensorboard",
    "seed": 42,
    "tracker_project_name": "cosmos2image-training",
    "tracker_run_name": "cosmos2image-lora",
    "train_batch_size": 1,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_disable_unconditional": true,
    "validation_guidance": 4.0,
    "validation_guidance_rescale": 0.0,
    "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
    "validation_num_inference_steps": 20,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_prompt_library": false,
    "validation_resolution": "512x512",
    "validation_seed": 42,
    "validation_step_interval": 500
}
```
</details>

> ℹ️ マルチ GPU ユーザーは、使用する GPU 数の設定については [このドキュメント](../OPTIONS.md#environment-configuration-variables) を参照してください。

そして `config/cosmos2image/lycoris_config.json`:

<details>
<summary>設定例を表示</summary>

```json
{
    "bypass_mode": true,
    "algo": "lokr",
    "multiplier": 1.0,
    "full_matrix": true,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 4,
    "apply_preset": {
        "target_module": [
            "Attention"
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 4
            }
        }
    }
}
```
</details>

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

このプロンプトライブラリを使用するには、設定に以下を追加します:
```json
"validation_prompt_library": "config/user_prompt_library.json"
```

多様なプロンプトのセットは、モデルが学習中に崩壊していないかを判断する助けになります。この例では、`<token>` という単語を被写体名（instance_prompt）に置き換えてください。

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```

#### CLIP スコアトラッキング

モデルのパフォーマンスをスコアリングするための評価を有効にしたい場合は、CLIP スコアの設定と解釈に関する情報について [このドキュメント](../evaluation/CLIP_SCORES.md) を参照してください。

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

フローマッチングモデルである Cosmos2 には、単純な小数値を使って学習されるタイムステップスケジュールの部分をシフトできる `shift` というプロパティがあります。

設定では `flow_schedule_auto_shift` が既定で有効になっており、解像度依存のタイムステップシフトを使用します。大きな画像には高いシフト値、小さな画像には低いシフト値が使われます。

#### データセットの考慮事項

モデルをトレーニングするには十分なデータセットが不可欠です。データセットサイズには制限があり、モデルを効果的にトレーニングできる十分な大きさのデータセットであることを確認する必要があります。最小限のデータセットサイズは `train_batch_size * gradient_accumulation_steps` に加え `vae_batch_size` より大きいことに注意してください。小さすぎるとデータセットは使用できません。

> ℹ️ 画像が少なすぎる場合、**no images detected in dataset** というメッセージが表示されることがあります。`repeats` 値を増やすことでこの制限を克服できます。

手元のデータセットに応じて、データセットディレクトリとデータローダー設定ファイルを異なる方法でセットアップする必要があります。この例では [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) を使用します。

以下を含む `--data_backend_config`（`config/cosmos2image/multidatabackend.json`）ドキュメントを作成します:

```json
[
  {
    "id": "pseudo-camera-10k-cosmos2",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/cosmos2/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/cosmos2/dreambooth-subject",
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
    "cache_dir": "cache/text/cosmos2",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> ℹ️ `.txt` のキャプションがある場合は `caption_strategy=textfile` を使用してください。
> caption_strategy のオプションと要件については [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

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

SimpleTuner ディレクトリから、トレーニングを開始するいくつかのオプションがあります:

**オプション 1（推奨 - pip install）:**
```bash
pip install 'simpletuner[cuda]'
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

### LoKr の推論を実行する

Cosmos2 は新しいモデルでドキュメントが限られているため、推論例は調整が必要になる可能性があります。基本的な構造は以下の通りです:

<details>
<summary>Python 推論例を表示</summary>

```py
import torch
from lycoris import create_lycoris_from_weights

# Model and adapter paths
model_id = 'nvidia/Cosmos-1.0-Predict-Image-Text2World-12B'
adapter_repo_id = 'your-username/your-cosmos2-lora'
adapter_filename = 'pytorch_lora_weights.safetensors'

def download_adapter(repo_id: str):
    import os
    from huggingface_hub import hf_hub_download
    adapter_filename = "pytorch_lora_weights.safetensors"
    cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
    cleaned_adapter_path = repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
    path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
    os.makedirs(path_to_adapter, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
    )

    return path_to_adapter_file

# Load the model and adapter

import torch
from diffusers import Cosmos2TextToImagePipeline

# Available checkpoints: nvidia/Cosmos-Predict2-2B-Text2Image, nvidia/Cosmos-Predict2-14B-Text2Image
model_id = "nvidia/Cosmos-Predict2-2B-Text2Image"
adapter_repo_id = "youruser/your-repo-name"

adapter_file_path = download_adapter(repo_id=adapter_repo_id)
pipe = Cosmos2TextToImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

pipe.to("cuda")

prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

output = pipe(
    prompt=prompt, negative_prompt=negative_prompt, generator=torch.Generator().manual_seed(1)
).images[0]
output.save("output.png")

```

</details>

## 注意事項とトラブルシューティングのヒント

### メモリに関する注意

Cosmos2 は学習中に量子化できないため、量子化モデルよりメモリ使用量が増えます。VRAM を抑えるための主な設定:

- `gradient_checkpointing` を有効化
- バッチサイズ 1 を使用
- メモリが厳しい場合は `adamw_8bit` オプティマイザの使用を検討
- 環境変数 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定すると、複数アスペクト比での学習時の VRAM 使用を最小化できます
- `--vae_cache_disable` を使用してオンライン VAE エンコードに切り替えると、ディスク節約になる一方、学習時間/メモリ圧は増えます

### 学習上の考慮事項

Cosmos2 は新しいモデルのため、最適な学習パラメータはまだ探索中です:

- 例では AdamW + `6e-5` の学習率を使用
- 複数解像度に対応するため flow schedule auto-shift を有効化
- CLIP 評価で学習進捗をモニタリング

### アスペクトバケッティング

設定では `disable_bucket_pruning` が true になっています。データセット特性に応じて調整してください。

### 複数解像度トレーニング

最初は 512px で学習し、後から高解像度に移行できます。`flow_schedule_auto_shift` はマルチ解像度学習に役立ちます。

### マスク損失

被写体またはスタイルをトレーニングしていて、一方または他方をマスクしたい場合は、Dreambooth ガイドの [マスク損失トレーニング](../DREAMBOOTH.md#masked-loss) セクションを参照してください。

### 既知の制限

- システムプロンプト処理は未実装
- 学習特性は引き続き調査中
- 量子化は未対応のため避けてください
