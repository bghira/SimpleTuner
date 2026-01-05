## HiDream クイックスタート

この例では、HiDream 用の Lycoris LoKr をトレーニングします。十分なメモリが必要です。

24G GPU は、広範なブロックオフロードや fused backward pass なしに成立する最低ラインになりそうです。Lycoris LoKr でも同様に動作します。

### ハードウェア要件

HiDream は合計 17B パラメータで、学習済み MoE ゲートにより任意時点でのアクティブは約 8B です。**4 つ**のテキストエンコーダと Flux VAE を使用します。

全体としてアーキテクチャは複雑で、Flux Dev からの派生（直接蒸留または継続的なファインチューニング）と思われます。同じ重みを共有しているように見える検証サンプルがあるためです。

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

> ⚠️ Hugging Face Hub にアクセスしにくい国にいるユーザーは、システムが使用する `$SHELL` に応じて `~/.bashrc` または `~/.zshrc` に `HF_ENDPOINT=https://hf-mirror.com` を追加してください。


手動で設定したい場合:

`config/config.json.example` を `config/config.json` にコピーします:

```bash
cp config/config.json.example config/config.json
```

そこで、以下の変数を変更する必要があります:

- `model_type` - `lora` に設定します。
- `lora_type` - `lycoris` に設定します。
- `model_family` - `hidream` に設定します。
- `model_flavour` - `full` に設定します。`dev` は蒸留されているため、蒸留を破ってでもやり切るつもりでない限り直接学習しにくいです。
  - 実際 `full` モデルも学習が難しいですが、蒸留されていない唯一のモデルです。
- `output_dir` - チェックポイントと検証画像を保存するディレクトリに設定します。フルパスの使用を推奨します。
- `train_batch_size` - 1、たぶん。
- `validation_resolution` - `1024x1024` または HiDream がサポートする他の解像度に設定します。
  - 他の解像度はカンマ区切りで指定できます: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - 推論時に慣れている値を使用します。2.5〜3.0 付近の低い値ほど現実的になります。
- `validation_num_inference_steps` - おおよそ 30 を使用します。
- `use_ema` - `true` に設定すると、メインの学習済みチェックポイントに加えてより滑らかな結果が得られます。

- `optimizer` - 使い慣れたオプティマイザを使用できますが、この例では `optimi-lion` を使います。
- `mixed_precision` - 最も効率的な学習のため `bf16` を推奨します。`no` にすると結果は良くなりますが、メモリ消費が増えて遅くなります。
- `gradient_checkpointing` - 無効化すると最速ですが、バッチサイズが制限されます。最低 VRAM を狙うには有効化が必須です。

HiDream の高度なオプションとして、MoE 補助損失を学習中に含めることができます。MoE 損失を追加すると、値は通常より大きくなります。

- `hidream_use_load_balancing_loss` - `true` に設定するとロードバランシング損失を有効化します。
- `hidream_load_balancing_loss_weight` - 補助損失の大きさです。既定値は `0.01` ですが、より積極的にするなら `0.1` や `0.2` にできます。

これらのオプションの影響は現時点では不明です。

最終的な config.json は以下のようになります:

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
    "validation_guidance": 3.0,
    "validation_guidance_rescale": "0.0",
    "vae_batch_size": 1,
    "train_batch_size": 1,
    "tracker_run_name": "eval_loss_test1",
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "output_dir": "output/models-hidream",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "hidream",
    "offload_during_startup": true,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 10000,
    "max_grad_norm": 0.01,
    "lycoris_config": "config/lycoris_config.json",
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "lora_type": "lycoris",
    "learning_rate": "4e-5",
    "gradient_checkpointing": "true",
    "grad_clip_method": "value",
    "eval_steps_interval": 100,
    "disable_benchmark": false,
    "data_backend_config": "config/hidream/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "int8-quanto",
    "text_encoder_3_precision": "int8-quanto",
    "text_encoder_4_precision": "int8-quanto",
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ マルチ GPU ユーザーは、使用する GPU 数の設定については [このドキュメント](../OPTIONS.md#environment-configuration-variables) を参照してください。

> ℹ️ この構成では T5 (#3) と Llama (#4) テキストエンコーダの精度を int8 にして 24G カード向けのメモリを節約しています。メモリに余裕がある場合は、これらのオプションを削除するか `no_change` に設定できます。

そしてシンプルな `config/lycoris_config.json`。追加の安定性のために `FeedForward` を削除してもよい点に注意してください。

<details>
<summary>設定例を表示</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 16384,
    "linear_alpha": 1,
    "full_matrix": true,
    "use_scalar": true,
    "factor": 16,
    "apply_preset": {
        "target_module": [
            "Attention",
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 16
            },
        }
    }
}
```
</details>

`config/lycoris_config.json` に `"use_scalar": true` を設定するか、`config/config.json` に `"init_lokr_norm": 1e-4` を設定すると学習が大幅に高速化します。両方を有効にすると少し遅くなるようです。`init_lokr_norm` を設定すると、ステップ 0 の検証画像がわずかに変わります。

`config/lycoris_config.json` に `FeedForward` モジュールを追加すると、すべてのエキスパートを含むより多くのパラメータが学習されます。ただし、エキスパートの学習はかなり難しいようです。

より簡単な方法として、エキスパート外の feed forward パラメータのみを学習する以下の `config/lycoris_config.json` を使用できます。

<details>
<summary>設定例を表示</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 16384,
    "linear_alpha": 1,
    "full_matrix": true,
    "use_scalar": true,
    "factor": 16,
    "apply_preset": {
        "name_algo_map": {
            "double_stream_blocks.*.block.attn*": {
                "factor": 16
            },
            "double_stream_blocks.*.block.ff_t*": {
                "factor": 16
            },
            "double_stream_blocks.*.block.ff_i.shared_experts*": {
                "factor": 16
            },
            "single_stream_blocks.*.block.attn*": {
                "factor": 16
            },
            "single_stream_blocks.*.block.ff_i.shared_experts*": {
                "factor": 16
            }
        },
        "use_fnmatch": true
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

トレーナーをこのプロンプトライブラリに向けるには、`config.json` の末尾に新しい行を追加して TRAINER_EXTRA_ARGS に追加します:
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

多様なプロンプトのセットは、モデルが学習中に崩壊していないかを判断する助けになります。この例では、`<token>` という単語を被写体名（instance_prompt）に置き換えてください。

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

> ℹ️ HiDream は 128 トークンにデフォルトで切り詰めます。

#### CLIP スコアトラッキング

モデルのパフォーマンスをスコアリングするための評価を有効にしたい場合は、CLIP スコアの設定と解釈に関する情報について [このドキュメント](../evaluation/CLIP_SCORES.md) を参照してください。

</details>

# 安定した評価損失

モデルのパフォーマンスをスコアリングするために安定した MSE 損失を使用したい場合は、評価損失の設定と解釈に関する情報について [このドキュメント](../evaluation/EVAL_LOSS.md) を参照してください。

#### 検証プレビュー

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

Tiny AutoEncoder のオーバーヘッドを削減するには、`validation_preview_steps` をより高い値（例: 3 または 5）に設定してください。`validation_num_inference_steps=20` と `validation_preview_steps=5` の場合、ステップ 5、10、15、20 でプレビュー画像を受け取ります。

#### Flow-matching スケジュールシフト

OmniGen、Sana、Flux、SD3 などのフローマッチングモデルには、単純な小数値を使って学習されるタイムステップスケジュールの部分をシフトできる `shift` というプロパティがあります。

`full` モデルは `3.0`、`dev` は `6.0` で学習されています。

実際には、こうした高いシフト値はモデルを破壊しがちです。`1.0` は良い出発点ですが変化が小さすぎる場合があり、`3.0` は高すぎる可能性があります。

##### 自動シフト
一般的に推奨されるアプローチは、いくつかの最近の研究に従って解像度依存のタイムステップシフト `--flow_schedule_auto_shift` を有効にすることです。これは大きな画像には高いシフト値を、小さな画像には低いシフト値を使用します。これにより安定したが潜在的に平凡なトレーニング結果が得られます。

##### 手動指定
_以下の例は Discord の General Awareness 氏のご協力によるものです_

`--flow_schedule_shift` 値を 0.1（非常に低い値）で使用すると、画像の細部のみが影響を受けます:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

`--flow_schedule_shift` 値を 4.0（非常に高い値）で使用すると、モデルの大きな構成的特徴や潜在的なカラースペースが影響を受けます:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### データセットの考慮事項

モデルをトレーニングするには十分なデータセットが不可欠です。データセットサイズには制限があり、モデルを効果的にトレーニングできる十分な大きさのデータセットであることを確認する必要があります。最小限のデータセットサイズは `train_batch_size * gradient_accumulation_steps` に加え `vae_batch_size` より大きいことに注意してください。小さすぎるとデータセットは使用できません。

> ℹ️ 画像が少なすぎる場合、**no images detected in dataset** というメッセージが表示されることがあります。`repeats` 値を増やすことでこの制限を克服できます。

手元のデータセットに応じて、データセットディレクトリとデータローダー設定ファイルを異なる方法でセットアップする必要があります。この例では [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) を使用します。

以下を含む `--data_backend_config`（`config/multidatabackend.json`）ドキュメントを作成します:

<details>
<summary>設定例を表示</summary>

```json
[
  {
    "id": "pseudo-camera-10k-hidream",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/hidream/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/hidream/dreambooth-subject",
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
    "cache_dir": "cache/text/hidream",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

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

### トレーニングの実行

SimpleTuner ディレクトリから、トレーニングを開始するいくつかのオプションがあります:

**オプション 1（推奨 - pip install）:**
```bash
pip install simpletuner[cuda]
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

新しいモデルのため、例は少し調整が必要です。以下は動作する例です:

<details>
<summary>Python 推論例を表示</summary>

```py
import torch
from helpers.models.hidream.pipeline import HiDreamImagePipeline
from helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
from lycoris import create_lycoris_from_weights
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

llama_repo = "unsloth/Meta-Llama-3.1-8B-Instruct"
model_id = 'HiDream-ai/HiDream-I1-Dev'
adapter_repo_id = 'bghira/hidream5m-photo-1mp-Prodigy'
adapter_filename = 'pytorch_lora_weights.safetensors'

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
    llama_repo,
)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    llama_repo,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)

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

adapter_file_path = download_adapter(repo_id=adapter_repo_id)
transformer = HiDreamImageTransformer2DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, subfolder="transformer")
pipeline = HiDreamImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    transformer=transformer,
)
lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

prompt = "Place your test prompt here."
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

## Optional: quantise the model to save on vram.
## Note: The model was quantised during training, and so it is recommended to do the same during inference time.
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # the pipeline is already in its target precision level
t5_embeds, llama_embeds, negative_t5_embeds, negative_llama_embeds, pooled_embeds, negative_pooled_embeds = pipeline.encode_prompt(
    prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt, num_images_per_prompt=1
)
# We'll unload the text encoders to save memory.
pipeline.text_encoder.to("meta")
pipeline.text_encoder_2.to("meta")
pipeline.text_encoder_3.to("meta")
pipeline.text_encoder_4.to("meta")
model_output = pipeline(
    t5_prompt_embeds=t5_embeds,
    llama_prompt_embeds=llama_embeds,
    pooled_prompt_embeds=pooled_embeds,
    negative_t5_prompt_embeds=negative_t5_embeds,
    negative_llama_prompt_embeds=negative_llama_embeds,
    negative_pooled_prompt_embeds=negative_pooled_embeds,
    num_inference_steps=30,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=3.2,
).images[0]

model_output.save("output.png", format="PNG")

```
</details>

## 注意事項とトラブルシューティングのヒント

### 最低 VRAM 設定

HiDream の最低 VRAM 構成は約 20〜22G です:

- OS: Ubuntu Linux 24
- GPU: 単一の NVIDIA CUDA デバイス（10G、12G）
- システムメモリ: 約 50G（多くなる場合も少なくなる場合もあります）
- ベースモデル精度:
  - Apple/AMD システム: `int8-quanto`（または `fp8-torchao`、`int8-torchao` は同様のメモリ使用プロファイル）
    - `int4-quanto` も動作しますが、精度が低下し結果が悪くなる可能性があります
  - NVIDIA システム: `nf4-bnb` が良好に動作すると報告されていますが、`int8-quanto` より遅くなります
- オプティマイザ: Lion 8Bit Paged、`bnb-lion8bit-paged`
- 解像度: 1024px
- バッチサイズ: 1、勾配蓄積ステップ 0
- DeepSpeed: 無効 / 未設定
- PyTorch: 2.7+
- `--quantize_via=cpu` を使用して、<=16G カードの起動時 outOfMemory を回避
- `--gradient_checkpointing` を有効化
- 小さな LoRA または Lycoris 設定（例: LoRA rank 1 または Lokr factor 25）
- 環境変数 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定すると、複数アスペクト比での学習時の VRAM 使用を最小化できます。

**注**: VAE 埋め込みとテキストエンコーダ出力の事前キャッシュは、より多くのメモリを使用して OOM になる場合があります。VAE タイリングとスライシングは既定で有効です。OOM が出る場合は `offload_during_startup=true` を有効にしてみてください。そうでなければ厳しいかもしれません。

Pytorch 2.7 と CUDA 12.8 を使用した NVIDIA 4090 で、速度は約 1 秒あたり 3 イテレーションでした。

### マスク損失

被写体またはスタイルをトレーニングしていて、一方または他方をマスクしたい場合は、Dreambooth ガイドの [マスク損失トレーニング](../DREAMBOOTH.md#masked-loss) セクションを参照してください。

### 量子化

速度/品質とメモリのトレードオフでは `int8` が最適ですが、`nf4` と `int4` も利用可能です。`int4` は HiDream では推奨されませんが、十分に長い学習を行えば一定の品質には到達します。

### 学習率

#### LoRA (--lora_type=standard)

- 小さな LoRA（rank 1〜8）には `4e-4` 前後の高い学習率が有効
- 大きな LoRA（rank 64〜256）には `6e-5` 前後の低い学習率が有効
- Diffusers の制約で `lora_alpha` を `lora_rank` と異なる値にすることは基本的にサポートされません（後段の推論ツールで扱うことを理解している場合を除く）。
  - 例えば `lora_alpha` を 1.0 にすると、すべての LoRA ランクで学習率を同じにできます。

#### LoKr (--lora_type=lycoris)

- LoKr には穏やかな学習率が適しています（AdamW で `1e-4`、Lion で `2e-5`）。
- 他のアルゴはさらなる検証が必要です。
- Prodigy は LoRA/LoKr に良い選択ですが、必要な学習率を過大評価したり肌をスムーズにしすぎる場合があります。

### 画像アーティファクト

HiDream は画像アーティファクトへの反応が不明ですが、Flux VAE を使用しており、細部の制約は類似しています。

最も一般的な問題は、学習率が高すぎる/バッチサイズが小さすぎることです。これにより、滑らかな肌、ぼやけ、ピクセル化などのアーティファクトが発生することがあります。

### アスペクトバケッティング

当初このモデルはアスペクトバケットにうまく反応しませんでしたが、コミュニティにより実装が改善されました。

### 複数解像度トレーニング

512px など低い解像度で最初に学習させることで速度を上げられますが、高解像度への一般化は不明です。512px → 1024px の順で段階的に学習するのが最善です。

1024px 以外の解像度で学習する場合は `--flow_schedule_auto_shift` を有効にすることを推奨します。低解像度は VRAM が少なく済むため、より大きなバッチサイズを使えます。

### フルランクチューニング

DeepSpeed は HiDream で大量のシステムメモリを使用しますが、十分に大きなシステムではフルチューニングは正常に動作します。

フルランクの代わりに、より安定でメモリフットプリントの小さい Lycoris LoKr が推奨されます。

PEFT LoRA はシンプルなスタイルには有用ですが、細部の維持は難しくなります。
