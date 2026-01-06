# ControlNet トレーニングガイド

## 背景

ControlNet モデルは、トレーニング時に与える条件データに応じて多様なタスクに対応できます。

当初は学習に多くのリソースが必要でしたが、現在は PEFT LoRA や Lycoris を使って同等のタスクをはるかに少ないリソースで学習できます。

例（Diffusers の ControlNet モデルカードより）:

![例](https://tripleback.net/public/controlnet-example-1.png)

左が条件入力として与えた「Canny エッジマップ」です。右側が ControlNet がベースの SDXL モデルから導いた出力です。

この使い方では、プロンプトは構図にほとんど関与せず、主に細部を埋める役割になります。

## ControlNet トレーニングの様子

学習初期の ControlNet は制御の兆しがほとんどありません:

![例](https://tripleback.net/public/controlnet-example-2.png)
(_Stable Diffusion 2.1 モデルで 4 ステップだけ学習した ControlNet_)

アンテロープのプロンプトが構図をほぼ支配しており、ControlNet の条件入力は無視されています。

時間とともに、条件入力が尊重されるようになります:

![例](https://tripleback.net/public/controlnet-example-3.png)
(_Stable Diffusion XL モデルで 100 ステップだけ学習した ControlNet_)

この段階で ControlNet の影響が少し見られるようになりますが、結果は非常に不安定でした。

この目的には 100 ステップをはるかに超える学習が必要です。

## データローダ設定例

データローダ設定は、通常のテキストから画像のデータセット設定にかなり近い構成です:

- メインの画像データは `antelope-data` セット
  - `conditioning_data` キーを設定し、このセットと対になる条件データの `id` を指定します。
  - ベースのセットは `dataset_type` を `image` にします。
- セカンダリのデータセットとして `antelope-conditioning` を設定
  - 名称は重要ではなく、ここでは説明用に `-data` と `-conditioning` を付けています。
  - `dataset_type` は `conditioning` に設定し、評価と条件入力トレーニングに使うことをトレーナーに伝えます。
- SDXL を学習する場合、条件入力は VAE エンコードされず、学習時にピクセル値のままモデルへ渡されます。これにより、学習開始時の VAE 埋め込み処理に追加の時間がかかりません。
- Flux、SD3、Auraflow、HiDream などの MMDiT モデルを学習する場合、条件入力は潜在表現にエンコードされ、学習中にオンデマンドで計算されます。
- ここではすべて `-controlnet` と明示していますが、通常のフル/LoRA チューニングで使ったテキスト埋め込みを再利用できます。ControlNet 入力はプロンプト埋め込みを変更しません。
- アスペクト比バケットやランダムクロップを使う場合、条件サンプルもメインの画像サンプルと同じ方法でクロップされます。

```json
[
    {
        "id": "antelope-data",
        "type": "local",
        "dataset_type": "image",
        "conditioning_data": "antelope-conditioning",
        "instance_data_dir": "datasets/animals/antelope-data",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an antelope",
        "metadata_backend": "discovery",
        "minimum_image_size": 512,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "cache_dir_vae": "cache/vae/sdxl/antelope-data",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "cache_file_suffix": "controlnet-sdxl"
    },
    {
        "id": "antelope-conditioning",
        "type": "local",
        "dataset_type": "conditioning",
        "instance_data_dir": "datasets/animals/antelope-conditioning",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an antelope",
        "metadata_backend": "discovery",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "minimum_image_size": 512,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel_area",
        "cache_file_suffix": "controlnet-sdxl"
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/sdxl-base/controlnet"
    }
]
```

## 条件画像入力の生成

SimpleTuner の ControlNet 対応はまだ新しく、現時点で利用できるトレーニングセット生成手段は 1 つです:

- [create_canny_edge.py](/scripts/toolkit/datasets/controlnet/create_canny_edge.py)
  - Canny モデルの学習用データセットを生成するための、非常に基本的な例です。
  - スクリプト内の `input_dir` と `output_dir` を変更する必要があります。

100 枚未満の小さなデータセットなら約 30 秒です。

## ControlNet モデルを学習するための設定変更

データローダ設定を整えるだけでは ControlNet モデルの学習は開始できません。

`config/config.json` の中で、次の値を設定してください:

```bash
"model_type": 'lora',
"controlnet": true,

# You may have to reduce TRAIN_BATCH_SIZE and RESOLUTION more than usual
"train_batch_size": 1
```

最終的には次のような設定になります:

```json
{
    "aspect_bucket_rounding": 2,
    "caption_dropout_probability": 0.1,
    "checkpoint_step_interval": 100,
    "checkpoints_total_limit": 5,
    "controlnet": true,
    "data_backend_config": "config/controlnet-sdxl/multidatabackend.json",
    "disable_benchmark": false,
    "gradient_checkpointing": true,
    "hub_model_id": "simpletuner-controlnet-sdxl-lora-test",
    "learning_rate": 3e-5,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 100,
    "max_train_steps": 1000,
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "sdxl",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "bnb-lion8bit",
    "output_dir": "output/controlnet-sdxl/models",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 42,
    "train_batch_size": 1,
    "use_ema": false,
    "vae_cache_ondemand": true,
    "validation_guidance": 4.2,
    "validation_guidance_rescale": 0.0,
    "validation_num_inference_steps": 20,
    "validation_resolution": "1024x1024",
    "validation_seed": 42,
    "validation_step_interval": 10,
    "validation_torch_compile": false
}
```

## 生成された ControlNet モデルでの推論

**フル**の ControlNet モデル（ControlNet LoRA ではない）で推論する SDXL の例を示します:

```py
# Update these values:
base_model = "stabilityai/stable-diffusion-xl-base-1.0"         # This is the model you used as `--pretrained_model_name_or_path`
controlnet_model_path = "diffusers/controlnet-canny-sdxl-1.0"   # This is the path to the resulting ControlNet checkpoint
# controlnet_model_path = "/path/to/controlnet/checkpoint-100"

# Leave the rest alone:
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'

image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    controlnet_model_path,
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

images = pipe(
    prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

images[0].save(f"hug_lab.png")
```
(_[Hugging Face SDXL ControlNet の例](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0) からのデモコード_)


## 自動データ拡張と条件生成

SimpleTuner は起動時に条件データセットを自動生成できるため、手動の前処理が不要になります。特に次の用途で便利です:
- 超解像トレーニング
- JPEG アーティファクト除去
- 深度誘導生成
- エッジ検出（Canny）

### 仕組み

条件データセットを手動で作成する代わりに、メインのデータセット設定で `conditioning` 配列を指定します。SimpleTuner は次を行います:
1. 起動時に条件画像を生成
2. 適切なメタデータを持つ別データセットを作成
3. メインのデータセットと自動的に紐付け

### パフォーマンス上の考慮事項

一部のジェネレータは CPU がボトルネックになりやすく、システムの CPU タスクが重い場合に遅くなることがあります。別のジェネレータは GPU リソースを必要とし、メインプロセス内で実行されるため、起動時間が長くなる場合があります。

**CPU ベースのジェネレータ（高速）:**
- `superresolution` - ぼかしとノイズ処理
- `jpeg_artifacts` - 圧縮シミュレーション
- `random_masks` - マスク生成
- `canny` - エッジ検出

**GPU ベースのジェネレータ（遅い）:**
- `depth` / `depth_midas` - Transformer モデルの読み込みが必要
- `segmentation` - セマンティックセグメンテーションモデル
- `optical_flow` - 動き推定

GPU ベースのジェネレータはメインプロセスで動作し、大規模データセットでは起動時間が大きく増える場合があります。

### 例: マルチタスク条件データセット

単一のソースデータセットから複数の条件タイプを生成する完全な例です:

```json
[
  {
    "id": "multitask-training",
    "type": "local",
    "instance_data_dir": "/datasets/high-quality-images",
    "caption_strategy": "filename",
    "resolution": 512,
    "conditioning": [
      {
        "type": "superresolution",
        "blur_radius": 2.0,
        "noise_level": 0.02,
        "captions": ["enhance image quality", "increase resolution", "sharpen"]
      },
      {
        "type": "jpeg_artifacts",
        "quality_range": [20, 40],
        "captions": ["remove compression", "fix jpeg artifacts"]
      },
      {
        "type": "canny",
        "low_threshold": 50,
        "high_threshold": 150
      }
    ]
  },
  {
    "id": "text-embed-cache",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/sdxl"
  }
]
```

この設定で行われること:
1. `/datasets/high-quality-images` から高品質画像を読み込む
2. 3 つの条件データセットを自動生成
3. 超解像と JPEG タスクに特定のキャプションを使用
4. Canny エッジデータセットには元の画像キャプションを使用

#### 生成データセットのキャプション戦略

生成した条件データのキャプションには次の 2 つの方法があります:

1. **ソースキャプションを使用**（デフォルト）: `captions` フィールドを省略
2. **カスタムキャプション**: 文字列または文字列配列を指定

「enhance」や「remove artifacts」のようなタスク特化の学習では、元の画像説明よりもカスタムキャプションのほうがうまく機能することが多いです。

### 起動時間の最適化

大規模データセットでは、条件生成が時間を要する場合があります。最適化のために:

1. **一度だけ生成**: 条件データはキャッシュされ、既に存在する場合は再生成されません
2. **CPU ジェネレータを使用**: 複数プロセスを使って高速化できます
3. **未使用タイプを無効化**: 必要なものだけ生成します
4. **事前生成**: `--skip_file_discovery=true` を指定して、探索と条件生成をスキップできます
5. **ディスク走査を回避**: 大規模データセットでは `preserve_data_backend_cache=True` を指定すると既存条件データの再走査を避けられ、起動時間が大幅に短縮されます。

生成処理は進行状況バーを表示し、中断しても再開できます。
