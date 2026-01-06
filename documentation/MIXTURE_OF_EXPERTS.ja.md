# Mixture-of-Experts

SimpleTuner は学習タスクを二分し、推論時の self-attention と cross-attention を二つの異なる重みセットに分割できるようにします。

この例では、SegMind と Hugging Face の共同成果である [SSD-1B](https://huggingface.co/segmind/ssd-1b) を使い、単一モデルより安定して学習でき、細部の再現性が高い 2 つの新しいモデルを作成します。

SSD-1B は小型のため、より軽量なハードウェアでも学習可能です。事前学習済み重みから開始するため Apache 2.0 ライセンスに従う必要がありますが、手続きは比較的簡単です。結果の重みは商用用途でも利用できます。

SDXL 0.9 と 1.0 では、フルのベースモデルと分割スケジュールの refiner が含まれていました。

- ベースモデルは 999 から 0 のステップで学習
  - 3B 超のパラメータで、単体で動作します。
- refiner は 199 から 0 のステップで学習
  - こちらも 3B 超のパラメータで、リソース的には過剰です。単体では大きな歪みやアニメ的バイアスが生じます。

この状況を改善する方法を見ていきます。


## ベースモデル（"Stage One"）

Mixture-of-Experts の最初の部分がベースモデルです。SDXL のケースと同様に 1000 ステップ全てで学習できますが、必須ではありません。次の設定では 1000 ステップ中 650 ステップのみを学習し、時間を節約しつつ安定性を高めます。

### 環境設定

`config/config.env` に次の値を設定します:

```bash
# Ensure these aren't incorrectly set.
export USE_BITFIT=false
export USE_DORA=false
# lora could be used here instead, but the concept hasn't been explored.
export MODEL_TYPE="full"
export MODEL_FAMILY="sdxl"
export MODEL_NAME="segmind/SSD-1B"
# The original Segmind model used a learning rate of 1e-5, which is
# probably too high for whatever batch size most users can pull off.
export LEARNING_RATE=4e-7

# We really want this as high as you can tolerate.
# - If training is very slow, ensure your CHECKPOINT_STEPS and VALIDATION_STEPS
#   are set low enough that you'll get a checkpoint every couple hours.
# - The original Segmind models used a batch size of 32 with 4 accumulations.
export TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1

# If you are running on a beefy machine that doesn't fully utilise its VRAM during training, set this to "false" and your training will go faster.
export USE_GRADIENT_CHECKPOINTING=true

# Enable first stage model training
export TRAINER_EXTRA_ARGS="--refiner_training --refiner_training_strength=0.35 --refiner_training_invert_schedule"

# Optionally reparameterise it to v-prediction/zero-terminal SNR. 'sample' prediction_type can be used instead for x-prediction.
# This will start out looking pretty terrible until about 1500-2500 steps have passed, but it could be very worthwhile.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --prediction_type=v_prediction --rescale_betas_zero_snr --training_scheduler_timestep_spacing=trailing"
```

### データローダ設定

データローダ設定に特別な注意は不要です。この手順の詳細は [dataloader 設定ガイド](DATALOADER.md) を参照してください。

### 検証

現時点では、SimpleTuner は stage one の評価で stage two モデルを使用しません。

将来的には、stage two が既に存在する場合や並行学習している場合に使用できるようにする予定です。

---

## Refiner モデル（"Stage Two"）

### SDXL refiner 学習との比較

- Segmind SSD-1B を両ステージに使う場合、テキスト埋め込みは 2 つの学習ジョブで共有**可能**
  - SDXL refiner は SDXL ベースモデルとは異なるテキスト埋め込みレイアウトを使用します。
- VAE 埋め込みは SDXL refiner と同様に共有**可能**で、両モデルは同じ入力レイアウトを使います。
- Segmind モデルでは美的スコアは使用せず、SDXL と同様のマイクロコンディショニング入力（例: クロップ座標）を使用します。
- モデルが小さいため学習が高速で、stage one のテキスト埋め込みも再利用できます。

### 環境設定

`config/config.env` の値を次のように更新し、stage two の学習に切り替えます。ベースモデル設定をコピーとして保持しておくと便利です。

```bash
# Update your OUTPUT_DIR value, so that we don't overwrite the stage one model checkpoints.
export OUTPUT_DIR="/some/new/path"

# We'll swap --refiner_training_invert_schedule for --validation_using_datasets
# - Train the end of the model instead of the beginning
# - Validate using images as input for partial denoising to evaluate fine detail improvements
export TRAINER_EXTRA_ARGS="--refiner_training --refiner_training_strength=0.35 --validation_using_datasets"

# Don't update these values if you've set them on the stage one. Be sure to use the same parameterisation for both models!
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --prediction_type=v_prediction --rescale_betas_zero_snr --training_scheduler_timestep_spacing=trailing"
```

### データセット形式

画像は純粋に高品質であるべきです。圧縮アーティファクト等が疑わしいデータセットは削除してください。

それ以外は、同じデータローダ設定を 2 つの学習ジョブで共有できます。

デモ用データセットが必要であれば、ライセンスが緩やかな [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) が良い選択です。

### 検証

stage two の refiner 学習では、各学習セットから画像を自動的に選び、検証時に部分的デノイズの入力として使います。

## CLIP スコア追跡

評価でモデル性能をスコア化したい場合は、CLIP スコアの設定と解釈について [このドキュメント](evaluation/CLIP_SCORES.md) を参照してください。

# 安定評価損失

安定 MSE 損失でモデル性能を評価したい場合は、評価損失の設定と解釈について [このドキュメント](evaluation/EVAL_LOSS.md) を参照してください。

## 推論で両モデルを組み合わせる

両モデルを簡単なスクリプトで結合して試すには、以下を参考にしてください:

```py
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, UniPCMultistepScheduler
from torch import float16, cuda
from torch.backends import mps

# For a training_refiner_strength of .35, you'll set the base model strength to 0.65.
# Formula: 1 - training_refiner_strength
training_refiner_strength = 0.35
base_model_power = 1 - training_refiner_strength
# Reduce this for lower quality but speed-up.
num_inference_steps = 40
# Update these to your local or hugging face hub paths.
stage_1_model_id = 'bghira/terminus-xl-velocity-v2'
stage_2_model_id = 'bghira/terminus-xl-refiner'
torch_device = 'cuda' if cuda.is_available() else 'mps' if mps.is_available() else 'cpu'

pipe = StableDiffusionXLPipeline.from_pretrained(stage_1_model_id, add_watermarker=False, torch_dtype=float16).to(torch_device)
pipe.scheduler = UniPCMultistepScheduler.from_pretrained(stage_1_model_id, subfolder="scheduler")
img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(stage_2_model_id).to(device=torch_device, dtype=float16)
img2img_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(stage_1_model_id, subfolder="scheduler")

prompt = "An astronaut riding a green horse"

# Important: update this to True if you reparameterised the models.
use_zsnr = True

image = pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_end=base_model_power,
    guidance_scale=9.2,
    guidance_rescale=0.7 if use_zsnr else 0.0,
    output_type="latent",
).images
image = img2img_pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_start=base_model_power,
    guidance_scale=9.2,
    guidance_rescale=0.7 if use_zsnr else 0.0,
    image=image,
).images[0]
image.save('demo.png', format="PNG")
```

実験できるポイント:
- `base_model_power` や `num_inference_steps` を調整（両パイプラインで同じ値にする必要があります）。
- `guidance_scale`、`guidance_rescale` を各ステージで別々に設定可能。コントラストとリアリズムに影響します。
- ベースと refiner で別々のプロンプトを使い、細部への誘導を変える。
