# T-LoRA（タイムステップ依存LoRA）

## 背景

標準的なLoRAファインチューニングは、すべてのディフュージョンタイムステップに対して均一に固定の低ランク適応を適用します。学習データが限られている場合（特に単一画像のカスタマイズ）、これは過学習につながります。意味的な情報がほとんど存在しない高ノイズタイムステップでモデルがノイズパターンを記憶してしまいます。

**T-LoRA**（[Soboleva et al., 2025](https://arxiv.org/abs/2507.05964)）は、現在のディフュージョンタイムステップに基づいてアクティブなLoRAランク数を動的に調整することでこの問題を解決します：

- **高ノイズ**（デノイジング初期、$t \to T$）：アクティブなランク数が少なくなり、情報量の少ないノイズパターンの記憶を防ぎます。
- **低ノイズ**（デノイジング後期、$t \to 0$）：アクティブなランク数が多くなり、モデルが細かいコンセプトの詳細を捉えることができます。

SimpleTunerのT-LoRAサポートは[LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)ライブラリ上に構築されており、`lycoris.modules.tlora`モジュールを含むLyCORISバージョンが必要です。

> 実験的：動画モデルでのT-LoRAは、時間圧縮がタイムステップ境界を越えてフレームを混合するため、結果が最適でない場合があります。

## クイックセットアップ

### 1. トレーニング設定を行う

`config.json`で、LyCORISと別のT-LoRA設定ファイルを使用します：

```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "lycoris_config": "config/lycoris_tlora.json",
    "validation_lycoris_strength": 1.0
}
```

### 2. LyCORIS T-LoRA設定を作成する

`config/lycoris_tlora.json`を作成します：

```json
{
    "algo": "tlora",
    "multiplier": 1.0,
    "linear_dim": 64,
    "linear_alpha": 32,
    "apply_preset": {
        "target_module": ["Attention", "FeedForward"]
    }
}
```

トレーニングを開始するために必要なのはこれだけです。以下のセクションでは、オプションのチューニングと推論について説明します。

## 設定リファレンス

### 必須フィールド

| フィールド | 型 | 説明 |
|-------|------|-------------|
| `algo` | string | `"tlora"`である必要があります |
| `multiplier` | float | LoRA強度の乗数。何をしているか理解していない限り`1.0`のままにしてください |
| `linear_dim` | int | LoRAランク。マスキングスケジュールにおける`max_rank`になります |
| `linear_alpha` | int | LoRAスケーリング係数（`tlora_alpha`とは別です） |

### オプションフィールド

| フィールド | 型 | デフォルト | 説明 |
|-------|------|---------|-------------|
| `tlora_min_rank` | int | `1` | 最も高いノイズレベルでの最小アクティブランク数 |
| `tlora_alpha` | float | `1.0` | マスキングスケジュールの指数。`1.0`は線形で、`1.0`より大きい値はより多くの容量を細部のステップに割り当てます |
| `apply_preset` | object | — | `target_module`と`module_algo_map`によるモジュールターゲティング |

### モデル固有のモジュールターゲット

ほとんどのモデルでは汎用的な`["Attention", "FeedForward"]`ターゲットが機能します。Flux 2（Klein）の場合は、カスタムクラス名を使用してください：

```json
{
    "algo": "tlora",
    "multiplier": 1.0,
    "linear_dim": 64,
    "linear_alpha": 32,
    "apply_preset": {
        "target_module": [
            "Flux2Attention", "Flux2FeedForward", "Flux2ParallelSelfAttention"
        ]
    }
}
```

モデルごとのモジュールターゲットの完全なリストについては、[LyCORISドキュメント](../LYCORIS.md)を参照してください。

## チューニングパラメータ

### `linear_dim`（ランク）

ランクが高いほどパラメータ数と表現力が増しますが、限られたデータでは過学習しやすくなります。元のT-LoRA論文では、SDXLの単一画像カスタマイズにランク64を使用しています。

### `tlora_min_rank`

最もノイズの多いタイムステップでのランクアクティベーションの下限を制御します。これを増やすと、モデルがより粗い構造を学習できますが、過学習防止の効果が減少します。デフォルトの`1`から始めて、収束が遅すぎる場合のみ上げてください。

### `tlora_alpha`（スケジュール指数）

マスキングスケジュールのカーブ形状を制御します：

- `1.0` — `min_rank`と`max_rank`の間の線形補間
- `> 1.0` — 高ノイズ時のマスキングがより積極的になり、ほとんどのランクはデノイジングの終盤付近でのみアクティブになります
- `< 1.0` — マスキングがより穏やかになり、ランクがより早くアクティブになります

<details>
<summary>スケジュールの可視化（ランク vs. タイムステップ）</summary>

`linear_dim=64`、`tlora_min_rank=1`の場合、1000ステップスケジューラでの例：

```
alpha=1.0 (linear):
  t=0   (clean)  → 64 active ranks
  t=250 (25%)    → 48 active ranks
  t=500 (50%)    → 32 active ranks
  t=750 (75%)    → 16 active ranks
  t=999 (noise)  →  1 active rank

alpha=2.0 (quadratic — biased toward detail):
  t=0   (clean)  → 64 active ranks
  t=250 (25%)    → 60 active ranks
  t=500 (50%)    → 48 active ranks
  t=750 (75%)    → 20 active ranks
  t=999 (noise)  →  1 active rank

alpha=0.5 (sqrt — biased toward structure):
  t=0   (clean)  → 64 active ranks
  t=250 (25%)    → 55 active ranks
  t=500 (50%)    → 46 active ranks
  t=750 (75%)    → 33 active ranks
  t=999 (noise)  →  1 active rank
```

</details>

## SimpleTunerパイプラインでの推論

SimpleTunerの同梱パイプラインにはT-LoRAサポートが組み込まれています。バリデーション中は、トレーニング時のマスキングパラメータが各デノイジングステップで自動的に再利用されるため、追加の設定は不要です。

トレーニング外でのスタンドアロン推論では、SimpleTunerのパイプラインを直接インポートし、`_tlora_config`属性を設定できます。これにより、ステップごとのマスキングがモデルのトレーニング時の設定と一致することが保証されます。

### SDXLの例

```py
import torch
from lycoris import create_lycoris_from_weights

# SimpleTunerの同梱SDXLパイプラインを使用（T-LoRAサポート内蔵）
from simpletuner.helpers.models.sdxl.pipeline import StableDiffusionXLPipeline
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
dtype = torch.bfloat16
device = "cuda"

# パイプラインコンポーネントを読み込む
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)

# LyCORIS T-LoRA重みを読み込んで適用する
lora_path = "path/to/pytorch_lora_weights.safetensors"
wrapper, _ = create_lycoris_from_weights(1.0, lora_path, unet)
wrapper.merge_to()

unet.to(device)

pipe = StableDiffusionXLPipeline(
    scheduler=scheduler,
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    unet=unet,
)

# T-LoRA推論マスキングを有効にする — トレーニング設定と一致させる必要があります
pipe._tlora_config = {
    "max_rank": 64,      # lycoris設定のlinear_dim
    "min_rank": 1,       # tlora_min_rank（デフォルト1）
    "alpha": 1.0,        # tlora_alpha（デフォルト1.0）
}

with torch.inference_mode():
    image = pipe(
        prompt="a sks dog riding a surfboard",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=5.0,
    ).images[0]

image.save("tlora_output.png")
```

### Fluxの例

```py
import torch
from lycoris import create_lycoris_from_weights

# SimpleTunerの同梱Fluxパイプラインを使用（T-LoRAサポート内蔵）
from simpletuner.helpers.models.flux.pipeline import FluxPipeline
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16
device = "cuda"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2")
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype)

# LyCORIS T-LoRA重みを読み込んで適用する
lora_path = "path/to/pytorch_lora_weights.safetensors"
wrapper, _ = create_lycoris_from_weights(1.0, lora_path, transformer)
wrapper.merge_to()

transformer.to(device)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
)

# T-LoRA推論マスキングを有効にする
pipe._tlora_config = {
    "max_rank": 64,
    "min_rank": 1,
    "alpha": 1.0,
}

with torch.inference_mode():
    image = pipe(
        prompt="a sks dog riding a surfboard",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=3.5,
    ).images[0]

image.save("tlora_flux_output.png")
```

> **注意：** SimpleTunerの同梱パイプライン（例：`simpletuner.helpers.models.flux.pipeline.FluxPipeline`）を使用する必要があります。標準のDiffusersパイプラインではありません。同梱パイプラインにのみ、ステップごとのT-LoRAマスキングロジックが含まれています。

### なぜ`merge_to()`だけでマスキングを省略できないのか？

`merge_to()`はLoRA重みをベースモデルに恒久的に統合します。これはフォワードパス中にLoRAパラメータをアクティブにするために必要です。しかし、T-LoRAはタイムステップ依存のランクマスキングで**トレーニング**されています。つまり、ノイズレベルに応じて特定のランクがゼロにされていました。推論時に同じマスキングを再適用しないと、すべてのランクがすべてのタイムステップで発火し、過飽和や焼けたような見た目の画像が生成されます。

パイプラインに`_tlora_config`を設定することで、デノイジングループが各モデルフォワードパスの前に正しいマスクを適用し、その後にクリアするようになります。

<details>
<summary>マスキングの内部動作</summary>

各デノイジングステップで、パイプラインは以下を呼び出します：

```python
from simpletuner.helpers.training.lycoris import apply_tlora_inference_mask, clear_tlora_mask

_tlora_cfg = getattr(self, "_tlora_config", None)
if _tlora_cfg:
    apply_tlora_inference_mask(
        timestep=int(t),
        max_timestep=self.scheduler.config.num_train_timesteps,
        max_rank=_tlora_cfg["max_rank"],
        min_rank=_tlora_cfg["min_rank"],
        alpha=_tlora_cfg["alpha"],
    )
try:
    noise_pred = self.unet(...)  # or self.transformer(...)
finally:
    if _tlora_cfg:
        clear_tlora_mask()
```

`apply_tlora_inference_mask`は以下の式を使用して、形状`(1, max_rank)`のバイナリマスクを計算します：

$$r = \left\lfloor\left(\frac{T - t}{T}\right)^\alpha \cdot (R_{\max} - R_{\min})\right\rfloor + R_{\min}$$

ここで$T$はスケジューラの最大タイムステップ、$R_{\max}$は`linear_dim`、$R_{\min}$は`tlora_min_rank`です。マスクの最初の$r$要素は`1.0`に設定され、残りは`0.0`に設定されます。このマスクは、LyCORISの`set_timestep_mask()`を介してすべてのT-LoRAモジュールにグローバルに設定されます。

フォワードパスが完了した後、`clear_tlora_mask()`がマスク状態を削除し、後続の操作に漏れないようにします。

</details>

<details>
<summary>バリデーション中のSimpleTunerの設定受け渡し方法</summary>

トレーニング中、T-LoRA設定辞書（`max_rank`、`min_rank`、`alpha`）はAcceleratorオブジェクトに保存されます。バリデーション実行時、`validation.py`がこの設定をパイプラインにコピーします：

```python
# setup_pipeline()
if getattr(self.accelerator, "_tlora_active", False):
    self.model.pipeline._tlora_config = self.accelerator._tlora_config

# clean_pipeline()
if hasattr(self.model.pipeline, "_tlora_config"):
    del self.model.pipeline._tlora_config
```

これは完全に自動化されており、バリデーション画像が正しいマスキングを使用するためにユーザーが設定する必要はありません。

</details>

## 上流：T-LoRA論文

<details>
<summary>論文の詳細とアルゴリズム</summary>

**T-LoRA: Single Image Diffusion Model Customization Without Overfitting**
Vera Soboleva, Aibek Alanov, Andrey Kuznetsov, Konstantin Sobolev
[arXiv:2507.05964](https://arxiv.org/abs/2507.05964) — AAAI 2026に採択

この論文は2つの相補的なイノベーションを紹介しています：

### 1. タイムステップ依存ランクマスキング

核心的な知見は、より高いディフュージョンタイムステップ（よりノイズの多い入力）は、低いタイムステップよりも過学習しやすいということです。高ノイズ時、潜在表現は意味的な信号がほとんどないランダムノイズで占められており、フルランクのアダプターをこれに対してトレーニングすると、ターゲットコンセプトを学習するのではなく、ノイズパターンを記憶することになります。

T-LoRAは、現在のタイムステップに基づいてアクティブなLoRAランクを制限する動的マスキングスケジュールでこの問題に対処します。

### 2. 直交重みパラメトリゼーション（オプション）

論文はまた、元のモデル重みのSVD分解を介してLoRA重みを初期化し、正則化損失を通じて直交性を強制することを提案しています。これによりアダプターコンポーネント間の独立性が確保されます。

SimpleTunerのLyCORIS統合は、過学習削減の主要な要因であるタイムステップマスキングコンポーネントに焦点を当てています。直交初期化はスタンドアロンT-LoRA実装の一部ですが、現在LyCORISの`tlora`アルゴリズムでは使用されていません。

### 引用

```bibtex
@misc{soboleva2025tlorasingleimagediffusion,
      title={T-LoRA: Single Image Diffusion Model Customization Without Overfitting},
      author={Vera Soboleva and Aibek Alanov and Andrey Kuznetsov and Konstantin Sobolev},
      year={2025},
      eprint={2507.05964},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05964},
}
```

</details>

## よくある落とし穴

- **推論時に`_tlora_config`を忘れた場合：** 画像が過飽和または焼けたような見た目になります。トレーニング済みのマスキングスケジュールに従う代わりに、すべてのランクがすべてのタイムステップで発火します。
- **標準Diffusersパイプラインの使用：** 標準パイプラインにはT-LoRAマスキングロジックが含まれていません。SimpleTunerの同梱パイプラインを使用する必要があります。
- **`linear_dim`の不一致：** `_tlora_config`の`max_rank`はトレーニング時に使用した`linear_dim`と一致する必要があります。そうでないとマスクの次元が間違ってしまいます。
- **動画モデル：** 時間圧縮がタイムステップ境界を越えてフレームを混合するため、タイムステップ依存のマスキング信号が弱まる可能性があります。結果が最適でない場合があります。
- **SDXL + FeedForwardモジュール：** SDXLでLyCORISを使用してFeedForwardモジュールをトレーニングするとNaN損失が発生する可能性があります。これはT-LoRA固有の問題ではなく、一般的なLyCORISの問題です。詳細については[LyCORISドキュメント](../LYCORIS.md#potential-problems)を参照してください。
