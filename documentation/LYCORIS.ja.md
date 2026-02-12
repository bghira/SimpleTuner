# LyCORIS

## 背景

[LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) は、より少ない VRAM でモデルをファインチューニングし、配布可能な重みを小さくできるパラメータ効率ファインチューニング（PEFT）手法の包括的なスイートです。

## LyCORIS の使用

LyCORIS を使用するには、`--lora_type=lycoris` を設定し、`--lycoris_config=config/lycoris_config.json` を指定します。`config/lycoris_config.json` は LyCORIS 設定ファイルのパスです。

`config.json` には次を記載します:
```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "lycoris_config": "config/lycoris_config.json",
    "validation_lycoris_strength": 1.0,
    ...the rest of your settings...
}
```


LyCORIS 設定ファイルの形式は次のとおりです:

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 10,
    "apply_preset": {
        "target_module": [
            "Attention",
            "FeedForward"
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 10
            },
            "FeedForward": {
                "factor": 4
            }
        }
    }
}
```

### フィールド

任意フィールド:
- LycorisNetwork.apply_preset 用の apply_preset
- 末尾に、選択したアルゴリズム固有のキーワード引数

必須フィールド:
- multiplier（期待が分かっている場合を除き 1.0 に設定）
- linear_dim
- linear_alpha

LyCORIS の詳細は、[ライブラリのドキュメント](https://github.com/KohakuBlueleaf/LyCORIS/tree/main/docs) を参照してください。

### Flux 2 (Klein) モジュールターゲット

Flux 2 モデルは、汎用の `Attention` や `FeedForward` ではなく、カスタムモジュールクラスを使用します。Flux 2 の LoKR 設定では以下をターゲットに設定してください：

- `Flux2Attention` — ダブルストリームアテンションブロック
- `Flux2FeedForward` — ダブルストリームフィードフォワードブロック
- `Flux2ParallelSelfAttention` — シングルストリーム並列アテンション+フィードフォワードブロック（QKV と MLP 投影が融合）

`Flux2ParallelSelfAttention` を含めるとシングルストリームブロックも学習対象となり、収束が改善する可能性がありますが、過学習のリスクが増加します。Flux 2 で LyCORIS LoKR の収束が困難な場合、このターゲットの追加を推奨します。

Flux 2 LoKR 設定例：

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
            "Flux2Attention", "Flux2FeedForward", "Flux2ParallelSelfAttention"
        ],
        "module_algo_map": {
            "Flux2FeedForward": {
                "factor": 4
            },
            "Flux2Attention": {
                "factor": 2
            },
            "Flux2ParallelSelfAttention": {
                "factor": 2
            }
        }
    }
}
```

### T-LoRA（タイムステップ依存 LoRA）

T-LoRA はトレーニング中にタイムステップ依存のランクマスキングを適用します。高ノイズレベル（デノイジング初期）ではアクティブな LoRA ランクが少なくなり、粗い構造を学習します。低ノイズレベル（デノイジング後期）ではより多くのランクがアクティブになり、細部を捉えます。この機能には `lycoris.modules.tlora` を含む LyCORIS バージョンが必要です。

T-LoRA 設定例：

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

任意の T-LoRA フィールド（同じ JSON に追加）：

- `tlora_min_rank`（整数、デフォルト `1`）— 最高ノイズレベルでのアクティブランクの最小数。
- `tlora_alpha`（浮動小数点数、デフォルト `1.0`）— マスキングスケジュールの指数。`1.0` で線形、`1.0` より大きい値はディテール側のステップにより多くの容量を割り当てます。

> **注意：** ビデオモデルで T-LoRA を使用すると、テンポラル圧縮がタイムステップ境界をまたいでフレームを混合するため、結果が最適でない場合があります。

バリデーション中、SimpleTuner は各デノイジングステップでタイムステップ依存のマスキングを自動的に適用し、推論がトレーニング条件と一致するようにします。追加の設定は不要です——トレーニング時のマスキングパラメータが自動的に再利用されます。

## 予想される問題

SDXL に Lycoris を使用すると、FeedForward モジュールの学習でモデルが壊れ、損失が `NaN`（Not-a-Number）に飛ぶことがあります。

SageAttention（`--sageattention_usage=training`）を使うと悪化しやすく、ほぼ確実に即失敗します。

解決策は、lycoris 設定から `FeedForward` モジュールを外し、`Attention` ブロックのみを学習することです。

## LyCORIS 推論例

FLUX.1-dev の簡単な推論スクリプトです。create_lycoris_from_weights で unet/transformer をラップして推論に使います。

```py
import torch

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import AutoModelForCausalLM, CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

from lycoris import create_lycoris_from_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
bfl_repo = "black-forest-labs/FLUX.1-dev"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer")

lycoris_safetensors_path = 'pytorch_lora_weights.safetensors'
lycoris_strength = 1.0
wrapper, _ = create_lycoris_from_weights(lycoris_strength, lycoris_safetensors_path, transformer)
wrapper.merge_to() # using apply_to() will be slower.

transformer.to(device, dtype=dtype)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
)

pipe.enable_sequential_cpu_offload()

with torch.inference_mode():
    image = pipe(
        prompt="a pokemon that looks like a pizza is eating a popsicle",
        width=1280,
        height=768,
        num_inference_steps=15,
        generator=generator,
        guidance_scale=3.5,
    ).images[0]
image.save('image.png')

# optionally, save a merged pipeline containing the LyCORIS baked-in:
pipe.save_pretrained('/path/to/output/pipeline')
```
