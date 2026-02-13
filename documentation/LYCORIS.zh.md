# LyCORIS

## 背景

[LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) 是一套完善的参数高效微调（PEFT）方法，允许在更少显存下微调模型，并生成更小、更易分发的权重。

## 使用 LyCORIS

要使用 LyCORIS，设置 `--lora_type=lycoris` 并指定 `--lycoris_config=config/lycoris_config.json`，其中 `config/lycoris_config.json` 为 LyCORIS 配置文件路径。

在 `config.json` 中写入：
```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "lycoris_config": "config/lycoris_config.json",
    "validation_lycoris_strength": 1.0,
    ...the rest of your settings...
}
```


LyCORIS 配置文件格式如下：

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

### 字段

可选字段：
- apply_preset，用于 LycorisNetwork.apply_preset
- 放在最后的、所选算法特有的关键字参数

必填字段：
- multiplier（除非你明确知道预期，否则应设为 1.0）
- linear_dim
- linear_alpha

更多信息请参考 [库文档](https://github.com/KohakuBlueleaf/LyCORIS/tree/main/docs)。

### Flux 2 (Klein) 模块目标

Flux 2 模型使用自定义模块类，而非通用的 `Attention` 和 `FeedForward` 名称。Flux 2 的 LoKR 配置应以下列模块为目标：

- `Flux2Attention` — 双流注意力块
- `Flux2FeedForward` — 双流前馈块
- `Flux2ParallelSelfAttention` — 单流并行注意力+前馈块（融合的 QKV 和 MLP 投影）

包含 `Flux2ParallelSelfAttention` 会训练单流块，可能改善收敛性，但会增加过拟合的风险。如果在 Flux 2 上使用 LyCORIS LoKR 难以收敛，建议添加此目标。

Flux 2 LoKR 配置示例：

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

### T-LoRA（时间步依赖 LoRA）

T-LoRA 在训练过程中应用时间步依赖的秩掩码。在高噪声水平（去噪早期）时，较少的 LoRA 秩处于活跃状态，学习粗略结构；在低噪声水平（去噪后期）时，更多秩被激活，捕捉细节。此功能需要包含 `lycoris.modules.tlora` 的 LyCORIS 版本。

T-LoRA 配置示例：

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

可选的 T-LoRA 字段（添加到同一 JSON 中）：

- `tlora_min_rank`（整数，默认 `1`）— 最高噪声水平时的最少活跃秩数。
- `tlora_alpha`（浮点数，默认 `1.0`）— 掩码调度指数。`1.0` 为线性；大于 `1.0` 的值会将更多容量分配给细节步骤。

> **注意：** 在视频模型中使用 T-LoRA 可能效果不佳，因为时间压缩会在时间步边界处混合帧。

在验证期间，SimpleTuner 会在每个去噪步骤自动应用时间步依赖的掩码，使推理与训练条件一致。无需额外配置——训练时的掩码参数会被自动复用。

## 潜在问题

在 SDXL 上使用 Lycoris 时，训练 FeedForward 模块可能会破坏模型并使损失变为 `NaN`。

使用 SageAttention（`--sageattention_usage=training`）会加剧该问题，几乎会立即失败。

解决方法是从 lycoris 配置中移除 `FeedForward` 模块，仅训练 `Attention` 块。

## LyCORIS 推理示例

以下是一个简单的 FLUX.1-dev 推理脚本，展示如何用 create_lycoris_from_weights 包装 unet 或 transformer 并用于推理。

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
