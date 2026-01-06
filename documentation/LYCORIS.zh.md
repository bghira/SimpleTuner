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
