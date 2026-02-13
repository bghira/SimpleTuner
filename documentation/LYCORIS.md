# LyCORIS

## Background

[LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) is an extensive suite of parameter-efficient fine-tuning (PEFT) methods that allow you to finetune models while using less VRAM and produces smaller distributable weights.

## Using LyCORIS

To use LyCORIS, set `--lora_type=lycoris` and then set `--lycoris_config=config/lycoris_config.json`, where `config/lycoris_config.json` is the location of your LyCORIS configuration file.

The following will go into your `config.json`:
```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "lycoris_config": "config/lycoris_config.json",
    "validation_lycoris_strength": 1.0,
    ...the rest of your settings...
}
```


The LyCORIS configuration file is in the format:

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

### Fields

Optional fields:
- apply_preset for LycorisNetwork.apply_preset
- any keyword arguments specific to the selected algorithm, at the end.

Mandatory fields:
- multiplier, which should be set to 1.0 only unless you know what to expect
- linear_dim
- linear_alpha

For more information on LyCORIS, please refer to the [documentation in the library](https://github.com/KohakuBlueleaf/LyCORIS/tree/main/docs).

### Flux 2 (Klein) module targets

Flux 2 models use custom module classes instead of the generic `Attention` and `FeedForward` names. A Flux 2 LoKR config should target:

- `Flux2Attention` — double-stream attention blocks
- `Flux2FeedForward` — double-stream feedforward blocks
- `Flux2ParallelSelfAttention` — single-stream parallel attention+feedforward blocks (fused QKV and MLP projections)

Including `Flux2ParallelSelfAttention` trains the single-stream blocks, which may improve convergence at the cost of increased risk of overfitting. If you are having difficulty getting LyCORIS LoKR to converge on Flux 2, adding this target is recommended.

Example Flux 2 LoKR config:

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

### T-LoRA (Timestep-dependent LoRA)

T-LoRA applies timestep-dependent rank masking during training. At high noise levels (early denoising) fewer LoRA ranks are active, learning coarse structure. At low noise levels (late denoising) more ranks activate, capturing fine detail. This requires a LyCORIS version that includes `lycoris.modules.tlora`.

Example T-LoRA config:

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

Optional T-LoRA fields (added to the same JSON):

- `tlora_min_rank` (integer, default `1`) — minimum number of active ranks at the highest noise level.
- `tlora_alpha` (float, default `1.0`) — masking schedule exponent. `1.0` is linear; values above `1.0` shift more capacity toward detail steps.

> **Note:** T-LoRA with video models may produce subpar results because temporal compression blends frames across timestep boundaries.

During validation, SimpleTuner automatically applies timestep-dependent masking at each denoising step so that inference matches the training conditions. No additional configuration is needed — the masking parameters from training are reused.

## Potential problems

When using Lycoris on SDXL, it's noted that training the FeedForward modules may break the model and send loss into `NaN` (Not-a-Number) territory.

This seems to be potentially exacerbated when using SageAttention (with `--sageattention_usage=training`), making it all but guaranteed that the model will immediately fail.

The solution is to remove the `FeedForward` modules from the lycoris config and train only the `Attention` blocks.

## LyCORIS Inference Example

Here is a simple FLUX.1-dev inference script showing how to wrap your unet or transformer with create_lycoris_from_weights and then use it for inference.

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
