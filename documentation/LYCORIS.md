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
