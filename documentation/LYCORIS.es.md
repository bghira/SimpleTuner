# LyCORIS

## Antecedentes

[LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) es un conjunto amplio de métodos de fine-tuning eficiente en parámetros (PEFT) que te permiten afinar modelos usando menos VRAM y producir pesos más pequeños para distribuir.

## Usar LyCORIS

Para usar LyCORIS, establece `--lora_type=lycoris` y luego `--lycoris_config=config/lycoris_config.json`, donde `config/lycoris_config.json` es la ubicación de tu archivo de configuración de LyCORIS.

Lo siguiente irá en tu `config.json`:
```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "lycoris_config": "config/lycoris_config.json",
    "validation_lycoris_strength": 1.0,
    ...the rest of your settings...
}
```


El archivo de configuración de LyCORIS tiene el formato:

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

### Campos

Campos opcionales:
- apply_preset para LycorisNetwork.apply_preset
- cualquier keyword argument específico al algoritmo seleccionado, al final.

Campos obligatorios:
- multiplier, que debería configurarse en 1.0 salvo que sepas qué esperar
- linear_dim
- linear_alpha

Para más información sobre LyCORIS, consulta la [documentación en la librería](https://github.com/KohakuBlueleaf/LyCORIS/tree/main/docs).

## Problemas potenciales

Al usar Lycoris en SDXL, se ha observado que entrenar los módulos FeedForward puede romper el modelo y llevar la pérdida a valores `NaN` (Not-a-Number).

Esto parece empeorar al usar SageAttention (con `--sageattention_usage=training`), haciendo prácticamente seguro que el modelo falle de inmediato.

La solución es eliminar los módulos `FeedForward` del config de lycoris y entrenar solo los bloques `Attention`.

## Ejemplo de inferencia con LyCORIS

Aquí hay un script simple de inferencia para FLUX.1-dev que muestra cómo envolver tu unet o transformer con create_lycoris_from_weights y luego usarlo para inferencia.

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
