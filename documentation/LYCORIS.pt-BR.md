# LyCORIS

## Contexto

[LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) e uma suite extensa de metodos de fine-tuning eficiente em parametros (PEFT) que permite ajustar modelos usando menos VRAM e produz pesos menores para distribuicao.

## Usando LyCORIS

Para usar LyCORIS, defina `--lora_type=lycoris` e depois `--lycoris_config=config/lycoris_config.json`, onde `config/lycoris_config.json` e o local do seu arquivo de configuracao do LyCORIS.

O seguinte vai no seu `config.json`:
```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "lycoris_config": "config/lycoris_config.json",
    "validation_lycoris_strength": 1.0,
    ...the rest of your settings...
}
```


O arquivo de configuracao do LyCORIS tem o formato:

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

Campos opcionais:
- apply_preset para LycorisNetwork.apply_preset
- quaisquer argumentos de palavra-chave especificos do algoritmo selecionado, no final.

Campos obrigatorios:
- multiplier, que deve ser definido como 1.0 apenas, a menos que voce saiba o que esperar
- linear_dim
- linear_alpha

Para mais informacoes sobre LyCORIS, consulte a [documentacao na biblioteca](https://github.com/KohakuBlueleaf/LyCORIS/tree/main/docs).

## Problemas potenciais

Ao usar Lycoris no SDXL, foi observado que treinar os modulos FeedForward pode quebrar o modelo e levar a loss para valores `NaN` (Not-a-Number).

Isso parece ser potencializado ao usar SageAttention (com `--sageattention_usage=training`), tornando quase garantido que o modelo falhe imediatamente.

A solucao e remover os modulos `FeedForward` do config do Lycoris e treinar apenas os blocos `Attention`.

## Exemplo de inferencia com LyCORIS

Aqui vai um script simples de inferencia do FLUX.1-dev mostrando como envolver seu unet ou transformer com create_lycoris_from_weights e depois usar isso para inferencia.

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
