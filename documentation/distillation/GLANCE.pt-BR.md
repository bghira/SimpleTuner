# LoRA de amostra unica estilo Glance

Glance e mais um “LoRA de imagem unica com schedule dividido” do que um pipeline de destilacao de verdade. Voce treina dois LoRAs pequenos na mesma imagem/caption: um nos steps iniciais (“Slow”), outro nos steps finais (“Fast”), e depois encadeia os dois na inferencia.

## O que voce ganha

- Dois LoRAs treinados em uma unica imagem/caption
- Timesteps de fluxo customizados em vez de amostragem CDF (`--flow_custom_timesteps`)
- Funciona com modelos flow-matching (Flux, familia SD3, Qwen-Image, etc.)

## Pre-requisitos

- Python 3.10–3.13, SimpleTuner instalado (`pip install 'simpletuner[cuda]'`)
  - CUDA 13 / Blackwell users (NVIDIA B-series GPUs): `pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130`
- Uma imagem e um arquivo de caption com o mesmo basename (ex.: `data/glance.png` + `data/glance.txt`)
- Um checkpoint de modelo de flow (o exemplo abaixo usa `black-forest-labs/FLUX.1-dev`)

## Passo 1 - Aponte um dataloader para sua unica amostra

Em `config/multidatabackend.json`, referencie seu par unico de imagem/texto. Um exemplo minimo:

```json
[
  {
    "backend": "simple",
    "resolution": 1024,
    "shuffle": true,
    "image_dir": "data",
    "caption_extension": ".txt"
  }
]
```

## Passo 2 - Treine o LoRA Slow (steps iniciais)

Crie `config/glance-slow.json` com a lista de steps iniciais:

```json
{
  "--model_type": "lora",
  "--model_family": "flux",
  "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
  "--data_backend_config": "config/multidatabackend.json",
  "--train_batch_size": 1,
  "--max_train_steps": 60,
  "--lora_rank": 32,
  "--output_dir": "output/glance-slow",
  "--flow_custom_timesteps": "1000,979.1915,957.5157,934.9171,911.3354"
}
```

Rode:

```bash
simpletuner train --config config/glance-slow.json
```

## Passo 3 - Treine o LoRA Fast (steps finais)

Copie o config para `config/glance-fast.json`, mude o caminho de saida e a lista de timesteps:

```json
{
  "--model_type": "lora",
  "--model_family": "flux",
  "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
  "--data_backend_config": "config/multidatabackend.json",
  "--train_batch_size": 1,
  "--max_train_steps": 60,
  "--lora_rank": 32,
  "--output_dir": "output/glance-fast",
  "--flow_custom_timesteps": "886.7053,745.0728,562.9505,320.0802,20.0"
}
```

Rode:

```bash
simpletuner train --config config/glance-fast.json
```

Notas:
- `--flow_custom_timesteps` sobrescreve a amostragem sigmoid/uniform/beta habitual para flow-matching e escolhe uniformemente da sua lista a cada step.
- Deixe as outras flags de flow schedule nos valores padrao; a lista customizada ignora elas.
- Se voce preferir sigmas em vez de timesteps 0-1000, forneca valores em `[0,1]`.

## Passo 4 - Use ambos os LoRAs juntos (com os sigmas corretos)

O Diffusers precisa ver o mesmo schedule de sigmas usado no treino. Converta as listas de timesteps 0-1000 em sigmas 0-1 e passe-as explicitamente; `num_inference_steps` deve corresponder ao tamanho de cada lista.

```python
import torch
from diffusers import FluxPipeline

device = "cuda"
base_model = "black-forest-labs/FLUX.1-dev"

# Converta os timesteps de treino para sigmas em [0, 1]
slow_sigmas = [t / 1000.0 for t in (1000.0, 979.1915, 957.5157, 934.9171, 911.3354)]
fast_sigmas = [t / 1000.0 for t in (886.7053, 745.0728, 562.9505, 320.0802, 20.0)]
generator = torch.Generator(device=device).manual_seed(0)

# Fase inicial
pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
pipe.load_lora_weights("output/glance-slow")
latents = pipe(
    prompt="your prompt",
    num_inference_steps=len(slow_sigmas),
    sigmas=slow_sigmas,
    output_type="latent",
    generator=generator,
).images

# Fase final (continue o mesmo schedule)
pipe_fast = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
pipe_fast.load_lora_weights("output/glance-fast")
image = pipe_fast(
    prompt="your prompt",
    num_inference_steps=len(fast_sigmas),
    sigmas=fast_sigmas,
    latents=latents,
    guidance_scale=1.0,
    generator=generator,
).images[0]
image.save("glance.png")
```

Reutilize o mesmo prompt e generator/seed para que o LoRA Fast retome exatamente onde o LoRA Slow parou, e mantenha as listas de sigma alinhadas com o `--flow_custom_timesteps` usado no treino.
