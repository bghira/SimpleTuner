# LoRA de una sola muestra estilo Glance

Glance es más bien un “LoRA de una sola imagen con un schedule dividido” que una canalización de destilación real. Entrenas dos LoRA diminutos sobre la misma imagen/caption: uno en los pasos más tempranos (“Slow”), otro en los pasos más tardíos (“Fast”), y luego los encadenas en inferencia.

## Lo que obtienes

- Dos LoRA entrenadas sobre una sola imagen/caption
- Timesteps de flujo personalizados en lugar de muestreo CDF (`--flow_custom_timesteps`)
- Funciona con modelos flow-matching (Flux, familia SD3, Qwen-Image, etc.)

## Requisitos previos

- Python 3.10–3.12, SimpleTuner instalado (`pip install simpletuner[cuda]`)
- Una imagen y un archivo de caption con el mismo basename (p. ej., `data/glance.png` + `data/glance.txt`)
- Un checkpoint de modelo flow (el ejemplo de abajo usa `black-forest-labs/FLUX.1-dev`)

## Paso 1 – Apunta un dataloader a tu muestra única

En `config/multidatabackend.json`, referencia tu par de imagen/texto. Un ejemplo mínimo:

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

## Paso 2 – Entrena la LoRA Slow (pasos tempranos)

Crea `config/glance-slow.json` con la lista de pasos tempranos:

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

Ejecuta:

```bash
simpletuner train --config config/glance-slow.json
```

## Paso 3 – Entrena la LoRA Fast (pasos tardíos)

Copia la config a `config/glance-fast.json`, cambia la ruta de salida y la lista de timesteps:

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

Ejecuta:

```bash
simpletuner train --config config/glance-fast.json
```

Notas:
- `--flow_custom_timesteps` sobrescribe el muestreo sigmoid/uniform/beta usual para flow-matching, y toma valores uniformemente de tu lista en cada paso.
- Deja los demás flags del schedule de flujo en sus valores predeterminados; la lista personalizada los omite.
- Si prefieres sigmas en lugar de timesteps 0–1000, proporciona valores en `[0,1]`.

## Paso 4 – Usa ambas LoRA juntas (con los sigmas correctos)

Diffusers necesita ver el mismo schedule de sigmas en el que entrenaste. Convierte las listas de timesteps 0–1000 a sigmas 0–1 y pásalos explícitamente; `num_inference_steps` debe coincidir con la longitud de cada lista.

```python
import torch
from diffusers import FluxPipeline

device = "cuda"
base_model = "black-forest-labs/FLUX.1-dev"

# Convert the training timesteps to sigmas in [0, 1]
slow_sigmas = [t / 1000.0 for t in (1000.0, 979.1915, 957.5157, 934.9171, 911.3354)]
fast_sigmas = [t / 1000.0 for t in (886.7053, 745.0728, 562.9505, 320.0802, 20.0)]
generator = torch.Generator(device=device).manual_seed(0)

# Early phase
pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
pipe.load_lora_weights("output/glance-slow")
latents = pipe(
    prompt="your prompt",
    num_inference_steps=len(slow_sigmas),
    sigmas=slow_sigmas,
    output_type="latent",
    generator=generator,
).images

# Late phase (continue the same schedule)
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

Reutiliza el mismo prompt y generator/seed para que la LoRA Fast continúe exactamente donde la LoRA Slow se detuvo, y mantén las listas de sigmas alineadas con `--flow_custom_timesteps` con los que entrenaste.
