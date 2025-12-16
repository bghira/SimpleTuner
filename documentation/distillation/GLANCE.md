# Glance-Style Single Sample LoRA

Glance is more of a “single-image LoRA with a split schedule” than a true distillation pipeline. You train two tiny LoRAs on the same image/caption: one on the earliest steps (“Slow”), one on the latest steps (“Fast”), then chain them at inference.

## What you get

- Two LoRAs trained on a single image/caption
- Custom flow timesteps instead of CDF sampling (`--flow_custom_timesteps`)
- Works with flow-matching models (Flux, SD3-family, Qwen-Image, etc.)

## Prerequisites

- Python 3.10–3.12, SimpleTuner installed (`pip install simpletuner[cuda]`)
- One image and one caption file with the same basename (e.g., `data/glance.png` + `data/glance.txt`)
- A flow-model checkpoint (example below uses `black-forest-labs/FLUX.1-dev`)

## Step 1 – Point a dataloader at your single sample

In `config/multidatabackend.json`, reference your single image/text pair. A minimal example:

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

## Step 2 – Train the Slow LoRA (early steps)

Create `config/glance-slow.json` with the early-step list:

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

Run:

```bash
simpletuner train --config config/glance-slow.json
```

## Step 3 – Train the Fast LoRA (late steps)

Copy the config to `config/glance-fast.json`, change the output path and timestep list:

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

Run:

```bash
simpletuner train --config config/glance-fast.json
```

Notes:
- `--flow_custom_timesteps` overrides the usual sigmoid/uniform/beta sampling for flow-matching, and picks uniformly from your list each step.
- Leave the other flow schedule flags at defaults; the custom list bypasses them.
- If you prefer sigmas instead of 0–1000 timesteps, provide values in `[0,1]`.

## Step 4 – Use both LoRAs together

At inference you run the base model twice: once with the Slow LoRA for the early steps (output latents), then reload/switch to the Fast LoRA and finish the remaining steps. A minimal Diffusers-style sketch:

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")

# Early phase
pipe.load_lora_weights("output/glance-slow")
latents = pipe(prompt="your prompt", num_inference_steps=5, output_type="latent").images

# Late phase
pipe_fast = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe_fast.load_lora_weights("output/glance-fast")
image = pipe_fast(prompt="your prompt", num_inference_steps=5, latents=latents, guidance_scale=1.0).images[0]
image.save("glance.png")
```

Keep the same prompt and seed between phases so the Fast LoRA continues from the Slow output. That’s the whole trick: two LoRAs, fixed timesteps, single image/caption. Nothing fancy—just a fast way to recreate the Glance-style split without a teacher.
