# Glance-Style Single Sample LoRA

Glance एक "single-image LoRA with a split schedule" जैसा है, न कि एक सच्चा distillation pipeline। आप उसी इमेज/कैप्शन पर दो छोटे LoRAs ट्रेन करते हैं: एक शुरुआती स्टेप्स ("Slow") पर, और दूसरा अंतिम स्टेप्स ("Fast") पर, फिर inference में उन्हें chain करते हैं।

## क्या मिलता है

- एक ही इमेज/कैप्शन पर ट्रेन किए गए दो LoRAs
- CDF sampling के बजाय custom flow timesteps (`--flow_custom_timesteps`)
- flow-matching मॉडल्स के साथ काम करता है (Flux, SD3-family, Qwen-Image, आदि)

## प्रीरिक्विज़िट्स

- Python 3.10–3.12, SimpleTuner इंस्टॉल (`pip install simpletuner[cuda]`)
- एक इमेज और एक कैप्शन फ़ाइल एक ही basename के साथ (उदाहरण: `data/glance.png` + `data/glance.txt`)
- एक flow-model checkpoint (नीचे उदाहरण में `black-forest-labs/FLUX.1-dev`)

## स्टेप 1 – डाटालोडर को अपने single sample पर इंगित करें

`config/multidatabackend.json` में अपने single image/text pair को रेफर करें। एक minimal उदाहरण:

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

## स्टेप 2 – Slow LoRA ट्रेन करें (early steps)

early-step सूची के साथ `config/glance-slow.json` बनाएँ:

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

चलाएँ:

```bash
simpletuner train --config config/glance-slow.json
```

## स्टेप 3 – Fast LoRA ट्रेन करें (late steps)

कॉन्फ़िग को `config/glance-fast.json` में कॉपी करें, output path और timestep list बदलें:

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

चलाएँ:

```bash
simpletuner train --config config/glance-fast.json
```

नोट्स:
- `--flow_custom_timesteps` flow-matching के लिए usual sigmoid/uniform/beta sampling को override करता है, और हर step पर आपकी सूची से uniformly चुनता है।
- अन्य flow schedule flags को डिफ़ॉल्ट पर रखें; custom list उन्हें bypass करती है।
- यदि आप 0–1000 timesteps की जगह sigmas पसंद करते हैं, तो `[0,1]` में मान दें।

## स्टेप 4 – दोनों LoRAs साथ में उपयोग करें (सही sigmas के साथ)

Diffusers को वही sigma schedule चाहिए जिस पर आपने ट्रेन किया है। 0–1000 timestep lists को 0–1 sigmas में बदलें और उन्हें स्पष्ट रूप से पास करें; `num_inference_steps` को हर सूची की लंबाई से मेल खाना चाहिए।

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

उसी prompt और generator/seed को reuse करें ताकि Fast LoRA वहीं से resume करे जहाँ Slow LoRA रुका था, और sigma lists को `--flow_custom_timesteps` के साथ align रखें।
