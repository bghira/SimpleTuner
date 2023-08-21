# Use Pytorch 2!
import torch
from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTextModel

# Any model currently on Huggingface Hub.
# model_id = 'junglerally/digital-diffusion'
# model_id = 'ptx0/realism-engine'
# model_id = 'ptx0/artius_v21'
# model_id = 'ptx0/pseudo-journey'
model_id = "ptx0/pseudo-journey-v2"
pipeline = DiffusionPipeline.from_pretrained(model_id)

# Optimize!
pipeline.unet = torch.compile(pipeline.unet)
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Remove this if you get an error.
torch.set_float32_matmul_precision("high")

pipeline.to("cuda")
prompts = {
    "woman": "a woman, hanging out on the beach",
    "man": "a man playing guitar in a park",
    "lion": "Explore the ++majestic beauty++ of untamed ++lion prides++ as they roam the African plains --captivating expressions-- in the wildest national geographic adventure",
    "child": "a child flying a kite on a sunny day",
    "bear": "best quality ((bear)) in the swiss alps cinematic 8k highly detailed sharp focus intricate fur",
    "alien": "an alien exploring the Mars surface",
    "robot": "a robot serving coffee in a cafe",
    "knight": "a knight protecting a castle",
    "menn": "a group of smiling and happy men",
    "bicycle": "a bicycle, on a mountainside, on a sunny day",
    "cosmic": "cosmic entity, sitting in an impossible position, quantum reality, colours",
    "wizard": "a mage wizard, bearded and gray hair, blue  star hat with wand and mystical haze",
    "wizarddd": "digital art, fantasy, portrait of an old wizard, detailed",
    "macro": "a dramatic city-scape at sunset or sunrise",
    "micro": "RNA and other molecular machinery of life",
    "gecko": "a leopard gecko stalking a cricket",
}
for shortname, prompt in prompts.items():
    # old prompt: ''
    image = pipeline(
        prompt=prompt,
        negative_prompt="malformed, disgusting, overexposed, washed-out",
        num_inference_steps=32,
        generator=torch.Generator(device="cuda").manual_seed(1641421826),
        width=1152,
        height=768,
        guidance_scale=7.5,
    ).images[0]
    image.save(f"test/{shortname}_nobetas.png", format="PNG")
