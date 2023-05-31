import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel

model_id = 'ptx0/pseudo-journey-v2'
pipeline = DiffusionPipeline.from_pretrained(model_id)

# Optimize!
pipeline.unet = torch.compile(pipeline.unet)
def enforce_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

     # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
    alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
     # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas

def patch_scheduler_betas(scheduler):
    scheduler.betas = enforce_zero_terminal_snr(scheduler.betas)
    return scheduler

scheduler = DDPMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler"
)
#pipeline.scheduler = patch_scheduler_betas(scheduler)
torch.set_float32_matmul_precision('high')
pipeline.to('cuda')
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
    "gecko": "a leopard gecko stalking a cricket"
}
for shortname, prompt in prompts.items():
    # old prompt: ''
    image = pipeline(prompt=prompt,
        negative_prompt='malformed, disgusting, overexposed, washed-out',
        num_inference_steps=32, generator=torch.Generator(device='cuda').manual_seed(1641421826), 
        width=1152, height=768, guidance_scale=7.5).images[0]
    image.save(f'test/{shortname}_nobetas.png', format="PNG")