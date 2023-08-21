import torch
from diffusers import StableDiffusionKDiffusionPipeline, AutoencoderKL

pipe = StableDiffusionKDiffusionPipeline.from_pretrained(
    "/models/pseudo-test",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe.set_scheduler("sample_dpmpp_2m")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", use_safetensors=True, torch_dtype=torch.float16
)
pipe.vae = vae
pipe.to("cuda")
image = pipe(
    prompt="best quality ((bear)) in the swiss alps cinematic 8k highly detailed sharp focus intricate fur",
    negative_prompt="malformed, disgusting",
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(42),
    width=1280,
    height=720,
    guidance_scale=3.0,
    use_karras_sigmas=True,
).images[0]
image.save("bear.png", format="PNG")
