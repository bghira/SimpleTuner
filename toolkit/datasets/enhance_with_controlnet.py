import torch
try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image
from diffusers import ControlNetModel, DiffusionPipeline
from diffusers.utils import load_image


def resize_for_condition_image(input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.bfloat16
)
pipe = DiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.0_noVAE",
    custom_pipeline="stable_diffusion_controlnet_img2img",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
).to("cuda" if torch.cuda.is_available() else "cpu")
from diffusers import DDIMScheduler

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.unet.set_attention_slice(1)
source_image = load_image(
    "/Volumes/models/training/datasets/animals/antelope/0e17715606.jpg"
)

condition_image = resize_for_condition_image(source_image, 1024)
image = pipe(
    prompt="best quality",
    negative_prompt="deformed eyes, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated",
    image=condition_image,
    controlnet_conditioning_image=condition_image,
    width=condition_image.size[0],
    height=condition_image.size[1],
    strength=1.0,
    generator=torch.manual_seed(20),
    num_inference_steps=32,
).images[0]

image.save("output.png")
