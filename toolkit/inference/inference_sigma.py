import torch
from diffusers import Transformer2DModel
from sigma import pixart_sigma_init_patched_inputs, PixArtSigmaPipeline

setattr(Transformer2DModel, "_init_patched_inputs", pixart_sigma_init_patched_inputs)
device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

transformer = Transformer2DModel.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    subfolder="transformer",
    use_safetensors=True,
)
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    transformer=transformer,
    use_safetensors=True,
)
pipe.to(device=device, dtype=torch.bfloat16)

# Enable memory optimizations.
# pipe.enable_model_cpu_offload()

prompt = "A small cactus with a happy face in the Sahara desert."
image = pipe(prompt).images[0]
image.save("./catcus.png")
