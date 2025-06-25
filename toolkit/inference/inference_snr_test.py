import torch

try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    DDIMScheduler,
)
from transformers import CLIPTextModel
from helpers.prompts import prompts

model_id = "/notebooks/datasets/models/pseudo-realism"
# model_id = 'stabilityai/stable-diffusion-2-1'
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.unet = torch.compile(pipe.unet)

scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
torch.set_float32_matmul_precision("high")
pipe.to("cuda")
negative_prompt = "cropped, out-of-frame, low quality, low res, oorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, synthetic, rendering"


def create_image_grid(images, ncols):
    # Assuming all images are the same size, get dimensions of the first image
    width, height = images[0].size

    # Create a new image of size that can fit all the small images in a grid
    grid_image = Image.new(
        "RGB", (width * ncols, height * ((len(images) + ncols - 1) // ncols))
    )

    # Loop through all images and paste them into the grid image
    for index, image in enumerate(images):
        row = index // ncols
        col = index % ncols
        grid_image.paste(image, (col * width, row * height))

    return grid_image


all_images = []

for shortname, prompt in prompts.items():
    for TIMESTEP_TYPE in ["trailing", "leading"]:
        for RESCALE_BETAS_ZEROS_SNR in [True, False]:
            for GUIDANCE_RESCALE in [0, 0.3, 0.5, 0.7]:
                for GUIDANCE in [5, 6, 7, 8, 9]:
                    pipe.scheduler = DDIMScheduler.from_config(
                        pipe.scheduler.config,
                        timestep_spacing=TIMESTEP_TYPE,
                        rescale_betas_zero_snr=RESCALE_BETAS_ZEROS_SNR,
                    )
                    generator = torch.Generator(device="cpu").manual_seed(0)
                    image = pipe(
                        prompt=prompt,
                        width=1152,
                        height=768,
                        negative_prompt=negative_prompt,
                        generator=generator,
                        num_images_per_prompt=1,
                        num_inference_steps=50,
                        guidance_scale=GUIDANCE,
                        guidance_rescale=GUIDANCE_RESCALE,
                    ).images[0]
                    image_path = f"test/{shortname}_{TIMESTEP_TYPE}_{RESCALE_BETAS_ZEROS_SNR}_{GUIDANCE}g_{GUIDANCE_RESCALE}r.png"
                    image.save(image_path, format="PNG")
                    all_images.append(Image.open(image_path))

# create image grid after all images are generated
image_grid = create_image_grid(all_images, 4)  # 4 is the number of columns
image_grid.save("image_comparison_grid.png", format="PNG")
