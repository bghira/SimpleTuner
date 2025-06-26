###
# This script will generate images for the same seed/prompt across many models and stitch the outputs together.
###

from diffusers import AutoPipelineForText2Image
from torch import manual_seed, float16
import os

try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image, ImageDraw, ImageFont
from helpers.prompts import prompts

# Define your pipelines and settings in a list of dictionaries
pipelines_info = [
    {
        "label": "velocity-v1",
        "pretrained_model": "ptx0/terminus-xl-velocity-v1",
        "settings": {
            "guidance_scale": 8.0,
            "guidance_rescale": 0.7,
            "num_inference_steps": 30,
            "negative_prompt": "blurry, cropped, ugly, upscaled",
        },
    },
    {
        "label": "gamma-v1",
        "pretrained_model": "ptx0/terminus-xl-gamma-v1",
        "settings": {
            "guidance_scale": 8.0,
            "guidance_rescale": 0.7,
            "num_inference_steps": 30,
            "negative_prompt": "blurry, cropped, ugly, upscaled",
        },
    },
    {
        "label": "gamma-v2",
        "pretrained_model": "ptx0/terminus-xl-gamma-v2",
        "settings": {
            "guidance_scale": 8.0,
            "guidance_rescale": 0.7,
            "num_inference_steps": 30,
            "negative_prompt": "blurry, cropped, ugly, upscaled",
        },
    },
    {
        "label": "otaku-v1",
        "pretrained_model": "ptx0/terminus-xl-otaku-v1",
        "settings": {
            "guidance_scale": 8.0,
            "guidance_rescale": 0.7,
            "num_inference_steps": 30,
            "negative_prompt": "blurry, cropped, ugly, upscaled",
        },
    },
    {
        "label": "gamma-training",
        "pretrained_model": "ptx0/terminus-xl-gamma-training",
        "settings": {
            "guidance_scale": 8.0,
            "guidance_rescale": 0.7,
            "num_inference_steps": 30,
            "negative_prompt": "blurry, cropped, ugly, upscaled",
        },
    },
    {
        "label": "gamma-v2-1",
        "pretrained_model": "ptx0/terminus-xl-gamma-v2-1",
        "settings": {
            "guidance_scale": 8.0,
            "guidance_rescale": 0.7,
            "num_inference_steps": 30,
            "negative_prompt": "blurry, cropped, ugly, upscaled",
        },
    },
    # {"label": "v2.1+LoRA", "pretrained_model": "ptx0/terminus-xl-gamma-v2-1", "lora": {"weights": "ptx0/simpletuner-lora-test", "weight_name": "pytorch_lora_weights.safetensors"}, "settings": {"guidance_scale": 8.0, "guidance_rescale": 0.7, "num_inference_steps": 30, "negative_prompt": "blurry, cropped, ugly, upscaled"}},
]


def combine_and_label_images(images_info, output_path):
    # Assume images_info is a list of tuples: (Image object, label)
    # Initial setup based on the first image dimensions and number of images
    label_height = 45
    total_width = sum(image.width for image, _ in images_info)
    max_height = max(image.height for image, _ in images_info) + label_height
    combined_image = Image.new("RGB", (total_width, max_height), "white")

    # Combine images and labels
    current_x = 0
    for image, label in images_info:
        combined_image.paste(image, (current_x, label_height))
        current_x += image.width

    # Adding labels using a uniform method for text placement
    draw = ImageDraw.Draw(combined_image)
    try:
        # Attempt to use a specific font
        font = ImageFont.truetype(
            ".venv/lib/python3.11/site-packages/cv2/qt/fonts/DejaVuSans.ttf", 40
        )  # Adjust font path and size
    except IOError:
        # Fallback to default font
        font = ImageFont.load_default()

    current_x = 0
    for _, label in images_info:
        draw.text((current_x + 10, 2), label, fill="black", font=font)
        current_x += image.width

    combined_image.save(output_path)


# Processing pipelines
base_pipeline = AutoPipelineForText2Image.from_pretrained(
    "ptx0/terminus-xl-gamma-v2", torch_dtype=float16
).to("cuda")
text_encoder_1 = base_pipeline.components["text_encoder"]
text_encoder_2 = base_pipeline.components["text_encoder_2"]
vae = base_pipeline.components["vae"]
for shortname, prompt in prompts.items():
    print(f"Processing: {shortname}")
    target_dir = f"inference/images/{shortname}"
    # Does the combined image exist? Skip it then.
    if os.path.exists(f"{target_dir}/combined_image.png"):
        continue
    os.makedirs(target_dir, exist_ok=True)

    images_info = []
    for pipeline_info in pipelines_info:
        image_path = f'{target_dir}/image-{pipeline_info["label"].replace("+", "plus").lower()}.png'
        if os.path.exists(image_path):
            continue
        # Initialize pipeline
        pipeline = AutoPipelineForText2Image.from_pretrained(
            pipeline_info["pretrained_model"],
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            vae=vae,
            torch_dtype=float16,
        ).to("cuda")

        # Load LoRA weights if specified
        if "lora" in pipeline_info:
            pipeline.load_lora_weights(
                pipeline_info["lora"]["weights"],
                weight_name=pipeline_info["lora"]["weight_name"],
            )

        # Generate image with specified settings
        settings = pipeline_info.get("settings", {})
        image = pipeline(prompt, generator=manual_seed(420420420), **settings).images[0]
        # Unload LoRA weights if they were loaded
        if "lora" in pipeline_info:
            pipeline.unload_lora_weights()
        del pipeline
        image.save(image_path, format="PNG")

        images_info.append((image, pipeline_info["label"]))

    # Combine and label images
    combine_and_label_images(images_info, f"{target_dir}/combined_image.png")
