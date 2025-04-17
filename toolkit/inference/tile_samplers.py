import pillow_jxl
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

# Placeholder URL
url = "https://sa1s3optim.patientpop.com/assets/images/provider/photos/2353184.jpg"

# Download the image from the URL
response = requests.get(url)
original_image = Image.open(BytesIO(response.content))

# Define target size (1 megapixel)
target_width = 1000
target_height = 1000

# Resize the image using different samplers
samplers = {
    "NEAREST": Image.NEAREST,
    "BOX": Image.BOX,
    "HAMMING": Image.HAMMING,
    "BILINEAR": Image.BILINEAR,
    "BICUBIC": Image.BICUBIC,
    "LANCZOS": Image.LANCZOS,
}

# Create a new image to combine the results
combined_width = target_width * len(samplers)
combined_height = target_height + 50  # Extra space for labels
combined_image = Image.new("RGB", (combined_width, combined_height), "white")
draw = ImageDraw.Draw(combined_image)

# Load a default font
try:
    font = ImageFont.load_default()
except IOError:
    font = None

# Resize and add each sampler result to the combined image
for i, (label, sampler) in enumerate(samplers.items()):
    resized_image = original_image.resize((target_width, target_height), sampler)
    combined_image.paste(resized_image, (i * target_width, 50))

    # Draw the label
    text_position = (i * target_width + 20, 15)
    draw.text(text_position, label, fill="black", font=font)

# Save or display the combined image
combined_image_path = "downsampled_image_comparison.png"
combined_image.save(combined_image_path)
combined_image_path
