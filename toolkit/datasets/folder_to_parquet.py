"""
This script exists to scan a folder and import all of the image data into equally sized parquet files.

Fields collected:

filename
image hash
width
height
luminance
image_data
"""

import os, argparse
import pillow_jxl
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_image_hash(image):
    """Calculate the hash of an image."""
    image = image.convert("L").resize((8, 8))
    pixels = list(image.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel > avg else "0" for pixel in pixels)
    return int(bits, 2)


def get_image_luminance(image):
    """Calculate the luminance of an image."""
    image = image.convert("L")
    pixels = list(image.getdata())
    return sum(pixels) / len(pixels)


def get_size(image):
    """Get the size of an image."""
    return image.size


argparser = argparse.ArgumentParser()
argparser.add_argument("input_folder", help="Folder to scan for images")
argparser.add_argument("output_folder", help="Folder to save parquet files")

args = argparser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

data = []
for root, _, files in os.walk(args.input_folder):
    for file in tqdm(files, desc="Processing images"):
        try:
            image = Image.open(os.path.join(root, file))
        except:
            continue

        width, height = get_size(image)
        luminance = get_image_luminance(image)
        image_hash = get_image_hash(image)
        # Get the smallest original compressed representation of the image
        file_data = open(os.path.join(root, file), "rb").read()
        image_data = np.frombuffer(file_data, dtype=np.uint8)

        data.append((file, image_hash, width, height, luminance, image_data))

df = pd.DataFrame(
    data, columns=["filename", "image_hash", "width", "height", "luminance", "image"]
)
df.to_parquet(os.path.join(args.output_folder, "images.parquet"), index=False)

print("Done!")
