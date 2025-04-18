"""
Walk through a LAION dataset and analyze it.
"""

import os
import json
import concurrent.futures
try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image


def get_aspect_ratio(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return image_path, width / height
    except Exception as e:
        os.remove(image_path)
        return None


def analyze_images(directory_path):
    aspect_ratios = {}
    good_count = 0
    bad_count = 0
    image_files = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.endswith(".jpg") or filename.endswith(".png")
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = {
            executor.submit(get_aspect_ratio, image_path): image_path
            for image_path in image_files
        }
        for future in concurrent.futures.as_completed(futures):
            image_path = futures[future]
            try:
                aspect_ratio = future.result()
                if aspect_ratio is not None:
                    image_path, aspect_ratio = aspect_ratio
                    aspect_ratio = round(aspect_ratio, 2)  # round to 2 decimal places
                    if aspect_ratio not in aspect_ratios:
                        aspect_ratios[aspect_ratio] = []
                    aspect_ratios[aspect_ratio].append(image_path)
                    good_count += 1
                else:
                    bad_count += 1
            except Exception as e:
                pass
    print(f"Good images: {good_count}, Bad images: {bad_count}")
    return aspect_ratios


def write_to_json(data, filename):
    with open(filename, "w") as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    image_directory = "/notebooks/datasets/laion-high-resolution/downloaded_images"
    output_file = "aspect_ratios.json"
    aspect_ratios = analyze_images(image_directory)
    write_to_json(aspect_ratios, output_file)
