import json, os
import threading, logging
from concurrent.futures import ThreadPoolExecutor

import pillow_jxl
from PIL import Image

# Allowed bucket values:
allowed = [1.0, 1.5, 0.67, 0.75, 1.78]

# Load from JSON.
with open("aspect_ratios.json", "r") as f:
    aspect_ratios = json.load(f)

new_bucket = {}
for bucket, indices in aspect_ratios.items():
    logging.info(f"{bucket}: {len(indices)}")
    if float(bucket) in allowed:
        logging.info(f"Bucket {bucket} in {allowed}")
        new_bucket[bucket] = aspect_ratios[bucket]

least_amount = None
for bucket, indices in aspect_ratios.items():
    if least_amount is None or len(indices) < least_amount:
        least_amount = len(indices)

# We don't want to limit square image training.
# buckets_to_skip = [ 1.0 ]
# for bucket, files in aspect_ratios.items():
#     if float(bucket) not in buckets_to_skip and len(files) > least_amount:
#         logging.info(f'We have to reduce the number of items in the bucket: {bucket}')
#         # 'files' is a list of full file paths. we need to delete them randomly until the value of least_amount is reached.


#         # Get a random sample of files to delete.
#         import random
#         random.shuffle(files)
#         files_to_delete = files[least_amount:]
#         logging.info(f'Files to delete: {len(files_to_delete)}')
#         for file in files_to_delete:
#             import os
#             os.remove(file)
def _resize_for_condition_image(self, input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def process_file(file):
    image = Image.open(file).convert("RGB")
    width, height = image.size
    if width < 900 or height < 900:
        logging.info(
            f"Image does not meet minimum size requirements: {file}, size {image.size}"
        )
        os.remove(file)
    else:
        logging.info(
            f"Image meets minimum size requirements for conditioning: {file}, size {image.size}"
        )
        image = _resize_for_condition_image(image, 1024)
        image.save(file)


def process_bucket(bucket, files):
    logging.info(f"Processing bucket {bucket}: {len(files)} files")
    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(process_file, files)


if __name__ == "__main__":
    # Load aspect ratios from the JSON file
    with open("aspect_ratios.json", "r") as f:
        aspect_ratios = json.load(f)

    threads = []
    for bucket, files in aspect_ratios.items():
        thread = threading.Thread(target=process_bucket, args=(bucket, files))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
