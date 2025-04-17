import os
import json
import time
import random
import logging
import requests
import pandas as pd
from tqdm import tqdm
from io import BytesIO
import pillow_jxl
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Configuration and setup
http = requests.Session()
timeouts = (6, 60)
output_path = "/Volumes/ml/datasets/pixilart"
output_dataframe_path = "pixilart.parquet"
subsets = ["highlighted", "rising", "popular", "featured"]
max_workers = 5
number_of_random_values = 1000
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)


# Function to fetch data from Pixilart API
def pixilart(sequence: int, subset: str, x_csrf_token: str, x_xsrf_token: str):
    url = f"https://www.pixilart.com/api/w/gallery/{sequence}/0/{subset}?user=true&liked=true&comments=true"
    headers = {
        "accept": "application/json, text/plain, */*",
        "cookie": 'pa_st={"challenge":0}; XSRF-TOKEN='
        + x_csrf_token
        + "; pixil_session="
        + x_xsrf_token,
        "dnt": "1",
        "priority": "u=1, i",
        "referer": "https://www.pixilart.com/gallery/staff-picks",
        "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "sec-gpc": "1",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "x-csrf-token": x_csrf_token,
        "x-requested-with": "XMLHttpRequest",
        "x-xsrf-token": x_xsrf_token,
    }

    for attempt in range(3):
        try:
            response = http.get(url, headers=headers, timeout=timeouts)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    return None


# Function to download images
def download_image(url, filename):
    response = requests.get(url)
    with BytesIO(response.content) as img_data:
        image = Image.open(img_data)
        image.save(filename)


# Load or initialize dataframe
if os.path.exists(output_dataframe_path):
    df = pd.read_parquet(output_dataframe_path)
else:
    df = pd.DataFrame(
        columns=[
            "subset",
            "sequence",
            "subset-sequence-element",
            "title",
            "description",
            "views",
            "filename",
            "pixel_size",
            "has_watermark",
            "image_hash",
            "image_url",
            "full_image_url",
            "likes_count",
            "comments_count",
            "width",
            "height",
            "date_created",
            "content_warning",
            "warning",
            "liked",
        ]
    )

# Generate random sequence values
random_values = range(1, number_of_random_values + 1)

# Process sequences and subsets
records_to_add = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for sequence in tqdm(random_values, desc="Sequences", position=1):
        for subset in tqdm(subsets, desc="Subsets", position=2, leave=False):
            if df[(df["subset"] == subset) & (df["sequence"] == sequence)].empty:
                result = pixilart(
                    sequence, subset, "your_x_csrf_token", "your_x_xsrf_token"
                )
                if result and "art" in result:
                    for idx, element in enumerate(result["art"], start=1):
                        record = {
                            "subset": subset,
                            "sequence": sequence,
                            "subset-sequence-element": f"{subset}.{sequence}.{idx}",
                            "title": element["title"],
                            "description": element["description"],
                            "filename": f"{element['image_id']}-{idx} {element['title']}.png",
                            "views": element["views"],
                            "image_hash": element["image_id"],
                            "image_url": element["image_url"],
                            "full_image_url": element["full_image_url"],
                            "likes_count": element["likes_count"],
                            "pixel_size": element.get("pixel_size", 0),
                            "has_watermark": element.get("has_watermark", False),
                            "comments_count": element["comments_count"],
                            "width": element["width"],
                            "height": element["height"],
                            "date_created": element["date_created"],
                            "content_warning": element.get("content_warning"),
                            "warning": str(element["warning"]),
                            "liked": element["liked"],
                        }
                        records_to_add.append(record)
                        image_filename = os.path.join(output_path, record["filename"])
                        executor.submit(
                            download_image, record["full_image_url"], image_filename
                        )

# Update dataframe and save to parquet
if records_to_add:
    new_records_df = pd.DataFrame(records_to_add)
    df = pd.concat([df, new_records_df], ignore_index=True)
    df.drop_duplicates(subset=["full_image_url"], inplace=True)
    df.to_parquet(output_dataframe_path)
