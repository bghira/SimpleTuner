import pillow_jxl
from PIL import Image
import os, logging
import csv
import shutil
import requests
import re
import sys
import piexif
from concurrent.futures import ThreadPoolExecutor


def calculate_luminance(img: Image):
    pixels = list(img.getdata())

    luminance_values = []
    for pixel in pixels:
        r, g, b = pixel[:3]  # Assuming the image is RGB or RGBA
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        luminance_values.append(luminance)

    # Return average luminance for the entire image
    avg_luminance = sum(luminance_values) / len(luminance_values)
    return avg_luminance


def get_camera_model(img):
    try:
        if "photoshop" in img.info:
            return None
    except:
        pass
    try:
        exif_data = piexif.load(img.info["exif"])
    except:
        return None

    if piexif.ImageIFD.Model in exif_data["0th"]:
        camera_model = exif_data["0th"][piexif.ImageIFD.Model]
        print(f"Camera Model: {camera_model}")
        return str(camera_model)
    else:
        return None


# Constants
FILE = "dataset.csv"  # The CSV file to read data from
OUTPUT_DIR = "/notebooks/datasets/camera"  # Directory to save images
NUM_WORKERS = 8  # Number of worker threads for parallel downloading
logger = logging.getLogger("root")
logger.setLevel(logging.INFO)
# Check if output directory exists, create if it does not
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except Exception as e:
        logging.info(f"Could not create output directory: {e}")
        sys.exit(1)

# Check if CSV file exists
if not os.path.exists(FILE):
    logging.info(f"Could not find CSV file: {FILE}")
    sys.exit(1)

import random


def content_to_filename(content):
    """
    Function to convert content to filename by stripping everything after '--',
    replacing non-alphanumeric characters and spaces, converting to lowercase,
    removing leading/trailing underscores, and limiting filename length to 128.
    """
    # Split on '--' and take the first part
    content = content.split("--", 1)[0]
    # Split on 'Upscaled by' and take the first part
    content = content.split(" - Upscaled by", 1)[0]
    # Remove URLs
    cleaned_content = re.sub(r"https*://\S*", "", content)

    # Replace non-alphanumeric characters and spaces, convert to lowercase, remove leading/trailing underscores
    cleaned_content = re.sub(r"[^a-zA-Z0-9 ]", "", cleaned_content)
    cleaned_content = cleaned_content.replace(" ", "_").lower().strip("_")

    # If cleaned_content is empty after removing URLs, generate a random filename
    if cleaned_content == "":
        cleaned_content = f"midjourney_{random.randint(0, 1000000)}"

    # Limit filename length to 128
    cleaned_content = (
        cleaned_content[:128] if len(cleaned_content) > 128 else cleaned_content
    )

    return cleaned_content + ".png"


def load_csv(file):
    """
    Function to load CSV data into a list of dictionaries
    """
    data = []
    with open(file, newline="") as csvfile:
        try:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        except Exception as e:
            logging.info(f"Could not advance reader: {e}")
    return data


conn_timeout = 6
read_timeout = 60
timeouts = (conn_timeout, read_timeout)


def _resize_for_condition_image(input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    aspect_ratio = round(W / H, 2)
    msg = f"Inspecting image of aspect {aspect_ratio} and size {W}x{H} to "
    if W < H:
        W = resolution
        H = int(resolution / aspect_ratio)  # Calculate the new height
    elif H < W:
        H = resolution
        W = int(resolution * aspect_ratio)  # Calculate the new width
    if W == resolution and H == resolution:
        logging.debug(f"Returning square image of size {resolution}x{resolution}")
        return input_image
    if W == H:
        W = resolution
        H = resolution
    msg = f"{msg} {W}x{H}."
    logging.info(msg)
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def fetch_image(info):
    """
    Function to fetch image from a URL and save it to disk if it is square.
    Enhanced to handle exceptions robustly.
    """
    filename = info["filename"]
    url = info["url"]
    current_file_path = os.path.join(OUTPUT_DIR, filename)

    # Skip download if file already exists
    if os.path.exists(current_file_path):
        logging.info(f"{filename} already exists, skipping download...")
        return

    try:
        r = requests.get(url, timeout=timeouts, stream=True)
        if r.status_code != 200:
            logging.warn(
                f"Failed to fetch {filename} from {url}. Status code: {r.status_code}"
            )
            return

        with open(current_file_path, "wb") as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

        image = Image.open(current_file_path)

        if image.width < 1024 or image.height < 1024:
            logging.info(f"Skipping small image ({image.width}x{image.height}).")
            image.close()
            os.remove(current_file_path)
            return

        camera_model = f"{get_camera_model(image)}".lower()
        if camera_model is None or camera_model == "none":
            image.close()
            os.remove(current_file_path)
            return

        if (
            "nikon" not in camera_model
            and "canon" not in camera_model
            and "fujifilm" not in camera_model
        ):
            logging.info(f"Skipping non-Canon/Nikon/Fujifilm: {camera_model}")
            image.close()
            os.remove(current_file_path)
            return

        current_luminance = calculate_luminance(image)
        if current_luminance > 0.12:
            logging.info(f"Skipping bright image ({current_luminance}).")
            image.close()
            os.remove(current_file_path)
            return

        logging.info(f"image luminance: {current_luminance}")
        image = _resize_for_condition_image(image, 1024)
        image.save(current_file_path, format="PNG")
        image.close()

        with open(os.path.join(OUTPUT_DIR, filename + ".txt"), "w") as f:
            f.write(info["TEXT"])

    except requests.RequestException as e:
        logging.error(f"Error fetching {filename} from {url}: {e}")
        if os.path.exists(current_file_path):
            os.remove(current_file_path)

    except Exception as e:
        logging.error(f"Unexpected error processing {filename}: {e}")
        if os.path.exists(current_file_path):
            os.remove(current_file_path)


def fetch_data(data):
    """
    Function to fetch all images specified in data
    """
    to_fetch = {}
    count = 0
    for row in data:
        try:
            if (
                float(int(row["WIDTH"])) < 1024
                or float(int(row["HEIGHT"])) < 1024
                or float(row["pwatermark"]) > 0.4
            ):
                continue
        except Exception as e:
            continue
        new_filename = content_to_filename(row["TEXT"])
        if new_filename not in to_fetch and count < 100000:
            to_fetch = {
                content_to_filename(row["TEXT"]): {
                    "url": row["URL"],
                    "filename": content_to_filename(row["TEXT"]),
                    "TEXT": row["TEXT"],
                }
                for row in data
                if row.get("WIDTH")
                and float(row.get("WIDTH", 0)) >= 1024
                and row.get("HEIGHT")
                and float(row.get("HEIGHT", 0)) >= 1024
                and row.get("pwatermark")
                and float(row.get("pwatermark", 1)) <= 0.4
            }
            count += 1
    logging.info(f"Fetching {len(to_fetch)} images...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(fetch_image, to_fetch.values())


def main():
    """
    Main function to load CSV and fetch images
    """
    data = load_csv(FILE)
    fetch_data(data)


if __name__ == "__main__":
    main()
