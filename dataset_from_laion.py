from PIL import Image
import os, logging
import csv
import shutil
import requests
import re
import sys

from concurrent.futures import ThreadPoolExecutor

# Constants
FILE = "laion.csv"  # The CSV file to read data from
OUTPUT_DIR = "/datasets/laion"  # Directory to save images
NUM_WORKERS = 64  # Number of worker threads for parallel downloading
logger = logging.getLogger('root')
logger.setLevel(logging.INFO)
# Check if output directory exists, create if it does not
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except Exception as e:
        logging.info(f'Could not create output directory: {e}')
        sys.exit(1)

# Check if CSV file exists
if not os.path.exists(FILE):
    logging.info(f'Could not find CSV file: {FILE}')
    sys.exit(1)

import random

def content_to_filename(content):
    """
    Function to convert content to filename by stripping everything after '--', 
    replacing non-alphanumeric characters and spaces, converting to lowercase, 
    removing leading/trailing underscores, and limiting filename length to 128.
    """
    # Split on '--' and take the first part
    content = content.split('--', 1)[0] 
    # Split on 'Upscaled by' and take the first part
    content = content.split(' - Upscaled by', 1)[0] 
    # Remove URLs
    cleaned_content = re.sub(r'https*://\S*', '', content)

    # Replace non-alphanumeric characters and spaces, convert to lowercase, remove leading/trailing underscores
    cleaned_content = re.sub(r'[^a-zA-Z0-9 ]', '', cleaned_content)
    cleaned_content = cleaned_content.replace(' ', '_').lower().strip('_')

    # If cleaned_content is empty after removing URLs, generate a random filename
    if cleaned_content == '':
        cleaned_content = f'midjourney_{random.randint(0, 1000000)}'
    
    # Limit filename length to 128
    cleaned_content = cleaned_content[:128] if len(cleaned_content) > 128 else cleaned_content

    return cleaned_content + '.png'

def load_csv(file):
    """
    Function to load CSV data into a list of dictionaries
    """
    data = []
    with open(file, newline='') as csvfile:
        try:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        except Exception as e:
            logging.info(f'Could not advance reader: {e}')
    return data

conn_timeout = 6
read_timeout = 60
timeouts = (conn_timeout, read_timeout)
def _resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    aspect_ratio = round(W / H, 3)
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
    img = input_image.resize((W, H), resample=Image.BICUBIC)
    return img


def fetch_image(info):
    """
    Function to fetch image from a URL and save it to disk if it is square
    """
    filename = info['filename']
    url = info['url']
    current_file_path = os.path.join(OUTPUT_DIR, filename)
    # Skip download if file already exists
    if os.path.exists(current_file_path):
        logging.info(f'{filename} already exists, skipping download...')
        return

    try:
        r = requests.get(url, timeout=timeouts, stream=True)
        if r.status_code == 200:
            with open(current_file_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            image = Image.open(current_file_path)
            image = _resize_for_condition_image(image, 1024)
            image.save(current_file_path, format='PNG')
            image.close()
        else:
            logging.warn(f'Could not fetch {filename} from {url} (status code {r.status_code})')
    except Exception as e:
        logging.info(f'Could not fetch {filename} from {url}: {e}')


def fetch_data(data):
    """
    Function to fetch all images specified in data
    """
    to_fetch = {}
    for row in data:
        if row['NSFW'] != 'NSFW':
            continue
        try:
            if float(row['WIDTH']) < 960 or float(row['HEIGHT']) < 960:
                continue
        except:
            logging.error(f'Could not download row: {row}')
            continue
        new_filename = content_to_filename(row['TEXT'])
        if new_filename not in to_fetch:
            to_fetch[new_filename] = {'url': row['URL'], 'filename': new_filename}
    logging.info(f'Fetching {len(to_fetch)} images...')
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(fetch_image, to_fetch.values())

def main():
    """
    Main function to load CSV and fetch images
    """
    data = load_csv(FILE)
    fetch_data(data)

if __name__ == '__main__':
    main()
