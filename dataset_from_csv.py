from PIL import Image
import os
import csv
import shutil
import requests
import re
import sys

from concurrent.futures import ThreadPoolExecutor

# Constants
FILE = "dataset.csv"  # The CSV file to read data from
OUTPUT_DIR = "/notebooks/images/datasets/midjourney"  # Directory to save images
NUM_WORKERS = 8  # Number of worker threads for parallel downloading

# Check if output directory exists, create if it does not
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except Exception as e:
        print(f'Could not create output directory: {e}')
        sys.exit(1)

# Check if CSV file exists
if not os.path.exists(FILE):
    print(f'Could not find CSV file: {FILE}')
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
            print(f'Could not advance reader: {e}')
    return data

conn_timeout = 6
read_timeout = 60
timeouts = (conn_timeout, read_timeout)

def fetch_image(info):
    """
    Function to fetch image from a URL and save it to disk if it is square
    """
    filename = info['filename']
    url = info['url']
    current_file_path = os.path.join(OUTPUT_DIR, filename)
    # Skip download if file already exists
    if os.path.exists(current_file_path):
        print(f'{filename} already exists, skipping download...')
        return

    try:
        r = requests.get(url, timeout=timeouts, stream=True)
        if r.status_code == 200:
            with open(current_file_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            image = Image.open(current_file_path)
            width, height = image.size
            if width != height:
                print(f'Image {filename} is not square ({width}x{height}), deleting...')
                os.remove(current_file_path)
                return
            print(f'Resizing image to {current_file_path}')
            image = image.resize((768, 768), resample=Image.LANCZOS)
            image.save(current_file_path, format='PNG')
            image.close()
        else:
            print(f'Could not fetch {filename} from {url} (status code {r.status_code})')
    except Exception as e:
        print(f'Could not fetch {filename} from {url}: {e}')


def fetch_data(data):
    """
    Function to fetch all images specified in data
    """
    to_fetch = {}
    for row in data:
        new_filename = content_to_filename(row['Content'])
        if "Variations" in row['Content'] or "Upscaled" not in row['Content']:
            continue
        if new_filename not in to_fetch:
            to_fetch[new_filename] = {'url': row['Attachments'], 'filename': new_filename}
    print(f'Fetching {len(to_fetch)} images...')
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
