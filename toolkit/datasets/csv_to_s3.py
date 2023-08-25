"""
This script loads up a folder of csv or parquet files and then downloads in parallel,
 confirming the images are valid and meet our minimum requirements, before uploading
 them to the S3 bucket.
"""

import os
import argparse
import logging
import boto3
import pandas as pd
from pathlib import Path
from PIL import Image, ExifTags
from concurrent.futures import ThreadPoolExecutor
import requests
import re
import shutil

# Constants
conn_timeout = 6
read_timeout = 60
timeouts = (conn_timeout, read_timeout)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

conn_timeout = 6
read_timeout = 60
timeouts = (conn_timeout, read_timeout)

def fetch_image(info):
    filename = info["filename"]
    url = info["url"]
    current_file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(current_file_path):
        logging.info(f"{filename} already exists, skipping download...")
        return
    try:
        r = requests.get(url, timeout=timeouts, stream=True)
        if r.status_code == 200:
            with open(current_file_path, "wb") as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            image = Image.open(current_file_path)
            width, height = image.size
            if width != height:
                logging.info(
                    f"Image {filename} is not square ({width}x{height}), deleting..."
                )
                os.remove(current_file_path)
                return
            logging.info(f"Resizing image to {current_file_path}")
            image = image.resize((768, 768), resample=Image.LANCZOS)
            image.save(current_file_path, format="PNG")
            image.close()
        else:
            logging.info(
                f"Could not fetch {filename} from {url} (status code {r.status_code})"
            )
    except Exception as e:
        logging.info(f"Could not fetch {filename} from {url}: {e}")



def fetch_data(data):
    to_fetch = {}
    for row in data:
        new_filename = content_to_filename(row["Content"])
        if "Variations" in row["Content"] or "Upscaled" not in row["Content"]:
            continue
        if new_filename not in to_fetch:
            to_fetch[new_filename] = {
                "url": row["Attachments"],
                "filename": new_filename,
            }
    logging.info(f"Fetching {len(to_fetch)} images...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(fetch_image, to_fetch.values())


def parse_args():
    parser = argparse.ArgumentParser(description="Filter and upload images from Parquet files to S3.")
    
    # AWS-related arguments
    parser.add_argument("--data_backend", choices=["local", "aws"], default="local", help="The data backend to use.")
    parser.add_argument("--aws_bucket_name", type=str, help="The AWS bucket name to use.")
    parser.add_argument("--aws_endpoint_url", type=str, help="The AWS server to use.")
    parser.add_argument("--aws_region_name", type=str, help="The AWS region to use.")
    parser.add_argument("--aws_access_key_id", type=str, help="AWS access key ID.")
    parser.add_argument("--aws_secret_access_key", type=str, help="AWS secret access key.")
    
    # Script-specific arguments
    parser.add_argument("--input_folder", type=str, required=True, help="Location of the Parquet files.")
    parser.add_argument("--pwatermark_threshold", type=float, default=0.0, help="Threshold for pwatermark value.")
    parser.add_argument("--aesthetic_threshold", type=int, default=0, help="Threshold for aesthetic score.")
    parser.add_argument("--caption_field", type=str, default="TEXT", help="Field to use for image filename.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads for downloading images.")
    
    return parser.parse_args()

# Additional functions for handling diverse input datasets

def get_uri_column(df):
    if 'URL' in df.columns:
        return 'URL'
    elif 'Attachments' in df.columns:
        return 'Attachments'
    else:
        logger.error("No recognized URI column found in the dataset.")
        return None

def main():
    args = parse_args()
    
    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        logger.error(f"Input folder '{args.input_folder}' does not exist.")
        return
    
    # Read Parquet file as DataFrame
    parquet_files = [f for f in Path(args.input_folder).glob("*.parquet")]
    for file in parquet_files:
        df = pd.read_parquet(file)
        
        # Determine the URI column
        uri_column = get_uri_column(df)
        if not uri_column:
            continue
        
        # Apply filters
        if 'pwatermark' in df.columns:
            df = df[df['pwatermark'] >= args.pwatermark_threshold]
        if 'aesthetic' in df.columns:
            df = df[df['aesthetic'] >= args.aesthetic_threshold]
        # TODO: Add more filters as needed

        # Fetch and process images
        fetch_data(df.to_dict(orient='records'))

if __name__ == "__main__":
    main()
