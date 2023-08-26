import os
import argparse
import logging
import boto3
import pandas as pd
from pathlib import Path
from PIL import Image, ExifTags
from tqdm import tqdm
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
    
    return cleaned_content


def resize_for_condition_image(input_image: Image, resolution: int):
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
    if W == H:
        W = resolution
        H = resolution
    msg = f"{msg} {W}x{H}."
    logger.debug(msg)
    img = input_image.resize((W, H), resample=Image.BICUBIC)
    return img


def fetch_image(info, args):
    filename = info["filename"]
    url = info["url"]
    current_file_path = os.path.join(args.temporary_folder, filename)
    if os.path.exists(current_file_path):
        return
    try:
        r = requests.get(url, timeout=timeouts, stream=True)
        if r.status_code == 200:
            with open(current_file_path, "wb") as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            image = Image.open(current_file_path)
            width, height = image.size
            if width < args.minimum_resolution or height < args.minimum_resolution:
                print(f"Image {filename} is too small ({width}x{height}), deleting...")
                os.remove(current_file_path)
                return
            image = resize_for_condition_image(image, args.condition_image_size)
            image.save(current_file_path, format="PNG")
            image.close()
        else:
            pass
    except Exception as e:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter and upload images from Parquet files to S3."
    )

    # AWS-related arguments
    parser.add_argument(
        "--data_backend",
        choices=["local", "aws"],
        default="local",
        help="The data backend to use.",
    )
    parser.add_argument(
        "--aws_bucket_name", type=str, help="The AWS bucket name to use."
    )
    parser.add_argument("--aws_endpoint_url", type=str, help="The AWS server to use.")
    parser.add_argument("--aws_region_name", type=str, help="The AWS region to use.")
    parser.add_argument("--aws_access_key_id", type=str, help="AWS access key ID.")
    parser.add_argument(
        "--aws_secret_access_key", type=str, help="AWS secret access key."
    )

    # Script-specific arguments
    parser.add_argument(
        "--input_folder", type=str, required=True, help="Location of the Parquet files."
    )
    parser.add_argument(
        "--temporary_folder",
        type=str,
        required=True,
        help="Location of temporary data during upload.",
    )
    parser.add_argument(
        "--pwatermark_threshold",
        type=float,
        default=0.0,
        help="Threshold for pwatermark value.",
    )
    parser.add_argument(
        "--aesthetic_threshold",
        type=int,
        default=0,
        help="Threshold for aesthetic score.",
    )
    parser.add_argument(
        "--caption_field",
        type=str,
        default=None,
        help="Field to use for image filename. Leave unset to auto-detect.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads for downloading images.",
    )
    parser.add_argument(
        "--max_num_files",
        type=int,
        default=1000000,
        help="Maximum number of files to process.",
    )
    # Filtering images
    parser.add_argument(
        "--minimum_resolution",
        type=int,
        default=768,
        help="Minimum resolution for images.",
    )
    parser.add_argument(
        "--condition_image_size",
        type=int,
        default=1024,
        help="This option will by default, resize the smaller edge of an image to 1024px.",
    )

    return parser.parse_args()


# Additional functions for handling diverse input datasets


def get_uri_column(df):
    if "URL" in df.columns:
        return "URL"
    elif "Attachments" in df.columns:
        return "Attachments"
    else:
        logger.error("No recognized URI column found in the dataset.")
        return None


def get_caption_column(df):
    if "Content" in df.columns:
        return "Content"
    elif "TEXT" in df.columns:
        return "TEXT"
    elif "all_captions" in df.columns:
        return "all_captions"


def initialize_s3_client(args):
    """Initialize the boto3 S3 client using the provided AWS credentials and settings."""
    s3_client = boto3.client(
        "s3",
        endpoint_url=args.aws_endpoint_url,
        region_name=args.aws_region_name,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
    )
    return s3_client


def content_to_filename(content):
    """Convert content to a suitable filename."""
    # Remove URLs
    cleaned_content = re.sub(r"https?://\S*", "", content)
    # Replace non-alphanumeric characters with underscore
    filename = re.sub(r"[^a-zA-Z0-9]", "_", cleaned_content)
    # Convert to lowercase and trim to 128 characters
    filename = filename.lower()[:128] + ".png"
    return filename


def valid_exif_data(image_path):
    """Check if the image contains EXIF data for camera make/model."""
    try:
        image = Image.open(image_path)
        for tag, value in image._getexif().items():
            if tag in ExifTags.TAGS and ExifTags.TAGS[tag] in ["Make", "Model"]:
                return True
    except:
        pass
    return False


def upload_to_s3(filename, args, s3_client):
    """Upload the specified file to the S3 bucket."""
    filename = os.path.join(args.temporary_folder, filename)
    object_name = os.path.basename(filename)

    # Check if the file exists just before uploading
    if not os.path.exists(filename):
        logger.error(f"File {filename} does not exist. Skipping upload.")
        return

    try:
        s3_client.upload_file(filename, args.aws_bucket_name, object_name)
        # Delete the local file after successful upload
        os.remove(filename)
    except Exception as e:
        logger.error("Error uploading {} to S3: {}".format(object_name, e))


def fetch_and_upload_image(info, args, s3_client):
    """Fetch the image, process it, and upload it to S3."""
    try:
        fetch_image(info, args)
    except Exception as e:
        pass
    upload_to_s3(info["filename"], args, s3_client)


def fetch_data(s3_client, data, args, uri_column):
    """Function to fetch all images specified in data and upload them to S3."""
    to_fetch = {}
    for row in data:
        new_filename = content_to_filename(row[args.caption_field])
        if (
            hasattr(args, "midjourney_data_checks")
            and args.midjourney_data_checks
            and (
                "Variations" in row[args.caption_field]
                or "Upscaled" not in row[args.caption_field]
            )
        ):
            continue
        if new_filename not in to_fetch:
            to_fetch[new_filename] = {
                "url": row[uri_column],
                "filename": new_filename,
                "args": args,
            }
    logging.info("Fetching {} images...".format(len(to_fetch)))
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(
                    fetch_and_upload_image,
                    to_fetch.values(),
                    [args] * len(to_fetch),
                    [s3_client] * len(to_fetch),
                ), total=len(to_fetch), desc="Fetching & Uploading Images"))


def main():
    args = parse_args()

    # Initialize S3 client
    s3_client = initialize_s3_client(args)

    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        logger.error(f"Input folder '{args.input_folder}' does not exist.")
        return

    # Read Parquet file as DataFrame
    parquet_files = [f for f in Path(args.input_folder).glob("*.parquet")]
    logger.info(f"Discovered catalogues: {parquet_files}")
    for file in tqdm(parquet_files, desc="Processing Parquet files"):
        logger.info(f"Loading file: {file}")
        df = pd.read_parquet(file)

        # Determine the URI column
        uri_column = get_uri_column(df)
        if args.caption_field is None:
            args.caption_field = get_caption_column(df)
        logger.info(f"Caption field: {args.caption_field}")
        if not uri_column:
            logger.warning(f"Row has no uri_column: {uri_column}")
            continue
        logger.info(f"URI field: {uri_column}")
        logger.info(f"Before filtering, we have {len(df)} rows.")
        # Apply filters
        if "pwatermark" in df.columns:
            logger.info(
                f"Applying pwatermark filter with threshold {args.pwatermark_threshold}"
            )
            df = df[df["pwatermark"] >= args.pwatermark_threshold]
            logger.info(f"Filtered to {len(df)} rows.")
        if "aesthetic" in df.columns:
            logger.info(
                f"Applying aesthetic filter with threshold {args.aesthetic_threshold}"
            )
            df = df[df["aesthetic"] >= args.aesthetic_threshold]
            logger.info(f"Filtered to {len(df)} rows.")
        if "WIDTH" in df.columns:
            logger.info(
                f"Applying minimum resolution filter with threshold {args.minimum_resolution}"
            )
            df = df[df["WIDTH"] >= args.minimum_resolution]
            logger.info(f"Filtered to {len(df)} rows.")
        if "HEIGHT" in df.columns:
            logger.info(
                f"Applying minimum resolution filter with threshold {args.minimum_resolution}"
            )
            df = df[df["HEIGHT"] >= args.minimum_resolution]
            logger.info(f"Filtered to {len(df)} rows.")
        # TODO: Add more filters as needed

        # Fetch and process images
        to_fetch = df.to_dict(orient="records")
        logger.info(f"Fetching {len(to_fetch)} images...")
        fetch_data(s3_client, to_fetch, args, uri_column)


if __name__ == "__main__":
    main()
