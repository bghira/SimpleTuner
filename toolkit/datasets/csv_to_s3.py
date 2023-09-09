import os
import argparse
import logging
import boto3
import pandas as pd
from pathlib import Path
from PIL import Image, ExifTags
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor
import requests
import re
import shutil
from botocore.config import Config

# Constants
conn_timeout = 6
read_timeout = 60
timeouts = (conn_timeout, read_timeout)

# Set up logging
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

http = requests.Session()
adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
http.mount("http://", adapter)
http.mount("https://", adapter)


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
        r = http.get(url, timeout=timeouts, stream=True)
        if r.status_code == 200:
            with open(current_file_path, "wb") as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            r.close()
            image = Image.open(current_file_path)
            width, height = image.size
            if width < args.minimum_resolution or height < args.minimum_resolution:
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
        default="aws",
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
        "--parquet_folder", type=str, help="Location of the Parquet files."
    )
    parser.add_argument("--csv_folder", type=str, help="Location of the CSV files.")
    parser.add_argument("--git_lfs_repo", type=str, help="The Git LFS repository URL.")
    parser.add_argument(
        "--temporary_folder",
        type=str,
        required=True,
        help="Location of temporary data during upload.",
    )
    parser.add_argument(
        "--pwatermark_threshold",
        type=float,
        default=0.7,
        help="Threshold for pwatermark value. A higher score indicates a more likely chance of a watermark. Default: 0.7",
    )
    parser.add_argument(
        "--aesthetic_threshold",
        type=int,
        default=5,
        help="Threshold for aesthetic score, where a low score indicates a lower-quality image, often containing text. Default: 5",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.33,
        help="The similarity score of an image describes how closely its caption followed the embed. Higher = better. Default: 0.33",
    )
    parser.add_argument(
        "--unsafe_threshold",
        type=float,
        default=0.5,
        help="The probability of an image containing harmful content. Values higher than this will be ignored, unless --inverse_unsafe_threshold is given. Default: 0.5",
    )
    parser.add_argument(
        "--invert_unsafe_threshold",
        action="store_true",
        help="If set, images with a probability of harmful content higher than --unsafe_threshold will be included. This may be useful for training eg. NSFW classifiers.",
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
    if "top_caption" in df.columns:
        return "top_caption"
    if "Content" in df.columns:
        return "Content"
    elif "TEXT" in df.columns:
        return "TEXT"
    elif "all_captions" in df.columns:
        return "all_captions"


def initialize_s3_client(args):
    """Initialize the boto3 S3 client using the provided AWS credentials and settings."""
    s3_config = Config(max_pool_connections=100)

    s3_client = boto3.client(
        "s3",
        endpoint_url=args.aws_endpoint_url,
        region_name=args.aws_region_name,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        config=s3_config,
    )
    return s3_client


def content_to_filename(content):
    """
    Function to convert content to filename by stripping everything after '--',
    replacing non-alphanumeric characters and spaces, converting to lowercase,
    removing leading/trailing underscores, and limiting filename length to 128.
    """
    # Remove URLs
    logger.debug(f"Converting content to filename: {content}")
    cleaned_content = str(content)
    if "https" in cleaned_content:
        cleaned_content = re.sub(r"https?://\S*", "", cleaned_content)
    # Replace non-alphanumeric characters with underscore
    filename = re.sub(r"[^a-zA-Z0-9]", "_", cleaned_content)
    # Remove any '*' character:
    filename = filename.replace("*", "")
    # Remove anything after ' - Upscaled by'
    filename = filename.split(" - Upscaled by", 1)[0]
    # Remove anything after '--'
    filename = filename.split("--", 1)[0]
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


def list_all_s3_objects(s3_client, bucket_name):
    paginator = s3_client.get_paginator("list_objects_v2")
    existing_files = set()

    for page in paginator.paginate(Bucket=bucket_name):
        if "Contents" in page:
            for item in page["Contents"]:
                existing_files.add(item["Key"])

    return existing_files


def upload_to_s3(filename, args, s3_client):
    """Upload the specified file to the S3 bucket."""
    filename = os.path.join(args.temporary_folder, filename)
    object_name = os.path.basename(filename)

    # Check if the file exists just before uploading
    if not os.path.exists(filename):
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
    thread_map(
        fetch_and_upload_image,
        to_fetch.values(),
        [args] * len(to_fetch),
        [s3_client] * len(to_fetch),
        desc="Fetching & Uploading Images",
        max_workers=args.num_workers,
    )


def main():
    args = parse_args()

    # Initialize S3 client
    s3_client = initialize_s3_client(args)

    # List existing files in the S3 bucket
    existing_files = list_all_s3_objects(s3_client, args.aws_bucket_name)
    logger.info(f"Found {len(existing_files)} existing files in the S3 bucket.")

    if args.git_lfs_repo:
        repo_path = os.path.join(args.temporary_folder, "git-lfs-repo")
        if not os.path.exists(repo_path):
            logger.info(f"Cloning Git LFS repo to {repo_path}")
            os.system(f"git lfs clone {args.git_lfs_repo} {repo_path}")
        else:
            logger.info(
                f"Git LFS repo already exists at {repo_path}. Using existing files."
            )
        # Do we have *.parquet files in the dir, or .csv files?
        parquet_file_list = [f for f in Path(repo_path).glob("*.parquet")]
        csv_file_list = [f for f in Path(repo_path).glob("*.csv")]
        if len(parquet_file_list) > 0:
            args.parquet_folder = repo_path
            logger.info(f"Using Parquet files from {args.parquet_folder}")
        if len(csv_file_list) > 0:
            args.csv_folder = repo_path
            logger.info(f"Using CSV files from {args.csv_folder}")

    # Check if input folder exists
    parquet_files = []
    if args.parquet_folder is not None:
        if not os.path.exists(args.parquet_folder):
            logger.error(f"Input folder '{args.parquet_folder}' does not exist.")
            return
        # Read Parquet file as DataFrame
        parquet_files = [f for f in Path(args.parquet_folder).glob("*.parquet")]
    csv_files = []
    if args.csv_folder is not None:
        if not os.path.exists(args.csv_folder):
            logger.error(f"Input folder '{args.csv_folder}' does not exist.")
            return
        # Read Parquet file as DataFrame
        csv_files = [f for f in Path(args.csv_folder).glob("*.csv")]
    all_files = parquet_files + csv_files
    logger.info(f"Discovered catalogues: {all_files}")

    total_files = len(all_files)
    for i, file in enumerate(
        tqdm(all_files, desc=f"Processing {total_files} Parquet files")
    ):
        if content_to_filename(file.name) in existing_files:
            logger.info(f"Skipping already processed file: {file}")
            continue
        logger.info(f"Loading file: {file}")
        if file.suffix == ".parquet":
            df = pd.read_parquet(file)
        elif file.suffix == ".csv":
            df = pd.read_csv(file)
        else:
            logger.warning(f"Unsupported file format: {file.suffix}")
            continue

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
            df = df[df["pwatermark"] <= args.pwatermark_threshold]
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
        if "similarity" in df.columns:
            logger.info(
                f"Applying similarity filter with threshold {args.similarity_threshold}"
            )
            df = df[df["similarity"] >= args.similarity_threshold]
            logger.info(f"Filtered to {len(df)} rows.")
        if "punsafe" in df.columns:
            logger.info(
                f"Applying unsafe filter with threshold {args.unsafe_threshold}"
            )
            if args.invert_unsafe_threshold:
                logger.info(
                    "Inverting unsafe threshold, so that more harmful content is included, rather than excluded."
                )
                df = df[df["punsafe"] >= args.unsafe_threshold]
            else:
                df = df[df["punsafe"] <= args.unsafe_threshold]
            logger.info(f"Filtered to {len(df)} rows.")

        # TODO: Add more filters as needed

        # Fetch and process images
        to_fetch = df.to_dict(orient="records")
        logger.info(f"Fetching {len(to_fetch)} images...")
        fetch_data(s3_client, to_fetch, args, uri_column)


if __name__ == "__main__":
    main()
