import os
import random
import argparse
import logging
import boto3
from io import BytesIO
import pandas as pd
from pathlib import Path
try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image, ExifTags
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from multiprocessing import Pool
import requests
import re
import shutil
from botocore.config import Config
from multiprocessing import Pool
from tqdm import tqdm

# Constants
conn_timeout = 6
read_timeout = 60
timeouts = (conn_timeout, read_timeout)

# Set up logging
logging.basicConfig(level=os.getenv("SIMPLETUNER_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
connection_logger = logging.getLogger("urllib3.connectionpool")
connection_logger.setLevel(logging.ERROR)
connection_logger = logging.getLogger("urllib3.connection")
connection_logger.setLevel(logging.ERROR)
pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)
pil_logger = logging.getLogger("PIL.Image")
pil_logger.setLevel("ERROR")
pil_logger = logging.getLogger("PIL.PngImagePlugin")
pil_logger.setLevel("ERROR")
loggers_to_silence = [
    "botocore.hooks",
    "botocore.auth",
    "botocore.httpsession",
    "botocore.parsers",
    "botocore.retryhandler",
    "botocore.loaders",
    "botocore.regions",
    "botocore.utils",
    "botocore.client",
    "botocore.handler",
    "botocore.handlers",
    "botocore.awsrequest",
]

for logger_name in loggers_to_silence:
    _logger = logging.getLogger(logger_name)
    _logger.setLevel("ERROR")

# Arguably, the most interesting one:
boto_logger = logging.getLogger("botocore.endpoint")
boto_logger.setLevel(os.environ.get("SIMPLETUNER_AWS_LOG_LEVEL", "ERROR"))

http = requests.Session()
adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
http.mount("http://", adapter)
http.mount("https://", adapter)


def shuffle_words_in_filename(filename):
    """Shuffle the words in a filename while keeping the file extension unchanged."""
    name, ext = os.path.splitext(filename)
    words = name.split(
        "_"
    )  # Assuming words in the filename are separated by underscores
    random.shuffle(words)
    return "_".join(words) + ext


def resize_for_condition_image(input_image: Image.Image, resolution: int):
    if resolution == 0:
        return input_image
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
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def object_exists_in_s3(s3_client, bucket_name, object_name):
    """Check if a specific object exists in the S3 bucket."""
    try:
        s3_client.head_object(Bucket=bucket_name, Key=object_name)
        return True
    except:
        return False


def calculate_luminance(image: Image):
    """Calculate the luminance of an image."""
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    luminance_value = 1 if brightness == 255 else brightness / scale
    logger.debug(f"Calculated luminance: {luminance_value}")
    return luminance_value


def fetch_image(info, args):
    filename = info["filename"]
    url = info["url"]
    # Constants
    conn_timeout = args.connection_timeout
    read_timeout = args.read_timeout
    timeouts = (conn_timeout, read_timeout)

    current_file_path = os.path.join(args.temporary_folder, filename)
    if os.path.exists(current_file_path):
        return
    try:
        r = http.get(url, timeout=timeouts, stream=True)
        if r.status_code == 200:
            image = Image.open(BytesIO(r.content))
            width, height = image.size
            if width < args.minimum_resolution or height < args.minimum_resolution:
                os.remove(current_file_path)
                return
            if args.only_exif_images and not valid_exif_data(current_file_path):
                os.remove(current_file_path)
                return
            if args.min_luminance is not None or args.max_luminance is not None:
                image_luminance = calculate_luminance(image)
                if args.min_luminance and image_luminance < args.min_luminance:
                    os.remove(current_file_path)
                    return
                if args.max_luminance and image_luminance > args.max_luminance:
                    os.remove(current_file_path)
                    return
            image = resize_for_condition_image(image, args.condition_image_size)
            return image
        else:
            pass
    except Exception as e:
        import traceback

        logger.error(f"Error: {e}, traceback: {traceback.format_exc()}")
        raise e


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
    parser.add_argument(
        "--connection_timeout",
        type=int,
        default=3,
        help="Connection timeout in seconds.",
    )
    parser.add_argument(
        "--midjourney_data_checks",
        action="store_true",
        help="If set, only images with certain entries in the caption will be included. This is useful for midjourney data checks.",
    )
    parser.add_argument(
        "--read_timeout",
        type=int,
        default=30,
        help="Read timeout in seconds.",
    )
    # Script-specific arguments
    parser.add_argument(
        "--local_folder",
        type=str,
        help="Location of local folder containing images to upload.",
    )
    parser.add_argument(
        "--parquet_folder", type=str, help="Location of the Parquet files."
    )
    parser.add_argument("--csv_folder", type=str, help="Location of the CSV files.")
    parser.add_argument("--git_lfs_repo", type=str, help="The Git LFS repository URL.")
    parser.add_argument(
        "--delete_after_processing",
        action="store_true",
        help="Delete original CSV/Parquet file after processing.",
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
        "--min_luminance",
        type=float,
        default=None,
        help="Minimum luminance threshold for images. If not provided, no lower cap is applied.",
    )
    parser.add_argument(
        "--max_luminance",
        type=float,
        default=None,
        help="Maximum luminance threshold for images. If not provided, only capping is applied.",
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
        default=0,
        help="Minimum resolution for images. Set to 0 to disable.",
    )
    parser.add_argument(
        "--minimum_pixel_area",
        type=int,
        default=1,
        help="Minimum pixel area for images, measured in megapixels. Set to 0 to disable.",
    )
    parser.add_argument(
        "--width_field",
        type=str,
        default=None,
        help=("Column name for image width. Auto-detected, if not supplied."),
    )
    parser.add_argument(
        "--height_field",
        type=str,
        default=None,
        help=(
            "The column name in the dataset for the image height. Auto-detected, if not supplied."
        ),
    )
    parser.add_argument(
        "--condition_image_size",
        type=int,
        default=0,
        help="This option will by default, resize the smaller edge of an image to 1024px.",
    )
    parser.add_argument(
        "--only_exif_images",
        action="store_true",
        help="If set, only images with EXIF data will be included.",
    )
    parser.add_argument(
        "--print_nonfatal_errors",
        action="store_true",
        help="If set, non-fatal errors will be printed. Remove this from the commandline to make output more streamlined/quieter.",
    )
    return parser.parse_args()


# Additional functions for handling diverse input datasets
def get_uri_column(df):
    if "url" in df.columns:
        return "url"
    if "URL" in df.columns:
        return "URL"
    elif "Attachments" in df.columns:
        return "Attachments"
    else:
        logger.error("No recognized URI column found in the dataset.")
        return None


def get_width_field(df):
    if "WIDTH" in df.columns:
        return "WIDTH"
    return "width"


def get_height_field(df):
    if "HEIGHT" in df.columns:
        return "HEIGHT"
    return "height"


def get_caption_column(df):
    if "top_caption" in df.columns:
        return "top_caption"
    if "Content" in df.columns:
        return "Content"
    if "caption" in df.columns:
        return "Content"
    if "TEXT" in df.columns:
        return "TEXT"
    if "all_captions" in df.columns:
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


def content_to_filename(content, args):
    """
    Function to convert content to filename by stripping everything after '--',
    replacing non-alphanumeric characters and spaces, converting to lowercase,
    removing leading/trailing underscores, and limiting filename length to 128.
    """
    logger.debug(f"Converting content to filename: {content}")
    filename = str(content)
    image_num_text = ""
    try:
        # Extract the "Image #" with its number using regex, careful not to grab anything past it.
        image_num_match = re.search(r" - Image #\d+", filename)
        if image_num_match:
            image_num_text = image_num_match.group(0).strip()
            filename = filename.replace(image_num_text, "")
            image_num_text = image_num_text.replace(
                " - ", "_"
            )  # Replace spaces and hyphens for consistency
        # Remove anything after '--'
        filename = filename.split("--", 1)[0]
        # Remove URLs
        filename = re.sub(r"https?://\S*", "", filename)
        # Replace non-alphanumeric characters with underscore
        # filename = re.sub(r"[^a-zA-Z0-9\s]", "_", filename)
        # Remove leading and trailing underscores
        filename = filename.strip("_")
        # Strip multiple whitespaces, replace with single whitespace
        filename = re.sub(r"\s+", " ", filename)
        # Strip surrounding whitespace
        filename = filename.strip()
        # Convert to lowercase and limit the length to accommodate the image number and extension
        max_length = 2048 - len(image_num_text) - 4  # 4 for the ".png"
        filename = (filename.lower()[:max_length] + image_num_text).rstrip("_") + ".png"
        logger.debug(f"-> Resulting filename: {filename}")
        return filename
    except Exception as e:
        if hasattr(args, "print_nonfatal_errors") and args.print_nonfatal_errors:
            logger.error(f"Encountered error processing filename: {e}")


def valid_exif_data(image_path):
    """Check if the image contains EXIF data typically associated with real cameras."""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        # If no EXIF data, return False
        if not exif_data:
            return False

        # List of tags to check for real camera evidence
        tags_to_check = ["Make", "Model", "DateTimeOriginal", "LensModel", "GPSInfo"]

        # Check if any of the relevant tags exist in the EXIF data
        for tag, value in exif_data.items():
            tagname = ExifTags.TAGS.get(tag, tag)
            if tagname in tags_to_check:
                return True

        # If "Software" tag exists, it might be edited or generated, but this is not a surefire method
        if "Software" in exif_data:
            software_name = exif_data["Software"].lower()
            if "photoshop" in software_name or "gimp" in software_name:
                return False

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        pass

    return False


def list_all_s3_objects(s3_client, bucket_name):
    paginator = s3_client.get_paginator("list_objects_v2")
    existing_files = set()
    png_files = set()
    for page in paginator.paginate(Bucket=bucket_name, MaxKeys=1000):
        if "Contents" in page:
            for item in page["Contents"]:
                existing_files.add(item["Key"])
                if item["Key"].endswith(".png"):
                    png_files.add(item["Key"])
                    logger.info(
                        f"Found {len(png_files)} existing png files in the S3 bucket."
                    )

    return existing_files


def upload_pil_to_s3(image: Image.Image, filename, args, s3_client):
    """Upload a PIL Image directly to S3 bucket"""
    if object_exists_in_s3(s3_client, args.aws_bucket_name, filename):
        return
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, args.aws_bucket_name, filename)
    buffer.close()


def upload_to_s3(filename, args, s3_client):
    """Upload the specified file to the S3 bucket with filename shuffling if needed."""
    local_path = os.path.join(args.temporary_folder, filename)
    # Just use the base filename without any directory prefix for S3
    object_name = os.path.basename(filename)

    # Check if the file exists just before uploading
    if not os.path.exists(local_path):
        return

    if object_exists_in_s3(s3_client, args.aws_bucket_name, object_name):
        try:
            os.remove(local_path)
        except:
            pass
        return

    try:
        s3_client.upload_file(local_path, args.aws_bucket_name, object_name)
        # Delete the local file after successful upload
        os.remove(local_path)
    except Exception as e:
        logger.error(f"Error uploading {object_name} to S3: {e}")


def upload_local_image_to_s3(image_path, args, s3_client):
    """Upload local image directly to the S3 bucket."""
    object_name = os.path.basename(image_path)

    # Check if the file exists just before uploading
    if not os.path.exists(image_path):
        return

    if object_exists_in_s3(s3_client, args.aws_bucket_name, object_name):
        try:
            os.remove(image_path)
        except:
            pass
        return

    try:
        s3_client.upload_file(image_path, args.aws_bucket_name, object_name)
        # Optionally, delete the local file after successful upload
        if args.delete_after_processing:
            os.remove(image_path)
    except Exception as e:
        logger.error(f"Error uploading {object_name} to S3: {e}")


def process_git_lfs_images(args, s3_client):
    """Scan the git-lfs-repo directory for image files and upload them."""
    repo_path = os.path.join(args.temporary_folder, "git-lfs-repo")
    image_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    for ext in image_exts:
        for image_path in Path(repo_path).rglob(f"*{ext}"):
            upload_local_image_to_s3(image_path, args, s3_client)


def process_local_folder_images(args, s3_client, existing_files: list):
    """Scan the local folder directory for image files, apply checks, and upload them in parallel."""
    if not os.path.exists(args.local_folder):
        logger.error(f"Local folder '{args.local_folder}' does not exist.")
        return

    image_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    all_images = [
        image_path
        for ext in image_exts
        for image_path in Path(args.local_folder).rglob(f"*{ext}")
    ]

    # Filter out already processed files
    images_to_process = [
        image_path for image_path in all_images if str(image_path) not in existing_files
    ]

    # Using Pool for parallel processing
    with Pool(processes=args.num_workers) as pool:
        list(
            tqdm(
                pool.imap(
                    process_and_upload,
                    [(image_path, args) for image_path in images_to_process],
                ),
                total=len(images_to_process),
            )
        )


def process_and_upload(image_path_args):
    image_path, args = image_path_args
    """Process and upload a single image."""
    # Place your existing logic here for processing a single image and uploading it to S3.
    # For example:
    try:
        image = Image.open(image_path)
    except Exception:
        return None
    width, height = image.size

    # Check minimum resolution
    if args.minimum_resolution > 0 and (
        width < args.minimum_resolution or height < args.minimum_resolution
    ):
        return None

    # Check luminance if required
    if args.min_luminance is not None or args.max_luminance is not None:
        luminance = calculate_luminance(image)
        if args.min_luminance is not None and luminance < args.min_luminance:
            return None
        if args.max_luminance is not None and luminance > args.max_luminance:
            return None

    # Reinitialize S3 client for each process
    s3_client = initialize_s3_client(args)

    # Resize image for condition if required
    image = resize_for_condition_image(image, args.condition_image_size)
    temp_path = os.path.join(args.temporary_folder, os.path.basename(image_path))
    try:
        image.save(temp_path, format="PNG")
        image.close()
    except:
        logger.error(f"Error saving image")
    # Upload to S3
    upload_local_image_to_s3(temp_path, args, s3_client)


def fetch_and_upload_image(info, args):
    """Fetch the image, process it, and upload it to S3."""
    try:
        s3_client = initialize_s3_client(args)
        image = fetch_image(info, args)
    except Exception as e:
        if args.print_nonfatal_errors:
            logger.error(f"Encountered error fetching file: {e}")
    upload_pil_to_s3(image, info["filename"], args, s3_client)


def fetch_data(data, args, uri_column):
    """Function to fetch all images specified in data and upload them to S3."""
    to_fetch = {}
    for row in data:
        new_filename = content_to_filename(row[args.caption_field], args)
        if (
            hasattr(args, "midjourney_data_checks")
            and args.midjourney_data_checks
            and "image #" not in row[args.caption_field].lower()
        ):
            # Midjourney's upscaler sucks. We only want single images, non-tiled.
            logger.debug(f"Skipping: {row[args.caption_field]}")
            continue
        if new_filename not in to_fetch:
            to_fetch[new_filename] = {
                "url": row[uri_column],
                "filename": new_filename,
                "args": args,
            }

    logging.info(f"Fetching {len(to_fetch)} images (truncated): {list(to_fetch)[:10]}")

    with Pool(processes=args.num_workers) as pool:
        results = pool.starmap(
            fetch_and_upload_image,
            [(item, args) for item in to_fetch.values()],
        )


def main():
    args = parse_args()

    # Initialize S3 client
    s3_client = initialize_s3_client(args)

    # List existing files in the S3 bucket
    existing_files = []
    existing_files = list_all_s3_objects(s3_client, args.aws_bucket_name)
    logger.info(f"Found {len(existing_files)} existing files in the S3 bucket.")

    # Process and upload images from the local folder
    if args.local_folder:
        process_local_folder_images(args, s3_client, existing_files)

    if args.git_lfs_repo:
        repo_path = os.path.join(args.temporary_folder, "git-lfs-repo")
        if not os.path.exists(repo_path):
            logger.info(f"Thin-cloning Git LFS repo to {repo_path}")
            os.system(
                f"env GIT_LFS_SKIP_SMUDGE=1 git clone {args.git_lfs_repo} {repo_path}"
            )
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
        # Process and upload images from the git-lfs-repo
        process_git_lfs_images(args, s3_client)

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
    random.shuffle(all_files)
    logger.info(f"Discovered catalogues: {all_files}")

    total_files = len(all_files)
    for i, file in enumerate(
        tqdm(all_files, desc=f"Processing {total_files} Parquet files")
    ):
        if content_to_filename(file.name, args) in existing_files:
            logger.info(f"Skipping already processed file: {file}")
            continue
        # If it's a parquet file from the Git LFS repo, pull it Just-in-Time
        if file.suffix == ".parquet":
            if args.git_lfs_repo:
                logger.info(f"Fetching {file} from Git LFS")
                # chdir to repo_path
                cwd = os.getcwd()
                os.chdir(repo_path)
                os.system(f"env GIT_LFS_SKIP_SMUDGE=0 git lfs pull -I {file.name}")
                # return to prev dir
                os.chdir(cwd)
            logger.info(f"Loading file: {file}")
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
        if args.height_field is None:
            args.height_field = get_height_field(df)
        if args.width_field is None:
            args.width_field = get_width_field(df)
        logger.info(
            f"Resolution fields: '{args.width_field}' and '{args.height_field}'"
        )

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
        if args.width_field in df.columns and args.minimum_resolution > 0:
            logger.info(
                f"Applying minimum resolution filter with threshold {args.minimum_resolution}"
            )
            df = df[df[args.width_field] >= args.minimum_resolution]
            logger.info(f"Filtered to {len(df)} rows.")
        if args.height_field in df.columns and args.minimum_resolution > 0:
            logger.info(
                f"Applying minimum resolution filter with threshold {args.minimum_resolution}"
            )
            df = df[df[args.height_field] >= args.minimum_resolution]
            logger.info(f"Filtered to {len(df)} rows.")
        if (
            args.width_field in df.columns
            and args.height_field in df.columns
            and args.minimum_pixel_area > 0
        ):
            # megapixel to pixel:
            minimum_pixel_area = args.minimum_pixel_area * 1000000
            logger.info(
                f"Applying minimum pixel area filter with threshold {minimum_pixel_area}"
            )
            df = df[df[args.width_field] * df[args.height_field] >= minimum_pixel_area]
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
        logger.info(f"Fetching {len(to_fetch)} images (truncated): {to_fetch[:5]}")
        fetch_data(to_fetch, args, uri_column)

        # Remove source file if argument is provided
        if args.delete_after_processing:
            try:
                os.remove(file)
            except:
                pass


if __name__ == "__main__":
    main()
