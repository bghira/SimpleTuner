import requests
import re
import os
import argparse
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image


def get_image_width(url):
    """Extract width from the image URL."""
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    return int(query_params.get("w", [0])[0])


def get_photo_id(url):
    """Extract photo ID from the image URL."""
    match = re.search(r"/photos/(\d+)", url)
    return match.group(1) if match else None


conn_timeout = 6
read_timeout = 60
timeouts = (conn_timeout, read_timeout)


def download_image(url, output_path, minimum_image_size: int, minimum_pixel_area: int):
    """Download an image."""
    response = requests.get(url, timeout=timeouts, stream=True)

    if response.status_code == 200:
        filename = os.path.basename(url.split("?")[0])
        file_path = os.path.join(output_path, filename)
        # Convert path to PNG:
        file_path = file_path.replace(".jpg", ".png")

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        # Check if the file meets the minimum size requirements
        image = Image.open(file_path)
        width, height = image.size
        if minimum_image_size > 0 and (
            width < minimum_image_size or height < minimum_image_size
        ):
            os.remove(file_path)
            return f"Nuked tiny image: {url}"
        if minimum_pixel_area > 0 and (width * height < minimum_pixel_area):
            os.remove(file_path)
            return f"Nuked tiny image: {url}"

        return f"Downloaded: {url}"
    return f"Failed to download: {url}"


def process_urls(urls, output_path, minimum_image_size: int, minimum_pixel_area: int):
    """Process a list of URLs."""
    # Simple URL list
    results = []
    for url in urls:
        result = download_image(
            url, output_path, minimum_image_size, minimum_pixel_area
        )
        results.append(result)
    return "\n".join(results)


def main(args):
    os.makedirs(args.output_path, exist_ok=True)

    url_groups = {}

    with open(args.file_path, "r") as file:
        for line in file:
            urls = line.strip().split()
            # Treat as a simple URL list
            url_groups[line] = urls

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_urls,
                urls,
                args.output_path,
                args.minimum_image_size,
                args.minimum_pixel_area,
            )
            for urls in url_groups.values()
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading images"
        ):
            if args.debug:
                print(future.result())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download smallest images from Pexels."
    )
    parser.add_argument(
        "--file_path", type=str, help="Path to the text file containing image URLs."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the directory where images will be saved.",
    )
    parser.add_argument(
        "--minimum_image_size",
        type=int,
        default=0,
        help="Both sides of the image must be larger than this. ZERO disables this.",
    )
    parser.add_argument(
        "--minimum_pixel_area",
        type=int,
        default=0,
        help="The total number of pixels in the image must be larger than this. ZERO disables this. Recommended value: 1024*1024",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of worker threads. Default is 64.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug messages.",
    )
    args = parser.parse_args()
    main(args)
