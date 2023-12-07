import requests
import re
import os
import argparse
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
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


def download_smallest_image(urls, output_path, minimum_image_size: int):
    """Download the smallest image based on width."""
    smallest_url = min(urls, key=get_image_width)
    response = requests.get(smallest_url, stream=True)

    if response.status_code == 200:
        filename = os.path.basename(smallest_url.split("?")[0])
        file_path = os.path.join(output_path, filename)

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        # Is the file >= 1024px on both sides?
        image = Image.open(file_path)
        width, height = image.size
        if width < minimum_image_size or height < minimum_image_size:
            os.remove(file_path)
            return f"Nuked tiny image: {smallest_url}"

        return f"Downloaded: {smallest_url}"
    return f"Failed to download: {smallest_url}"


def main(args):
    os.makedirs(args.output_path, exist_ok=True)

    url_groups = {}

    with open(args.file_path, "r") as file:
        for line in file:
            urls = line.strip().split()
            for url in urls:
                photo_id = get_photo_id(url)
                if photo_id:
                    if photo_id not in url_groups:
                        url_groups[photo_id] = []
                    url_groups[photo_id].append(url)

    # Using ThreadPoolExecutor to parallelize downloads
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_smallest_image, urls, args.output_path, args.minimum_image_size
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
        help="Both sides of the image must be larger than this.",
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
