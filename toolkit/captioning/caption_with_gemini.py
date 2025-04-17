import os
import logging
import argparse
import re
import random
import base64
import requests
import time
import io
import pandas as pd
import google.generativeai as genai
import pillow_jxl
from PIL import Image
from tqdm import tqdm
import pyarrow.parquet as pq

logger = logging.getLogger("Captioner")

# Configure the API Key for Google Generative AI
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


# Function to upload image and generate caption using Google Generative AI
def generate_caption_with_genai(image_path, query_str: str):
    image_input = genai.upload_file(
        path=image_path,
        mime_type="image/jpeg",  # Adjust if your image is a different type
    )
    prompt = f"{query_str}: {image_input.uri}"
    model = genai.GenerativeModel(model_name="gemini-pro-vision")
    attempt = 0
    while attempt < 3:
        try:
            caption = model.generate_content([prompt, image_input])
            return caption.text
        except Exception as e:
            logging.error(f"Error generating caption: {e}")
            attempt += 1
            if attempt >= 3:
                raise Exception(f"Failed to generate caption after 3 attempts: {e}")


def get_quota_reset_time(headers):
    # Example header: "X-Quota-Reset: 3600" (in seconds)
    print(f"Headers: {headers}")
    reset_time = headers.get("X-Quota-Reset", 3600)
    return int(reset_time)


def calculate_sleep_time(reset_time):
    # Current time in seconds since the epoch
    current_time = int(time.time())
    return max(0, reset_time - current_time)


def process_and_evaluate_image(args, image_path):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)
        image = Image.open(io.BytesIO(response.content))
    else:
        image = Image.open(image_path)

    def resize_for_condition_image(input_image: Image.Image, resolution: int):
        if resolution == 0:
            return input_image
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        aspect_ratio = round(W / H, 2)
        if W < H:
            W = resolution
            H = int(resolution / aspect_ratio)
        elif H < W:
            H = resolution
            W = int(resolution * aspect_ratio)
        if W == H:
            W = resolution
            H = resolution
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img

    resized_image_path = "/tmp/resized_image.jpg"
    resize_for_condition_image(image, 384).save(resized_image_path)

    result = generate_caption_with_genai(resized_image_path, args.query_str)
    return result


def process_directory(args, image_dir, output_parquet):
    records = []
    parquet_path = f"{output_parquet}.{os.path.basename(image_dir)}.parquet"
    if os.path.exists(parquet_path):
        existing_df = pd.read_parquet(parquet_path)
        processed_files = set(existing_df["filename"].tolist())
    else:
        existing_df = pd.DataFrame()
        processed_files = set()

    for filename in tqdm(os.listdir(image_dir), desc="Processing Images"):
        full_filepath = os.path.join(image_dir, filename)
        if os.path.isdir(full_filepath):
            logging.info(f"Found directory to traverse: {full_filepath}")
            process_directory(args, full_filepath, output_parquet)
        elif (
            filename.lower().endswith((".jpg", ".png"))
            and filename not in processed_files
        ):
            try:
                logging.info(f"Attempting to load image: {filename}")
                with Image.open(full_filepath) as image:
                    logging.info(f"Processing image: {filename}, data: {image}")
                    best_match = process_and_evaluate_image(args, full_filepath)
                    logging.info(f"Best match for {filename}: {best_match}")

                    records.append({"filename": filename, "caption": best_match})
                    if len(records) % 10 == 0:  # Save every 10 records
                        save_parquet(records, parquet_path, existing_df)

            except Exception as e:
                import traceback

                logging.error(
                    f"Error processing {filename}: {str(e)}, traceback: {traceback.format_exc()}"
                )
                if "check quota" in str(e):
                    import sys

                    sys.exit(1)

    save_parquet(records, parquet_path, existing_df)
    logging.info(f"Processed Parquet file saved to {output_parquet}")


def save_parquet(records, parquet_path, existing_df):
    df = pd.DataFrame(records)
    if not df.empty:
        combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(
            subset=["filename"]
        )
        combined_df.to_parquet(parquet_path, engine="pyarrow")
        logging.info(f"Saved {len(records)} records to {parquet_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process images and generate captions."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing the images."
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        required=True,
        help="Path to the output Parquet dataset.",
    )
    parser.add_argument(
        "--query_str",
        type=str,
        default="Provide the most detailed caption.",
        help="The query string to use for captioning. This instructs the model how to behave.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "fp16"],
        default="fp16",
        help=("Precision for loading the model. Default: fp16"),
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    process_directory(args, args.input_dir, args.output_parquet)


if __name__ == "__main__":
    main()
