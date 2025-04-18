import os
import logging
import requests
import random
import argparse
import base64
try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image
from tqdm import tqdm
import io
import pandas as pd

logger = logging.getLogger("Captioner")


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Define the prompt template for GPT-4
def apply_prompt_template(prompt):
    return [{"type": "text", "text": prompt}]


# Load the existing parquet file if it exists
def load_input_parquet(parquet_path: str):
    df = pd.read_parquet(path=parquet_path)
    return df


# Function to get image captions from GPT-4 API
def get_image_captions(image_path, api_key, prompt):
    base64_image = encode_image(image_path)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": apply_prompt_template(prompt)
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    if response.status_code == 200:
        result = response.json()
        caption = result["choices"][0]["message"]["content"]
        return caption
    else:
        logger.error(f"Error {response.status_code}: {response.text}")
        return "Error in captioning"


def process_and_evaluate_image(args, image_path: str, api_key):
    return get_image_captions(image_path, api_key, args.query_str)


def process_directory(
    args,
    image_dir,
    output_parquet,
    api_key,
    input_parquet=None,
    original_query_str=None,
):
    records = []
    parquet_path = f"{output_parquet}.{os.path.basename(image_dir)}.parquet"
    print(f"Parquet: {parquet_path}")
    total_to_process = 10000
    total_processed = 0
    for filename in tqdm(os.listdir(image_dir), desc="Processing Images"):
        if input_parquet is not None:
            current_caption = input_parquet[input_parquet["filename"] == filename]
            hint_column = args.input_parquet_hint_column
            hint_value = None
            if hint_column is not None and hint_column != "":
                try:
                    hint_value = current_caption[hint_column].values[0]
                except:
                    hint_value = None
                if hint_value is not None and not hint_value == "":
                    if original_query_str is not None:
                        args.query_str = original_query_str
                    args.query_str = args.query_str.replace("%s", hint_value)
                    logger.info(
                        f"Using query string: {args.query_str} for hint value: {hint_value}"
                    )
            try:
                if (
                    not current_caption.empty
                    and not current_caption["caption"].isnull().values[0]
                ):
                    logger.debug(f"Already has caption: {current_caption['caption']}")
                    continue
            except:
                logger.debug(f"Error checking for existing caption: {current_caption}")
        full_filepath = os.path.join(image_dir, filename)
        if os.path.isdir(full_filepath):
            logger.info(f"Found directory to traverse: {full_filepath}")
            process_directory(
                args,
                full_filepath,
                output_parquet,
                api_key,
                input_parquet=input_parquet,
                original_query_str=original_query_str,
            )
            args.query_str = original_query_str
            original_query_str = None
        elif filename.lower().endswith((".jpg", ".png", ".jpeg")):
            try:
                logger.debug(f"Attempting to load image: {filename}")
                with Image.open(full_filepath) as image:
                    logger.debug(f"Processing image: {filename}, data: {image}")
                    best_match = process_and_evaluate_image(
                        args, full_filepath, api_key
                    )
                    total_processed += 1
                    logger.debug(f"Best match for {filename}: {best_match}")

                    with Image.open(full_filepath) as img_file:
                        image_bytes = img_file.tobytes()

                    records.append({"filename": filename, "caption": best_match})
                    if total_processed >= total_to_process:
                        break

            except Exception as e:
                import traceback

                logger.error(
                    f"Error processing {filename}: {str(e)}, traceback: {traceback.format_exc()}"
                )
    new_df = pd.DataFrame(records)
    if input_parquet is not None:
        input_parquet.set_index("filename", inplace=True)
        new_df.set_index("filename", inplace=True)
        combined_df = input_parquet.combine_first(new_df).reset_index()
    else:
        combined_df = new_df
    combined_df = combined_df.drop_duplicates(subset=["filename"])
    combined_df.to_parquet(parquet_path, engine="pyarrow")
    logger.info(f"Processed Parquet file saved to {output_parquet}")


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
        default="What’s in this image?",
        help="The query string to use for captioning. This instructs the model how to behave.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for accessing the GPT-4 API.",
    )
    parser.add_argument(
        "--input_parquet",
        type=str,
        default=None,
        help="Path to the input Parquet dataset which will be adjusted to have the new column.",
    )
    parser.add_argument(
        "--input_parquet_hint_column",
        type=str,
        default="title",
        help="When set, the column to use as a hint for the input query str placement value. Default: title",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    input_database = None
    if args.input_parquet:
        if not os.path.exists(args.input_parquet):
            raise ValueError("The parquet file specified as input did not exist.")

        input_database = load_input_parquet(args.input_parquet)

    process_directory(
        args,
        args.input_dir,
        args.output_parquet,
        args.api_key,
        input_parquet=input_database,
        original_query_str=str(args.query_str),
    )


if __name__ == "__main__":
    main()
