import os
import logging
import re
import random
import argparse
import base64
import torch

try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image
from tqdm import tqdm
import requests
import io
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)
import pandas as pd
import torch.nn as nn

logger = logging.getLogger("Captioner")


# load the existing parquet file if it exists
def load_input_parquet(parquet_path: str):
    df = pd.read_parquet(path=parquet_path)
    return df


# Load Florence model
def load_model(model_name_or_path="microsoft/Florence-2-large-ft"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,  # cache_dir="/home/user/storage/hf_cache"
    )
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,  # cache_dir="/home/user/storage/hf_cache"
    )
    return (
        model.to("mps" if torch.backends.mps.is_available() else "cuda"),
        processor,
    )


# Function to evaluate BLIP3 model
def eval_model(args, image, model, processor, task="<MORE_DETAILED_CAPTION>"):
    inputs = processor(
        text=f"{task}{args.query_str}", images=image, return_tensors="pt"
    )

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        pixel_values=inputs["pixel_values"].to(model.device, dtype=model.dtype),
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    prediction = processor.post_process_generation(
        generated_text,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(image.width, image.height),
    )

    # If it doesn't end on a complete sentence, remove everything after the final '.'
    # if not prediction.endswith("."):
    #     # Remove everything after the final '.'
    #     prediction = re.sub(r"\.[^.]*$", ".", prediction)

    return prediction[task]


def process_and_evaluate_image(args, image_path: str, model, image_processor):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)
        image = Image.open(io.BytesIO(response.content))
    else:
        image = Image.open(image_path)

    result = eval_model(
        args,
        image,
        model,
        image_processor,
    )
    logger.info(f"Result for captioning: {result}")
    return result


def process_directory(
    args,
    image_dir,
    output_parquet,
    model,
    image_processor,
    input_parquet=None,
    original_query_str=None,
    total_to_process: int = None,
):
    records = []
    parquet_path = f"{output_parquet}.{os.path.basename(image_dir)}.parquet"
    print(f"Parquet: {parquet_path}")
    total_processed = 0
    for filename in tqdm(os.listdir(image_dir), desc="Processing Images"):
        if input_parquet is not None:
            # if the caption column at the filename position is non-empty, skip
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
                model,
                image_processor,
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
                        args, full_filepath, model, image_processor
                    )
                    total_processed += 1
                    logger.debug(f"Best match for {filename}: {best_match}")

                    # with Image.open(full_filepath) as img_file:
                    #     image_bytes = img_file.tobytes()

                    records.append({"filename": filename, "caption": best_match})
                    if (
                        total_to_process is not None
                        and total_processed >= total_to_process
                    ):
                        break

            except Exception as e:
                import traceback

                logger.error(
                    f"Error processing {filename}: {str(e)}, traceback: {traceback.format_exc()}"
                )
                if "CUDA error" in str(e):
                    import sys

                    sys.exit(1)
    new_df = pd.DataFrame(records)
    if input_parquet is not None:
        # Merge new_df with input_parquet
        input_parquet.set_index("filename", inplace=True)
        new_df.set_index("filename", inplace=True)
        combined_df = input_parquet.combine_first(new_df).reset_index()
    else:
        combined_df = new_df
    # reduce duplicates by "filename" contents
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
        "--model_name",
        type=str,
        default="microsoft/Florence-2-large-ft",
        help="Model name to use for captioning.",
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
        default="",
        help="The query string to use for captioning. This instructs the model how to behave. Not normally needed for Florence",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "fp16"],
        default="fp16",
        help=("Precision for loading the model. Default: fp16"),
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
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="The maximum number of tokens to generate. Default: 1024",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help=(
            "Whether to use sampling for generation. Makes model more responsive to input prompts."
            " If not set, greedy decoding is used. Default: False"
        ),
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

    model, image_processor = load_model(args.model_name)
    process_directory(
        args,
        args.input_dir,
        args.output_parquet,
        model,
        image_processor,
        input_parquet=input_database,
        original_query_str=str(args.query_str),
    )


if __name__ == "__main__":
    main()
