import os
from pathlib import Path
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
import pandas as pd
import torch.nn as nn
import numpy as np
import glob
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

logger = logging.getLogger("Captioner")


# load the existing parquet file if it exists
def load_input_parquet(parquet_path: str):
    df = pd.read_parquet(path=parquet_path)
    return df


# Load InterVL2-8B model, only need 24G VRAM,if you wanna to load bigger models like 26B or 72B,you should need 1-3 80G A100
def load_model(model_name_or_path="OpenGVLab/InternVL2-8B"):
    model = (
        AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )

    return (model, tokenizer)


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
        default="OpenGVLab/InternVL2-8B",
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
        help="The query string to use for captioning. This instructs the model how to behave. Not normally needed for InternVL",
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


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    # print(f"Original image filename: {image_file}")
    image = Image.open(image_file).convert("RGB")
    width, height = image.size
    # print(f"Original image size: {image.size}")
    mode = image.mode
    # print(f"Original image mode: {mode}")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    # print(f"Size after dynamic_preprocess: {images[0].size}")
    pixel_values = [transform(image) for image in images]
    # print(f"Size after transform: {pixel_values[0].shape}")
    pixel_values = torch.stack(pixel_values)
    return pixel_values, width, height


def process_directory(
    args,
    image_dir,
    output_parquet,
    model,
    tokenizer,
    max_new_tokens,
    input_parquet=None,
    original_query_str=None,
    total_to_process: int = None,
):
    records = []
    directory_path = Path(image_dir)
    parquet_path = f"{output_parquet}.{directory_path.name}.parquet"
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
                tokenizer,
                input_parquet=input_parquet,
                original_query_str=original_query_str,
            )
            args.query_str = original_query_str
            original_query_str = None
        elif filename.lower().endswith((".jpg", ".png", ".jpeg")):
            try:
                logger.debug(f"Attempting to load image: {filename}")
                logger.debug(f"Processing image: {filename}")
                # set the max number of tiles in `max_num`
                pixel_values, width, height = load_image(full_filepath, max_num=12)
                pixel_values, width, height = (
                    pixel_values.to(torch.bfloat16).to(device),
                    width,
                    height,
                )
                generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

                question = "<image>\n" + str(original_query_str)
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=None,
                    return_history=False,
                )

                total_processed += 1
                logger.debug(f"Best match for {filename}: {response}")

                # with Image.open(full_filepath) as img_file:
                #     image_bytes = img_file.tobytes()

                records.append(
                    {
                        "filename": filename,
                        "caption": response,
                        "width": width,
                        "height": height,
                    }
                )
                if total_to_process is not None and total_processed >= total_to_process:
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


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    input_database = None
    if args.input_parquet:
        if not os.path.exists(args.input_parquet):
            raise ValueError("The parquet file specified as input did not exist.")

        input_database = load_input_parquet(args.input_parquet)

    model, tokenizer = load_model(args.model_name)
    process_directory(
        args,
        args.input_dir,
        args.output_parquet,
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        input_parquet=input_database,
        original_query_str=str(args.query_str),
    )


if __name__ == "__main__":
    main()
