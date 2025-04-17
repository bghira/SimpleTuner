import os
import logging
import re
import random
import argparse
import base64
import torch
import pillow_jxl
from PIL import Image
from tqdm import tqdm
import requests
import io
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    StoppingCriteria,
)
import pandas as pd
import torch.nn as nn

logger = logging.getLogger("Captioner")


# Define the prompt template for BLIP3
def apply_prompt_template(prompt):
    s = (
        "<|system|>\nYou are an image tagger. Provide image tags only separated by commas.<|end|>\n"
        f"<|user|>\n<image>\n{prompt}\n<|end|>\n<|assistant|>\n"
    )
    return s


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence) :].tolist()
        return self.eos_sequence in last_ids


# load the existing parquet file if it exists
def load_input_parquet(parquet_path: str):
    df = pd.read_parquet(path=parquet_path)
    return df


# Load BLIP3 model
def load_blip3_model():
    model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    tokenizer = model.update_special_tokens(tokenizer)
    return (
        model.to("mps" if torch.backends.mps.is_available() else "cuda"),
        tokenizer,
        image_processor,
    )


# Function to evaluate BLIP3 model
def eval_blip3_model(query, raw_image, model, tokenizer, image_processor):
    prompt = apply_prompt_template(query)
    inputs = image_processor(
        [raw_image], return_tensors="pt", image_aspect_ratio="anyres"
    )
    language_inputs = tokenizer([prompt], return_tensors="pt")
    inputs.update(language_inputs)
    inputs = {name: tensor.to("mps" if torch.backends.mps.is_available() else "cuda") for name, tensor in inputs.items()}
    generated_text = model.generate(
        **inputs,
        image_size=[raw_image.size],
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        max_new_tokens=512,
        top_p=0.95,
        top_k=50,
        num_beams=1,
        stopping_criteria=[EosListStoppingCriteria()],
    )
    prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split(
        "<|end|>"
    )[0]
    # If it doesn't end on a complete sentence, remove everything after the final '.'
    # if not prediction.endswith("."):
    #     # Remove everything after the final '.'
    #     prediction = re.sub(r"\.[^.]*$", ".", prediction)

    return prediction


def process_and_evaluate_image(
    args, image_path: str, model, tokenizer, image_processor
):
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

    result = eval_blip3_model(
        args.query_str,
        resize_for_condition_image(image, 384),
        model,
        tokenizer,
        image_processor,
    )
    print(f"Result for captioning: {result}")
    return result


def process_directory(
    args,
    image_dir,
    output_parquet,
    model,
    tokenizer,
    image_processor,
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
                        args, full_filepath, model, tokenizer, image_processor
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

    model, tokenizer, image_processor = load_blip3_model()
    process_directory(
        args,
        args.input_dir,
        args.output_parquet,
        model,
        tokenizer,
        image_processor,
        input_parquet=input_database,
        original_query_str=str(args.query_str),
    )


if __name__ == "__main__":
    main()
