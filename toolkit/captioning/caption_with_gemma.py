import os, json
import logging
import argparse
import requests
try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

logger = logging.getLogger("Captioner")


# Function to load PaliGemma model and processor
def load_pali_gemma_model(args):
    model_id = args.model_path
    model = (
        PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        .to(torch.float32)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def generate_caption_with_pali_gemma(
    image_path, processor, model, query_strings, do_sample, temperature
):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)

    model_inputs = processor(
        text=query_strings,
        images=[image] * len(query_strings),
        return_tensors="pt",
        padding=True,
    )
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=do_sample,
            # temperature=temperature,
            # top_p=0.5,
            # top_k=80
        )
        outputs = []
        for _generation in generation:
            decoded = processor.decode(
                _generation[input_len:], skip_special_tokens=True
            )
            outputs.append(decoded)
    return outputs


def process_and_evaluate_image(args, image_path, model, processor):
    query_strings = ["caption en"]
    result = generate_caption_with_pali_gemma(
        image_path, processor, model, query_strings, do_sample=True, temperature=1.2
    )
    return result[0]
    output = {query: result for query, result in zip(query_strings, result)}
    print(f"Output: {json.dumps(output, indent=4)}")
    return output


def process_directory(args, image_dir, output_parquet, model, processor):
    records = []
    parquet_path = f"{output_parquet}.{os.path.basename(image_dir)}.parquet"
    print(f"Parquet: {parquet_path}")
    for filename in tqdm(os.listdir(image_dir), desc="Processing Images"):
        full_filepath = os.path.join(image_dir, filename)
        if os.path.isdir(full_filepath):
            logging.info(f"Found directory to traverse: {full_filepath}")
            process_directory(args, full_filepath, output_parquet, model, processor)
        elif filename.lower().endswith((".jpg", ".png")):
            try:
                logging.info(f"Attempting to load image: {filename}")
                with Image.open(full_filepath) as image:
                    logging.debug(f"Processing image: {filename}, data: {image}")
                    best_match = process_and_evaluate_image(
                        args, full_filepath, model, processor
                    )
                    logging.info(f"Best match for {filename}: {best_match}")

                    with Image.open(full_filepath) as img_file:
                        image_bytes = img_file.tobytes()

                    records.append(
                        {
                            "filename": filename,
                            "caption": best_match,
                            "image": image_bytes,
                        }
                    )

            except Exception as e:
                import traceback

                logging.error(
                    f"Error processing {filename}: {str(e)}, traceback: {traceback.format_exc()}"
                )
                if "CUDA error" in str(e):
                    import sys

                    sys.exit(1)

    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, engine="pyarrow")
    logging.info(f"Processed Parquet file saved to {output_parquet}")


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
        "--precision",
        type=str,
        choices=["bf16", "fp16"],
        default="fp16",
        help=("Precision for loading the model. Default: fp16"),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="google/paligemma-3b-mix-224",
        help=("Model path to load. Default: google/paligemma-3b-mix-224"),
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    model, processor = load_pali_gemma_model(args)
    process_directory(args, args.input_dir, args.output_parquet, model, processor)


if __name__ == "__main__":
    main()
