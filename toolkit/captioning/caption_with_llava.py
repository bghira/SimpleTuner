import os, logging, re, random, argparse, json, torch
import pillow_jxl
from PIL import Image
from tqdm import tqdm
import requests, io

try:
    from transformers import BitsAndBytesConfig
except:
    pass
try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
except:
    pass
from transformers import AutoProcessor, LlavaForConditionalGeneration

logger = logging.getLogger("Captioner")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process images and generate captions."
    )
    parser.add_argument(
        "--input_dir", type=str, required=False, help="Directory containing the images."
    )
    parser.add_argument(
        "--input_parquet",
        type=str,
        default=None,
        help="Path to the input Parquet dataset.",
    )
    parser.add_argument(
        "--parquet_image_column",
        type=str,
        default="image_url",
        help="Column name in the Parquet file that contains the image URLs.",
    )
    parser.add_argument(
        "--parquet_target_column",
        type=str,
        default="llava_caption",
        help="Column name in the Parquet file where the generated captions will be saved.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Directory to save processed images.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for processing."
    )
    parser.add_argument(
        "--caption_strategy",
        type=str,
        default="filename",
        choices=["filename", "text"],
        help="Strategy for saving captions.",
    )
    parser.add_argument(
        "--filter_list",
        type=str,
        default=None,
        help=(
            "Path to a txt file containing terms or sentence fragments to filter out."
            " These will be removed from the final caption."
        ),
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Interval to save progress (number of files processed).",
    )
    parser.add_argument(
        "--delete_after_caption",
        action="store_true",
        default=False,
        help="Delete *input* image files after captioning.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "fp16", "fp4", "fp8"],
        default="fp4",
        help=(
            "When loading CogVLM, you can load it in fp16, bf16, fp8 or fp4 precision to reduce memory. Default: fp4"
        ),
    )
    parser.add_argument(
        "--disable_filename_cleaning",
        action="store_true",
        default=False,
        help="Disable filename cleaning. This may result in filenames that are too long for some operating systems, but better captions.",
    )
    parser.add_argument(
        "--query_str",
        type=str,
        default="Caption this image accurately, with as few words as possible.",
        help="The query string to use for captioning. This instructs the model how to behave.",
    )
    parser.add_argument(
        "--model_path", type=str, default="llava-hf/llava-1.5-7b-hf", help="Model path"
    )
    parser.add_argument(
        "--progress_file",
        type=str,
        default="progress.json",
        help="File to save progress",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=90,
        help=(
            "The maximum number of tokens a stable diffusion model can reasonably use is 77."
            " This allows returning longer than 77 token long captions, though their utility may be reduced."
        ),
    )
    args = parser.parse_args()
    if not args.input_dir and not args.input_parquet:
        parser.error("Either --input_dir or --input_parquet must be provided.")
    if args.input_dir and args.input_parquet:
        parser.error("Only one of --input_dir or --input_parquet can be provided.")
    if args.output_dir and args.input_parquet:
        parser.error(
            "--output_dir cannot be provided when --input_parquet is provided."
        )
    if not args.output_dir and args.input_dir:
        parser.error("--output_dir must be provided when --input_dir is provided.")

    return parser.parse_args()


import pandas as pd


def process_parquet_dataset(args, model, processor):
    df = pd.read_parquet(args.input_parquet)
    if args.parquet_image_column not in df.columns:
        logger.error(
            f"Image column '{args.parquet_image_column}' not found in the Parquet file."
        )
        return
    if args.parquet_target_column not in df.columns:
        df[args.parquet_target_column] = ""

    for index, row in tqdm(
        df.iterrows(), total=df.shape[0], desc="Processing Parquet Dataset"
    ):
        image_url = row[args.parquet_image_column]
        if not image_url:
            continue
        if row[args.parquet_target_column]:
            continue
        try:
            caption = process_and_evaluate_image(args, image_url, model, processor)
            df.at[index, args.parquet_target_column] = caption
            # Save the DataFrame back to a Parquet file
            output_parquet_path = (
                args.input_parquet.rsplit(".", 1)[0] + "_captioned.parquet"
            )
            df.to_parquet(output_parquet_path, engine="pyarrow")
        except Exception as e:
            logger.error(f"Error processing image at {image_url}: {str(e)}")

    logger.info(f"Processed Parquet file saved to {output_parquet_path}")


# Function to load LLaVA model
def load_llava_model(
    model_path: str = "llava-hf/llava-1.5-7b-hf", precision: str = "fp4"
):
    try:
        bnb_config = BitsAndBytesConfig()
        if precision == "fp4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif precision == "fp8":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            bnb_config = None
        torch_dtype = None
    except:
        bnb_config = None
        torch_dtype = torch.float16
    if "1.6" in model_path:
        logger.info("Using LLaVA 1.6+ model.")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="mps" if torch.backends.mps.is_available() else "auto",
        )
    else:
        logger.info("Using LLaVA 1.5 model.")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
    if "1.6" in model_path:
        logger.info("Using LLaVA 1.6+ model processor.")
        autoprocessor_cls = LlavaNextProcessor
    else:
        logger.info("Using LLaVA 1.5 model processor.")
        autoprocessor_cls = AutoProcessor
    processor = autoprocessor_cls.from_pretrained(model_path)

    return model, processor


# Function to evaluate the model
def eval_model(args, image_file, model, processor):

    if type(processor) == LlavaNextProcessor:
        logging.info("Using LLaVA 1.6+ model.")
        prompt = f"<|im_start|>system\nAnswer the question carefully, without speculation.<|im_end|><|im_start|>user\n<image>\n{args.query_str}<|im_end|><|im_start|>assistant\n"
        inputs = processor(text=prompt, images=image_file, return_tensors="pt").to(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logging.info(f"Inputs: {inputs}")
    else:
        prompt = f"<image>\nUSER: {args.query_str}\nASSISTANT:"
        images = [image_file]
        inputs = processor(text=prompt, images=images, return_tensors="pt").to(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs,
            max_length=args.max_new_tokens,
        )

    outputs = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    # Pull everything after "ASSISTANT":
    outputs = outputs.split("ASSISTANT:")[1].strip()
    return outputs


def process_and_evaluate_image(args, image_path: str, model, processor):
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

    result = eval_model(args, resize_for_condition_image(image, 256), model, processor)
    print(f"Result for captioning: {result}")
    return result


# Function to convert content to filename
def content_to_filename(args, content):
    """
    Function to convert content to filename by stripping everything after '--',
    replacing non-alphanumeric characters and spaces, converting to lowercase,
    removing leading/trailing underscores, and limiting filename length to 128.
    """
    # If --disable_filename_cleaning is provided, we just append ".png":
    if args.disable_filename_cleaning:
        return f"{content}.png"
    # Split on '--' and take the first part
    content = content.split("--")[0]

    # Remove URLs
    cleaned_content = re.sub(r"https*://\S*", "", content)

    # Replace non-alphanumeric characters and spaces, convert to lowercase, remove leading/trailing underscores
    # cleaned_content = re.sub(r"[^a-zA-Z0-9 ]", "", cleaned_content)
    # cleaned_content = cleaned_content.replace(" ", "_").lower().strip("_")

    # If cleaned_content is empty after removing URLs, generate a random filename
    if cleaned_content == "":
        cleaned_content = f"midjourney_{random.randint(0, 1000000)}"

    # Limit filename length to 128
    cleaned_content = (
        cleaned_content[:128] if len(cleaned_content) > 128 else cleaned_content
    )

    return cleaned_content + ".png"


# Main processing function with progress saving
def process_directory(args, image_dir, output_dir, progress_file, model, processor):
    # Load progress if exists
    if os.path.exists(progress_file):
        with open(progress_file, "r") as file:
            processed_files = json.load(file)
    else:
        processed_files = {}

    for filename in tqdm(os.listdir(image_dir), desc="Processing Images"):
        full_filepath = os.path.join(image_dir, filename)
        if image_dir in processed_files and filename in processed_files[image_dir]:
            logging.info(f"File has already been processed: {filename}")
            continue

        if os.path.isdir(full_filepath):
            logging.info(f"Found directory to traverse: {full_filepath}")
            process_directory(
                args, full_filepath, output_dir, progress_file, model, processor
            )  # Recursive call for directories
        elif filename.lower().endswith((".jpg", ".png")):
            try:
                logging.info(f"Attempting to load image: {filename}")
                with Image.open(full_filepath) as image:
                    logging.info(f"Processing image: {filename}, data: {image}")
                    best_match = process_and_evaluate_image(
                        args, full_filepath, model, processor
                    )
                    logging.info(f"Best match for {filename}: {best_match}")
                    # Save based on caption strategy
                    new_filename = (
                        content_to_filename(args, best_match)
                        if args.caption_strategy == "filename"
                        else filename
                    )
                    new_filepath = os.path.join(output_dir, new_filename)

                    # Ensure no overwriting
                    counter = 1
                    while os.path.exists(new_filepath):
                        new_filepath = os.path.join(
                            output_dir,
                            f"{new_filename.rsplit('.', 1)[0]}_{counter}.{new_filename.rsplit('.', 1)[1]}",
                        )
                        counter += 1

                    image.save(new_filepath)

                if args.caption_strategy == "text":
                    with open(new_filepath + ".txt", "w") as f:
                        f.write(best_match)

                # Update progress
                if image_dir not in processed_files:
                    processed_files[image_dir] = {}
                processed_files[image_dir][filename] = best_match
                with open(progress_file, "w") as file:
                    json.dump(processed_files, file)

            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                if "CUDA error" in str(e):
                    import sys

                    sys.exit(1)
                if "name too long" in str(e):
                    # Loop and try to reduce the filename length until it works:
                    exception_error = str(e)
                    while "name too long" in exception_error:
                        # Cut the word down by one character:
                        new_filename = new_filename[:-1]
                        try:
                            new_filepath = os.path.join(output_dir, new_filename)
                            # Try to save again
                            image.save(new_filepath)
                            exception_error = ""
                        except Exception as e:
                            exception_error = str(e)


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    # Ensure output directory exists
    if args.output_dir is not None and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load model
    model, processor = load_llava_model(args.model_path, args.precision)

    if args.input_parquet:
        # Process Parquet dataset
        process_parquet_dataset(args, model, processor)
    else:
        # Process directory
        process_directory(
            args,
            args.input_dir,
            args.output_dir,
            args.progress_file,
            model,
            processor,
        )


if __name__ == "__main__":
    main()
