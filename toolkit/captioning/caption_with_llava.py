import os, logging, re, random, argparse, json, torch
from PIL import Image
from tqdm import tqdm
import requests

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model

logger = logging.getLogger("Captioner")

# Directory where the images are located
input_directory_path = "/notebooks/datasets/Hands"
output_dir = "/notebooks/datasets/caption_output"
caption_strategy = "filename"


# Function to load LLaVA model
def load_llava_model(model_path: str = "liuhaotian/llava-v1.5-7b"):
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
    )
    return tokenizer, model, image_processor, context_len


# Function to evaluate the model
def eval_model(args, tokenizer, model, image_processor, context_len):
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = "llava_v1"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    images = [args.image_file]
    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs


def process_and_evaluate_image(
    image_path: str,
    tokenizer,
    model,
    image_processor,
    context_len: int,
    model_path: str = "liuhaotian/llava-v1.5-7b",
    prompt: str = "Describe what the hand is doing.",
):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)
        image = Image.open(io.BytesIO(response.content))
    else:
        image = Image.open(image_path)

    llava_args = type(
        "Args",
        (),
        {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": prompt,
            "conv_mode": None,
            "image_file": image,
            "sep": ",",
            "temperature": 0.2,
            "top_p": 0.1,
            "num_beams": 1,
            "max_new_tokens": context_len,
        },
    )()

    return eval_model(llava_args, tokenizer, model, image_processor, context_len)


# Function to convert content to filename
def content_to_filename(content):
    """
    Function to convert content to filename by stripping everything after '--',
    replacing non-alphanumeric characters and spaces, converting to lowercase,
    removing leading/trailing underscores, and limiting filename length to 128.
    """
    # Split on '--' and take the first part
    content = content.split("--")[0]

    # Remove URLs
    cleaned_content = re.sub(r"https*://\S*", "", content)

    # Replace non-alphanumeric characters and spaces, convert to lowercase, remove leading/trailing underscores
    cleaned_content = re.sub(r"[^a-zA-Z0-9 ]", "", cleaned_content)
    cleaned_content = cleaned_content.replace(" ", "_").lower().strip("_")

    # If cleaned_content is empty after removing URLs, generate a random filename
    if cleaned_content == "":
        cleaned_content = f"midjourney_{random.randint(0, 1000000)}"

    # Limit filename length to 128
    cleaned_content = (
        cleaned_content[:128] if len(cleaned_content) > 128 else cleaned_content
    )

    return cleaned_content + ".png"


# Main processing function with progress saving
def process_directory(
    image_dir, output_dir, progress_file, model, tokenizer, image_processor, context_len
):
    # Load progress if exists
    if os.path.exists(progress_file):
        with open(progress_file, "r") as file:
            processed_files = json.load(file)
    else:
        processed_files = {}

    for filename in tqdm(os.listdir(image_dir), desc="Processing Images"):
        full_filepath = os.path.join(image_dir, filename)
        if filename in processed_files:
            continue

        if os.path.isdir(full_filepath):
            process_directory(
                full_filepath, model, context_len
            )  # Recursive call for directories
        elif filename.lower().endswith((".jpg", ".png")):
            try:
                with Image.open(full_filepath) as image:
                    best_match = process_and_evaluate_image(
                        full_filepath, tokenizer, model, image_processor, context_len
                    )
                    logging.info(f"Best match for {filename}: {best_match}")
                    # Save based on caption strategy
                    new_filename = (
                        content_to_filename(best_match)
                        if caption_strategy == "filename"
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

                if caption_strategy == "text":
                    with open(new_filepath + ".txt", "w") as f:
                        f.write(best_match)

                # Update progress
                processed_files[filename] = best_match
                with open(progress_file, "w") as file:
                    json.dump(processed_files, file)

            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Image Captioning with LLaVA model")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory path"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory path"
    )
    parser.add_argument(
        "--model_path", type=str, default="liuhaotian/llava-v1.5-7b", help="Model path"
    )
    parser.add_argument(
        "--progress_file",
        type=str,
        default="progress.json",
        help="File to save progress",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load model
    tokenizer, model, image_processor, context_len = load_llava_model(args.model_path)

    # Process directory
    process_directory(
        args.input_dir,
        args.output_dir,
        args.progress_file,
        model,
        tokenizer,
        image_processor,
        context_len,
    )


if __name__ == "__main__":
    main()
