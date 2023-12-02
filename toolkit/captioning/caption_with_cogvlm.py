import os, torch, logging, xformers, accelerate, re, random, argparse
from tqdm.auto import tqdm
from PIL import Image
import requests

logger = logging.getLogger("Captioner")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process images and generate captions."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing the images."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
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
    return parser.parse_args()

def load_filter_list(filter_list_path):
    if filter_list_path and os.path.exists(filter_list_path):
        with open(filter_list_path, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    return []

def eval_image(
    image: Image,
    model,
    tokenizer,
    query: str = "Describe the image accurately and concisely, with few fill words.",
):
    inputs = model.build_conversation_input_ids(
        tokenizer, query=query, history=[], images=[image]
    )  # chat mode
    inputs = {
        "input_ids": inputs["input_ids"].unsqueeze(0).to("cuda"),
        "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to("cuda"),
        "attention_mask": inputs["attention_mask"].unsqueeze(0).to("cuda"),
        "images": [[inputs["images"][0].to("cuda").to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_new_tokens": 77, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(outputs[0])


def content_to_filename(content, filter_terms):
    """
    Function to convert content to filename by stripping specified terms,
    replacing non-alphanumeric characters and spaces, converting to lowercase,
    removing leading/trailing underscores, and limiting filename length to 128.
    """
    for term in filter_terms:
        content = content.replace(term, "")
    # Split on '--' and take the first part
    content = content.split("--")[0]

    # Remove URLs
    cleaned_content = re.sub(r"https*://\S*", "", content)

    cleaned_content = cleaned_content.replace("The image showcases a ", "")
    cleaned_content = cleaned_content.replace(" appears to be", "")
    cleaned_content = cleaned_content.replace(" its ", " ")

    # Remove </s>
    cleaned_content = cleaned_content.replace("</s>", "")

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


def process_directory(
    image_dir,
    output_dir,
    model,
    tokenizer,
    processed_files,
    caption_strategy,
    save_interval,
    progress_file,
    filter_terms
):
    processed_file_counter = 0

    for filename in tqdm(
        os.listdir(image_dir),
        desc="Processing Images",
        unit="images",
        leave=False,
        position=0,
        mininterval=0.5,
    ):
        if filename in processed_files:
            continue

        full_filepath = os.path.join(image_dir, filename)
        if os.path.isdir(full_filepath):
            process_directory(
                full_filepath,
                output_dir,
                model,
                tokenizer,
                processed_files,
                caption_strategy,
            )
        elif filename.lower().endswith((".jpg", ".png")):
            try:
                with Image.open(full_filepath) as image:
                    best_match = eval_image(image, model, tokenizer)
                    logging.info(f"Best match for {filename}: {best_match}")
                    # Save based on caption strategy
                    new_filename = (
                        content_to_filename(best_match, filter_terms)
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

            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
        processed_files.add(filename)
        processed_file_counter += 1
        # Save progress at specified intervals
        if processed_file_counter % save_interval == 0:
            with open(progress_file, "w") as f:
                f.writelines("\n".join(processed_files))

    # Save remaining progress
    with open(progress_file, "w") as f:
        f.writelines("\n".join(processed_files))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info("Loading CogVLM model. This should only occur once.")
    from transformers import AutoModelForCausalLM, LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    model = (
        AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm-chat-hf",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .to("cuda")
        .eval()
    )
    logger.info("Completed loading model.")
    processed_files = set()
    progress_file = os.path.join(args.input_dir, "processed_files.txt")

    # Load progress if exists
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            processed_files = set(f.read().splitlines())

    # Load a list of "filter terms". This is a text file that might look like:
    #
    # The image showcases a
    # closeup image of a
    # anotherterm
    #
    # Each filter term should be on its own line. The entire contents of the row will be removed.
    filter_terms = load_filter_list(args.filter_list)

    # Process directory
    process_directory(
        args.input_dir,
        args.output_dir,
        model,
        tokenizer,
        processed_files,
        args.caption_strategy,
        args.save_interval,
        progress_file,
        filter_terms
    )

    # Save progress
    with open(progress_file, "w") as f:
        f.writelines("\n".join(processed_files))


if __name__ == "__main__":
    main()
