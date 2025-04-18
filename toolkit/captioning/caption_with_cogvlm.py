import os, torch, logging, accelerate, re, random, argparse, io
from tqdm.auto import tqdm
try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image
import requests, boto3
from botocore.config import Config

logger = logging.getLogger("Captioner")


def upload_to_s3(s3_client, bucket_name, image_data, object_name):
    try:
        in_memory_file = io.BytesIO()
        # Save PIL image to the bytes buffer
        image_data.save(in_memory_file, format="PNG")
        in_memory_file.seek(0)  # Move to the beginning of the buffer

        s3_client.upload_fileobj(
            in_memory_file,
            bucket_name,
            object_name,
        )
    except Exception as e:
        logger.error(f"Error uploading {object_name} to bucket {bucket_name}: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process images and generate captions."
    )
    parser.add_argument(
        "--multidatabackend_config",
        type=str,
        required=True,
        help="Path to the multidatabackend config JSON file.",
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing the images."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Directory to save processed images.",
    )
    parser.add_argument(
        "--target_backend_id",
        type=str,
        default=None,
        help=(
            "When this is provided, the script will upload the captioned file directly to an S3 backend in your multidatabackend.json file."
        ),
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
        type=bool,
        default=True,
        help="Disable filename cleaning. This may result in filenames that are too long for some operating systems, but better captions.",
    )
    parser.add_argument(
        "--query_str",
        type=str,
        default="Caption this image accurately, with as few words as possible.",
        help="The query string to use for captioning. This instructs the model how to behave.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/cogvlm2-llama3-chat-19B",
        help="Model path to load. Default: THUDM/cogvlm2-llama3-chat-19B, also useful is THUDM/cogvlm-chat-hf to utilise CogVLM v1.1",
    )
    return parser.parse_args()


def load_filter_list(filter_list_path):
    if filter_list_path and os.path.exists(filter_list_path):
        with open(filter_list_path, "r") as file:
            return [line.strip() for line in file if line.strip()]
    return []


def eval_image(
    image: Image.Image,
    model,
    tokenizer,
    torch_dtype,
    query: str,
):
    inputs = model.build_conversation_input_ids(
        tokenizer, query=query, history=[], images=[image]
    )  # chat mode
    inputs = {
        "input_ids": inputs["input_ids"].unsqueeze(0).to("cuda"),
        "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to("cuda"),
        "attention_mask": inputs["attention_mask"].unsqueeze(0).to("cuda"),
        "images": [[inputs["images"][0].to("cuda").to(torch_dtype)]],
    }
    gen_kwargs = {"max_new_tokens": 77, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(outputs[0])


def content_to_filename(content, filter_terms, disable_filename_cleaning: bool = False):
    """
    Function to convert content to filename by stripping specified terms,
    replacing non-alphanumeric characters and spaces, converting to lowercase,
    removing leading/trailing underscores, and limiting filename length to 230.
    """
    cleaned_content = content.replace("</s>", "")
    if disable_filename_cleaning:
        return f"{cleaned_content}.png"
    for term in filter_terms:
        cleaned_content = cleaned_content.replace(term, "")
    # Split on '--' and take the first part
    cleaned_content = cleaned_content.split("--")[0]

    # Remove URLs
    cleaned_content = re.sub(r"https*://\S*", "", content)

    cleaned_content = cleaned_content.replace("The image showcases a ", "")
    cleaned_content = cleaned_content.replace(" appears to be", "")
    cleaned_content = cleaned_content.replace(" its ", " ")

    # Remove </s>

    # Replace non-alphanumeric characters and spaces, convert to lowercase, remove leading/trailing underscores
    cleaned_content = re.sub(r"[^a-zA-Z0-9 ]", "", cleaned_content)
    cleaned_content = cleaned_content.replace(" ", "_").lower().strip("_")

    # If cleaned_content is empty after removing URLs, generate a random filename
    if cleaned_content == "":
        cleaned_content = f"midjourney_{random.randint(0, 1000000)}"

    # Limit filename length to 230
    cleaned_content = (
        cleaned_content[:230] if len(cleaned_content) > 230 else cleaned_content
    )

    return cleaned_content + ".png"


def process_directory(
    args,
    image_dir,
    output_dir,
    model,
    tokenizer,
    processed_files,
    caption_strategy,
    save_interval,
    progress_file,
    filter_terms,
    torch_dtype,
    query_str: str,
):
    processed_file_counter = 0
    bucket_name = None
    if args.target_backend_id:
        if not args.multidatabackend_config:
            raise ValueError(
                "A --multidatabackend_config must be provided when --target_backend_id is provided."
            )
        # Load the config
        import json

        with open(args.multidatabackend_config) as file:
            multidatabackend_config = json.load(file)
        for backend in multidatabackend_config:
            if backend["id"] == args.target_backend_id and backend["type"] != "aws":
                raise ValueError(
                    "Only S3 backends are supported for --target_backend_id."
                )
            elif backend["id"] == args.target_backend_id:
                config = backend
                bucket_name = config["aws_bucket_name"]
                break
        if not bucket_name:
            raise ValueError(
                f"The backend id '{args.target_backend_id}' was not found in the multidatabackend config."
            )

        s3_config = Config(max_pool_connections=100)

        s3_client = boto3.client(
            "s3",
            endpoint_url=backend["aws_endpoint_url"],
            region_name=backend.get("aws_region_name", None),
            aws_access_key_id=backend["aws_access_key_id"],
            aws_secret_access_key=backend["aws_secret_access_key"],
            config=s3_config,
        )

    for filename in tqdm(
        os.listdir(image_dir),
        desc=f"Processing directory {image_dir}",
        unit="images",
        leave=True,
        position=0,
        mininterval=0.5,
    ):
        full_filepath = os.path.join(image_dir, filename)
        if os.path.isdir(full_filepath):
            process_directory(
                args,
                full_filepath,
                output_dir,
                model,
                tokenizer,
                processed_files,
                caption_strategy,
                save_interval,
                progress_file,
                filter_terms,
                torch_dtype,
                query_str,
            )
        elif filename.lower().endswith((".jpg", ".png", ".jpeg")):
            if filename in processed_files:
                # Remove the original file if args.delete_after_caption
                if args.delete_after_caption:
                    os.remove(full_filepath)
                continue

            try:
                with Image.open(full_filepath) as image:
                    # Convert to RGB
                    image = image.convert("RGB")
                    best_match = eval_image(
                        image, model, tokenizer, torch_dtype, query_str
                    )
                    logging.debug(f"Best match for {filename}: {best_match}")
                    # Save based on caption strategy
                    new_filename = (
                        content_to_filename(
                            best_match, filter_terms, args.disable_filename_cleaning
                        )
                        if caption_strategy == "filename"
                        else filename
                    )
                    if args.output_dir:
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
                    else:
                        new_filepath = full_filepath
                    if args.target_backend_id:
                        upload_to_s3(s3_client, bucket_name, image, new_filename)

                    # Remove the original file if args.delete_after_caption
                    if args.delete_after_caption:
                        os.remove(full_filepath)

                if caption_strategy == "text":
                    with open(new_filepath + ".txt", "w", encoding="utf-8") as f:
                        f.write(best_match)

            except Exception as e:
                import traceback

                logging.error(
                    f"Error processing {filename}: {str(e)}, traceback: {traceback.format_exc()}"
                )
        processed_files.add(filename)
        processed_file_counter += 1
        # Save progress at specified intervals
        if processed_file_counter % save_interval == 0:
            with open(progress_file, "w", encoding="utf-8") as f:
                f.writelines("\n".join(processed_files))

    # Save remaining progress
    with open(progress_file, "w", encoding="utf-8") as f:
        f.writelines("\n".join(processed_files))


def main():
    args = parse_args()
    send_to_cuda = load_in_4bit = load_in_8bit = False
    torch_dtype = torch.bfloat16
    if args.precision == "bf16" or args.precision == "fp16":
        send_to_cuda = True
        if args.precision == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    elif args.precision == "fp4":
        load_in_4bit = True
    elif args.precision == "fp8":
        load_in_8bit = True
    logging.basicConfig(level=logging.INFO)

    # Ensure output directory exists
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info("Loading CogVLM model. This should only occur once.")
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer

    logger.info(f"Loading CogVLM in {args.precision} precision.")
    if "cogvlm2" in args.model_path and torch.backends.mps.is_available():
        logger.warning(
            "Can not run CogVLM 2 on MPS because Triton is unavailable. Falling back to CogVLM 1.1"
        )
    elif "cogvlm2" in args.model_path:
        import sysconfig

        print(sysconfig.get_paths()["include"])
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
    else:
        tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    ).eval()
    if send_to_cuda:
        logger.info(f"Sending model to CUDA.")
        model.to("cuda")
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
        args,
        args.input_dir,
        args.output_dir,
        model,
        tokenizer,
        processed_files,
        args.caption_strategy,
        args.save_interval,
        progress_file,
        filter_terms,
        torch_dtype,
        args.query_str,
    )

    # Save progress
    with open(progress_file, "w", encoding="utf-8") as f:
        f.writelines("\n".join(processed_files))


if __name__ == "__main__":
    main()
