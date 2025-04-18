import threading
import queue
import requests
import torch, base64, logging, io, time
import argparse
try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image
from io import BytesIO
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm as tq
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize queues
job_queue = queue.Queue(maxsize=64)
submission_queue = queue.Queue()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process images and generate captions."
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "fp16", "fp4", "fp8"],
        default="fp8",
        help=(
            "When loading CogVLM, you can load it in fp16, bf16, fp8 or fp4 precision to reduce memory. Default: fp4"
        ),
    )
    parser.add_argument(
        "--disable_compile",
        type=bool,
        default=False,
        help=(
            "Provide --disable_compile=true to disable Torch compile. Required on Mac and AMD, perhaps. Default: false."
        ),
    )

    parser.add_argument(
        "--query_str",
        type=str,
        default="Caption this image accurately, without speculation. Just describe what you see.",
        help="The query string to use for captioning. This instructs the model how to behave.",
    )
    parser.add_argument(
        "--backend_url",
        type=str,
        required=True,
        help=(
            "The URL of the backend to use for processing. This should be the URL of the backend that will be used to process the images."
        ),
    )
    parser.add_argument(
        "--job_type",
        type=str,
        choices=["caption", "vae", "dataset_upload"],
        required=True,
        help=("The type of encoding to produce."),
    )
    parser.add_argument(
        "--client_id",
        type=str,
        required=True,
        help=("The client ID to use to login to the backend."),
    )
    parser.add_argument(
        "--secret",
        type=str,
        required=True,
        help=("The secret to use to login to the backend."),
    )
    parser.add_argument(
        "--eval_backend",
        type=str,
        default="transformers",
        choices=["oobabooga", "transformers"],
        help=(
            "If transformers is provided, it will load the model using the transformers library. If oobabooga is provided, it will use the API for a running installation of Oobabooga's text-generation-webui."
        ),
    )
    parser.add_argument(
        "--aws_config",
        type=str,
        default=None,
        help=("If provided, can post images directly to S3."),
    )
    parser.add_argument(
        "--aws_endpoint_url",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--aws_region_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--aws_access_key_id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--aws_secret_access_key",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=(
            "When querying the backend, how many jobs to request at once. Decreasing this will put more load on the remote server. Do not do that. Default: 16"
        ),
    )
    return parser.parse_args()


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


def eval_image_with_ooba(image: Image.Image, query: str) -> str:
    CONTEXT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"

    img_str = base64.b64encode(image).decode("utf-8")
    prompt = (
        CONTEXT
        + f'### Human: {query} \n<img src="data:image/jpeg;base64,{img_str}">### Assistant: '
    )

    data = {
        "mode": "instruct",  # chat, instruct
        "character": "Example",
        #           "instruction_template": "LLaVA-v1",
        "messages": [{"role": "system", "content": prompt}],
    }

    response = requests.post("http://127.0.0.1:5000/v1/chat/completions", json=data)

    if response.status_code != 200:
        print(
            f"Request failed with status {response.status_code}. Response: {response.text}"
        )
        import sys

        sys.exit(1)
    else:
        return response.json()["choices"][0]["message"]["content"]


def encode_images(accelerator, vae, images, image_transform):
    """
    Encode a batch of input images. Images must be the same dimension.
    """

    pixel_values = [
        image_transform(image).to(accelerator.device, dtype=vae.dtype)
        for image in images
    ]

    with torch.no_grad():
        processed_images = torch.stack(pixel_values).to(
            accelerator.device, dtype=torch.bfloat16
        )
        latents = vae.encode(processed_images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    return latents


from math import sqrt


def round_to_nearest_multiple(value, multiple: int = 64):
    """Round a value to the nearest multiple."""
    rounded = round(value / multiple) * multiple
    return max(rounded, multiple)  # Ensure it's at least the value of 'multiple'


def calculate_new_size_by_pixel_area(W: int, H: int, megapixels: float = 1.0):
    aspect_ratio = W / H
    total_pixels = megapixels * 1e6  # Convert megapixels to pixels

    W_new = int(round(sqrt(total_pixels * aspect_ratio)))
    H_new = int(round(sqrt(total_pixels / aspect_ratio)))

    # Ensure they are divisible by 64
    W_new = round_to_nearest_multiple(W_new)
    H_new = round_to_nearest_multiple(H_new)

    return W_new, H_new


def initialize_s3_client(args):
    """Initialize the boto3 S3 client using the provided AWS credentials and settings."""
    import boto3
    from botocore.config import Config

    s3_config = Config(max_pool_connections=100)

    s3_client = boto3.client(
        "s3",
        endpoint_url=args.aws_endpoint_url,
        region_name=args.aws_region_name,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        config=s3_config,
    )
    return s3_client


def submit_response(args, files, object_etag, task):
    submission_response = requests.post(
        f"{args.backend_url}/?action=submit_job",
        files=files,
        params={
            "result": object_etag,
            "job_id": task["data_id"],
            "client_id": args.client_id,
            "secret": args.secret,
            "status": "success",
            "job_type": "dataset_upload",
        },
    )


def upload_sample(args, image, task, local_progress_bar, files=None):
    if args.aws_config:
        s3_client = initialize_s3_client(args)
    if image is None:
        return None
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)
    files = None
    if args.aws_config:
        # post the image to s3
        attempt = 0
        while attempt < 3:
            try:
                tq.write(
                    f"Attempting to upload to {args.aws_bucket_name}: image_data/{task['data_id']}.png"
                )
                before_time = time.time()
                response_meta = s3_client.put_object(
                    Bucket=args.aws_bucket_name,
                    Key=f"image_data/{task['data_id']}.png",
                    Body=image_buffer,
                )
                after_time = time.time()
                tq.write(f"Received data in {after_time - before_time} seconds.")
                object_etag = response_meta["ETag"]
                tq.write(f"Object Etag: {object_etag}")
                break
            except Exception as e:
                tq.write(f"Failed to upload image to s3: {e}. Attempting again.")
                attempt += 1
                time.sleep(5)
        if attempt == 3:
            tq.write(f"Failed to upload image to s3. Skipping.")
            local_progress_bar.update(1)
            return None
    else:
        object_etag = None
        files = {
            "image_file": (
                "image.png",
                image_buffer,
                "image/png",
            ),
        }
    before_time = time.time()
    submit_response(args, files, object_etag, task)
    after_time = time.time()
    tq.write(f"Submitted result in {after_time - before_time} seconds.")


def load_image_from_url(url):
    tq.write(f"Begin load URL: {url}")
    before_time = time.time()
    attempts = 0
    while attempts < 3:
        try:
            result = Image.open(
                io.BytesIO(requests.get(url, timeout=10).content)
            ).convert("RGB")
            break
        except Exception as e:
            tq.write(f"-> [error] Could not load image from {url}. Retrying...")
            attempts += 1
            time.sleep(5)
    if attempts == 3:
        tq.write(f"-> [error] Failed to load image from {url}.")
        return None
    after_time = time.time()
    tq.write(f"Received image in {after_time - before_time} seconds.")
    return result


def load_images_in_parallel(tasks):
    images = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_url = {
            executor.submit(load_image_from_url, task["URL"]): task for task in tasks
        }
        for future in as_completed(future_to_url):
            task = future_to_url[future]
            try:
                image = future.result()
                if image is not None:
                    images[task["data_id"]] = image
            except Exception as exc:
                print(f'{task["data_id"]} generated an exception: {exc}')
    return images


def main():
    args = parse_args()
    if args.aws_config:
        with open(args.aws_config, "r") as f:
            import json

            aws_config = json.load(f)
            args.aws_endpoint_url = aws_config["aws_endpoint_url"]
            args.aws_region_name = aws_config["aws_region_name"]
            args.aws_access_key_id = aws_config["aws_access_key_id"]
            args.aws_secret_access_key = aws_config["aws_secret_access_key"]
            args.aws_bucket_name = aws_config["aws_bucket_name"]
    logging.basicConfig(level=logging.INFO)
    import warnings

    warnings.filterwarnings("ignore")
    accelerator_project_config = ProjectConfiguration()
    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with=None,
        project_config=accelerator_project_config,
    )

    if args.job_type == "vae":
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            force_upcast=False,
            torch_dtype=torch.bfloat16,
        ).to(accelerator.device)
        if not args.disable_compile:
            vae = torch.compile(vae, fullgraph=True)
        from torchvision import transforms

        image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    elif args.eval_backend == "transformers" and args.job_type == "caption":
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

        if accelerator.is_main_process:
            tq.write("Loading CogVLM model. This should only occur once.")
        from transformers import AutoModelForCausalLM, LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        if accelerator.is_main_process:
            tq.write(f"Loading CogVLM in {args.precision} precision.")

        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm-chat-hf",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        ).eval()
        if send_to_cuda:
            if accelerator.is_main_process:
                tq.write(f"Sending model to CUDA.")
            model.to(accelerator.device)
        if accelerator.is_main_process:
            tq.write("Completed loading model.")
    # Split device by : and use 2nd half as position
    local_bar_pos = 1
    cuda_device = f"{accelerator.device}"
    if ":" in cuda_device:
        local_bar_pos = int(cuda_device.split(":")[1]) + 1
    local_progress_bar = tq(
        desc="Local progress         ",
        dynamic_ncols=False,
        ascii=False,
        # We want this to be positioned by the accelerator rank
        position=local_bar_pos,
        leave=True,
        ncols=125,
        disable=not accelerator.is_main_process,
    )
    initial_cluster_progress = 0
    global_progress_bar = tq(
        desc="Global cluster progress",
        dynamic_ncols=False,
        ascii=False,
        position=0,
        leave=True,
        ncols=125,
        # We want to disable if not on main proc.
        disable=not accelerator.is_main_process,
    )

    # Query backend for tasks, and loop.
    has_set_total = False
    while True:
        try:
            # Query backend for tasks
            before_time = time.time()
            response = requests.get(
                f"{args.backend_url}/?action=list_jobs",
                timeout=30,
                params={
                    "client_id": args.client_id,
                    "secret": args.secret,
                    "count": args.batch_size,
                    "job_type": args.job_type,
                },
            )
            after_time = time.time()
            tq.write(f"Received job in {after_time - before_time} seconds.")
            # 403? Exit.
            if response.status_code == 403:
                tq.write("Access denied. Exiting.")
                break
            # 500? Wait.
            if response.status_code == 500:
                tq.write("Server error. Waiting.")
                time.sleep(30)
                continue
            # Decode the JSON response?
            try:
                response_json = response.json()
            except:
                tq.write(f"Could not decode JSON response: {response.text}")
                time.sleep(5)
                continue
            # Example:
            # [
            #     {
            #         "data_id": 474,
            #         "URL": "https://images.pexels.com/photos/474/black-and-white-car-vehicle-vintage.jpg?cs=srgb&dl=pexels-gratisography-474.jpg&fm=jpg",
            #         "pending": 0,
            #         "result": None,
            #         "submitted_at": None,
            #         "attempts": 0,
            #         "error": None,
            #         "client_id": None,
            #         "updated_at": "2024-02-21 02:32:32",
            #     }
            # ]

            # Now, we evaluate the caption for each image.
            if response_json[0]["job_type"] == "dataset_upload":
                # Prepare the file data for uploading
                tq.write(f"Received data upload job: {response_json[0]['data_id']}.")
                # Grab all images in the batch at once via threads:
                images = load_images_in_parallel(response_json)
                with ThreadPoolExecutor(max_workers=20) as executor:
                    future_to_task = {
                        executor.submit(
                            upload_sample,
                            args,
                            image,
                            {"data_id": data_id},
                            local_progress_bar,
                        ): data_id
                        for data_id, image in images.items()
                    }
                    for future in as_completed(future_to_task):
                        data_id = future_to_task[future]
                        try:
                            future.result()
                        except Exception as exc:
                            print(f"{data_id} upload generated an exception: {exc}")
                # close all images
                for image in images.values():
                    image.close()
                continue
            for task in response_json:
                if not has_set_total:
                    initial_cluster_progress = task.get("completed_jobs", 0)
                    global_progress_bar.total = task.get("remaining_jobs", 0)
                    local_progress_bar.total = task.get("remaining_jobs", 0)
                    has_set_total = True
                # Skip if the task is pending or has a result
                if "pending" not in task:
                    tq.write(f"Received invalid task: {task}. Skipping.")
                    continue
                if (task["pending"] == 1 and args.job_type == "vae") or (
                    task["result"] and args.job_type == "dataset_upload"
                ):
                    tq.write(
                        f"-> [warning] Task {task['data_id']} is pending? {task['pending']} or has a result {task['result']}. Skipping."
                    )
                    local_progress_bar.update(1)
                    continue
                # Load the image
                attempt_count = 0
                while attempt_count < 3:
                    try:
                        # tq.write(f"Loading image from {task['URL']}.")
                        response = requests.get(task["URL"], timeout=10)
                        image = Image.open(io.BytesIO(response.content)).convert("RGB")
                        break
                    except Exception as e:
                        tq.write(
                            f'-> [error] Could not load image from {task["URL"]}. {"Retrying..." if attempt_count < 2 else "Dropping sample."}'
                        )
                        # Upload the error using endpoint?action=submit_job&job_id=data_id&error=message&status=error
                        requests.post(
                            f"{args.backend_url}/?action=submit_job",
                            params={
                                "client_id": args.client_id,
                                "secret": args.secret,
                                "error": e,
                                "job_id": {task["data_id"]},
                            },
                        )
                        attempt_count += 1
                        time.sleep(5)
                if attempt_count == 3:
                    tq.write(f"-> [error] Failed to load image from {task['URL']}.")
                    local_progress_bar.update(1)
                    continue
                if "job_type" not in task:
                    tq.write(f"Received invalid task: {task}. Skipping.")
                    continue
                if task["job_type"] == "caption":
                    # Generate the caption
                    caption_source = "cogvlm"
                    if args.eval_backend == "transformers":
                        caption = eval_image(
                            image, model, tokenizer, torch_dtype, args.query_str
                        )
                    elif args.eval_backend == "oobabooga":
                        # only really llava is supported by oobabooga, so we will assume here.
                        caption_source = "llava"
                        image_to_bytes = io.BytesIO()
                        image.save(image_to_bytes, format="JPEG")
                        image_to_bytes = image_to_bytes.getvalue()
                        caption = eval_image_with_ooba(image_to_bytes, args.query_str)
                    # Upload the caption using endpoint?action=submit_job&job_id=data_id&result=caption&status=success
                    submission_response = requests.post(
                        f"{args.backend_url}/?action=submit_job",
                        params={
                            "result": caption,
                            "job_id": task["data_id"],
                            "client_id": args.client_id,
                            "secret": args.secret,
                            "caption_source": caption_source,
                            "status": "success",
                        },
                    )
                elif task["job_type"] == "vae":
                    (
                        target_width,
                        target_height,
                    ) = calculate_new_size_by_pixel_area(image.width, image.height)
                    # Resize image
                    resized_image = image.resize(
                        (target_width, target_height), resample=Image.LANCZOS
                    )
                    # Generate the latent vector
                    latents = encode_images(
                        accelerator, vae, [resized_image], image_transforms
                    )
                    # Unsqueeze the latents into separate objects:
                    latents = latents[0]
                    # Save the tensor to a BytesIO object (in-memory file)
                    latents_buffer = BytesIO()
                    torch.save(latents, latents_buffer)
                    latents_buffer.seek(
                        0
                    )  # Important: move back to the start of the buffer
                    files = {
                        "result_file": (
                            "latents.pt",
                            latents_buffer,
                            "application/octet-stream",
                        ),
                    }
                    submission_response = requests.post(
                        f"{args.backend_url}/?action=submit_job",
                        files=files,
                        params={
                            "result": "encoded latents",
                            "job_id": task["data_id"],
                            "client_id": args.client_id,
                            "secret": args.secret,
                            "status": "success",
                            "job_type": "vae",
                        },
                    )
                    # print(f"Submission response: {submission_response.text}")
                image.close()

                current_cluster_progress = (
                    task.get("completed_jobs", 0) - initial_cluster_progress
                )
                global_progress_bar.n = current_cluster_progress
                local_progress_bar.update(1)
            global_progress_bar.refresh()

        except Exception as e:
            import traceback

            tq.write(f"An error occurred: {e}, traceback: {traceback.format_exc()}")
            time.sleep(15)


if __name__ == "__main__":
    main()
