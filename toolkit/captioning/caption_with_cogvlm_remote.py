import torch, logging, argparse, io, base64
from PIL import Image
import requests, time
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm as tq
from io import BytesIO


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
        choices=["caption", "vae"],
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
    return parser.parse_args()


def eval_image(
    image: Image,
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


def eval_image_with_ooba(image: Image, query: str) -> str:
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


def round_to_nearest_multiple(value, multiple):
    """Round a value to the nearest multiple."""
    rounded = round(value / multiple) * multiple
    return max(rounded, multiple)  # Ensure it's at least the value of 'multiple'


def calculate_new_size_by_pixel_area(W: int, H: int, megapixels: float = 1.0):
    aspect_ratio = W / H
    total_pixels = megapixels * 1e6  # Convert megapixels to pixels

    W_new = int(round(sqrt(total_pixels * aspect_ratio)))
    H_new = int(round(sqrt(total_pixels / aspect_ratio)))

    # Ensure they are divisible by 64
    W_new = round_to_nearest_multiple(W_new, 64)
    H_new = round_to_nearest_multiple(H_new, 64)

    return W_new, H_new


def main():
    args = parse_args()
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
        vae = torch.compile(vae, fullgraph=True)
        from torchvision import transforms

        image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    elif args.eval_backend == "transformers" and not args.job_type == "vae":
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
            response = requests.get(
                f"{args.backend_url}/?action=list_jobs",
                timeout=30,
                params={
                    "client_id": args.client_id,
                    "secret": args.secret,
                    "count": 16,
                    "job_type": args.job_type,
                },
            )
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
                if task["pending"] == 1 or task["result"]:
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
                    # Prepare the file data for uploading
                    image_buffer = BytesIO()
                    image.save(image_buffer, format="PNG")
                    image_buffer.seek(0)
                    files = {
                        "result_file": (
                            "latents.pt",
                            latents_buffer,
                            "application/octet-stream",
                        ),
                        "image_file": (
                            "image.png",
                            image_buffer,
                            "image/png",
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
