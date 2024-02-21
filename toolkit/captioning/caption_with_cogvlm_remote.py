import torch, logging, argparse, io, base64
from PIL import Image
import requests, time
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed


logger = logging.getLogger("Captioner")


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


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.eval_backend == "transformers":
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

        logger.info("Loading CogVLM model. This should only occur once.")
        from transformers import AutoModelForCausalLM, LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        logger.info(f"Loading CogVLM in {args.precision} precision.")
        accelerator_project_config = ProjectConfiguration()
        accelerator = Accelerator(
            mixed_precision="fp16",
            log_with=None,
            project_config=accelerator_project_config,
        )
        print(f"Device: {accelerator.device}")

        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm-chat-hf",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        ).eval()
        if send_to_cuda:
            logger.info(f"Sending model to CUDA.")
            model.to(accelerator.device)
        logger.info("Completed loading model.")

    # Query backend for tasks, and loop.
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
                },
            )
            # 403? Exit.
            if response.status_code == 403:
                logger.error("Access denied. Exiting.")
                break
            # 500? Wait.
            if response.status_code == 500:
                logger.error("Server error. Waiting.")
                time.sleep(30)
                continue
            # Decode the JSON response?
            try:
                response_json = response.json()
            except:
                logger.error("Could not decode JSON response. Exiting.")
                break
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
                # Skip if the task is pending or has a result
                if task["pending"] == 1 or task["result"]:
                    continue
                # Load the image
                try:
                    logger.info(f"Loading image from {task['URL']}.")
                    response = requests.get(task["URL"], timeout=10)
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                except Exception as e:
                    logger.error(f"Could not load image from {task['URL']}.")
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
                    continue
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
                logger.info(f"Result: {caption}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            time.sleep(15)


if __name__ == "__main__":
    main()
