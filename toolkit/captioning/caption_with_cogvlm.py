import os, torch, logging, xformers, accelerate, re, random
from PIL import Image
import requests

logger = logging.getLogger("Captioner")

# Directory where the images are located
input_directory_path = "/notebooks/datasets/Hands"
output_dir = "/notebooks/datasets/caption_output"
caption_strategy = "filename"
batch_size = 16  # You can change this value as needed

def eval_image(image: Image, model, tokenizer, query: str = "Describe the image accurately."):
    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_new_tokens": 77, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0])

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
    
    # Remove </s>
    cleaned_content = cleaned_content.replace("</s>", "")
    
    return cleaned_content + ".png"


def process_directory(image_dir="images", model=None, context_len: int = 77):
    if model is None:
        logger.info("Loading CogVLM model. This should only occur once.")
        from transformers import AutoModelForCausalLM, LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-chat-hf',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()
        logger.info("Completed loading model.")

    for filename in os.listdir(image_dir):
        full_filepath = os.path.join(image_dir, filename)
        if os.path.isdir(full_filepath):
            process_directory(full_filepath, model)
        elif filename.lower().endswith((".jpg", ".png")):
            try:
                with Image.open(full_filepath) as image:
                    best_match = eval_image(
                        image, model, tokenizer
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

            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_directory(input_directory_path)
