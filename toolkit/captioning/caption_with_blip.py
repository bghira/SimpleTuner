import os, torch, logging, re, random
import pillow_jxl
from PIL import Image
from clip_interrogator import Config, Interrogator, LabelTable, load_list
from clip_interrogator import clip_interrogator
clip_interrogator.CAPTION_MODELS.update({
    'unography': 'unography/blip-large-long-cap',           # 1.9GB
})
print(f"Models supported: {clip_interrogator.CAPTION_MODELS}")

# Directory where the images are located
input_directory_path = "/Volumes/datasets/photo-concept-bucket/image_data"
output_dir = "/Volumes/datasets/photo-concept-bucket/image_data_captioned"
caption_strategy = "text"


def content_to_filename(content):
    """
    Function to convert content to filename by stripping everything after '--',
    replacing non-alphanumeric characters and spaces, converting to lowercase,
    removing leading/trailing underscores, and limiting filename length to 128.
    """
    # Split on '--' and take the first part
    content = content.split("--", 1)[0]

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


def interrogator(
    clip_model_name="ViT-H-14/laion2b_s32b_b79k", blip_model="unography"
):
    # Create an Interrogator instance with the latest CLIP model for Stable Diffusion 2.1
    conf = Config(
        clip_model_name=clip_model_name, clip_offload=True, caption_offload=True, caption_max_length=170, device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    conf.caption_model_name = blip_model
    ci = Interrogator(conf)
    return ci


def load_terms(filename, interrogator_instance):
    # Load your list of terms
    table = LabelTable(load_list(filename), "terms", interrogator_instance)
    logging.debug(f"Loaded {len(table)} terms from {filename}")
    return table


def process_directory(image_dir="images", terms_file=None, active_interrogator=None):
    if active_interrogator is None:
        active_interrogator = interrogator()
    if terms_file is not None:
        table = load_terms(terms_file, active_interrogator)

    for filename in os.listdir(image_dir):
        full_filepath = os.path.join(image_dir, filename)
        if os.path.isdir(full_filepath):
            process_directory(full_filepath, terms_file, active_interrogator)
        elif filename.lower().endswith((".jpg", ".png")):
            try:
                image = Image.open(full_filepath).convert("RGB")
                if terms_file:
                    best_match = table.rank(
                        active_interrogator.image_to_features(image), top_count=1
                    )[0]
                else:
                    best_match = active_interrogator.generate_caption(image)

                logging.info(f"Best match for {filename}: {best_match}")

                # Save based on caption strategy
                new_filename = (
                    content_to_filename(best_match)
                    if caption_strategy == "filename"
                    else filename
                )
                new_filepath = os.path.join(output_dir, new_filename)

                if caption_strategy == "text":
                    with open(new_filepath + ".txt", "w") as f:
                        f.write(best_match)
                else:
                    # Ensure no overwriting
                    counter = 1
                    while os.path.exists(new_filepath):
                        new_filepath = os.path.join(
                            output_dir,
                            f"{new_filename.rsplit('.', 1)[0]}_{counter}.{new_filename.rsplit('.', 1)[1]}",
                        )
                        counter += 1

                    image.save(new_filepath)
                image.close()


            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_directory(input_directory_path)
