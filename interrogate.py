import os, logging, xformers, accelerate, re
from PIL import Image
from clip_interrogator import Config, Interrogator, LabelTable, load_list

# Directory where the images are located
output_dir = '/models/training/datasets/processed_animals'

def content_to_filename(content):
    """
    Function to convert content to filename by stripping everything after '--', 
    replacing non-alphanumeric characters and spaces, converting to lowercase, 
    removing leading/trailing underscores, and limiting filename length to 128.
    """
    # Split on '--' and take the first part
    content = content.split('--', 1)[0] 

    # Remove URLs
    cleaned_content = re.sub(r'https*://\S*', '', content)

    # Replace non-alphanumeric characters and spaces, convert to lowercase, remove leading/trailing underscores
    cleaned_content = re.sub(r'[^a-zA-Z0-9 ]', '', cleaned_content)
    cleaned_content = cleaned_content.replace(' ', '_').lower().strip('_')

    # If cleaned_content is empty after removing URLs, generate a random filename
    if cleaned_content == '':
        cleaned_content = f'midjourney_{random.randint(0, 1000000)}'
    
    # Limit filename length to 128
    cleaned_content = cleaned_content[:128] if len(cleaned_content) > 128 else cleaned_content

    return cleaned_content + '.png'

def interrogator(clip_model_name = "ViT-H-14/laion2b_s32b_b79k"):
    # Create an Interrogator instance with the latest CLIP model for Stable Diffusion 2.1
    ci = Interrogator(Config(clip_model_name=clip_model_name))
    return ci

def load_terms(filename, interrogator):
    # Load your list of terms
    table = LabelTable(load_list('terms.txt'), 'terms', interrogator)
    logging.debug(f'Loaded {len(table)} terms from {filename}')
    return table

active_interrogator = None
def process_directory(image_dir = 'images', terms_file = None):
    # Go through all the images in the directory
    global active_interrogator
    if active_interrogator is None:
        active_interrogator = interrogator()
    if terms_file is not None:
        table = load_terms(terms_file, active_interrogator)

    for filename in os.listdir(image_dir):
        # Is the file a directory?
        if os.path.isdir(os.path.join(image_dir, filename)):
            # Recursively process the directory
            process_directory(os.path.join(image_dir, filename), terms_file)
        elif filename.endswith(".jpg") or filename.endswith(".png"):
            # Open and convert the image
            image = Image.open(os.path.join(image_dir, filename)).convert('RGB')
            if terms_file is not None:
                # Get the best match for the image
                best_match = table.rank(active_interrogator.image_to_features(image), top_count=1)[0]
            else:
                best_match = active_interrogator.generate_caption(image)

            # Print the result
            logging.info(f'Best match for {filename}: {best_match}')
            # Write the best match to {filename}.txt:
            image.save(os.path.join(output_dir, content_to_filename(best_match)))
            

if __name__ == "__main__":
    process_directory('/models/training/datasets/animals')
