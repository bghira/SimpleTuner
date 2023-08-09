---

## interrogate.py - Image Captioning Script

`interrogate.py` processes a directory of images, generating captions using CLIP and BLIP models. The generated captions can either be used to rename the image files or be saved alongside the image in a text file.

### Requirements

- Python 3.9
- PIL (Pillow)
- accelerate
- xformers
- CLIP Interrogator (specifically the modules `Config`, `Interrogator`, `LabelTable`, and `load_list`)

### Configuration

- `input_directory_path`: The directory where the images to be processed are located.
- `output_dir`: The directory where the processed images (if renamed) and their captions (if saved as text) will be stored.
- `caption_strategy`: Determines how captions are used. Can be 'filename' (rename the image files using the captions) or 'text' (save captions in a separate text file).

### How to Use

1. Ensure all dependencies are installed.
2. Configure the script by setting the appropriate values for `input_directory_path`, `output_dir`, and `caption_strategy`.
3. Run the script using `python script_name.py`.

### Features

- **Recursive Processing**: The script can process nested directories of images.
- **Filename Safety**: Image files are renamed based on their captions in a way that ensures unique and safe filenames.
- **Error Handling**: Errors encountered during image processing are logged, ensuring the script continues with the remaining images.

### Known Issues

- Ensure there are no cyclic references in the directory structure, as this could lead to infinite recursion.