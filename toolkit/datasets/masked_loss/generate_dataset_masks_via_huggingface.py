import argparse
import os
import shutil
from gradio_client import Client, handle_file


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Mask images in a directory using Florence SAM Masking."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory to save masked images.",
    )
    parser.add_argument(
        "--text_input",
        type=str,
        default="person",
        help='Text prompt for masking (default: "person").',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SkalskiP/florence-sam-masking",
        help='Model name to use (default: "SkalskiP/florence-sam-masking").',
    )
    args = parser.parse_args()

    input_path = args.input_dir
    output_path = args.output_dir
    text_input = args.text_input
    model_name = args.model

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Initialize the Gradio client
    client = Client(model_name)

    # Get all files in the input directory
    files = os.listdir(input_path)

    # Iterate over all files
    for file in files:
        # Construct the full file path
        full_path = os.path.join(input_path, file)
        # Check if the file is an image
        if os.path.isfile(full_path) and full_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            # Define the path for the output mask
            mask_path = os.path.join(output_path, file)
            # Skip if the mask already exists
            if os.path.exists(mask_path):
                print(f"Mask already exists for {file}, skipping.")
                continue
            # Predict the mask
            try:
                mask_filename = client.predict(
                    image_input=handle_file(full_path),
                    text_input=text_input,
                    api_name="/process_image",
                )
                # Move the generated mask to the output directory
                shutil.move(mask_filename, mask_path)
                print(f"Saved mask to {mask_path}")
            except Exception as e:
                print(f"Failed to process {file}: {e}")


if __name__ == "__main__":
    main()
