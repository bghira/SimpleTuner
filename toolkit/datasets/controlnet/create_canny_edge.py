import os
import cv2
import pillow_jxl
from PIL import Image


def generate_canny_edge_dataset(input_dir, output_dir_original, output_dir_edges):
    # Create output directories if they do not exist
    if not os.path.exists(output_dir_original):
        os.makedirs(output_dir_original)
    if not os.path.exists(output_dir_edges):
        os.makedirs(output_dir_edges)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_dir, filename)
            original_image = Image.open(image_path)
            original_image.save(os.path.join(output_dir_original, filename))

            # Read image in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Apply Canny edge detection
            edges = cv2.Canny(image, 100, 200)

            # Save edge image
            edge_image_path = os.path.join(output_dir_edges, filename)
            cv2.imwrite(edge_image_path, edges)
            print(f"Processed {filename}")


if __name__ == "__main__":
    input_dir = (
        "/Volumes/ml/datasets/animals/antelope"  # Update this to your folder path
    )
    output_dir_original = "/Volumes/ml/datasets/canny-edge/animals/antelope-data"  # Update this to your desired output path for originals
    output_dir_edges = "/Volumes/ml/datasets/canny-edge/animals/antelope-conditioning"  # Update this to your desired output path for edges
    generate_canny_edge_dataset(input_dir, output_dir_original, output_dir_edges)
