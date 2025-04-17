import argparse
import os
import pillow_jxl
import PIL
import cv2  # Import OpenCV for image processing
import numpy as np
import supervision as sv
import torch
from PIL import Image, ImageOps
from typing import Union, Any, Tuple, Dict
from unittest.mock import patch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from gradio_client import handle_file
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Constants
FLORENCE_CHECKPOINT = "microsoft/Florence-2-large"
FLORENCE_OPEN_VOCABULARY_DETECTION_TASK = "<OPEN_VOCABULARY_DETECTION>"

SAM_CONFIG = "sam2_hiera_l.yaml"
SAM_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"


def load_sam_image_model(
    device: torch.device, config: str = SAM_CONFIG, checkpoint: str = SAM_CHECKPOINT
) -> SAM2ImagePredictor:
    model = build_sam2(config, checkpoint, device=device)
    return SAM2ImagePredictor(sam_model=model)


def run_sam_inference(
    model: Any, image: Image.Image, detections: sv.Detections
) -> sv.Detections:
    image_np = np.array(image.convert("RGB"))
    model.set_image(image_np)
    masks, scores, _ = model.predict(box=detections.xyxy, multimask_output=False)

    # Ensure mask dimensions are correct
    if len(masks.shape) == 4:
        masks = np.squeeze(masks)

    detections.mask = masks.astype(bool)
    return detections


def fixed_get_imports(filename: Union[str, os.PathLike]) -> list[str]:
    """Workaround for specific import issues."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def load_florence_model(
    device: torch.device, checkpoint: str = FLORENCE_CHECKPOINT
) -> Tuple[Any, Any]:
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = (
            AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
            .to(device)
            .eval()
        )
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        return model, processor


def run_florence_inference(
    model: Any,
    processor: Any,
    device: torch.device,
    image: Image.Image,
    task: str,
    text: str = "",
) -> Tuple[str, Dict]:
    prompt = task + text
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    response = processor.post_process_generation(
        generated_text, task=task, image_size=image.size
    )
    return generated_text, response


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
        "--invert_mask",
        action="store_true",
        help="Invert the mask to ignore the segmented portion instead of isolate it.",
    )
    parser.add_argument(
        "--mask_padding",
        type=int,
        default=0,
        help="Number of pixels to pad the mask (default: 0).",
    )
    parser.add_argument(
        "--mask_blur",
        type=int,
        default=0,
        help="Amount of Gaussian blur to apply to the mask edges (default: 0).",
    )
    args = parser.parse_args()
    if args.input_dir is None or args.output_dir is None:
        import sys

        sys.exit(1)

    input_path = args.input_dir
    output_path = args.output_dir
    text_input = args.text_input

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Retrieve model
    from huggingface_hub import hf_hub_download

    print(f"Downloading SAM2 to {os.getcwd()}/checkpoints.")
    hf_hub_download(
        "SkalskiP/florence-sam-masking",
        repo_type="space",
        subfolder="checkpoints",
        local_dir="./",
        filename="sam2_hiera_large.pt",
    )

    # Load models
    FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
    SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

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
                image_input = Image.open(full_path)
                # Convert to RGB
                image_input = image_input.convert("RGB")
                _, result = run_florence_inference(
                    model=FLORENCE_MODEL,
                    processor=FLORENCE_PROCESSOR,
                    device=DEVICE,
                    image=image_input,
                    task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                    text=text_input,
                )
                detections = sv.Detections.from_lmm(
                    lmm=sv.LMM.FLORENCE_2,
                    result=result,
                    resolution_wh=image_input.size,
                )
                if len(detections) == 0:
                    print(f"No objects detected in {file}.")
                    continue
                detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
                # Combine masks if multiple detections
                combined_mask = np.any(detections.mask, axis=0)

                # Apply mask padding if specified
                if args.mask_padding > 0:
                    kernel = np.ones((3, 3), np.uint8)
                    combined_mask = cv2.dilate(
                        combined_mask.astype(np.uint8),
                        kernel,
                        iterations=args.mask_padding,
                    )

                # Apply mask blurring if specified
                if args.mask_blur > 0:
                    combined_mask = combined_mask.astype(np.float32)
                    ksize = args.mask_blur * 2 + 1  # Kernel size must be odd
                    combined_mask = cv2.GaussianBlur(combined_mask, (ksize, ksize), 0)

                # Convert mask to image
                if args.mask_blur > 0:
                    mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8))
                else:
                    mask_image = Image.fromarray(combined_mask.astype(np.uint8) * 255)

                # Invert masks if necessary
                if args.invert_mask:
                    mask_image = ImageOps.invert(mask_image)

                mask_image.save(mask_path)
                print(f"Saved mask to {mask_path}")
            except Exception as e:
                print(f"Failed to process {file}: {e}")
    # Clean up
    os.remove("checkpoints/sam2_hiera_large.pt")
    os.rmdir("checkpoints")


if __name__ == "__main__":
    main()
