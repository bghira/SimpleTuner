import argparse
import importlib
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jxl"}


def generate_canny_edge_dataset(input_dir, output_dir_original, output_dir_edges):
    source = Path(input_dir).resolve()
    originals = Path(output_dir_original).resolve()
    conditioning = Path(output_dir_edges).resolve()
    if not source.is_dir():
        raise ValueError("The input directory must exist.")
    for first, second in ((source, originals), (source, conditioning), (originals, conditioning)):
        if first.is_relative_to(second) or second.is_relative_to(first):
            raise ValueError("The input, original output, and conditioning output directories must be disjoint.")

    for image_path in sorted(source.rglob("*")):
        suffix = image_path.suffix.lower()
        if not image_path.is_file() or suffix not in IMAGE_SUFFIXES:
            continue
        if suffix == ".jxl":
            importlib.import_module("pillow_jxl")

        relative_path = image_path.relative_to(source)
        original_path = originals / relative_path
        edge_path = conditioning / relative_path
        if not original_path.resolve().is_relative_to(originals) or not edge_path.resolve().is_relative_to(conditioning):
            raise ValueError(f"An output path for {relative_path} resolves outside its dataset directory.")
        original_path.parent.mkdir(parents=True, exist_ok=True)
        edge_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            edges = cv2.Canny(np.asarray(image.convert("L")), 100, 200)
        Image.fromarray(edges).save(edge_path)
        shutil.copy2(image_path, original_path)
        print(f"Processed {relative_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create matching originals and Canny conditioning images.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir_original", type=Path)
    parser.add_argument("output_dir_edges", type=Path)
    arguments = parser.parse_args()
    try:
        generate_canny_edge_dataset(arguments.input_dir, arguments.output_dir_original, arguments.output_dir_edges)
    except ValueError as exc:
        parser.error(str(exc))
