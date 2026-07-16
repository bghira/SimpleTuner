#!/usr/bin/env python3
import importlib
import json
import logging
import os
import sys
from pathlib import Path

# Suppress logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("simpletuner.helpers.models")
logger.setLevel(logging.ERROR)


def extract_metadata_from_module(module, module_path: str, family_name: str) -> dict | None:
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and hasattr(obj, "NAME") and getattr(obj, "__module__", None) == module.__name__:
            prediction_type = None
            if hasattr(obj, "PREDICTION_TYPE"):
                pt = getattr(obj, "PREDICTION_TYPE")
                # Handle both enum and direct value
                prediction_type = getattr(pt, "value", pt) if pt else None

            return {
                "class_name": name,
                "flavour_choices": list(obj.get_flavour_choices()) if hasattr(obj, "get_flavour_choices") else [],
                "module_path": module_path,
                "name": getattr(obj, "NAME", family_name.replace("_", " ").title()),
                "prediction_type": prediction_type,
            }
    return None


def main():
    model_dir = Path("simpletuner/helpers/models")
    if not model_dir.exists():
        print(f"Error: {model_dir} not found.")
        return

    metadata = {}

    # We need to add the project root to sys.path to import simpletuner
    sys.path.insert(0, os.getcwd())

    for subdir in sorted(model_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("__"):
            model_file = subdir / "model.py"
            if model_file.exists():
                module_path = f"simpletuner.helpers.models.{subdir.name}.model"
                try:
                    # Temporary import to extract NAME
                    module = importlib.import_module(module_path)
                    module_metadata = extract_metadata_from_module(module, module_path, subdir.name)
                    if module_metadata is not None:
                        metadata[subdir.name] = module_metadata
                except Exception as exc:
                    print(f"Failed to extract metadata for {subdir.name}: {exc}")

    output_path = model_dir / "model_metadata.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        f.write("\n")

    print(f"Wrote model metadata to {output_path}")


if __name__ == "__main__":
    main()
