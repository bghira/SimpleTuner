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


def main():
    model_dir = Path("simpletuner/helpers/models")
    if not model_dir.exists():
        print(f"Error: {model_dir} not found.")
        return

    metadata = {}

    # We need to add the project root to sys.path to import simpletuner
    sys.path.insert(0, os.getcwd())

    for subdir in model_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("__"):
            model_file = subdir / "model.py"
            if model_file.exists():
                module_path = f"simpletuner.helpers.models.{subdir.name}.model"
                try:
                    # Temporary import to extract NAME
                    module = importlib.import_module(module_path)
                    # Find the class that inherits from VideoModelFoundation or similar
                    # and was registered in ModelRegistry.
                    # Since we can't easily peek into ModelRegistry without triggering more imports,
                    # we'll look for classes in the module.
                    for name, obj in module.__dict__.items():
                        if isinstance(obj, type) and hasattr(obj, "NAME"):
                            # This looks like our model class
                            metadata[subdir.name] = {
                                "class_name": name,
                                "module_path": module_path,
                                "name": getattr(obj, "NAME", subdir.name.replace("_", " ").title()),
                                "flavour_choices": (
                                    list(obj.get_flavour_choices()) if hasattr(obj, "get_flavour_choices") else []
                                ),
                            }
                            # Assume one model per directory
                            break
                except Exception as exc:
                    print(f"Failed to extract metadata for {subdir.name}: {exc}")

    output_path = model_dir / "model_metadata.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Wrote model metadata to {output_path}")


if __name__ == "__main__":
    main()
