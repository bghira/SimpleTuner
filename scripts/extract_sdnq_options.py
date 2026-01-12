#!/usr/bin/env python3
import json
import logging
import os
import sys

# Suppress logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("sdnq")
logger.setLevel(logging.ERROR)


def main():
    try:
        import sdnq.optim
    except ImportError:
        print("sdnq not found.")
        return

    # Map of simpletuner optimizer names to sdnq class names
    # This mirrors the imports in simpletuner/helpers/training/optimizer_param.py
    # from sdnq.optim import CAME as SDNQCAME
    # from sdnq.optim import Adafactor as SDNQAdafactor
    # from sdnq.optim import AdamW as SDNQAdamW
    # from sdnq.optim import Lion as SDNQLion
    # from sdnq.optim import Muon as SDNQMuon

    available_optimizers = []

    # Check for presence of classes in sdnq.optim
    mapping = {"CAME": "CAME", "Adafactor": "Adafactor", "AdamW": "AdamW", "Lion": "Lion", "Muon": "Muon"}

    found_classes = []
    for sdnq_name, local_name in mapping.items():
        if hasattr(sdnq.optim, sdnq_name):
            found_classes.append(sdnq_name)

    output_path = os.path.join(os.path.dirname(__file__), "../simpletuner/helpers/training/sdnq_options.json")
    output_path = os.path.abspath(output_path)

    with open(output_path, "w") as f:
        json.dump({"available_classes": found_classes}, f, indent=4)

    print(f"Wrote sdnq options to {output_path}")


if __name__ == "__main__":
    main()
