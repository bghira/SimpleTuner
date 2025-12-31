"""
Examples command for SimpleTuner CLI.

Handles listing and copying example configurations.
"""

import json
import shutil
from pathlib import Path
from typing import List

from .train import get_examples_dir, list_examples


def find_referenced_files(config_path: Path) -> List[str]:
    """Find files referenced in config.json that should be copied."""
    referenced_files = []

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        fields_to_check = [
            "data_backend_config",
            "validation_prompt_library",
            "controlnet_config",
            "reference_config",
            "lycoris_config",
        ]

        for field in fields_to_check:
            if field in config:
                value = config[field]
                if isinstance(value, str):
                    if "examples/" in value and value.endswith(".json"):
                        filename = Path(value).name
                        referenced_files.append(filename)

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not parse config file {config_path}: {e}")

    return referenced_files


def update_config_paths(config_path: Path, referenced_files: List[str]) -> None:
    """Update config.json paths to point to local referenced files."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        fields_to_check = [
            "data_backend_config",
            "validation_prompt_library",
            "controlnet_config",
            "reference_config",
            "lycoris_config",
        ]

        updated = False
        for field in fields_to_check:
            if field in config:
                value = config[field]
                if isinstance(value, str) and "examples/" in value:
                    filename = Path(value).name
                    if filename in referenced_files:
                        config[field] = filename
                        updated = True
                        print(f"  Updated {field}: {value} -> {filename}")

        if updated:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"  Updated config.json with local file paths")

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not update config paths: {e}")


def copy_example(example_name: str, dest: str = None) -> bool:
    """Copy an example to destination directory."""
    examples_dir = get_examples_dir()
    example_path = examples_dir / example_name

    if not example_path.exists():
        print(f"Error: Example '{example_name}' not found.")
        print(f"Available examples: {', '.join(list_examples())}")
        return False

    if dest is None:
        dest = "."

    dest_path = Path(dest)

    try:
        if example_path.is_dir():
            dest_example = dest_path / example_name
            if dest_example.exists():
                print(f"Error: Destination '{dest_example}' already exists.")
                return False
            shutil.copytree(example_path, dest_example)
            print(f"Copied example directory '{example_name}' to '{dest_example}'")

            config_json = dest_example / "config.json"
            if config_json.exists():
                referenced_files = find_referenced_files(config_json)
                if referenced_files:
                    print(f"Found {len(referenced_files)} referenced file(s) to copy...")

                    for ref_file in referenced_files:
                        source_file = examples_dir / ref_file
                        if source_file.exists():
                            dest_file = dest_example / ref_file
                            shutil.copy2(source_file, dest_file)
                            print(f"  Copied referenced file: {ref_file}")
                        else:
                            print(f"  Warning: Referenced file not found: {ref_file}")

                    update_config_paths(config_json, referenced_files)
        else:
            dest_file = dest_path / example_path.name
            if dest_file.exists():
                print(f"Error: Destination '{dest_file}' already exists.")
                return False
            shutil.copy2(example_path, dest_file)
            print(f"Copied example file '{example_name}' to '{dest_file}'")

        return True
    except Exception as e:
        print(f"Error copying example: {e}")
        return False


def cmd_examples(args) -> int:
    """Handle examples command."""
    if args.action == "list":
        examples = list_examples()
        if not examples:
            print("No examples found.")
            return 1

        print("Available examples:")
        for example in examples:
            print(f"  {example}")
        return 0

    elif args.action == "copy":
        if not args.name:
            print("Error: Example name required for copy action.")
            return 1

        success = copy_example(args.name, args.dest)
        return 0 if success else 1

    else:
        print(f"Unknown examples action: {args.action}")
        return 1
