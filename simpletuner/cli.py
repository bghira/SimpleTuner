#!/usr/bin/env python3
"""
SimpleTuner CLI - Command-line interface for SimpleTuner
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def find_config_file() -> Optional[str]:
    """Find config file in current directory or config/ subdirectory."""
    # Check for config.json in current directory
    for config_name in ["config.json", "config.toml", "config.env"]:
        if os.path.exists(config_name):
            return config_name

    # Check for config files in config/ subdirectory
    config_dir = Path("config")
    if config_dir.exists():
        for config_name in ["config.json", "config.toml", "config.env"]:
            config_path = config_dir / config_name
            if config_path.exists():
                return str(config_path)

    return None


def get_examples_dir() -> Path:
    """Get the path to the examples directory."""
    # Find simpletuner package directory
    import simpletuner

    simpletuner_dir = Path(simpletuner.__file__).parent
    return simpletuner_dir / "examples"


def list_examples() -> List[str]:
    """List all available examples (directories only)."""
    examples_dir = get_examples_dir()
    if not examples_dir.exists():
        return []

    examples = []
    for item in examples_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            examples.append(item.name)

    return sorted(examples)


def find_referenced_files(config_path: Path) -> List[str]:
    """Find files referenced in config.json that should be copied."""
    referenced_files = []

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Look for common fields that reference files in examples directory
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
                    # Check if it references a file in examples directory
                    if "examples/" in value and value.endswith(".json"):
                        # Extract just the filename from paths like "config/examples/file.json"
                        filename = Path(value).name
                        referenced_files.append(filename)

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not parse config file {config_path}: {e}")

    return referenced_files


def copy_example(example_name: str, dest: Optional[str] = None) -> bool:
    """Copy an example to destination directory."""
    examples_dir = get_examples_dir()
    example_path = examples_dir / example_name

    if not example_path.exists():
        print(f"Error: Example '{example_name}' not found.")
        print(f"Available examples: {', '.join(list_examples())}")
        return False

    # Determine destination
    if dest is None:
        dest = "."

    dest_path = Path(dest)

    try:
        if example_path.is_dir():
            # Copy directory
            dest_example = dest_path / example_name
            if dest_example.exists():
                print(f"Error: Destination '{dest_example}' already exists.")
                return False
            shutil.copytree(example_path, dest_example)
            print(f"Copied example directory '{example_name}' to '{dest_example}'")

            # Check for referenced files in config.json
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

                    # Update config.json to use local paths
                    update_config_paths(config_json, referenced_files)

        else:
            # Copy file
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


def update_config_paths(config_path: Path, referenced_files: List[str]) -> None:
    """Update config.json paths to point to local referenced files."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Update paths for referenced files
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
                        # Update to use local path
                        config[field] = filename
                        updated = True
                        print(f"  Updated {field}: {value} -> {filename}")

        if updated:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"  Updated config.json with local file paths")

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not update config paths: {e}")


def setup_environment_from_example(example_name: str) -> dict:
    """Setup environment variables for training with example."""
    env = os.environ.copy()

    # Set ENV variable to point to the example
    env["ENV"] = f"examples/{example_name}"

    # Find the example config file
    examples_dir = get_examples_dir()
    example_path = examples_dir / example_name

    # Check if it's a directory with config files or a standalone config file
    if example_path.is_dir():
        config_path = example_path / "config"
        config_base = str(config_path)

        # Check for different config file types
        if (example_path / "config.json").exists():
            env["CONFIG_BACKEND"] = "json"
            env["CONFIG_PATH"] = str(example_path / "config")
        elif (example_path / "config.toml").exists():
            env["CONFIG_BACKEND"] = "toml"
            env["CONFIG_PATH"] = str(example_path / "config")
        elif (example_path / "config.env").exists():
            env["CONFIG_BACKEND"] = "env"
            env["CONFIG_PATH"] = str(example_path / "config")
        else:
            raise ValueError(f"No config file found in example {example_name}")
    elif example_path.is_file():
        # Standalone config file
        if example_path.suffix == ".json":
            env["CONFIG_BACKEND"] = "json"
            env["CONFIG_PATH"] = str(example_path.with_suffix(""))
        elif example_path.suffix == ".toml":
            env["CONFIG_BACKEND"] = "toml"
            env["CONFIG_PATH"] = str(example_path.with_suffix(""))
        else:
            raise ValueError(f"Unsupported config file type: {example_path.suffix}")
    else:
        raise ValueError(f"Example {example_name} not found")

    # Set other default environment variables (similar to train.sh)
    if "TQDM_NCOLS" not in env:
        env["TQDM_NCOLS"] = "125"
    if "TQDM_LEAVE" not in env:
        env["TQDM_LEAVE"] = "false"

    env["TOKENIZERS_PARALLELISM"] = "false"

    # Platform-specific settings
    platform = os.uname().sysname
    env["PLATFORM"] = platform
    if platform == "Darwin":
        env["MIXED_PRECISION"] = "no"

    # Training defaults
    if "TRAINING_NUM_PROCESSES" not in env:
        env["TRAINING_NUM_PROCESSES"] = "1"
    if "TRAINING_NUM_MACHINES" not in env:
        env["TRAINING_NUM_MACHINES"] = "1"
    if "MIXED_PRECISION" not in env:
        env["MIXED_PRECISION"] = "bf16"
    if "TRAINING_DYNAMO_BACKEND" not in env:
        env["TRAINING_DYNAMO_BACKEND"] = "no"

    return env


def find_accelerate_config() -> Optional[str]:
    """Find accelerate configuration file."""
    # Check HF_HOME first
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        config_path = Path(hf_home) / "accelerate" / "default_config.yaml"
        if config_path.exists():
            return str(config_path)

    # Fallback to default cache location
    home = Path.home()
    config_path = home / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"
    if config_path.exists():
        return str(config_path)

    return None


def run_training(example: Optional[str] = None, env: Optional[str] = None, extra_args: Optional[list] = None) -> int:
    """Run training with specified example or environment."""
    # Set ENV first so config.env loading works correctly
    if env:
        os.environ["ENV"] = env

    # Load environment variables from config.env files early (like train.sh does)
    from simpletuner.helpers.configuration.loader import load_env_variables

    load_env_variables()

    # Setup environment after loading config.env files
    training_env = os.environ.copy()

    # Check for config file if no example/env specified
    if not example and not env:
        config_file = find_config_file()
        if not config_file:
            print("Error: No config file found in current directory or config/ subdirectory.")
            print("Expected: config.json, config.toml, or config.env")
            print("Or use: simpletuner train example=<example_name>")
            return 1

        print(f"Using config file: {config_file}")

        # Set environment variables for local config
        config_path = Path(config_file)
        if config_path.suffix == ".json":
            training_env["CONFIG_BACKEND"] = "json"
            # If config is in current directory, use absolute path
            if config_path.parent == Path("."):
                training_env["CONFIG_PATH"] = str(Path.cwd() / config_path.with_suffix(""))
            else:
                training_env["CONFIG_PATH"] = str(config_path.with_suffix(""))
        elif config_path.suffix == ".toml":
            training_env["CONFIG_BACKEND"] = "toml"
            if config_path.parent == Path("."):
                training_env["CONFIG_PATH"] = str(Path.cwd() / config_path.with_suffix(""))
            else:
                training_env["CONFIG_PATH"] = str(config_path.with_suffix(""))
        elif config_path.suffix == ".env":
            training_env["CONFIG_BACKEND"] = "env"
            if config_path.parent == Path("."):
                training_env["CONFIG_PATH"] = str(Path.cwd() / config_path.with_suffix(""))
            else:
                training_env["CONFIG_PATH"] = str(config_path.with_suffix(""))

    if example:
        # Validate example exists
        available_examples = list_examples()
        if example not in available_examples:
            print(f"Error: Example '{example}' not found.")
            print(f"Available examples: {', '.join(available_examples)}")
            return 1

        training_env = setup_environment_from_example(example)
        print(f"Using example: {example}")
    elif env:
        training_env["ENV"] = env
        print(f"Using environment: {env}")

    # Find simpletuner train.py
    import simpletuner

    simpletuner_dir = Path(simpletuner.__file__).parent
    train_py = simpletuner_dir / "train.py"

    if not train_py.exists():
        print(f"Error: train.py not found at {train_py}")
        return 1

    # Setup accelerate command
    accelerate_config = find_accelerate_config()

    if accelerate_config:
        print(f"Using Accelerate config file: {accelerate_config}")
        cmd = [
            "accelerate",
            "launch",
            f"--config_file={accelerate_config}",
            str(train_py),
        ]
    else:
        print("Accelerate config file not found. Using environment variables.")
        cmd = [
            "accelerate",
            "launch",
            f"--mixed_precision={training_env.get('MIXED_PRECISION', 'bf16')}",
            f"--num_processes={training_env.get('TRAINING_NUM_PROCESSES', '1')}",
            f"--num_machines={training_env.get('TRAINING_NUM_MACHINES', '1')}",
            f"--dynamo_backend={training_env.get('TRAINING_DYNAMO_BACKEND', 'no')}",
            str(train_py),
        ]

    # Add any extra accelerate args
    accelerate_extra_args = training_env.get("ACCELERATE_EXTRA_ARGS", "")
    if accelerate_extra_args:
        # Insert extra args before the train.py script
        accel_extra = accelerate_extra_args.split()
        cmd = cmd[:-1] + accel_extra + cmd[-1:]

    # Add any extra train.py arguments after the script path
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")

    # Run the training
    try:
        result = subprocess.run(cmd, env=training_env)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return 130
    except Exception as e:
        print(f"Error running training: {e}")
        return 1


def cmd_train(args) -> int:
    """Handle train command."""
    example = getattr(args, "example", None)
    env = getattr(args, "env", None)
    extra_args = []

    # Parse key=value arguments
    if hasattr(args, "args") and args.args:
        for arg in args.args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key == "example" and not example:
                    example = value
                elif key == "env" and not env:
                    env = value
                else:
                    # Pass through unknown arguments to train.py
                    extra_args.append(f"--{key}={value}")
            else:
                # Pass through flag-style arguments
                if arg.startswith("--"):
                    extra_args.append(arg)
                else:
                    extra_args.append(f"--{arg}")

    return run_training(example=example, env=env, extra_args=extra_args)


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


def cmd_configure(args) -> int:
    """Handle configure command."""
    output_file = getattr(args, "output_file", "config.json")

    # Import and run the configure module
    try:
        # Set up sys.argv for the configure script
        import sys

        from simpletuner.configure import main as configure_main

        original_argv = sys.argv.copy()

        # Only pass the output file if it exists (for editing existing configs)
        # Otherwise, start fresh and let the user save to the output file
        if Path(output_file).exists():
            sys.argv = ["configure.py", output_file]
            print(f"Loading existing configuration from: {output_file}")
        else:
            sys.argv = ["configure.py"]
            print(f"Creating new configuration. Will save to: {output_file}")

        try:
            configure_main()
            return 0
        except KeyboardInterrupt:
            print("\nConfiguration cancelled by user.")
            return 130
        except Exception as e:
            print(f"Error running configuration wizard: {e}")
            return 1
        finally:
            sys.argv = original_argv

    except ImportError as e:
        print(f"Error importing configuration module: {e}")
        return 1


def cmd_server(args) -> int:
    """Handle server command."""
    host = getattr(args, "host", "0.0.0.0")
    port = getattr(args, "port", None)  # Will be determined by mode
    reload = getattr(args, "reload", False)
    mode = getattr(args, "mode", "trainer")

    # Determine port based on mode if not specified
    if port is None:
        if mode == "trainer":
            port = 8001
        elif mode == "callback":
            port = 8002
        else:  # unified
            port = 8001

    print(f"Starting SimpleTuner {mode} server on {host}:{port}")

    try:
        import uvicorn

        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        # Map mode string to enum
        server_mode = {"trainer": ServerMode.TRAINER, "callback": ServerMode.CALLBACK, "unified": ServerMode.UNIFIED}.get(
            mode, ServerMode.TRAINER
        )

        # Create app with specified mode
        app = create_app(mode=server_mode)

        # Create necessary directories
        os.makedirs("static/css", exist_ok=True)
        os.makedirs("static/js", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        os.makedirs("configs", exist_ok=True)

        # Run the server
        uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")
        return 0
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        return 130
    except ImportError as e:
        print(f"Error importing server dependencies: {e}")
        print("Make sure FastAPI and uvicorn are installed.")
        return 1
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1


def get_version() -> str:
    """Get SimpleTuner version."""
    try:
        import simpletuner

        return getattr(simpletuner, "__version__", "unknown")
    except:
        return "unknown"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="simpletuner",
        description="SimpleTuner - Fine-tune diffusion models with ease",
    )
    parser.add_argument("--version", "-v", action="version", version=f"SimpleTuner {get_version()}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Run training",
        description="Run training with automatic config detection or examples",
        epilog="""
Examples:
  simpletuner train                           # Use config.json in current directory
  simpletuner train --example sd3.peft-lora  # Use example configuration
  simpletuner train example=sd3.peft-lora    # Alternative syntax for examples
  simpletuner train --env custom-path        # Use custom environment path
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_group = train_parser.add_mutually_exclusive_group()
    train_group.add_argument("--example", "-e", help="Use example configuration (e.g., sd3.peft-lora)")
    train_group.add_argument("--env", help="Use custom environment path")
    # Add support for positional arguments like example=value
    train_parser.add_argument(
        "args",
        nargs="*",
        help="Additional arguments in key=value format (e.g., example=sd3.peft-lora)",
    )
    train_parser.set_defaults(func=cmd_train)

    # Examples command
    examples_parser = subparsers.add_parser("examples", help="Manage examples")
    examples_subparsers = examples_parser.add_subparsers(dest="action", help="Examples actions")

    # examples list
    list_parser = examples_subparsers.add_parser("list", help="List available examples")

    # examples copy
    copy_parser = examples_subparsers.add_parser("copy", help="Copy example to local directory")
    copy_parser.add_argument("name", help="Example name to copy")
    copy_parser.add_argument("dest", nargs="?", help="Destination directory (default: current)")

    examples_parser.set_defaults(func=cmd_examples)

    # Configure command
    configure_parser = subparsers.add_parser(
        "configure",
        help="Interactive configuration wizard",
        description="Run the interactive configuration wizard to create training configs",
    )
    configure_parser.add_argument(
        "output_file",
        nargs="?",
        default="config.json",
        help="Output configuration file (default: config.json)",
    )
    configure_parser.set_defaults(func=cmd_configure)

    # Server command
    server_parser = subparsers.add_parser(
        "server",
        help="Start SimpleTuner web server",
        description="Start the SimpleTuner web server for training management",
    )
    server_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the server to (default: 8001 for trainer, 8002 for callback)",
    )
    server_parser.add_argument(
        "--mode",
        choices=["trainer", "callback", "unified"],
        default="trainer",
        help="Server mode: trainer (8001), callback (8002), or unified (both in single process)",
    )
    server_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    server_parser.set_defaults(func=cmd_server)

    # Parse args and run command
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
