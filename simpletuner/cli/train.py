"""
Training command for SimpleTuner CLI.

Handles the train command for running local training with examples or environments.
"""

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from .common import (
    _candidate_config_paths,
    _extract_cli_override,
    _kill_process_group,
    _terminate_process_group,
    _validate_environment_config,
    find_config_file,
)


def get_examples_dir() -> Path:
    """Get the path to the examples directory."""
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


def setup_environment_from_example(example_name: str) -> dict:
    """Setup environment variables for training with example."""
    env = os.environ.copy()
    env["ENV"] = f"examples/{example_name}"

    examples_dir = get_examples_dir()
    example_path = examples_dir / example_name

    if example_path.is_dir():
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

    if "TQDM_NCOLS" not in env:
        env["TQDM_NCOLS"] = "125"
    if "TQDM_LEAVE" not in env:
        env["TQDM_LEAVE"] = "false"

    env["TOKENIZERS_PARALLELISM"] = "false"

    platform = os.uname().sysname
    env["PLATFORM"] = platform
    if platform == "Darwin":
        env["MIXED_PRECISION"] = "no"

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
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        config_path = Path(hf_home) / "accelerate" / "default_config.yaml"
        if config_path.exists():
            return str(config_path)

    home = Path.home()
    config_path = home / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"
    if config_path.exists():
        return str(config_path)

    return None


def run_training(example: Optional[str] = None, env: Optional[str] = None, extra_args: Optional[list] = None) -> int:
    """Run training with specified example or environment."""
    extra_args = list(extra_args or [])

    if env:
        os.environ["ENV"] = env

    from simpletuner.helpers.configuration.loader import load_env_variables

    load_env_variables()

    training_env = os.environ.copy()

    # Extract accelerate-related overrides from extra args
    overrides = {
        "accelerate_config": None,
        "accelerate_extra_args": None,
        "num_processes": None,
        "num_machines": None,
        "dynamo_backend": None,
    }
    cleaned_extra_args: List[str] = []

    idx = 0
    while idx < len(extra_args):
        arg = extra_args[idx]
        consumed_next = False

        def _extract_value(current_arg: str) -> Optional[str]:
            nonlocal consumed_next
            if "=" in current_arg:
                return current_arg.split("=", 1)[1]
            if idx + 1 < len(extra_args):
                candidate = extra_args[idx + 1]
                if not candidate.startswith("--"):
                    consumed_next = True
                    return candidate
            return None

        if arg.startswith("--accelerate_config"):
            value = _extract_value(arg)
            if value:
                overrides["accelerate_config"] = value
        elif arg.startswith("--accelerate_extra_args"):
            value = _extract_value(arg)
            if value:
                overrides["accelerate_extra_args"] = value
        elif arg.startswith("--num_processes"):
            value = _extract_value(arg)
            if value:
                overrides["num_processes"] = value
        elif arg.startswith("--num_machines"):
            value = _extract_value(arg)
            if value:
                overrides["num_machines"] = value
        elif arg.startswith("--dynamo_backend"):
            value = _extract_value(arg)
            if value:
                overrides["dynamo_backend"] = value
        else:
            cleaned_extra_args.append(arg)

        if consumed_next:
            idx += 1
        idx += 1

    extra_args = cleaned_extra_args

    config_backend_cli = _extract_cli_override(extra_args, ("config_backend", "config-backend"))
    config_backend_env = training_env.get(
        "SIMPLETUNER_CONFIG_BACKEND",
        training_env.get("CONFIG_BACKEND", training_env.get("CONFIG_TYPE")),
    )
    config_backend_override = config_backend_cli or config_backend_env

    config_path_cli = _extract_cli_override(extra_args, ("config_path", "config-path"))
    config_path_override = config_path_cli or training_env.get("CONFIG_PATH")

    if env:
        try:
            _validate_environment_config(env, config_backend_override, config_path_override)
        except FileNotFoundError as validation_error:
            print(f"Error: {validation_error}")
            return 1

    if overrides["accelerate_config"]:
        training_env["ACCELERATE_CONFIG_PATH"] = os.path.expanduser(str(overrides["accelerate_config"]))
    if overrides["accelerate_extra_args"]:
        training_env["ACCELERATE_EXTRA_ARGS"] = str(overrides["accelerate_extra_args"])
    if overrides["num_processes"]:
        training_env["TRAINING_NUM_PROCESSES"] = str(overrides["num_processes"])
    if overrides["num_machines"]:
        training_env["TRAINING_NUM_MACHINES"] = str(overrides["num_machines"])
    if overrides["dynamo_backend"]:
        training_env["TRAINING_DYNAMO_BACKEND"] = str(overrides["dynamo_backend"])

    if not example and not env:
        config_file = find_config_file()
        if not config_file:
            print("Error: No config file found in current directory or config/ subdirectory.")
            print("Expected: config.json, config.toml, or config.env")
            print("Or use: simpletuner train example=<example_name>")
            return 1

        print(f"Using config file: {config_file}")

        config_path = Path(config_file)
        if config_path.suffix == ".json":
            training_env["CONFIG_BACKEND"] = "json"
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

    import simpletuner

    simpletuner_dir = Path(simpletuner.__file__).parent
    train_py = simpletuner_dir / "train.py"

    if not train_py.exists():
        print(f"Error: train.py not found at {train_py}")
        return 1

    accelerate_config = None
    accelerate_env_path = training_env.get("ACCELERATE_CONFIG_PATH")
    if accelerate_env_path:
        candidate = Path(accelerate_env_path).expanduser()
        if candidate.exists():
            accelerate_config = str(candidate)
        else:
            print(f"Accelerate config override not found at {candidate}. Falling back to discovery.")
            training_env.pop("ACCELERATE_CONFIG_PATH", None)

    if accelerate_config is None:
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

    accelerate_extra_args = training_env.get("ACCELERATE_EXTRA_ARGS", "")
    if accelerate_extra_args:
        accel_extra = shlex.split(accelerate_extra_args)
        cmd = cmd[:-1] + accel_extra + cmd[-1:]

    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")

    process = None
    try:
        if os.name != "nt":
            process = subprocess.Popen(cmd, env=training_env, preexec_fn=os.setsid)
        else:
            process = subprocess.Popen(cmd, env=training_env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

        returncode = process.wait()
        return returncode

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        if process and process.poll() is None:
            print("Terminating training processes...")
            _terminate_process_group(process)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing training processes...")
                _kill_process_group(process)
                process.wait()
        return 130
    except Exception as e:
        print(f"Error running training: {e}")
        if process and process.poll() is None:
            _terminate_process_group(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
        return 1
    finally:
        if process and process.poll() is None:
            _terminate_process_group(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)


def cmd_train(args) -> int:
    """Handle train command."""
    example = getattr(args, "example", None)
    env = getattr(args, "env", None)
    extra_args = []

    if hasattr(args, "args") and args.args:
        for arg in args.args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key == "example" and not example:
                    example = value
                elif key == "env" and not env:
                    env = value
                else:
                    extra_args.append(f"--{key}={value}")
            else:
                if arg.startswith("--"):
                    extra_args.append(arg)
                else:
                    extra_args.append(f"--{arg}")

    return run_training(example=example, env=env, extra_args=extra_args)
