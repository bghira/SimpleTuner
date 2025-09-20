import json
import logging
import os
from pathlib import Path

# Set up logging
from simpletuner.helpers.training.multi_process import _get_rank

logger = logging.getLogger("SimpleTuner")
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


def normalize_args(args_dict):
    """
    Normalize arguments, ensuring they have '--' at the start if necessary.

    :param args_dict: A dictionary of arguments that may or may not have '--' prefixes.
    :return: A normalized dictionary of arguments.
    """
    normalized = []
    for key, value in args_dict.items():
        # Add -- prefix if not present
        if (type(value) is bool and value) or value == "true":
            if not key.startswith("--"):
                normalized_key = f"--{key}"
            else:
                normalized_key = key
        elif type(value) is bool and not value or value == "false":
            logger.warning(f"Skipping false argument: {key}")
            continue
        else:
            if isinstance(value, str):
                value = _resolve_example_path(value)
            if not key.startswith("--"):
                normalized_key = f"--{key}={value}"
            else:
                normalized_key = f"{key}={value}"
        normalized.append(normalized_key)
    return normalized


def _candidate_json_paths(env: str | None) -> list[Path]:
    candidates: list[Path] = []

    override = os.environ.get("CONFIG_PATH")
    if override:
        override_path = Path(override)
        if override_path.is_dir():
            candidates.append(override_path / "config.json")
        else:
            if override_path.suffix:
                candidates.append(override_path)
            candidates.append(override_path.with_suffix(".json"))

    if env and env != "default":
        env_path = Path(env)
        candidates.append(Path("config") / env_path / "config.json")

        package_root = Path(__file__).resolve().parents[2]
        candidates.append(package_root / env_path / "config.json")
        candidates.append(package_root / env_path / "config" / "config.json")

    default_path = Path("config/config.json")
    candidates.append(default_path)

    seen = set()
    ordered_candidates = []
    for path in candidates:
        path = path.expanduser()
        if path in seen:
            continue
        seen.add(path)
        ordered_candidates.append(path)

    return ordered_candidates


def load_json_config():
    """
    Load configuration from a JSON file that directly specifies command-line arguments.

    :param json_path: The path to the JSON file.
    :return: A dictionary containing the configuration.
    """
    config_json_path = None
    env = os.environ.get(
        "SIMPLETUNER_ENVIRONMENT",
        os.environ.get("SIMPLETUNER_ENV", os.environ.get("ENV", None)),
    )

    for candidate in _candidate_json_paths(env):
        if candidate.is_file():
            config_json_path = candidate
            break

    if config_json_path is None:
        attempted = ", ".join(str(path) for path in _candidate_json_paths(env))
        raise ValueError(f"JSON configuration file not found. Paths tried: {attempted}")

    with open(config_json_path, "r") as file:
        try:
            config = json.load(file)
            logger.info(f"[CONFIG.JSON] Loaded configuration from {config_json_path}")
            return normalize_args(config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file {config_json_path}: {e}")


def _resolve_example_path(value: str) -> str:
    if not value.startswith("config/examples/"):
        return value

    try:
        relative_path = Path(value).relative_to("config")
    except ValueError:
        return value

    package_root = Path(__file__).resolve().parents[2]
    candidate = package_root / relative_path

    if candidate.exists():
        return str(candidate)

    return value
