import json
import logging
import os
from pathlib import Path

from simpletuner.helpers.configuration.cli_utils import mapping_to_cli_args
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger("SimpleTuner")

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


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
        env_path = Path(env).expanduser()
        if env_path.suffix:
            candidates.append(env_path)
            if not env_path.is_absolute():
                candidates.append(Path.cwd() / env_path)
        else:
            candidates.append(env_path / "config.json")
            first_segment = env_path.parts[0] if env_path.parts else ""
            if first_segment != "config":
                candidates.append(Path("config") / env_path / "config.json")

            package_root = Path(__file__).resolve().parents[2]
            candidates.append(package_root / env_path / "config.json")
            candidates.append(package_root / env_path / "config" / "config.json")
    else:
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

            def _transform(key: str, value: object) -> object:
                if isinstance(value, str):
                    return _resolve_example_path(value)
                return value

            return mapping_to_cli_args(config, transform=_transform)
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


def normalize_args(args_dict: dict[str, object]) -> list[str]:
    """Backward-compatible helper returning CLI args for the given mapping."""

    return mapping_to_cli_args(args_dict)
