import os
import json
import logging

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
            if not key.startswith("--"):
                normalized_key = f"--{key}={value}"
            else:
                normalized_key = f"{key}={value}"
        normalized.append(normalized_key)
    return normalized


def load_json_config():
    """
    Load configuration from a JSON file that directly specifies command-line arguments.

    :param json_path: The path to the JSON file.
    :return: A dictionary containing the configuration.
    """
    config_json_path = "config/config.json"
    env = os.environ.get(
        "SIMPLETUNER_ENVIRONMENT",
        os.environ.get("SIMPLETUNER_ENV", os.environ.get("ENV", None)),
    )
    if env and env != "default":
        config_json_path = f"config/{env}/config.json"

    if not os.path.isfile(config_json_path):
        raise ValueError(f"JSON configuration file not found: {config_json_path}")

    with open(config_json_path, "r") as file:
        try:
            config = json.load(file)
            logger.info(f"[CONFIG.JSON] Loaded configuration from {config_json_path}")
            return normalize_args(config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file {config_json_path}: {e}")
