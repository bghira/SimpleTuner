import os
import toml
import logging

# Set up logging
from helpers.training.multi_process import _get_rank

logger = logging.getLogger("SimpleTuner")
from helpers.training.multi_process import should_log

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
        if type(value) is bool and value or value == "true":
            if not key.startswith("--"):
                normalized_key = f"--{key}"
            else:
                normalized_key = key
        elif type(value) is bool and not value or value == "false":
            logger.warning(f"Skipping false argument: {key}")
            continue
        else:
            print(f"Value: {value}, type: {type(value)}")
            if not key.startswith("--"):
                normalized_key = f"--{key}={value}"
            else:
                normalized_key = f"{key}={value}"
        normalized.append(normalized_key)
    return normalized


def load_toml_config():
    """
    Load configuration from a TOML file that directly specifies command-line arguments.

    :param toml_path: The path to the TOML file.
    :return: A dictionary containing the configuration.
    """
    config_toml_path = "config/config.toml"
    env = os.environ.get(
        "SIMPLETUNER_ENVIRONMENT",
        os.environ.get("SIMPLETUNER_ENV", os.environ.get("ENV", None)),
    )
    if env and env != "default":
        config_toml_path = f"config/{env}/config.toml"

    if not os.path.isfile(config_toml_path):
        raise ValueError(f"Can not find config file: {config_toml_path}")

    with open(config_toml_path, "r") as file:
        try:
            config = toml.load(file)
            logger.info(f"[CONFIG.TOML] Loaded configuration from {config_toml_path}")
            toml_config = config
        except toml.TomlDecodeError as e:
            logger.error(f"Failed to parse TOML file {config_toml_path}: {e}")
            toml_config = {}
    normalized_config = normalize_args(toml_config)
    logger.info(
        f"[CONFIG] Loaded and normalized TOML configuration: {normalized_config}"
    )

    return normalized_config
