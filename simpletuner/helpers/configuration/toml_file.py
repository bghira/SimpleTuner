import logging
import os

import toml

from simpletuner.helpers.configuration.cli_utils import mapping_to_cli_args
from simpletuner.helpers.training.multi_process import should_log

# Standard named logger so it inherits global handlers.
logger = logging.getLogger("SimpleTuner")
logger.setLevel(logging._nameToLevel.get(str(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")).upper(), logging.INFO))


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
        except toml.TomlDecodeError as e:
            logger.error(f"Failed to parse TOML file {config_toml_path}: {e}")
            config = {}

    cli_args = mapping_to_cli_args(config)
    logger.info(f"[CONFIG] Loaded and normalized TOML configuration: {cli_args}")

    return cli_args
