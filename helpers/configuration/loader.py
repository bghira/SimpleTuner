import os
import logging
from helpers.configuration import toml_file, json_file, env_file, cmd_args
from helpers.training.state_tracker import StateTracker
import sys

logger = logging.getLogger("SimpleTuner")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

helpers = {
    "json": json_file.load_json_config,
    "toml": toml_file.load_toml_config,
    "env": env_file.load_env_config,
    "cmd": cmd_args.parse_cmdline_args,
}

default_config_paths = {
    "json": "config.json",
    "toml": "config.toml",
    "env": "config.env",
}


def attach_env_to_path_if_not_present(backend: str, env: str = None):
    backend_cfg_path = default_config_paths.get(backend)
    if env and env != "default":
        return f"config/{env}/{backend_cfg_path}"
    return f"config/{backend_cfg_path}"


def load_config(args: dict = None, exit_on_error: bool = False):
    # Check if help is requested; bypass configuration loading if true
    if "-h" in sys.argv or "--help" in sys.argv:
        return helpers["cmd"]()

    mapped_config = args
    if mapped_config is None or not mapped_config:
        config_backend = os.environ.get(
            "SIMPLETUNER_CONFIG_BACKEND",
            os.environ.get("CONFIG_BACKEND", os.environ.get("CONFIG_TYPE", "env")),
        ).lower()
        config_env = os.environ.get(
            "SIMPLETUNER_ENVIRONMENT",
            os.environ.get("SIMPLETUNER_ENV", os.environ.get("ENV", "default")),
        )
        config_backend_path = "config"
        if config_env and config_env != "default" and config_env is not None:
            config_backend_path = os.path.join("config", config_env)
        StateTracker.set_config_path(config_backend_path)
        logger.info("Using {} configuration backend.".format(config_backend))
        mapped_config = helpers[config_backend]()
        if config_backend == "cmd":
            return mapped_config

    # Other configs need to be passed through parse_cmdline_args to be made whole and have complete defaults and safety checks applied.
    configuration = helpers["cmd"](
        input_args=mapped_config, exit_on_error=exit_on_error
    )

    return configuration
