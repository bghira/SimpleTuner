import logging
import os
import sys

from simpletuner.helpers.configuration import cmd_args, env_file, json_file, toml_file
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger("SimpleTuner")
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")

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


def auto_detect_config_backend(config_env: str = "default"):
    """Auto-detect configuration backend based on file existence, defaulting to JSON."""
    config_base_path = "config"
    if config_env and config_env != "default":
        config_base_path = os.path.join("config", config_env)

    # Check for config files in priority order: json, toml, env
    for backend, filename in [
        ("json", "config.json"),
        ("toml", "config.toml"),
        ("env", "config.env"),
    ]:
        config_path = os.path.join(config_base_path, filename)
        if os.path.exists(config_path):
            return backend

    # Default to json if no config files found (will be created or cause error later)
    return "json"


def load_env_variables():
    """
    Load environment variables from config.env files, mimicking train.sh behavior.
    This should happen early, before any other configuration loading.
    """
    # Source base config.env if it exists (like train.sh line 10)
    base_env_path = "config/config.env"
    if os.path.exists(base_env_path):
        env_config = env_file.parse_env_file(base_env_path)
        for key, value in env_config.items():
            if isinstance(value, list):
                value = " ".join(value)
            if value and isinstance(value, str):
                os.environ[key] = value

    # Source environment-specific config.env if ENV is set (like train.sh line 88)
    env = os.environ.get("ENV", os.environ.get("SIMPLETUNER_ENV", "default"))
    if env and env != "default":
        env_specific_path = f"config/{env}/config.env"
        if os.path.exists(env_specific_path):
            env_config = env_file.parse_env_file(env_specific_path)
            for key, value in env_config.items():
                if isinstance(value, list):
                    value = " ".join(value)
                if value and isinstance(value, str):
                    os.environ[key] = value


def load_config(args: dict = None, exit_on_error: bool = False):
    # Load environment variables from config.env files first (like train.sh does)
    load_env_variables()

    # Check if help is requested; bypass configuration loading if true
    if "-h" in sys.argv or "--help" in sys.argv:
        return helpers["cmd"]()

    if args is not None and hasattr(args, "__dict__"):
        args = args.__dict__.copy()
        args.pop("should_abort", None)

    mapped_config = args
    if mapped_config is None or not mapped_config:
        config_env = os.environ.get(
            "SIMPLETUNER_ENVIRONMENT",
            os.environ.get("SIMPLETUNER_ENV", os.environ.get("ENV", "default")),
        )

        # Auto-detect config backend unless explicitly set
        config_backend = os.environ.get(
            "SIMPLETUNER_CONFIG_BACKEND",
            os.environ.get("CONFIG_BACKEND", os.environ.get("CONFIG_TYPE")),
        )

        if not config_backend:
            config_backend = auto_detect_config_backend(config_env)
        else:
            config_backend = config_backend.lower()

        config_backend_path = "config"
        if config_env and config_env != "default" and config_env is not None:
            config_backend_path = os.path.join("config", config_env)
        StateTracker.set_config_path(config_backend_path)
        logger.info("Using {} configuration backend.".format(config_backend))
        mapped_config = helpers[config_backend]()
        if config_backend == "cmd":
            return mapped_config

    if isinstance(mapped_config, dict):
        # Convert dict to list of command-line style arguments
        list_arguments = []
        for arg_name, value in mapped_config.items():
            if isinstance(value, str) and value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            try:
                float(value)
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False
            if value is not None and value != "":
                if isinstance(value, str) and value.lower() in ["true", "false"]:
                    if value.lower() == "true":
                        list_arguments.append(f"{arg_name}")
                    else:
                        continue
                elif value is False:
                    continue
                elif value is True:
                    list_arguments.append(f"{arg_name}")
                elif is_numeric:
                    list_arguments.append(f"{arg_name}={value}")
                else:
                    list_arguments.append(f"{arg_name}={value}")
        mapped_config = list_arguments

    configuration = helpers["cmd"](input_args=mapped_config, exit_on_error=exit_on_error)

    return configuration
