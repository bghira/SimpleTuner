import json

env_to_args_map = {
    "RESUME_CHECKPOINT": "--resume_from_checkpoint",
    "DATALOADER_CONFIG": "--data_backend_config",
    "ASPECT_BUCKET_ROUNDING": "--aspect_bucket_rounding",
    "TRAINING_SEED": "--seed",
    "USE_EMA": "--use_ema",
    "USE_XFORMERS": "--enable_xformers_memory_efficient_attention",
    "MINIMUM_RESOLUTION": "--minimum_image_size",
    "OUTPUT_DIR": "--output_dir",
    "USE_DORA": "--use_dora",
    "USE_BITFIT": "--layer_freeze_strategy=bitfit",
    "LORA_TYPE": "--lora_type",
    "LYCORIS_CONFIG": "--lycoris_config",
    "PUSH_TO_HUB": "--push_to_hub",
    "PUSH_CHECKPOINTS": "--push_checkpoints_to_hub",
    "MAX_NUM_STEPS": "--max_train_steps",
    "NUM_EPOCHS": "--num_train_epochs",
    "CHECKPOINTING_STEPS": "--checkpointing_steps",
    "CHECKPOINTING_LIMIT": "--checkpoints_total_limit",
    "HUB_MODEL_NAME": "--hub_model_id",
    "MODEL_CARD_SAFE_FOR_WORK": "--model_card_safe_for_work",
    "TRACKER_PROJECT_NAME": "--tracker_project_name",
    "TRACKER_RUN_NAME": "--tracker_run_name",
    "MODEL_TYPE": "--model_type",
    "MODEL_NAME": "--pretrained_model_name_or_path",
    "MODEL_FAMILY": "--model_family",
    "TRAIN_BATCH_SIZE": "--train_batch_size",
    "USE_GRADIENT_CHECKPOINTING": "--gradient_checkpointing",
    "CAPTION_DROPOUT_PROBABILITY": "--caption_dropout_probability",
    "RESOLUTION_TYPE": "--resolution_type",
    "RESOLUTION": "--resolution",
    "VALIDATION_SEED": "--validation_seed",
    "VALIDATION_STEPS": "--validation_steps",
    "VALIDATION_RESOLUTION": "--validation_resolution",
    "VALIDATION_GUIDANCE": "--validation_guidance",
    "VALIDATION_GUIDANCE_RESCALE": "--validation_guidance_rescale",
    "VALIDATION_NUM_INFERENCE_STEPS": "--validation_num_inference_steps",
    "VALIDATION_PROMPT": "--validation_prompt",
    "ALLOW_TF32": "--allow_tf32",
    "MIXED_PRECISION": "--mixed_precision",
    "OPTIMIZER": "--optimizer",
    "LEARNING_RATE": "--learning_rate",
    "LR_SCHEDULE": "--lr_scheduler",
    "LR_WARMUP_STEPS": "--lr_warmup_steps",
    "BASE_MODEL_PRECISION": "--base_model_precision",
    "TRAINING_NUM_PROCESSES": "--num_processes",
    "TRAINING_NUM_MACHINES": "--num_machines",
    "VALIDATION_TORCH_COMPILE": "--validation_torch_compile",
    "TRAINER_DYNAMO_BACKEND": "--dynamo_backend",
    "VALIDATION_GUIDANCE_REAL": "--validation_guidance_real",
    "VALIDATION_NO_CFG_UNTIL_TIMESTEP": "--validation_no_cfg_until_timestep",
    "TRAINING_SCHEDULER_TIMESTEP_SPACING": "--training_scheduler_timestep_spacing",
    "INFERENCE_SCHEDULER_TIMESTEP_SPACING": "--inference_scheduler_timestep_spacing",
    "GRADIENT_ACCUMULATION_STEPS": "--gradient_accumulation_steps",
    "TRAINING_DYNAMO_BACKEND": "--dynamo_backend",
    "LR_END": "--lr_end",
    "FLUX_GUIDANCE_VALUE": "--flux_guidance_value",
    "FLUX_LORA_TARGET": "--flux_lora_target",
    "VALIDATION_NEGATIVE_PROMPT": "--validation_negative_prompt",
    "METADATA_UPDATE_INTERVAL": "--metadata_update_interval",
    "READ_BATCH_SIZE": "--read_batch_size",
    "WRITE_BATCH_SIZE": "--write_batch_size",
    "AWS_MAX_POOL_CONNECTIONS": "--aws_max_pool_connections",
    "TORCH_NUM_THREADS": "--torch_num_threads",
    "IMAGE_PROCESSING_BATCH_SIZE": "--image_processing_batch_size",
    "DISABLE_BENCHMARK": "--disable_benchmark",
}

import logging
import os
import subprocess

from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger("SimpleTuner")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


def parse_env_file(file_path):
    """
    Parse a single .env file and return its contents as a dictionary.
    """
    config_file_contents = {}
    if not os.path.isfile(file_path):
        return config_file_contents

    logger.info(f"[CONFIG.ENV] Loading environment variables from {file_path}")

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            # Skip comments and empty lines
            line = line.strip()
            if line.startswith("#") or line == "":
                continue

            # Remove 'export ' from the start (with space)
            if line.startswith("export "):
                line = line[7:]

            # Handle `+=` for appending values
            if "+=" in line:
                key, value = line.split("+=", 1)
                key, value = (
                    key.strip(),
                    value.strip('"').strip("'").strip().split(),
                )
                # Append each element to the existing key's list or create a new list
                if key in config_file_contents:
                    config_file_contents[key].extend(value)
                else:
                    config_file_contents[key] = value
            else:
                # Regular `=` assignment
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Handle quoted values
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]

                    # Split into words for consistency with old behavior
                    config_file_contents[key] = value.split() if value else []
                else:
                    logger.warning(f"[CONFIG.ENV] Skipping malformed line {line_num} in {file_path}: {line}")

    return config_file_contents


def load_env():
    """
    Load environment variables from .env files using topological search.
    Searches from specific env directory up to current directory, merging configs.
    """
    env = os.environ.get(
        "SIMPLETUNER_ENVIRONMENT",
        os.environ.get("SIMPLETUNER_ENV", os.environ.get("ENV", None)),
    )

    # Define search paths in order of precedence (later files override earlier ones)
    search_paths = []

    # 1. Root config.env (lowest precedence)
    search_paths.append("config.env")

    # 2. Base config directory
    search_paths.append("config/config.env")

    # 3. Environment-specific config (highest precedence)
    if env and env != "default":
        search_paths.append(f"config/{env}/config.env")

    # Load and merge configs
    merged_config = {}
    loaded_files = []

    for config_path in search_paths:
        file_config = parse_env_file(config_path)
        if file_config:  # Only merge if file exists and has content
            loaded_files.append(config_path)
            # Merge configs - later files override earlier ones
            for key, value in file_config.items():
                if "+=" in key:  # Handle append operations
                    base_key = key.replace("+=", "").strip()
                    if base_key in merged_config:
                        if isinstance(merged_config[base_key], list):
                            merged_config[base_key].extend(value)
                        else:
                            merged_config[base_key] = merged_config[base_key].split() + value
                    else:
                        merged_config[base_key] = value
                else:
                    merged_config[key] = value

    # Convert lists to single string values with spaces, if needed
    for key, value in merged_config.items():
        if isinstance(value, list):
            if value and "${" in str(value[0]):
                continue
            merged_config[key] = " ".join(value)

    if loaded_files:
        logger.info(f"[CONFIG.ENV] Successfully loaded config from: {', '.join(loaded_files)}")
    else:
        logger.warning("[CONFIG.ENV] No config.env files found in search paths")

    return merged_config


def load_env_config():
    """
    Map the environment variables to command-line arguments.
    NOTE: Environment variables from config.env files should be loaded separately
    by the configuration loader before this function is called.

    :return: List of command-line arguments.
    """
    # Get variables from actual environment (which should have been loaded from config.env files)
    mapped_args = []
    # Loop through the environment variable to argument mapping
    ignored_accelerate_kwargs = [
        "--num_processes",
        "--num_machines",
        "--dynamo_backend",
    ]
    for env_var, arg_name in env_to_args_map.items():
        if arg_name in ignored_accelerate_kwargs:
            # These are handled by accelerate via environment variables, not command-line args
            continue
        value = os.environ.get(env_var, None)
        # strip 's from the outside of value
        if value is not None and value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        if value is not None and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        try:
            float(value)
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False
        if value is not None:
            # Handle booleans by checking their string value
            if value.lower() in ["true", "false"]:
                if value.lower() == "true":
                    mapped_args.append(f"{arg_name}")
            elif is_numeric:
                # Handle numeric values
                mapped_args.append(f"{arg_name}={value}")
            else:
                # Add the argument and its value to the list
                mapped_args.append(f"{arg_name}={value}")
    extra_args = os.environ.get("TRAINER_EXTRA_ARGS", None)
    if extra_args:
        print(f"Extra args: {extra_args}")
        if type(extra_args) is list:
            for value in extra_args:
                if "${" in value:
                    continue
                mapped_args.extend(value.split())
        else:
            mapped_args.extend(extra_args.split())

    logger.info(f"Loaded environment variables: {json.dumps(mapped_args, indent=4)}")
    return mapped_args
