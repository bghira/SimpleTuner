import json

env_to_args_map = {
    "RESUME_CHECKPOINT": "--resume_from_checkpoint",
    "DATALOADER_CONFIG": "--data_backend_config",
    "ASPECT_BUCKET_ROUNDING": "--aspect_bucket_rounding",
    "TRAINING_SEED": "--seed",
    "ACCELERATE_CONFIG_PATH": "--accelerate_config",
    "ACCELERATE_EXTRA_ARGS": "--accelerate_extra_args",
    "USE_EMA": "--use_ema",
    "USE_XFORMERS": "--enable_xformers_memory_efficient_attention",
    "MINIMUM_RESOLUTION": "--minimum_image_size",
    "OUTPUT_DIR": "--output_dir",
    "USE_DORA": "--use_dora",
    "USE_BITFIT": "--layer_freeze_strategy=bitfit",
    "LORA_TYPE": "--lora_type",
    "LYCORIS_CONFIG": "--lycoris_config",
    "PUSH_TO_HUB": "--push_to_hub",
    "PUSH_TO_HUB_BACKGROUND": "--push_to_hub_background",
    "PUSH_CHECKPOINTS": "--push_checkpoints_to_hub",
    "PUBLISHING_CONFIG": "--publishing_config",
    "MAX_NUM_STEPS": "--max_train_steps",
    "NUM_EPOCHS": "--num_train_epochs",
    "CHECKPOINTING_STEPS": "--checkpoint_step_interval",
    "CHECKPOINT_STEP_INTERVAL": "--checkpoint_step_interval",
    "CHECKPOINT_EPOCH_INTERVAL": "--checkpoint_epoch_interval",
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
    "ENABLE_CHUNKED_FEED_FORWARD": "--enable_chunked_feed_forward",
    "FEED_FORWARD_CHUNK_SIZE": "--feed_forward_chunk_size",
    "CAPTION_DROPOUT_PROBABILITY": "--caption_dropout_probability",
    "RESOLUTION_TYPE": "--resolution_type",
    "RESOLUTION": "--resolution",
    "VALIDATION_SEED": "--validation_seed",
    "VALIDATION_STEPS": "--validation_step_interval",
    "VALIDATION_STEP_INTERVAL": "--validation_step_interval",
    "VALIDATION_EPOCH_INTERVAL": "--validation_epoch_interval",
    "VALIDATION_RESOLUTION": "--validation_resolution",
    "VALIDATION_GUIDANCE": "--validation_guidance",
    "VALIDATION_METHOD": "--validation_method",
    "VALIDATION_EXTERNAL_SCRIPT": "--validation_external_script",
    "VALIDATION_EXTERNAL_BACKGROUND": "--validation_external_background",
    "POST_CHECKPOINT_SCRIPT": "--post_checkpoint_script",
    "POST_UPLOAD_SCRIPT": "--post_upload_script",
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
    "QUANTIZE_VIA": "--quantize_via",
    "QUANTIZATION_CONFIG": "--quantization_config",
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
    config_file_contents = {}
    if not os.path.isfile(file_path):
        return config_file_contents

    logger.info(f"[CONFIG.ENV] Loading environment variables from {file_path}")

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith("#") or line == "":
                continue

            if line.startswith("export "):
                line = line[7:]

            if "+=" in line:
                key, value = line.split("+=", 1)
                key, value = (
                    key.strip(),
                    value.strip('"').strip("'").strip().split(),
                )
                if key in config_file_contents:
                    config_file_contents[key].extend(value)
                else:
                    config_file_contents[key] = value
            else:
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]

                    config_file_contents[key] = value.split() if value else []
                else:
                    logger.warning(f"[CONFIG.ENV] Skipping malformed line {line_num} in {file_path}: {line}")

    return config_file_contents


def load_env():
    env = os.environ.get(
        "SIMPLETUNER_ENVIRONMENT",
        os.environ.get("SIMPLETUNER_ENV", os.environ.get("ENV", None)),
    )

    search_paths = []

    search_paths.append("config.env")

    search_paths.append("config/config.env")

    if env and env != "default":
        search_paths.append(f"config/{env}/config.env")

    merged_config = {}
    loaded_files = []

    for config_path in search_paths:
        file_config = parse_env_file(config_path)
        if file_config:
            loaded_files.append(config_path)
            for key, value in file_config.items():
                if "+=" in key:
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
    mapped_args = []
    ignored_accelerate_kwargs = [
        "--dynamo_backend",
    ]
    for env_var, arg_name in env_to_args_map.items():
        if arg_name in ignored_accelerate_kwargs:
            continue
        value = os.environ.get(env_var, None)
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
            if value.lower() in ["true", "false"]:
                if value.lower() == "true":
                    mapped_args.append(f"{arg_name}")
            elif is_numeric:
                mapped_args.append(f"{arg_name}={value}")
            else:
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
