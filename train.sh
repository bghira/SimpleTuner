#!/usr/bin/env bash

# Pull config from config.env
[ -f "config/config.env" ] && source config/config.env

# If the user has not provided VENV_PATH, we will assume $(pwd)/.venv
if [ -z "${VENV_PATH}" ]; then
    # what if we have VIRTUAL_ENV? use that instead
    if [ -n "${VIRTUAL_ENV}" ]; then
        export VENV_PATH="${VIRTUAL_ENV}"
    else
        export VENV_PATH="$(pwd)/.venv"
    fi
fi
if [ -z "${DISABLE_LD_OVERRIDE}" ]; then
    export NVJITLINK_PATH="$(find "${VENV_PATH}" -name nvjitlink -type d)/lib"
    # if it's not empty, we will add it to LD_LIBRARY_PATH at the front:
    if [ -n "${NVJITLINK_PATH}" ]; then
        export LD_LIBRARY_PATH="${NVJITLINK_PATH}:${LD_LIBRARY_PATH}"
    fi
fi

export TOKENIZERS_PARALLELISM=false
export PLATFORM
PLATFORM=$(uname -s)
if [[ "$PLATFORM" == "Darwin" ]]; then
    export MIXED_PRECISION="no"
fi

if [ -z "${ACCELERATE_EXTRA_ARGS}" ]; then
    ACCELERATE_EXTRA_ARGS=""
fi

if [ -z "${TRAINING_NUM_PROCESSES}" ]; then
    echo "Set custom env vars permanently in config/config.env:"
    printf "TRAINING_NUM_PROCESSES not set, defaulting to 1.\n"
    TRAINING_NUM_PROCESSES=1
fi

if [ -z "${TRAINING_NUM_MACHINES}" ]; then
    printf "TRAINING_NUM_MACHINES not set, defaulting to 1.\n"
    TRAINING_NUM_MACHINES=1
fi

if [ -z "${MIXED_PRECISION}" ]; then
    printf "MIXED_PRECISION not set, defaulting to bf16.\n"
    MIXED_PRECISION=bf16
fi

if [ -z "${TRAINING_DYNAMO_BACKEND}" ]; then
    printf "TRAINING_DYNAMO_BACKEND not set, defaulting to no.\n"
    TRAINING_DYNAMO_BACKEND="no"
fi

if [ -z "${ENV}" ]; then
    printf "ENV not set, defaulting to default.\n"
    export ENV="default"
fi
export ENV_PATH=""
if [[ "$ENV" != "default" ]]; then
    export ENV_PATH="${ENV}/"
fi

if [ -z "${CONFIG_BACKEND}" ]; then
    if [ -n "${CONFIG_TYPE}" ]; then
        export CONFIG_BACKEND="${CONFIG_TYPE}"
    fi
fi

if [ -z "${CONFIG_BACKEND}" ]; then
    export CONFIG_BACKEND="env"
    export CONFIG_PATH="config/${ENV_PATH}config"
    if [ -f "${CONFIG_PATH}.json" ]; then
        export CONFIG_BACKEND="json"
    elif [ -f "${CONFIG_PATH}.toml" ]; then
        export CONFIG_BACKEND="toml"
    elif [ -f "${CONFIG_PATH}.env" ]; then
        export CONFIG_BACKEND="env"
    fi
    echo "Using ${CONFIG_BACKEND} backend: ${CONFIG_PATH}.${CONFIG_BACKEND}"
fi

# Update dependencies
if [ -z "${DISABLE_UPDATES}" ]; then
    echo 'Updating dependencies. Set DISABLE_UPDATES to prevent this.'
    if [ -f "pyproject.toml" ] && [ -f "poetry.lock" ]; then
        nvidia-smi 2> /dev/null && poetry install
        uname -s | grep -q Darwin && poetry install -C install/apple
        rocm-smi 2> /dev/null && poetry install -C install/rocm
    fi
fi
# Run the training script.
if [[ -z "${ACCELERATE_CONFIG_PATH}" ]]; then
    ACCELERATE_CONFIG_PATH="${HOME}/enwik8-usw2/sd35lora/SimpleTuner/base_config.yaml"

# If config doesn't work, use this:
# ORIGINAL: 
# ACCELERATE_CONFIG_PATH="${HOME}/.cache/huggingface/accelerate/default_config.yaml"
# MOD:
# ACCELERATE_CONFIG_PATH="${HOME}/SimpleTuner/base_config.yaml"

fi
if [ -f "${ACCELERATE_CONFIG_PATH}" ]; then
    echo "Using Accelerate config file: ${ACCELERATE_CONFIG_PATH}"
    accelerate launch --config_file="${ACCELERATE_CONFIG_PATH}" train.py
else
    echo "Accelerate config file not found: ${ACCELERATE_CONFIG_PATH}. Using values from config.env."
    accelerate launch ${ACCELERATE_EXTRA_ARGS} --mixed_precision="${MIXED_PRECISION}" --num_processes="${TRAINING_NUM_PROCESSES}" --num_machines="${TRAINING_NUM_MACHINES}" --dynamo_backend="${TRAINING_DYNAMO_BACKEND}" train.py

fi

exit 0
