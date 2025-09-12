#!/usr/bin/env bash

# Pull config from config.env
[ -f "config/config.env" ] && source config/config.env

# If the user has not provided VENV_PATH, we will assume $(pwd)/.venv
if [ -z "${VENV_PATH}" ]; then
    # what if we have VIRTUAL_ENV? use that instead
    if [ -n "${VIRTUAL_ENV}" ]; then
        export VENV_PATH="${VIRTUAL_ENV}"
    elif [ -d "$PWD/.venv" ]; then
        export VENV_PATH="$PWD/.venv"
    elif [ -d "$PWD/venv" ]; then
        export VENV_PATH="$PWD/venv"
    fi
fi

# If a venv hasn't already been activated, activate it now
if [[ -z "${VIRTUAL_ENV}" ]]; then
    source "${VENV_PATH}/bin/activate"
fi

if [ -z "${DISABLE_LD_OVERRIDE}" ]; then
    export NVJITLINK_PATH="$(find "${VENV_PATH}" -name nvjitlink -type d)/lib"
    # if it's not empty, we will add it to LD_LIBRARY_PATH at the front:
    if [ -n "${NVJITLINK_PATH}" ]; then
        export LD_LIBRARY_PATH="${NVJITLINK_PATH}:${LD_LIBRARY_PATH}"
    fi
fi

if [ -z "${TQDM_NCOLS}" ]; then
    export TQDM_NCOLS=125
fi
if [ -z "${TQDM_LEAVE}" ]; then
    export TQDM_LEAVE=false
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
    # Handle backwards compatibility: if ENV starts with "examples/", redirect to simpletuner/examples/
    if [[ "$ENV" == examples/* ]]; then
        export ENV_PATH="simpletuner/${ENV}/"
    else
        export ENV_PATH="${ENV}/"
    fi
    [ -f "config/$ENV_PATH/config.env" ] && source "config/$ENV_PATH/config.env"

fi

if [ -z "${CONFIG_BACKEND}" ]; then
    if [ -n "${CONFIG_TYPE}" ]; then
        export CONFIG_BACKEND="${CONFIG_TYPE}"
    fi
fi

if [ -z "${CONFIG_BACKEND}" ]; then
    export CONFIG_BACKEND="env"
    # Handle examples path differently - look directly in the examples directory
    if [[ "$ENV" == examples/* ]]; then
        export CONFIG_PATH="${ENV_PATH}config"
    else
        export CONFIG_PATH="config/${ENV_PATH}config"
    fi
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
        nvidia-smi > /dev/null 2>&1 && poetry install
        uname -s | grep -q Darwin && poetry install -C install/apple
        rocm-smi > /dev/null 2>&1 && poetry install -C install/rocm
    fi
fi
if [[ -z "${ACCELERATE_CONFIG_PATH}" ]]; then
    # Look for accelerate config in HF_HOME first, otherwise fallback to $HOME
    if [[ -f "${HF_HOME}/accelerate/default_config.yaml" ]]; then
        ACCELERATE_CONFIG_PATH="${HF_HOME}/accelerate/default_config.yaml"
    else
        ACCELERATE_CONFIG_PATH="${HOME}/.cache/huggingface/accelerate/default_config.yaml"
    fi
fi
# Run the training script.
if [ -f "${ACCELERATE_CONFIG_PATH}" ]; then
    echo "Using Accelerate config file: ${ACCELERATE_CONFIG_PATH}"
    accelerate launch --config_file="${ACCELERATE_CONFIG_PATH}" simpletuner/train.py
else
    echo "Accelerate config file not found: ${ACCELERATE_CONFIG_PATH}. Using values from config.env."
    accelerate launch ${ACCELERATE_EXTRA_ARGS} --mixed_precision="${MIXED_PRECISION}" --num_processes="${TRAINING_NUM_PROCESSES}" --num_machines="${TRAINING_NUM_MACHINES}" --dynamo_backend="${TRAINING_DYNAMO_BACKEND}" simpletuner/train.py

fi

exit 0
