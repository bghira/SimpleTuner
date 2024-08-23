#!/usr/bin/env bash

# Pull the default config.
[ -f "config/config.env.example" ] && source config/config.env.example
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
    echo $NVJITLINK_PATH
fi

export PLATFORM
PLATFORM=$(uname -s)

if [ -z "${ACCELERATE_EXTRA_ARGS}" ]; then
    ACCELERATE_EXTRA_ARGS=""
fi

if [ -z "${TRAINING_NUM_PROCESSES}" ]; then
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

export PURE_BF16_ARGS=""
if ! [ -z "$PURE_BF16" ] && [[ "$PURE_BF16" == "true" ]]; then
    PURE_BF16_ARGS="--adam_bfloat16"
    if [[ "$MIXED_PRECISION" != "no" ]]; then
        MIXED_PRECISION="bf16"
    fi
fi

if [ -z "${TRAINING_SEED}" ]; then
    printf "TRAINING_SEED not set, defaulting to 0.\n"
    TRAINING_SEED=0
fi

if [ -z "${TRAINING_DYNAMO_BACKEND}" ]; then
    printf "TRAINING_DYNAMO_BACKEND not set, defaulting to 'no'.\n"
    TRAINING_DYNAMO_BACKEND=no
fi
# Check that the rest of the parameters are not blank:
if [ -z "${MODEL_NAME}" ]; then
    printf "MODEL_NAME not set, exiting.\n"
    exit 1
fi
export LYCORIS_CONFIG_ARG=""
if [ -n "$LYCORIS_CONFIG" ]; then
    export LYCORIS_CONFIG_ARG="--lycoris_config=${LYCORIS_CONFIG}"
fi
if [ -n "$LORA_TYPE" ]; then
    export LORA_TYPE_ARG="--lora_type=${LORA_TYPE}"
fi
if [ -n "$LORA_RANK" ]; then
    export LORA_RANK_ARG="--lora_rank=${LORA_RANK}"
fi
if [ -n "$BASE_MODEL_PRECISION" ]; then
    export BASE_MODEL_PRECISION_ARG="--base_model_precision=${BASE_MODEL_PRECISION}"
fi
if [ -z "${RESOLUTION}" ]; then
    printf "RESOLUTION not set, exiting.\n"
    exit 1
fi
if [ -z "${OUTPUT_DIR}" ]; then
    printf "OUTPUT_DIR not set, exiting.\n"
    exit 1
fi
if [ -z "${CHECKPOINTING_STEPS}" ]; then
    printf "CHECKPOINTING_STEPS not set, exiting.\n"
    exit 1
fi
if [ -z "${CHECKPOINTING_LIMIT}" ]; then
    printf "CHECKPOINTING_LIMIT not set, exiting.\n"
    exit 1
fi
if [ -z "${VALIDATION_STEPS}" ]; then
    printf "VALIDATION_STEPS not set, exiting.\n"
    exit 1
fi
if [ -z "${TRACKER_PROJECT_NAME}" ]; then
    printf "TRACKER_PROJECT_NAME not set, exiting.\n"
    exit 1
fi
if [ -z "${TRACKER_RUN_NAME}" ]; then
    printf "TRACKER_RUN_NAME not set, exiting.\n"
    exit 1
fi
if [ -z "${VALIDATION_PROMPT}" ]; then
    printf "VALIDATION_PROMPT not set, exiting.\n"
    exit 1
fi
if [ -z "${VALIDATION_GUIDANCE}" ]; then
    printf "VALIDATION_GUIDANCE not set, exiting.\n"
    exit 1
fi
if [ -z "${VALIDATION_GUIDANCE_REAL}" ]; then
    printf "VALIDATION_GUIDANCE_REAL not set, defaulting to 1.0.\n"
    export VALIDATION_GUIDANCE_REAL=1.0
fi
if [ -z "${VALIDATION_NO_CFG_UNTIL_TIMESTEP}" ]; then
    printf "VALIDATION_NO_CFG_UNTIL_TIMESTEP not set, exiting.\n"
    exit 1
fi
if [ -z "${VALIDATION_GUIDANCE_RESCALE}" ]; then
    printf "VALIDATION_GUIDANCE_RESCALE not set, exiting.\n"
    exit 1
fi
if [ -z "${VALIDATION_RESOLUTION}" ]; then
    printf "VALIDATION_RESOLUTION not set, defaulting to RESOLUTION.\n"
    export VALIDATION_RESOLUTION=$RESOLUTION
fi
if [ -z "${LEARNING_RATE}" ]; then
    printf "LEARNING_RATE not set, exiting.\n"
    exit 1
fi
if [ -z "${LR_SCHEDULE}" ]; then
    printf "LR_SCHEDULE not set, exiting.\n"
    exit 1
fi
export LR_END_ARG=""
if [ -n "${LR_END}" ]; then
    export LR_END_ARG="--lr_end=${LR_END}"
fi
if [ -z "${TRAIN_BATCH_SIZE}" ]; then
    printf "TRAIN_BATCH_SIZE not set, exiting.\n"
    exit 1
fi
if [ -z "${CAPTION_DROPOUT_PROBABILITY}" ]; then
    printf "CAPTION_DROPOUT_PROBABILITY not set, exiting.\n"
    exit 1
fi
if [ -z "${RESUME_CHECKPOINT}" ]; then
    printf "RESUME_CHECKPOINT not set, exiting.\n"
    exit 1
fi
if [ -z "${DEBUG_EXTRA_ARGS}" ]; then
    printf "DEBUG_EXTRA_ARGS not set, defaulting to empty.\n"
    DEBUG_EXTRA_ARGS=""
fi
if [ -z "${TRAINER_EXTRA_ARGS}" ]; then
    printf "TRAINER_EXTRA_ARGS not set, defaulting to empty.\n"
    TRAINER_EXTRA_ARGS=""
fi
export MINIMUM_RESOLUTION_ARG=""
if [ -z "$MINIMUM_RESOLUTION" ]; then
    printf "MINIMUM_RESOLUTION not set, you might have problems with upscaled images.\n"
else
    export MINIMUM_RESOLUTION_ARG="--minimum_image_size=${MINIMUM_RESOLUTION}"
fi

if [ -z "$RESOLUTION_TYPE" ]; then
    printf "RESOLUTION_TYPE not set, defaulting to pixel.\n"
    export RESOLUTION_TYPE="pixel"
fi
if [ -z "$LR_WARMUP_STEPS" ]; then
    printf "LR_WARMUP_STEPS not set, defaulting to 0.\n"
    export LR_WARMUP_STEPS=0
fi
if [ -z "$VALIDATION_NEGATIVE_PROMPT" ]; then
    printf "VALIDATION_NEGATIVE_PROMPT not set, defaulting to empty.\n"
    export VALIDATION_NEGATIVE_PROMPT=""
fi

if [ -z "$VALIDATION_SEED" ]; then
    printf "VALIDATION_SEED is unset, randomising validation seeds.\n"
    export VALIDATION_ARGS="--validation_randomize"
else
    export VALIDATION_ARGS="--validation_seed=${VALIDATION_SEED}"
fi
if [ -z "$VAE_BATCH_SIZE" ]; then
    printf "VAE_BATCH_SIZE not set, defaulting to 1. This may slow down VAE caching.\n"
    export VAE_BATCH_SIZE=1
fi
if [ -z "$METADATA_UPDATE_INTERVAL" ]; then
    printf "METADATA_UPDATE_INTERVAL not set, defaulting to 120 seconds.\n"
    export METADATA_UPDATE_INTERVAL=120
fi
if [ -n "$STABLE_DIFFUSION_3" ] && [[ "$STABLE_DIFFUSION_3" == "true" ]]; then
    export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --sd3"
    echo "Disabling Xformers for Stable Diffusion 3 (https://github.com/huggingface/diffusers/issues/8535)"
    export XFORMERS_ARG=""
fi
if [ -n "$PIXART_SIGMA" ] && [[ "$PIXART_SIGMA" == "true" ]]; then
    export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --pixart_sigma"
fi
if [ -n "$STABLE_DIFFUSION_LEGACY" ] && [[ "$STABLE_DIFFUSION_LEGACY" == "true" ]]; then
    export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --legacy"
fi
if [ -n "$KOLORS" ] && [[ "$KOLORS" == "true" ]]; then
    export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --kolors"
fi
if [ -n "$SMOLDIT" ] && [[ "$SMOLDIT" == "true" ]]; then
    export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --smoldit"
fi
if [ -n "$FLUX" ] && [[ "$FLUX" == "true" ]]; then
    export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --flux"
    # if --flux_guidance_value is in TRAINER_EXTRA_ARGS, we will not add it again.
    if [[ "${TRAINER_EXTRA_ARGS}" != *"--flux_guidance_value"* ]]; then
        export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --flux_guidance_value=${FLUX_GUIDANCE_VALUE}"
    fi
    if [[ "${TRAINER_EXTRA_ARGS}" != *"--flux_lora_target"* ]]; then
        export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --flux_lora_target=${FLUX_LORA_TARGET}"
    fi
fi


if [ -z "$MAX_WORKERS" ]; then
    printf "MAX_WORKERS not set, defaulting to 32.\n"
    export MAX_WORKERS=32
fi
if [ -z "$READ_BATCH_SIZE" ]; then
    printf "READ_BATCH_SIZE not set, defaulting to 25.\n"
    export READ_BATCH_SIZE=25
fi
if [ -z "$WRITE_BATCH_SIZE" ]; then
    printf "WRITE_BATCH_SIZE not set, defaulting to 64.\n"
    export WRITE_BATCH_SIZE=64
fi
if [ -z "$AWS_MAX_POOL_CONNECTIONS" ]; then
    printf "AWS_MAX_POOL_CONNECTIONS not set, defaulting to 128.\n"
    export AWS_MAX_POOL_CONNECTIONS=128
fi
if [ -z "$TORCH_NUM_THREADS" ]; then
    printf "TORCH_NUM_THREADS not set, defaulting to 8.\n"
    export TORCH_NUM_THREADS=8
fi
if [ -z "$IMAGE_PROCESSING_BATCH_SIZE" ]; then
    printf "IMAGE_PROCESSING_BATCH_SIZE not set, defaulting to 32.\n"
    export IMAGE_PROCESSING_BATCH_SIZE=32
fi

export EMA_ARGS=""
if [ -n "$USE_EMA" ] && [[ "$USE_EMA" == "true" ]]; then
    if [ -z "$EMA_DECAY" ]; then
        printf "EMA_DECAY not set, defaulting to 0.9999.\n"
        export EMA_DECAY=0.9999
    fi
    export EMA_ARGS="--use_ema --ema_decay=${EMA_DECAY}"
fi
export OPTIMIZER_ARG="--optimizer=${OPTIMIZER}"

export DELETE_ARGS=""
if ! [ -z "$DELETE_SMALL_IMAGES" ] && [ $DELETE_SMALL_IMAGES -eq 1 ]; then
    export DELETE_ARGS="${DELETE_ARGS} --delete_unwanted_images"
fi
if ! [ -z "$DELETE_ERRORED_IMAGES" ] && [ $DELETE_ERRORED_IMAGES -eq 1 ]; then
    export DELETE_ARGS="${DELETE_ARGS} --delete_problematic_images"
fi

if [ -z "$VALIDATION_NUM_INFERENCE_STEPS" ]; then
    printf "VALIDATION_NUM_INFERENCE_STEPS not set, defaulting to 15.\n"
    export VALIDATION_NUM_INFERENCE_STEPS=15
fi

if [ -z "$TRAINING_SCHEDULER_TIMESTEP_SPACING" ]; then
    printf "TRAINING_SCHEDULER_TIMESTEP_SPACING not set, defaulting to 'trailing'.\n"
    export TRAINING_SCHEDULER_TIMESTEP_SPACING='trailing'
fi
if [ -z "$INFERENCE_SCHEDULER_TIMESTEP_SPACING" ]; then
    printf "INFERENCE_SCHEDULER_TIMESTEP_SPACING not set, defaulting to 'trailing'.\n"
    export INFERENCE_SCHEDULER_TIMESTEP_SPACING='trailing'
fi

export XFORMERS_ARG="--enable_xformers_memory_efficient_attention"
if ! [ -z "$USE_XFORMERS" ] && [[ "$USE_XFORMERS" == "false" ]]; then
    export XFORMERS_ARG=""
fi
if [[ "$PLATFORM" == "Darwin" ]]; then
    export XFORMERS_ARG=""
    export MIXED_PRECISION="no"
    echo "Disabled Xformers on MacOS, as it is not yet supported."
    echo "Overridden MIXED_PRECISION to 'no' for MacOS, as autocast is not supported by MPS."
fi

if [ -z "$DATALOADER_CONFIG" ]; then
    printf "DATALOADER_CONFIG not set, cannot continue. See multidatabackend.json.example.\n"
    exit 1
fi
if ! [ -f "$DATALOADER_CONFIG" ]; then
    printf "DATALOADER_CONFIG file %s not found, cannot continue.\n" "${DATALOADER_CONFIG}"
    exit 1
fi
if [ -z "$MAX_NUM_STEPS" ] && [ -z "$NUM_EPOCHS" ]; then
    echo "Neither MAX_NUM_STEPS or NUM_EPOCHS were defined."
    exit 1
fi
if [ -z "$MAX_NUM_STEPS" ]; then
    export MAX_NUM_STEPS=0
fi
if [ -z "$NUM_EPOCHS" ]; then
    export NUM_EPOCHS=0
fi
if [ "$MAX_NUM_STEPS" -lt 1 ] && [ "$NUM_EPOCHS" -lt 1 ]; then
    echo "Both MAX_NUM_STEPS {$MAX_NUM_STEPS} and NUM_EPOCHS {$NUM_EPOCHS} cannot be zero."
    exit 1
fi
export SNR_GAMMA_ARG=""
if [ -n "$MIN_SNR_GAMMA" ]; then
    export SNR_GAMMA_ARG="--snr_gamma=${MIN_SNR_GAMMA}"
fi

export GRADIENT_ARG="--gradient_checkpointing"
if ! [ -z "$USE_GRADIENT_CHECKPOINTING" ] && [[ "$USE_GRADIENT_CHECKPOINTING" == "false" ]]; then
    export GRADIENT_ARG=""
fi

if [ -z "$GRADIENT_ACCUMULATION_STEPS" ]; then
    export GRADIENT_ACCUMULATION_STEPS=1
fi

export TF32_ARG=""
if [ -n "$ALLOW_TF32" ] && [[ "$ALLOW_TF32" == "true" ]]; then
    export TF32_ARG="--allow_tf32"
fi
if [[ "$PLATFORM" == "Darwin" ]]; then
    export TF32_ARG=""
fi

export DORA_ARGS=""
if [[ "$MODEL_TYPE" == "full" ]] && [[ "$USE_DORA" != "false" ]]; then
    echo "Cannot use DoRA with a full u-net training task. Disabling DoRA."
elif [[ "$MODEL_TYPE" == "lora" ]] && [[ "$USE_DORA" != "false" ]]; then
    echo "Enabling DoRA."
    DORA_ARGS="--use_dora"
fi

export BITFIT_ARGS=""
if [[ "$USE_BITFIT" == "true" ]]; then
    echo "Enabling BitFit."
    BITFIT_ARGS="--layer_freeze_strategy=bitfit"
fi

# if PUSH_TO_HUB is set, ~/.cache/huggingface/token needs to exist and have a valid token.
# they can use huggingface-cli login to generate the token.
if [ -n "$PUSH_TO_HUB" ] && [[ "$PUSH_TO_HUB" == "true" ]]; then
    export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --push_to_hub --hub_model_id=${HUB_MODEL_NAME}"
    if [ -n "$PUSH_CHECKPOINTS" ] && [[ "$PUSH_CHECKPOINTS" == "true" ]]; then
        export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --push_checkpoints_to_hub"
    fi
fi

export ASPECT_BUCKET_ROUNDING_ARGS=""
if [ -n "$ASPECT_BUCKET_ROUNDING" ]; then
    export ASPECT_BUCKET_ROUNDING_ARGS="--aspect_bucket_rounding=${ASPECT_BUCKET_ROUNDING}"
fi

export MAX_NUM_STEPS_ARGS=""
if [ -n "$MAX_NUM_STEPS" ] && [[ "$MAX_NUM_STEPS" != 0 ]]; then
    export MAX_NUM_STEPS_ARGS="--max_train_steps=${MAX_NUM_STEPS}"
fi

export CONTROLNET_ARGS=""
if [ -n "$CONTROLNET" ] && [[ "$CONTROLNET" == "true" ]]; then
    export CONTROLNET_ARGS="--controlnet"
fi


# Run the training script.
accelerate launch ${ACCELERATE_EXTRA_ARGS} --mixed_precision="${MIXED_PRECISION}" --num_processes="${TRAINING_NUM_PROCESSES}" --num_machines="${TRAINING_NUM_MACHINES}" --dynamo_backend="${TRAINING_DYNAMO_BACKEND}" train.py \
    --model_type="${MODEL_TYPE}" ${DORA_ARGS} --pretrained_model_name_or_path="${MODEL_NAME}" ${XFORMERS_ARG} ${GRADIENT_ARG} --set_grads_to_none --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --resume_from_checkpoint="${RESUME_CHECKPOINT}" ${DELETE_ARGS} ${SNR_GAMMA_ARG} --data_backend_config="${DATALOADER_CONFIG}" \
    --num_train_epochs=${NUM_EPOCHS} ${MAX_NUM_STEPS_ARGS} --metadata_update_interval=${METADATA_UPDATE_INTERVAL} \
    ${OPTIMIZER_ARG} --learning_rate="${LEARNING_RATE}" --lr_scheduler="${LR_SCHEDULE}" --seed "${TRAINING_SEED}" --lr_warmup_steps="${LR_WARMUP_STEPS}" \
    --output_dir="${OUTPUT_DIR}" ${BITFIT_ARGS} ${ASPECT_BUCKET_ROUNDING_ARGS} \
    --inference_scheduler_timestep_spacing="${INFERENCE_SCHEDULER_TIMESTEP_SPACING}" --training_scheduler_timestep_spacing="${TRAINING_SCHEDULER_TIMESTEP_SPACING}" \
    ${DEBUG_EXTRA_ARGS}	${TF32_ARG} --mixed_precision="${MIXED_PRECISION}" ${TRAINER_EXTRA_ARGS} \
    --train_batch="${TRAIN_BATCH_SIZE}" --max_workers=$MAX_WORKERS --read_batch_size=$READ_BATCH_SIZE --write_batch_size=$WRITE_BATCH_SIZE --caption_dropout_probability=${CAPTION_DROPOUT_PROBABILITY} \
    --torch_num_threads=${TORCH_NUM_THREADS} --image_processing_batch_size=${IMAGE_PROCESSING_BATCH_SIZE} --vae_batch_size=$VAE_BATCH_SIZE \
    --validation_prompt="${VALIDATION_PROMPT}" --num_validation_images=1 --validation_num_inference_steps="${VALIDATION_NUM_INFERENCE_STEPS}" ${VALIDATION_ARGS} \
    ${MINIMUM_RESOLUTION_ARG} --resolution="${RESOLUTION}" --validation_resolution="${VALIDATION_RESOLUTION}" \
    --resolution_type="${RESOLUTION_TYPE}" ${LYCORIS_CONFIG_ARG} ${LORA_TYPE_ARG} ${LORA_RANK_ARG} ${BASE_MODEL_PRECISION_ARG} ${LR_END_ARG} \
    --checkpointing_steps="${CHECKPOINTING_STEPS}" --checkpoints_total_limit="${CHECKPOINTING_LIMIT}" \
    --validation_steps="${VALIDATION_STEPS}" --tracker_run_name="${TRACKER_RUN_NAME}" --tracker_project_name="${TRACKER_PROJECT_NAME}" \
    --validation_guidance="${VALIDATION_GUIDANCE}" --validation_guidance_real="${VALIDATION_GUIDANCE_REAL}" --validation_guidance_rescale="${VALIDATION_GUIDANCE_RESCALE}" --validation_negative_prompt="${VALIDATION_NEGATIVE_PROMPT}" ${EMA_ARGS} ${CONTROLNET_ARGS}

exit 0
