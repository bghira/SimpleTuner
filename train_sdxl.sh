#!/bin/bash
# Pull the default config.
source sdxl-env.sh.example
# Pull config from env.sh
[ -f "sdxl-env.sh" ] && source sdxl-env.sh

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
if [ -z "${NUM_EPOCHS}" ]; then
    printf "NUM_EPOCHS not set, exiting.\n"
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
if [ -z "$MINIMUM_RESOLUTION" ]; then
    printf "MINIMUM_RESOLUTION not set, defaulting to RESOLUTION.\n"
    export MINIMUM_RESOLUTION=$RESOLUTION
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
if [ -n "$STABLE_DIFFUSION_3" ]; then
    export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --sd3"
fi

export EMA_ARGS=""
if [ -n "$USE_EMA" ] && [[ "$USE_EMA" == "true" ]]; then
    if [ -z "$EMA_DECAY" ]; then
        printf "EMA_DECAY not set, defaulting to 0.9999.\n"
        export EMA_DECAY=0.9999
    fi
    export EMA_ARGS="--use_ema --ema_decay=${EMA_DECAY}"
fi
# OPTIMIZER can be "adamw", "adamw8bit", "adafactor", "dadaptation" and we'll use case-switch to detect and set --use_8bit_adam, --use_adafactor_optimizer, --use_dadapt_optimizer or nothing for plain adam.
export OPTIMIZER_ARG=""
case $OPTIMIZER in
    "adamw")
        export OPTIMIZER_ARG=""
        ;;
    "adamw8bit")
        export OPTIMIZER_ARG="--use_8bit_adam"
        ;;
    "adamw_bf16")
        export OPTIMIZER_ARG="--adam_bfloat16"
        ;;
    "adafactor")
        export OPTIMIZER_ARG="--use_adafactor_optimizer"
        ;;
    "dadaptation")
        export OPTIMIZER_ARG="--use_dadapt_optimizer"
        ;;
    "prodigy")
        export OPTIMIZER_ARG="--use_prodigy_optimizer"
        ;;
    *)
        echo "Unknown optimizer requested: $OPTIMIZER"
        exit 1
        ;;
esac

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
fi

if [ -z "$DATALOADER_CONFIG" ]; then
    printf "DATALOADER_CONFIG not set, cannot continue. See multidatabackend.json.example.\n"
    exit 1
fi
if ! [ -f "$DATALOADER_CONFIG" ]; then
    printf "DATALOADER_CONFIG file %s not found, cannot continue.\n" "${DATALOADER_CONFIG}"
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
    printf "GRADIENT_ACCUMULATION_STEPS not set, defaulting to 1.\n"
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
if [[ "$USE_BITFIT" != "false" ]]; then
    echo "Enabling BitFit."
    BITFIT_ARGS="--freeze_unet_strategy=bitfit"
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

export MAX_TRAIN_STEPS_ARGS=""
if [ -n "$MAX_NUM_STEPS" ] && [[ "$MAX_NUM_STEPS" != 0 ]]; then
    export MAX_TRAIN_STEPS_ARGS="--max_train_steps=${MAX_NUM_STEPS}"
fi

export CONTROLNET_ARGS=""
if [ -n "$CONTROLNET" ] && [[ "$CONTROLNET" == "true" ]]; then
    export CONTROLNET_ARGS="--controlnet"
fi


# Run the training script.
accelerate launch ${ACCELERATE_EXTRA_ARGS} --mixed_precision="${MIXED_PRECISION}" --num_processes="${TRAINING_NUM_PROCESSES}" --num_machines="${TRAINING_NUM_MACHINES}" --dynamo_backend="${TRAINING_DYNAMO_BACKEND}" train_sdxl.py \
    --model_type="${MODEL_TYPE}" ${DORA_ARGS} --pretrained_model_name_or_path="${MODEL_NAME}" ${XFORMERS_ARG} ${GRADIENT_ARG} --set_grads_to_none --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --resume_from_checkpoint="${RESUME_CHECKPOINT}" ${DELETE_ARGS} ${SNR_GAMMA_ARG} --data_backend_config="${DATALOADER_CONFIG}" \
    --num_train_epochs=${NUM_EPOCHS} ${MAX_TRAIN_STEPS_ARGS} --metadata_update_interval=${METADATA_UPDATE_INTERVAL} \
    ${OPTIMIZER_ARG} --learning_rate="${LEARNING_RATE}" --lr_scheduler="${LR_SCHEDULE}" --seed "${TRAINING_SEED}" --lr_warmup_steps="${LR_WARMUP_STEPS}" \
    --output_dir="${OUTPUT_DIR}" ${BITFIT_ARGS} ${ASPECT_BUCKET_ROUNDING_ARGS} \
    --inference_scheduler_timestep_spacing="${INFERENCE_SCHEDULER_TIMESTEP_SPACING}" --training_scheduler_timestep_spacing="${TRAINING_SCHEDULER_TIMESTEP_SPACING}" \
    ${DEBUG_EXTRA_ARGS}	${TF32_ARG} --mixed_precision="${MIXED_PRECISION}" ${TRAINER_EXTRA_ARGS} \
    --train_batch="${TRAIN_BATCH_SIZE}" --caption_dropout_probability=${CAPTION_DROPOUT_PROBABILITY} \
    --validation_prompt="${VALIDATION_PROMPT}" --num_validation_images=1 --validation_num_inference_steps="${VALIDATION_NUM_INFERENCE_STEPS}" ${VALIDATION_ARGS} \
    --minimum_image_size="${MINIMUM_RESOLUTION}" --resolution="${RESOLUTION}" --validation_resolution="${VALIDATION_RESOLUTION}" \
    --resolution_type="${RESOLUTION_TYPE}" \
    --checkpointing_steps="${CHECKPOINTING_STEPS}" --checkpoints_total_limit="${CHECKPOINTING_LIMIT}" \
    --validation_steps="${VALIDATION_STEPS}" --tracker_run_name="${TRACKER_RUN_NAME}" --tracker_project_name="${TRACKER_PROJECT_NAME}" \
    --validation_guidance="${VALIDATION_GUIDANCE}" --validation_guidance_rescale="${VALIDATION_GUIDANCE_RESCALE}" --validation_negative_prompt="${VALIDATION_NEGATIVE_PROMPT}" ${EMA_ARGS} ${CONTROLNET_ARGS}

exit 0