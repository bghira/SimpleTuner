#!/bin/bash
# Pull the default config.
source sdxl-env.sh.example
# Pull config from env.sh
[ -f "sdxl-env.sh" ] && source sdxl-env.sh

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

if [ -z "$DATALOADER_CONFIG" ]; then
    printf "DATALOADER_CONFIG not set, cannot continue. See multidatabackend.json.example.\n"
    exit 1
fi
if ! [ -f "$DATALOADER_CONFIG" ]; then
    printf "DATALOADER_CONFIG file %s not found, cannot continue.\n" "${DATALOADER_CONFIG}"
    exit 1
fi

export SNR_GAMMA_ARG=""
if ! [ -z "$MIN_SNR_GAMMA" ]; then
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

# Run the training script.
accelerate launch ${ACCELERATE_EXTRA_ARGS} --mixed_precision="${MIXED_PRECISION}" --num_processes="${TRAINING_NUM_PROCESSES}" --num_machines="${TRAINING_NUM_MACHINES}" --dynamo_backend="${TRAINING_DYNAMO_BACKEND}" train_sdxl.py \
--pretrained_model_name_or_path="${MODEL_NAME}" ${XFORMERS_ARG} ${GRADIENT_ARG} --set_grads_to_none --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
--resume_from_checkpoint="${RESUME_CHECKPOINT}" ${DELETE_ARGS} ${SNR_GAMMA_ARG} --data_backend_config="${DATALOADER_CONFIG}" \
--num_train_epochs=${NUM_EPOCHS} --max_train_steps=${MAX_NUM_STEPS} --metadata_update_interval=${METADATA_UPDATE_INTERVAL} \
--learning_rate="${LEARNING_RATE}" --lr_scheduler="${LR_SCHEDULE}" --seed "${TRAINING_SEED}" --lr_warmup_steps="${LR_WARMUP_STEPS}" \
--output_dir="${OUTPUT_DIR}" \
--inference_scheduler_timestep_spacing="${INFERENCE_SCHEDULER_TIMESTEP_SPACING}" --training_scheduler_timestep_spacing="${TRAINING_SCHEDULER_TIMESTEP_SPACING}" \
${DEBUG_EXTRA_ARGS}	--mixed_precision="${MIXED_PRECISION}" --vae_dtype="${MIXED_PRECISION}" ${TRAINER_EXTRA_ARGS} \
--train_batch="${TRAIN_BATCH_SIZE}" --caption_dropout_probability=${CAPTION_DROPOUT_PROBABILITY} \
--validation_prompt="${VALIDATION_PROMPT}" --num_validation_images=1 --validation_num_inference_steps="${VALIDATION_NUM_INFERENCE_STEPS}" ${VALIDATION_ARGS} \
--minimum_image_size="${MINIMUM_RESOLUTION}" --resolution="${RESOLUTION}" --validation_resolution="${VALIDATION_RESOLUTION}" \
--resolution_type="${RESOLUTION_TYPE}" \
--checkpointing_steps="${CHECKPOINTING_STEPS}" --checkpoints_total_limit="${CHECKPOINTING_LIMIT}" \
--validation_steps="${VALIDATION_STEPS}" --tracker_run_name="${TRACKER_RUN_NAME}" --tracker_project_name="${TRACKER_PROJECT_NAME}" \
--validation_guidance="${VALIDATION_GUIDANCE}" --validation_guidance_rescale="${VALIDATION_GUIDANCE_RESCALE}" --validation_negative_prompt="${VALIDATION_NEGATIVE_PROMPT}"

exit 0