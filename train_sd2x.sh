#!/bin/bash

# Pull the default config.
source sd21-env.sh.example
# Pull config from env.sh
source sd21-env.sh

if [ -z "${PROTECT_JUPYTER_FOLDERS}" ]; then
    # We had no value for protecting the folders, so we nuke them!
    echo "Deleting Jupyter notebook folders in 5 seconds if you do not cancel out."
    echo "These folders are generally useless, and will cause problems if they remain."
    export seconds
    seconds=4
    for ((i=seconds;i>0;i--)); do
        echo -n "."
        sleep 1
    done
    echo "." # Newline
    echo "YOUR TIME HAS COME."
    if [ -n "${INSTANCE_DIR}" ]; then
     find "${INSTANCE_DIR}" -type d -name ".ipynb_checkpoints" -exec rm -vr {} \;
    fi
    find "${OUTPUT_DIR}" -type d -name ".ipynb_checkpoints" -exec rm -vr {} \;
    find "." -type d -name ".ipynb_checkpoints" -exec rm -vr {} \;
fi

if [ -z "$TRAINER_EXTRA_ARGS" ]; then
    export TRAINER_EXTRA_ARGS=""
fi

accelerate launch  \
--num_processes="${TRAINING_NUM_PROCESSES}" --num_machines="${TRAINING_NUM_MACHINES}" --mixed_precision="${MIXED_PRECISION}" --dynamo_backend="${TRAINING_DYNAMO_BACKEND}" \
train_sd21.py \
--pretrained_model_name_or_path="${MODEL_NAME}"  \
--instance_data_dir="${INSTANCE_DIR}" \
--output_dir="${OUTPUT_DIR}" \
--resolution="${RESOLUTION}" \
--minimum_image_size="${MINIMUM_RESOLUTION}" \
--validation_resolution="${VALIDATION_RESOLUTION}" \
--resolution_type="${RESOLUTION_TYPE}" \
--train_batch_size="${TRAIN_BATCH_SIZE}" \
--seed "${TRAINING_SEED}" \
--learning_rate="${LEARNING_RATE}" \
--lr_end="${LEARNING_RATE_END}" \
--lr_scheduler="${LR_SCHEDULE}" \
--num_train_epochs="${NUM_EPOCHS}" \
--mixed_precision="${MIXED_PRECISION}" \
--checkpointing_steps="${CHECKPOINTING_STEPS}" \
--checkpoints_total_limit=10 \
--allow_tf32 \
--resume_from_checkpoint="${RESUME_CHECKPOINT}" \
--use_8bit_adam \
--train_text_encoder --text_encoder_limit="${TEXT_ENCODER_LIMIT}" \
--freeze_encoder --freeze_encoder_strategy="${TEXT_ENCODER_FREEZE_STRATEGY}" --freeze_encoder_before="${TEXT_ENCODER_FREEZE_BEFORE}" --freeze_encoder_after="${TEXT_ENCODER_FREEZE_AFTER}" \
--gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" --gradient_checkpointing \
--seen_state_path="${SEEN_STATE_PATH}" \
--state_path="${STATE_PATH}" \
--caption_dropout_probability="${CAPTION_DROPOUT_PROBABILITY}" \
--caption_strategy="${CAPTION_STRATEGY}" \
--data_backend_config="${DATALOADER_CONFIG}" ${TRAINER_EXTRA_ARGS}