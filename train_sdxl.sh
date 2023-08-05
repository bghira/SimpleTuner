#!/bin/bash
# Pull the default config.
source sdxl-env.sh.example
# Pull config from env.sh
[ -f "sdxl-env.sh" ] && source sdxl-env.sh

accelerate launch ${ACCELERATE_EXTRA_ARGS} --mixed_precision="${MIXED_PRECISION}" --num_processes="${TRAINING_NUM_PROCESSES}" --num_machines="${TRAINING_NUM_MACHINES}" --dynamo_backend="${TRAINING_DYNAMO_BACKEND}" train_sdxl.py \
	--pretrained_model_name_or_path="${MODEL_NAME}" \
	--resume_from_checkpoint="${RESUME_CHECKPOINT}" \
	--learning_rate="${LEARNING_RATE}" --seed "${TRAINING_SEED}" \
	--instance_data_dir="${INSTANCE_DIR}" --seen_state_path="${SEEN_STATE_PATH}" \
	${DEBUG_EXTRA_ARGS}	--mixed_precision="${MIXED_PRECISION}" --vae_dtype="${MIXED_PRECISION}" ${TRAINER_EXTRA_ARGS} \
	--train_batch="${TRAIN_BATCH_SIZE}" \
	--validation_prompt="${VALIDATION_PROMPT}" --num_validation_images=1 \
	--resolution="${RESOLUTION}" --validation_resolution="${RESOLUTION}" \
	--checkpointing_steps="${CHECKPOINTING_STEPS}" --checkpoints_total_limit="${CHECKPOINTING_LIMIT}" \
	--validation_steps="${VALIDATION_STEPS}" --tracker_run_name="${TRACKER_RUN_NAME}"

exit 0