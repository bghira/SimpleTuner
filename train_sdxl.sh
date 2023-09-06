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
if [ -z "${INSTANCE_DIR}" ]; then
	printf "INSTANCE_DIR not set, exiting.\n"
	exit 1
fi
if [ -z "${OUTPUT_DIR}" ]; then
	printf "OUTPUT_DIR not set, exiting.\n"
	exit 1
fi
if [ -z "${SEEN_STATE_PATH}" ]; then
	printf "SEEN_STATE_PATH not set, exiting.\n"
	exit 1
fi
if [ -z "${STATE_PATH}" ]; then
	printf "STATE_PATH not set, exiting.\n"
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
  find . -type d -name ".ipynb_checkpoints" -exec rm -vr {} \;
fi

# Run the training script.

accelerate launch ${ACCELERATE_EXTRA_ARGS} --mixed_precision="${MIXED_PRECISION}" --num_processes="${TRAINING_NUM_PROCESSES}" --num_machines="${TRAINING_NUM_MACHINES}" --dynamo_backend="${TRAINING_DYNAMO_BACKEND}" train_sdxl.py \
	--pretrained_model_name_or_path="${MODEL_NAME}" \
	--resume_from_checkpoint="${RESUME_CHECKPOINT}" \
	--num_train_epochs="${NUM_EPOCHS}" \
	--learning_rate="${LEARNING_RATE}" --lr_scheduler="${LR_SCHEDULE}" --seed "${TRAINING_SEED}" \
	--instance_data_dir="${INSTANCE_DIR}" --seen_state_path="${SEEN_STATE_PATH}" --state_path="${STATE_PATH}" --output_dir="${OUTPUT_DIR}" \
	${DEBUG_EXTRA_ARGS}	--mixed_precision="${MIXED_PRECISION}" --vae_dtype="${MIXED_PRECISION}" ${TRAINER_EXTRA_ARGS} \
	--train_batch="${TRAIN_BATCH_SIZE}" --caption_dropout_probability=${CAPTION_DROPOUT_PROBABILITY} \
	--validation_prompt="${VALIDATION_PROMPT}" --num_validation_images=1 \
	--resolution="${RESOLUTION}" --validation_resolution="${VALIDATION_RESOLUTION}" \
	--checkpointing_steps="${CHECKPOINTING_STEPS}" --checkpoints_total_limit="${CHECKPOINTING_LIMIT}" \
	--validation_steps="${VALIDATION_STEPS}" --tracker_run_name="${TRACKER_RUN_NAME}" --tracker_project_name="${TRACKER_PROJECT_NAME}" \
	--validation_guidance="${VALIDATION_GUIDANCE}" --validation_guidance_rescale="${VALIDATION_GUIDANCE_RESCALE}"

exit 0