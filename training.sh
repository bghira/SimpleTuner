#!/bin/bash

# Pull the default config.
source sd21-env.sh.example
# Pull config from env.sh
source sd21-env.sh

accelerate launch  \
  --num_processes="${TRAINING_NUM_PROCESSES}" --num_machines="${TRAINING_NUM_MACHINES}" --mixed_precision="${MIXED_PRECISION}" --dynamo_backend="${TRAINING_DYNAMO_BACKEND}" \
  train_dreambooth.py \
  --pretrained_model_name_or_path="${MODEL_NAME}"  \
  --instance_data_dir="${INSTANCE_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --resolution="${RESOLUTION}" \
  --train_batch_size="${TRAIN_BATCH_SIZE}" \
  --seed "${TRAINING_SEED}" \
  --learning_rate="${LEARNING_RATE}" \
  --learning_rate_end="${LEARNING_RATE_END}" \
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
  --use_original_images="${USE_ORIGINAL_IMAGES}" \
  --seen_state_path="${SEEN_STATE_PATH}" \
  --state_path="${STATE_PATH}" \
  --caption_dropout_interval="${CAPTION_DROPOUT_INTERVAL}" \
  --caption-strategy="${CAPTION_STRATEGY}"


  #--prepend_instance_prompt --instance_prompt="${INSTANCE_PROMPT}" \
  #--max_train_steps=${MAX_NUM_STEPS} \
  #--print_filenames \

#--lr_warmup_steps=${LR_WARMUP_STEPS} \
