#!/bin/bash

# Pull config from env.sh
source env.sh

accelerate launch  \
  --num_processes=2 --num_machines=1 --mixed_precision=${MIXED_PRECISION} --dynamo_backend='no' \
  train_dreambooth.py \
  --pretrained_model_name_or_path="${MODEL_NAME}"  \
  --instance_data_dir="${INSTANCE_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --resolution=${RESOLUTION} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --seed 420420420 \
  --learning_rate=${LEARNING_RATE} \
  --lr_scheduler=${LR_SCHEDULE} \
  --num_train_epochs=${NUM_EPOCHS} \
  --mixed_precision=${MIXED_PRECISION} \
  --checkpointing_steps=${CHECKPOINTING_STEPS} \
  --checkpoints_total_limit=10 \
  --allow_tf32 \
  --resume_from_checkpoint=${RESUME_CHECKPOINT} \
  --offset_noise --noise_offset=0.1 --input_pertubation=0.1 \
  --use_8bit_adam \
  --train_text_encoder \
  --freeze_encoder --freeze_encoder_strategy='before' --freeze_encoder_before=17 --freeze_encoder_after=23 \
  --scale_lr --gradient_accumulation_steps 1 --gradient_checkpointing --snr_gamma 5.0


  #--prepend_instance_prompt --instance_prompt="${INSTANCE_PROMPT}" \
  #--max_train_steps=${MAX_NUM_STEPS} \
  #--print_filenames \

#--lr_warmup_steps=${LR_WARMUP_STEPS} \
