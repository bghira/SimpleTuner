#!/bin/bash
export RESUME_CHECKPOINT="latest"
export CHECKPOINTING_STEPS=100
export NUM_INSTANCE_IMAGES=22976 #@param {type:"integer"}
export LEARNING_RATE=4e-7 #@param {type:"number"}

# Configure these values.
export MODEL_NAME="ptx0/realism-engine"
export INSTANCE_PROMPT="lotr style "
#export MODEL_NAME="/notebooks/datasets/models/pipeline"
export BASE_DIR="/models/training/datasets"
export INSTANCE_DIR="${BASE_DIR}/lotr"
export OUTPUT_DIR="${BASE_DIR}/models"

#export MAX_NUM_STEPS=$((NUM_INSTANCE_IMAGES * 80))
#export MAX_NUM_STEPS=10000
export NUM_EPOCHS=4
export LR_SCHEDULE="polynomial"
export LR_WARMUP_STEPS=$((MAX_NUM_STEPS / 10))

export TRAIN_BATCH_SIZE=1
export RESOLUTION=768
export MIXED_PRECISION="bf16"

accelerate launch  \
  --num_processes=1 --num_machines=1 --mixed_precision=${MIXED_PRECISION} \
  train_dreambooth.py \
  --pretrained_model_name_or_path="${MODEL_NAME}"  \
  --instance_data_dir="${INSTANCE_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --resolution=${RESOLUTION} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --seed 420420420 \
  --scale_lr \
  --learning_rate=${LEARNING_RATE} \
  --lr_scheduler=${LR_SCHEDULE} \
  --lr_warmup_steps=${LR_WARMUP_STEPS} \
  --num_train_epochs=${NUM_EPOCHS} \
  --mixed_precision=${MIXED_PRECISION} \
  --checkpointing_steps=${CHECKPOINTING_STEPS} \
  --checkpoints_total_limit=10 \
  --prepend_instance_prompt --instance_prompt="${INSTANCE_PROMPT}" \
  --allow_tf32 \
  --resume_from_checkpoint=${RESUME_CHECKPOINT} \
  --train_text_encoder \
  --freeze_encoder --freeze_encoder_strategy='before' --freeze_encoder_before=17 --freeze_encoder_after=20 \
  --offset_noise --noise_offset=0.1 \
  --print_filenames \
  --use_8bit_adam \
  --snr_gamma 5.0

  #--max_train_steps=${MAX_NUM_STEPS} \
  #--gradient_accumulation_steps=2 \  
  #--gradient_checkpointing