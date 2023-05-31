#!/bin/bash
export RESUME_CHECKPOINT="latest"
export CHECKPOINTING_STEPS=100
export NUM_INSTANCE_IMAGES=22976 #@param {type:"integer"}
export LEARNING_RATE=4e-7 #@param {type:"number"}

# Configure these values.
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
#export MODEL_NAME="/notebooks/datasets/models/pipeline"
export BASE_DIR="/notebooks/datasets"
export INSTANCE_DIR="${BASE_DIR}/midjourney"
export OUTPUT_DIR="${BASE_DIR}/models"

# Regularization data config.
# This helps retain previous knowledge from the model.
export CLASS_PROMPT="a person"
export CLASS_DIR="${BASE_DIR}/processed_faces"
export NUM_CLASS_IMAGES=7000

#export MAX_NUM_STEPS=$((NUM_INSTANCE_IMAGES * 80))
#export MAX_NUM_STEPS=10000
export NUM_EPOCHS=4
export LR_SCHEDULE="polynomial"
export LR_WARMUP_STEPS=$((MAX_NUM_STEPS / 10))

export TRAIN_BATCH_SIZE=3
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
  --use_8bit_adam \
  --seed 420420420 \
  --scale_lr \
  --learning_rate=${LEARNING_RATE} \
  --lr_scheduler=${LR_SCHEDULE} \
  --lr_warmup_steps=${LR_WARMUP_STEPS} \
  --num_train_epochs=${NUM_EPOCHS} \
  --mixed_precision=${MIXED_PRECISION} \
  --checkpointing_steps=${CHECKPOINTING_STEPS} \
  --checkpoints_total_limit=10 \
  --allow_tf32 \
  --resume_from_checkpoint=${RESUME_CHECKPOINT} \
  --train_text_encoder \
  --freeze_encoder --freeze_encoder_strategy='after' --freeze_encoder_after=17 \
  --offset_noise --noise_offset=0.1 \
  --snr_gamma 5.0 
  #--max_train_steps=${MAX_NUM_STEPS} \
  #--gradient_accumulation_steps=2 \  
  #--gradient_checkpointing