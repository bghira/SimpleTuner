#!/bin/bash
# =============================================================================
# RunPod Entrypoint: LongCat Video Full Finetune
# =============================================================================
set -e

echo "============================================="
echo "  LongCat Video Full Finetune - RunPod"
echo "============================================="
echo ""

# -----------------------------------------------------------------------------
# 1. Check required environment variables
# -----------------------------------------------------------------------------
echo "[1/6] Checking configuration..."

# Check required secrets (can come as RUNPOD_SECRET_ or directly)
AWS_BUCKET_NAME="${AWS_BUCKET_NAME:-${RUNPOD_SECRET_AWS_BUCKET_NAME}}"
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-${RUNPOD_SECRET_AWS_ACCESS_KEY_ID}}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-${RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY}}"

if [ -z "$AWS_BUCKET_NAME" ]; then
    echo "ERROR: AWS_BUCKET_NAME not set!"
    echo "Please configure the AWS_BUCKET_NAME secret in your RunPod template"
    exit 1
fi

if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "ERROR: AWS_ACCESS_KEY_ID not set!"
    echo "Please configure the AWS_ACCESS_KEY_ID secret in your RunPod template"
    exit 1
fi

if [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "ERROR: AWS_SECRET_ACCESS_KEY not set!"
    echo "Please configure the AWS_SECRET_ACCESS_KEY secret in your RunPod template"
    exit 1
fi

echo "  [OK] AWS credentials configured"

# -----------------------------------------------------------------------------
# 2. Set default values for optional variables
# -----------------------------------------------------------------------------
echo "[2/6] Applying configuration..."

# AWS
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DATA_PREFIX="${AWS_DATA_PREFIX:-}"
export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-}"

# Training
export USE_PARQUET="${USE_PARQUET:-false}"
export AUTO_START_TRAINING="${AUTO_START_TRAINING:-false}"
export MODEL_TYPE="${MODEL_TYPE:-full}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-30000}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-2500}"
export VALIDATION_EVERY_N_STEPS="${VALIDATION_EVERY_N_STEPS:-2500}"
export NUM_GPUS="${NUM_GPUS:-8}"

# LoRA (if set, uses LoRA instead of full finetune)
export LORA_RANK="${LORA_RANK:-}"

# Precision
export BASE_MODEL_PRECISION="${BASE_MODEL_PRECISION:-bf16}"

# Directories
export CONFIG_DIR="${CONFIG_DIR:-/workspace/config}"
export CACHE_DIR="${CACHE_DIR:-/workspace/cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output}"
export SIMPLETUNER_DIR="${SIMPLETUNER_DIR:-/workspace/SimpleTuner}"

echo "  [OK] Bucket: $AWS_BUCKET_NAME"
echo "  [OK] Region: $AWS_REGION"
echo "  [OK] Prefix: ${AWS_DATA_PREFIX:-<bucket root>}"
echo "  [OK] Training type: ${LORA_RANK:+LoRA rank $LORA_RANK}${LORA_RANK:-Full Finetune}"
echo "  [OK] Precision: $BASE_MODEL_PRECISION"
echo "  [OK] Learning Rate: $LEARNING_RATE"
echo "  [OK] Max Steps: $MAX_TRAIN_STEPS"
echo "  [OK] GPUs: $NUM_GPUS"
echo "  [OK] Use Parquet: $USE_PARQUET"
echo "  [OK] Auto Start: $AUTO_START_TRAINING"

# -----------------------------------------------------------------------------
# 3. Generate configuration files
# -----------------------------------------------------------------------------
echo "[3/6] Generating configuration files..."

# Determine model_type
if [ -n "$LORA_RANK" ]; then
    EFFECTIVE_MODEL_TYPE="lora"
    LORA_CONFIG='"lora_rank": '$LORA_RANK','
else
    EFFECTIVE_MODEL_TYPE="full"
    LORA_CONFIG=""
fi

# Generate training_config.json
cat > "${CONFIG_DIR}/training_config.json" << EOF
{
  "model_type": "${EFFECTIVE_MODEL_TYPE}",
  "model_family": "longcat_video",
  "model_flavour": "final",
  "pretrained_model_name_or_path": null,

  "base_model_precision": "${BASE_MODEL_PRECISION}",
  "attention_implementation": "sdpa",

  "output_dir": "${OUTPUT_DIR}",
  "data_backend_config": "${CONFIG_DIR}/databackend.json",

  ${LORA_CONFIG}
  "train_batch_size": ${TRAIN_BATCH_SIZE},
  "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS},
  "gradient_checkpointing": true,

  "learning_rate": ${LEARNING_RATE},
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 500,
  "max_train_steps": ${MAX_TRAIN_STEPS},
  "checkpointing_steps": ${CHECKPOINTING_STEPS},
  "checkpoints_total_limit": 5,

  "optimizer": "adamw_bf16",
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_weight_decay": 0.01,
  "adam_epsilon": 1e-8,
  "max_grad_norm": 1.0,

  "fsdp_enable": true,
  "fsdp_version": 2,
  "fsdp_state_dict_type": "SHARDED_STATE_DICT",
  "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
  "fsdp_cpu_ram_efficient_loading": true,

  "validation_prompt": "A professional video showing the main subject in high quality, smooth motion, natural lighting",
  "validation_negative_prompt": "",
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0,
  "validation_every_n_steps": ${VALIDATION_EVERY_N_STEPS},
  "num_validation_videos": 2,

  "seed": 42,
  "mixed_precision": "bf16",
  "report_to": "tensorboard",
  "logging_dir": "/workspace/logs",

  "dataloader_num_workers": 4,
  "allow_tf32": true
}
EOF

echo "  [OK] training_config.json generated"

# Generate databackend.json
if [ "$USE_PARQUET" = "true" ] && [ -f "${CONFIG_DIR}/metadata.parquet" ]; then
    # Parquet mode
    cat > "${CONFIG_DIR}/databackend.json" << EOF
[
  {
    "id": "longcat-video-dataset",
    "type": "aws",
    "dataset_type": "video",

    "aws_bucket_name": "${AWS_BUCKET_NAME}",
    "aws_region_name": "${AWS_REGION}",
    "aws_endpoint_url": ${AWS_ENDPOINT_URL:+\"$AWS_ENDPOINT_URL\"}${AWS_ENDPOINT_URL:-null},
    "aws_access_key_id": "${AWS_ACCESS_KEY_ID}",
    "aws_secret_access_key": "${AWS_SECRET_ACCESS_KEY}",
    "aws_data_prefix": "${AWS_DATA_PREFIX}",

    "metadata_backend": "parquet",
    "caption_strategy": "parquet",
    "parquet": {
      "path": "${CONFIG_DIR}/metadata.parquet",
      "filename_column": "filename",
      "caption_column": "caption",
      "width_column": "width",
      "height_column": "height"
    },

    "resolution": 480,
    "resolution_type": "pixel_area",
    "minimum_image_size": 256,
    "maximum_image_size": 1024,

    "crop": true,
    "crop_style": "center",
    "crop_aspect": "preserve",

    "cache_dir_vae": "${CACHE_DIR}/vae",
    "preserve_data_backend_cache": true,
    "skip_file_discovery": "vae,aspect,metadata",

    "video": {
      "bucket_strategy": "aspect_ratio",
      "min_frames": 1,
      "max_frames": 500,
      "frame_interval": 4
    },

    "repeats": 0
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "${CACHE_DIR}/text",
    "write_batch_size": 128
  }
]
EOF
    echo "  [OK] databackend.json generated (Parquet mode)"
else
    # Direct S3 mode
    cat > "${CONFIG_DIR}/databackend.json" << EOF
[
  {
    "id": "longcat-video-dataset",
    "type": "aws",
    "dataset_type": "video",

    "aws_bucket_name": "${AWS_BUCKET_NAME}",
    "aws_region_name": "${AWS_REGION}",
    "aws_endpoint_url": ${AWS_ENDPOINT_URL:+\"$AWS_ENDPOINT_URL\"}${AWS_ENDPOINT_URL:-null},
    "aws_access_key_id": "${AWS_ACCESS_KEY_ID}",
    "aws_secret_access_key": "${AWS_SECRET_ACCESS_KEY}",
    "aws_data_prefix": "${AWS_DATA_PREFIX}",

    "caption_strategy": "textfile",

    "resolution": 480,
    "resolution_type": "pixel_area",
    "minimum_image_size": 256,
    "maximum_image_size": 1024,

    "crop": true,
    "crop_style": "center",
    "crop_aspect": "preserve",

    "cache_dir_vae": "${CACHE_DIR}/vae",
    "preserve_data_backend_cache": true,

    "video": {
      "bucket_strategy": "aspect_ratio",
      "min_frames": 1,
      "max_frames": 500,
      "frame_interval": 4
    },

    "repeats": 0
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "${CACHE_DIR}/text",
    "write_batch_size": 128
  }
]
EOF
    echo "  [OK] databackend.json generated (direct S3 mode)"
fi

# -----------------------------------------------------------------------------
# 4. Create convenience scripts
# -----------------------------------------------------------------------------
echo "[4/6] Creating convenience scripts..."

# Script to generate Parquet
cat > /workspace/generate_parquet.sh << 'SCRIPT'
#!/bin/bash
echo "Generating Parquet metadata..."
python /workspace/scripts/generate_metadata_parquet.py \
    --bucket "$AWS_BUCKET_NAME" \
    --prefix "$AWS_DATA_PREFIX" \
    --output "${CONFIG_DIR}/metadata.parquet" \
    --region "$AWS_REGION" \
    ${AWS_ENDPOINT_URL:+--endpoint-url "$AWS_ENDPOINT_URL"} \
    --workers 32

if [ -f "${CONFIG_DIR}/metadata.parquet" ]; then
    echo "[OK] Parquet generated successfully!"
    echo "Set USE_PARQUET=true to use it in your next training run"
else
    echo "[ERROR] Failed to generate Parquet"
    exit 1
fi
SCRIPT
chmod +x /workspace/generate_parquet.sh

# Script to start training
cat > /workspace/start_training.sh << 'SCRIPT'
#!/bin/bash
echo "============================================="
echo "  Starting LongCat Video Training"
echo "============================================="

cd "${SIMPLETUNER_DIR}"

# Configure environment
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# Detect NVLink
if nvidia-smi topo -m 2>/dev/null | grep -q "NV"; then
    echo "NVLink detected"
    export NCCL_P2P_LEVEL=NVL
fi

echo "Starting with ${NUM_GPUS} GPUs..."
echo ""

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    -m simpletuner.train \
    --config "${CONFIG_DIR}/training_config.json" \
    "$@"
SCRIPT
chmod +x /workspace/start_training.sh

# Script to monitor training
cat > /workspace/monitor.sh << 'SCRIPT'
#!/bin/bash
echo "Starting training monitor..."
echo "TensorBoard: http://localhost:6006"
echo ""
tensorboard --logdir /workspace/logs --port 6006 --bind_all &
watch -n 5 nvidia-smi
SCRIPT
chmod +x /workspace/monitor.sh

echo "  [OK] Scripts created in /workspace/"
echo "    - generate_parquet.sh"
echo "    - start_training.sh"
echo "    - monitor.sh"

# -----------------------------------------------------------------------------
# 5. Display system information
# -----------------------------------------------------------------------------
echo "[5/6] System information..."

echo "  Python: $(python --version 2>&1 | cut -d' ' -f2)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.version.cuda)')"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "  GPUs: ${GPU_COUNT}x ${GPU_NAME} (${GPU_MEM})"
fi

# -----------------------------------------------------------------------------
# 6. Final action
# -----------------------------------------------------------------------------
echo "[6/6] Finalizing setup..."

if [ "$USE_PARQUET" = "true" ] && [ ! -f "${CONFIG_DIR}/metadata.parquet" ]; then
    echo ""
    echo "  [WARNING] USE_PARQUET=true but metadata.parquet does not exist"
    echo "  Run: /workspace/generate_parquet.sh"
    echo ""
fi

if [ "$AUTO_START_TRAINING" = "true" ]; then
    echo ""
    echo "============================================="
    echo "  AUTO_START_TRAINING=true"
    echo "  Starting training automatically..."
    echo "============================================="
    echo ""

    # Start TensorBoard in background
    tensorboard --logdir /workspace/logs --port 6006 --bind_all &

    # Start training
    exec /workspace/start_training.sh
else
    echo ""
    echo "============================================="
    echo "  Setup complete!"
    echo "============================================="
    echo ""
    echo "  Next steps:"
    echo "    1. (Optional) Generate Parquet: /workspace/generate_parquet.sh"
    echo "    2. Start training: /workspace/start_training.sh"
    echo "    3. Monitor: /workspace/monitor.sh"
    echo ""
    echo "  Configuration files:"
    echo "    - ${CONFIG_DIR}/training_config.json"
    echo "    - ${CONFIG_DIR}/databackend.json"
    echo ""
    echo "  Directories:"
    echo "    - VAE Cache: ${CACHE_DIR}/vae"
    echo "    - Text Cache: ${CACHE_DIR}/text"
    echo "    - Output: ${OUTPUT_DIR}"
    echo "    - Logs: /workspace/logs"
    echo ""

    # Keep container running (for RunPod JupyterLab/SSH)
    if [ -f /start.sh ]; then
        exec /start.sh
    else
        # Fallback: keep running
        tail -f /dev/null
    fi
fi
