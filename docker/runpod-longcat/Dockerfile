# =============================================================================
# RunPod Template: LongCat Video Full Finetune
# =============================================================================
# Pre-configured image for full finetuning LongCat Video (13.6B)
# with SimpleTuner, optimized for 8x A100 80GB / H200
#
# Required secrets in RunPod:
#   - AWS_BUCKET_NAME (required)
#   - AWS_ACCESS_KEY_ID (required)
#   - AWS_SECRET_ACCESS_KEY (required)
#   - AWS_REGION (optional, default: us-east-1)
#   - AWS_DATA_PREFIX (optional, default: "")
#
# Optional environment variables:
#   - USE_PARQUET: "true" or "false" (default: false)
#   - AUTO_START_TRAINING: "true" or "false" (default: false)
#   - LEARNING_RATE: (default: 1e-5)
#   - MAX_TRAIN_STEPS: (default: 30000)
#   - TRAIN_BATCH_SIZE: (default: 1)
#   - GRADIENT_ACCUMULATION_STEPS: (default: 4)
#   - LORA_RANK: if set, uses LoRA instead of full finetune
#   - NUM_GPUS: (default: 8)
# =============================================================================

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

LABEL maintainer="danielxmed"
LABEL description="LongCat Video Full Finetune with SimpleTuner"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/workspace/models
ENV TORCH_HOME=/workspace/torch_cache
ENV SIMPLETUNER_DIR=/workspace/SimpleTuner
ENV CONFIG_DIR=/workspace/config
ENV CACHE_DIR=/workspace/cache
ENV OUTPUT_DIR=/workspace/output

# Working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    vim \
    htop \
    nvtop \
    tmux \
    screen \
    rsync \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Clone and install SimpleTuner (using fork with RunPod fixes)
RUN git clone https://github.com/danielxmed/SimpleTuner.git ${SIMPLETUNER_DIR} \
    && cd ${SIMPLETUNER_DIR} \
    && pip install --no-cache-dir -e ".[cuda]"

# Install additional dependencies
RUN pip install --no-cache-dir \
    boto3 \
    pyarrow \
    pandas \
    tqdm \
    opencv-python-headless \
    tensorboard \
    wandb \
    accelerate \
    deepspeed

# Create required directories
RUN mkdir -p ${CONFIG_DIR} \
    ${CACHE_DIR}/vae \
    ${CACHE_DIR}/text \
    ${OUTPUT_DIR} \
    /workspace/logs

# Copy scripts and configurations
COPY scripts/ /workspace/scripts/
COPY configs/ ${CONFIG_DIR}/

# Make scripts executable
RUN find /workspace/scripts -type f \( -name "*.sh" -o -name "*.py" \) -exec chmod +x {} \;

# Entry script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose ports
EXPOSE 8888 6006 22

# Entrypoint
ENTRYPOINT ["/entrypoint.sh"]
