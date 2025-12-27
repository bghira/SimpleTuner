# =============================================================================
# RunPod Template: LongCat Video Full Finetune
# =============================================================================
# Imagem pré-configurada para full finetune do LongCat Video (13.6B)
# com SimpleTuner, otimizada para 8x A100 80GB / H200
#
# Secrets necessários no RunPod:
#   - AWS_BUCKET_NAME (obrigatório)
#   - AWS_ACCESS_KEY_ID (obrigatório)
#   - AWS_SECRET_ACCESS_KEY (obrigatório)
#   - AWS_REGION (opcional, default: us-east-1)
#   - AWS_DATA_PREFIX (opcional, default: "")
#
# Variáveis de ambiente opcionais:
#   - USE_PARQUET: "true" ou "false" (default: false)
#   - AUTO_START_TRAINING: "true" ou "false" (default: false)
#   - LEARNING_RATE: (default: 1e-5)
#   - MAX_TRAIN_STEPS: (default: 30000)
#   - TRAIN_BATCH_SIZE: (default: 1)
#   - GRADIENT_ACCUMULATION_STEPS: (default: 4)
#   - LORA_RANK: se definido, usa LoRA em vez de full finetune
#   - NUM_GPUS: (default: 8)
# =============================================================================

# Base image com PyTorch 2.5.1, CUDA 12.4, Ubuntu 22.04
FROM runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL maintainer="danielxmed"
LABEL description="LongCat Video Full Finetune with SimpleTuner"
LABEL version="1.0.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/workspace/models
ENV TORCH_HOME=/workspace/torch_cache
ENV SIMPLETUNER_DIR=/workspace/SimpleTuner
ENV CONFIG_DIR=/workspace/config
ENV CACHE_DIR=/workspace/cache
ENV OUTPUT_DIR=/workspace/output

# Diretório de trabalho
WORKDIR /workspace

# Instala dependências do sistema
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
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Clona e instala SimpleTuner
RUN git clone https://github.com/bghira/SimpleTuner.git ${SIMPLETUNER_DIR} \
    && cd ${SIMPLETUNER_DIR} \
    && pip install --no-cache-dir -e ".[cuda]"

# Instala dependências adicionais
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

# Cria diretórios necessários
RUN mkdir -p ${CONFIG_DIR} \
    ${CACHE_DIR}/vae \
    ${CACHE_DIR}/text \
    ${OUTPUT_DIR} \
    /workspace/logs

# Copia scripts e configurações
COPY scripts/ /workspace/scripts/
COPY configs/ ${CONFIG_DIR}/

# Torna scripts executáveis
RUN chmod +x /workspace/scripts/*.sh /workspace/scripts/*.py

# Script de entrada
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expõe portas
EXPOSE 8888 6006 22

# Entrypoint
ENTRYPOINT ["/entrypoint.sh"]
