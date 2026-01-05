FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Hardware Architecture
ENV TORCH_CUDA_ARCH_LIST=8.9
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

# Settings
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/workspace/huggingface
ENV SIMPLETUNER_WORKSPACE=/workspace/simpletuner
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 1. System Dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    wget \
    curl \
    vim \
    tmux \
    htop \
    rsync \
    net-tools \
    openssh-server \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# 2. Python Environment & Core Deps
RUN python${PYTHON_VERSION} -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
       "huggingface_hub[cli,hf_transfer]" \
       wandb \
       mpi4py \
       ninja

# 3. Install SimpleTuner
# Using 'release' branch for stability. Change to 'main' for latest features.
ARG SIMPLETUNER_BRANCH=release
RUN git clone https://github.com/bghira/SimpleTuner --branch $SIMPLETUNER_BRANCH \
    && cd SimpleTuner \
    && pip install --no-cache-dir -e .[jxl] \
    && pip install --no-build-isolation --no-cache-dir sageattention==1.0.6

# 4. Setup Runtime
COPY --chmod=755 docker-start.sh /start.sh
VOLUME /workspace

# SSH & WebUI Ports
EXPOSE 22 8001

ENTRYPOINT [ "/start.sh" ]
