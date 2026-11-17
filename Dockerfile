FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04

# Hardware Architecture
ENV TORCH_CUDA_ARCH_LIST=9.0
ENV CUDA_HOME=/usr/local/cuda-13.2
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
# Install the CUDA 13.2 stack before building optional CUDA extensions.
RUN python${PYTHON_VERSION} -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cu132 \
       "torch==2.15.0+cu132" \
       "torchvision==0.29.2+cu132" \
       "torchaudio==2.11.1+cu132" \
       "nvidia-cutlass-dsl[cu13]>=4.8.1,<4.9" \
       "apache-tvm-ffi>=0.1.14,<0.2" \
       huggingface_hub \
       wandb \
       mpi4py \
       ninja \
       "bitsandbytes>=0.50.2" \
       "deepspeed>=0.20.0" \
       "torchao>=0.18.0,<0.19.0" \
       "nvidia-ml-py>=12.555" \
       "lm-eval>=0.4.4"

# 3. Install SimpleTuner
# Use main for current model integrations such as Boogu-Image.
ARG SIMPLETUNER_BRANCH=main
RUN git clone https://github.com/bghira/SimpleTuner --branch $SIMPLETUNER_BRANCH \
    && cd SimpleTuner \
    && pip install --no-cache-dir -e ".[jxl]" \
    && pip install --no-build-isolation --no-cache-dir sageattention==1.0.6

# 4. Setup Runtime
COPY --chmod=755 docker-start.sh /start.sh
VOLUME /workspace

# SSH & WebUI Ports
EXPOSE 22 8001

ENTRYPOINT [ "/start.sh" ]
