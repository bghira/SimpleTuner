# SimpleTuner needs CU141
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION=3.11

# Prevent commands from blocking for input during build
ENV DEBIAN_FRONTEND=noninteractive

# /workspace is the default volume for Runpod & other hosts
WORKDIR /workspace

# Base system dependencies (including Python ${PYTHON_VERSION} toolchain)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        htop \
        inotify-tools \
        iputils-ping \
        less \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        net-tools \
        nvtop \
        openssh-client \
        openssh-server \
        p7zip-full \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        rsync \
        tmux \
        tldr \
        unzip \
        vim \
        wget \
        zip && \
    rm -rf /var/lib/apt/lists/*

# Configure git to support LFS and credential storage
RUN git config --global credential.helper store && \
    git lfs install

# Create a dedicated virtual environment with the requested Python version
RUN python${PYTHON_VERSION} -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Use the virtual environment for all subsequent Python work
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Ensure SSH access. Not needed for Runpod but is required on Vast and other Docker hosts
EXPOSE 22/tcp

# HuggingFace cache location and platform hint for setup.py
ENV HF_HOME=/workspace/huggingface
ENV SIMPLETUNER_PLATFORM=cuda

# Install supporting CLIs ahead of the project install
RUN pip install --no-cache-dir "huggingface_hub[cli]" wandb

# Copy project into image and install with setup.py so dependency logic is reused
COPY . /workspace/SimpleTuner
WORKDIR /workspace/SimpleTuner
RUN pip install --no-cache-dir .

# Copy start script with exec permissions
COPY --chmod=755 docker-start.sh /start.sh

# Return to default workspace location
WORKDIR /workspace

# Dummy entrypoint
ENTRYPOINT [ "/start.sh" ]
