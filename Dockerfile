# SimpleTuner needs CU118
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# /workspace is the default volume for Runpod & other hosts
WORKDIR /workspace

# Update apt-get
RUN apt-get update -y

# Prevents different commands from being stuck by waiting
# on user input during build
ENV DEBIAN_FRONTEND noninteractive

# Install openssh & git
RUN apt-get install -y --no-install-recommends openssh-server \
                                               openssh-client \
                                               git \
                                               git-lfs

# Installl misc unix libraries
RUN apt-get install -y wget \
                       curl \
                       tmux \
                       tldr \
                       nvtop \
                       vim \
                       rsync \
                       net-tools \
                       less \
                       iputils-ping \
                       7zip \
                       zip \
                       unzip \
                       htop \
                       inotify-tools

# Set up git to support LFS, and to store credentials; useful for Huggingface Hub
RUN git config --global credential.helper store && \
    git lfs install

# Install Python VENV
RUN apt-get install -y python3.10-venv

# Ensure SSH access. Not needed for Runpod but is required on Vast and other Docker hosts
EXPOSE 22/tcp

# Install misc Python & CUDA Libraries
RUN apt-get update -y && apt-get install -y python3 python3-pip libcudnn8 libcudnn8-dev
RUN python3 -m pip install pip --upgrade

# HF
ARG HUGGING_FACE_HUB_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN
ENV HF_HOME=/workspace/huggingface

RUN pip3 install "huggingface_hub[cli]"

RUN huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN" --add-to-git-credential

# WanDB
ARG WANDB_TOKEN
ENV WANDB_TOKEN=$WANDB_TOKEN

RUN pip3 install wandb

RUN wandb login "$WANDB_TOKEN"

# Clone SimpleTuner
RUN git clone https://github.com/bghira/SimpleTuner --branch release
# RUN git clone https://github.com/bghira/SimpleTuner --branch main # Uncomment to use latest (possibly unstable) version

# Install SimpleTuner
RUN pip3 install poetry
RUN cd SimpleTuner && python3 -m venv .venv && poetry install --no-root
RUN chmod +x SimpleTuner/train.sh

# Copy start script with exec permissions
COPY --chmod=755 docker-start.sh /start.sh

# Dummy entrypoint
ENTRYPOINT [ "/start.sh" ]
