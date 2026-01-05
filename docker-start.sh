#!/bin/bash

# Export useful ENV variables, including all Runpod specific vars, to /etc/rp_environment
# This file can then later be sourced in a login shell
echo "Exporting environment variables..."
printenv |
	grep -E '^RUNPOD_|^PATH=|^HF_HOME=|^HF_TOKEN=|^HUGGING_FACE_HUB_TOKEN=|^WANDB_API_KEY=|^WANDB_TOKEN=|^_=' |
	sed 's/^\(.*\)=\(.*\)$/export \1="\2"/' >>/etc/rp_environment

# Add it to Bash login script
echo 'source /etc/rp_environment' >>~/.bashrc

# Vast.ai uses $SSH_PUBLIC_KEY
if [[ $SSH_PUBLIC_KEY ]]; then
	PUBLIC_KEY="${SSH_PUBLIC_KEY}"
fi

# Runpod uses $PUBLIC_KEY
if [[ $PUBLIC_KEY ]]; then
	mkdir -p ~/.ssh
	chmod 700 ~/.ssh
	echo "${PUBLIC_KEY}" >>~/.ssh/authorized_keys
	chmod 700 -R ~/.ssh
fi

# Start SSH server
service ssh start

# Login to HF
if [[ -n "${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}" ]]; then
	huggingface-cli login --token "${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}" --add-to-git-credential
else
	echo "HF_TOKEN or HUGGING_FACE_HUB_TOKEN not set; skipping login"
fi

# Login to WanDB
if [[ -n "${WANDB_API_KEY:-$WANDB_TOKEN}" ]]; then
	wandb login "${WANDB_API_KEY:-$WANDB_TOKEN}"
else
	echo "WANDB_API_KEY or WANDB_TOKEN not set; skipping login"
fi

# Start SimpleTuner Server
echo "Starting SimpleTuner Server..."
# We bind to 0.0.0.0 to be accessible outside the container
simpletuner server --host 0.0.0.0 --port 8001
