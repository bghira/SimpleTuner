#!/bin/bash

echo "Running dependency installation for Ubuntu."
apt -y install cuda-cupti-11-8 cuda-cupti-11-7 libnccl-dev cuda-cupti-dev-11-7 cuda-cupti-dev-11-8
echo "Creating python venv"
python -m venv .venv
echo "Activating venv"
. .venv/bin/activate
echo "Installing poetry"
pip install -U pip poetry
# echo "Installing torch via Pytorch repo, for NVIDIA compatibility."
# pip3 install -U boto3 botocore urllib3 xformers torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --force
echo "Using poetry to install project dependencies"
poetry install

if ! [ -f 'config/config.env' ]; then
	echo "Copying SDXL example config to config/config.env"
	cp config/config.env.example config/config.env
fi
echo "Done."
