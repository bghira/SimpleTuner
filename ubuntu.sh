#!/bin/bash

echo "Running dependency installation for Ubuntu."
apt -y install nvidia-cuda-dev nvidia-cuda-toolkit
echo "Creating python venv"
python -m venv .venv
echo "Activating venv"
. .venv/bin/activate
echo "Installing poetry"
pip install -U pip poetry
echo "Installing torch via Pytorch repo, for NVIDIA compatibility."
pip3 install xformers torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --force
echo "Using poetry to install project dependencies"
poetry install

if ! [ -f 'sdxl-env.sh' ]; then
	echo "Copying SDXL example config to sdxl-env.sh"
	cp sdxl-env.sh.example sdxl-env.sh
fi
if ! [ -f 'env.sh' ]; then
	echo "Copying SD 2.1 example config to sd21-env.sh"
	cp sd21-env.sh.example sd21-env.sh
fi
echo "Done."
