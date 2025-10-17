# SimpleTuner WebUI Tutorial

## Introduction

This tutorial will help you get started with the SimpleTuner Web interface.

## Installing requirements

For Ubuntu systems, start by installing the required packages:

```bash
apt -y install python3.12-venv python3.12-dev
apt -y install libopenmpi-dev openmpi-bin cuda-toolkit-12-8 libaio-dev # if you're using DeepSpeed
```

## Installing SimpleTuner

> **NOTE:** Currently, we're installing from the webui-phase-one branch. This will be merged into main soon, at which time you can simply `pip install simpletuner[cuda]` for a streamlined experience.

```bash
git clone --branch=feature/webui-phase-one https://github.com/bghira/SimpleTuner
```

```bash
cd SimpleTuner/
python3.12 -m venv .venv
. .venv/bin/activate
```

### CUDA-specific dependencies

NVIDIA users will have to use the CUDA extras to pull in all the right dependencies:

```bash
pip install -e '.[cuda]'
```

There are other extras for users on apple and rocm hardware, see the [installation instructions](../INSTALL.md).

## Starting the server

To start the server with SSL on port 8080:

```bash
# for DeepSpeed, we'll need CUDA_HOME pointing to the correct location
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

simpletuner server --ssl --port 8080
```

Now, visit https://localhost:8080 in your web browser. You may need to forward the port over SSH, for example:

```bash
ssh -L 8080:localhost:8080 user@remote-server
```

## Using the WebUI

Once you have the page loaded, you'll be asked onboarding questions to set up your environment.

If you've been using SimpleTuner before without a WebUI, you can point to your existing config/ folder and all of your environments will be auto-discovered.

For new users, the default location of your configs and datasets will `~/.simpletuner/` and it's recommended to move your datasets somewhere with more space.

## Downloading a dataset

If you stuck with the defaults, you can download a dataset to `~/.simpletuner/datasets/`:

```bash
cd ~/.simpletuner/datasets/
huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --include=train/ --local-dir=pseudo-camera-10k
```