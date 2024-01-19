## Setup

1. Clone the repository and install the dependencies:

```bash
git clone https://github.com/bghira/SimpleTuner --branch release
python -m venv .venv
pip3 install -U poetry pip
poetry install --no-root
```

You will need to install some Linux-specific dependencies (Ubuntu is used here):

> ⚠️ This command can break certain container deployments. If it does, you'll have to redeploy the container.

```bash
apt -y install nvidia-cuda-dev nvidia-cuda-toolkit
```

If you get an error about missing cudNN library, you will want to install torch manually (replace 118 with your CUDA version if not using 11.8):

```bash
pip3 install xformers torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --force
```

Alternatively, Pytorch Nightly may be used (currently Torch 2.3) with Xformers 0.0.21dev (note that this includes torchtriton now):

```bash
pip3 install --pre torch torchvision torchaudio torchtriton --extra-index-url https://download.pytorch.org/whl/nightly/cu118 --force
pip3 install --pre git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

If the egg install for Xformers does not work, try including `xformers` on the first line, and run only that:

```bash
pip3 install --pre xformers torch torchvision torchaudio torchtriton --extra-index-url https://download.pytorch.org/whl/nightly/cu118 --force
```

2. For SD2.1, copy `sd21-env.sh.example` to `env.sh` - be sure to fill out the details. Try to change as little as possible.

For SDXL, copy `sdxl-env.sh.example` to `sdxl-env.sh` and then fill in the details.

For both training scripts, any missing values from your user config will fallback to the defaults.

3. If you are using `--report_to='wandb'` (the default), the following will help you report your statistics:

```bash
wandb login
```

Follow the instructions that are printed, to locate your API key and configure it.

Once that is done, any of your training sessions and validation data will be available on Weights & Biases.

4. For SD2.1, run the `training.sh` script, probably by redirecting the output to a log file:

```bash
bash training.sh > /path/to/training-$(date +%s).log 2>&1
```

For SDXL, run the `train_sdxl.sh` script, redirecting outputs to the log file:

```bash
bash train_sdxl.sh > /path/to/training-$(date +%s).log 2>&1
```

> ⚠️ At this point, the commands will work, but further configuration is required. See [the tutorial](/TUTORIAL.md) for more information.