## Setup

For users that wish to make use of Docker or another container orchestration platform, see [this document](/documentation/DOCKER.md) first.

### Installation

For  users operating on Windows 10 or newer, an installation guide based on Docker and WSL is available here [this document](/documentation/DOCKER.md).

Clone the SimpleTuner repository and set up the python venv:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

# if python --version shows 3.11 you can just also use the 'python' command here.
python3.11 -m venv .venv

source .venv/bin/activate
```

> ℹ️ You can use your own custom venv path by setting `export VENV_PATH=/path/to/.venv` in your `config/config.env` file.

**Note:** We're currently installing the `release` branch here; the `main` branch may contain experimental features that might have better results or lower memory use.

Install SimpleTuner with automatic platform detection:

```bash
# Basic installation (auto-detects CUDA/ROCm/Apple)
pip install -e .

# With JPEG XL support
pip install -e .[jxl]
```

**Note:** The setup.py automatically detects your platform (CUDA/ROCm/Apple) and installs the appropriate dependencies.

#### NVIDIA Hopper / Blackwell follow-up steps

Optionally, Hopper (or newer) equipment can make use of FlashAttention3 for improved inference and training performance when making use of `torch.compile`

You'll need to run the following sequence of commands from your SimpleTuner directory, with your venv active:

```bash
git clone https://github.com/Dao-AILab/flash-attention
pushd flash-attention
  pushd hopper
    python setup.py install
  popd
popd
```

> ⚠️ Managing the flash_attn build is poorly-supported in SimpleTuner, currently. This can break on updates, requiring you to re-run this build procedure manually from time-to-time.

#### AMD ROCm follow-up steps

The following must be executed for an AMD MI300X to be useable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
  python3 -m pip install --upgrade pip
  python3 -m pip install .
popd
```

### All platforms

- 2a. **Option One (Recommended)**: Run `simpletuner configure`
- 2b. **Option Two**: Copy `config/config.json.example` to `config/config.json` and then fill in the details.

> ⚠️ For users located in countries where Hugging Face Hub is not readily accessible, you should add `HF_ENDPOINT=https://hf-mirror.com` to your `~/.bashrc` or `~/.zshrc` depending on which `$SHELL` your system uses.

#### Multiple GPU training

**Note**: For MultiGPU setup, you will have to set all of these variables in `config/config.env`

```bash
TRAINING_NUM_PROCESSES=1
TRAINING_NUM_MACHINES=1
TRAINING_DYNAMO_BACKEND='no'
# this is auto-detected, and not necessary. but can be set explicitly.
CONFIG_BACKEND='json'
```

Any missing values from your user config will fallback to the defaults.

3. If you are using `--report_to='wandb'` (the default), the following will help you report your statistics:

```bash
wandb login
```

Follow the instructions that are printed, to locate your API key and configure it.

Once that is done, any of your training sessions and validation data will be available on Weights & Biases.

> ℹ️ If you would like to disable Weights & Biases or Tensorboard reporting entirely, use `--report-to=none`


4. Launch training with simpletuner; logs will be written to `debug.log`

```bash
simpletuner train
```

> ⚠️ At this point, if you used `simpletuner configure`, you are done! If not - these commands will work, but further configuration is required. See [the tutorial](/documentation/TUTORIAL.md) for more information.

### Run unit tests

To run unit tests to ensure that installation has completed successfully:

```bash
python -m unittest discover tests/
```

## Advanced: Multiple configuration environments

For users who train multiple models or need to quickly switch between different datasets or settings, two environment variables are inspected at startup.

To use them:

```bash
simpletuner train env=default config_backend=env
```

- `env` will default to `default`, which points to the typical `SimpleTuner/config/` directory that this guide helped you configure
  - Using `simpletuner train env=pixart` would use `SimpleTuner/config/pixart` directory to find `config.env`
- `config_backend` will default to `env`, which uses the typical `config.env` file this guide helped you configure
  - Supported options: `env`, `json`, `toml`, or `cmd` if you rely on running `train.py` manually
  - Using `simpletuner train config_backend=json` would search for `SimpleTuner/config/config.json` instead of `config.env`
  - Similarly, `config_backend=toml` will use `config.env`

You can create `config/config.env` that contains one or both of these values:

```bash
ENV=default
CONFIG_BACKEND=json
```

They will be remembered upon subsequent runs. Note that these can be added in addition to the multiGPU options described [above](#multiple-gpu-training).
