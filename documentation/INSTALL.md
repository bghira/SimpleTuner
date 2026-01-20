# Setup

For users that wish to make use of Docker or another container orchestration platform, see [this document](DOCKER.md) first.

## Installation

For  users operating on Windows 10 or newer, an installation guide based on Docker and WSL is available here [this document](DOCKER.md).

### Pip installation method

You can simply install SimpleTuner using pip, which is recommended for most users:

```bash
# for CUDA
pip install 'simpletuner[cuda]'
# for CUDA 13 / Blackwell (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
# for ROCm
pip install 'simpletuner[rocm]'
# for Apple Silicon
pip install 'simpletuner[apple]'
# for CPU-only (not recommended)
pip install 'simpletuner[cpu]'
# for JPEG XL support (optional)
pip install 'simpletuner[jxl]'

# development requirements (optional, only for submitting PRs or running tests)
pip install 'simpletuner[dev]'
```

### Git repository method

For local development or testing, you can clone the SimpleTuner repository and set up the python venv:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

# if python --version shows 3.11 or 3.12, you may want to upgrade to 3.13.
python3.13 -m venv .venv

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

> ℹ️ **ROCm acceleration defaults**: When SimpleTuner detects a HIP-enabled PyTorch build it automatically exports `PYTORCH_TUNABLEOP_ENABLED=1` (unless you already set it) so TunableOp kernels are available. On MI300/gfx94x devices we also set `HIPBLASLT_ALLOW_TF32=1` by default, enabling hipBLASLt’s TF32 paths without requiring manual environment tweaks.

### All platforms

- 2a. **Option One (Recommended)**: Run `simpletuner configure`
- 2b. **Option Two**: Copy `config/config.json.example` to `config/config.json` and then fill in the details.

> ⚠️ For users located in countries where Hugging Face Hub is not readily accessible, you should add `HF_ENDPOINT=https://hf-mirror.com` to your `~/.bashrc` or `~/.zshrc` depending on which `$SHELL` your system uses.

#### Multiple GPU training

SimpleTuner now includes **automatic GPU detection and configuration** through the WebUI. Upon first load, you'll be guided through an onboarding step that detects your GPUs and configures Accelerate automatically.

##### WebUI Auto-Detection (Recommended)

When you first launch the WebUI or use `simpletuner configure`, you'll encounter an "Accelerate GPU Defaults" onboarding step that:

1. **Automatically detects** all available GPUs on your system
2. **Shows GPU details** including name, memory, and device IDs
3. **Recommends optimal settings** for multi-GPU training
4. **Offers three configuration modes:**

   - **Auto Mode** (Recommended): Uses all detected GPUs with optimal process count
   - **Manual Mode**: Select specific GPUs or set a custom process count
   - **Disabled Mode**: Single GPU training only

**How it works:**
- The system detects your GPU hardware via CUDA/ROCm
- Calculates optimal `--num_processes` based on available devices
- Sets `CUDA_VISIBLE_DEVICES` automatically when specific GPUs are selected
- Saves your preferences for future training runs

##### Manual Configuration

If not using the WebUI, you can control GPU visibility directly in your `config.json`:

```json
{
  "accelerate_visible_devices": [0, 1, 2],
  "num_processes": 3
}
```

This will restrict training to GPUs 0, 1, and 2, launching 3 processes.

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

> ⚠️ At this point, if you used `simpletuner configure`, you are done! If not - these commands will work, but further configuration is required. See [the tutorial](TUTORIAL.md) for more information.

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

## Training Data

A publicly-available dataset is available [on Hugging Face Hub](https://huggingface.co/datasets/bghira/pseudo-camera-10k) with approximately 10k images with captions as filenames, ready for use with SimpleTuner.

You can organize images in a single folder or neatly organize them into subdirectories.

### Image Selection Guidelines

**Quality Requirements:**
- No JPEG artifacts or blurry images - modern models will pick these up
- Avoid grainy CMOS sensor noise (will appear in all generated images)
- No watermarks, badges, or signatures (these will be learned)
- Movie frames generally don't work due to compression (use production stills instead)

**Technical Specifications:**
- Images optimally divisible by 64 (allows reuse without resizing)
- Mix square and non-square images for balanced capabilities
- Use varied, high-quality datasets for best results

### Captioning

SimpleTuner provides [captioning scripts](/scripts/toolkit/README.md) for mass-renaming files. Caption formats supported:
- Filename as caption (default)
- Text files with `--caption_strategy=textfile`
- JSONL, CSV, or advanced metadata files

**Recommended captioning tools:**
- **InternVL2**: Best quality but slow (small datasets)
- **BLIP3**: Best lightweight option with good instruction following
- **Florence2**: Fastest but some dislike outputs

### Training Batch Size

Your maximum batch size depends on VRAM and resolution:
```
vram use = batch size * resolution + base_requirements
```

**Key principles:**
- Use highest batch size possible without VRAM issues
- Higher resolution = more VRAM = lower batch size
- If batch size 1 at 128x128 doesn't work, hardware is insufficient

#### Multi-GPU Dataset Requirements

When training with multiple GPUs, your dataset must be large enough for the **effective batch size**:
```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

**Example:** With 4 GPUs and `train_batch_size=4`, you need at least 16 samples per aspect bucket.

**Solutions for small datasets:**
- Use `--allow_dataset_oversubscription` to auto-adjust repeats
- Manually set `repeats` in your dataloader config
- Reduce batch size or GPU count

See [DATALOADER.md](DATALOADER.md#multi-gpu-training-and-dataset-sizing) for complete details.

## Publishing to Hugging Face Hub

To automatically push models to Hub upon completion, add to `config/config.json`:

```json
{
  "push_to_hub": true,
  "hub_model_name": "your-model-name"
}
```

Login before training:
```bash
huggingface-cli login
```

## Debugging

Enable detailed logging by adding to `config/config.env`:

```bash
export SIMPLETUNER_LOG_LEVEL=DEBUG
export SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG
```

A `debug.log` file will be created in the project root with all log entries.
