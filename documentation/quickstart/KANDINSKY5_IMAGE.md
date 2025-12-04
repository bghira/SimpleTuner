# Kandinsky 5.0 Image Quickstart

In this example, we'll be training a Kandinsky 5.0 Image LoRA.

## Hardware requirements

Kandinsky 5.0 employs a **huge 7B parameter Qwen2.5-VL text encoder** in addition to a standard CLIP encoder and the Flux VAE. This places significant demand on both VRAM and System RAM.

Simply loading the Qwen encoder requires roughly **14GB** of memory on its own. When training a rank-16 LoRA with full gradient checkpointing:

- **24GB VRAM** is the comfortable minimum (RTX 3090/4090).
- **16GB VRAM** is possible but requires aggressive offloading and likely `int8` quantization of the base model.

You'll need:

- **System RAM**: At least 32GB, ideally 64GB, to handle the initial model load without crashing.
- **GPU**: NVIDIA RTX 3090 / 4090 or professional cards (A6000, A100, etc.).

### Memory offloading (recommended)

Given the size of the text encoder, you should almost certainly use grouped offloading if you are on consumer hardware. This offloads the transformer blocks to CPU memory when they are not actively being computed.

Add the following to your `config.json`:

<details>
<summary>View example config</summary>

```json
{
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "group_offload_blocks_per_group": 1,
  "group_offload_use_stream": true
}
```
</details>

- `--group_offload_use_stream`: Only works on CUDA devices.
- **Do not** combine this with `--enable_model_cpu_offload`.

Additionally, set `"offload_during_startup": true` in your `config.json` to reduce VRAM usage during the initialization and caching phase. This ensures the text encoder and VAE are not loaded simultaneously.

## Prerequisites

Make sure that you have python installed; SimpleTuner does well with 3.10 through 3.12.

You can check this by running:

```bash
python --version
```

If you don't have python 3.12 installed on Ubuntu, you can try the following:

```bash
apt -y install python3.12 python3.12-venv
```

## Installation

Install SimpleTuner via pip:

```bash
pip install simpletuner[cuda]
```

For manual installation or development setup, see the [installation documentation](/documentation/INSTALL.md).

## Setting up the environment

### Web interface method

The SimpleTuner WebUI makes setup fairly straightforward. To run the server:

```bash
simpletuner server
```

Access it at http://localhost:8001.

### Manual / command-line method

To run SimpleTuner via command-line tools, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

#### Configuration file

An experimental script, `configure.py`, may help you skip this section:

```bash
simpletuner configure
```

If you prefer to manually configure:

Copy `config/config.json.example` to `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

You will need to modify the following variables:

- `model_type`: `lora`
- `model_family`: `kandinsky5-image`
- `model_flavour`:
  - `t2i-lite-sft`: (Default) The standard SFT checkpoint. Best for fine-tuning styles/characters.
  - `t2i-lite-pretrain`: The pretrain checkpoint. Better for teaching entirely new concepts from scratch.
  - `i2i-lite-sft` / `i2i-lite-pretrain`: For image-to-image training. Requires conditioning images in your dataset.
- `output_dir`: Where to save your checkpoints.
- `train_batch_size`: Start with `1`.
- `gradient_accumulation_steps`: Use `1` or higher to simulate larger batches.
- `validation_resolution`: `1024x1024` is standard for this model.
- `validation_guidance`: `5.0` is the recommended default for Kandinsky 5.
- `flow_schedule_shift`: `1.0` is the default. Adjusting this changes how the model prioritizes details vs composition (see below).

#### Validation prompts

Inside `config/config.json` is the "primary validation prompt". You can also create a library of prompts in `config/user_prompt_library.json`:

<details>
<summary>View example config</summary>

```json
{
  "portrait": "A high quality portrait of a woman, cinematic lighting, 8k",
  "landscape": "A beautiful mountain landscape at sunset, oil painting style"
}
```
</details>

Enable it by adding this to your `config.json`:

<details>
<summary>View example config</summary>

```json
{
  "user_prompt_library": "config/user_prompt_library.json"
}
```
</details>

#### Flow schedule shifting

Kandinsky 5 is a flow-matching model. The `shift` parameter controls the noise distribution during training and inference.

- **Shift 1.0 (Default)**: Balanced training.
- **Lower Shift (< 1.0)**: Focuses training more on high-frequency details (texture, noise).
- **Higher Shift (> 1.0)**: Focuses training more on low-frequency details (composition, color, structure).

If your model learns styles well but fails on composition, try increasing the shift. If it learns composition but lacks texture, try decreasing it.

#### Quantised model training

You can reduce VRAM usage significantly by quantizing the transformer to 8-bit.

In `config.json`:

<details>
<summary>View example config</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "base_model_default_dtype": "bf16"
```
</details>

> **Note**: We do not recommend quantizing the text encoders (`no_change`) as Qwen2.5-VL is sensitive to quantization effects and is already the heaviest part of the pipeline.

### Advanced Experimental Features

<details>
<summary>Show advanced experimental details</summary>


SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> ⚠️ These features increase the computational overhead of training.

#### Dataset considerations

You will need a dataset configuration file, e.g., `config/multidatabackend.json`.

```json
[
  {
    "id": "my-image-dataset",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/my_images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "crop": true,
    "crop_aspect": "square",
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/kandinsky5",
    "disabled": false
  }
]
```

Then create your dataset directory:

```bash
mkdir -p datasets/my_images
</details>

# Copy your images and .txt caption files here
```

#### Login to WandB and Huggingface Hub

```bash
wandb login
huggingface-cli login
```

### Executing the training run

**Option 1 (Recommended):**

```bash
simpletuner train
```

**Option 2 (Legacy):**

```bash
./train.sh
```

## Notes & troubleshooting tips

### Lowest VRAM config

To run on 16GB or constrained 24GB setups:

1.  **Enable Group Offload**: `--enable_group_offload`.
2.  **Quantize Base Model**: Set `"base_model_precision": "int8-quanto"`.
3.  **Batch Size**: Keep it at `1`.

### Artifacts and "Burnt" images

If validation images look over-saturated or noisy ("burnt"):

- **Check Guidance**: Ensure `validation_guidance` is around `5.0`. Higher values (like 7.0+) often fry the image on this model.
- **Check Flow Shift**: Extreme `flow_schedule_shift` values can cause instability. Stick to `1.0` to start.
- **Learning Rate**: 1e-4 is standard for LoRA, but if you see artifacts, try lowering to 5e-5.

### TREAD training

Kandinsky 5 supports [TREAD](/documentation/TREAD.md) for faster training by dropping tokens.

Add to `config.json`:

<details>
<summary>View example config</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

This drops 50% of tokens in the middle layers, speeding up the transformer pass.
