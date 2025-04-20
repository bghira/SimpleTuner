# HiDream Inference with LoRA Support

This script allows you to run inference with the HiDream model, optionally applying a LoRA adapter.

## Features

- HiDream model inference
- Optional LoRA adapter support
- Configurable generation parameters
- Batch processing of prompts from text file
- Option for cfg_zero

## Installation

First install the required dependencies:

`pip install autoroot autorootcwd`

## Inference 

```bash
python infer_hidream_lora.py \
  --prompt prompts.txt \
  --adapter_path /path/to/your/adapter.safetensors \
  --lora_scale 0.9 \
  --output_dir outputs/lora_run
```