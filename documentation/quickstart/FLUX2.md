# FLUX.2 Quickstart

This guide covers training LoRAs on FLUX.2-dev, Black Forest Labs' latest image generation model featuring a Mistral-3 text encoder.

## Model Overview

FLUX.2-dev introduces significant architectural changes from FLUX.1:

- **Text Encoder**: Mistral-Small-3.1-24B instead of CLIP+T5
- **Architecture**: 8 DoubleStreamBlocks + 48 SingleStreamBlocks
- **Latent Channels**: 32 VAE channels → 128 after pixel shuffle (vs 16 in FLUX.1)
- **VAE**: Custom VAE with batch normalization and pixel shuffling
- **Embedding Dimension**: 15,360 (stacked from layers 10, 20, 30 of Mistral)

## Hardware Requirements

FLUX.2 has significant resource requirements due to the Mistral-3 text encoder:

### VRAM Requirements

The 24B Mistral text encoder alone requires significant VRAM:

| Component | bf16 | int8 | int4 |
|-----------|------|------|------|
| Mistral-3 (24B) | ~48GB | ~24GB | ~12GB |
| FLUX.2 Transformer | ~24GB | ~12GB | ~6GB |
| VAE + overhead | ~4GB | ~4GB | ~4GB |

| Configuration | Approximate Total VRAM |
|--------------|------------------------|
| bf16 everything | ~76GB+ |
| int8 text encoder + bf16 transformer | ~52GB |
| int8 everything | ~40GB |
| int4 text encoder + int8 transformer | ~22GB |

### System RAM

- **Minimum**: 96GB system RAM (loading 24B text encoder requires substantial memory)
- **Recommended**: 128GB+ for comfortable operation

### Recommended Hardware

- **Minimum**: 2x 48GB GPUs (A6000, L40S) with FSDP2 or DeepSpeed
- **Recommended**: 4x H100 80GB with fp8-torchao
- **With heavy quantization (int4)**: 2x 24GB GPUs may work but is experimental

Multi-GPU distributed training (FSDP2 or DeepSpeed) is essentially required for FLUX.2 due to the combined size of the Mistral-3 text encoder and transformer.

## Prerequisites

### Python Version

FLUX.2 requires Python 3.10 or later with recent transformers:

```bash
python --version  # Should be 3.10+
pip install transformers>=4.45.0
```

### Model Access

FLUX.2-dev requires access approval on Hugging Face:

1. Visit [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)
2. Accept the license agreement
3. Ensure you're logged in to Hugging Face CLI

## Installation

```bash
pip install simpletuner[cuda]
```

For development setup:
```bash
git clone https://github.com/bghira/SimpleTuner
cd SimpleTuner
pip install -e ".[cuda]"
```

## Configuration

### Web Interface

```bash
simpletuner server
```

Access http://localhost:8001 and select FLUX.2 as the model family.

### Manual Configuration

Create `config/config.json`:

<details>
<summary>View example config</summary>

```json
{
  "model_type": "lora",
  "model_family": "flux2",
  "model_flavour": "dev",
  "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-dev",
  "output_dir": "/path/to/output",
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "gradient_checkpointing": true,
  "mixed_precision": "bf16",
  "learning_rate": 1e-4,
  "lr_scheduler": "constant",
  "max_train_steps": 10000,
  "validation_resolution": "1024x1024",
  "validation_num_inference_steps": 20,
  "flux_guidance_mode": "constant",
  "flux_guidance_value": 1.0,
  "lora_rank": 16
}
```
</details>

### Key Configuration Options

#### Guidance Configuration

FLUX.2 uses guidance embedding similar to FLUX.1:

<details>
<summary>View example config</summary>

```json
{
  "flux_guidance_mode": "constant",
  "flux_guidance_value": 1.0
}
```
</details>

Or for random guidance during training:

<details>
<summary>View example config</summary>

```json
{
  "flux_guidance_mode": "random-range",
  "flux_guidance_min": 1.0,
  "flux_guidance_max": 5.0
}
```
</details>

#### Quantization (Memory Optimization)

For reduced VRAM usage:

<details>
<summary>View example config</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "int8-quanto",
  "base_model_default_dtype": "bf16"
}
```
</details>

#### TREAD (Training Acceleration)

FLUX.2 supports TREAD for faster training:

<details>
<summary>View example config</summary>

```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.5, "start_layer_idx": 2, "end_layer_idx": -2}
    ]
  }
}
```
</details>

### Advanced Experimental Features

<details>
<summary>Show advanced experimental details</summary>


SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> ⚠️ These features increase the computational overhead of training.

</details>

### Dataset Configuration

Create `config/multidatabackend.json`:

<details>
<summary>View example config</summary>

```json
[
  {
    "id": "my-dataset",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux2/my-dataset",
    "instance_data_dir": "datasets/my-dataset",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux2",
    "write_batch_size": 64
  }
]
```
</details>

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

### LoRA Targets

Available LoRA target presets:

- `all` (default): All attention and MLP layers
- `attention`: Only attention layers (qkv, proj)
- `mlp`: Only MLP/feed-forward layers
- `tiny`: Minimal training (just qkv layers)

<details>
<summary>View example config</summary>

```json
{
  "--flux_lora_target": "all"
}
```
</details>

## Training

### Login to Services

```bash
huggingface-cli login
wandb login  # optional
```

### Start Training

```bash
simpletuner train
```

Or via script:

```bash
./train.sh
```

### Memory Offloading

For memory-constrained setups, FLUX.2 supports group offloading for both the transformer and optionally the Mistral-3 text encoder:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
--group_offload_text_encoder
```

The `--group_offload_text_encoder` flag is recommended for FLUX.2 since the 24B Mistral text encoder benefits significantly from offloading during text embedding caching. You can also add `--group_offload_vae` to include the VAE in offloading during latent caching.

## Validation Prompts

Create `config/user_prompt_library.json`:

<details>
<summary>View example config</summary>

```json
{
  "portrait_subject": "a professional portrait photograph of <subject>, studio lighting, high detail",
  "artistic_subject": "an artistic interpretation of <subject> in the style of renaissance painting",
  "cinematic_subject": "a cinematic shot of <subject>, dramatic lighting, film grain"
}
```
</details>

## Inference

### Using Trained LoRA

FLUX.2 LoRAs can be loaded with the SimpleTuner inference pipeline or compatible tools once community support develops.

### Guidance Scale

- Training with `flux_guidance_value=1.0` works well for most use cases
- At inference, use normal guidance values (3.0-5.0)

## Differences from FLUX.1

| Aspect | FLUX.1 | FLUX.2 |
|--------|--------|--------|
| Text Encoder | CLIP-L/14 + T5-XXL | Mistral-Small-3.1-24B |
| Embedding Dim | CLIP: 768, T5: 4096 | 15,360 (3×5,120) |
| Latent Channels | 16 | 32 (→128 after pixel shuffle) |
| VAE | AutoencoderKL | Custom (BatchNorm) |
| VAE Scale Factor | 8 | 16 (8×2 pixel shuffle) |
| Transformer Blocks | 19 joint + 38 single | 8 double + 48 single |

## Troubleshooting

### Out of Memory During Startup

- Enable `--offload_during_startup=true`
- Use `--quantize_via=cpu` for text encoder quantization
- Reduce `--vae_batch_size`

### Slow Text Embedding

Mistral-3 is large; consider:
- Pre-caching all text embeddings before training
- Using text encoder quantization
- Batch processing with larger `write_batch_size`

### Training Instability

- Lower learning rate (try 5e-5)
- Increase gradient accumulation steps
- Enable gradient checkpointing
- Use `--max_grad_norm=1.0`

### CUDA Out of Memory

- Enable quantization (`int8-quanto` or `int4-quanto`)
- Enable gradient checkpointing
- Reduce batch size
- Enable group offloading
- Use TREAD for token routing efficiency

## Advanced: TREAD Configuration

TREAD (Token Routing for Efficient Architecture-agnostic Diffusion) speeds up training by selectively processing tokens:

<details>
<summary>View example config</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 4,
        "end_layer_idx": -4
      }
    ]
  }
}
```
</details>

- `selection_ratio`: Fraction of tokens to keep (0.5 = 50%)
- `start_layer_idx`: First layer to apply routing
- `end_layer_idx`: Last layer (negative = from end)

Expected speedup: 20-40% depending on configuration.

## See Also

- [FLUX.1 Quickstart](FLUX.md) - For FLUX.1 training
- [TREAD Documentation](/documentation/TREAD.md) - Detailed TREAD configuration
- [LoRA Training Guide](/documentation/LORA.md) - General LoRA training tips
- [Dataloader Configuration](/documentation/DATALOADER.md) - Dataset setup
