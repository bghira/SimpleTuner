# Mage-Flow Quickstart

In this example, we'll be training a Mage-Flow LoRA.

Mage-Flow is Microsoft's 4B rectified-flow image generation and editing family. It uses a native-resolution MMDiT transformer, Qwen3-VL text conditioning, and Mage-VAE, a 128-channel latent tokenizer with 16x downsampling. SimpleTuner supports the text-to-image checkpoints and the edit checkpoints under one model family.

## Hardware requirements

Mage-Flow is smaller than Flux.1 and Qwen-Image, but it is still a transformer model with a large frozen Qwen3-VL text encoder. Treat full-resolution training as a high-memory GPU workload until you have measured your own dataset.

Good starting points:

- **bf16, 512px, batch 1** for smoke tests and caption/crop checks
- **bf16, 1024px, batch 1** for normal LoRA experiments on large GPUs
- **fp8wo-torchao**, 1024px, batch 1 when VRAM is tight on Ada/Hopper or newer NVIDIA GPUs
- **Turbo flavours**, 4 validation steps, when fast validation feedback matters

You will want:

- **minimum**: a 24GB NVIDIA GPU for reduced-resolution or quantised LoRA experiments
- **recommended**: 48GB or more for comfortable 1024px LoRA work
- **ideal**: 80GB-class cards for larger batches, edit training, and fewer memory compromises

Apple GPUs are not a recommended training target for Mage-Flow.

## Installation

Install SimpleTuner:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

For a development checkout, follow the [installation documentation](../INSTALL.md).

Mage-Flow uses packed variable-length attention. To use FlashAttention 2 without building the `flash-attn` package locally, set `"attention_mechanism": "flash-attn-varlen-hub"` so SimpleTuner loads the Hugging Face Hub kernel. Leave the default `diffusers` value for PyTorch SDPA.

## Setting up the environment

### Web interface method

The WebUI can create and edit the config:

```bash
simpletuner server
```

The server listens on port 8001 by default.

### Manual / command-line method

Copy an existing transformer LoRA config and edit the important values:

```bash
cp config/config.json.example config/config.json
```

For text-to-image LoRA training:

```json
{
  "model_family": "mageflow",
  "model_flavour": "base",
  "model_type": "lora",
  "pretrained_model_name_or_path": "microsoft/Mage-Flow-Base",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "optimizer": "optimi-lion",
  "learning_rate": 1e-4,
  "lora_rank": 32,
  "train_batch_size": 1,
  "resolution": 1024,
  "validation_resolution": "1024x1024",
  "validation_num_inference_steps": 30,
  "validation_guidance": 5.0
}
```

The supported `model_flavour` values are:

- `base` - `microsoft/Mage-Flow-Base`, text-to-image base checkpoint
- `default` - `microsoft/Mage-Flow`, RL-aligned text-to-image checkpoint
- `turbo` - `microsoft/Mage-Flow-Turbo`, 4-step text-to-image checkpoint
- `edit-base` - `microsoft/Mage-Flow-Edit-Base`, instruction edit base checkpoint
- `edit` - `microsoft/Mage-Flow-Edit`, RL-aligned instruction edit checkpoint
- `edit-turbo` - `microsoft/Mage-Flow-Edit-Turbo`, 4-step instruction edit checkpoint

For edit LoRA training, switch the flavour and model path:

```json
{
  "model_family": "mageflow",
  "model_flavour": "edit-turbo",
  "pretrained_model_name_or_path": "microsoft/Mage-Flow-Edit-Turbo",
  "validation_num_inference_steps": 4
}
```

## Mage Flow (Edit) Considerations

The Mage-Flow edit checkpoints do not require a conditioning or reference dataset. Microsoft trained the edit models jointly on generation and editing tasks, so the generative prior is preserved. In SimpleTuner you can keep using a normal image dataset for subject, style, or concept LoRA finetuning even when `model_flavour` is `edit-base`, `edit`, or `edit-turbo`.

Use paired source/target data only when you specifically want to train edit behavior. SimpleTuner routes edit flavours through the edit-capable pipeline automatically; when no conditioning image is provided, validation and prompt encoding use the text-to-image path.

## Dataloader configuration

For ordinary subject or style LoRAs, use the normal image dataloader:

```json
[
  {
    "id": "dreambooth-1024",
    "type": "local",
    "instance_data_dir": "/path/to/images",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "minimum_image_size": 128,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel",
    "metadata_backend": "discovery",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "cache_dir_vae": "cache/vae/mageflow/dreambooth-1024"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/mageflow"
  }
]
```

For optional edit-behavior training, use paired source/target data and configure conditioning images the same way you would for other edit models. The caption should describe the edit instruction, not just the target image. The conditioning image is encoded as Mage-VAE latents and appended to the noisy target sequence during training.

## Validation prompts

Mage-Flow supports long prompts and native resolutions from 512px to 2048px. Start with a small prompt set while tuning:

```json
{
  "validation_prompt": "a detailed studio portrait of <token>, soft directional light, natural skin texture",
  "validation_negative_prompt": "blurry, low quality, cropped",
  "validation_resolution": "1024x1024",
  "validation_guidance": 5.0,
  "validation_num_inference_steps": 4
}
```

Use about 20 steps for `default`, 30 steps for `base`, and 4 steps for `turbo` / `edit-turbo`.

## Memory presets

Mage-Flow exposes built-in RAMTorch and Musubi block swap presets in the memory optimisation menu. Use RAMTorch when you want CPU-resident transformer weights; use Musubi block swap when you want to stream only the last transformer blocks during forward and backward. These presets are mutually exclusive in the configurator.

## Quantisation notes

Start with `bf16` when it fits. If it does not, prefer FP8 weight-only TorchAO on GPUs with float8 support:

```json
{
  "base_model_precision": "fp8wo-torchao",
  "quantize_via": "cpu"
}
```

In Mage-Flow LoRA smoke tests, int8 quantisation produced suspicious loss spikes compared with FP8 weight-only TorchAO. Avoid int8 Mage-Flow presets unless you validate the loss curve on your dataset. NF4 and other SimpleTuner quantisation presets may also be useful for LoRA training. Keep the text encoder frozen; the initial cache pass is expensive but it should not be trained for normal LoRA runs.

## Implementation notes

SimpleTuner vendors the upstream MIT Mage-Flow transformer, VAE, and inference utilities, then wraps them in native `DiffusionPipeline` classes for validation and save-hook consistency. The model family is `mageflow`; the helper lives under `simpletuner/helpers/models/mageflow`.
