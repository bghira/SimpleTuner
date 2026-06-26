# Krea2 Quickstart

In this example, we'll be training a Krea2 LoRA.

Krea2 is a large flow-matching image transformer using Qwen-style text conditioning and the Qwen Image VAE. In SimpleTuner, the default target is PEFT LoRA training on the transformer. The text encoder and VAE are used for caching and validation, then moved out of the way before the training loop.

The starter example lives here:

```bash
simpletuner/examples/krea2.peft-lora/config.json
```

## Hardware requirements

Krea2 is much heavier than SDXL-style UNets. It is happiest on a high-memory CUDA GPU, and the 1024px configuration should be treated as an H100/A100-80G/L40S-class workload unless you are using quantisation or offload.

On a single H100 80GB, the practical starting points are:

- **bf16, 512px, batch 1** for fast smoke tests and dataset checks
- **bf16, 1024px, batch 1** for a realistic full-resolution run
- **int8-torchao, 1024px, batch 1-4** when VRAM headroom is more important than step speed
- **compile reduce-overhead** only after the uncompiled run is stable, because compile can add a large amount of VRAM

You will need:

- **the realistic minimum**: an NVIDIA GPU with at least 24GB VRAM for reduced-resolution or quantised experiments
- **recommended**: 48GB or more for comfortable 512px work
- **ideal**: 80GB H100/A100-class cards for 1024px and compile experiments

Apple GPUs are not currently a recommended target for Krea2 training.

## Prerequisites

Make sure Python is installed. SimpleTuner supports Python 3.10 through 3.13.

```bash
python --version
```

If Python 3.13 is not installed on Ubuntu:

```bash
apt -y install python3.13 python3.13-venv
```

### Container image dependencies

For Vast, RunPod, TensorDock, and similar CUDA images, install CUDA toolkit headers when you need packages that compile CUDA extensions:

```bash
apt -y install nvidia-cuda-toolkit
```

CUDA 13 users should use an image whose CUDA runtime, CUDA headers, NVRTC, and PyTorch wheel are from the same generation. Transformer-style training paths can become very sensitive to mismatched CUDA packages.

## Installation

Install SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

For a development checkout, follow the [installation documentation](../INSTALL.md).

## Setting up the environment

### Web interface method

The SimpleTuner WebUI can create and edit the config for you:

```bash
simpletuner server
```

The server listens on port 8001 by default.

### Manual / command-line method

Copy the example config and edit it:

```bash
cp simpletuner/examples/krea2.peft-lora/config.json config/config.json
```

The important values are:

- `model_type` - set this to `lora`.
- `model_family` - set this to `krea2`.
- `model_flavour` - set this to `raw`.
- `pretrained_model_name_or_path` - set this to `krea/Krea-2-Raw`.
- `mixed_precision` - keep this at `bf16` on modern NVIDIA GPUs.
- `gradient_checkpointing` - keep this enabled unless you are deliberately measuring memory.
- `fuse_qkv_projections` - keep this enabled. Krea2 supports permanent QKV fusion for the attention projections, and the LoRA target changes to the fused projection.
- `train_batch_size` - start at 1. Increase after the run is stable.
- `resolution` - the top-level value is less important than the dataloader's own `resolution`; make sure the dataloader is actually set to the resolution you intend to test.
- `validation_resolution` - use `1024x1024` for full-resolution validation.
- `base_model_precision` - use `no_change` for bf16 or `int8-torchao` for TorchAO int8 weight-only training.
- `quantize_via` - use `cpu` for TorchAO int8 when startup GPU memory is tight.

A conservative bf16 starting point:

```json
{
  "model_family": "krea2",
  "model_flavour": "raw",
  "model_type": "lora",
  "pretrained_model_name_or_path": "krea/Krea-2-Raw",
  "base_model_precision": "no_change",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "fuse_qkv_projections": true,
  "optimizer": "optimi-lion",
  "learning_rate": 1e-4,
  "lora_rank": 64,
  "train_batch_size": 1,
  "resolution": 1024,
  "validation_resolution": "1024x1024"
}
```

For TorchAO int8:

```json
{
  "base_model_precision": "int8-torchao",
  "quantize_via": "cpu"
}
```

For reduce-overhead compile:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_mode": "reduce-overhead",
  "dynamo_use_regional_compilation": true
}
```

Compile should be considered a batch-size-1 performance option first. It can make larger batches OOM even when the same batch fits without compile.

## Dataloader configuration

Krea2 uses the same general image dataloader structure as the other image transformer models. The example config uses a small Domokun dataset:

```json
[
  {
    "id": "dreambooth-1024",
    "type": "huggingface",
    "dataset_name": "RareConcepts/Domokun",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "minimum_image_size": 128,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel",
    "metadata_backend": "huggingface",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "cache_dir_vae": "cache/vae/krea2/dreambooth-1024"
  },
  {
    "id": "alt-embed-cache",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/krea2"
  }
]
```

For your own dataset, switch `type` to `local`, set `instance_data_dir`, and choose a caption strategy. Subject LoRAs commonly start with `caption_strategy=instanceprompt`; style LoRAs usually do better with captions or filenames.

512px and 1024px datasets can both be useful. 512px runs are much faster and are good for catching bad captions, broken crops, or learning-rate mistakes. 1024px runs are the better signal for final quality.

## Validation prompts

Krea2 validation is expensive enough that it is worth keeping the prompt set small while tuning. Start with one or two prompts that clearly show whether the subject or style is being learned. Once the run is stable, add a prompt library.

Example:

```json
{
  "validation_prompt": "a studio portrait of <token>, soft directional light, detailed fabric texture",
  "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
  "validation_num_inference_steps": 28,
  "validation_guidance": 4.5,
  "validation_resolution": "1024x1024"
}
```

Krea2 can overfit small datasets quickly. Do not rely on a single validation prompt; a small prompt library makes collapse or prompt memorisation much easier to spot.

## Reference image training

Krea2 supports optional reference-latent conditioning for edit-style datasets:

```json
{
  "krea2_reference_latents": true
}
```

This mode expects paired reference data. The reference latents must match the target latent shape. It is intended for Qwen Edit-style paired conditioning, not for generic SDEdit/img2img training.

Use this only when the dataset is built around paired examples. For ordinary subject/style LoRA training, leave it disabled.

## Quantisation notes

`int8-torchao` stores the base transformer weights in int8 and trains bf16 LoRA weights on top. On H100 it reduced peak VRAM substantially, but the tested path was slower than bf16 at the same resolution and batch size.

That tradeoff is still useful:

- use bf16 when the run fits and speed matters
- use int8 when the run otherwise does not fit
- expect int8 startup to take longer because the model is quantised before training
- remeasure if you change PyTorch, TorchAO, CUDA, or the attention backend

## Performance notes

The following results were measured on a single NVIDIA H100 80GB using the real SimpleTuner trainer, Krea2 LoRA, fused QKV projections, gradient checkpointing, and a small Domokun dataset. VRAM was sampled externally with `nvidia-smi`.

These numbers are not hardware guarantees. Treat them as comparative data showing how this recipe behaved on one H100 system. Different drivers, CUDA builds, PyTorch builds, dataloaders, optimizers, LoRA ranks, and attention backends can move the numbers.

### Fused QKV + checkpointing, compile off

| Precision | Resolution | Batch | Steady s/step | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 512 | 1 | 0.353 | 31.10 GiB |
| bf16 | 512 | 4 | 1.230 | 39.31 GiB |
| bf16 | 512 | 8 | 2.430 | 50.32 GiB |
| bf16 | 1024 | 1 | 0.990 | 33.28 GiB |
| bf16 | 1024 | 4 | 3.850 | 48.35 GiB |
| bf16 | 1024 | 8 | 7.690 | 67.88 GiB |
| int8-torchao | 512 | 1 | 0.535 | 18.10 GiB |
| int8-torchao | 512 | 4 | 1.690 | 27.46 GiB |
| int8-torchao | 512 | 8 | 3.220 | 40.52 GiB |
| int8-torchao | 1024 | 1 | 1.330 | 20.35 GiB |
| int8-torchao | 1024 | 4 | 4.850 | 36.99 GiB |
| int8-torchao | 1024 | 8 | 9.520 | 58.84 GiB |

### Fused QKV + checkpointing + reduce-overhead compile

| Precision | Resolution | Batch | Status | Steady s/step | Peak VRAM |
| --- | ---: | ---: | --- | ---: | ---: |
| bf16 | 512 | 1 | ok | 0.260 | 41.20 GiB |
| bf16 | 512 | 4 | OOM | - | 79.07 GiB |
| bf16 | 512 | 8 | OOM | - | 79.10 GiB |
| bf16 | 1024 | 1 | ok | 0.704 | 63.71 GiB |
| bf16 | 1024 | 4 | OOM | - | 79.11 GiB |
| bf16 | 1024 | 8 | OOM | - | 78.40 GiB |
| int8-torchao | 512 | 1 | ok | 0.410 | 30.93 GiB |
| int8-torchao | 512 | 4 | ok | 1.300 | 78.60 GiB |
| int8-torchao | 512 | 8 | OOM | - | 79.12 GiB |
| int8-torchao | 1024 | 1 | ok | 0.990 | 58.68 GiB |
| int8-torchao | 1024 | 4 | OOM | - | 78.92 GiB |
| int8-torchao | 1024 | 8 | OOM | - | 78.09 GiB |

The useful conclusion is straightforward: on this H100, compile was a strong batch-size-1 speedup, but it increased VRAM enough to make most larger batches fail. Uncompiled bf16 was the best general-purpose choice when the model fit. Int8 was the memory-saving choice, not the speed choice.

## Executing the training run

From the SimpleTuner directory:

```bash
simpletuner train
```

or, from a development checkout:

```bash
.venv/bin/python -m simpletuner.cli train
```

Training begins by caching text embeddings and VAE latents. If you change captions, image resolution, crop settings, or reference-image settings, clear the relevant cache or use a new cache directory.

## Troubleshooting

### The run is using 512px even though `resolution` says 1024

The dataloader has its own `resolution`, `maximum_image_size`, and `target_downsample_size`. Those values decide the actual training image size. Update the dataloader JSON, not just the top-level model config.

### Compile OOMs but uncompiled training fits

This is expected for Krea2 on 80GB cards. Compile reduce-overhead can reserve much more memory for graphs/cudagraphs. Lower the batch size or disable compile.

### Int8 uses less VRAM but trains slower

That matched the H100 measurements above. TorchAO int8 is useful for capacity; it is not automatically a throughput improvement for this training path.

### The validation image is not following the reference image

Confirm that `krea2_reference_latents=true` is enabled and that the validation dataset is using paired reference data. Plain validation prompts do not exercise the reference-latent path.

### The model overfits quickly

Use fewer steps, lower the learning rate, add more prompts for validation, or increase dataset variety. Krea2 is large enough to memorise small subject datasets quickly.
