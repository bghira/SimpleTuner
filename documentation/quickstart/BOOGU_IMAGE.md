# Boogu-Image 0.1 Quickstart

This guide covers LoRA and LyCORIS LoKr training for Boogu-Image 0.1 in SimpleTuner. Boogu-Image is a flow-matching image model with text-to-image, turbo, and edit flavours. The SimpleTuner integration uses local pipeline and transformer code, and the exported pipeline checkpoints are hosted under the `SimpleTuner` Hugging Face namespace.

The included starter configs are:

```bash
simpletuner/examples/boogu-image-v0.1.peft-lora/config.json
simpletuner/examples/boogu-image-v0.1.lycoris-lokr/config.json
```

## Hardware requirements

Boogu-Image should be treated like a large transformer image model. For first runs, use 1024px training, batch size 1, bf16 mixed precision, and gradient checkpointing.

Recommended starting points:

- **Best default:** `v0.1-base`, bf16 LoRA weights, rank 16.
- **Lower VRAM:** use an FP8 flavour such as `v0.1-base-fp8`, `v0.1-turbo-fp8`, or `v0.1-edit-fp8`.
- **Fast validation / inference target:** use the turbo flavour, noting the assistant LoRA status below.
- **Editing:** use `v0.1-edit` or `v0.1-edit-fp8` with paired conditioning data.

Observed memory depends on rank, optimiser, validation resolution, offload, compile settings, and whether FP8 weights are used. A single H100 can train the provided PEFT LoRA example for 1000 steps at 1024px with validation and benchmark samples enabled.

For smaller cards, start with FP8 weights, rank 8-16, `train_batch_size=1`, gradient checkpointing, and model or group offload.

### Memory offloading

Grouped module offloading can reduce VRAM pressure when the transformer weights are the bottleneck:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

Optional disk offload:

```bash
--group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams are only effective on CUDA; SimpleTuner disables them on ROCm, MPS, and CPU backends.
- Do not combine group offload with other CPU offload strategies.
- Prefer fast local NVMe when offloading to disk.

### Torch compile and attention

On NVIDIA GPUs, use the Hugging Face hub kernel attention aliases when available:

```json
{
  "attention_mechanism": "flash-attn-3-hub",
  "dynamo_backend": "inductor",
  "dynamo_use_regional_compilation": true
}
```

If compiled validation produces black images on a specific GPU or driver stack, disable torch compile first and re-test before changing the training recipe.

## Installation

Install SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

For manual installation or development setup, see the [installation documentation](../INSTALL.md).

## Setting up the environment

### Web interface method

The SimpleTuner WebUI can create a Boogu-Image training config. To run the server:

```bash
simpletuner server
```

Open http://localhost:8001 and choose `boogu_image` as the model family.

### Manual / command-line method

To run SimpleTuner from the command line, configure the trainer, dataset backend, and model flavour.

Copy `config/config.json.example` to `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Set or review these values:

- `model_type` - Set this to `lora`.
- `lora_type` - Use `standard` for PEFT LoRA or `lycoris` for LyCORIS LoKr.
- `model_family` - Set this to `boogu_image`.
- `model_flavour` - Choose one of:
  - `v0.1-base` - standard text-to-image training target.
  - `v0.1-base-fp8` - base model exported with FP8 weights.
  - `v0.1-turbo` - turbo text-to-image target.
  - `v0.1-turbo-fp8` - turbo model exported with FP8 weights.
  - `v0.1-edit` - image editing target.
  - `v0.1-edit-fp8` - edit model exported with FP8 weights.
- `pretrained_model_name_or_path` - Usually leave unset and let the flavour choose the `SimpleTuner/Boogu-Image-0.1-*` pipeline.
- `output_dir` - Set this to the directory where checkpoints and validation images should be stored.
- `train_batch_size` - Start with `1`.
- `resolution` - Start with `1024`.
- `resolution_type` - Use `pixel_area` for multi-aspect bucket training.
- `validation_resolution` - Use `1024x1024` for normal validation; multi-aspect values may be comma-separated.
- `validation_guidance` - Start around `4.0` for base/edit flavours.
- `validation_num_inference_steps` - Start around `30` for base/edit flavours. Turbo can use fewer steps.
- `mixed_precision` - Use `bf16` on modern NVIDIA GPUs.
- `gradient_checkpointing` - Keep this enabled.
- `flow_schedule_shift` - The examples use `3`.
- `base_model_precision` - Use `no_change` for the standard exported pipeline or an FP8 flavour for FP8 loading.

Minimal PEFT LoRA config:

```json
{
  "model_type": "lora",
  "model_family": "boogu_image",
  "model_flavour": "v0.1-base",
  "lora_type": "standard",
  "lora_rank": 16,
  "lora_alpha": 16,
  "output_dir": "output/models-boogu-image-v0.1",
  "train_batch_size": 1,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.0,
  "validation_num_inference_steps": 30,
  "validation_prompt": "a polished product photo of a ceramic mug on a walnut desk",
  "validation_steps": 50,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "flow_schedule_shift": 3,
  "optimizer": "adamw_bf16",
  "learning_rate": 1e-4,
  "lr_scheduler": "constant_with_warmup",
  "lr_warmup_steps": 10,
  "max_train_steps": 1000,
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "data_backend_config": "config/examples/multidatabackend-small-dreambooth-1024px.json"
}
```

## Running the examples

PEFT LoRA:

```bash
simpletuner train example=boogu-image-v0.1.peft-lora
```

LyCORIS LoKr:

```bash
simpletuner train example=boogu-image-v0.1.lycoris-lokr
```

Git checkout / development form:

```bash
simpletuner train env=examples/boogu-image-v0.1.peft-lora
simpletuner train env=examples/boogu-image-v0.1.lycoris-lokr
```

Legacy wrapper form:

```bash
ENV=examples/boogu-image-v0.1.peft-lora ./train.sh
ENV=examples/boogu-image-v0.1.lycoris-lokr ./train.sh
```

## FP8 flavours

Use the `-fp8` model flavours when you want the exported FP8 pipeline weights:

```json
{
  "model_family": "boogu_image",
  "model_flavour": "v0.1-base-fp8"
}
```

The same pattern applies to `v0.1-turbo-fp8` and `v0.1-edit-fp8`. These are flavour-level pipeline choices, so you do not need to point SimpleTuner at Boogu `.bin` files.

## Turbo assistant LoRA

SimpleTuner enables the assistant LoRA code path for `v0.1-turbo` and `v0.1-turbo-fp8`. The adapter path is currently a `None` placeholder because no separate assistant adapter path has been published for this integration yet.

Until that adapter exists, use turbo flavours as exported pipeline targets and validate output quality directly. For the most predictable training baseline, start with `v0.1-base`.

## Edit training

Boogu edit flavours require paired conditioning data. Use the same paired-reference dataset structure described in the [Qwen Image Edit quickstart](./QWEN_EDIT.md): the main dataset supplies target images and captions, and the conditioning dataset supplies the reference/edit input images.

For text-to-image LoRA runs, use the base or turbo flavours instead.

## Validation prompts

Inside `config/config.json`, `validation_prompt` is the primary validation prompt. For broader coverage, add a validation prompt library:

```json
{
  "product": "a polished product photo of <token> on a walnut desk",
  "studio": "a clean studio portrait of <token> with softbox lighting",
  "cinematic": "a cinematic scene featuring <token>, detailed lighting, shallow depth of field"
}
```

Point the trainer at it:

```json
{
  "validation_prompt_library": "config/user_prompt_library.json"
}
```

Use prompts that are distinct enough to reveal overfitting, prompt collapse, and style drift.

## Inference

After training, load the saved adapter with the matching Boogu-Image pipeline flavour. For SimpleTuner training outputs, the important file is usually:

```bash
output/models-boogu-image-v0.1/pytorch_lora_weights.safetensors
```

Use the same base flavour for inference that you used for training, especially when comparing validation samples across checkpoints.
