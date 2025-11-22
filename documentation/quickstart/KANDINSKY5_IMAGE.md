# Kandinsky 5.0 Image Quickstart

In this example, we'll train a Kandinsky 5.0 Image LoRA (Lite checkpoints) using the dual text encoders (Qwen2.5-VL + CLIP) and the Flux VAE.

## Notes on this model
- Uses the Qwen2.5-VL 7B vision-language encoder; ensure enough system RAM to host it or precompute text embeddings.
- Image variants use the Flux VAE (scaling factor 0.3611, shift baked into the VAE config).
- I2I flavours expect visual conditioning: the transformer is `visual_cond=true` and the pipeline appends encoded image latents plus a mask.

## Prerequisites
- Python 3.10–3.12 works well with SimpleTuner.
- Install system packages for your environment (CUDA toolkit on common cloud images, ROCm extras on AMD if needed).

## Installation

```bash
pip install simpletuner[cuda]  # or [apple]/[rocm] as appropriate
```

For development installs, see [INSTALL.md](/documentation/INSTALL.md).

## Configuration highlights
You can use `simpletuner configure` or edit `config/config.json` directly.

- `model_type`: `lora`
- `model_family`: `kandinsky5-image`
- `model_flavour`: one of:
  - `t2i-lite-sft` (default)
  - `t2i-lite-pretrain`
  - `i2i-lite-sft`
  - `i2i-lite-pretrain`
- `pretrained_model_name_or_path`: set to the corresponding HF repo above if overriding the flavour.
- `train_batch_size`: start with 1–2; raise only if memory allows.
- `validation_resolution`: e.g., `1024x1024` (divisible by 16).
- `validation_guidance`: use a typical CFG-style value around 5.0 to mirror released configs.
- I2I flavours: supply conditioning images and ensure conditioning latents are available in the dataloader; SimpleTuner will append the mask automatically.

### Text encoder considerations
- Qwen2.5-VL is large; if memory is tight, precompute embeddings via the Text Embedding Cache, or quantize the text encoder if your workflow allows it.
- Keep the dual encoders loaded (Qwen + CLIP) for both positive and negative prompts.

### Offloading (optional)
If GPU memory is tight, consider enabling grouped offload:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1
```

Do not combine with `--enable_model_cpu_offload`.

## Running the trainer
Launch the WebUI:

```bash
simpletuner server
```

Or run the CLI trainer once your config and dataloader are set:

```bash
simpletuner train --config config/config.json
```

Monitor validation images to confirm the LoRA is learning; adjust learning rate and rank if you see overfitting or collapse.
