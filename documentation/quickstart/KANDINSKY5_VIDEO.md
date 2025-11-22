# Kandinsky 5.0 Video Quickstart

In this example, we'll train a Kandinsky 5.0 Video LoRA (Lite/Pro SFT or pretrain) using the HunyuanVideo VAE and Qwen2.5-VL + CLIP dual text encoders.

## Notes on this model
- Uses Qwen2.5-VL (7B) for text; plan for enough system RAM or precompute embeddings if memory is constrained.
- Video variants use the HunyuanVideo 3D VAE (temporal compression 4, spatial compression 8).
- I2V flavours require first-frame conditioning: the pipeline encodes the input image into the visual_cond channels and sets a mask on the first frame.
- Default schedules are flow-matching (FlowMatch Euler) with guidance around 5.0 in released configs.

## Prerequisites
- Python 3.10–3.12 is recommended.
- Install CUDA toolkit or ROCm extras as needed for your environment.

## Installation

```bash
pip install simpletuner[cuda]  # or [apple]/[rocm] as appropriate
```

See [INSTALL.md](/documentation/INSTALL.md) for development setups.

## Configuration highlights
Use `simpletuner configure` or edit `config/config.json`.

- `model_type`: `lora`
- `model_family`: `kandinsky5-video`
- `model_flavour`: choose one of (SFT/pretrain only):
  - Lite 5s/10s: `t2v-lite-sft-5s`, `t2v-lite-pretrain-5s`, `t2v-lite-sft-10s`, `t2v-lite-pretrain-10s`
  - Pro: `t2v-pro-sft-5s-hd`, `t2v-pro-pretrain-5s-hd`, `t2v-pro-sft-10s-hd`, `t2v-pro-pretrain-10s-hd`
- `pretrained_model_name_or_path`: set to the corresponding HF repo above if overriding the flavour.
- `train_batch_size`: start at 1 for video; increase cautiously based on VRAM.
- `validation_num_video_frames`: use values divisible by 4 plus 1 (e.g., 121 for 5s, 241 for 10s) to respect the VAE temporal stride.
- `validation_resolution`: e.g., `512x768` or other 16-divisible pairs supported by the checkpoint.
- `validation_guidance`: around 5.0 matches released defaults.
- I2V runs: include conditioning images and ensure conditioning latents are passed; the pipeline handles first-frame masks.

### Text encoder considerations
- Qwen2.5-VL is large; precompute embeddings or use quantization if you need to reduce memory pressure. Keep both Qwen and CLIP encoders available for positive and negative prompts.

### Offloading (optional)
Group offloading can help if VRAM is tight:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1
```

Avoid combining with `--enable_model_cpu_offload`.

## Running the trainer
Start the WebUI:

```bash
simpletuner server
```

Or run via CLI once config/dataloader are ready:

```bash
simpletuner train --config config/config.json
```

Monitor validation videos; adjust learning rate, rank, and guidance if you observe quality regressions.
