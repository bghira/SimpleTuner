# Chroma 1 Quickstart

![image](https://github.com/user-attachments/assets/3c8a12c6-9d45-4dd4-9fc8-6b7cd3ed51dd)

Chroma 1 is an 8.9B-parameter, trimmed variant of Flux.1 Schnell released by Lodestone Labs. This guide walks through configuring SimpleTuner for LoRA training.

## Hardware requirements

Despite the smaller parameter count, memory usage is close to Flux Schnell:

- Quantising the base transformer can still use **≈40–50 GB** of system RAM.
- Rank-16 LoRA training typically consumes:
  - ~28 GB VRAM without base quantisation
  - ~16 GB VRAM with int8 + bf16
  - ~11 GB VRAM with int4 + bf16
  - ~8 GB VRAM with NF4 + bf16
- Realistic GPU minimum: **RTX 3090 / RTX 4090 / L40S** class cards or better.
- Works well on **Apple M-series (MPS)** for LoRA training, and on AMD ROCm.
- 80 GB-class accelerators or multi-GPU setups are recommended for full-rank fine-tuning.

## Prerequisites

Chroma shares the same runtime expectations as the Flux guide:

- Python **3.10 – 3.12**
- A supported accelerator backend (CUDA, ROCm, or MPS)

Check your Python version:

```bash
python3 --version
```

Install SimpleTuner (CUDA example):

```bash
pip install simpletuner[cuda]
```

For backend-specific setup details (CUDA, ROCm, Apple), refer to the [installation guide](/documentation/INSTALL.md).

## Launching the web UI

```bash
simpletuner server
```

The UI will be available at http://localhost:8001.

## Configuration via CLI

`simpletuner configure` walks you through the core settings. The key values for Chroma are:

- `model_type`: `lora`
- `model_family`: `chroma`
- `model_flavour`: one of
  - `base` (default, balanced quality)
  - `hd` (higher fidelity, more compute hungry)
  - `flash` (fast but unstable – not recommended for production)
- `pretrained_model_name_or_path`: leave empty to use the flavour mapping above
- `model_precision`: keep default `bf16`
- `flux_fast_schedule`: leave **disabled**; Chroma has its own adaptive sampling

### Example manual configuration snippet

```jsonc
{
  "model_type": "lora",
  "model_family": "chroma",
  "model_flavour": "base",
  "output_dir": "/workspace/chroma-output",
  "network_rank": 16,
  "learning_rate": 2.0e-4,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "pretrained_model_name_or_path": null
}
```

> ⚠️ If Hugging Face access is slow in your region, export `HF_ENDPOINT=https://hf-mirror.com` before launching.

## Dataset & dataloader

Chroma uses the same dataloader format as Flux. Refer to the [general tutorial](/documentation/TUTORIAL.md) or the [web UI tutorial](/documentation/webui/TUTORIAL.md) for dataset preparation and prompt libraries.

## Training options specific to Chroma

- `flux_lora_target`: controls which transformer modules receive LoRA adapters (`all`, `all+ffs`, `context`, `tiny`, etc.). The defaults mirror Flux and work well for most cases.
- `flux_guidance_mode`: `constant` or `random-range`. Guidance embeds are optional for Chroma but can stabilise validation renders.
- `flux_attention_masked_training`: enable if your text embeds were cached with padding masks; disables when unset.
- `flow_schedule_shift`/`flow_schedule_auto_shift`: available but usually **not** required—Chroma applies a quadratic tail boost internally.
- `flux_t5_padding`: set to `zero` if you prefer to zero padded tokens before masking.

## Automatic tail timestep sampling

Flux used a log-normal schedule that under-sampled high-noise / low-noise extremes. Chroma’s training helper applies a quadratic (`σ ↦ σ²` / `1-(1-σ)²`) remapping to the sampled sigmas so tail regions are visited more often. This requires **no extra configuration**—it is built into the `chroma` model family.

## Validation & sampling tips

- `validation_guidance_real` maps directly to the pipeline’s `guidance_scale`. Leave it at `1.0` for single-pass sampling, or raise it to `2.0`–`3.0` if you want classifier-free guidance during validation renders.
- Use 20 inference steps for quick previews; 28–32 for higher quality.
- Negative prompts remain optional; the base model is already de-distilled.
- The model only supports text-to-image at the moment; img2img support will arrive in a later update.

## Troubleshooting

- **OOM at startup**: enable `offload_during_startup` or quantise the base model (`base_model_precision: int8-quanto`).
- **Training diverges early**: ensure gradient checkpointing is on, lower `learning_rate` to `1e-4`, and verify captions are diverse.
- **Validation repeats the same pose**: lengthen prompts; flow-matching models collapse when prompt variety is low.

For advanced topics—DeepSpeed, FSDP2, TREAD, evaluation metrics—see the shared guides linked throughout the README.
