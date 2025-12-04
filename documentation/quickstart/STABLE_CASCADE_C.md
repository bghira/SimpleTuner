# Stable Cascade Stage C Quickstart

This guide walks through configuring SimpleTuner to fine-tune the **Stable Cascade Stage C prior**. Stage C learns the text-to-image prior that feeds the Stage B/C decoder stack, so good training hygiene here directly improves the downstream decoder outputs. We'll focus on LoRA training, but the same steps apply to full fine-tunes if you have the VRAM to spare.

> **Heads-up:** Stage C uses the 1B+ parameter CLIP-G/14 text encoder and an EfficientNet-based autoencoder. Make sure torchvision is installed and expect large text-embed caches (roughly 5–6× larger per prompt than SDXL).

## Hardware Requirements

- **LoRA training:** 20–24 GB VRAM (RTX 3090/4090, A6000, etc.)
- **Full-model training:** 48 GB+ VRAM recommended (A6000, A100, H100). DeepSpeed/FSDP2 offload can lower the requirement but introduces complexity.
- **System RAM:** 32 GB recommended so the CLIP-G text encoder and caching threads do not starve.
- **Disk:** Allocate at least ~50 GB for prompt-cache files. The Stage C CLIP-G embeddings are ~4–6 MB each.

## Prerequisites

1. Python 3.12 (matching the project `.venv`).
2. CUDA 12.1+ or ROCm 5.7+ for GPU acceleration (or Apple Metal for M-series Macs, though Stage C is mostly tested on CUDA).
3. `torchvision` (required for the Stable Cascade autoencoder) and `accelerate` for launching training.

Check your Python version:

```bash
python --version
```

Install missing packages (Ubuntu example):

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
```

## Installation

Follow the standard SimpleTuner installation (pip or source). For a typical CUDA workstation:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install simpletuner[cuda]
```

For contributors or anyone hacking on the repo directly, install from source and then run `pip install -e .[cuda,dev]`.

## Environment Setup

### 1. Copy the base config

```bash
cp config/config.json.example config/config.json
```

Set the following keys (values shown are a good baseline for Stage C):

| Key | Recommendation | Notes |
| --- | -------------- | ----- |
| `model_family` | `"stable_cascade"` | Required to load Stage C components |
| `model_flavour` | `"stage-c"` (or `"stage-c-lite"`) | Lite flavour trims parameters if you only have ~18 GB VRAM |
| `model_type` | `"lora"` | Full fine-tune works but requires substantially more memory |
| `mixed_precision` | `"no"` | Stage C refuses to run mixed precision unless you set `i_know_what_i_am_doing=true`; fp32 is the safe choice |
| `gradient_checkpointing` | `true` | Saves 3–4 GB of VRAM |
| `vae_batch_size` | `1` | The Stage C autoencoder is heavy; keep it small |
| `validation_resolution` | `"1024x1024"` | Matches the downstream decoder expectations |
| `stable_cascade_use_decoder_for_validation` | `true` | Ensures validation uses the combined prior+decoder pipeline |
| `stable_cascade_decoder_model_name_or_path` | `"stabilityai/stable-cascade"` | Change to a local path if you have a custom Stage B/C decoder |
| `stable_cascade_validation_prior_num_inference_steps` | `20` | Prior denoising steps |
| `stable_cascade_validation_prior_guidance_scale` | `3.0–4.0` | CFG on the prior |
| `stable_cascade_validation_decoder_guidance_scale` | `0.0–0.5` | Decoder CFG (0.0 is photorealistic, >0.0 adds more prompt adherence) |

#### Example `config/config.json`

```json
{
  "base_model_precision": "int8-torchao",
  "checkpoint_step_interval": 100,
  "data_backend_config": "config/stable_cascade/multidatabackend.json",
  "gradient_accumulation_steps": 2,
  "gradient_checkpointing": true,
  "hub_model_id": "stable-cascade-stage-c-lora",
  "learning_rate": 1e-4,
  "lora_alpha": 16,
  "lora_rank": 16,
  "lora_type": "standard",
  "lr_scheduler": "cosine",
  "max_train_steps": 30000,
  "mixed_precision": "no",
  "model_family": "stable_cascade",
  "model_flavour": "stage-c",
  "model_type": "lora",
  "optimizer": "adamw_bf16",
  "output_dir": "output/stable_cascade_stage_c",
  "report_to": "wandb",
  "seed": 42,
  "stable_cascade_decoder_model_name_or_path": "stabilityai/stable-cascade",
  "stable_cascade_decoder_subfolder": "decoder_lite",
  "stable_cascade_use_decoder_for_validation": true,
  "stable_cascade_validation_decoder_guidance_scale": 0.0,
  "stable_cascade_validation_prior_guidance_scale": 3.5,
  "stable_cascade_validation_prior_num_inference_steps": 20,
  "train_batch_size": 4,
  "use_ema": true,
  "vae_batch_size": 1,
  "validation_guidance": 4.0,
  "validation_negative_prompt": "ugly, blurry, low-res",
  "validation_num_inference_steps": 30,
  "validation_prompt": "a cinematic photo of a shiba inu astronaut",
  "validation_resolution": "1024x1024"
}
```

Key takeaways:

- `model_flavour` accepts `stage-c` and `stage-c-lite`. Use lite if you're short on VRAM or prefer the distilled prior.
- Keep `mixed_precision` at `"no"`. If you override it, set `i_know_what_i_am_doing=true` and be ready for NaNs.
- Enabling `stable_cascade_use_decoder_for_validation` wires the prior output into the Stage B/C decoder so the validation gallery shows real images instead of prior latents.

### 2. Configure the data backend

Create `config/stable_cascade/multidatabackend.json`:

```json
[
  {
    "id": "primary",
    "type": "local",
    "dataset_type": "images",
    "instance_data_dir": "/data/stable-cascade",
    "resolution": "1024x1024",
    "bucket_resolutions": ["1024x1024", "896x1152", "1152x896"],
    "crop": true,
    "crop_style": "random",
    "minimum_image_size": 768,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "caption_strategy": "filename",
    "prepend_instance_prompt": false,
    "repeats": 1
  },
  {
    "id": "stable-cascade-text-cache",
    "type": "local",
    "dataset_type": "text_embeds",
    "cache_dir": "/data/cache/stable-cascade/text",
    "default": true
  }
]
```

Tips:

- Stage C latents are derived from an autoencoder, so stick to 1024×1024 (or a tight range of portrait/landscape buckets). The decoder expects ~24×24 latent grids from a 1024px input.
- Keep the `target_downsample_size` at 1024 so narrow crops don't explode aspect ratios beyond ~2:1.
- Always configure a dedicated text-embed cache. Without one, every run will spend 30–60 minutes re-embedding captions with CLIP-G.

### 3. Prompt library (optional)

Create `config/stable_cascade/prompt_library.json`:

```json
{
  "portrait": "a cinematic portrait photograph lit by studio strobes",
  "landscape": "a sweeping ultra wide landscape with volumetric lighting",
  "product": "a product render on a seamless background, dramatic reflections",
  "stylized": "digital illustration in the style of a retro sci-fi book cover"
}
```

Enable it in your config by adding `"validation_prompt_library": "config/stable_cascade/prompt_library.json"`.

## Training

1. Activate your environment and launch Accelerate configuration if you have not already:

```bash
source .venv/bin/activate
accelerate config
```

2. Start training:

```bash
accelerate launch simpletuner/train.py \
  --config_file config/config.json \
  --data_backend_config config/stable_cascade/multidatabackend.json
```

During the first epoch, monitor:

- **Text cache throughput** – Stage C will log cache progress. Expect ~8–12 prompts/sec on high-end GPUs.
- **VRAM usage** – Aim for <95% utilization to avoid OOMs when validation runs.
- **Validation outputs** – The combined pipeline should emit full-resolution PNGs into `output/<run>/validation/`.

## Validation & Inference Notes

- The Stage C prior on its own only produces image embeddings. The SimpleTuner validation wrapper automatically feeds them through the decoder when `stable_cascade_use_decoder_for_validation=true`.
- To swap the decoder flavour, set `stable_cascade_decoder_subfolder` to `"decoder"`, `"decoder_lite"`, or a custom folder containing the Stage B or Stage C weights.
- For quicker previews, lower `stable_cascade_validation_prior_num_inference_steps` to ~12 and `validation_num_inference_steps` to 20. Once satisfied, raise them back for higher quality.

## Advanced Experimental Features

SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.
*   **[Diff2Flow](/documentation/experimental/DIFF2FLOW.md):** allows training Stable Cascade with a Flow Matching objective.

> ⚠️ These features increase the computational overhead of training.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "Stable Cascade Stage C requires --mixed_precision=no" | Set `"mixed_precision": "no"` or add `"i_know_what_i_am_doing": true` (not recommended) |
| Validation only shows priors (green noise) | Ensure `stable_cascade_use_decoder_for_validation` is `true` and the decoder weights are downloaded |
| Text embed caching takes hours | Use SSD/NVMe for the cache directory and avoid network mounts. Consider pruning prompts or pre-computing with `simpletuner-text-cache` CLI |
| Autoencoder import error | Install torchvision inside your `.venv` (`pip install torchvision --extra-index-url https://download.pytorch.org/whl/cu124`). Stage C needs EfficientNet weights |

## Next Steps

- Experiment with `lora_rank` (8–32) and `learning_rate` (5e-5 to 2e-4) depending on subject complexity.
- Attach ControlNet/conditioning adapters to Stage B after training the prior.
- If you need faster iteration, train the `stage-c-lite` flavour and keep the `decoder_lite` weights for validation.

Happy tuning!
