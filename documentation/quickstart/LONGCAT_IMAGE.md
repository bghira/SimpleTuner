# LongCat‑Image Quickstart

LongCat‑Image is a 6B bilingual (zh/en) text‑to‑image model that uses flow matching and the Qwen‑2.5‑VL text encoder. This guide walks you through setup, data prep, and running a first training/validation with SimpleTuner.

---

## 1) Hardware requirements (what to expect)

- VRAM: 16–24 GB covers 1024px LoRA at `int8-quanto` or `fp8-torchao`. Full bf16 runs may need ~24 GB.
- System RAM: ~32 GB is normally enough.
- Apple MPS: supported for inference/preview; we already downcast pos‑ids to float32 on MPS to avoid dtype issues.

---

## 2) Prerequisites (step‑by‑step)

1. Python 3.10–3.12 verified:
   ```bash
   python --version
   ```
2. (Linux/CUDA) On fresh images, install the usual build/toolchain bits:
   ```bash
   apt -y update
   apt -y install build-essential nvidia-cuda-toolkit
   ```
3. Install SimpleTuner with the right extras for your backend:
   ```bash
   pip install "simpletuner[cuda]"   # CUDA
   pip install "simpletuner[mps]"    # Apple Silicon
   pip install "simpletuner[cpu]"    # CPU-only
   ```
4. Quantisation is built in (`int8-quanto`, `int4-quanto`, `fp8-torchao`) and does not need extra manual installs in normal setups.

---

## 3) Environment setup

### Web UI (most guided)
```bash
simpletuner server
```
Visit http://localhost:8001 and pick model family `longcat_image`.

### CLI baseline (config/config.json)

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_image",
  "model_flavour": "final",                // options: final, dev
  "pretrained_model_name_or_path": null,   // auto-selected from flavour; override with a local path if needed
  "base_model_precision": "int8-quanto",   // good default; fp8-torchao also works
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 16,
  "learning_rate": 1e-4,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 30
}
```

**Key defaults to keep**
- Flow matching scheduler is automatic; no special schedule flags needed.
- Aspect buckets stay 64‑pixel aligned; do not lower `aspect_bucket_alignment`.
- Max token length 512 (Qwen‑2.5‑VL).

Optional memory savers (pick what matches your hardware):
- `--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`
- Lower `lora_rank` (4–8) and/or use `int8-quanto` base precision.
- If validation OOMs, reduce `validation_resolution` or steps first.

### Fast config creation (one-time)
```bash
cp config/config.json.example config/config.json
```
Edit the fields above (model_family, flavour, precision, paths). Point `output_dir` and dataset paths to your storage.

### Start training (CLI)
```bash
simpletuner train --config config/config.json
```
or launch the WebUI and start a run from the Jobs page after selecting the same config.

---

## 4) Dataloader pointers (what to supply)

- Standard captioned image folders (textfile/JSON/CSV) work. Include both zh/en if you want bilingual strength to persist.
- Keep bucket edges on the 64px grid. If you train multi‑aspect, list several resolutions (e.g., `1024x1024,1344x768`).
- The VAE is KL with shift+scale; caches use the built‑in scaling factor automatically.

---

## 5) Validation and inference

- Guidance: 4–6 is a good start; leave the negative prompt empty.
- Steps: ~30 for speed checks; 40–50 for best quality.
- Validation preview works out of the box; latents are unpacked before decoding to avoid channel mismatches.

Example (CLI validate):
```bash
simpletuner validate \
  --model_family longcat_image \
  --model_flavour final \
  --validation_resolution 1024x1024 \
  --validation_num_inference_steps 30 \
  --validation_guidance 4.5
```

---

## 6) Troubleshooting

- **MPS float64 errors**: handled internally; keep your config on float32/bf16.
- **Channel mismatch in previews**: fixed by unpacking latents pre‑decode (included in this guide’s code).
- **OOM**: lower `validation_resolution`, reduce `lora_rank`, enable group offload, or switch to `int8-quanto` / `fp8-torchao`.
- **Slow tokenisation**: Qwen‑2.5‑VL caps at 512 tokens; avoid very long prompts.

---

## 7) Flavour selection
- `final`: main release (best quality).
- `dev`: mid‑training checkpoint for experiments/fine‑tunes.
