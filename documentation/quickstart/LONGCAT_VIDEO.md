# LongCat‑Video Quickstart

LongCat‑Video is a 13.6B bilingual text‑to‑video and image‑to‑video model that uses flow matching, the Qwen‑2.5‑VL text encoder, and a Wan‑style VAE. This guide covers setup, data prep, and running validation/training end‑to‑end with SimpleTuner.

---

## 1) Hardware expectations

- **VRAM**: 24–48 GB is recommended for 720p LoRA; 480p runs fit in ~24 GB with gradient checkpointing and low ranks.
- **System RAM**: ~64 GB is comfortable for dataloading multi‑frame clips.
- **Apple MPS**: supported for previews; positional encodings are downcast to float32 automatically.

---

## 2) Prerequisites

1. Verify Python 3.12 (SimpleTuner ships a `.venv` by default):
   ```bash
   python --version
   ```
2. Install SimpleTuner with the backend that matches your hardware:
   ```bash
   pip install "simpletuner[cuda]"   # NVIDIA
   pip install "simpletuner[mps]"    # Apple Silicon
   pip install "simpletuner[cpu]"    # CPU-only
   ```
3. Quantisation is built in (`int8-quanto`, `fp8-torchao`); no extra wheels needed for typical setups.

---

## 3) Environment setup

### Web UI
```bash
simpletuner server
```
Open http://localhost:8001 and pick model family `longcat_video`.

### CLI baseline config (`config/config.json`)

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_video",
  "model_flavour": "final",
  "pretrained_model_name_or_path": null,      // auto-selected from flavour
  "base_model_precision": "bf16",             // int8-quanto/fp8-torchao also work for LoRA
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "learning_rate": 1e-4,
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0
}
```

**Key defaults to keep**
- Flow‑matching scheduler with shift `12.0` is automatic; no custom noise flags required.
- Aspect buckets stay 64‑pixel aligned (`aspect_bucket_alignment` is enforced at 64).
- Max token length 512 (Qwen‑2.5‑VL).

Optional VRAM savers:
- Reduce `lora_rank` (4–8) and use `int8-quanto` base precision.
- Enable group offload: `--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`.
- Lower `validation_resolution`, frames, or steps first if previews OOM.

### Start training (CLI)
```bash
simpletuner train --config config/config.json
```
Or launch the Web UI and submit a job with the same config.

---

## 4) Dataloader guidance

- Use captioned video datasets; each sample should provide frames (or a short clip) plus a text caption. `dataset_type: video` is handled automatically via `VideoToTensor`.
- Keep frame dimensions on the 64px grid (e.g., 480x832, 720p buckets). Height/width must be divisible by `vae_scale_factor_spatial * 2` (Wan VAE + 1x2x2 patching).
- For image‑to‑video runs, include a conditioning image per sample; SimpleTuner will place it in the first latent frame when present.

---

## 5) Validation & inference

- Guidance: 3.5–5.0 works well; empty negative prompts are auto‑generated when CFG is enabled.
- Steps: 35–45 for quality checks; lower for quick previews.
- Frames: 93 by default (aligns with the VAE temporal stride of 4).

Examples (CLI):
```bash
# Text-to-video
simpletuner validate \
  --model_family longcat_video \
  --model_flavour final \
  --validation_resolution 480x832 \
  --validation_num_video_frames 93 \
  --validation_num_inference_steps 40 \
  --validation_guidance 4.0

# Image-to-video (condition on a single frame)
simpletuner validate \
  --model_family longcat_video \
  --model_flavour final \
  --validation_conditioning_image_path /path/to/start_frame.png \
  --validation_resolution 480x832 \
  --validation_num_video_frames 93 \
  --validation_num_inference_steps 40 \
  --validation_guidance 4.0
```

Latent previews are unpacked before decoding to avoid channel mismatches.

---

## 6) Troubleshooting

- **Height/width errors**: ensure both are divisible by 16 (64‑aligned buckets are safest).
- **MPS float64 warnings**: handled internally; keep precision at bf16/float32.
- **OOM**: drop validation resolution or frames, lower LoRA rank, enable group offload, or switch to `int8-quanto`.
- **Blank negatives with CFG**: if you omit a negative prompt, the pipeline inserts an empty one automatically.

---

## 7) Flavours

- `final`: main LongCat‑Video release (text‑to‑video + image‑to‑video in one checkpoint).
