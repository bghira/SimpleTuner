# LongCat‑Video Quickstart

LongCat‑Video is a 13.6B bilingual (zh/en) text‑to‑video and image‑to‑video model that uses flow matching, the Qwen‑2.5‑VL text encoder, and the Wan VAE. This guide walks you through setup, data prep, and running a first training/validation with SimpleTuner.

---

## 1) Hardware requirements (what to expect)

- 13.6B transformer + Wan VAE: expect higher VRAM than image models; start with `train_batch_size=1`, gradient checkpointing, and low LoRA ranks.
- System RAM: more than 32 GB is helpful for multi‑frame clips; keep datasets on fast storage.
- Apple MPS: supported for previews; positional encodings downcast to float32 automatically.

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
3. Quantisation is built in (`int8-quanto`, `int4-quanto`, `fp8-torchao`) and does not need extra manual installs in normal setups.

---

## 3) Environment setup

### Web UI
```bash
simpletuner server
```
Open http://localhost:8001 and pick model family `longcat_video`.

### CLI baseline (config/config.json)

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
- Flow‑matching scheduler with shift `12.0` is automatic; no custom noise flags needed.
- Aspect buckets stay 64‑pixel aligned; `aspect_bucket_alignment` is enforced at 64.
- Max token length 512 (Qwen‑2.5‑VL); the pipeline auto‑adds empty negatives when CFG is on and no negative prompt is provided.
- Frames must satisfy `(num_frames - 1)` divisible by the VAE temporal stride (default 4). The default 93 frames already matches this.

Optional VRAM savers:
- Reduce `lora_rank` (4–8) and use `int8-quanto` base precision.
- Enable group offload: `--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`.
- Lower `validation_resolution`, frames, or steps first if previews OOM.
- Attention defaults: on CUDA, LongCat‑Video will automatically use the bundled block‑sparse Triton kernel when it's available and fall back to the standard dispatcher otherwise. No toggle needed. If you specifically want xFormers, set `attention_mechanism: "xformers"` in your config/CLI.

### Start training (CLI)
```bash
simpletuner train --config config/config.json
```
Or launch the Web UI and submit a job with the same config.

---

## 4) Dataloader guidance

- Use captioned video datasets; each sample should provide frames (or a short clip) plus a text caption. `dataset_type: video` is handled automatically via `VideoToTensor`.
- Keep frame dimensions on the 64px grid (e.g., 480x832, 720p buckets). Height/width must be divisible by the Wan VAE stride (16px with the built‑in settings) and by 64 for bucketing.
- For image‑to‑video runs, include a conditioning image per sample; it is placed in the first latent frame and kept fixed during sampling.
- LongCat‑Video is 30 fps by design. The default 93 frames is ~3.1 s; if you change frame counts, keep `(frames - 1) % 4 == 0` and remember duration scales with fps.

### Video bucket strategy

In your dataset's `video` section, you can configure how videos are grouped:
- `bucket_strategy`: `aspect_ratio` (default) groups by spatial aspect ratio. `resolution_frames` groups by `WxH@F` format (e.g., `480x832@93`) for mixed-resolution/duration datasets.
- `frame_interval`: When using `resolution_frames`, round frame counts to this interval (e.g., set to 4 to match the VAE temporal stride).

---

## 5) Validation & inference

- Guidance: 3.5–5.0 works well; empty negative prompts are auto‑generated when CFG is enabled.
- Steps: 35–45 for quality checks; lower for quick previews.
- Frames: 93 by default (aligns with the VAE temporal stride of 4).
- Need more headroom for previews or training? Set `musubi_blocks_to_swap` (try 4–8) and optionally `musubi_block_swap_device` to stream the last transformer blocks from CPU while running forward/backward. Expect extra transfer overhead but lower VRAM peaks.

- Validation runs from the `validation_*` fields in your config or via the WebUI preview tab after `simpletuner server` is started. Use those paths for quick checks instead of a standalone CLI subcommand.
- For dataset-driven validation (including I2V), set `validation_using_datasets: true` and point `eval_dataset_id` at your validation split. If that split is marked `is_i2v` and has linked conditioning frames, the pipeline keeps the first frame fixed automatically.
- Latent previews are unpacked before decoding to avoid channel mismatches.

---

## 6) Troubleshooting

- **Height/width errors**: ensure both are divisible by 16 and stay on the 64px grid.
- **MPS float64 warnings**: handled internally; keep precision at bf16/float32.
- **OOM**: drop validation resolution or frames first, lower LoRA rank, enable group offload, or switch to `int8-quanto`/`fp8-torchao`.
- **Blank negatives with CFG**: if you omit a negative prompt, the pipeline inserts an empty one automatically.

---

## 7) Flavours

- `final`: main LongCat‑Video release (text‑to‑video + image‑to‑video in one checkpoint).
