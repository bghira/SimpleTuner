# LongCat‑Video Edit (Image‑to‑Video) Quickstart

This guide walks you through training and validating the image‑to‑video workflow for LongCat‑Video. You don’t need to flip flavours; the same `final` checkpoint covers both text‑to‑video and image‑to‑video. The difference comes from your datasets and validation settings.

---

## 1) Model differences vs base LongCat‑Video

|                               | Base (text2video) | Edit / I2V |
| ----------------------------- | ----------------- | ---------- |
| Flavour                       | `final`           | `final` (same weights) |
| Conditioning                  | none              | **requires conditioning frame** (first latent kept fixed) |
| Text encoder                  | Qwen‑2.5‑VL       | Qwen‑2.5‑VL (same) |
| Pipeline                      | TEXT2IMG          | IMG2VIDEO |
| Validation inputs             | prompt only       | prompt **and** conditioning image |
| Buckets / stride              | 64px buckets, `(frames-1)%4==0` | same |

**Core defaults you inherit**
- Flow matching with shift `12.0`.
- Aspect buckets enforced at 64px.
- Qwen‑2.5‑VL text encoder; empty negatives auto‑added when CFG is on.
- Default frames: 93 (satisfies `(frames-1)%4==0`).

---

## 2) Config changes (CLI/WebUI)

```jsonc
{
  "model_family": "longcat_video",
  "model_flavour": "final",
  "model_type": "lora",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "learning_rate": 1e-4,
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0,
  "validation_using_datasets": true,
  "eval_dataset_id": "longcat-video-val"
}
```

Keep `aspect_bucket_alignment` at 64. The first latent frame holds the start image; leave it intact. Stick with 93 frames (already matches the VAE stride rule `(frames - 1) % 4 == 0`) unless you have a strong reason to change it.

Quick setup:
```bash
cp config/config.json.example config/config.json
```
Fill in `model_family`, `model_flavour`, `output_dir`, `data_backend_config`, and `eval_dataset_id`. Leave the defaults above unless you know you need different values.

---

## 3) Dataloader: pair clips with start frames

- Create two datasets:
  - **Clips**: the target videos + captions (edit instructions). Mark them `is_i2v: true` and set `conditioning_data` to the start-frame dataset ID.
  - **Start frames**: one image per clip, same filenames, no captions.
- Keep both on the 64px grid (e.g., 480x832). Height/width must be divisible by 16. Frame counts must meet `(frames - 1) % 4 == 0`; 93 is already valid.
- Use separate VAE caches for clips vs start frames.

Example `multidatabackend.json`:
```jsonc
[
  {
    "id": "longcat-video-train",
    "type": "local",
    "dataset_type": "video",
    "is_i2v": true,
    "instance_data_dir": "/data/video-clips",
    "caption_strategy": "textfile",
    "resolution": 480,
    "cache_dir_vae": "/cache/vae/longcat/video",
    "conditioning_data": ["longcat-video-cond"]
  },
  {
    "id": "longcat-video-cond",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/data/video-start-frames",
    "caption_strategy": null,
    "resolution": 480,
    "cache_dir_vae": "/cache/vae/longcat/video-cond"
  }
]
```

---

## 4) Validation specifics

- Add a small validation split with the same paired structure as training. Set `validation_using_datasets: true` and point `eval_dataset_id` to that split (e.g., `longcat-video-val`) so validation pulls the start frame automatically.
- WebUI previews: start `simpletuner server`, choose LongCat‑Video edit, and upload the start frame + prompt.
- Guidance: 3.5–5.0 works; empty negatives are auto‑filled when CFG is on.
- The conditioning frame stays fixed during sampling; only later frames denoise.

---

## 5) Training start (CLI)

After config and dataloader are set:
```bash
simpletuner train --config config/config.json
```
Ensure conditioning frames are present in the training data so the pipeline can build conditioning latents.

---

## 6) Troubleshooting

- **Missing conditioning image**: provide a conditioning dataset via `conditioning_data` with matching filenames; set `eval_dataset_id` to your validation split ID.
- **Height/width errors**: keep dimensions divisible by 16 and on the 64px grid.
- **First frame drifts**: lower guidance (3.5–4.0) or reduce steps.
- **OOM**: lower validation resolution/frames, reduce `lora_rank`, enable group offload, or use `int8-quanto`/`fp8-torchao`.
