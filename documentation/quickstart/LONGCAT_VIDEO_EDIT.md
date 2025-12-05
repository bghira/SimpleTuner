# LongCat‑Video Edit (Image‑to‑Video) Quickstart

Read [LONGCAT_VIDEO.md](/documentation/quickstart/LONGCAT_VIDEO.md) first; this note only lists what changes for the image‑to‑video flavour.

---

## What’s different
- Uses the same `final` checkpoint; conditioning comes from a start frame instead of pure text.
- Keep `aspect_bucket_alignment` at 64 and frame counts aligned to the VAE stride (93 frames by default).
- The first latent frame is treated as conditioning; it is kept fixed during sampling.

## Config tweaks

```jsonc
{
  "model_family": "longcat_video",
  "model_flavour": "final",
  "model_type": "lora",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "validation_conditioning_image_path": "/data/start.png",   // required for validation
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0
}
```

## Dataloader tips
- Each sample should include a conditioning image (first frame) plus the target clip and caption.
- Ensure conditioning frames are already resized/cropped to the same buckets as the target clip.

## Validation/inference

```bash
simpletuner validate \
  --model_family longcat_video \
  --model_flavour final \
  --validation_conditioning_image_path /path/to/start_frame.png \
  --validation_resolution 480x832 \
  --validation_num_video_frames 93 \
  --validation_num_inference_steps 40 \
  --validation_guidance 4.0
```

## Troubleshooting
- If the first frame drifts, lower guidance (3.5–4.0) or reduce steps.
- Height/width must stay on the 64px grid; mismatched buckets will raise shape errors.
