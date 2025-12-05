# LongCat‑Image Edit Quickstart

This is the edit/img2img variant of LongCat‑Image. Read [LONGCAT_IMAGE.md](/documentation/quickstart/LONGCAT_IMAGE.md) first; this file only lists what changes for the edit flavour.

---

## 1) Model differences vs base LongCat‑Image

|                               | Base (text2img) | Edit |
| ----------------------------- | --------------- | ---- |
| Flavour                       | `final` / `dev` | `edit` |
| Conditioning                  | none            | **requires conditioning latents (reference image)** |
| Text encoder                  | Qwen‑2.5‑VL     | Qwen‑2.5‑VL **with vision context** (prompt encoding needs ref image) |
| Pipeline                      | TEXT2IMG        | IMG2IMG/EDIT |
| Validation inputs             | prompt only     | prompt **and** reference |

---

## 2) Config changes (CLI/WebUI)

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_image",
  "model_flavour": "edit",
  "base_model_precision": "int8-quanto",      // fp8-torchao also fine; helps fit 16–24 GB
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "learning_rate": 5e-5,
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 40,
  "validation_resolution": "768x768"
}
```

Keep `aspect_bucket_alignment` at 64. Do not disable conditioning latents; the edit pipeline expects them.

Fast config creation:
```bash
cp config/config.json.example config/config.json
```
Then set `model_family`, `model_flavour`, dataset paths, and output_dir.

---

## 3) Dataloader: paired edit + reference

Use two aligned datasets: **edit images** (caption = edit instruction) and **reference images**. The edit dataset’s `conditioning_data` must point to the reference dataset ID. Filenames must match 1‑to‑1.

```jsonc
[
  {
    "id": "edit-images",
    "type": "local",
    "instance_data_dir": "/data/edits",
    "caption_strategy": "textfile",
    "resolution": 768,
    "cache_dir_vae": "/cache/vae/longcat/edit",
    "conditioning_data": ["ref-images"]
  },
  {
    "id": "ref-images",
    "type": "local",
    "instance_data_dir": "/data/refs",
    "caption_strategy": null,
    "resolution": 768,
    "cache_dir_vae": "/cache/vae/longcat/ref"
  }
]
```

Notes:
- Aspect buckets: keep on the 64px grid.
- Reference captions are optional; if present they replace edit captions (usually undesired).
- VAE caches for edit and reference should be separate paths.
- If you see cache misses or shape errors, clear the VAE caches for both datasets and regenerate.

---

## 4) Validation specifics

- Validation needs reference images to produce conditioning latents. Point the validation split of `edit-images` to `ref-images` via `conditioning_data`.
- Guidance: 4–6 works well; keep negative prompt empty.
- Preview callbacks are supported; latents are unpacked for decoders automatically.
- If validation fails due to missing conditioning latents, check that the validation dataloader includes both edit and reference entries with matching filenames.

---

## 5) Inference / validation commands

Quick CLI validation:
```bash
simpletuner validate \
  --model_family longcat_image \
  --model_flavour edit \
  --validation_resolution 768x768 \
  --validation_guidance 4.5 \
  --validation_num_inference_steps 40
```

WebUI: choose the **Edit** pipeline, supply both the source image and the edit instruction.

---

## 6) Training start (CLI)

After config and dataloader are set:
```bash
simpletuner train --config config/config.json
```
Ensure the reference dataset is present during training so conditioning latents can be computed or loaded from cache.

---

## 7) Troubleshooting

- **Missing conditioning latents**: ensure the reference dataset is wired via `conditioning_data` and filenames match.
- **MPS dtype errors**: the pipeline auto‑downgrades pos‑ids to float32 on MPS; keep the rest at float32/bf16.
- **Channel mismatch in previews**: previews un‑patchify latents before decoding (keep this SimpleTuner version).
- **OOM during edit**: lower validation resolution/steps, reduce `lora_rank`, enable group offload, and prefer `int8-quanto`/`fp8-torchao`.
