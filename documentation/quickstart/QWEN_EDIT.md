# Qwen Image Edit Quickstart

This guide covers the **edit** flavours of Qwen Image that SimpleTuner supports:

- `edit-v1` – single reference image per training example. The reference image is encoded with the Qwen2.5-VL text encoder and cached as **conditioning image embeds**.
- `edit-v2` (“edit plus”) – up to three reference images per sample, encoded into VAE latents on the fly.

Both variants inherit most of the base [Qwen Image quickstart](./QWEN_IMAGE.md); the sections below focus on what is *different* when fine-tuning the edit checkpoints.

---

## 1. Hardware checklist

The base model is still **20 B parameters**:

| Requirement | Recommendation |
|-------------|----------------|
| GPU VRAM    | 24 G minimum (with int8/nf4 quantisation) • 40 G+ strongly recommended |
| Precision   | `mixed_precision=bf16`, `base_model_precision=int8-quanto` (or `nf4-bnb`) |
| Batch size  | Must remain `train_batch_size=1`; use gradient accumulation for the effective batch |

All other training prerequisites from the [Qwen Image guide](./QWEN_IMAGE.md) apply (Python ≥ 3.10, CUDA 12.x image, etc.).

---

## 2. Configuration highlights

Inside `config/config.json`:

```jsonc
{
  "model_type": "lora",
  "model_family": "qwen_image",
  "model_flavour": "edit-v1",      // or "edit-v2"
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.0,
  "validation_num_inference_steps": 30,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "base_model_precision": "int8-quanto",
  "quantize_via": "cpu",
  "quantize_activations": false,
  "flow_schedule_shift": 1.73,
  "data_backend_config": "config/qwen_edit/multidatabackend.json"
}
```

- EMA runs on the CPU by default and is safe to leave enabled unless you need faster checkpoints.
- `validation_resolution` must be reduced (e.g. `768x768`) on 24 G cards.
- `match_target_res` may be added under `model_kwargs` for `edit-v2` if you want the control images to inherit the target resolution instead of the default 1 MP packing:

```jsonc
"model_kwargs": {
  "match_target_res": true
}
```

---

## 3. Dataloader layout

Both flavours expect **paired datasets**: an edit image, optional edit caption, and one or more control/reference images that share the **exact same filenames**.

### 3.1 edit‑v1 (single control image)

The main dataset should reference one conditioning dataset **and** a conditioning-image-embed cache:

```jsonc
[
  {
    "id": "qwen-edit-images",
    "type": "local",
    "instance_data_dir": "/datasets/qwen-edit/images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": ["qwen-edit-reference"],
    "conditioning_image_embeds": "qwen-edit-ref-embeds",
    "cache_dir_vae": "cache/vae/qwen-edit-images"
  },
  {
    "id": "qwen-edit-reference",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit/reference",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-reference"
  },
  {
    "id": "qwen-edit-ref-embeds",
    "type": "local",
    "dataset_type": "conditioning_image_embeds",
    "cache_dir": "cache/conditioning_image_embeds/qwen-edit"
  }
]
```

- `conditioning_type=reference_strict` guarantees that crops match the edit image. Use `reference_loose` only if the reference can be aspect-mismatched.
- The `conditioning_image_embeds` entry stores the Qwen2.5-VL visual tokens produced for each reference. If omitted, SimpleTuner will create a default cache under `cache/conditioning_image_embeds/<dataset_id>`.

### 3.2 edit‑v2 (multi‑control)

For `edit-v2`, list every control dataset under `conditioning_data`. Each entry supplies one additional control frame. You do **not** need a conditioning-image-embed cache because latents are computed on the fly.

```jsonc
[
  {
    "id": "qwen-edit-plus-images",
    "type": "local",
    "instance_data_dir": "/datasets/qwen-edit-plus/images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": [
      "qwen-edit-plus-reference-a",
      "qwen-edit-plus-reference-b",
      "qwen-edit-plus-reference-c"
    ],
    "cache_dir_vae": "cache/vae/qwen-edit-plus/images"
  },
  {
    "id": "qwen-edit-plus-reference-a",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit-plus/reference_a",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-plus/ref_a"
  },
  {
    "id": "qwen-edit-plus-reference-b",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit-plus/reference_b",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-plus/ref_b"
  },
  {
    "id": "qwen-edit-plus-reference-c",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit-plus/reference_c",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-plus/ref_c"
  }
]
```

Use as many control datasets as you have reference images (1–3). SimpleTuner keeps them aligned per sample by matching filenames.

---

## 4. Running the trainer

The quickest smoke test is to run one of the example presets:

```bash
simpletuner train example=qwen_image.edit-v1-lora
# or
simpletuner train example=qwen_image.edit-v2-lora
```

When launching manually:

```bash
simpletuner train \
  --config config/config.json \
  --data config/qwen_edit/multidatabackend.json
```

### Tips

- Keep `caption_dropout_probability` at `0.0` unless you have a reason to train without the edit instruction.
- For long training jobs, reduce validation cadence (`validation_step_interval`) so that expensive edit validations do not dominate runtime.
- Qwen edit checkpoints ship without a guidance head; `validation_guidance` typically lives in the **3.5–4.5** range.

---

## 5. Validation previews

If you want to preview the reference image alongside the validation output, store your validation edit/reference pairs in a dedicated dataset (same layout as the training split) and set:

```jsonc
{
  "eval_dataset_id": "qwen-edit-val"
}
```

SimpleTuner will reuse the conditioning images from that dataset during validation.

---

### Troubleshooting

- **`ValueError: Control tensor list length does not match batch size`** – ensure every conditioning dataset contains files for *all* edit images. Empty folders or mismatched filenames trigger this error.
- **Out of memory during validation** – lower `validation_resolution`, `validation_num_inference_steps`, or quantise further (`base_model_precision=int2-quanto`) before retrying.
- **Cache not found errors** when using `edit-v1` – double-check that the main dataset’s `conditioning_image_embeds` field matches an existing cache dataset entry.

---

You are now ready to adapt the base Qwen Image quickstart to edit training. For full configuration options (text encoder caching, multi-backend sampling, etc.), re-use the guidance from [FLUX_KONTEXT.md](./FLUX_KONTEXT.md) – the dataset pairing workflow is the same, only the model family changes to `qwen_image`.
