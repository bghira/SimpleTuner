# Kontext \[dev] Mini Quick‑start

> 📝  Kontext shares 90 % of its training workflow with Flux, so this file only lists what *differs*.  When a step is **not** mentioned here, follow the original [instructions](/documentation/quickstart/FLUX.md)

**Note**: All LoRAs trained for Kontext will run perfectly on [Runware](https://runware.ai), the world's **fastest** (~5s-15s depending on settings) and **most affordable inference platform** for Flux dev, schnell, and kontext.

---

## 1. Model overview

|                                                  | Flux‑dev               | Kontext‑dev                                 |
| ------------------------------------------------ | -------------------    | ------------------------------------------- |
| License                                          | Non‑commercial         | Non‑commercial                              |
| Guidance                                         | Distilled (CFG ≈ 1)     | Distilled (CFG ≈ 1)                         |
| Variants available                               | *dev*, schnell,\[pro]   | *dev*, \[pro, max]                          |
| T5 sequence length                               | 512 dev, 256 schnell   | 512 dev                                     |
| Typical 1024 px inference time<br>(4090 @ CFG 1)  | ≈ 20 s                  | **≈ 80 s**                                  |
| VRAM for 1024 px LoRA @ int8‑quanto               | 18 G                   | **24 G**                                    |

Kontext keeps the Flux transformer backbone but introduces **paired‑reference conditioning**.  Two operating modes exist:

* `reference_loose` (✅ stable, default) – reference can differ in aspect‐ratio/size from the edit.
  - Currently, the only (truly) supported mode. Images are scanned for metadata, aspect bucketed, and cropped independently of each other.
  - This may be an issue for setups where you'd like to ensure the alignment of the edit and reference images, such as in a dataloader that uses a single image per file name.
* `reference_strict` (✅ stable) – reference is pre‑transformed exactly like the edit crop.
  - This is how you should configure your datasets if you need perfect alignment between crops / aspect bucketing between your edit and reference images.
  - Originally required `--vae_cache_ondemand` and some increased VRAM usage, but no longer does.
  - Duplicates the crop / aspect bucket metadata from the source dataset at startup, so you don't have to.


---

## 2. Hardware requirements

* **System RAM**: quantisation still needs 50 GB.
* **GPU**: 3090 (24 G) is the realistic minimum for 1024 px training **with int8‑quanto**.
  * Hopper H100/H200 systems with Flash Attention 3 can enable `--fuse_qkv_projections` to greatly speed up training.
  * If you train at 512 px you can squeeze into a 12 G card, but expect slow batches (sequence length remains large).


---

## 3. Quick configuration diff

Below is the *smallest* set of changes you need in `config/config.json` compared with your typical Flux training configuration.

```jsonc
{
  "model_family":   "flux",
  "model_flavour": "kontext",                       // <‑‑ change
  "base_model_precision": "int8-quanto",            // fits on 24 G at 1024 px
  "gradient_checkpointing": true,
  "fuse_qkv_projections": false,                    // <‑‑ use this to speed up training on Hopper H100/H200 systems. WARNING: requires flash-attn manually installed.
  "lora_rank": 16,
  "learning_rate": 1e-5,
  "optimizer": "optimi-lion",                       // <‑‑ use Lion for faster results, and adamw_bf16 for slower but possibly more stable results.
  "max_train_steps": 10000,
  "validation_guidance": 2.5,                       // <‑‑ kontext really does best with a guidance value of 2.5
  "validation_resolution": "1024x1024",
  "conditioning_multidataset_sampling": "random"    // <-- setting this to "combined" when you have two conditioning datasets defined will show them simultaneously instead of switching between them.
}
```

### Dataloader snippet (multi‑data‑backend)

If you've manually curated an image-pair dataset, you can configure it using two separate directories: one for the edit images and one for the reference images.

The `conditioning_data` field in the edit dataset should point to the reference dataset's `id`.

```jsonc
[
  {
    "id": "my-edited-images",
    "type": "local",
    "instance_data_dir": "/path/to/datasets/edited-images",     // <-- use absolute paths
    "conditioning_data": [
      "my-reference-images"                                        // <‑‑ this should be your "id" of the reference set
                                                                // you could specify a second set to alternate between or combine them, e.g. ["reference-images", "reference-images2"]
    ],
    "resolution": 1024,
    "caption_strategy": "textfile"                              // <-- these captions should contain the edit instructions
  },
  {
    "id": "my-reference-images",
    "type": "local",
    "instance_data_dir": "/path/to/datasets/reference-images",  // <-- use absolute paths
    "conditioning_type": "reference_strict",                    // <‑‑ if this is set to reference_loose, the images are cropped independently of the edit images
    "resolution": 1024,
    "caption_strategy": null,                                   // <‑‑ no captions needed for references, but if available, will be used INSTEAD of the edit captions
                                                                // NOTE: you cannot define separate conditioning captions when using conditioning_multidataset_sampling=combined.
                                                                // Only the edit datasets' captions will be used.
  }
]
```

*Every edit image **must** have 1‑to‑1 matching file names and extensions in `reference-images/`.*  SimpleTuner will automatically staple the reference embedding to the edit’s conditioning.

If you'd like a demo dataset of how to set this up, you can use this [Kontext Max derived demo dataset](https://huggingface.co/datasets/terminusresearch/KontextMax-Edit-smol) which contains reference and edit images along with their caption textfiles.

### Automatic Reference-Edit Pair Generation

If you don't have pre-existing reference-edit pairs, SimpleTuner can automatically generate them from a single dataset. This is particularly useful for training models for:
- Image enhancement / super-resolution
- JPEG artifact removal
- Deblurring
- Other restoration tasks

#### Example: Deblurring Training Dataset

```jsonc
[
  {
    "id": "high-quality-images",
    "type": "local",
    "instance_data_dir": "/path/to/sharp-images",
    "conditioning_data": "high-quality-images_conditioning_superresolution",
    "resolution": 1024,
    "caption_strategy": "textfile",
    "conditioning": [
      {
        "type": "superresolution",
        "blur_radius": 3.0,
        "blur_type": "gaussian",
        "add_noise": true,
        "noise_level": 0.02,
        "captions": ["enhance sharpness", "deblur", "increase clarity", "sharpen image"]
      }
    ]
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/kontext"
  }
]
```

This configuration will:
1. Take your high-quality sharp images and create blurred versions (these become the "edit" images)
2. Use the originals as reference images
3. Train Kontext to enhance/deblur based on the reference

> **NOTE**: You can't define `captions` on a conditioning dataset when using `conditioning_multidataset_sampling=combined`. The edit dataset's captions will be used instead.

#### Example: JPEG Artifact Removal

```jsonc
[
  {
    "id": "pristine-images",
    "type": "local",
    "instance_data_dir": "/path/to/pristine-images",
    "conditioning_data": "pristine-images_conditioning_jpeg_artifacts",
    "resolution": 1024,
    "caption_strategy": "textfile",
    "conditioning": [
      {
        "type": "jpeg_artifacts",
        "quality_mode": "range",
        "quality_range": [10, 30],
        "compression_rounds": 2,
        "captions": ["remove compression artifacts", "restore quality", "fix jpeg artifacts"]
      }
    ]
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/kontext"
  }
]
```

#### Important Notes

1. **Generation happens at startup**: The degraded versions are created automatically when training begins
2. **Caching**: Generated images are saved, so subsequent runs won't regenerate them
3. **Caption strategy**: The `captions` field in the conditioning config provides task-specific prompts that work better than generic image descriptions
4. **Performance**: These CPU-based generators (blur, JPEG) are fast and use multiple processes
5. **Disk space**: Ensure you have enough disk space for the generated images, as they can be large! Unfortunately, there is no ability to create them on-demand yet.

For more conditioning types and advanced configurations, see the [ControlNet documentation](/documentation/CONTROLNET.md).

---

## 4. Training tips specific to Kontext

1. **Longer sequences → slower steps.**  Expect \~0.4 it/s on a single 4090 at 1024 px, rank‑1 LoRA, bf16 + int8.
2. **Explore for the correct settings.**  Not a lot is known about fine-tuning Kontext; for safety, stay at `1e‑5` (Lion) or `5e‑4` (AdamW).
3. **Watch VRAM spikes during VAE caching.**  If you OOM, add `--offload_during_startup=true`, lower your `resolution`, or possibly enable VAE tiling via your `config.json`.
4. **You can train it without reference images, but not currently via SimpleTuner.**  Currently, things are somewhat hardcoded toward requiring conditional images to be supplied, but you can provide normal datasets alongside your edit pairs to allow it to learn subjects and likeness.
5. **Guidance re‑distillation.**  Like Flux‑dev, Kontext‑dev is CFG‑distilled; if you need diversity, retrain with `validation_guidance_real > 1` and use an Adaptive‑Guidance node at inference, though this will take a LOT longer to converge, and will require a large rank LoRA or a Lycoris LoKr to succeed.
6. **Full-rank training is probably a waste of time.** Kontext is designed to be trained with low rank, and full rank training will likely not yield any better results than a Lycoris LoKr, which will typically outperform a Standard LoRA with less work chasing the best parameters.  If you want to try it anyway, you'll have to use DeepSpeed.

---

## 5. Inference gotchas

- Match your training and inference precision levels; int8 training will do best with int8 inference and so on.
- It's going to be very slow due to the fact that two images are running through the system at a time.  Expect 80 s per 1024 px edit on a 4090.
  - For fast remote inference, use Runware's [modelUpload feature](https://runware.ai/docs/en/image-inference/model-upload) via the UI or API to upload your LoRA and run it on their servers, often reaching 15 seconds per 50 step generation (vs 80 seconds on a 4090).

---

## 6. Troubleshooting cheat‑sheet

| Symptom                                 | Likely cause               | Quick fix                                              |
| --------------------------------------- | -------------------------- | ------------------------------------------------------ |
| OOM during quantisation                 | Not enough **system** RAM  | Use `quantize_via=cpu`                                 |
| Ref image ignored / no edit applied     | Dataloader mis‑pairing     | Ensure identical filenames & `conditioning_data` field |
| Square grid artifacts                   | Low‑quality edits dominate | Make higher-quality dataset, lower LR, avoid Lion      |

---

## 7. Credits & further reading

*Original research & checkpoints:* **@Black‑Forest‑Labs**
*Implementation wrappers & docs:* **@bghira** / **@Beinsezii**, and others at **@Runware**
*Work sponsored by*: **[Runware](https://runware.ai)** & **Terminus Research Group**

For advanced tuning options (LoKr, NF4 quant, DeepSpeed, etc.) consult [the original quickstart for Flux](/documentation/quickstart/FLUX.md) – every flag works the same unless stated otherwise above.
