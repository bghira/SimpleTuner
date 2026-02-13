# CREPA (video regularization)

Cross-frame Representation Alignment (CREPA) is a light regularizer for video models. It nudges each frame's hidden states toward a frozen vision encoder's features from the current frame **and its neighbours**, improving temporal consistency without changing your main loss.

> **Looking for image models?** See [IMAGE_REPA.md](IMAGE_REPA.md) for REPA support on image DiT models (Flux, SD3, etc.) and U-REPA for UNet models (SDXL, SD1.5, Kolors).

## When to use it

- You are training on videos with complex motion, scene changes, or occlusions.
- You are fine-tuning a video DiT (LoRA or full) and see flicker/identity drift between frames.
- Supported model families: `kandinsky5_video`, `ltxvideo`, `sanavideo`, and `wan` (other families do not expose the CREPA hooks).
- You have extra VRAM (CREPA adds ~1–2GB depending on settings) for the DINO encoder and VAE, which must remain in memory during training for decoding latents to pixels.

## Quick setup (WebUI)

1. Open **Training → Loss functions**.
2. Enable **CREPA**.
3. Set **CREPA Block Index** to an encoder-side layer. Start with:
   - Kandinsky5 Video: `8`
   - LTXVideo / Wan: `8`
   - SanaVideo: `10`
4. Leave **Weight** at `0.5` to start.
5. Keep **Adjacent Distance** at `1` and **Temporal Decay** at `1.0` for a setup that closely matches the original CREPA paper.
6. Use the defaults for the vision encoder (`dinov2_vitg14`, resolution `518`). Change only if you know you need a smaller encoder (e.g., `dinov2_vits14` + image size `224` to save VRAM).
7. Train as normal. CREPA adds an auxiliary loss and logs `crepa_loss` / `crepa_alignment_score` / `crepa_similarity_self`.

## Quick setup (config JSON / CLI)

Add the following to your `config.json` or CLI args:

```json
{
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 0.5,
  "crepa_adjacent_distance": 1,
  "crepa_adjacent_tau": 1.0,
  "crepa_encoder": "dinov2_vitg14",
  "crepa_encoder_image_size": 518
}
```

## Tuning knobs

- `crepa_spatial_align`: keep patch-level structure (default). Set `false` to pool if memory is tight.
- `crepa_normalize_by_frames`: keeps loss scale stable across clip lengths (default). Turn off if you want longer clips to contribute more.
- `crepa_drop_vae_encoder`: free memory if you only ever **decode** latents (unsafe if you need to encode pixels).
- `crepa_adjacent_distance=0`: behaves like per-frame REPA* (no neighbour help); combine with `crepa_adjacent_tau` for distance decay.
- `crepa_cumulative_neighbors=true` (config only): use all offsets `1..d` instead of just the nearest neighbours.
- `crepa_use_backbone_features=true`: skip the external encoder and align to a deeper transformer block; set `crepa_teacher_block_index` to choose the teacher.
- Encoder size: downshift to `dinov2_vits14` + `224` if VRAM is tight; keep `dinov2_vitg14` + `518` for best quality.

## Coefficient scheduling

CREPA supports scheduling the coefficient (`crepa_lambda`) over training with warmup, decay, and automatic cutoff based on similarity threshold. This is particularly useful for text2video training where CREPA may cause horizontal/vertical stripes or a washed-out feel if applied too strongly for too long.

### Basic scheduling

```json
{
  "crepa_enabled": true,
  "crepa_lambda": 0.5,
  "crepa_scheduler": "cosine",
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 5000,
  "crepa_lambda_end": 0.0
}
```

This configuration:
1. Ramps CREPA weight from 0 to 0.5 over the first 100 steps
2. Decays from 0.5 to 0.0 using a cosine schedule over 5000 steps
3. After step 5100, CREPA is effectively disabled

### Scheduler types

- `constant`: No decay, weight stays at `crepa_lambda` (default)
- `linear`: Linear interpolation from `crepa_lambda` to `crepa_lambda_end`
- `cosine`: Smooth cosine annealing (recommended for most cases)
- `polynomial`: Polynomial decay with configurable power via `crepa_power`

### Step-based cutoff

For a hard cutoff after a specific step:

```json
{
  "crepa_cutoff_step": 3000
}
```

CREPA is completely disabled after step 3000.

### Similarity-based cutoff

This is the most flexible approach—CREPA automatically disables when the similarity metric plateaus, indicating the model has learned sufficient temporal alignment:

```json
{
  "crepa_similarity_threshold": 0.9,
  "crepa_similarity_ema_decay": 0.99,
  "crepa_threshold_mode": "permanent"
}
```

- `crepa_similarity_threshold`: When the exponential moving average of similarity reaches this value, CREPA cuts off
- `crepa_similarity_ema_decay`: Smoothing factor (0.99 ≈ 100-step window)
- `crepa_threshold_mode`: `permanent` (stays off) or `recoverable` (can re-enable if similarity drops)

### Recommended configurations

**For image2video (i2v)**:
```json
{
  "crepa_scheduler": "constant",
  "crepa_lambda": 0.5
}
```
Standard CREPA works well for i2v since the reference frame anchors consistency.

**For text2video (t2v)**:
```json
{
  "crepa_scheduler": "cosine",
  "crepa_lambda": 0.5,
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 0,
  "crepa_lambda_end": 0.1,
  "crepa_similarity_threshold": 0.85,
  "crepa_threshold_mode": "permanent"
}
```
Decays CREPA over training and cuts off when similarity saturates to prevent artifacts.

**For solid backgrounds (t2v)**:
```json
{
  "crepa_cutoff_step": 2000
}
```
Early cutoff prevents stripe artifacts on uniform backgrounds.

<details>
<summary>How it works (practitioner)</summary>

- Captures hidden states from a chosen DiT block, projects them through a LayerNorm+Linear head, and aligns them to frozen vision features.
- By default it encodes pixel frames with DINOv2; backbone mode reuses a deeper transformer block instead.
- Aligns each frame to its neighbours with an exponential decay on distance (`crepa_adjacent_tau`); cumulative mode optionally sums all offsets up to `d`.
- Spatial/temporal alignment resamples tokens so DiT patches and encoder patches line up before cosine similarity; the loss averages over patches and frames.

</details>

<details>
<summary>Technical (SimpleTuner internals)</summary>

- Implementation: `simpletuner/helpers/training/crepa.py`; registered from `ModelFoundation._init_crepa_regularizer` and attached to the trainable model (projector lives on the model for optimizer coverage).
- Hidden-state capture: video transformers stash `crepa_hidden_states` (and optionally `crepa_frame_features`) when `crepa_enabled` is true; backbone mode can also pull `layer_{idx}` from the shared hidden-state buffer.
- Loss path: decodes latents with the VAE to pixels unless `crepa_use_backbone_features` is on; normalizes projected hidden states and encoder features, applies distance-weighted cosine similarity, logs `crepa_loss` / `crepa_alignment_score` / `crepa_similarity_self`, and adds the scaled loss.
- Interaction: runs before LayerSync so both can reuse the hidden-state buffer; clears the buffer afterward. Requires a valid block index and a hidden size inferred from the transformer config.

</details>

## Common pitfalls

- Enabling CREPA on unsupported families leads to missing hidden states; stick to `kandinsky5_video`, `ltxvideo`, `sanavideo`, or `wan`.
- **Block index too high** → “hidden states not returned”. Lower the index; it is zero-based on the transformer blocks.
- **VRAM spikes** → try `crepa_spatial_align=false`, a smaller encoder (`dinov2_vits14` + `224`), or a lower block index.
- **Backbone mode errors** → set both `crepa_block_index` (student) and `crepa_teacher_block_index` (teacher) to layers that exist.
- **Out of memory** → If RamTorch is not helping, your only solution may be a larger GPU—if H200 or B200 do not work, please file an issue report.
