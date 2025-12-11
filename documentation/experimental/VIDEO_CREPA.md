# CREPA (video regularization)

Cross-frame Representation Alignment (CREPA) is a light regularizer for video models. It nudges each frame’s hidden states toward a frozen vision encoder’s features from the current frame **and its neighbours**, improving temporal consistency without changing your main loss.

## When to use it

- You're training on videos
- Your data has complex motion, scene changes, or occlusions
- You are fine-tuning a video DiT (LoRA or full) and see flicker/identity drift between frames
- Supported model families: `kandinsky5_video`, `ltxvideo`, `sanavideo`, and `wan`
  - Other families simply ignore the toggle
- You have extra VRAM (CREPA adds ~1-2GB depending on settings) for the DINO encoder and VAE, which must remain in memory during training for decoding latents to pixels

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
7. Train as normal. CREPA adds an auxiliary loss and logs `crepa_loss` / `crepa_similarity`.

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

Optional quality/VRAM tweaks:

- `crepa_spatial_align`: `true` keeps patch-level structure (default).
  - Set `false` to pool if memory is tight.
- `crepa_normalize_by_frames`: `true` keeps loss scale stable across clip lengths (default).
  - Turn off if you want longer clips to contribute more.
- `crepa_drop_vae_encoder`: free memory if you only ever **decode** latents (unsafe if you need to encode pixels).
- `crepa_adjacent_distance=0`: behaves like per-frame REPA* (no neighbour help).
- `crepa_cumulative_neighbors=true` (config only): use all offsets `1..d` instead of just the nearest neighbours.

## Common pitfalls

- **Block index too high** → you’ll see “hidden states not returned”. Lower the index; it is zero-based on the transformer blocks.
- **VRAM spikes** → try `crepa_spatial_align=false`, a smaller encoder (`dinov2_vits14` + `224`), or a lower block index.
- **Out of memory** → If RamTorch is not helping, your only solution may be a larger GPU - if H200 or B200 do not work, please file an issue report.
