# REPA & U-REPA (image regularization)

Representation Alignment (REPA) is a regularization technique that aligns diffusion model hidden states with frozen vision encoder features (typically DINOv2). This improves generation quality and training efficiency by leveraging pre-trained visual representations.

SimpleTuner supports two variants:

- **REPA** for DiT-based image models (Flux, SD3, Chroma, Sana, PixArt, etc.) - PR #2562
- **U-REPA** for UNet-based image models (SDXL, SD1.5, Kolors) - PR #2563

> **Looking for video models?** See [VIDEO_CREPA.md](VIDEO_CREPA.md) for CREPA support on video models with temporal alignment.

## When to use it

### REPA (DiT models)
- You are training DiT-based image models and want faster convergence
- You notice quality issues or want stronger semantic grounding
- Supported model families: `flux`, `flux2`, `sd3`, `chroma`, `sana`, `pixart`, `hidream`, `auraflow`, `lumina2`, and others

### U-REPA (UNet models)
- You are training UNet-based image models (SDXL, SD1.5, Kolors)
- You want to leverage representation alignment optimized for UNet architectures
- U-REPA uses **mid-block** alignment (not early layers) and adds **manifold loss** for better relative similarity structure

## Quick setup (WebUI)

### For DiT models (REPA)

1. Open **Training -> Loss functions**.
2. Enable **CREPA** (the same option enables REPA for image models).
3. Set **CREPA Block Index** to an early encoder-side layer:
   - Flux / Flux2: `8`
   - SD3: `8`
   - Chroma: `8`
   - Sana / PixArt: `10`
4. Set **Weight** to `0.5` to start.
5. Keep defaults for the vision encoder (`dinov2_vitg14`, resolution `518`).

### For UNet models (U-REPA)

1. Open **Training -> Loss functions**.
2. Enable **U-REPA**.
3. Set **U-REPA Weight** to `0.5` (paper default).
4. Set **U-REPA Manifold Weight** to `3.0` (paper default).
5. Keep defaults for the vision encoder.

## Quick setup (config JSON / CLI)

### For DiT models (REPA)

```json
{
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 0.5,
  "crepa_encoder": "dinov2_vitg14",
  "crepa_encoder_image_size": 518
}
```

### For UNet models (U-REPA)

```json
{
  "urepa_enabled": true,
  "urepa_lambda": 0.5,
  "urepa_manifold_weight": 3.0,
  "urepa_model": "dinov2_vitg14",
  "urepa_encoder_image_size": 518
}
```

## Key differences: REPA vs U-REPA

| Aspect | REPA (DiT) | U-REPA (UNet) |
|--------|-----------|---------------|
| Architecture | Transformer blocks | UNet with mid-block |
| Alignment point | Early transformer layers | Mid-block (bottleneck) |
| Hidden state shape | `(B, S, D)` sequence | `(B, C, H, W)` convolutional |
| Loss components | Cosine alignment | Cosine + Manifold loss |
| Default weight | 0.5 | 0.5 |
| Config prefix | `crepa_*` | `urepa_*` |

## U-REPA specifics

U-REPA adapts REPA for UNet architectures with two key innovations:

### Mid-block alignment
Unlike DiT-based REPA which uses early transformer layers, U-REPA extracts features from the UNet's **mid-block** (bottleneck). This is where the UNet has the most semantic information compressed.

- **SDXL/Kolors**: Mid-block outputs `(B, 1280, 16, 16)` for 1024x1024 images
- **SD1.5**: Mid-block outputs `(B, 1280, 8, 8)` for 512x512 images

### Manifold loss
In addition to cosine alignment, U-REPA adds a **manifold loss** that aligns the relative similarity structure:

```
L_manifold = ||sim(y[i],y[j]) - sim(h[i],h[j])||^2_F
```

This ensures that if two encoder patches are similar, the corresponding projected patches should also be similar. The `urepa_manifold_weight` parameter (default 3.0) controls the balance between direct alignment and manifold alignment.

## Tuning knobs

### REPA (DiT models)
- `crepa_lambda`: Alignment loss weight (default 0.5)
- `crepa_block_index`: Which transformer block to tap (0-indexed)
- `crepa_spatial_align`: Interpolate tokens to match (default true)
- `crepa_encoder`: Vision encoder model (default `dinov2_vitg14`)
- `crepa_encoder_image_size`: Input resolution (default 518)

### U-REPA (UNet models)
- `urepa_lambda`: Alignment loss weight (default 0.5)
- `urepa_manifold_weight`: Manifold loss weight (default 3.0)
- `urepa_model`: Vision encoder model (default `dinov2_vitg14`)
- `urepa_encoder_image_size`: Input resolution (default 518)
- `urepa_use_tae`: Use Tiny AutoEncoder for faster decoding

## Coefficient scheduling

Both REPA and U-REPA support scheduling to decay the regularization over training:

```json
{
  "crepa_scheduler": "cosine",
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 5000,
  "crepa_lambda_end": 0.0
}
```

For U-REPA, use the `urepa_` prefix:

```json
{
  "urepa_scheduler": "cosine",
  "urepa_warmup_steps": 100,
  "urepa_cutoff_step": 5000
}
```

<details>
<summary>How it works (practitioner)</summary>

### REPA (DiT)
- Captures hidden states from a chosen transformer block
- Projects through LayerNorm + Linear to encoder dimension
- Computes cosine similarity with frozen DINOv2 features
- Interpolates spatial tokens to match if counts differ

### U-REPA (UNet)
- Registers a forward hook on UNet mid_block
- Captures convolutional features `(B, C, H, W)`
- Reshapes to sequence `(B, H*W, C)` and projects to encoder dimension
- Computes both cosine alignment and manifold loss
- Manifold loss aligns the pairwise similarity structure

</details>

<details>
<summary>Technical (SimpleTuner internals)</summary>

### REPA
- Implementation: `simpletuner/helpers/training/crepa.py` (`CrepaRegularizer` class)
- Mode detection: `CrepaMode.IMAGE` for image models, automatically set via `crepa_mode` property
- Hidden states stored in `crepa_hidden_states` key of model output

### U-REPA
- Implementation: `simpletuner/helpers/training/crepa.py` (`UrepaRegularizer` class)
- Mid-block capture: `simpletuner/helpers/utils/hidden_state_buffer.py` (`UNetMidBlockCapture`)
- Hidden size inferred from `block_out_channels[-1]` (1280 for SDXL/SD1.5/Kolors)
- Only enabled for `MODEL_TYPE == ModelTypes.UNET`
- Hidden states stored in `urepa_hidden_states` key of model output

</details>

## Common pitfalls

- **Wrong model type**: REPA (`crepa_*`) is for DiT models; U-REPA (`urepa_*`) is for UNet models. Using the wrong one will have no effect.
- **Block index too high** (REPA): Lower the index if you get "hidden states not returned" errors.
- **VRAM spikes**: Try a smaller encoder (`dinov2_vits14` + image size `224`) or enable `use_tae` for decoding.
- **Manifold weight too high** (U-REPA): If training becomes unstable, reduce `urepa_manifold_weight` from 3.0 to 1.0.

## References

- [REPA paper](https://arxiv.org/abs/2402.17750) - Representation Alignment for Generation
- [U-REPA paper](https://arxiv.org/abs/2410.xxxxx) - Universal REPA for UNet architectures (NeurIPS 2025)
- [DINOv2](https://github.com/facebookresearch/dinov2) - Self-supervised vision encoder
