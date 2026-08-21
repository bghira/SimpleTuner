# iREPA

iREPA improves representation alignment by preserving spatial structure in the alignment path. It replaces the tokenwise linear projector with a spatial convolution and z-score normalizes each teacher feature channel over the image patches.

SimpleTuner applies iREPA through the existing alignment engine for the loaded backbone:

- Transformer image models use REPA/CREPA hidden-state capture.
- Transformer video models apply the convolution and normalization independently to each frame, then retain CREPA's temporal-neighbour loss.
- UNet image models use the U-REPA mid-block capture and manifold loss.

The implementation derives rectangular token grids from the clean latent shape. It does not assume square training buckets.

## Configuration

```json
{
  "irepa_enabled": true,
  "irepa_spatial_norm_alpha": 0.6,
  "irepa_projector_kernel_size": 3,
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 1.0
}
```

Enable `crepa_enabled` with iREPA for transformer backbones. Enable `urepa_enabled` with iREPA for UNet backbones. The corresponding `crepa_*` or `urepa_*` settings control the teacher, loss weight, capture point, and schedule.

- `irepa_spatial_norm_alpha=0.6` matches the latent-diffusion reference recipe. `1.0` fully removes the spatial mean.
- `irepa_projector_kernel_size=3` is the published architecture.
- Use an early transformer block for spatial alignment. The best block remains model-dependent.

## Requirements

iREPA requires a captured hidden state with spatial patch tokens and clean latents for grid recovery. The selected vision encoder must expose patch tokens. The convolution projector is attached to the trained backbone and optimized with it; it is used only by the training loss.

Use full-model training or standard PEFT LoRA. PEFT stores the projector as a trainable module in the adapter checkpoint. LyCORIS does not support the auxiliary projector and is rejected.

For video, the convolution never mixes adjacent frames. Temporal mixing remains controlled by `crepa_adjacent_distance` and `crepa_adjacent_tau`.

Reference: [What Matters for Representation Alignment: Global Information or Spatial Structure?](https://arxiv.org/abs/2512.10794)
