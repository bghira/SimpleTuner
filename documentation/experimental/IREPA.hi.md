# iREPA

iREPA alignment path में spatial structure को बनाए रखकर representation alignment सुधारता है। यह per-token linear projector को spatial convolution से बदलता है और हर image के patch dimension पर teacher feature channels को z-score normalize करता है।

SimpleTuner backbone के अनुसार मौजूदा alignment engine चुनता है: Transformer image models REPA/CREPA उपयोग करते हैं; Transformer video models हर frame पर iREPA लगाकर CREPA temporal-neighbour loss रखते हैं; UNet image models U-REPA mid-block और manifold loss उपयोग करते हैं। Rectangular token grid clean latent shape से निकाली जाती है।

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

Transformer के लिए iREPA के साथ `crepa_enabled` या UNet के लिए `urepa_enabled` भी चालू करें। संबंधित `crepa_*` या `urepa_*` settings teacher, weight, capture layer और schedule नियंत्रित करती हैं। `0.6` latent-diffusion reference recipe है; kernel `3` paper architecture है।

iREPA को spatial patch tokens वाले hidden states और grid recovery के लिए clean latents चाहिए। Video convolution frames को आपस में mix नहीं करता।

Full-model या standard PEFT LoRA training उपयोग करें। Auxiliary projector save न कर पाने के कारण LyCORIS supported नहीं है।

संदर्भ: [What Matters for Representation Alignment: Global Information or Spatial Structure?](https://arxiv.org/abs/2512.10794)
