# Slider LoRA Targeting

This flag lets SimpleTuner aim LoRA training at the “slider-friendly” parts of the UNet: self-attention (`attn1`), convolutions, and time-embedding projections. It mirrors the Concept Sliders choice to avoid cross-attention so language features stay untouched.

- Enable with `--slider_lora_target true` (or `"slider_lora_target": true` in config).
- Only applies to standard PEFT LoRA. LyCORIS users should set targets in their `lycoris_config.json`.
- Assistant LoRA stacks are left alone; only the trainable adapter switches targets.
- With ControlNet enabled, the slider targets take precedence over the ControlNet-specific list.

## Default slider targets

Most models share this pattern:

```
[
  "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
  "attn1.to_qkv", "to_qkv",
  "proj_in", "proj_out",
  "conv_in", "conv_out",
  "time_embedding.linear_1", "time_embedding.linear_2"
]
```

Special cases:

- Kandinsky 5 (image/video): `["attn1.to_query", "attn1.to_key", "attn1.to_value", "conv_in", "conv_out", "time_embedding.linear_1", "time_embedding.linear_2"]`

## Model coverage

- Uses the default slider list (self-attn/conv/time): SD1.x, SDXL, Lumina2, Sana, HiDream, Wan, Kolors, LTXVideo, Qwen-Image, DeepFloyd, Stable Cascade, Cosmos, SD3, Z-Image/Turbo, Ace Step, and other to_q/to_k-based families.
- Flux/Flux2/Chroma/AuraFlow (MMDiT/DiT-style) use visual-only targets to avoid language branches: `["to_q", "to_k", "to_v", "to_out.0", "to_qkv"]` (Flux2 includes `attn.to_*`/`attn.to_qkv_mlp_proj`).
- Uses the Kandinsky list: Kandinsky5 Image/Video models.

If a model is missing a match in its checkpoints, fallback is to the standard LoRA targets.

## Quick usage

```json
{
  "model_type": "lora",
  "lora_type": "standard",
  "slider_lora_target": true,
  "lora_rank": 16,
  "learning_rate": 5e-5
}
```

## LyCORIS (manual targeting)

Set slider-friendly targets directly in `lycoris_config.json` (SimpleTuner will not auto-route LyCORIS):

<details>
<summary>Most models (attn1/to_q style, including Stable Diffusion, SDXL, SD3, Wan, HiDream, Z-Image, DeepFloyd, Stable Cascade, etc.)</summary>

```json
{
  "algo": "lokr",
  "multiplier": 1.0,
  "linear_dim": 4,
  "linear_alpha": 1,
  "apply_preset": {
    "target_module": [
      "attn1.to_q",
      "attn1.to_k",
      "attn1.to_v",
      "attn1.to_out.0",
      "conv_in",
      "conv_out",
      "time_embedding.linear_1",
      "time_embedding.linear_2"
    ]
  }
}
```

Works for: SD1.x, SDXL, Flux/Flux2, Lumina2, Chroma, AuraFlow, Sana, HiDream, Wan, Kolors, PixArt, LTXVideo, Qwen-Image, DeepFloyd, Cosmos, SD3, Z-Image/Turbo, Ace Step.
</details>

<details>
<summary>Kandinsky 5 (Image/Video)</summary>

```json
{
  "algo": "lokr",
  "multiplier": 1.0,
  "linear_dim": 4,
  "linear_alpha": 1,
  "apply_preset": {
    "target_module": [
      "attn1.to_query",
      "attn1.to_key",
      "attn1.to_value",
      "conv_in",
      "conv_out",
      "time_embedding.linear_1",
      "time_embedding.linear_2"
    ]
  }
}
```
</details>

<details>
<summary>Flux / Flux2 / Chroma / AuraFlow (MMDiT/DiT visual stream)</summary>

```json
{
  "algo": "lokr",
  "multiplier": 1.0,
  "linear_dim": 4,
  "linear_alpha": 1,
  "apply_preset": {
    "target_module": [
      "attn.to_q",
      "attn.to_k",
      "attn.to_v",
      "attn.to_out.0",
      "attn.to_qkv_mlp_proj"
    ]
  }
}
```

For Flux/Chroma without the `attn.` prefix, drop it: `["to_q","to_k","to_v","to_out.0","to_qkv"]`. Avoid `add_*` projections to keep text/context branches untouched.
</details>

Adjust `target_module` if your checkpoint uses different naming; keep it scoped to self-attn/conv/time-embed layers for slider-style behaviour.
