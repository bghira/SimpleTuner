# Slider LoRA Targeting

In this guide, we'll be training a slider-style adapter in SimpleTuner. We'll be using Z-Image Turbo because it trains quickly, ships an Apache 2.0 license, and gives great results for its size—even with distilled weights.

For the full compatibility matrix (LoRA, LyCORIS, full-rank), see the Sliders column in [documentation/QUICKSTART.md](QUICKSTART.md); this guide applies to all architectures.

Slider targeting works with standard LoRA, LyCORIS (including `full`), and ControlNet. The toggle is available in both CLI and WebUI; everything ships in SimpleTuner, no extra installs needed.

## Step 1 — Follow the base setup

- **CLI**: Walk through `documentation/quickstart/ZIMAGE.md` for environment, install, hardware notes, and the starter `config.json`.
- **WebUI**: Use `documentation/webui/TUTORIAL.md` to run the trainer wizard; pick Z-Image Turbo as usual.

Everything from those guides can be followed until you reach the point of configuring a dataset because sliders only change where adapters are placed and how data is sampled.

## Step 2 — Enable slider targets

- CLI: add `"slider_lora_target": true` (or pass `--slider_lora_target true`).
- WebUI: Model → LoRA Config → Advanced → check “Use slider LoRA targets”.

For LyCORIS, keep `lora_type: "lycoris"` and for `lycoris_config.json`, use the presets in the details section below.

## Step 3 — Build slider-friendly datasets

Concept sliders learn from a contrastive dataset of "opposites". Create small before/after pairs (4–6 pairs is enough to start, more if you have them):

- **Positive bucket**: “more of the concept” (e.g., brighter eyes, stronger smile, extra sand). Set `"slider_strength": 0.5` (any positive value).
- **Negative bucket**: “less of the concept” (e.g., dimmer eyes, neutral expression). Set `"slider_strength": -0.5` (any negative value).
- **Neutral bucket (optional)**: regular examples. Omit `slider_strength` or set it to `0`.

It's not necessary to keep filenames matched across positive/negative folders - just ensure you have an equal number of samples in each bucket.

## Step 4 — Point the dataloader at your buckets

- Use the same dataloader JSON pattern from the Z-Image quickstart.
- Add `slider_strength` to each backend entry. SimpleTuner will:
  - Rotate batches **positive → negative → neutral** so both directions stay fresh.
  - Still honor each backend’s probability, so your weighting knobs keep working.

You don’t need extra flags—just the `slider_strength` fields.

## Step 5 — Train

Use the usual command (`simpletuner train ...`) or start from the WebUI. Slider targeting is automatic once the flag is on.

## Step 6 — Validate (optional slider tweaks)

Prompt libraries can carry per-prompt adapter scales for A/B checks:

```json
{
  "plain": "regular prompt",
  "slider_plus": { "prompt": "same prompt", "adapter_strength": 1.2 },
  "slider_minus": { "prompt": "same prompt", "adapter_strength": 0.5 }
}
```

If omitted, validation uses your global strength.

---

## References & details

<details>
<summary>Why these targets? (technical)</summary>

SimpleTuner routes slider LoRAs to self-attention, conv/proj, and time-embedding layers to mimic Concept Sliders’ “leave text alone” rule. ControlNet runs still honour slider targeting. Assistant adapters stay frozen.
</details>

<details>
<summary>Default slider target lists (by architecture)</summary>

- General (SD1.x, SDXL, SD3, Lumina2, Wan, HiDream, LTXVideo, Qwen-Image, Cosmos, Stable Cascade, etc.):

  ```json
  [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn1.to_qkv", "to_qkv",
    "proj_in", "proj_out",
    "conv_in", "conv_out",
    "time_embedding.linear_1", "time_embedding.linear_2"
  ]
  ```

- Flux / Flux2 / Chroma / AuraFlow (visual stream only):

  ```json
  ["to_q", "to_k", "to_v", "to_out.0", "to_qkv"]
  ```

  Flux2 variants include `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0`, `attn.to_qkv_mlp_proj`.

- Kandinsky 5 (image/video):

  ```json
  ["attn1.to_query", "attn1.to_key", "attn1.to_value", "conv_in", "conv_out", "time_embedding.linear_1", "time_embedding.linear_2"]
  ```

</details>

<details>
<summary>LyCORIS presets (LoKr example)</summary>

Most models:

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

Flux/Chroma/AuraFlow: swap targets to `["attn.to_q","attn.to_k","attn.to_v","attn.to_out.0","attn.to_qkv_mlp_proj"]` (drop `attn.` when checkpoints omit it). Avoid `add_*` projections to keep text/context untouched.

Kandinsky 5: use `attn1.to_query/key/value` plus `conv_*` and `time_embedding.linear_*`.
</details>

<details>
<summary>How sampling works (technical)</summary>

Backends tagged with `slider_strength` are grouped by sign and sampled in a fixed cycle: positive → negative → neutral. Within each group, usual backend probabilities apply. Exhausted backends are removed and the cycle continues with what’s left.
</details>
