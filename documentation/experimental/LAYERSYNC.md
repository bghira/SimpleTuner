# LayerSync (SimpleTuner)

LayerSync is a “teach yourself” nudge for transformer models: one layer (the student) learns to line up with a stronger layer (the teacher). It is light, self-contained, and has no extra models to download.

## When to use it

- You are training transformer families that expose hidden states (e.g., Flux/Flux Kontext/Flux.2, PixArt Sigma, SD3/SDXL, Sana, Wan, Qwen Image/Edit, Hunyuan Video, LTXVideo, Kandinsky5 Video, Chroma, ACE-Step, HiDream, Cosmos/LongCat/Z-Image/Auraflow).
- You want a built-in regularizer without shipping an external teacher checkpoint.
- You are seeing mid-training drift or unstable heads and want to pull a mid-layer back toward a deeper teacher.
- You have a bit of VRAM headroom to hold student/teacher activations for the current step.

## Quick setup (WebUI)

1. Open **Training → Loss functions**.
2. Enable **LayerSync**.
3. Set **Student Block** to a mid-layer and **Teacher Block** to a deeper one. On 24-layer DiT-style models (Flux, PixArt, SD3), start with `8` → `16`; on shorter stacks, keep the teacher a few blocks deeper than the student.
4. Leave **Weight** at `0.2` (defaults to this when LayerSync is enabled).
5. Train normally; logs will include `layersync_loss` and `layersync_similarity`.

## Quick setup (config JSON / CLI)

```json
{
  "layersync_enabled": true,
  "layersync_student_block": 8,
  "layersync_teacher_block": 16,
  "layersync_lambda": 0.2
}
```

## Tuning knobs

- `layersync_student_block` / `layersync_teacher_block`: 1-based-friendly indexing; we try `idx-1` first, then `idx`.
- `layersync_lambda`: scales the cosine loss; must be > 0 when enabled (defaults to `0.2`).
- Teacher defaults to the student block when omitted, making the loss self-similarity.
- VRAM: activations for both layers are kept until the aux loss runs; disable LayerSync (or CREPA) if you need to free memory.
- Plays fine with CREPA/TwinFlow; they share the same hidden-state buffer.

<details>
<summary>How it works (practitioner)</summary>

- Computes negative cosine similarity between flattened student and teacher tokens; higher weight pushes the student toward the teacher’s features.
- Teacher tokens are always detached to avoid gradients flowing backward.
- Handles 3D `(B, S, D)` and 4D `(B, T, P, D)` hidden states for both image and video transformers.
- Upstream option mapping:
  - `--encoder-depth` → `--layersync_student_block`
  - `--gt-encoder-depth` → `--layersync_teacher_block`
  - `--reg-weight` → `--layersync_lambda`
- Defaults: off by default; when enabled and unset, `layersync_lambda=0.2`.

</details>

<details>
<summary>Technical (SimpleTuner internals)</summary>

- Implementation: `simpletuner/helpers/training/layersync.py`; invoked from `ModelFoundation._apply_layersync_regularizer`.
- Hidden-state capture: triggered when LayerSync or CREPA requests it; transformers store states as `layer_{idx}` via `_store_hidden_state`.
- Layer resolution: tries 1-based then 0-based indices; errors if the requested layers are missing.
- Loss path: normalizes student/teacher tokens, computes mean cosine similarity, logs `layersync_loss` and `layersync_similarity`, and adds the scaled loss to the main objective.
- Interaction: runs after CREPA so both can reuse the same buffer; clears the buffer afterward.

</details>

## Common pitfalls

- Missing student block → startup error; set `layersync_student_block` explicitly.
- Weight ≤ 0 → startup error; keep the default `0.2` if unsure.
- Requesting blocks deeper than the model exposes → “LayerSync could not find layer” errors; lower the indices.
- Enabling on models that do not expose transformer hidden states (Kolors, Lumina2, Stable Cascade C, Kandinsky5 Image, OmniGen) will fail; stick to transformer-backed families.
- VRAM spikes: lower the block indices or disable CREPA/LayerSync to free the hidden-state buffer.

Use LayerSync when you want a cheap, built-in regularizer to gently steer intermediate representations without adding external teachers.
