# Self-Flow (internal alignment)

Self-Flow is a CREPA mode that replaces the external vision encoder with a cleaner EMA teacher view of the same model. It follows the Black Forest Labs paper idea closely: train the student on a mixed tokenwise noise schedule, run the EMA teacher on a cleaner view, and align internal hidden states while keeping the normal generative loss.

Unlike REPA / VIDEO_CREPA, Self-Flow does not need DINOv2. Unlike LayerSync, it does not align two layers from the same forward pass at the same noise level. The asymmetry comes from the student seeing mixed corruption while the teacher sees the cleaner target.

> **Looking for external-encoder alignment?** See [IMAGE_REPA.md](IMAGE_REPA.md) for REPA / U-REPA and [VIDEO_CREPA.md](VIDEO_CREPA.md) for temporal CREPA.

## When to use it

- You want the BFL-style self-supervised regularizer instead of an external encoder.
- You are training a transformer family that already exposes Self-Flow hooks in SimpleTuner.
- You want the same backbone regularizer to help standard generation, editing, and multimodal training without shipping DINO checkpoints.
- You already use EMA or are willing to enable it. Self-Flow requires an EMA teacher.

Supported families currently include:

- Image / edit: `flux`, `flux2`, `sd3`, `pixart`, `sana`, `qwen_image`, `chroma`, `hidream`, `auraflow`, `lumina2`, `z_image`, `z_image_omni`, `kandinsky5_image`, `longcat_image`, `omnigen`, `ace_step`
- Video / multimodal: `wan`, `wan_s2v`, `ltxvideo`, `ltxvideo2`, `sanavideo`, `kandinsky5_video`, `hunyuanvideo`, `longcat_video`, `cosmos`, `anima`

## Quick setup (WebUI)

1. Open **Training → Loss functions**.
2. Enable **CREPA**.
3. Set **CREPA Feature Source** to `self_flow`.
4. Set **CREPA Block Index** to an earlier student block. Start with `8` on 24-layer DiTs and `10` on deeper/wider stacks.
5. Set **CREPA Teacher Block Index** to a deeper teacher block. Good starting points are `16` or `20`.
6. Keep **Weight** at `0.5` to start.
7. Set **Self-Flow Mask Ratio** to:
   - `0.25` for image models
   - `0.10` for video models
   - `0.50` for audio-heavy models such as `ace_step`
8. Make sure **EMA** is enabled.
9. Do **not** combine it with TwinFlow.

Logs will include the normal CREPA metrics (`crepa_loss`, `crepa_alignment_score`) plus the standard training loss.

## Quick setup (config JSON / CLI)

```json
{
  "use_ema": true,
  "crepa_enabled": true,
  "crepa_feature_source": "self_flow",
  "crepa_block_index": 8,
  "crepa_teacher_block_index": 16,
  "crepa_lambda": 0.5,
  "crepa_self_flow_mask_ratio": 0.25
}
```

Legacy configs can still use:

```json
{
  "crepa_self_flow": true
}
```

Prefer `crepa_feature_source=self_flow` for new configs.

## Tuning knobs

- `crepa_block_index`: student block to supervise. Earlier blocks usually work better.
- `crepa_teacher_block_index`: deeper EMA teacher block. Required for Self-Flow.
- `crepa_lambda`: alignment strength. Start at `0.5`; reduce if generations look over-regularized.
- `crepa_self_flow_mask_ratio`: fraction of tokens that receive the alternate timestep. Must stay in `[0.0, 0.5]`.
- `crepa_scheduler`, `crepa_warmup_steps`, `crepa_decay_steps`, `crepa_lambda_end`, `crepa_cutoff_step`: same scheduling controls as CREPA. They work well if you want Self-Flow to decay later in training.
- `crepa_use_backbone_features`: a different mode. Do not combine it with Self-Flow.
- `crepa_feature_source=self_flow`: preferred selector for the mode.

## Sampling / validation

Self-Flow changes training, not the basic inference algorithm.

- Training uses mixed tokenwise noise on the student and a cleaner EMA teacher view.
- Validation loss still evaluates the requested homogeneous timestep schedule.
- Normal sampling stays unchanged. You do **not** run dual-timestep masking at inference.

If you want better sampler defaults, tune them after training as you would for any other model. Self-Flow itself does not require a new sampler.

<details>
<summary>How it works (practitioner)</summary>

- Sample two timesteps and assign them across tokens with a random mask.
- Build a student view with mixed corruption and a teacher view with the cleaner timestep.
- Run the student normally and the EMA teacher under `no_grad`.
- Align an earlier student layer to a deeper teacher layer with cosine similarity while still training on the normal generative loss.
- On edit / context architectures such as `flux2`, reference tokens stay clean while target image tokens receive the mixed schedule.

</details>

<details>
<summary>Technical (SimpleTuner internals)</summary>

- Source selection lives in `simpletuner/helpers/training/crepa.py` via `CrepaFeatureSource.SELF_FLOW`.
- Shared batch builders are in `ModelFoundation._prepare_image_crepa_self_flow_batch` and `_prepare_video_crepa_self_flow_batch`.
- The EMA teacher pass is run from `ImageModelFoundation.auxiliary_loss` / `VideoModelFoundation.auxiliary_loss` through `_run_crepa_teacher_forward`.
- Validation and training-time inference now rebuild homogeneous eval batches when `custom_timesteps` are requested, so eval loss is not polluted by the mixed Self-Flow training batch.
- Families that support Self-Flow implement `supports_crepa_self_flow()` and a model-specific `_prepare_crepa_self_flow_batch()` when they need custom token handling.

</details>

## Common pitfalls

- **EMA disabled**: Self-Flow requires `use_ema=true`.
- **Teacher block unset**: Set `crepa_teacher_block_index`; startup validation will reject missing values.
- **TwinFlow enabled**: not supported together.
- **Wrong family**: only model families that implement `supports_crepa_self_flow()` can use this mode.
- **Mask ratio too high**: stay at or below `0.5`; aggressive values can make training unstable.
- **Expecting a special sampler**: inference stays standard. Self-Flow is a training regularizer, not a new generation-time schedule.
- **Confusing it with backbone mode**: `crepa_use_backbone_features=true` is not Self-Flow. Self-Flow requires the cleaner EMA teacher view.

## References

- [Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](https://bfl.ai/research/self-flow)
