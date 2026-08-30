# NextLat

NextLat is an auxiliary training objective that teaches a transformer to make its hidden states predictive of the next hidden state.

The original Next-Latent Prediction paper studies language-style transformers and argues that standard next-token prediction does not strongly require the model to compress history into stable internal states. NextLat adds a self-supervised latent-space transition objective: from the current hidden state, predict the next hidden state. In SimpleTuner, this idea is adapted as an experimental regularizer for supported transformer model families.

Inference is unchanged. NextLat adds training loss and a small predictor module; it does not require a different sampler.

## ELI5

Standard training says: "given what you have seen, predict the next output."

NextLat adds: "also make your internal notes good enough to predict your next internal notes."

For image, video, and audio models, those "internal notes" are hidden tokens inside the transformer. If the model learns smooth hidden-state transitions, it may learn a more coherent plan for how the sample should evolve across tokens, frames, patches, or RVQ code positions.

## What NextLat Changes

NextLat adds a small predictor beside the main model. During training:

1. SimpleTuner captures hidden states from one transformer block.
2. The predictor receives each hidden token except the last.
3. It predicts the following hidden token.
4. The real following hidden token is detached and used as the target.
5. The resulting auxiliary loss is added to the normal training loss.

The base model still trains on its normal objective. NextLat is a side objective that encourages useful hidden-state dynamics.

## Pseudocode

```text
for each batch:
    prediction = model(batch)
    main_loss = normal_training_loss(prediction, target)

    hidden = captured_hidden_states_from_selected_block
    current_hidden = hidden tokens 0..N-2
    next_hidden = hidden tokens 1..N-1

    predicted_next_hidden = nextlat_predictor(current_hidden)
    nextlat_loss = distance(predicted_next_hidden, stop_gradient(next_hidden))

    total_loss = main_loss + nextlat_weight * nextlat_loss
    train_on(total_loss)
```

If a model family exposes a compatible logits head, SimpleTuner can also add optional KL agreement:

```text
predicted_logits = logits_head(predicted_next_hidden)
target_logits = logits_head(stop_gradient(next_hidden))
kl_loss = agreement_loss(predicted_logits, target_logits)

total_loss += nextlat_kl_weight * kl_loss
```

Most users should leave the KL weight at `0` unless a model guide recommends it.

## SimpleTuner Behavior

SimpleTuner's NextLat implementation is intentionally narrow:

- It works on transformer families that expose hidden states.
- It captures one block, chosen by `nextlat_block_index`.
- `-1` means the final supported block.
- It flattens image, video, audio, or token hidden states into a sequence before comparing adjacent tokens.
- It predicts one step ahead in hidden-token order.
- The target hidden state is detached, so the auxiliary predictor learns to match the model state without pulling the target state backward through itself.
- The predictor is saved as an extra trainable module when the training mode supports saving it.

NextLat requires a training setup that optimizes and saves the predictor. In practice, use standard PEFT LoRA or full-model training unless the model guide states another adapter mode is supported.

## Quick Setup

### WebUI

1. Open **Training → Loss functions**.
2. Enable **NextLat**.
3. Keep **NextLat Block Index** at `-1` for the first run.
4. Set **NextLat Weight** to a small positive value.
5. Leave **NextLat State Loss** at `smooth_l1`.
6. Leave **NextLat KL Weight** at `0` unless the model guide recommends it.

### Config JSON / CLI

```json
{
  "nextlat_enabled": true,
  "nextlat_block_index": -1,
  "nextlat_weight": 0.05,
  "nextlat_state_loss": "smooth_l1",
  "nextlat_kl_weight": 0.0
}
```

## Settings

- `nextlat_enabled`: turns NextLat on.
- `nextlat_block_index`: zero-based transformer block to capture. `-1` uses the final supported block.
- `nextlat_weight`: multiplier for the auxiliary hidden-state prediction loss. Must be greater than zero when NextLat is enabled.
- `nextlat_state_loss`: distance function for hidden-state prediction. `smooth_l1` is the default; `mse` is also available.
- `nextlat_kl_weight`: optional KL agreement loss when the model family provides a compatible logits head for predicted hidden states.

## Choosing Values

Start small. NextLat is a regularizer, not a replacement for the main objective.

| Situation | Suggested start |
| --- | --- |
| First transformer LoRA run | `nextlat_block_index=-1`, `nextlat_weight=0.02` to `0.05` |
| AR/RVQ planner | late block, `smooth_l1`, small positive weight |
| Video transformer | middle-to-late block if the final block feels too restrictive |
| Unstable auxiliary loss | lower `nextlat_weight` before changing the block |
| Model guide recommends KL | set `nextlat_kl_weight` only to the documented value |

Use fixed validation prompts or clips when tuning. A good NextLat setting should improve stability or coherence without making the model underfit the main target.

## Logs

When NextLat is active, logs may include:

- `nextlat_loss`: weighted auxiliary loss added to the training objective.
- `nextlat_state_loss`: raw hidden-state prediction loss.
- `nextlat_kl_loss`: optional KL term, only when enabled and supported.

The raw state loss is mainly useful for trend tracking. It does not have to match the scale of the main diffusion, flow, or token loss.

## Compatibility

See the feature table in the [Quick Start](../QUICKSTART.md) for current family-level support.

The general requirements are:

- The model must expose transformer hidden states.
- The selected block must exist and be capturable.
- The captured sequence must contain at least two hidden tokens.
- The training mode must save the NextLat predictor.

NextLat naturally pairs with features that already capture hidden states, such as LayerSync, Internal Guidance, and CREPA, but it still adds memory pressure because hidden states must remain available until the auxiliary loss is computed.

## What To Expect

NextLat is most likely to help when the model benefits from coherent internal transitions:

- autoregressive audio-code or RVQ planners
- video transformers with temporal structure
- image transformers where token order carries useful spatial structure
- multimodal models that need a stable internal plan

It is less likely to help on very small experiments where the auxiliary objective dominates the main loss, or on model families that do not expose useful hidden states.

## Practical Advice

- Start with a short ablation run before committing to a long training job.
- Keep `nextlat_weight` small; raise it only if validation improves.
- Prefer `smooth_l1` unless you have a reason to make large hidden-state errors more expensive.
- Use `-1` first, then try a middle-to-late block if the final block over-constrains training.
- Leave KL disabled unless the model family documentation says it is wired up.
- If VRAM increases too much, lower batch size or disable other hidden-state regularizers.

## References

- [Next-Latent Prediction paper](https://arxiv.org/abs/2511.05963)
- [NextLat reference code](https://github.com/JaydenTeoh/NextLat)
