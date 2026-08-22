# Explorative Modeling (XM)

Explorative Modeling, abbreviated as XM in SimpleTuner, is a training-time technique that lets the model try more than one possible hidden choice for the same supervised example, then learns only from the choice that best matches the target.

The original Explorative Modeling work frames this as adding a third scaling axis to generative training: besides more data and more parameters, the model can spend extra training compute exploring more candidate generations. In SimpleTuner, XM is implemented as an experimental training objective for supported image, video, audio, and autoregressive model families.

Inference is unchanged. XM only changes how a training batch is built, scored, and reduced into a loss.

## ELI5

Imagine asking a student to draw a target image, but allowing them to make four rough attempts before grading. Instead of averaging all four attempts, you grade the one that is closest to the target and teach from that attempt.

That is the core XM idea:

1. Make several candidates for the same training sample.
2. Run the model on all candidates.
3. Score each candidate against the real target.
4. Keep the best candidate for that sample or token block.
5. Backpropagate only the selected loss.

This helps when the target can be explained in several valid ways. A single forced path can teach the model to average possibilities. Multiple explored paths let the model commit to one plausible mode.

## What XM Changes

XM does not add a new inference sampler, a new checkpoint format, or a second teacher model. It changes training selection:

- Standard training samples one candidate and learns from it.
- XM samples `K` candidates and learns from the lowest-loss candidate.
- Higher `K` gives the model more exploration, but costs more training compute.

For diffusion and flow models, the candidate is usually the noise used to construct the noised latent at the sampled timestep.

For autoregressive token models, such as RVQ/audio planners, the candidate is a learned route embedding that gives the model several possible internal paths through the same supervised token sequence.

## SimpleTuner Behavior

### Diffusion and Flow Models

For supported diffusion or flow-matching families, use `xm_training_target=noise`.

SimpleTuner:

1. Samples the normal training timestep or sigma.
2. Repeats the batch `xm_candidate_count` times.
3. Generates a different noise tensor for each repeated candidate.
4. Builds noised latents from each candidate noise.
5. Runs the model on the expanded candidate batch.
6. Computes the normal training loss for each candidate.
7. Selects the lowest-loss candidate per original sample.
8. Backpropagates the selected loss.

The model still learns the same prediction type it normally learns: flow velocity, epsilon, v-prediction, or sample prediction depending on the family.

### Autoregressive and RVQ Models

For supported autoregressive planners, use `xm_training_target=route`.

SimpleTuner:

1. Adds a small learned route embedding table with one route per XM candidate.
2. Repeats each supervised token sequence across route candidates.
3. Inserts the route signal into the model input.
4. Computes token losses for each route.
5. Selects the best route for the whole sample or for configured token blocks.
6. Backpropagates only the selected route loss.

This is useful for global language-model style planners that predict RVQ audio codes or other discrete token streams. The route embedding gives the model multiple internal explanations for the same target sequence without changing inference-time decoding.

## Pseudocode

```text
for each batch:
    candidates = []

    for candidate_id in 1..K:
        candidate_input = make_candidate(batch, candidate_id)
        prediction = model(candidate_input)
        loss = compare(prediction, target)
        candidates.append(loss)

    selected_loss = minimum_loss_per_sample_or_block(candidates)
    train_on(selected_loss)
```

For diffusion:

```text
candidate_input = add_noise(clean_latent, random_noise_candidate, timestep)
loss = diffusion_or_flow_loss(model(candidate_input), training_target)
```

For autoregressive route selection:

```text
candidate_input = add_route_embedding(token_sequence, route_candidate)
loss = token_loss(model(candidate_input), target_tokens)
```

## Quick Setup

### WebUI

1. Open **Training → Loss functions**.
2. Enable **XM**.
3. Set **XM Candidates** to `2` or `4`.
4. Choose **XM Training Target**:
   - `noise` for diffusion or flow models.
   - `route` for autoregressive/RVQ planners.
5. Keep **XM Selection Scope** at `sample` unless the model guide recommends block selection.
6. Leave **XM Block Size** at `0` unless using route-based block selection.

### Config JSON / CLI

```json
{
  "xm_enabled": true,
  "xm_candidate_count": 4,
  "xm_training_target": "noise",
  "xm_selection_scope": "sample",
  "xm_block_size": 0
}
```

For route-based AR/RVQ training:

```json
{
  "xm_enabled": true,
  "xm_candidate_count": 4,
  "xm_training_target": "route",
  "xm_selection_scope": "block",
  "xm_block_size": 16
}
```

## Settings

- `xm_enabled`: turns XM on.
- `xm_candidate_count`: number of candidates per training sample. Must be at least `2` when XM is enabled. Start with `2`; use `4` when you have enough training throughput.
- `xm_training_target`: candidate type. Use `noise` for diffusion/flow models and `route` for autoregressive token planners.
- `xm_selection_scope`: winner selection granularity. `sample` chooses one winner for the whole sample. `block` chooses winners over token or frame blocks when the model family supports it.
- `xm_block_size`: token or frame span for block-level selection. `0` means the full supervised sequence.

## Choosing Values

Start conservatively:

| Situation | Suggested start |
| --- | --- |
| Image or video diffusion LoRA | `xm_candidate_count=2`, `xm_training_target=noise`, `xm_selection_scope=sample` |
| Larger batch or high ambiguity dataset | Try `xm_candidate_count=4` |
| RVQ/audio planner route selection | `xm_training_target=route`, `xm_selection_scope=block`, block size from the model guide |
| First run on a new family | Keep block size `0` and compare validation against a non-XM baseline |

Increasing candidates raises compute. Until model-specific batching is optimized further, expect cost to scale roughly with the number of candidates.

## Logs

When XM is active, training logs may include:

- `xm_loss`: selected loss after candidate choice.
- `xm_candidate_loss_mean`: average loss across candidates before selection.
- `xm_candidate_0_wins`, `xm_candidate_1_wins`, etc.: how often each candidate won.
- `xm_route_usage` or per-route usage entries for AR/RVQ route models.

Useful signs:

- Candidates win at nonzero rates instead of one candidate always winning.
- Validation improves at the same or slightly higher training loss.
- Route usage is not completely collapsed for long periods.

Concerning signs:

- One candidate wins almost always from the start.
- Loss drops sharply but validation gets worse.
- Memory or step time increases beyond what your batch size can tolerate.

## Compatibility

See the feature table in the [Quick Start](../QUICKSTART.md) for current family-level support.

The general rules are:

- Diffusion/flow XM uses noise candidates and sample-level selection.
- AR/RVQ XM uses route candidates and may support block-level selection.
- Unsupported families fail explicitly rather than silently ignoring the option.

For diffusion noise-candidate XM, SimpleTuner currently treats these features as incompatible unless a model family explicitly states otherwise:

- TwinFlow
- Scheduled Sampling
- `input_perturbation`
- CREPA self-flow
- stochastic segmentation masked loss

Mask-style inpainting loss can be used where the model family supports it. Segmentation masks are stricter because stochastic segment selection would make candidate comparison ambiguous.

## How It Relates to Other Features

- **MixFlow** changes the training trajectory for flow models. XM changes candidate selection at a fixed supervised target.
- **Diff2Flow** changes the target used by legacy diffusion models. XM can select candidates before reducing the loss where supported.
- **NextLat** regularizes hidden-state dynamics. XM chooses among candidate routes or noises.
- **LayerSync and CREPA** align representations. XM is about selecting the most explanatory candidate.

## Practical Advice

- Use fixed validation seeds when comparing XM against a baseline.
- Lower the batch size if `xm_candidate_count` causes VRAM pressure.
- Do not judge XM only by training loss. Since it chooses easier candidate paths, validation quality and sample diversity matter more.
- For AR/RVQ models, avoid block size `1` unless the model guide explicitly recommends it. Per-token route switching can be too unstable.
- Keep the first run short. XM is easy to ablate: same model, same dataset, same seed, only XM on/off.

## References

- [Explorative Modeling project page](https://explorative-modeling.github.io/)
- [Explorative Modeling paper](https://arxiv.org/abs/2607.27372)
