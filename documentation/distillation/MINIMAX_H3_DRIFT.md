# MiniMax H3 Drift Distillation

MiniMax H3 is a distilled flow-matching video/audio model. During normal LoRA or LyCORIS training, the adapter learns the dataset target directly, but the base model's distilled behavior can drift: guidance behavior, modality balance, and the packed video/audio sequence layout can move away from the frozen checkpoint even when the training loss looks reasonable.

H3 drift distillation keeps the adapter close to the frozen MiniMax H3 prediction while still allowing the normal supervised fine-tuning loss to teach the dataset. It is not a separate student/teacher checkpoint and it does not cache teacher latents. SimpleTuner uses the same model twice on the same prepared batch:

1. Run the trainable adapter path and compute the normal MiniMax H3 SFT loss.
2. Temporarily disable the adapter.
3. Run the frozen base path under `torch.no_grad()` with the same prompt embeddings, noised latents, timesteps, masks, and packed layout.
4. Compare the adapter prediction to the frozen-base prediction with a video/audio MSE loss.
5. Re-enable the adapter and backpropagate the combined loss.

The combined objective is:

```text
total = sft_loss_weight * normal_h3_loss + loss_weight * frozen_base_prediction_mse
```

When an inner distiller is configured, the objective becomes:

```text
total = inner_distiller_loss + sft_loss_weight * normal_h3_loss + loss_weight * frozen_base_prediction_mse
```

This makes the adapter learn the dataset while being penalized for changing unrelated base behavior too aggressively.

## When To Use It

Use `h3_drift` for MiniMax H3 LoRA or LyCORIS training unless you are intentionally trying to remove or replace the original distillation behavior. It is especially useful for:

- short concept or style LoRAs where the adapter should not change the model's guidance behavior;
- FL2VA or Ref2VA training where the model must keep its packed conditioning layout stable;
- joint audio/video training where video can otherwise dominate the loss simply because it has more elements;
- quantized H3 flavours such as `convrot-int8` and `convrot-int4`, where adapter-only training is the expected path.

Do not use it for full-rank MiniMax H3 training. The implementation intentionally rejects non-LoRA training because there is no frozen base path to compare against once the full transformer weights are being updated.

## Quick Config

Add the distiller to a MiniMax H3 LoRA config:

```json
{
  "model_family": "minimaxh3",
  "model_flavour": "convrot-int8",
  "model_type": "lora",

  "distillation_method": "h3_drift",
  "distillation_config": {
    "h3_drift": {
      "loss_weight": 0.5,
      "sft_loss_weight": 1.0,
      "balance": "token",
      "video_weight": 1.0,
      "audio_weight": 1.0
    }
  }
}
```

The checked-in MiniMax H3 example configs enable this by default. `loss_weight: 0.5` is a practical starter value: the normal dataset loss remains primary, but drift has enough weight to catch adapters that quickly move away from the base prediction.

## Configuration Keys

- `loss_weight`: multiplier for the frozen-base prediction loss. Start at `0.25` to `0.5` for narrow visual LoRAs. Use `1.0` when validation shows the adapter is breaking the base model behavior or when training longer multimodal adapters.
- `sft_loss_weight`: multiplier for the normal MiniMax H3 training loss. Keep this at `1.0` for ordinary fine-tuning. Setting it to `0.0` turns the run into pure base-prediction following and is usually only useful for debugging.
- `balance`: `token` or `modality`. `token` averages by valid element count, so video naturally dominates video/audio batches. `modality` averages the video and audio modality means after applying `video_weight` and `audio_weight`.
- `video_weight`: multiplier for the video prediction drift term.
- `audio_weight`: multiplier for the audio prediction drift term.
- `inner_distillation_method`: optional nested SimpleTuner distiller to run inside H3 drift, such as `anyflow`, `dmd`, `perflow`, `flow_dpo`, or `self_forcing`.
- `inner_distillation_config`: configuration mapping passed to the nested distiller.

The defaults are `loss_weight: 1.0`, `sft_loss_weight: 1.0`, `balance: "token"`, and equal video/audio weights. The examples set a lighter drift value because most downstream LoRAs still need the dataset target to move the adapter.

## Composing Another Distiller

`h3_drift` can wrap another distiller when you want a step-distillation or preference objective while still preserving MiniMax H3's frozen-base behavior. For example, this runs AnyFlow target preparation first, then adds the H3 drift reference loss:

```json
{
  "distillation_method": "h3_drift",
  "distillation_config": {
    "h3_drift": {
      "loss_weight": 0.5,
      "sft_loss_weight": 1.0,
      "inner_distillation_method": "anyflow",
      "inner_distillation_config": {
        "stage": "forward",
        "diffusion_ratio": 0.5,
        "consistency_ratio": 0.25,
        "central_difference_epsilon": 0.005,
        "loss_weight": 1.0
      }
    }
  }
}
```

The wrapper delegates batch preparation, validation scheduler hooks, distillation cache generation, caption-batch support, and discriminator/generator lifecycle hooks to the inner distiller. The inner distiller still performs its own compatibility checks, so a method that does not support MiniMax H3 will fail during setup instead of silently falling back.

`sft_loss_weight` remains the normal MiniMax H3 objective even when the inner distiller rewrites `target`. If the inner distiller adds FlowMap/AnyFlow timestep conditioning, H3 drift computes that SFT term with an additional adapter-enabled forward pass after removing the inner timestep keys. That keeps the standard 30-step H3 inference path anchored while the inner distiller trains the step-distilled path.

## Video And Audio Modes

MiniMax H3 can train video-only targets or joint audio/video targets:

```json
{
  "minimax_h3_target_mode": "video"
}
```

`auto` resolves to video-only. Set `minimax_h3_target_mode` globally, or `h3_target_mode` / `minimax_h3_target_mode` inside a data backend entry, to `av` when you want target audio rows included.

The drift distiller follows the prepared batch:

- video-only batches compare only `model_prediction`;
- audio/video batches compare both `model_prediction` and `audio_prediction`;
- `audio_latent_mask` masks out missing generated audio rows;
- existing sample weights and visual loss masks are respected.

If an `av` backend has no audio latents, MiniMax H3 builds zero audio rows and masks audio loss. That avoids training against fake audio targets while still allowing video training to proceed.

## Maintaining CFG-Distilled Behavior

MiniMax H3 is CFG-distilled. The base checkpoint is expected to work with `validation_guidance: 1.0`, `validation_guidance_real: 1.0`, and `validation_disable_unconditional: true`. Negative prompting is not part of the base training contract.

SimpleTuner still has pipeline support for real CFG and negative prompt encoding because users may train adapters or merged checkpoints that partially remove the original distillation. `h3_drift` is the opposite pressure: it helps keep the adapter close to the base model's distilled conditional prediction. If your goal is to teach negative prompt behavior or de-distill H3, reduce `loss_weight` or disable `h3_drift`; otherwise the reference loss will push the adapter back toward the base model.

## Reading The Logs

The distiller adds separate log values:

- `h3_drift_loss`: unweighted combined video/audio drift loss;
- `h3_drift_video_loss`: video drift mean;
- `h3_drift_audio_loss`: audio drift mean;
- `h3_drift_video_elements` and `h3_drift_audio_elements`: valid element counts after masks;
- `h3_drift_weighted_loss`: `h3_drift_loss * loss_weight`;
- `h3_drift_sft_loss`: normal MiniMax H3 loss after `sft_loss_weight`;
- `h3_drift_inner_total`: the nested distiller's `total` value, when `inner_distillation_method` is enabled;
- `total`: final loss returned to the trainer.

Use these values to diagnose whether the adapter is learning the dataset or mostly fighting the reference. If `h3_drift_loss` climbs while validation quality gets more erratic, increase `loss_weight` or reduce learning rate. If the adapter barely learns the concept, reduce `loss_weight`, raise rank, or train longer.

## Cost And Memory

H3 drift distillation adds one extra forward pass per training step, but it does not keep a second transformer in memory. The reference pass reuses the same model with adapters disabled and no gradients. When wrapping an inner FlowMap/AnyFlow distiller with `sft_loss_weight` enabled, it also runs a normal-path adapter forward for the SFT anchor. Peak VRAM still increases because activations, compilation caches, and temporary buffers can overlap with the normal step, but the cost is closer to extra forwards than a second trainable model.

This makes it compatible with the usual MiniMax H3 memory features: quantized ConvRot checkpoints, RamTorch, musubi block swap, gradient checkpointing, and attention offload. Benchmark each preset after enabling the distiller because the extra forward can change which checkpointing or attention backend is fastest.

## Troubleshooting

- **`H3 drift distillation only supports low-rank LoRA/LyCORIS training`**: set `model_type` to `lora`. Full-rank H3 drift is intentionally unsupported.
- **`H3 drift prediction contains no target modality`**: the model output did not contain a video or audio prediction. Check `minimax_h3_target_mode`, latent cache shape, and model family.
- **Audio loss is always zero**: the batch is video-only, `minimax_h3_target_mode` resolves to `video`, or `audio_latent_mask` excludes every audio row.
- **Adapter learning is too weak**: lower `loss_weight` from `0.5` to `0.25`, increase rank, or use `balance: "token"` if `modality` was over-emphasizing audio.
- **Audio quality drifts while video looks stable**: use `balance: "modality"` or raise `audio_weight` for joint audio/video training.
