# Flow-DPO and Masked Flow-DPO

Flow-DPO is an experimental distillation method for flow-matching models that trains a low-rank adapter from paired preferred/rejected samples. It is intended for LoRA/LyCORIS training only. SimpleTuner does not support full-model Flow-DPO, and text encoder training is blocked for all distillation methods.

SimpleTuner uses the existing reference dataset system for the rejected side of each pair. The normal `image` or `video` dataset supplies the preferred sample, and a paired `conditioning` dataset with `conditioning_type=reference_strict` supplies the rejected sample. See the [`conditioning_type`](../DATALOADER.md#conditioning_type) and [`conditioning_data`](../DATALOADER.md#conditioning_data) dataloader sections for the reference pairing rules.

## What It Does

For each batch, SimpleTuner:

1. Runs the adapter-enabled model on the preferred latents.
2. Runs the adapter-enabled model on the rejected latents using the same prompt, noise, and timestep.
3. Disables the LoRA/LyCORIS adapter and runs the same two predictions as the frozen reference.
4. Applies the Flow-DPO margin loss:

```text
win_adv  = L(reference_win, target_win) - L(policy_win, target_win)
lose_adv = L(policy_lose, target_lose) - L(reference_lose, target_lose)
loss     = -logsigmoid(beta / 2 * (win_adv + lose_adv))
```

For flow-matching models, the target is `noise - latents`.

## Masked Flow-DPO

If the batch also includes a `conditioning` dataset with `conditioning_type=mask` or `conditioning_type=segmentation`, SimpleTuner applies that mask to the DPO prediction errors before reducing them. This concentrates the preference signal on the region that differs between the preferred and rejected samples.

`anchor_alpha` adds an optional global MSE regularizer between adapter-enabled and adapter-disabled predictions on both the preferred and rejected samples. The anchor is unmasked, so it constrains whole-frame drift rather than only the masked region.

## Configuration

Minimal setup:

```bash
--model_type=lora
--distillation_method=flow_dpo
--flow_custom_timesteps=801,694,548,338
--flow_timesteps_mode=round-robin
```

Common `distillation_config` keys:

```json
{
  "flow_dpo": {
    "beta": 1.0,
    "auto_beta": true,
    "auto_beta_target_gf": 0.2,
    "auto_beta_decay": 0.99,
    "norm_type": "sum",
    "mask_dilate": 1,
    "anchor_alpha": 0.0,
    "sft_loss_weight": 0.0
  }
}
```

- `norm_type=sum` matches the usual Flow-DPO formulation. `mean` averages all latent elements, and `masked_mean` averages over active mask elements when a mask is present.
- `auto_beta=true` adapts beta from the running margin magnitude. This is useful for small paired datasets where a fixed beta can saturate the sigmoid.
- `flow_timesteps_mode=fixed-list` randomly samples from `flow_custom_timesteps`.
- `flow_timesteps_mode=round-robin` cycles through `flow_custom_timesteps` for even timestep coverage. Distributed ranks are offset from each other, and checkpoints save the cursor so resumed runs continue the same microbatch sequence.
- `sft_loss_weight` defaults to `0.0`, so Flow-DPO does not mix in the normal diffusion loss unless you explicitly request it.

SimpleTuner logs the core Flow-DPO health values: beta, margin, win/lose advantages, policy/reference errors, negative-margin percentage, and gradient factor. The extended reward-hacking detector metrics shown in the original demo model card are analysis tooling from that release and are not all emitted by SimpleTuner yet.

## Dataset Shape

The rejected dataset must be paired to the preferred dataset with `reference_strict`:

```json
[
  {
    "id": "preferred",
    "dataset_type": "image",
    "type": "local",
    "instance_data_dir": "/data/win",
    "conditioning_data": ["rejected"]
  },
  {
    "id": "rejected",
    "dataset_type": "conditioning",
    "conditioning_type": "reference_strict",
    "type": "local",
    "instance_data_dir": "/data/lose",
    "source_dataset_id": "preferred"
  }
]
```

Add a mask conditioning dataset to the same `conditioning_data` list when using masked Flow-DPO.

## Limits

Flow-DPO currently requires:

- A flow-matching model.
- `model_type=lora`.
- A paired `reference_strict` conditioning dataset.
- No text encoder training.

It does not load a second full copy of the model weights. The reference pass disables the trained adapter, including LyCORIS multipliers, then re-enables it for the trainable policy path.
