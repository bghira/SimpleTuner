# Flow-DPO और Masked Flow-DPO

Flow-DPO flow-matching models के लिए experimental distillation method है, जो preferred/rejected paired samples से low-rank adapter train करता है। SimpleTuner में यह केवल LoRA/LyCORIS के लिए है। Full-model Flow-DPO supported नहीं है, और distillation methods के साथ text encoder training blocked है।

SimpleTuner existing reference dataset system reuse करता है। Normal `image` या `video` dataset preferred sample देता है, और paired `conditioning` dataset `conditioning_type=reference_strict` के साथ rejected sample देता है। Pairing rules के लिए [`conditioning_type`](../DATALOADER.hi.md#conditioning_type) और [`conditioning_data`](../DATALOADER.hi.md#conditioning_data) देखें।

## यह क्या करता है

हर batch में SimpleTuner:

1. Adapter-enabled model को preferred latents पर चलाता है।
2. उसी prompt, noise, और timestep से rejected latents पर चलाता है।
3. LoRA/LyCORIS adapter disable करके frozen reference के रूप में दोनों predictions चलाता है।
4. Flow-DPO margin loss लगाता है।

```text
win_adv  = L(reference_win, target_win) - L(policy_win, target_win)
lose_adv = L(policy_lose, target_lose) - L(reference_lose, target_lose)
loss     = -logsigmoid(beta / 2 * (win_adv + lose_adv))
```

Flow-matching target `noise - latents` है।

## Masked Flow-DPO

अगर batch में `conditioning_type=mask` या `conditioning_type=segmentation` dataset भी है, तो SimpleTuner DPO prediction errors पर reduction से पहले mask apply करता है। इससे preference signal preferred और rejected samples के बदलने वाले region पर केंद्रित रहता है।

`anchor_alpha` preferred side पर global MSE regularizer जोड़ता है, adapter-enabled और adapter-disabled preferred prediction के बीच। यह anchor win-only है और mask use नहीं करता।

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

- `norm_type=sum` common Flow-DPO formulation से match करता है।
- `auto_beta=true` running margin magnitude से beta adapt करता है, जो small paired datasets में उपयोगी है।
- `flow_timesteps_mode=fixed-list` `flow_custom_timesteps` से random sample करता है।
- `flow_timesteps_mode=round-robin` `flow_custom_timesteps` को cycle करता है।
- `sft_loss_weight` default `0.0` है, इसलिए normal diffusion loss mix नहीं होती।

## Dataset Shape

Rejected dataset को preferred dataset से `reference_strict` के रूप में pair करें:

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

Masked Flow-DPO के लिए mask conditioning dataset को भी उसी `conditioning_data` list में जोड़ें।

## Limits

Flow-DPO currently requires:

- Flow-matching model.
- `model_type=lora`.
- Paired `reference_strict` conditioning dataset.
- No text encoder training.

यह model weights की दूसरी full copy load नहीं करता। Reference pass trained adapter को disable करता है, LyCORIS multipliers सहित, फिर policy path के लिए re-enable करता है।
