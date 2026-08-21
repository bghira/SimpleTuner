# Internal Guidance

Internal Guidance adds an auxiliary denoising head to an early diffusion-transformer block. The head predicts the same target as the final head. During sampling, their difference provides guidance without another model or an unconditional pass.

## Configuration

```json
{
  "internal_guidance_enabled": true,
  "internal_guidance_loss_weight": 0.5,
  "internal_guidance_block_index": 7,
  "validation_internal_guidance_scale": 1.4
}
```

- `internal_guidance_loss_weight`: multiplier for the intermediate loss. The reference implementation uses `0.5`.
- `internal_guidance_block_index`: zero-based capture block. When omitted, SimpleTuner selects one quarter of the transformer depth.
- `validation_internal_guidance_scale`: sampling scale. `1.0` disables extrapolation. The reference implementation uses `1.4` for its primary result.

The training loss is:

```text
loss = final_loss + weight * intermediate_loss
```

The guided prediction is:

```text
guided = intermediate + scale * (final - intermediate)
```

The projection is zero initialized and saved with standard PEFT LoRA adapters or full-model checkpoints. LyCORIS is rejected because it does not preserve the auxiliary module through the PEFT `modules_to_save` path. Autoregressive models, UNets, and the ACE-Step v1.5 decoder layout are not supported.

## Diffusers inference

Attach the head before loading the remaining adapter tensors into a transformer that implements SimpleTuner's hidden-state buffer contract:

```python
from safetensors.torch import load_file

from simpletuner.helpers.training.internal_guidance import (
    attach_internal_guidance_head_from_state_dict,
    internal_guidance_inference,
    internal_guidance_lora_state_dict,
)

state_dict = load_file("pytorch_lora_weights.safetensors")
attach_internal_guidance_head_from_state_dict(pipe.transformer, state_dict)
pipe.transformer.load_lora_adapter(
    internal_guidance_lora_state_dict(state_dict),
    prefix="transformer",
)

with internal_guidance_inference(pipe.transformer, scale=1.4):
    image = pipe(prompt).images[0]
```

Standard LoRA loaders reject the auxiliary tensors, so `internal_guidance_lora_state_dict` removes them after `attach_internal_guidance_head_from_state_dict` loads their trained values. Loading on the transformer with `prefix="transformer"` handles SimpleTuner's component-prefixed checkpoint keys without relying on pipeline-specific loaders. The transformer forward must accept `hidden_states_buffer` and store `layer_{index}`. SimpleTuner's supported diffusion-transformer implementations provide this contract. The buffer retains only the selected block when Internal Guidance is used alone.

Monitor `internal_guidance_loss` and `internal_guidance_unweighted_loss`. Sweep sampling scales from `1.0` to `1.8`; the best value is model- and checkpoint-dependent.
