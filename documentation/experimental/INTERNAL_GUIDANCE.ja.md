# Internal Guidance

Internal Guidance は diffusion transformer の前段ブロックに補助 denoising head を追加します。この head は final head と同じ target を予測します。Sampling では両者の差を使い、別モデルや unconditional pass なしで guidance を行います。

## 設定

```json
{
  "internal_guidance_enabled": true,
  "internal_guidance_loss_weight": 0.5,
  "internal_guidance_block_index": 7,
  "validation_internal_guidance_scale": 1.4
}
```

- `internal_guidance_loss_weight`: intermediate loss の重み。reference implementation は `0.5`。
- `internal_guidance_block_index`: 0-based の capture block。省略時は transformer depth の 1/4。
- `validation_internal_guidance_scale`: sampling scale。`1.0` は extrapolation 無効。reference は `1.4`。

```text
loss = final_loss + weight * intermediate_loss
guided = intermediate + scale * (final - intermediate)
```

Projection は zero initialization され、standard PEFT LoRA または full-model checkpoint に保存されます。LyCORIS、autoregressive model、UNet、ACE-Step v1.5 decoder は非対応です。

## Diffusers inference

```python
from safetensors.torch import load_file
from simpletuner.helpers.training.internal_guidance import attach_internal_guidance_head_from_state_dict, internal_guidance_inference, internal_guidance_lora_state_dict

state_dict = load_file("pytorch_lora_weights.safetensors")
attach_internal_guidance_head_from_state_dict(pipe.transformer, state_dict)
pipe.transformer.load_lora_adapter(internal_guidance_lora_state_dict(state_dict), prefix="transformer")
with internal_guidance_inference(pipe.transformer, scale=1.4):
    image = pipe(prompt).images[0]
```

Transformer は `hidden_states_buffer` を受け取り `layer_{index}` を保存する必要があります。`internal_guidance_loss` を監視し、sampling scale `1.0` から `1.8` を試してください。
