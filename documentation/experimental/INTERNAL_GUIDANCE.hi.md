# Internal Guidance

Internal Guidance diffusion transformer के शुरुआती block पर auxiliary denoising head जोड़ता है। यह head final head वाला ही target predict करता है। Sampling के समय दोनों predictions का अंतर बिना दूसरे model या unconditional pass के guidance देता है।

## Configuration

```json
{
  "internal_guidance_enabled": true,
  "internal_guidance_loss_weight": 0.5,
  "internal_guidance_block_index": 7,
  "validation_internal_guidance_scale": 1.4
}
```

- `internal_guidance_loss_weight`: intermediate loss का weight; reference implementation `0.5` उपयोग करता है।
- `internal_guidance_block_index`: zero-based block index। खाली छोड़ने पर transformer depth का एक चौथाई चुना जाता है।
- `validation_internal_guidance_scale`: sampling scale। `1.0` extrapolation बंद करता है; reference में `1.4` है।

```text
loss = final_loss + weight * intermediate_loss
guided = intermediate + scale * (final - intermediate)
```

Projection zero-initialized है और standard PEFT LoRA या full checkpoint के साथ save होता है। LyCORIS, autoregressive, UNet और ACE-Step v1.5 decoder supported नहीं हैं।

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

Transformer को `hidden_states_buffer` स्वीकार करके `layer_{index}` store करना होगा। `internal_guidance_loss` monitor करें और `1.0` से `1.8` तक sampling scale sweep करें।
