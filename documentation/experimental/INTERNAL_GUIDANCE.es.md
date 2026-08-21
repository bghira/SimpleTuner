# Internal Guidance

Internal Guidance añade una cabeza de denoising auxiliar a un bloque temprano del diffusion transformer. La cabeza predice el mismo objetivo que la cabeza final. Durante el muestreo, su diferencia proporciona guidance sin otro modelo ni una pasada incondicional.

## Configuración

```json
{
  "internal_guidance_enabled": true,
  "internal_guidance_loss_weight": 0.5,
  "internal_guidance_block_index": 7,
  "validation_internal_guidance_scale": 1.4
}
```

- `internal_guidance_loss_weight`: peso de la pérdida intermedia; la referencia usa `0.5`.
- `internal_guidance_block_index`: bloque capturado, con índice desde cero. Si se omite, se usa un cuarto de la profundidad.
- `validation_internal_guidance_scale`: escala de muestreo. `1.0` desactiva la extrapolación; la referencia usa `1.4`.

```text
loss = final_loss + weight * intermediate_loss
guided = intermediate + scale * (final - intermediate)
```

La proyección se inicializa a cero y se guarda con LoRA PEFT estándar o checkpoints completos. LyCORIS, modelos autoregresivos, UNet y el decoder ACE-Step v1.5 no son compatibles.

## Inferencia con Diffusers

```python
from safetensors.torch import load_file
from simpletuner.helpers.training.internal_guidance import attach_internal_guidance_head_from_state_dict, internal_guidance_inference, internal_guidance_lora_state_dict

state_dict = load_file("pytorch_lora_weights.safetensors")
attach_internal_guidance_head_from_state_dict(pipe.transformer, state_dict)
pipe.transformer.load_lora_adapter(internal_guidance_lora_state_dict(state_dict), prefix="transformer")
with internal_guidance_inference(pipe.transformer, scale=1.4):
    image = pipe(prompt).images[0]
```

El transformer debe aceptar `hidden_states_buffer` y guardar `layer_{index}`. Supervise `internal_guidance_loss` y pruebe escalas entre `1.0` y `1.8`.
