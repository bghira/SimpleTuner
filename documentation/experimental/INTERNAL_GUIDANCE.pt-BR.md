# Internal Guidance

Internal Guidance adiciona uma cabeça auxiliar de denoising a um bloco inicial do diffusion transformer. Ela prevê o mesmo alvo da cabeça final. Durante a amostragem, a diferença entre as duas fornece guidance sem outro modelo ou uma passada incondicional.

## Configuração

```json
{
  "internal_guidance_enabled": true,
  "internal_guidance_loss_weight": 0.5,
  "internal_guidance_block_index": 7,
  "validation_internal_guidance_scale": 1.4
}
```

- `internal_guidance_loss_weight`: peso da loss intermediária; a referência usa `0.5`.
- `internal_guidance_block_index`: bloco zero-based. Quando omitido, usa um quarto da profundidade.
- `validation_internal_guidance_scale`: escala de sampling. `1.0` desativa a extrapolação; a referência usa `1.4`.

```text
loss = final_loss + weight * intermediate_loss
guided = intermediate + scale * (final - intermediate)
```

A projeção usa inicialização zero e é salva com LoRA PEFT padrão ou checkpoints completos. LyCORIS, modelos autoregressivos, UNet e o decoder ACE-Step v1.5 não são compatíveis.

## Inferência com Diffusers

```python
from safetensors.torch import load_file
from simpletuner.helpers.training.internal_guidance import attach_internal_guidance_head_from_state_dict, internal_guidance_inference, internal_guidance_lora_state_dict

state_dict = load_file("pytorch_lora_weights.safetensors")
attach_internal_guidance_head_from_state_dict(pipe.transformer, state_dict)
pipe.transformer.load_lora_adapter(internal_guidance_lora_state_dict(state_dict), prefix="transformer")
with internal_guidance_inference(pipe.transformer, scale=1.4):
    image = pipe(prompt).images[0]
```

O transformer deve aceitar `hidden_states_buffer` e armazenar `layer_{index}`. Monitore `internal_guidance_loss` e teste escalas entre `1.0` e `1.8`.
