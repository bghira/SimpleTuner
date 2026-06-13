# Guia rapido do zlab i1

Este guia cobre treinamento LoRA para [zlab-princeton i1](https://huggingface.co/zlab-princeton/i1-3B). O i1 e um transformer flow-matching de 3B publicado com receita JAX/TPU e pesos PyTorch de inferencia. O SimpleTuner treina esse modelo com uma integracao PyTorch nativa e usa uma conversao Diffusers safetensors em [`bghira/zlab-i1-diffusers`](https://huggingface.co/bghira/zlab-i1-diffusers).

O i1 usa o VAE do FLUX.2, um text encoder T5Gemma, latentes de 32 canais e uma caption nula aprendida para CFG.

## Requisitos de hardware

Para LoRA em 1024px, comece com:

- uma GPU moderna de 24G com quantizacao int8 para LoRAs pequenas
- 40G+ para uma experiencia mais folgada
- multi-GPU para ranks maiores, datasets maiores ou menos quantizacao

Os exemplos usam `int8-quanto`, `bf16`, `gradient_checkpointing=true` e `train_batch_size=1`. CUDA e o caminho esperado; Apple GPU nao e recomendada.

## Exemplos incluidos

```bash
simpletuner train example=zlab-i1.peft-lora
simpletuner train example=zlab-i1.lycoris-lokr
```

Comece pelo PEFT LoRA. Use LyCORIS LoKr quando voce quiser essa fatorizacao em vez de LoRA padrao.

## Configuracao principal

```json
{
  "model_type": "lora",
  "model_family": "zlab_i1",
  "model_flavour": "3b",
  "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.2-dev",
  "validation_resolution": "1024x1024",
  "validation_guidance": 12.0,
  "validation_guidance_rescale": 0.7,
  "validation_num_inference_steps": 250
}
```

O flavour `3b` resolve para `bghira/zlab-i1-diffusers`, com o transformer no subfolder Diffusers padrao `transformer/` e pesos safetensors. Defina `pretrained_transformer_model_name_or_path` apenas ao testar uma conversao propria.

## Validacao

A validacao funciona pelo pipeline nativo do i1. Para um smoke test rapido:

```bash
simpletuner train example=zlab-i1.peft-lora validation_num_inference_steps=4 num_eval_images=1
```

Quatro passos apenas comprovam que o pipeline gera e salva imagens. Use 250 passos antes de julgar qualidade.

## Recursos avançados

i1 usa os caminhos comuns de transformers do SimpleTuner:

- TwinFlow funciona em modo flow-matching nativo. O timestep do i1 é ignorado como no upstream, então o TwinFlow altera a trajetória latente ruidosa e o alvo, não adiciona um novo embedding temporal.
- CREPA Self-Flow e LayerSync usam o buffer de hidden states dos tokens de imagem. Configure índices de bloco contra as 29 camadas transformer do i1.
- TREAD roteia apenas tokens de imagem. Os tokens de texto ficam intactos para preservar a semântica da máscara do T5Gemma.
- A validação aceita CFG Zero*, skip de CFG via `validation_no_cfg_until_timestep` e skip-layer guidance via `validation_guidance_skip_layers`.
- RamTorch, Musubi block swap e VAE tiling são suportados. Mantenha RamTorch e Musubi como opções mutuamente exclusivas.

## Dataset

O i1 precisa de cache VAE proprio porque espera latentes de 32 canais do VAE do FLUX.2. Nao reutilize caches de SDXL, Flux.1, PixArt ou outras familias.

```json
[
  {
    "id": "my-i1-dataset",
    "type": "local",
    "instance_data_dir": "/datasets/my-subject",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zlab_i1/my-i1-dataset"
  }
]
```

Comece com o exemplo PEFT sem alteracoes, confira benchmark base, loss finita, imagem de validacao e `pytorch_lora_weights.safetensors`; depois troque dataset e prompts.
