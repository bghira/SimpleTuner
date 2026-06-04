# Ideogram 4 Quickstart

Este guia cobre o treinamento de LoRA para Ideogram 4 no SimpleTuner. Ideogram 4 é um modelo de imagem flow-matching com cerca de 9B parâmetros, forte em tipografia, layout e prompts complexos. O checkpoint público é distribuído em FP8; o SimpleTuner usa essa versão FP8 por padrão.

Configuração inicial:

```bash
simpletuner/examples/ideogram-fp8.peft-lora/config.json
```

## Hardware e precisão

Pontos de partida recomendados:

- **Padrão:** pesos base FP8, pesos LoRA treináveis em bf16, rank 16-32.
- **Baixa VRAM:** NF4 para o modelo base.
- **Alta VRAM:** pesos bf16-upcast se houver VRAM suficiente e você quiser evitar carregamento quantizado.

Em GPUs NVIDIA de 80G, LoRA FP8 ou bf16-upcast em 1024px com batch size 1 deve caber normalmente. Em GPUs menores, comece com FP8 ou NF4, rank 8-16, gradient checkpointing e offload. GPUs Apple não são recomendadas para treinamento do Ideogram 4.

## Configuração

Copie a configuração e o dataloader de exemplo:

```bash
mkdir -p config/examples
cp simpletuner/examples/ideogram-fp8.peft-lora/config.json config/config.json
cp simpletuner/examples/multidatabackend-ideogram-dreambooth-1024px.json config/examples/multidatabackend-ideogram-dreambooth-1024px.json
```

Campos importantes:

```json
{
  "model_type": "lora",
  "model_family": "ideogram",
  "model_flavour": "fp8",
  "base_model_precision": "no_change",
  "quantize_via": "cpu",
  "mixed_precision": "bf16",
  "train_batch_size": 1,
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "gradient_checkpointing": true,
  "ideogram_auto_json": true,
  "ideogram_validation": true,
  "ideogram_schedule_mu": 0.0,
  "ideogram_schedule_std": 1.5
}
```

FP8 é a primeira recomendação:

```json
{
  "model_flavour": "fp8",
  "base_model_precision": "no_change",
  "quantize_via": "cpu"
}
```

Para pouca VRAM, use NF4:

```json
{
  "base_model_precision": "nf4-bnb",
  "base_model_default_dtype": "bf16",
  "quantize_via": "cpu"
}
```

## Validação

A validação do Ideogram fica desativada até você optar por ela:

```json
{
  "ideogram_validation": true
}
```

Este é um flag temporário. O caminho upstream de inferência CFG do Ideogram espera um transformer unconditional separado, enquanto o SimpleTuner atualmente treina apenas o transformer conditional por padrão. Com o flag ativo, a validação usa o transformer conditional também para o negative/unconditional pass, permitindo verificar prompts e negative prompts.

## Formato das captions

Ideogram 4 funciona melhor com captions JSON estruturadas. Campos recomendados:

- `high_level_description`
- `style_description`
- `style_description.color_palette` com cores hex
- `compositional_deconstruction.background`
- `compositional_deconstruction.elements`
- `bbox` opcional no formato `[x1, y1, x2, y2]`

Se o dataset mistura texto comum e JSON, mantenha:

```json
{
  "ideogram_auto_json": true
}
```

Prompts em texto são embrulhados no schema JSON do Ideogram; captions JSON existentes são normalizadas e preservadas. Captions JSON escritas manualmente continuam sendo melhores, especialmente quando descrevem composição, fundo, elementos e cores.

## Prompt upsampling

Opcionalmente:

```json
{
  "ideogram_prompt_upsample": true,
  "ideogram_prompt_enhancer_head_id": "diffusers/qwen3-vl-8b-instruct-lm-head"
}
```

Isso reescreve prompts com o prompt upsampler do Ideogram antes da conversão para JSON. É mais lento; deixe desligado até confirmar que o caminho básico de treinamento funciona.

## LoRA e LyCORIS

O PEFT LoRA padrão mira projections de attention:

```json
{
  "lora_type": "standard",
  "lora_rank": 32
}
```

LyCORIS/LoKr pode mirar as classes `Attention` e `FeedForward` expostas pelo Ideogram. Full-matrix LoKr pode gerar arquivos muito grandes; use LoRA padrão para iteração rápida.

## Expectativa de loss

O loss do Ideogram pode parecer alto comparado a outros modelos. Valores perto ou acima de `1.0` não significam automaticamente que o modelo quebrou ou que as imagens de validação ficarão incoerentes.

Em testes, o Ideogram produziu imagens coerentes mesmo com step loss variando por volta de `0.3-1.3`, com spikes ocasionais. Julgue pelo resultado das validações, aderência ao prompt e se o loss está explodindo, não pela expectativa de um scalar muito baixo.

## Treinamento

```bash
simpletuner train
```

Checkout de desenvolvimento:

```bash
CONFIG_BACKEND=json CONFIG_PATH=config/config.json .venv/bin/python simpletuner/train.py
```
