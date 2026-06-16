# Prompt2Effect

Prompt2Effect e um fluxo experimental somente por CLI para treinar uma hiper-rede que gera pesos PEFT LoRA a partir de um prompt de efeito. Ele fica separado do treinador normal de denoising de imagem/video do SimpleTuner.

A diferenca importante e que Prompt2Effect nao faz o treinamento da hiper-rede levar 3,3 segundos. Ele move o trabalho caro para uma etapa unica de treinamento sobre uma biblioteca de LoRAs de efeito ja existentes. Depois que essa hiper-rede existe, gerar um novo LoRA a partir de um prompt e uma unica passada forward.

## O Que Ele Treina

As amostras de treinamento sao checkpoints LoRA existentes, nao arquivos de midia:

- um prompt de efeito
- um checkpoint PEFT LoRA para esse efeito
- um modelo base fixo e um esquema fixo de camadas alvo

A etapa de preparacao converte cada atualizacao LoRA em fatores canonicos por SVD. A perda de treinamento e MSE normalizado sobre esses fatores LoRA canonicos, nao uma perda de difusao sobre latentes.

## Familias Suportadas

Os scripts atualmente suportam:

- `ltxvideo2`
- sabores I2V de `wan`
- `hunyuanvideo`

O artefato gerado e um arquivo normal `pytorch_lora_weights.safetensors` com chaves PEFT `lora_A`, `lora_B` e `alpha`.

## Arquivos

Prompt2Effect fica em `scripts/prompt2effect/`:

- `prepare.py`: valida um manifesto de LoRAs e grava alvos canonicos por SVD.
- `train.py`: treina a hiper-rede Prompt2Effect.
- `generate.py`: emite um PEFT LoRA a partir de uma hiper-rede treinada e um prompt de efeito.

Isso nao aparece na WebUI.

## Manifesto

Crie um arquivo JSONL com um LoRA de efeito por linha:

```json
{"id":"blue_mood","effect_prompt":"blue mood cinematic atmosphere","lora_path":"/path/to/pytorch_lora_weights.safetensors"}
```

Todos os LoRAs em uma execucao Prompt2Effect devem usar o mesmo esquema de modulos alvo e as mesmas dimensoes de entrada/saida. Use `--rank` na preparacao para escolher o rank LoRA canonico/gerado; se omitido, o rank do primeiro LoRA sera usado.

## Preparar Alvos

```bash
.venv/bin/python scripts/prompt2effect/prepare.py \
  --manifest /path/to/effects.jsonl \
  --output_dir cache/prompt2effect/wan-i2v-targets \
  --model_family wan \
  --model_flavour i2v-14b-2.1
```

Opcoes uteis:

- `--model_family`: `ltxvideo2`, `wan` ou `hunyuanvideo`.
- `--base_model`: substitui o repo ou caminho local do modelo base.
- `--model_flavour`: usa um valor conhecido da familia quando `--base_model` nao e fornecido.
- `--target_modules`: sufixos PEFT separados por virgula, `default` ou `all-linear`.
- `--rank`: rank do LoRA gerado. O padrao e o rank do primeiro LoRA fonte.
- `--component_subfolder`: subpasta do componente do modelo base. O padrao e a subpasta transformer da familia.

`prepare.py` grava:

- `schema.json`
- `targets.safetensors`

Ele falha se um LoRA nao tiver modulos obrigatorios, tiver modulos inesperados ou nao corresponder aos formatos dos tensores do modelo base.

## Treinar

```bash
.venv/bin/python scripts/prompt2effect/train.py \
  --prepared_dir cache/prompt2effect/wan-i2v-targets \
  --output_dir output/prompt2effect/wan-i2v \
  --text_encoder_model google/t5-v1_1-base \
  --max_train_steps 10000
```

O codificador de texto fica congelado e apenas codifica prompts de efeito. Os pesos do modelo base tambem ficam congelados e sao usados como condicionamento estrutural para a hiper-rede.

Por padrao, os pesos base ficam na CPU. Use `--base_weights_device training` somente quando as camadas alvo selecionadas couberem no acelerador.

## Gerar Um LoRA

```bash
.venv/bin/python scripts/prompt2effect/generate.py \
  --checkpoint output/prompt2effect/wan-i2v/prompt2effect_hypernetwork.pt \
  --prompt "blue mood cinematic atmosphere" \
  --output output/blue_mood_prompt2effect
```

O diretorio de saida contera `pytorch_lora_weights.safetensors`. Carregue-o como qualquer outro PEFT LoRA do SimpleTuner/Diffusers.

## Limites

- Apenas PEFT LoRA linear. LyCORIS, LoRA convolucional, vetores de magnitude DoRA e tensores sidecar arbitrarios ainda nao sao suportados neste fluxo.
- Uma hiper-rede fica vinculada a uma familia de modelo, formato do modelo base, esquema de modulos alvo e rank.
- Os scripts nao sao integrados ao Accelerate, a WebUI ou ao gerenciador principal de checkpoints do SimpleTuner.
- A qualidade do treinamento depende da quantidade e diversidade dos LoRAs de efeito fonte. Poucos LoRAs bastam para testar o caminho, nao para esperar generalizacao.
- LoRAs gerados devem ser validados normalmente antes de publicacao ou uso em fluxos de producao.
