# ERNIE-Image [base / turbo] Guia Rápido

Este guia mostra como treinar uma LoRA para ERNIE-Image. ERNIE-Image é a família single-stream flow-matching da Baidu, e o SimpleTuner suporta os sabores `base` e `turbo`.

## Requisitos de hardware

ERNIE não é um modelo pequeno. Planeje algo parecido com outros transformers single-stream grandes:

- o alvo mais realista é uma GPU com 24 GB ou mais usando quantização int8 + LoRA em bf16
- 16 GB pode funcionar com offload agressivo e RamTorch, mas com iteração mais lenta
- multi-GPU, FSDP2 e offload extra para CPU/RAM também ajudam

Apple GPUs não são recomendadas para treino.

## Instalação

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Veja também a [documentação de instalação](../INSTALL.md).

## Configurando o ambiente

### WebUI

```bash
simpletuner server
```

Depois selecione a família ERNIE no assistente de treino.

### Linha de comando

O caminho mais simples é usar o exemplo incluído:

- config de exemplo: `simpletuner/examples/ernie.peft-lora/config.json`
- env local executável: `config/ernie-example/config.json`

Execute:

```bash
simpletuner train --env ernie-example
```

Se for configurar manualmente, use:

- `model_type`: `lora`
- `model_family`: `ernie`
- `model_flavour`: `base` ou `turbo`
- `pretrained_model_name_or_path`:
  - `base`: `baidu/ERNIE-Image`
  - `turbo`: `baidu/ERNIE-Image-Turbo`
- `resolution`: comece com `512`
- `train_batch_size`: `1`
- `ramtorch`: `true`
- `ramtorch_text_encoder`: `true`
- `gradient_checkpointing`: `true`

O exemplo usa:

- `max_train_steps: 100`
- `optimizer: optimi-lion`
- `learning_rate: 1e-4`
- `validation_guidance: 4.0`
- `validation_num_inference_steps: 20`

### Assistant LoRA no Turbo

O ERNIE Turbo já tem suporte a assistant LoRA, mas ainda não existe um caminho padrão de adaptador.

- flavour suportado: `turbo`
- nome de peso padrão: `pytorch_lora_weights.safetensors`
- valor que você precisa fornecer: `assistant_lora_path`

Se você tiver um assistant adapter próprio:

```json
{
  "assistant_lora_path": "your-org/your-ernie-turbo-assistant-lora",
  "assistant_lora_weight_name": "pytorch_lora_weights.safetensors"
}
```

Se não quiser usar:

```json
{
  "disable_assistant_lora": true
}
```

### Dataset e captions

O exemplo usa:

- `dataset_name`: `RareConcepts/Domokun`
- `caption_strategy`: `instanceprompt`
- `instance_prompt`: `🟫`

Isso serve para smoke test, mas o ERNIE normalmente responde melhor a texto real do que a um gatilho de um único token. Para treino real, prefira captions mais descritivas.

### Recursos avançados

ERNIE também suporta:

- TREAD
- LayerSync
- captura de hidden states estilo REPA / CREPA
- carregamento de assistant LoRA no turbo

Primeiro faça o treino básico funcionar; depois ative os extras.
