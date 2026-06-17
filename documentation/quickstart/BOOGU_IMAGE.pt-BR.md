# Guia de início rápido do Boogu-Image 0.1

Este guia cobre treinamento LoRA e LyCORIS LoKr para Boogu-Image 0.1 no SimpleTuner. Boogu-Image é um modelo de imagem com flow matching e flavours de texto-para-imagem, turbo e edição. A integração do SimpleTuner usa código local de pipeline e transformer, e os pipelines exportados ficam no namespace `SimpleTuner` do Hugging Face.

Configs iniciais incluídas:

```bash
simpletuner/examples/boogu-image-v0.1.peft-lora/config.json
simpletuner/examples/boogu-image-v0.1.lycoris-lokr/config.json
```

## Requisitos de hardware

Trate o Boogu-Image como um modelo transformer de imagem grande. Para primeiras execuções, use 1024px, batch size 1, precisão mista bf16 e gradient checkpointing.

Pontos de partida recomendados:

- **Padrão:** `v0.1-base`, pesos LoRA bf16, rank 16.
- **Menos VRAM:** use um flavour FP8 como `v0.1-base-fp8`, `v0.1-turbo-fp8` ou `v0.1-edit-fp8`.
- **Validação/inferência rápida:** use turbo, observando o estado do assistant LoRA abaixo.
- **Edição:** use `v0.1-edit` ou `v0.1-edit-fp8` com dados condicionais pareados.

O uso de memória depende de rank, otimizador, resolução de validação, offload, compile e uso de FP8. Uma única H100 consegue treinar o exemplo PEFT LoRA por 1000 steps a 1024px com amostras de benchmark e validação habilitadas.

Em GPUs menores, comece com pesos FP8, rank 8-16, `train_batch_size=1`, gradient checkpointing e model/group offload.

### Offload de memória

Group offload pode reduzir pressão de VRAM quando os pesos do transformer são o gargalo:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

Offload opcional para disco:

```bash
--group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams só são efetivos em CUDA; o SimpleTuner desativa em ROCm, MPS e CPU.
- Não combine group offload com outras estratégias de CPU offload.
- Prefira NVMe local rápido para offload em disco.

### Torch compile e atenção

Em GPUs NVIDIA, use aliases de atenção dos kernels do Hugging Face Hub quando disponíveis:

```json
{
  "attention_mechanism": "flash-attn-3-hub",
  "dynamo_backend": "inductor",
  "dynamo_use_regional_compilation": true
}
```

Se a validação compilada gerar imagens pretas em uma combinação específica de GPU/driver, desative primeiro o torch compile e teste novamente antes de mudar a receita.

## Instalação

Instale o SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'

# Usuários CUDA 13 / Blackwell
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Para instalação manual ou setup de desenvolvimento, veja a [documentação de instalação](../INSTALL.md).

## Configuração do ambiente

### Interface web

A WebUI do SimpleTuner pode criar uma configuração para Boogu-Image:

```bash
simpletuner server
```

Abra http://localhost:8001 e escolha `boogu_image` como família de modelo.

### Método manual / linha de comando

Copie `config/config.json.example` para `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Revise estes valores:

- `model_type` - `lora`.
- `lora_type` - `standard` para PEFT LoRA ou `lycoris` para LyCORIS LoKr.
- `model_family` - `boogu_image`.
- `model_flavour` - `v0.1-base`, `v0.1-base-fp8`, `v0.1-turbo`, `v0.1-turbo-fp8`, `v0.1-edit` ou `v0.1-edit-fp8`.
- `pretrained_model_name_or_path` - normalmente deixe sem definir para o flavour escolher o pipeline `SimpleTuner/Boogu-Image-0.1-*`.
- `output_dir` - diretório para checkpoints e imagens de validação.
- `train_batch_size` - comece com `1`.
- `resolution` - comece com `1024`.
- `resolution_type` - use `pixel_area` para buckets multi-aspecto.
- `validation_resolution` - use `1024x1024`; múltiplos tamanhos podem ser separados por vírgulas.
- `validation_guidance` - comece perto de `4.0` para base/edit.
- `validation_num_inference_steps` - comece perto de `30`; turbo pode usar menos steps.
- `mixed_precision` - use `bf16` em GPUs NVIDIA modernas.
- `gradient_checkpointing` - mantenha habilitado.
- `flow_schedule_shift` - os exemplos usam `3`.

Config PEFT LoRA mínima:

```json
{
  "model_type": "lora",
  "model_family": "boogu_image",
  "model_flavour": "v0.1-base",
  "lora_type": "standard",
  "lora_rank": 16,
  "lora_alpha": 16,
  "output_dir": "output/models-boogu-image-v0.1",
  "train_batch_size": 1,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.0,
  "validation_num_inference_steps": 30,
  "validation_prompt": "a polished product photo of a ceramic mug on a walnut desk",
  "validation_steps": 50,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "flow_schedule_shift": 3,
  "optimizer": "adamw_bf16",
  "learning_rate": 1e-4,
  "lr_scheduler": "constant_with_warmup",
  "lr_warmup_steps": 10,
  "max_train_steps": 1000,
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "data_backend_config": "config/examples/multidatabackend-small-dreambooth-1024px.json"
}
```

## Executar os exemplos

```bash
simpletuner train example=boogu-image-v0.1.peft-lora
simpletuner train example=boogu-image-v0.1.lycoris-lokr
```

Forma para checkout de desenvolvimento:

```bash
simpletuner train env=examples/boogu-image-v0.1.peft-lora
simpletuner train env=examples/boogu-image-v0.1.lycoris-lokr
```

## Flavours FP8

Use os flavours `-fp8` para carregar os pesos FP8 exportados:

```json
{
  "model_family": "boogu_image",
  "model_flavour": "v0.1-base-fp8"
}
```

O mesmo vale para `v0.1-turbo-fp8` e `v0.1-edit-fp8`. Você não precisa apontar o SimpleTuner para arquivos `.bin` do Boogu.

## Assistant LoRA do Turbo

O SimpleTuner habilita o caminho de assistant LoRA para `v0.1-turbo` e `v0.1-turbo-fp8`. O path do adaptador atualmente é um placeholder `None`, porque ainda não há um adaptador separado publicado para esta integração.

Até esse adaptador existir, use turbo como pipeline exportado e valide a qualidade diretamente. Para o baseline mais previsível, comece com `v0.1-base`.

## Treinamento de edição

Os flavours de edição exigem dados condicionais pareados. Use a mesma estrutura de dataset de referência pareada descrita no [quickstart do Qwen Image Edit](./QWEN_EDIT.md).

Para LoRA texto-para-imagem, use os flavours base ou turbo.

## Prompts de validação

`validation_prompt` é o prompt principal de validação. Para cobertura maior, adicione uma biblioteca:

```json
{
  "product": "a polished product photo of <token> on a walnut desk",
  "studio": "a clean studio portrait of <token> with softbox lighting",
  "cinematic": "a cinematic scene featuring <token>, detailed lighting, shallow depth of field"
}
```

Aponte a configuração para ela:

```json
{
  "validation_prompt_library": "config/user_prompt_library.json"
}
```

Use prompts bem diferentes para detectar overfitting, colapso de prompt e drift de estilo.

## Inferência

Depois do treino, carregue o adaptador salvo com o mesmo flavour de pipeline usado no treino. O arquivo principal geralmente é:

```bash
output/models-boogu-image-v0.1/pytorch_lora_weights.safetensors
```
