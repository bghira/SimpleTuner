# Guia rápido do Mage-Flow

Este guia cobre treino LoRA do Mage-Flow no SimpleTuner. Mage-Flow é a família rectified-flow 4B da Microsoft para geração e edição de imagens, com transformer MMDiT em resolução nativa, condicionamento Qwen3-VL e Mage-VAE com latentes de 128 canais e downsample 16x.

## Hardware

Mage-Flow é menor que Flux.1 e Qwen-Image, mas ainda usa um transformer grande e um text encoder Qwen3-VL congelado.

Pontos de partida:

- `bf16`, 512px, batch 1 para smoke tests
- `bf16`, 1024px, batch 1 para LoRA normal em GPUs maiores
- `int8-torchao` ou NF4 quando faltar VRAM
- flavours Turbo com 4 passos de validação

Use 24GB como mínimo prático para testes reduzidos ou quantizados, 48GB para 1024px com mais conforto e 80GB para edit training ou batches maiores.

## Configuração

Instale o SimpleTuner:

```bash
pip install 'simpletuner[cuda]'
```

Configuração inicial para texto para imagem:

```json
{
  "model_family": "mageflow",
  "model_flavour": "base",
  "model_type": "lora",
  "pretrained_model_name_or_path": "microsoft/Mage-Flow-Base",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "optimizer": "optimi-lion",
  "learning_rate": 1e-4,
  "lora_rank": 32,
  "train_batch_size": 1,
  "resolution": 1024,
  "validation_resolution": "1024x1024",
  "validation_num_inference_steps": 30,
  "validation_guidance": 5.0
}
```

Flavours disponíveis:

- `base` - `microsoft/Mage-Flow-Base`
- `default` - `microsoft/Mage-Flow`
- `turbo` - `microsoft/Mage-Flow-Turbo`
- `edit-base` - `microsoft/Mage-Flow-Edit-Base`
- `edit` - `microsoft/Mage-Flow-Edit`
- `edit-turbo` - `microsoft/Mage-Flow-Edit-Turbo`

Para edição:

```json
{
  "model_family": "mageflow",
  "model_flavour": "edit-turbo",
  "pretrained_model_name_or_path": "microsoft/Mage-Flow-Edit-Turbo",
  "validation_num_inference_steps": 4
}
```

Os flavours de edição exigem dataset com imagem de condicionamento. O SimpleTuner troca automaticamente para a pipeline de edição durante `check_user_config`, como no Flux Kontext.

## Dataloader

Para LoRA de subject/style, use o dataloader de imagem normal:

```json
[
  {
    "id": "dreambooth-1024",
    "type": "local",
    "instance_data_dir": "/path/to/images",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "resolution": 1024,
    "resolution_type": "pixel",
    "metadata_backend": "discovery",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "cache_dir_vae": "cache/vae/mageflow/dreambooth-1024"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/mageflow"
  }
]
```

Para edição, use pares source/target. A legenda deve ser a instrução de edição, não apenas uma descrição do alvo.

## Presets de memoria

Mage-Flow inclui presets de RAMTorch e Musubi block swap no menu de otimizacao de memoria. Use RAMTorch para manter pesos do transformer na RAM da CPU; use Musubi block swap para transmitir apenas os ultimos blocos durante forward e backward. Eles sao mutuamente exclusivos no configurador.

## Validação e quantização

Use cerca de 20 passos para `default`, 30 para `base` e 4 para `turbo` / `edit-turbo`.

```json
{
  "base_model_precision": "int8-torchao",
  "quantize_via": "cpu"
}
```

O SimpleTuner vendoriza o código MIT do Mage-Flow e o envolve em pipelines Diffusers nativas para validação.
