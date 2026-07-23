# Guia rápido do Mage-Flow

Este guia cobre treino LoRA do Mage-Flow no SimpleTuner. Mage-Flow é a família rectified-flow 4B da Microsoft para geração e edição de imagens, com transformer MMDiT em resolução nativa, condicionamento Qwen3-VL e Mage-VAE com latentes de 128 canais e downsample 16x.

## Hardware

Mage-Flow é menor que Flux.1 e Qwen-Image, mas ainda usa um transformer grande e um text encoder Qwen3-VL congelado.

Pontos de partida:

- `bf16`, 512px, batch 1 para smoke tests
- `bf16`, 1024px, batch 1 para LoRA normal em GPUs maiores
- `fp8wo-torchao` quando faltar VRAM em GPUs NVIDIA Ada/Hopper ou mais novas
- flavours Turbo com 4 passos de validação

Use 24GB como mínimo prático para testes reduzidos ou quantizados, 48GB para 1024px com mais conforto e 80GB para edit training ou batches maiores.

## Configuração

Instale o SimpleTuner:

```bash
pip install 'simpletuner[cuda]'
```

Mage-Flow usa atencao empacotada de comprimento variavel. Para usar FlashAttention 2 sem compilar o pacote `flash-attn` localmente, defina `"attention_mechanism": "flash-attn-varlen-hub"` para que o SimpleTuner carregue o kernel pelo Hugging Face Hub. Mantenha o valor padrao `diffusers` para PyTorch SDPA.

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

## Mage Flow (Edit) Considerations

Os checkpoints Mage-Flow edit nao exigem dataset de condicionamento ou referencia. A Microsoft treinou os modelos edit em conjunto em tarefas de geracao e edicao, entao o prior generativo e preservado. No SimpleTuner, voce pode continuar usando um dataset normal de imagens para LoRA de sujeito, estilo ou conceito mesmo quando `model_flavour` for `edit-base`, `edit` ou `edit-turbo`.

Use pares source/target apenas quando quiser treinar comportamento de edicao. O SimpleTuner usa automaticamente a pipeline compativel com edicao; quando nenhuma imagem de condicionamento e fornecida, a validacao e o prompt encoding usam o caminho text-to-image.

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

Para treinar comportamento de edicao opcional, use pares source/target. A legenda deve ser a instrucao de edicao, nao apenas uma descricao do alvo.

## Presets de memoria

Mage-Flow inclui presets de RAMTorch e Musubi block swap no menu de otimizacao de memoria. Use RAMTorch para manter pesos do transformer na RAM da CPU; use Musubi block swap para transmitir apenas os ultimos blocos durante forward e backward. Eles sao mutuamente exclusivos no configurador.

## Validação e quantização

Use cerca de 20 passos para `default`, 30 para `base` e 4 para `turbo` / `edit-turbo`.

```json
{
  "base_model_precision": "fp8wo-torchao",
  "quantize_via": "cpu"
}
```

Em smoke tests de LoRA para Mage-Flow, a quantização int8 produziu picos suspeitos de loss quando comparada com FP8 weight-only TorchAO. Evite presets int8 para Mage-Flow a menos que você valide a curva de loss no seu dataset. NF4 e outros presets de quantização ainda podem ser úteis.

O SimpleTuner vendoriza o código MIT do Mage-Flow e o envolve em pipelines Diffusers nativas para validação.
