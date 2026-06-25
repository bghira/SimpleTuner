# Guia Rápido do Krea2

Este guia cobre treinamento LoRA do Krea2 no SimpleTuner. Krea2 é um transformer grande de imagem com flow matching, condicionamento de texto estilo Qwen e VAE do Qwen Image. Ele é mais confortável em GPUs NVIDIA com muita memória.

O exemplo inicial está em:

```bash
simpletuner/examples/krea2.peft-lora/config.json
```

## Ponto de Partida Recomendado

Para a primeira execução, use a configuração de exemplo e mantenha o modelo conservador:

```json
{
  "model_family": "krea2",
  "model_flavour": "raw",
  "model_type": "lora",
  "pretrained_model_name_or_path": "krea/Krea-2-Raw",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "fuse_qkv_projections": true,
  "train_batch_size": 1,
  "base_model_precision": "no_change"
}
```

Krea2 é nativo em 1024px, mas 512px e 768px são úteis para iteração rápida e checagem de dataset. Use um dataloader de 1024px depois que a execução estiver estável.

## Notas de Hardware

Krea2 pode treinar em bf16 em uma H100 de 80GB com batch 1 em 1024px. Batches maiores couberam sem compile nos nossos testes, mas compile adiciona memória de grafo/cudagraph suficiente para causar OOM em muitas configurações maiores.

TorchAO int8 weight-only reduz bastante a VRAM, mas não foi mais rápido que bf16 no caminho de treinamento testado. Use quando memória for mais importante que tempo por passo.

Recomendações:

- Use `bf16` quando couber.
- Use `int8-torchao` quando precisar de folga de memória.
- Mantenha `gradient_checkpointing=true`.
- Mantenha `fuse_qkv_projections=true`.
- Use `dynamo_backend=inductor`, `dynamo_mode=reduce-overhead` e `dynamo_use_regional_compilation=true` apenas depois de confirmar que batch/resolução cabem.

## Valores Principais de Configuração

```json
{
  "model_family": "krea2",
  "model_flavour": "raw",
  "model_type": "lora",
  "pretrained_model_name_or_path": "krea/Krea-2-Raw",
  "base_model_precision": "no_change",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "fuse_qkv_projections": true,
  "optimizer": "optimi-lion",
  "learning_rate": 1e-4,
  "lora_rank": 64,
  "train_batch_size": 1,
  "resolution": 1024,
  "validation_resolution": "1024x1024"
}
```

Para TorchAO int8:

```json
{
  "base_model_precision": "int8-torchao",
  "quantize_via": "cpu"
}
```

Para compile reduce-overhead:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_mode": "reduce-overhead",
  "dynamo_use_regional_compilation": true
}
```

## Treinamento com Imagem de Referência

Krea2 suporta condicionamento opcional por latentes de referência para datasets de edição. Ative quando o dataloader fornecer imagens de referência pareadas ou latentes de referência em cache:

```json
{
  "krea2_reference_latents": true
}
```

Os latentes de referência devem ter a mesma forma dos latentes alvo.

## Configuração do Dataloader

Krea2 usa a estrutura geral de dataloader de imagem dos outros modelos transformer. A resolução real de treino vem do JSON do dataloader, não apenas de `resolution` no config principal. Para treinar em 1024px, confirme que `resolution`, `maximum_image_size` e `target_downsample_size` também são 1024 no dataloader.

Datasets em 512px são úteis para testes rápidos, checar captions e encontrar crops ruins. Para sinal de qualidade final, 1024px costuma ser mais representativo.

Para datasets locais, use `type: local`, defina `instance_data_dir` e escolha uma estratégia de caption. Para subject LoRA pequeno, `caption_strategy=instanceprompt` é um bom começo. Para estilos, filenames ou captions completas tendem a funcionar melhor.

## Validação

Validação do Krea2 é cara, então comece com poucos prompts. Um único prompt pode esconder overfit ou memorização. Depois que o run estiver estável, adicione uma pequena prompt library.

```json
{
  "validation_prompt": "a studio portrait of <token>, soft directional light, detailed fabric texture",
  "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
  "validation_num_inference_steps": 28,
  "validation_guidance": 4.5,
  "validation_resolution": "1024x1024"
}
```

## Notas de Quantização

`int8-torchao` armazena os pesos base do transformer em int8 e treina pesos LoRA bf16 por cima. Na H100 reduziu bastante a VRAM, mas foi mais lento que bf16 neste caminho de treinamento. Pense nele como uma opção de capacidade, não como garantia de throughput.

## Resultados de Benchmark

As medições abaixo foram feitas em uma NVIDIA H100 80GB usando o trainer real do SimpleTuner, Krea2 LoRA, QKV fusionado, gradient checkpointing e um dataset pequeno Domokun. A VRAM foi amostrada externamente com `nvidia-smi`. Use estes valores apenas como comparação; versões diferentes de PyTorch, CUDA, driver, dataset, rank LoRA, otimizador, backend de atenção e GPU podem mudar os resultados.

### QKV Fusionado + Checkpointing, Compile Desligado

| Precisão | Resolução | Batch | s/passo estável | Pico VRAM |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 512 | 1 | 0.353 | 31.10 GiB |
| bf16 | 512 | 4 | 1.230 | 39.31 GiB |
| bf16 | 512 | 8 | 2.430 | 50.32 GiB |
| bf16 | 1024 | 1 | 0.990 | 33.28 GiB |
| bf16 | 1024 | 4 | 3.850 | 48.35 GiB |
| bf16 | 1024 | 8 | 7.690 | 67.88 GiB |
| int8-torchao | 512 | 1 | 0.535 | 18.10 GiB |
| int8-torchao | 512 | 4 | 1.690 | 27.46 GiB |
| int8-torchao | 512 | 8 | 3.220 | 40.52 GiB |
| int8-torchao | 1024 | 1 | 1.330 | 20.35 GiB |
| int8-torchao | 1024 | 4 | 4.850 | 36.99 GiB |
| int8-torchao | 1024 | 8 | 9.520 | 58.84 GiB |

### QKV Fusionado + Checkpointing + Compile Reduce-Overhead

| Precisão | Resolução | Batch | Estado | s/passo estável | Pico VRAM |
| --- | ---: | ---: | --- | ---: | ---: |
| bf16 | 512 | 1 | ok | 0.260 | 41.20 GiB |
| bf16 | 512 | 4 | OOM | - | 79.07 GiB |
| bf16 | 512 | 8 | OOM | - | 79.10 GiB |
| bf16 | 1024 | 1 | ok | 0.704 | 63.71 GiB |
| bf16 | 1024 | 4 | OOM | - | 79.11 GiB |
| bf16 | 1024 | 8 | OOM | - | 78.40 GiB |
| int8-torchao | 512 | 1 | ok | 0.410 | 30.93 GiB |
| int8-torchao | 512 | 4 | ok | 1.300 | 78.60 GiB |
| int8-torchao | 512 | 8 | OOM | - | 79.12 GiB |
| int8-torchao | 1024 | 1 | ok | 0.990 | 58.68 GiB |
| int8-torchao | 1024 | 4 | OOM | - | 78.92 GiB |
| int8-torchao | 1024 | 8 | OOM | - | 78.09 GiB |

## Orientação Prática

- Para iterar mais rápido em uma H100, use bf16, QKV fusionado, checkpointing, compile ligado e batch 1.
- Para batches efetivos maiores, prefira bf16 sem compile e aumente `train_batch_size` até a VRAM virar o limite.
- Para execuções com pouca memória, use `int8-torchao`; espere menos VRAM, mas passos mais lentos.
- Compile ajuda em batch 1, mas pode consumir VRAM suficiente para fazer batches maiores falharem.

## Problemas Comuns

- Se você esperava 1024px mas o log mostra 512px, revise o JSON do dataloader.
- Se compile causa OOM mas o run sem compile cabe, reduza batch size ou desligue compile.
- Se int8 usa menos VRAM mas é mais lento, isso corresponde aos nossos testes H100.
- Se a imagem de referência não afeta a validação, confirme `krea2_reference_latents=true` e dataset pareado.
- Se overfitar rápido, reduza learning rate, reduza steps ou aumente a variedade do dataset.
