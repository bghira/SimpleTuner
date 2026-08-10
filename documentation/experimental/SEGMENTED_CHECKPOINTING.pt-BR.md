# Checkpointing Segmentado

Checkpointing segmentado e o meio-termo entre checkpoint em todo block e nenhum checkpoint.

Ele usa o backend de activation checkpointing do PyTorch. O SimpleTuner executa um grupo contiguo de transformer blocks em uma chamada de checkpoint e passa o hidden state retornado para o proximo grupo. Grupos mais largos recomputam menos no backward, mas mantem mais activations vivas.

Para CPU offload e FFN-only checkpointing, veja [Unsloth-style checkpointing](UNSLOTH_CHECKPOINTING.pt-BR.md#controls). A regra curta continua em [Decision Rule](UNSLOTH_CHECKPOINTING.pt-BR.md#decision-rule).

## Controles

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch",
  "gradient_checkpointing_interval": 2
}
```

Nos caminhos whole-block suportados, `gradient_checkpointing_interval` e a largura do segment. `2` significa checkpoint dos blocks `0-1`, `2-3`, `4-5` e assim por diante.

Para controle mais fino de VRAM, adicione stride:

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch",
  "gradient_checkpointing_interval": 2,
  "gradient_checkpointing_segment_stride": 4
}
```

Isso faz checkpoint dos blocks `0-1`, executa `2-3` normalmente, faz checkpoint de `4-5`, executa `6-7` normalmente e repete. O stride deve ser pelo menos o interval; schedules sobrepostos nao sao validos.

Caminhos segmented whole-block suportados: Flux.1, Flux.2, HunyuanVideo, Krea 2, LongCat Image, LongCat Video, LTXVideo 0.9, LTXVideo2, Lumina2, MageFlow, PixArt, SD3, SanaVideo, Z-Image, ZLab I1 e Wan.

Stable Cascade stage C tambem suporta interval e stride, mas aplica o schedule a sequencia de micro-blocos Res/Timestep/Attention do UNet em vez de grupos transformer whole-block.

Algumas familias usam semantica de interval especifica do modelo:

| Family | `gradient_checkpointing_interval` | `gradient_checkpointing_segment_stride` |
| --- | --- | --- |
| Sana | Checkpoint a cada N-th block | Ignorado |
| Stable Cascade stage C | Checkpoint de micro-blocos UNet por interval | Stride alterna janelas UNet checkpointed e non-checkpointed |
| SD1x, SDXL | Sem suporte segmented whole-block | Ignorado |

Nao compare linhas stride quando stride e ignorado. Se os numeros forem identicos, normalmente a option nao foi aplicada.

## Quando Usar

Use depois que o checkpointing normal por block couber, mas custar tempo demais por step. Comece com `2`. Se houver VRAM, tente `2` com stride `4` em modelos muito profundos.

Nao espere ajuda quando o peak vem principalmente de pesos treinaveis, optimizer state, validation, cache de VAE, block swapping ou routing. O SimpleTuner volta para o caminho per-block mais seguro quando uma feature do modelo precisa desse controle.

`dynamo_use_regional_compilation` nao e ganho universal. Ele ajudou ou foi neutro em alguns image models, mas foi ruim nos perfis Wan/RamTorch e LTXVideo2 abaixo.

## Benchmarks

Medido com exemplos reais do SimpleTuner em pods single-GPU H100 e L40S. Validation e checkpoint saves ficaram desativados, cache preparation foi excluida, e o compile/setup do primeiro step dentro do train loop e excluido quando ha timing post-warmup.

Cada celula medida e `post-warmup sec/step / peak VRAM GiB`. Celulas somente com status significam: `OOM` ficou sem memoria de GPU, `failed` nao chegou aos training steps medidos, `unsupported` significa que a opcao nao estava conectada para essa familia, e `not run` significa que o sweep nao incluiu essa combinacao.

Compare modos dentro da mesma familia primeiro. Comparacoes entre familias sao aproximadas porque resolution, frame count, attention backend, model depth, trainable adapter type e dataset shape mudam.

A matriz abaixo e a fonte de verdade para este sweep. Notas especificas por modelo marcam caveats quando uma linha e coverage data em vez de recomendacao.

<!-- full-sweep-matrix:start -->

### Resultados Por Familia

### ACE Step 1.5

Example: `ace_step-v1-5.peft-lora`. Resolution: 512.

Note: This sweep did not produce a usable ACE Step throughput row. The status-only entries below should be treated as coverage gaps, not as a recommendation.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | OOM | OOM |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | OOM | OOM |
| bf16 | interval2 | OOM | OOM |
| bf16 | seg2-stride4 | OOM | OOM |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | OOM | OOM |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | OOM | OOM |
| int8-sdnq-hadamard | interval2 | OOM | OOM |
| int8-sdnq-hadamard | seg2-stride4 | OOM | OOM |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | OOM | OOM |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | OOM | OOM |
| fp8-torchao | interval2 | OOM | OOM |
| fp8-torchao | seg2-stride4 | OOM | OOM |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### Destilacao AnyFlow no Anima

Exemplo: `anima-anyflow-stage1.peft-lora`. Resolucao: 1024x1024. Esta linha mede destilacao AnyFlow usando Anima, nao o exemplo LoRA de Anima puro. Use `anima.peft-lora` para treinamento de imagem Anima puro em 1024x1024.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 1.022 / 17.26 | 0.719 / 17.21 |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 1.232 / 5.60 | 0.903 / 5.56 |
| bf16 | interval2 | 1.252 / 5.60 | 0.897 / 5.56 |
| bf16 | seg2-stride4 | 1.244 / 5.60 | 0.898 / 5.56 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 5.417 / 18.61 | 4.974 / 18.57 |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 4.562 / 4.36 | 4.019 / 4.31 |
| int8-sdnq-hadamard | interval2 | 3.723 / 4.36 | 3.196 / 4.31 |
| int8-sdnq-hadamard | seg2-stride4 | 3.658 / 4.36 | 3.140 / 4.31 |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | 2.242 / 45.71 | OOM |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 2.810 / 5.51 | 2.576 / 5.46 |
| fp8-torchao | interval2 | 2.846 / 5.51 | 2.581 / 5.46 |
| fp8-torchao | seg2-stride4 | 2.766 / 5.51 | 2.567 / 5.46 |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### AuraFlow

Example: `auraflow.peft-lora`. Resolution: 1024x1024.

Nota: AuraFlow suporta quantizacao com SDNQ e TorchAO. As linhas quantizadas sem checkpointing estao abaixo; as linhas quantizadas com checkpointing precisam de uma nova medicao completa antes de receber numeros aqui.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.180 / 19.19 | 0.233 / 19.12 |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 1.824 / 13.37 | 0.833 / 13.32 |
| bf16 | interval2 | 1.764 / 16.21 | 0.877 / 16.14 |
| bf16 | seg2-stride4 | 1.771 / 16.21 | 0.887 / 16.14 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 0.642 / 12.88 | 0.610 / 12.87 |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | not run | not run |
| int8-sdnq-hadamard | interval2 | not run | not run |
| int8-sdnq-hadamard | seg2-stride4 | not run | not run |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | 0.621 / 24.53 | 0.757 / 24.44 |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | not run | not run |
| fp8-torchao | interval2 | not run | not run |
| fp8-torchao | seg2-stride4 | not run | not run |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### Boogu Image

Example: `boogu-image-v0.1.peft-lora`. Resolution: 1024x1024.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.694 / 59.14 | OOM |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 0.907 / 23.44 | 2.641 / 23.39 |
| bf16 | interval2 | 0.912 / 23.44 | 2.649 / 23.39 |
| bf16 | seg2-stride4 | 0.911 / 23.44 | 2.648 / 23.39 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 1.488 / 53.24 | OOM |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 1.878 / 15.20 | 3.309 / 15.15 |
| int8-sdnq-hadamard | interval2 | 1.630 / 34.12 | 2.577 / 34.07 |
| int8-sdnq-hadamard | seg2-stride4 | 1.656 / 34.11 | 2.574 / 34.06 |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | OOM | OOM |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 1.713 / 18.48 | 3.778 / 18.44 |
| fp8-torchao | interval2 | 1.731 / 18.48 | 3.777 / 18.44 |
| fp8-torchao | seg2-stride4 | 1.721 / 18.48 | 3.779 / 18.44 |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### Chroma

Example: `chroma.peft-lora`. Resolution: 1024x1024.

Note: Checkpointed Chroma rows use `attention_mechanism=native-efficient`, which was the stable attention path for this sweep.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.454 / 26.18 | 0.559 / 26.13 |
| bf16 | activation-offload | 4.873 / 18.74 | 4.793 / 18.69 |
| bf16 | layer | 1.276 / 17.67 | 1.430 / 17.63 |
| bf16 | interval2 | 1.204 / 21.80 | 1.349 / 21.75 |
| bf16 | seg2-stride4 | 1.200 / 21.78 | 1.382 / 21.74 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 1.083 / 21.42 | 1.061 / 21.37 |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 1.714 / 10.41 | 1.646 / 10.36 |
| int8-sdnq-hadamard | interval2 | 1.443 / 15.72 | 1.391 / 15.68 |
| int8-sdnq-hadamard | seg2-stride4 | 1.428 / 15.71 | 1.323 / 15.67 |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | 1.122 / 45.44 | OOM |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 1.821 / 10.64 | 2.104 / 10.60 |
| fp8-torchao | interval2 | 1.447 / 27.49 | 1.871 / 27.44 |
| fp8-torchao | seg2-stride4 | 1.431 / 27.49 | 1.877 / 27.44 |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### Cosmos 2 Image

Example: `cosmos2image.lycoris-lokr`. Resolution: 512x512.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.336 / 8.05 | 0.316 / 8.00 |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 0.567 / 4.09 | 0.559 / 4.04 |
| bf16 | interval2 | 0.595 / 4.09 | 0.544 / 4.04 |
| bf16 | seg2-stride4 | 0.598 / 4.09 | 0.546 / 4.04 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 0.831 / 6.56 | 0.783 / 6.56 |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 1.315 / 2.44 | 1.288 / 2.39 |
| int8-sdnq-hadamard | interval2 | 1.346 / 2.44 | 1.321 / 2.39 |
| int8-sdnq-hadamard | seg2-stride4 | 1.413 / 2.44 | 1.274 / 2.39 |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | 0.850 / 13.30 | 0.840 / 13.25 |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 1.692 / 2.70 | 1.560 / 2.65 |
| fp8-torchao | interval2 | 1.626 / 2.70 | 1.544 / 2.65 |
| fp8-torchao | seg2-stride4 | 1.607 / 2.70 | 1.555 / 2.65 |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### Cosmos 3

Example: `cosmos3-edge-image-24g.lycoris-lokr`. Resolution: 1024 px.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 2.747 / 8.90 | 2.904 / 8.86 |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 2.602 / 8.90 | 2.606 / 8.86 |
| bf16 | interval2 | 2.658 / 8.90 | 2.567 / 8.86 |
| bf16 | seg2-stride4 | 2.628 / 8.90 | 2.899 / 8.86 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 17.112 / 6.21 | 17.965 / 6.16 |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 3.238 / 6.21 | 2.891 / 6.16 |
| int8-sdnq-hadamard | interval2 | 3.253 / 6.21 | 2.916 / 6.16 |
| int8-sdnq-hadamard | seg2-stride4 | 3.254 / 6.21 | 2.923 / 6.16 |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | 1.849 / 17.84 | 1.559 / 17.80 |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 1.869 / 17.84 | 1.599 / 17.79 |
| fp8-torchao | interval2 | 1.855 / 17.84 | 1.600 / 17.80 |
| fp8-torchao | seg2-stride4 | 1.859 / 17.84 | 1.523 / 17.80 |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### ERNIE 4.5 Image

Example: `ernie.peft-lora`. Resolution: 512x512.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.711 / 15.61 | 1.282 / 15.56 |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 1.110 / 4.94 | 1.840 / 4.90 |
| bf16 | interval2 | 0.876 / 10.16 | 1.536 / 10.12 |
| bf16 | seg2-stride4 | 0.874 / 10.16 | 1.532 / 10.12 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 2.782 / 13.75 | 2.722 / 13.70 |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 4.262 / 2.98 | 4.393 / 2.94 |
| int8-sdnq-hadamard | interval2 | 3.547 / 8.26 | 3.380 / 8.21 |
| int8-sdnq-hadamard | seg2-stride4 | 3.366 / 8.26 | 3.457 / 8.21 |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | 2.432 / 14.03 | 2.303 / 13.98 |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 3.920 / 3.26 | 4.008 / 3.22 |
| fp8-torchao | interval2 | 3.316 / 8.54 | 2.973 / 8.49 |
| fp8-torchao | seg2-stride4 | 2.994 / 8.54 | 3.046 / 8.49 |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### HeartMula

Example: `heartmula.peft-lora`. Treinamento com tokens de áudio.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | failed | failed |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | failed | failed |
| bf16 | interval2 | unsupported | unsupported |
| bf16 | seg2-stride4 | unsupported | unsupported |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | OOM | OOM |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | failed | failed |
| int8-sdnq-hadamard | interval2 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | failed | failed |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | failed | failed |
| fp8-torchao | interval2 | unsupported | unsupported |
| fp8-torchao | seg2-stride4 | unsupported | unsupported |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### HiDream

Example: `hidream.peft-lora`. Resolution: 512x512.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.515 / 44.58 | OOM |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 0.928 / 33.57 | 1.043 / 33.52 |
| bf16 | interval2 | 0.971 / 33.57 | 1.002 / 33.52 |
| bf16 | seg2-stride4 | 0.952 / 33.57 | 1.010 / 33.52 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | not measured | not measured |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 2.497 / 17.99 | 2.058 / 17.93 |
| int8-sdnq-hadamard | interval2 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | not measured | not measured |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 4.920 / 18.10 | 4.338 / 18.05 |
| fp8-torchao | interval2 | unsupported | unsupported |
| fp8-torchao | seg2-stride4 | unsupported | unsupported |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

SDNQ Hadamard numbers use `sdnq_compile_mode=eager`. The compiled SDNQ path quantizes HiDream, but this sweep spent the first training step in Inductor dequantizer compilation, so it is not listed as a throughput row.

### HunyuanVideo

Exemplo: `hunyuanvideo-1.5-t2v.peft-lora`. Forma de training: video buckets de 480 pixel-area, 48 frames, batch 2.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | OOM | OOM |
| bf16 | activation-offload | not run | not run |
| bf16 | layer | 7.682 / 26.35 | 22.816 / 26.30 |
| bf16 | interval2 | 7.398 / 26.11 | 22.772 / 26.06 |
| bf16 | seg2-stride4 | OOM | OOM |
| bf16 | seg2-stride4-offload | 10.679 / 58.37 | not run |
| int8-sdnq-hadamard | none | not run | not run |
| int8-sdnq-hadamard | activation-offload | not run | not run |
| int8-sdnq-hadamard | layer | 11.765 / 25.96 | 34.464 / 25.92 |
| int8-sdnq-hadamard | interval2 | not run | not run |
| int8-sdnq-hadamard | seg2-stride4 | not run | not run |
| int8-sdnq-hadamard | seg2-stride4-offload | not run | not run |
| fp8-torchao | none | not run | not run |
| fp8-torchao | activation-offload | not run | not run |
| fp8-torchao | layer | 10.516 / 33.55 | 32.003 / 33.53 |
| fp8-torchao | interval2 | not run | not run |
| fp8-torchao | seg2-stride4 | not run | not run |
| fp8-torchao | seg2-stride4-offload | not run | not run |

HunyuanVideo usa muitas activations nessa forma de training. Per-block e interval-2 cabem bem; sem checkpointing nao cabe em um H100 de 80 GB. `seg2-stride4` so coube neste sweep com attention activation offload, e essa linha e uma saida para memoria, nao uma recomendacao de velocidade. SDNQ Hadamard funciona, mas formas variaveis de conditioning ainda disparam compilacao de kernels dinamicos durante a medicao.

### Ideogram 4.0

Exemplo: `ideogram-fp8.peft-lora`. Resolucao: 1024x1024. O flavour fp8 usa o checkpoint fp8 nativo weight-only do Ideogram 4 (`base_model_precision=no_change`).

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| fp8-native | none | failed | OOM |
| fp8-native | activation-offload | unsupported | unsupported |
| fp8-native | layer | 1.033 / 12.57 | 3.101 / 11.82 |
| fp8-native | interval2 | 1.030 / 12.33 | 3.098 / 11.82 |
| fp8-native | seg2-stride4 | 1.031 / 12.33 | 3.088 / 11.82 |
| fp8-native | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 0.702 / 61.78 | OOM |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 1.033 / 12.33 | 3.030 / 11.82 |
| int8-sdnq-hadamard | interval2 | 1.032 / 12.33 | 3.034 / 11.82 |
| int8-sdnq-hadamard | seg2-stride4 | 1.028 / 12.33 | 3.035 / 11.82 |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |

### Kandinsky 5 Image

Example: `kandinsky5-image-6b-t2i.lycoris-lokr`. Resolution: 1024x1024.

Note: batch 3 at 1024x1024 needs full checkpointing on both cards. SDNQ with Hadamard is the best low-VRAM row; H100 can also use partial checkpointing with SDNQ, but only near the top of the card.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | OOM | OOM |
| bf16 | activation-offload | 6.956 / 25.52 | 10.189 / 25.42 |
| bf16 | layer | 6.590 / 25.58 | 9.458 / 25.55 |
| bf16 | interval2 | OOM | OOM |
| bf16 | seg2-stride4 | OOM | OOM |
| bf16 | seg2-stride4-offload | OOM | OOM |
| int8-sdnq-hadamard | none | OOM | OOM |
| int8-sdnq-hadamard | activation-offload | 7.186 / 20.00 | 10.141 / 19.95 |
| int8-sdnq-hadamard | layer | 6.830 / 20.12 | 9.362 / 20.08 |
| int8-sdnq-hadamard | interval2 | 5.746 / 75.67 | OOM |
| int8-sdnq-hadamard | seg2-stride4 | OOM | OOM |
| int8-sdnq-hadamard | seg2-stride4-offload | 6.057 / 71.46 | OOM |
| fp8-torchao | none | OOM | OOM |
| fp8-torchao | activation-offload | 8.319 / 24.29 | 14.716 / 24.24 |
| fp8-torchao | layer | 7.976 / 24.40 | 13.949 / 24.35 |
| fp8-torchao | interval2 | OOM | OOM |
| fp8-torchao | seg2-stride4 | OOM | OOM |
| fp8-torchao | seg2-stride4-offload | OOM | OOM |

### Kandinsky 5 Video

Example: `kandinsky5-video-2b-t2v.peft-lora`. Resolution: 768x512, 81f.

Kandinsky 5 video is activation-heavy at this frame count. Full block checkpointing is the practical baseline on both cards. On H100, `interval2` and `seg2-stride4` are faster when they fit; on L40S, SDNQ `interval2` is the only partial-checkpoint row here that fits without attention activation offload.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | OOM | OOM |
| bf16 | activation-offload | 2.580 / 9.49 | 7.136 / 9.45 |
| bf16 | layer | 2.267 / 9.83 | 6.379 / 9.79 |
| bf16 | interval2 | 1.967 / 44.57 | OOM |
| bf16 | seg2-stride4 | 1.971 / 46.62 | OOM |
| bf16 | seg2-stride4-offload | 2.275 / 37.99 | 6.249 / 37.94 |
| int8-sdnq-hadamard | none | OOM | OOM |
| int8-sdnq-hadamard | activation-offload | 2.844 / 8.07 | 7.234 / 8.02 |
| int8-sdnq-hadamard | layer | 2.460 / 8.40 | 6.509 / 8.35 |
| int8-sdnq-hadamard | interval2 | 2.126 / 43.12 | 5.641 / 43.08 |
| int8-sdnq-hadamard | seg2-stride4 | 2.125 / 45.19 | OOM |
| int8-sdnq-hadamard | seg2-stride4-offload | 2.451 / 36.56 | 6.322 / 36.51 |
| fp8-torchao | none | OOM | OOM |
| fp8-torchao | activation-offload | 3.867 / 15.11 | 11.501 / 15.06 |
| fp8-torchao | layer | 3.579 / 15.28 | 10.822 / 15.24 |
| fp8-torchao | interval2 | OOM | OOM |
| fp8-torchao | seg2-stride4 | OOM | OOM |
| fp8-torchao | seg2-stride4-offload | OOM | OOM |

### Kolors

Example: `kolors.peft-lora`. Resolution: 1024x1024.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.635 / 7.22 | 0.628 / 7.17 |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 1.118 / 5.36 | 1.065 / 5.31 |
| bf16 | interval2 | 1.105 / 5.36 | 1.068 / 5.31 |
| bf16 | seg2-stride4 | 1.110 / 5.36 | 1.072 / 5.31 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 1.860 / 5.68 | 1.726 / 5.63 |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 2.967 / 3.38 | 2.803 / 3.33 |
| int8-sdnq-hadamard | interval2 | 2.770 / 3.32 | 2.655 / 3.27 |
| int8-sdnq-hadamard | seg2-stride4 | 2.805 / 3.32 | 2.742 / 3.27 |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | 1.629 / 8.79 | 1.637 / 8.75 |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 3.044 / 3.50 | 3.013 / 3.45 |
| fp8-torchao | interval2 | 3.040 / 3.50 | 3.019 / 3.45 |
| fp8-torchao | seg2-stride4 | 2.988 / 3.50 | 2.935 / 3.45 |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### Krea 2

Exemplo: `krea2.peft-lora`. Resolucao de treino: crop quadrado de 512 px. Configuracao de validacao do exemplo: 1024x1024; a validacao ficou desativada no benchmark.

A tabela principal usa regional compilation. Isso ajuda o step speed do Krea2, mas nao e uma comparacao limpa de VRAM: o compiled graph/workspace mantem o pico perto do pico sem checkpointing em varios modos. Uma execucao de controle bf16 com regional compilation desativado mostrou que o checkpointing esta conectado e tem o formato esperado de memoria/velocidade:

A linha `activation-offload` aqui significa full-block checkpointing mais attention activation offload. Contra full-block `layer` checkpointing sozinho, attention offload nao reduziu o peak VRAM do Krea2 neste shape; principalmente adicionou CPU transfer overhead.

| Mode | H100 no-compile | L40S no-compile |
| --- | ---: | ---: |
| none | 0.272 / 40.09 | 0.661 / 40.01 |
| layer | 0.371 / 30.06 | 0.919 / 30.01 |
| seg2-stride4 | 0.317 / 34.75 | 0.788 / 34.70 |
| activation-offload | 0.657 / 30.50 | 1.341 / 30.30 |

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.275 / 40.06 | 0.662 / 40.01 |
| bf16 | activation-offload | 0.416 / 34.62 | 0.822 / 34.57 |
| bf16 | layer | 0.268 / 40.06 | 0.665 / 40.01 |
| bf16 | interval2 | 0.274 / 40.06 | 0.663 / 40.01 |
| bf16 | seg2-stride4 | 0.279 / 40.06 | 0.663 / 40.01 |
| bf16 | seg2-stride4-offload | 0.404 / 34.62 | 0.819 / 34.57 |
| int8-sdnq-hadamard | none | 0.462 / 27.01 | 0.807 / 26.96 |
| int8-sdnq-hadamard | activation-offload | 0.773 / 21.57 | 1.012 / 21.53 |
| int8-sdnq-hadamard | layer | 0.473 / 27.01 | 0.802 / 26.96 |
| int8-sdnq-hadamard | interval2 | 0.474 / 27.01 | 0.804 / 26.96 |
| int8-sdnq-hadamard | seg2-stride4 | 0.472 / 27.01 | 0.802 / 26.96 |
| int8-sdnq-hadamard | seg2-stride4-offload | 0.744 / 21.57 | 1.007 / 21.53 |
| fp8-torchao | none | 0.689 / 51.63 | OOM |
| fp8-torchao | activation-offload | 0.975 / 37.77 | 2.058 / 37.73 |
| fp8-torchao | layer | 0.689 / 51.63 | OOM |
| fp8-torchao | interval2 | 0.684 / 51.63 | OOM |
| fp8-torchao | seg2-stride4 | 0.674 / 51.63 | OOM |
| fp8-torchao | seg2-stride4-offload | 0.965 / 37.77 | 2.053 / 37.73 |

### LongCat Image

Exemplo: `longcat-image.peft-lora`. Resolucao de treino: 512 px quadrada; resolucao de validacao: 1024x1024. As linhas usam `attention_mechanism=native-flash`.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.193 / 16.47 | 0.262 / 16.42 |
| bf16 | activation-offload | 0.544 / 12.73 | 0.543 / 12.69 |
| bf16 | layer | 0.327 / 12.38 | 0.370 / 12.34 |
| bf16 | interval2 | 0.257 / 14.38 | 0.313 / 14.34 |
| bf16 | seg2-stride4 | 0.263 / 14.36 | 0.316 / 14.31 |
| bf16 | seg2-stride4-offload | 0.446 / 13.42 | 0.492 / 13.38 |
| int8-sdnq-hadamard | none | 0.578 / 12.54 | 0.537 / 12.45 |
| int8-sdnq-hadamard | activation-offload | 1.184 / 7.43 | 1.185 / 7.39 |
| int8-sdnq-hadamard | layer | 0.901 / 7.19 | 0.911 / 7.09 |
| int8-sdnq-hadamard | interval2 | 0.718 / 9.73 | 0.695 / 9.68 |
| int8-sdnq-hadamard | seg2-stride4 | 0.735 / 9.72 | 0.717 / 9.68 |
| int8-sdnq-hadamard | seg2-stride4-offload | 1.056 / 8.49 | 1.011 / 8.45 |
| fp8-torchao | none | 0.602 / 25.19 | 0.834 / 25.14 |
| fp8-torchao | activation-offload | 1.662 / 7.75 | 1.844 / 7.70 |
| fp8-torchao | layer | 0.984 / 7.57 | 1.080 / 7.53 |
| fp8-torchao | interval2 | 0.750 / 16.15 | 0.938 / 16.10 |
| fp8-torchao | seg2-stride4 | 0.760 / 16.13 | 0.961 / 16.09 |
| fp8-torchao | seg2-stride4-offload | 1.287 / 13.02 | 1.653 / 12.98 |

### LongCat Video

Exemplo: `longcat-video.peft-lora+ramtorch`. Resolucao: 832x480, 81f. As linhas usam `attention_mechanism=native-flash`.

LongCat Video usa muitas activations neste shape. Full per-block checkpointing e a linha pratica. As linhas partial checkpoint (`interval2`, `seg2-stride4`) nao cabem aqui, mesmo com attention activation offload na linha strided. Attention activation offload simples cabe para bf16 e SDNQ, mas e muito mais lento que full checkpointing.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | OOM (77.86 GiB) | OOM (43.30 GiB) |
| bf16 | activation-offload | 25.774 / 37.36 | 49.149 / 37.14 |
| bf16 | layer | 7.448 / 23.73 | 24.866 / 23.68 |
| bf16 | interval2 | OOM (76.41 GiB) | OOM (42.80 GiB) |
| bf16 | seg2-stride4 | OOM (76.42 GiB) | OOM (43.06 GiB) |
| bf16 | seg2-stride4-offload | OOM (76.72 GiB) | OOM (42.29 GiB) |
| int8-sdnq-hadamard | none | OOM (77.40 GiB) | OOM (43.59 GiB) |
| int8-sdnq-hadamard | activation-offload | 30.887 / 35.28 | 61.270 / 35.24 |
| int8-sdnq-hadamard | layer | 8.444 / 21.60 | 25.164 / 21.55 |
| int8-sdnq-hadamard | interval2 | OOM (76.01 GiB) | OOM (42.47 GiB) |
| int8-sdnq-hadamard | seg2-stride4 | OOM (77.01 GiB) | OOM (43.02 GiB) |
| int8-sdnq-hadamard | seg2-stride4-offload | OOM (76.42 GiB) | OOM (42.57 GiB) |
| fp8-torchao | none | OOM (75.87 GiB) | OOM (41.28 GiB) |
| fp8-torchao | activation-offload | 30.163 / 47.88 | OOM (40.11 GiB) |
| fp8-torchao | layer | 8.343 / 34.16 | 24.659 / 34.07 |
| fp8-torchao | interval2 | OOM (74.63 GiB) | OOM (40.85 GiB) |
| fp8-torchao | seg2-stride4 | OOM (75.37 GiB) | OOM (41.19 GiB) |
| fp8-torchao | seg2-stride4-offload | OOM (75.55 GiB) | OOM (40.94 GiB) |

### LTXVideo 0.9.5

Example: `ltxvideo-0.9.5-t2v.peft-lora`. Resolution: 768x512, 49f.

Os numeros sao segundos warm por step / GiB pico. A media da execucao completa inclui setup e compile, e fica nos artifacts do sweep.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.274 / 8.77 | 0.275 / 8.72 |
| bf16 | layer | 0.462 / 4.21 | 0.459 / 4.16 |
| bf16 | interval2 | 0.449 / 4.30 | 0.433 / 4.25 |
| bf16 | seg2-stride4 | 0.359 / 6.40 | 0.357 / 6.35 |
| int8-sdnq-hadamard | none | 0.688 / 7.05 | 0.640 / 6.90 |
| int8-sdnq-hadamard | layer | 1.094 / 2.59 | 1.081 / 2.44 |
| int8-sdnq-hadamard | interval2 | 1.112 / 2.57 | 1.073 / 2.53 |
| int8-sdnq-hadamard | seg2-stride4 | 0.887 / 4.61 | 0.817 / 4.56 |
| fp8-torchao | none | 0.655 / 17.45 | 0.735 / 17.41 |
| fp8-torchao | layer | 1.226 / 2.94 | 1.206 / 2.89 |
| fp8-torchao | interval2 | 1.259 / 3.40 | 1.233 / 3.35 |
| fp8-torchao | seg2-stride4 | 0.933 / 9.96 | 0.901 / 9.91 |
| fp8wo-torchao | none | 0.328 / 10.06 | 0.325 / 10.01 |
| fp8wo-torchao | layer | 0.567 / 2.64 | 0.540 / 2.59 |
| fp8wo-torchao | interval2 | 0.555 / 2.84 | 0.531 / 2.79 |
| fp8wo-torchao | seg2-stride4 | 0.443 / 6.20 | 0.432 / 6.15 |

As linhas de attention activation offload nao sao suportadas para LTXVideo 0.9 neste sweep.

### LTXVideo2 2.3

Example: `ltxvideo2-2.3-dev-720p-single-gpu.peft-lora+sdnq-hadamard`. Resolution: 1280x704, 49f.

Note: LTXVideo2 2.3 should be read from the no-regional-compile rows in this sweep; regional compile raised memory pressure for this model.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | OOM | OOM |
| bf16 | activation-offload | 6.350 / 58.36 | OOM |
| bf16 | layer | 3.993 / 47.95 | OOM |
| bf16 | interval2 | 3.977 / 48.83 | OOM |
| bf16 | seg2-stride4 | OOM | OOM |
| bf16 | seg2-stride4-offload | 5.554 / 75.64 | OOM |
| int8-sdnq-hadamard | none | OOM | OOM |
| int8-sdnq-hadamard | activation-offload | 10.102 / 38.20 | 9.852 / 38.15 |
| int8-sdnq-hadamard | layer | 7.753 / 27.78 | 7.579 / 27.73 |
| int8-sdnq-hadamard | interval2 | 7.733 / 28.66 | 7.288 / 28.61 |
| int8-sdnq-hadamard | seg2-stride4 | 6.522 / 61.50 | OOM |
| int8-sdnq-hadamard | seg2-stride4-offload | 8.659 / 55.48 | OOM |
| fp8-torchao | none | OOM | OOM |
| fp8-torchao | activation-offload | failed | 22.917 / 38.57 |
| fp8-torchao | layer | 8.580 / 30.27 | 10.381 / 30.22 |
| fp8-torchao | interval2 | 8.660 / 33.71 | 10.661 / 33.66 |
| fp8-torchao | seg2-stride4 | OOM | OOM |
| fp8-torchao | seg2-stride4-offload | failed | OOM |

### Lumina2

Example: `lumina2.peft-lora`. Resolution: 512x512.

Nota: Lumina2 agora usa o caminho segmented whole-block. `interval2` faz checkpoint de cada segmento de dois blocks; `seg2-stride4` checkpointa dois blocks, deixa os dois seguintes manterem activations e repete. Attention activation offload nao fez parte desta rodada de Lumina2.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.235 / 15.99 | 0.332 / 15.94 |
| bf16 | layer | 0.384 / 6.60 | 0.457 / 6.56 |
| bf16 | interval2 | 0.356 / 6.87 | 0.424 / 6.82 |
| bf16 | seg2-stride4 | 0.295 / 11.43 | 0.377 / 11.38 |
| int8-sdnq-hadamard | none | 0.584 / 13.59 | 0.541 / 13.55 |
| int8-sdnq-hadamard | layer | 0.899 / 4.21 | 0.827 / 4.16 |
| int8-sdnq-hadamard | interval2 | 0.865 / 4.48 | 0.835 / 4.43 |
| int8-sdnq-hadamard | seg2-stride4 | 0.719 / 9.03 | 0.707 / 8.99 |
| fp8-torchao | none | 0.598 / 28.03 | 0.763 / 27.98 |
| fp8-torchao | layer | 0.950 / 5.93 | 0.967 / 5.88 |
| fp8-torchao | interval2 | 0.938 / 6.70 | 0.974 / 6.66 |
| fp8-torchao | seg2-stride4 | 0.765 / 17.36 | 0.901 / 17.32 |
| fp8wo-torchao | none | 0.273 / 17.97 | 0.389 / 17.93 |
| fp8wo-torchao | layer | 0.452 / 4.99 | 0.525 / 4.94 |
| fp8wo-torchao | interval2 | 0.427 / 5.40 | 0.522 / 5.35 |
| fp8wo-torchao | seg2-stride4 | 0.360 / 11.68 | 0.466 / 11.64 |

### MageFlow

Example: `mageflow-image-24g.peft-lora`. Resolution: 1024x1024.

Nota: o caminho de imagem com shapes variaveis do MageFlow em 1024px se beneficia principalmente de attention activation offload e FP8 weight-only. Os modos de block checkpointing sao validos, mas nao reduziram o pico de residencia medido neste sweep.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 7.735 / 36.74 | 7.658 / 36.69 |
| bf16 | activation-offload | 8.902 / 23.05 | 9.477 / 23.00 |
| bf16 | layer | 7.805 / 36.74 | 7.504 / 36.69 |
| bf16 | interval2 | 8.036 / 36.74 | 7.762 / 36.69 |
| bf16 | seg2-stride4 | 7.882 / 36.74 | 7.531 / 36.69 |
| bf16 | seg2-stride4-offload | 8.833 / 23.05 | 9.288 / 23.01 |
| int8-sdnq-hadamard | none | 81.016 / 37.38 | 94.991 / 37.34 |
| fp8wo-torchao | none | 5.454 / 36.86 | 5.772 / 36.82 |
| fp8wo-torchao | activation-offload | 6.295 / 23.18 | 6.738 / 23.14 |
| fp8wo-torchao | seg2-stride4 | 5.542 / 36.86 | 5.595 / 36.82 |

### OmniGen

Example: `omnigen.lycoris-lokr`. Resolution: 1024x1024.

Nota: OmniGen usa prompts como token IDs em vez de embeddings de texto em cache. Estas linhas medem os caminhos suportados sem checkpointing e com checkpointing torch de bloco completo; os controles interval, segmented-stride e attention-offload nao estao implementados para esta familia neste sweep.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.425 / 14.20 | 0.293 / 14.24 |
| bf16 | layer | 0.597 / 10.13 | 0.389 / 10.09 |
| int8-sdnq-hadamard | none | 1.523 / 11.00 | 1.388 / 11.06 |
| int8-sdnq-hadamard | layer | 1.312 / 6.75 | 1.064 / 6.70 |
| fp8-torchao | none | 0.690 / 19.25 | 0.608 / 19.30 |
| fp8-torchao | layer | 1.069 / 7.09 | 0.824 / 7.04 |
| fp8wo-torchao | none | 0.454 / 17.71 | 0.377 / 17.73 |
| fp8wo-torchao | layer | 0.646 / 6.98 | 0.534 / 6.94 |

### PixArt

Example: `pixart.lycoris-lokr`. Resolution: 1024x1024.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 1.700 / 41.73 | 1.734 / 41.67 |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 2.433 / 5.63 | 2.346 / 5.58 |
| bf16 | interval2 | 2.440 / 6.01 | 2.348 / 5.96 |
| bf16 | seg2-stride4 | 2.092 / 23.90 | 2.072 / 23.86 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 1.905 / 47.58 | OOM |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 2.738 / 6.07 | 2.937 / 6.03 |
| int8-sdnq-hadamard | interval2 | 2.734 / 6.03 | 2.943 / 5.99 |
| int8-sdnq-hadamard | seg2-stride4 | 2.336 / 26.85 | 2.596 / 26.81 |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | OOM | OOM |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 3.254 / 8.06 | 4.646 / 8.01 |
| fp8-torchao | interval2 | 3.260 / 9.66 | 4.649 / 9.61 |
| fp8-torchao | seg2-stride4 | 2.827 / 63.27 | OOM |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### Qwen Image

Example: `qwen_image.peft-lora`. Resolution: 1024x1024.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | OOM | OOM |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 1.202 / 41.04 | 3.377 / 40.99 |
| bf16 | interval2 | 1.201 / 41.04 | 3.382 / 40.99 |
| bf16 | seg2-stride4 | 1.205 / 41.03 | 3.385 / 40.99 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 1.675 / 63.48 | OOM |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 2.640 / 24.09 | 3.928 / 24.05 |
| int8-sdnq-hadamard | interval2 | 2.722 / 24.09 | 3.918 / 24.05 |
| int8-sdnq-hadamard | seg2-stride4 | 2.663 / 24.09 | 3.919 / 24.05 |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | OOM | OOM |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 3.088 / 25.34 | 6.172 / 25.29 |
| fp8-torchao | interval2 | 3.095 / 25.34 | 6.173 / 25.29 |
| fp8-torchao | seg2-stride4 | 3.125 / 25.34 | 6.141 / 25.29 |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### Sana

Example: `sana.lycoris-lokr`. Resolution: 1024x1024.

Note: Sana has interval checkpointing; stride is not a separate segmented schedule for this family in the measured rows.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.529 / 23.71 | 0.597 / 23.67 |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 0.529 / 23.72 | 0.596 / 23.66 |
| bf16 | interval2 | 0.530 / 23.72 | 0.598 / 23.66 |
| bf16 | seg2-stride4 | unsupported | unsupported |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 0.554 / 22.92 | 0.590 / 22.88 |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 0.554 / 22.92 | 0.589 / 22.88 |
| int8-sdnq-hadamard | interval2 | 0.556 / 22.92 | 0.591 / 22.88 |
| int8-sdnq-hadamard | seg2-stride4 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | 0.633 / 33.67 | 0.753 / 33.62 |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 0.633 / 33.67 | 0.755 / 33.62 |
| fp8-torchao | interval2 | 0.631 / 33.67 | 0.759 / 33.62 |
| fp8-torchao | seg2-stride4 | unsupported | unsupported |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### SanaVideo

Example: `sanavideo-2b-480p.peft-lora`. Resolution: 832x480, 49f.

Note: SanaVideo usa linear attention, entao attention activation offload continua unsupported. Segmented whole-block checkpointing e suportado no caminho standard.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.599 / 59.15 | OOM |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 0.597 / 59.15 | OOM |
| bf16 | interval2 | not run | OOM |
| bf16 | seg2-stride4 | not run | OOM |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 0.641 / 58.36 | OOM |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 0.641 / 58.36 | OOM |
| int8-sdnq-hadamard | interval2 | not run | OOM |
| int8-sdnq-hadamard | seg2-stride4 | not run | OOM |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | OOM | OOM |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | OOM | OOM |
| fp8-torchao | interval2 | OOM | OOM |
| fp8-torchao | seg2-stride4 | OOM | OOM |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### SD 1.x

Example: `sd1x-dreamshaper.peft-lora`. Resolution: 512x512.

Nota: SD1x usa o caminho UNet do diffusers. O checkpointing por camada funciona, mas os controles interval e segmented stride nao estao conectados para esta familia.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.181 / 2.87 | 0.176 / 2.83 |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 0.305 / 1.98 | 0.292 / 1.93 |
| bf16 | interval2 | unsupported | unsupported |
| bf16 | seg2-stride4 | unsupported | unsupported |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 0.446 / 3.07 | 0.431 / 3.04 |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 0.701 / 1.79 | 0.666 / 1.74 |
| int8-sdnq-hadamard | interval2 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | 0.410 / 4.23 | 0.401 / 4.18 |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 0.785 / 1.96 | 0.756 / 1.91 |
| fp8-torchao | interval2 | unsupported | unsupported |
| fp8-torchao | seg2-stride4 | unsupported | unsupported |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### SD3

Example: `sd3.peft-lora`. Resolution: 1024x1024.

Nota: SD3 usa checkpointing segmentado contiguo real no caminho transformer simples. Attention activation offload e suportado; reduz bastante a VRAM, mas custa throughput.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.529 / 34.59 | 1.189 / 34.53 |
| bf16 | activation-offload | 1.335 / 9.68 | 2.850 / 9.63 |
| bf16 | layer | 0.721 / 7.67 | 1.607 / 7.62 |
| bf16 | interval2 | 0.723 / 8.93 | 1.606 / 8.88 |
| bf16 | seg2-stride4 | 0.620 / 20.14 | 1.398 / 20.09 |
| bf16 | seg2-stride4-offload | 1.229 / 12.21 | 2.639 / 12.16 |
| int8-sdnq-hadamard | none | 0.858 / 33.66 | 1.381 / 33.61 |
| int8-sdnq-hadamard | activation-offload | 1.845 / 8.90 | 3.297 / 8.85 |
| int8-sdnq-hadamard | layer | 1.265 / 6.58 | 1.857 / 6.53 |
| int8-sdnq-hadamard | interval2 | 1.264 / 7.30 | 1.858 / 7.26 |
| int8-sdnq-hadamard | seg2-stride4 | 1.049 / 19.02 | 1.625 / 18.98 |
| int8-sdnq-hadamard | seg2-stride4-offload | 1.527 / 12.42 | 3.023 / 12.37 |
| fp8-torchao | none | OOM | OOM |
| fp8-torchao | activation-offload | 4.707 / 10.27 | 9.814 / 10.23 |
| fp8-torchao | layer | 1.575 / 9.13 | 3.279 / 9.09 |
| fp8-torchao | interval2 | 1.576 / 12.62 | 3.282 / 12.58 |
| fp8-torchao | seg2-stride4 | 1.376 / 45.70 | OOM |
| fp8-torchao | seg2-stride4-offload | 4.184 / 26.17 | 8.714 / 26.12 |
| fp8wo-torchao | none | 0.567 / 35.21 | 1.252 / 35.17 |
| fp8wo-torchao | activation-offload | 1.392 / 8.71 | 2.921 / 8.67 |
| fp8wo-torchao | layer | 0.795 / 5.69 | 1.736 / 5.65 |
| fp8wo-torchao | interval2 | 0.797 / 7.08 | 1.734 / 7.03 |
| fp8wo-torchao | seg2-stride4 | 0.679 / 19.43 | 1.490 / 19.38 |
| fp8wo-torchao | seg2-stride4-offload | 1.257 / 12.00 | 2.676 / 11.96 |

### SDXL

Example: `sdxl.lycoris-lokr`. Resolution: 1024x1024.

Note: SDXL has real layer checkpointing. Interval and stride rows are included as coverage data, not as segmented-support recommendations.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.606 / 13.03 | 0.585 / 12.98 |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 1.080 / 6.53 | 1.029 / 6.48 |
| bf16 | interval2 | unsupported | unsupported |
| bf16 | seg2-stride4 | unsupported | unsupported |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 1.741 / 13.72 | 1.643 / 13.68 |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 2.820 / 4.64 | 2.647 / 4.59 |
| int8-sdnq-hadamard | interval2 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | 1.608 / 26.22 | 1.582 / 26.16 |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | 2.939 / 5.10 | 2.890 / 5.04 |
| fp8-torchao | interval2 | unsupported | unsupported |
| fp8-torchao | seg2-stride4 | unsupported | unsupported |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### Stable Cascade

Example: `cascade-stage-c.lycoris-lokr`. Resolution: 1024x1024.

Nota: stage C usa o prior em precision completa. Estas linhas foram executadas com `mixed_precision=no` e `base_model_precision=no_change`; linhas de precision base quantizada nao sao significativas para este modelo. Os modos interval e stride operam sobre a sequencia de micro-blocos Res/Timestep/Attention do UNet.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.884 / 51.52 | OOM |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 1.179 / 22.68 | 2.135 / 22.61 |
| bf16 | interval2 | 1.032 / 36.99 | 1.871 / 36.92 |
| bf16 | seg2-stride4 | 1.032 / 37.20 | 1.870 / 37.13 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | unsupported | unsupported |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | unsupported | unsupported |
| int8-sdnq-hadamard | interval2 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | unsupported | unsupported |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | unsupported | unsupported |
| fp8-torchao | interval2 | unsupported | unsupported |
| fp8-torchao | seg2-stride4 | unsupported | unsupported |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### Wan 2.1 T2V 1.3B

Example: `wan2.1-t2v-1.3b-480p-single-gpu.peft-lora+ramtorch`. Resolution: 832x480, 81f.

Note: Wan 1.3B should be read from the no-regional-compile/RamTorch rows; regional compile was not a useful throughput setting in this sweep.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 1.407 / 71.90 | OOM |
| bf16 | activation-offload | 3.472 / 8.78 | 7.179 / 8.66 |
| bf16 | layer | 2.099 / 4.73 | 4.459 / 4.68 |
| bf16 | interval2 | 2.139 / 6.32 | 4.514 / 6.27 |
| bf16 | seg2-stride4 | 1.806 / 39.25 | 3.921 / 39.21 |
| bf16 | seg2-stride4-offload | 2.993 / 22.66 | 6.493 / 22.61 |
| int8-sdnq-hadamard | none | 1.850 / 71.91 | OOM |
| int8-sdnq-hadamard | activation-offload | 4.204 / 8.72 | 7.387 / 8.68 |
| int8-sdnq-hadamard | layer | 2.790 / 4.70 | 4.989 / 4.65 |
| int8-sdnq-hadamard | interval2 | 2.874 / 6.29 | 5.093 / 6.24 |
| int8-sdnq-hadamard | seg2-stride4 | 2.558 / 39.27 | 4.393 / 39.22 |
| int8-sdnq-hadamard | seg2-stride4-offload | 3.711 / 22.67 | 6.695 / 22.63 |
| fp8-torchao | none | 1.727 / 73.57 | OOM |
| fp8-torchao | activation-offload | 4.061 / 10.08 | 7.404 / 9.96 |
| fp8-torchao | layer | 2.607 / 5.98 | 4.888 / 5.93 |
| fp8-torchao | interval2 | 2.744 / 7.57 | 4.916 / 7.52 |
| fp8-torchao | seg2-stride4 | 2.246 / 40.55 | 4.245 / 40.50 |
| fp8-torchao | seg2-stride4-offload | 3.602 / 24.02 | 6.683 / 23.91 |

### Wan 2.1 T2V 14B

Example: `wan2.1-t2v-14b-480p-single-gpu.peft-lora+ramtorch`. Resolution: 832x480, 81f.

Note: Wan 14B is mainly a fit test for activation savings. Status-only cells are still useful because they show which combinations reached the memory limit.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | OOM | OOM |
| bf16 | activation-offload | 13.144 / 36.80 | OOM |
| bf16 | layer | 7.162 / 16.28 | 21.770 / 16.23 |
| bf16 | interval2 | 7.172 / 19.62 | 21.777 / 19.58 |
| bf16 | seg2-stride4 | OOM | OOM |
| bf16 | seg2-stride4-offload | OOM | OOM |
| int8-sdnq-hadamard | none | unsupported | unsupported |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | unsupported | unsupported |
| int8-sdnq-hadamard | interval2 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | failed | failed |
| fp8-torchao | activation-offload | failed | failed |
| fp8-torchao | layer | failed | failed |
| fp8-torchao | interval2 | failed | failed |
| fp8-torchao | seg2-stride4 | failed | failed |
| fp8-torchao | seg2-stride4-offload | failed | failed |

### Wan S2V

Example: `wan-s2v-14b-480p.peft-lora+ramtorch`. Resolution: 832x480, 81f.

Note: Wan S2V is included as coverage data for the video/audio path. Treat failed cells as implementation coverage gaps.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | failed | failed |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | failed | failed |
| bf16 | interval2 | unsupported | unsupported |
| bf16 | seg2-stride4 | unsupported | unsupported |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | failed | failed |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | failed | failed |
| int8-sdnq-hadamard | interval2 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4 | unsupported | unsupported |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8-torchao | none | failed | failed |
| fp8-torchao | activation-offload | unsupported | unsupported |
| fp8-torchao | layer | failed | failed |
| fp8-torchao | interval2 | unsupported | unsupported |
| fp8-torchao | seg2-stride4 | unsupported | unsupported |
| fp8-torchao | seg2-stride4-offload | unsupported | unsupported |

### Z-Image Turbo

Example: `z-image-turbo.peft-lora`. Resolution: 1024x1024.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.243 / 21.25 | 0.316 / 21.21 |
| bf16 | activation-offload | 0.837 / 13.24 | 0.805 / 13.19 |
| bf16 | layer | 0.479 / 12.87 | 0.493 / 12.83 |
| bf16 | interval2 | 0.452 / 13.04 | 0.477 / 12.99 |
| bf16 | seg2-stride4 | 0.349 / 16.88 | 0.400 / 16.83 |
| bf16 | seg2-stride4-offload | 0.681 / 15.03 | 0.736 / 14.99 |
| int8-sdnq-hadamard | none | 0.645 / 15.60 | 0.615 / 15.55 |
| int8-sdnq-hadamard | activation-offload | 1.439 / 7.61 | 1.382 / 7.56 |
| int8-sdnq-hadamard | layer | 1.046 / 7.25 | 1.021 / 7.20 |
| int8-sdnq-hadamard | interval2 | 1.074 / 7.41 | 0.996 / 7.36 |
| int8-sdnq-hadamard | seg2-stride4 | 0.867 / 11.25 | 0.841 / 11.20 |
| int8-sdnq-hadamard | seg2-stride4-offload | 1.202 / 9.39 | 1.162 / 9.35 |
| fp8-torchao | none | 1.232 / 37.50 | 1.476 / 37.46 |
| fp8-torchao | activation-offload | 3.623 / 7.97 | 3.564 / 7.93 |
| fp8-torchao | layer | 2.319 / 7.93 | 2.344 / 7.88 |
| fp8-torchao | interval2 | 2.336 / 8.80 | 2.309 / 8.75 |
| fp8-torchao | seg2-stride4 | 1.843 / 22.55 | 1.930 / 22.50 |
| fp8-torchao | seg2-stride4-offload | 2.947 / 15.57 | 3.243 / 15.52 |

### ZLab I1

Example: `zlab-i1.peft-lora`. Resolution: 1024x1024.

Nota: ZLab I1 carrega seus skip tensors estilo U-Net pelo estado segmented checkpoint. Attention activation offload nao esta conectado para esta familia.

| Precision | Mode | H100 | L40S |
| --- | --- | ---: | ---: |
| bf16 | none | 0.462 / 22.21 | 0.865 / 22.16 |
| bf16 | activation-offload | unsupported | unsupported |
| bf16 | layer | 0.693 / 7.79 | 1.148 / 7.75 |
| bf16 | interval2 | 0.676 / 8.30 | 1.152 / 8.25 |
| bf16 | seg2-stride4 | 0.567 / 14.97 | 1.014 / 14.92 |
| bf16 | seg2-stride4-offload | unsupported | unsupported |
| int8-sdnq-hadamard | none | 0.861 / 19.21 | 0.926 / 19.16 |
| int8-sdnq-hadamard | activation-offload | unsupported | unsupported |
| int8-sdnq-hadamard | layer | 1.385 / 4.78 | 1.265 / 4.74 |
| int8-sdnq-hadamard | interval2 | 1.298 / 5.30 | 1.277 / 5.26 |
| int8-sdnq-hadamard | seg2-stride4 | 1.073 / 11.98 | 1.098 / 11.93 |
| int8-sdnq-hadamard | seg2-stride4-offload | unsupported | unsupported |
| fp8wo-torchao | none | 0.504 / 25.08 | 0.930 / 25.02 |
| fp8wo-torchao | activation-offload | unsupported | unsupported |
| fp8wo-torchao | layer | 0.772 / 5.12 | 1.280 / 5.07 |
| fp8wo-torchao | interval2 | 0.759 / 5.84 | 1.290 / 5.79 |
| fp8wo-torchao | seg2-stride4 | 0.633 / 15.12 | 1.115 / 15.08 |
| fp8wo-torchao | seg2-stride4-offload | unsupported | unsupported |

<!-- full-sweep-matrix:end -->
