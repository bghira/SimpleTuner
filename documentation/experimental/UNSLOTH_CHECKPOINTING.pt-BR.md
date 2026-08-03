# Checkpointing estilo Unsloth

Resumo curto: use quando o job quase cabe, e tente FFN-only primeiro quando o modelo suportar.

O backend `unsloth` descarrega activations salvas para a CPU. O backend `torch` descarta essas activations e recalcula no backward. Unsloth pode comprar os últimos GiB para aumentar batch, resolução ou frames. Não é velocidade grátis. Se o run já cabe com `torch`, normalmente `torch` continua sendo o melhor default.

## Controles

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch-ffn"
}
```

`gradient_checkpointing_backend` tem quatro valores úteis:

| Valor | Escopo | Caminho | Quando usar |
| --- | --- | --- | --- |
| `torch` | bloco inteiro | recompute | Você precisa do maior corte de memória antes de CPU offload. |
| `torch-ffn` | feed-forward | recompute | Você quer o ganho barato depois que Flash Attention já cuidou da atenção. |
| `unsloth` | bloco inteiro | CPU offload | Torch layer checkpointing ainda não cabe. |
| `unsloth-ffn` | feed-forward | CPU offload | FFN-only com torch quase cabe e CPU offload pode comprar o resto. |

Em famílias compatíveis, você também pode checkpointar menos blocos:

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch-ffn",
  "gradient_checkpointing_interval": 2
}
```

`gradient_checkpointing_interval: 2` faz checkpoint de chunks contiguos de dois blocos em caminhos whole-block compatíveis. Valores maiores recomputam menos e mantêm mais activations na VRAM.

Nesses caminhos segmentados, `gradient_checkpointing_segment_stride` também funciona com `unsloth`. Trate como alavanca para caber, não para acelerar: os blocos pulados ficam na GPU, enquanto os blocos checkpointed ainda usam CPU offload para tensores salvos. Para o resumo torch-only e benchmarks por modelo, veja [Segmented Checkpointing](SEGMENTED_CHECKPOINTING.md).

`gradient_checkpointing_offload_attention` e independente do backend. Em blocos compatíveis com separacao attention/FFN, faz offload das activations salvas do lado attention. Pode rodar sozinho ou ser combinado com `torch`, `torch-ffn`, `unsloth` ou `unsloth-ffn` quando o modelo suportar esse backend.

`gradient_checkpointing_offload_pin_memory_max_buckets` controla o pooling de CPU pinned para tensores salvos offloaded. O padrao e `12` buckets de tensor distintos; use `0` para usar apenas memoria CPU normal.

`torch-ffn` e `unsloth-ffn` atualmente suportam Chroma, Flux, Krea 2, LTXVideo2, MageFlow, Wan e Z-Image. Outras famílias falham claramente até seus blocos exporem a mesma fronteira segura.

## O tradeoff

- `torch`: descarta activations intermediárias e recalcula no backward.
- `unsloth`: salva parte desses tensores na CPU e copia de volta para o backward.
- `*-ffn`: checkpointa só o lado feed-forward em modelos com uma fronteira FFN limpa.
- Flash Attention já evita materializar a matriz grande de atenção. Esse "checkpointing grátis" é principalmente da atenção, não do bloco transformer inteiro.
- CPU offload ajuda mais quando activations são grandes e o pico não vem de parâmetros ou optimizer.

Requer CUDA e RAM de CPU suficiente. Banda PCIe importa. Se as cópias CPU-GPU ficarem expostas, o step fica mais lento.

## Nosso sweep

Bloco transformer sintético, bf16, flash SDPA, pesos base congelados, batch 1. Não são garantias de modelo; mostram o formato do tradeoff.

### Latents de imagem empacotados

Com packing 2x2, `64x64`, `128x128` e `256x256` viram `1024`, `4096` e `16384` tokens.

| GPU | Tokens | Sem checkpoint | Torch FFN | Unsloth FFN | Torch layer | Unsloth layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 80GB | 1024 | 0.0166s / 4.43 GiB | 0.0191s / 4.08 GiB | 0.0233s / 4.00 GiB | 0.0231s / 3.64 GiB | 0.0265s / 3.56 GiB |
| H100 80GB | 4096 | 0.0948s / 7.43 GiB | 0.1029s / 6.02 GiB | 0.1157s / 5.67 GiB | 0.1233s / 4.26 GiB | 0.1358s / 3.93 GiB |
| H100 80GB | 16384 | 0.8781s / 19.39 GiB | 0.9117s / 13.77 GiB | 0.9632s / 12.36 GiB | 1.1157s / 6.72 GiB | 1.1662s / 5.41 GiB |
| L40S | 1024 | 0.0500s / 4.39 GiB | 0.0575s / 4.04 GiB | 0.0627s / 3.95 GiB | 0.0666s / 3.60 GiB | 0.0725s / 3.51 GiB |
| L40S | 4096 | 0.2461s / 7.38 GiB | 0.2729s / 5.97 GiB | 0.2933s / 5.62 GiB | 0.3169s / 4.21 GiB | 0.3369s / 3.88 GiB |
| L40S | 16384 | 1.8153s / 19.35 GiB | 1.9639s / 13.72 GiB | 2.0250s / 12.31 GiB | 2.3360s / 6.67 GiB | 2.4218s / 5.36 GiB |

Em `1024` tokens, o offload extra é ruído a menos que você já esteja sem VRAM. Em `16384` tokens, `torch-ffn` é o passo barato e whole-layer checkpointing é a grande alavanca para caber. `unsloth` compra cerca de `1.3 GiB` além do torch layer checkpointing.

### Transformer maior

`32` camadas congeladas, largura `4096`, `3072` tokens:

| GPU | Sem checkpoint | Torch FFN | Unsloth FFN | Torch layer | Unsloth layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| H100 80GB | 0.1943s / 14.56 GiB | 0.2138s / 11.65 GiB | 0.2317s / 10.92 GiB | 0.2527s / 8.01 GiB | 0.2722s / 7.30 GiB |
| L40S | 0.5045s / 14.51 GiB | 0.5640s / 11.60 GiB | 0.5932s / 10.88 GiB | 0.6491s / 7.96 GiB | 0.6864s / 7.26 GiB |

Com pesos completos treináveis, gradientes e optimizer dominaram o pico, então `unsloth` não economizou mais que `torch` naquele run sintético. PEFT fica mais perto do caso com pesos congelados.

## Regra prática

1. Se cabe sem checkpointing, deixe desligado.
2. Se não cabe, tente `gradient_checkpointing_backend: torch-ffn`.
3. Se ainda estiver apertado, tente `torch`.
4. Se torch layer checkpointing ainda não couber, tente `unsloth-ffn` e depois `unsloth`.
5. Se o modelo suporta `gradient_checkpointing_interval`, use `2` ou mais só depois que o run já couber e você quiser recuperar velocidade.

Vale a pena quando permite usar o batch, resolução, frames ou rank que você queria. Não vale muito para poucos tokens ou quando o pico vem de pesos treináveis, gradientes, optimizer, cache VAE ou validação.

## Notas

- Com FSDP activation checkpointing, SimpleTuner desativa o checkpointing de modelo para evitar conflito.
- `torch-ffn` e `unsloth-ffn` exigem suporte do modelo. SimpleTuner falha explicitamente em vez de rodar outro escopo em silêncio.
- `gradient_checkpointing_interval: 1` equivale ao checkpointing normal de cada bloco.
- Algumas famílias não têm interval checkpointing. SimpleTuner avisa e ignora o intervalo.
- `torch.compile` não salvou o caminho de offload no nosso sweep.
