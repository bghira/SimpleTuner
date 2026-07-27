# Checkpointing estilo Unsloth

Resumen corto: úsalo cuando el job casi cabe, y prueba FFN-only primero si el modelo lo soporta.

El backend `unsloth` descarga activaciones guardadas a CPU. El backend `torch` las descarta y las recalcula en backward. Unsloth puede comprar los últimos GiB para subir batch, resolución o frames. No es velocidad gratis. Si el run ya cabe con `torch`, normalmente `torch` sigue siendo el mejor default.

## Controles

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch-ffn"
}
```

`gradient_checkpointing_backend` tiene cuatro valores útiles:

| Valor | Scope | Ruta | Cuándo usarlo |
| --- | --- | --- | --- |
| `torch` | bloque completo | recompute | Necesitas el mayor ahorro integrado antes de CPU offload. |
| `torch-ffn` | feed-forward | recompute | Quieres el win barato después de que Flash Attention ya cubrió attention memory. |
| `unsloth` | bloque completo | CPU offload | Torch layer checkpointing aún no cabe. |
| `unsloth-ffn` | feed-forward | CPU offload | FFN-only con torch casi cabe y CPU offload puede comprar el resto. |

En familias compatibles también puedes checkpointar menos bloques:

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch-ffn",
  "gradient_checkpointing_interval": 2
}
```

`gradient_checkpointing_interval: 2` checkpointa cada dos bloques compatibles. Valores más altos hacen menos checkpointing y dejan más activaciones en VRAM.

`torch-ffn` y `unsloth-ffn` soportan actualmente bloques estilo Flux.1 y MageFlow. Otras familias fallan claramente hasta que sus bloques expongan el mismo límite seguro.

## Qué intercambia

- `torch`: descarta activaciones intermedias y las recalcula en backward.
- `unsloth`: guarda parte de esos tensores en CPU y los copia de vuelta para backward.
- `*-ffn`: checkpointa solo el lado feed-forward en modelos con un límite FFN limpio.
- Flash Attention ya evita materializar la matriz grande de atención. Esa idea de "checkpointing gratis" aplica sobre todo a atención, no a todo el bloque transformer.
- CPU offload ayuda más cuando las activaciones son grandes y el pico no viene de parámetros u optimizer.

Requiere CUDA y suficiente RAM de CPU. El ancho de banda PCIe importa. Si las copias CPU-GPU quedan expuestas, el step se vuelve más lento.

## Nuestro sweep

Bloque transformer sintético, bf16, flash SDPA, pesos base congelados, batch 1. No son garantías de modelo; enseñan la forma del tradeoff.

### Latents de imagen empaquetados

Con empaquetado 2x2, `64x64`, `128x128` y `256x256` pasan a `1024`, `4096` y `16384` tokens.

| GPU | Tokens | Sin checkpoint | Torch FFN | Unsloth FFN | Torch layer | Unsloth layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 80GB | 1024 | 0.0166s / 4.43 GiB | 0.0191s / 4.08 GiB | 0.0233s / 4.00 GiB | 0.0231s / 3.64 GiB | 0.0265s / 3.56 GiB |
| H100 80GB | 4096 | 0.0948s / 7.43 GiB | 0.1029s / 6.02 GiB | 0.1157s / 5.67 GiB | 0.1233s / 4.26 GiB | 0.1358s / 3.93 GiB |
| H100 80GB | 16384 | 0.8781s / 19.39 GiB | 0.9117s / 13.77 GiB | 0.9632s / 12.36 GiB | 1.1157s / 6.72 GiB | 1.1662s / 5.41 GiB |
| L40S | 1024 | 0.0500s / 4.39 GiB | 0.0575s / 4.04 GiB | 0.0627s / 3.95 GiB | 0.0666s / 3.60 GiB | 0.0725s / 3.51 GiB |
| L40S | 4096 | 0.2461s / 7.38 GiB | 0.2729s / 5.97 GiB | 0.2933s / 5.62 GiB | 0.3169s / 4.21 GiB | 0.3369s / 3.88 GiB |
| L40S | 16384 | 1.8153s / 19.35 GiB | 1.9639s / 13.72 GiB | 2.0250s / 12.31 GiB | 2.3360s / 6.67 GiB | 2.4218s / 5.36 GiB |

En `1024` tokens, el offload extra casi no importa salvo que ya estés al límite. En `16384` tokens, `torch-ffn` es el paso barato y whole-layer checkpointing es la palanca grande para caber. `unsloth` compra alrededor de `1.3 GiB` más que torch layer checkpointing.

### Transformer más grande

`32` capas congeladas, ancho `4096`, `3072` tokens:

| GPU | Sin checkpoint | Torch FFN | Unsloth FFN | Torch layer | Unsloth layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| H100 80GB | 0.1943s / 14.56 GiB | 0.2138s / 11.65 GiB | 0.2317s / 10.92 GiB | 0.2527s / 8.01 GiB | 0.2722s / 7.30 GiB |
| L40S | 0.5045s / 14.51 GiB | 0.5640s / 11.60 GiB | 0.5932s / 10.88 GiB | 0.6491s / 7.96 GiB | 0.6864s / 7.26 GiB |

Con pesos completos entrenables, gradientes y optimizer dominaron el pico, así que `unsloth` no ahorró más que `torch` en ese run sintético. PEFT se parece más al caso de pesos congelados.

## Regla práctica

1. Si cabe sin checkpointing, déjalo apagado.
2. Si no cabe, prueba `gradient_checkpointing_backend: torch-ffn`.
3. Si sigue muy justo, prueba `torch`.
4. Si torch layer checkpointing aún no cabe, prueba `unsloth-ffn` y luego `unsloth`.
5. Si el modelo soporta `gradient_checkpointing_interval`, usa `2` o más solo después de que el run ya quepa y quieras recuperar velocidad.

Vale la pena cuando permite usar el batch, resolución, frames o rank que querías. No vale mucho para tokens pequeños ni cuando el pico viene de pesos entrenables, gradientes, optimizer, caché VAE o validación.

## Notas

- Con FSDP activation checkpointing, SimpleTuner desactiva el checkpointing de modelo para evitar conflictos.
- `torch-ffn` y `unsloth-ffn` requieren soporte del modelo. SimpleTuner falla de forma explícita en vez de ejecutar otro scope en silencio.
- `gradient_checkpointing_interval: 1` equivale al checkpointing normal de cada bloque.
- Algunas familias no tienen interval checkpointing. SimpleTuner avisa e ignora el intervalo.
- `torch.compile` no rescató el camino de offload en nuestro sweep.
