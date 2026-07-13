# Metal Flash Attention

`metal-flash-attention` envia llamadas SDPA MPS elegibles en Apple Silicon al backend PyTorch FFI de Universal Metal Flash Attention (UMFA). Es experimental y esta pensado para rutas FLUX FP32/FP16/BF16 donde PyTorch SDPA usa mas memoria o llega a limites de MPSGraph con secuencias largas.

## Requisitos

- Apple Silicon con MPS disponible.
- Xcode command line tools con toolchain Metal.
- SimpleTuner instalado con dependencias Apple. El extra Apple requiere PyTorch `>=2.13.0`.
- Un build UMFA que exponga `metal_flash_attention_autograd`, registre la dispatch key PyTorch `MPS` y exponga `clear_quantization_mode`. Los aliases cuantizados tambien requieren `metal_quantized_flash_attention_autograd`, `set_quantization_mode`, `QUANT_INT8`, `QUANT_INT4` y `QUANT_BLOCK_WISE`.

SimpleTuner enruta la atencion por el dispatcher MPS SDPA de PyTorch, que los builds UMFA actuales registran. Son elegibles las llamadas MPS FP32/FP16/BF16 4D con cualquier cantidad de heads (single-head funciona) y cualquier sequence length, incluidos layouts transposed estilo FLUX, masks bool/additive de hasta 4D y llamadas causales. Las llamadas elegibles se codifican directamente en el command stream MPS de PyTorch — sin sincronizacion por llamada y sin promocion a FP32 para entradas FP16/BF16. El entrenamiento causal tambien es elegible — el backward causal pasa paridad exacta de gradientes. Las llamadas con dropout o `enable_gqa` hacen fallback a PyTorch SDPA. Builds UMFA antiguos registrados en `PrivateUse1` en vez de `MPS` usan silenciosamente PyTorch SDPA nativo.

## Compilar E Instalar UMFA

Usa el mismo entorno Python que ejecuta SimpleTuner:

```bash
export ST_ROOT=/path/to/SimpleTuner
export UMFA_ROOT=/path/to/universal-metal-flash-attention
export PYTHON="$ST_ROOT/.venv/bin/python"
```

Compila el paquete Swift e instala el paquete PyTorch FFI:

```bash
cd "$UMFA_ROOT"
git submodule update --init --recursive
swift build -c release

cd "$UMFA_ROOT/examples/pytorch-custom-op-ffi"
"$PYTHON" -m pip install --upgrade pip setuptools wheel pybind11 numpy
"$PYTHON" -m pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir .
```

Comprueba que la extension exporta el binding autograd:

```bash
"$PYTHON" - <<'PY'
import metal_sdpa_extension

print("metal_flash_attention_autograd" in dir(metal_sdpa_extension))
print([name for name in dir(metal_sdpa_extension) if "attention" in name])
PY
```

Comprueba que SimpleTuner acepta la extension:

```bash
cd "$ST_ROOT"
"$PYTHON" - <<'PY'
from simpletuner.helpers.training.attention_backend import (
    get_metal_flash_attention_unavailable_reason,
    is_metal_flash_attention_available,
)

print("available", is_metal_flash_attention_available())
print("reason", get_metal_flash_attention_unavailable_reason())
PY
```

Resultado esperado:

```text
available True
reason None
```

Si el resultado dice que la salida UMFA esta detached, recompila UMFA desde una version que implemente `metal_flash_attention_autograd`.

## Activar En SimpleTuner

Configura el mecanismo de atencion:

```json
{
  "attention_mechanism": "metal-flash-attention"
}
```

La atencion FP32, FP16 y BF16 corre de forma nativa: las entradas FP16/BF16 usan kernels de baja precision (BF16 mantiene acumulacion softmax en FP32) y la salida se produce en el dtype de entrada, asi que `mixed_precision: bf16` funciona sin forzar FP32 en ningun lado. SimpleTuner todavia hace fallback con dropout y `enable_gqa`.

Tambien hay aliases cuantizados:

- `metal-flash-attention-int8`
- `metal-flash-attention-int4`

Estos configuran el modo global de cuantizacion de UMFA con cuantizacion blockwise (`quant_mode=2`) y usan la entrada autograd cuantizada para llamadas despachadas directamente:

- `metal-flash-attention-int8`: `set_quantization_mode(ext.QUANT_INT8, ext.QUANT_BLOCK_WISE)`
- `metal-flash-attention-int4`: `set_quantization_mode(ext.QUANT_INT4, ext.QUANT_BLOCK_WISE)`

SimpleTuner limpia ese modo al volver a UMFA FP32 u otro backend de atencion. Tambien ejecuta un check inicial adicional que verifica salidas conectadas a autograd, gradientes multi-head finitos, SDPA masked en el dispatcher y ausencia de fallback PyTorch antes de habilitar cualquiera de los aliases.

Tanto el dispatcher regular como el quantized soportan bool masks (`True` significa atender), additive float masks, batched masks como `[B, H, S_q, S_kv]` y broadcast masks como `[B, 1, 1, S_kv]`. Las bool masks all-true se detectan y se saltan como fast path. Las llamadas con mask siguen en la ruta in-stream; la expansion del mask se codifica en el mismo command buffer que el kernel de atencion.

Para verificar que el dispatcher MPS usa la ruta esperada durante una ejecucion, inspecciona los contadores de dispatch de UMFA:

```python
import metal_sdpa_extension as ext

print(ext.get_dispatch_stats())
```

Para inferencia/validacion sin cuantizar, `fp32_instream` debe aumentar mientras `pytorch_fallback` queda en `0` (cuenta la atencion con cualquier dtype de entrada — el nombre se refiere a la ruta de dispatch, no al dtype de computo). Para entrenamiento quantized de Z-Image, `quantized_autograd` debe aumentar en su lugar. Si el `encoder_attention_mask` es all-true, `mask_all_true_skipped` tambien aumenta. Las llamadas por el entry point de RoPE fusionado cuentan en `rope_instream`.

## RoPE + SDPA fusionado

La extension expone `metal_sdpa_extension.rope_scaled_dot_product_attention(query, key, value, rope_cos, rope_sin, attn_mask=None, is_causal=False, scale=None)`, que aplica rotary embeddings interleaved-pair a Q/K en la GPU justo antes de la atencion — un solo submit de command buffer, sin pasadas eager de rotacion, sin materializaciones FP32. Cubre la convencion RoPE compartida por FLUX.1, FLUX.2, Krea2 y Z-Image (la formulacion complex-multiply de Z-Image es la misma rotacion); los modelos solo difieren en el formato de tabla, que adapta el caller.

- Los tensores son BHSD; las vistas strided (p. ej. `transpose(1, 2)` de una proyeccion BSHD, o vistas `unbind` de QKV fusionado) se consumen sin copias.
- `rope_cos`/`rope_sin` son tablas pair-duplicated (`cos[2i] == cos[2i+1]`), con forma `[S, D]`, `[1, S, D]` o por muestra `[B, S, D]`; cualquier dtype float se normaliza a FP32 internamente.
- El entrenamiento fluye por la ruta fusionada: un autograd custom aplica la rotacion inversa (la misma rotacion pairwise con sin negado — RoPE es ortonormal) a dQ/dK en backward, devolviendo gradientes respecto a Q/K pre-RoPE. Verificado exacto contra una referencia diferenciable en FP32, incluidas tablas batched por muestra. Causal tambien esta soportado con gradientes; las llamadas con mask o GQA que requieren gradientes siguen usando rotacion eager.
- Las entradas GQA (menos heads de K/V) se rotan con la cantidad de heads de K/V y se expanden despues.

En la forma DiT de Z-Image `(1, 30, 4128, 128)` en BF16, la ruta fusionada midio 4.4 ms/layer mas rapida que rotacion eager + SDPA en un benchmark de cadena de 12 layers. La integracion por modelo en SimpleTuner esta pendiente; hasta entonces el entry point esta disponible para uso directo.

## Sequence Lengths De FLUX

Para la ruta LoRA FLUX.1 cuadrada probada, la forma de atencion es `B,H,S,D = 1,24,S,128`. El sequence length escala con el area de imagen:

| Resolucion | `S` esperado | Forma de atencion |
| --- | ---: | --- |
| 512 | 1536 | `(1, 24, 1536, 128)` |
| 1024 | 6144 | `(1, 24, 6144, 128)` |
| 2048 | 24576 | `(1, 24, 24576, 128)` |

Por tanto, `S=24576` es la forma sintetica objetivo para una muestra FLUX.1 cuadrada de 2048px en esta configuracion.

## Prueba Sintetica De Memoria

Numeros de probes aislados FP32 forward+backward en MPS con `B=1,H=24,D=128`. Miden picos de `torch.mps.driver_allocated_memory()` y no incluyen pesos del modelo, optimizador, cache VAE, carga de datos ni checkpoints.

| Sequence Length | Pico PyTorch SDPA | Pico UMFA | Resultado |
| ---: | ---: | ---: | --- |
| 1536 | 1.010 GiB | 1.009 GiB | ambos pasan |
| 6144 | 11.983 GiB | 1.009 GiB | ambos pasan |
| 8192 | 20.516 GiB | 1.009 GiB | ambos pasan |
| 10240 | falla | 1.009 GiB | SDPA llega a un limite de MPSGraph |
| 24576 | no probado tras la falla SDPA | 3.008 GiB | UMFA pasa |
| 65536 | no probado tras la falla SDPA | 6.040 GiB | UMFA pasa |

Falla SDPA en `S=10240`:

```text
RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX
```

Paridad directa en la forma de 1024px (`S=6144`):

```text
forward max_abs=6.11e-07
loss_sdpa=0.0004459388
loss_metal=0.0004459389
q/k/v gradient mean_abs <= 4.1e-16
```

## Probe FLUX De Un Paso

Probe real con FLUX.1 LoRA FP32 de un paso, `train_batch_size=1`, sin cuantizacion, sin validacion y dataset Domokun pequeno. Estos numeros son para memoria y forma, no calidad.

| Run | Resultado | Tiempo de paso | Pico RSS del arbol | Notas |
| --- | --- | ---: | ---: | --- |
| 1024px PyTorch SDPA | pasa | 40.66s | 46.991 GiB | baseline |
| 1024px UMFA | pasa | 47.64s | 49.758 GiB | autograd activo; paridad directa `S=6144` pasa |
| 2048px UMFA, sin VAE tiling | falla antes del paso | n/a | n/a | falla creando latentes VAE 2048px |
| 2048px PyTorch SDPA, sin VAE tiling | falla antes del paso | n/a | n/a | misma falla de cache VAE antes de atencion |
| 2048px UMFA, VAE tiling activado | pasa | 512.66s | 46.747 GiB | 27 latentes VAE cacheados; `Metal SDPA backend initialized successfully`; `step_loss=0.256` |
| 2048px PyTorch SDPA, VAE tiling activado | crash en el paso | n/a | 46.292 GiB | proceso salio con `rc=-11` despues de entrar al primer paso; sin traceback Python |

Las primeras ejecuciones 2048px no activaron `vae_enable_tiling` y no llegaron a la atencion del transformer:

```text
MPS backend out of memory (MPS allocated: 20.48 GiB, other allocations: 146.27 GiB, max allowed: 163.20 GiB).
```

Con `vae_enable_tiling=true`, la cache VAE 2048px completo y UMFA completo el paso de training. La ejecucion PyTorch SDPA equivalente reutilizo la cache VAE tiled ya caliente, entro al primer paso de training y crasheo sin traceback Python. UMFA reduce presion de memoria de atencion, pero no resuelve todos los cuellos de botella FLUX de alta resolucion; la generacion de cache VAE debe configurarse aparte.

## Troubleshooting

- Falta `metal_flash_attention_autograd`: recompila UMFA con soporte autograd y reinstala el paquete FFI.
- `available False`: lee `get_metal_flash_attention_unavailable_reason()`.
- Fallback inesperado: verifica tensores MPS FP32/FP16/BF16 4D BHSD (cualquier cantidad de heads y sequence length; single-head esta soportado), `dropout_p=0` y que `enable_gqa` no este activo. Las masks bool/additive de hasta 4D son elegibles. Las llamadas causales son elegibles con y sin gradientes. Confirma que el build UMFA registra la dispatch key PyTorch `MPS` y expone `get_dispatch_stats()`; builds registrados solo en `PrivateUse1` son bypassed por tensores `torch.device("mps")`.
- 2048px falla antes de atencion: probablemente es presion de memoria de cache VAE, no memoria de UMFA. Activa `vae_enable_tiling=true` o genera/reusa latentes con un cache workflow de menor memoria.
