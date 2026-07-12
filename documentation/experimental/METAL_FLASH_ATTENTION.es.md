# Metal Flash Attention

`metal-flash-attention` envia llamadas SDPA MPS elegibles en Apple Silicon al backend PyTorch FFI de Universal Metal Flash Attention (UMFA). Es experimental y esta pensado para rutas FLUX FP32 donde PyTorch SDPA usa mas memoria o llega a limites de MPSGraph con secuencias largas.

## Requisitos

- Apple Silicon con MPS disponible.
- Xcode command line tools con toolchain Metal.
- SimpleTuner instalado con dependencias Apple. El extra Apple requiere PyTorch `>=2.13.0`.
- Un build UMFA que exponga `metal_flash_attention_autograd`. Builds antiguos solo-forward son rechazados por SimpleTuner.

SimpleTuner solo despacha a UMFA llamadas MPS FP32 4D con al menos cuatro heads y sequence length 64 o mayor. FP16/BF16, masked, causal, grouped-query, tiny, 2D y llamadas no-MPS vuelven a PyTorch SDPA.

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
  "attention_mechanism": "metal-flash-attention",
  "mixed_precision": "no",
  "base_model_default_dtype": "fp32"
}
```

`mixed_precision=no` y defaults FP32 son importantes para la integracion actual. SimpleTuner hace fallback en lugar de enviar atencion BF16/FP16 a UMFA.

Tambien hay aliases cuantizados:

- `metal-flash-attention-int8`
- `metal-flash-attention-int4`

Estos llaman `metal_quantized_flash_attention_autograd` de UMFA con cuantizacion blockwise (`quant_mode=2`). SimpleTuner ejecuta un check inicial adicional que verifica salidas conectadas a autograd y gradientes multi-head finitos antes de habilitar cualquiera de los aliases.

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
- Fallback inesperado: verifica tensores MPS FP32 4D BHSD, cuatro heads o mas, sequence length 64 o mas, `dropout_p=0`, `is_causal=False`, sin mask y sin GQA.
- 2048px falla antes de atencion: probablemente es presion de memoria de cache VAE, no memoria de UMFA. Activa `vae_enable_tiling=true` o genera/reusa latentes con un cache workflow de menor memoria.
