# Metal Flash Attention

`metal-flash-attention` envia chamadas SDPA MPS elegiveis em Apple Silicon para a extensao PyTorch FFI do Universal Metal Flash Attention (UMFA). E experimental e atualmente voltado para caminhos FLUX FP32 onde PyTorch SDPA usa mais memoria ou atinge limites do MPSGraph em sequence lengths longos.

## Requisitos

- Apple Silicon com MPS disponivel.
- Xcode command line tools com toolchain Metal.
- SimpleTuner instalado com dependencias Apple. O extra Apple requer PyTorch `>=2.13.0`.
- Um build UMFA que exponha `metal_flash_attention_autograd`, registre a dispatch key PyTorch `MPS` e exponha `clear_quantization_mode`. Os aliases quantizados tambem exigem `metal_quantized_flash_attention_autograd`, `set_quantization_mode`, `QUANT_INT8`, `QUANT_INT4` e `QUANT_BLOCK_WISE`.

SimpleTuner despacha diretamente para UMFA chamadas MPS FP32 4D sem mask, com pelo menos quatro heads e sequence length 64 ou maior. Outras chamadas passam pelo PyTorch SDPA. Com um build UMFA atual, o dispatcher MPS SDPA do PyTorch e registrado pelo UMFA, entao chamadas quantizadas com mask ainda podem chegar ao UMFA pelo dispatcher; builds antigos registrados em `PrivateUse1` em vez de `MPS` sao ignorados silenciosamente por tensores `torch.device("mps")`.

## Build E Instalacao Do UMFA

Use o mesmo Python environment que executa SimpleTuner:

```bash
export ST_ROOT=/path/to/SimpleTuner
export UMFA_ROOT=/path/to/universal-metal-flash-attention
export PYTHON="$ST_ROOT/.venv/bin/python"
```

Compile o pacote Swift e instale o pacote PyTorch FFI:

```bash
cd "$UMFA_ROOT"
git submodule update --init --recursive
swift build -c release

cd "$UMFA_ROOT/examples/pytorch-custom-op-ffi"
"$PYTHON" -m pip install --upgrade pip setuptools wheel pybind11 numpy
"$PYTHON" -m pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir .
```

Verifique se a extensao exporta o binding autograd:

```bash
"$PYTHON" - <<'PY'
import metal_sdpa_extension

print("metal_flash_attention_autograd" in dir(metal_sdpa_extension))
print([name for name in dir(metal_sdpa_extension) if "attention" in name])
PY
```

Verifique se SimpleTuner aceita a extensao:

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

Se a saida disser que o output UMFA esta detached, recompile UMFA a partir de uma versao que implemente `metal_flash_attention_autograd`.

## Ativar No SimpleTuner

Configure o mecanismo de atencao:

```json
{
  "attention_mechanism": "metal-flash-attention",
  "mixed_precision": "no",
  "base_model_default_dtype": "fp32"
}
```

`mixed_precision=no` e defaults FP32 sao importantes para a integracao atual. SimpleTuner faz fallback em vez de enviar atencao BF16/FP16 para UMFA.

Aliases quantizados tambem estao disponiveis:

- `metal-flash-attention-int8`
- `metal-flash-attention-int4`

Eles configuram o modo global de quantizacao do UMFA com quantizacao blockwise (`quant_mode=2`) e usam a entrada autograd quantizada para chamadas despachadas diretamente:

- `metal-flash-attention-int8`: `set_quantization_mode(ext.QUANT_INT8, ext.QUANT_BLOCK_WISE)`
- `metal-flash-attention-int4`: `set_quantization_mode(ext.QUANT_INT4, ext.QUANT_BLOCK_WISE)`

SimpleTuner limpa esse modo ao voltar para UMFA FP32 ou outro backend de atencao. Ele tambem executa um check inicial adicional que verifica saidas ligadas ao autograd, gradientes multi-head finitos, SDPA masked no dispatcher e ausencia de fallback PyTorch antes de habilitar qualquer alias.

O dispatcher quantizado suporta bool masks (`True` significa attend), additive float masks, masks batched como `[B, H, S_q, S_kv]` e masks broadcast como `[B, 1, 1, S_kv]`. Bool masks all-true sao detectadas e ignoradas como fast path.

Para verificar se o dispatcher MPS esta usando o caminho esperado durante uma execucao, inspecione os contadores de dispatch do UMFA:

```python
import metal_sdpa_extension as ext

print(ext.get_dispatch_stats())
```

Em training Z-Image, `quantized_autograd` deve aumentar enquanto `pytorch_fallback` fica em `0`. Se `encoder_attention_mask` for all-true, `mask_all_true_skipped` tambem deve aumentar.

## Sequence Lengths Do FLUX

No caminho FLUX.1 LoRA quadrado testado, a forma de atencao e `B,H,S,D = 1,24,S,128`. O sequence length escala com a area da imagem:

| Resolucao | `S` esperado | Forma de atencao |
| --- | ---: | --- |
| 512 | 1536 | `(1, 24, 1536, 128)` |
| 1024 | 6144 | `(1, 24, 6144, 128)` |
| 2048 | 24576 | `(1, 24, 24576, 128)` |

Assim, `S=24576` e a forma sintetica alvo para uma amostra FLUX.1 quadrada de 2048px nesta configuracao.

## Probe Sintetico De Memoria

Numeros de probes isolados FP32 forward+backward em MPS com `B=1,H=24,D=128`. Eles medem picos de `torch.mps.driver_allocated_memory()` e nao incluem pesos do modelo, optimizer, cache VAE, data loading ou checkpointing.

| Sequence Length | Pico PyTorch SDPA | Pico UMFA | Resultado |
| ---: | ---: | ---: | --- |
| 1536 | 1.010 GiB | 1.009 GiB | ambos passam |
| 6144 | 11.983 GiB | 1.009 GiB | ambos passam |
| 8192 | 20.516 GiB | 1.009 GiB | ambos passam |
| 10240 | falha | 1.009 GiB | SDPA atinge limite do MPSGraph |
| 24576 | nao testado apos falha SDPA | 3.008 GiB | UMFA passa |
| 65536 | nao testado apos falha SDPA | 6.040 GiB | UMFA passa |

Falha SDPA em `S=10240`:

```text
RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX
```

Paridade direta na forma de 1024px (`S=6144`):

```text
forward max_abs=6.11e-07
loss_sdpa=0.0004459388
loss_metal=0.0004459389
q/k/v gradient mean_abs <= 4.1e-16
```

## Probe FLUX De Um Passo

O probe real usou um config FLUX.1 LoRA FP32 de um passo com `train_batch_size=1`, sem quantizacao, sem validacao e o dataset pequeno Domokun. Estes numeros sao para memoria e forma, nao qualidade.

| Run | Resultado | Tempo do passo | Pico RSS da arvore | Notas |
| --- | --- | ---: | ---: | --- |
| 1024px PyTorch SDPA | passa | 40.66s | 46.991 GiB | baseline |
| 1024px UMFA | passa | 47.64s | 49.758 GiB | autograd ativo; paridade direta `S=6144` passa |
| 2048px UMFA, sem VAE tiling | falha antes do passo | n/a | n/a | falha criando latentes VAE 2048px |
| 2048px PyTorch SDPA, sem VAE tiling | falha antes do passo | n/a | n/a | mesma falha de cache VAE antes da atencao |
| 2048px UMFA, VAE tiling ativado | passa | 512.66s | 46.747 GiB | 27 latentes VAE em cache; `Metal SDPA backend initialized successfully`; `step_loss=0.256` |
| 2048px PyTorch SDPA, VAE tiling ativado | crash no passo | n/a | 46.292 GiB | processo saiu com `rc=-11` depois de entrar no primeiro passo; sem traceback Python |

As primeiras execucoes 2048px nao ativaram `vae_enable_tiling` e nao chegaram a atencao do transformer:

```text
MPS backend out of memory (MPS allocated: 20.48 GiB, other allocations: 146.27 GiB, max allowed: 163.20 GiB).
```

Com `vae_enable_tiling=true`, a cache VAE 2048px completou e a execucao UMFA completou o passo de training. A execucao PyTorch SDPA equivalente reutilizou a cache VAE tiled ja quente, entrou no primeiro passo de training e crasheou sem traceback Python. UMFA reduz a pressao de memoria da atencao, mas nao resolve todos os gargalos FLUX de alta resolucao; a geracao da cache VAE ainda precisa ser configurada separadamente.

## Troubleshooting

- `metal_flash_attention_autograd` ausente: recompile UMFA com suporte autograd e reinstale o pacote FFI.
- `available False`: leia `get_metal_flash_attention_unavailable_reason()`.
- Fallback inesperado: verifique tensores MPS FP32 4D BHSD, heads >= 4, sequence length >= 64, `dropout_p=0`, `is_causal=False` e sem GQA. Para caminhos quantizados com mask, confirme que o build UMFA registra a dispatch key PyTorch `MPS` e expoe `get_dispatch_stats()`; builds registrados apenas em `PrivateUse1` sao bypassados por tensores `torch.device("mps")`.
- 2048px falha antes da atencao: provavelmente e pressao de memoria da cache VAE, nao memoria de atencao UMFA. Ative `vae_enable_tiling=true` ou gere/reuse latentes com um cache workflow de menor memoria.
