# Metal Flash Attention

`metal-flash-attention` envia chamadas SDPA MPS elegiveis em Apple Silicon para a extensao PyTorch FFI do Universal Metal Flash Attention (UMFA). E experimental e atualmente voltado para caminhos FLUX FP32/FP16/BF16 onde PyTorch SDPA usa mais memoria ou atinge limites do MPSGraph em sequence lengths longos.

## Requisitos

- Apple Silicon com MPS disponivel.
- Xcode command line tools com toolchain Metal.
- SimpleTuner instalado com dependencias Apple. O extra Apple requer PyTorch `>=2.13.0`.
- Um build UMFA que exponha `metal_flash_attention_autograd`, registre a dispatch key PyTorch `MPS` e exponha `clear_quantization_mode`. Os aliases quantizados tambem exigem `metal_quantized_flash_attention_autograd`, `set_quantization_mode`, `QUANT_INT8`, `QUANT_INT4` e `QUANT_BLOCK_WISE`.

O SimpleTuner roteia a atencao pelo dispatcher MPS SDPA do PyTorch, que os builds UMFA atuais registram. Sao elegiveis chamadas MPS FP32/FP16/BF16 4D com qualquer quantidade de heads (single-head funciona) e qualquer sequence length, incluindo layouts transposed estilo FLUX, masks bool/additive de ate 4D e chamadas causais. As chamadas elegiveis sao codificadas diretamente no command stream MPS do PyTorch — sem sincronizacao por chamada e sem promocao a FP32 para entradas FP16/BF16. O treinamento causal tambem e elegivel — o backward causal passa paridade exata de gradientes. Chamadas com dropout ou `enable_gqa` fazem fallback para PyTorch SDPA. Builds UMFA antigos registrados em `PrivateUse1` em vez de `MPS` usam silenciosamente o PyTorch SDPA nativo.

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
  "attention_mechanism": "metal-flash-attention"
}
```

A atencao FP32, FP16 e BF16 roda de forma nativa: entradas FP16/BF16 usam kernels de baixa precisao (BF16 mantem acumulacao do softmax em FP32) e a saida e produzida no dtype de entrada, entao `mixed_precision: bf16` funciona sem forcar FP32 em lugar nenhum. O SimpleTuner ainda faz fallback com dropout e `enable_gqa`.

Aliases quantizados tambem estao disponiveis:

- `metal-flash-attention-int8`
- `metal-flash-attention-int4`

Eles configuram o modo global de quantizacao do UMFA com quantizacao blockwise (`quant_mode=2`) e usam a entrada autograd quantizada para chamadas despachadas diretamente:

- `metal-flash-attention-int8`: `set_quantization_mode(ext.QUANT_INT8, ext.QUANT_BLOCK_WISE)`
- `metal-flash-attention-int4`: `set_quantization_mode(ext.QUANT_INT4, ext.QUANT_BLOCK_WISE)`

SimpleTuner limpa esse modo ao voltar para UMFA FP32 ou outro backend de atencao. Ele tambem executa um check inicial adicional que verifica saidas ligadas ao autograd, gradientes multi-head finitos, SDPA masked no dispatcher e ausencia de fallback PyTorch antes de habilitar qualquer alias.

Tanto o dispatcher regular quanto o quantized suportam bool masks (`True` significa atender), additive float masks, batched masks como `[B, H, S_q, S_kv]` e broadcast masks como `[B, 1, 1, S_kv]`. Bool masks all-true sao detectadas e puladas como fast path. Chamadas com mask permanecem na rota in-stream; a expansao da mask e codificada no mesmo command buffer do kernel de atencao.

Para verificar se o dispatcher MPS esta usando o caminho esperado durante uma execucao, inspecione os contadores de dispatch do UMFA:

```python
import metal_sdpa_extension as ext

print(ext.get_dispatch_stats())
```

Para inferencia/validacao sem quantizacao, `fp32_instream` deve aumentar enquanto `pytorch_fallback` fica em `0` (atencao com qualquer dtype de entrada conta aqui — o nome se refere a rota de dispatch, nao ao dtype de computo). Para treinamento quantized do Z-Image, `quantized_autograd` deve aumentar no lugar. Se o `encoder_attention_mask` for all-true, `mask_all_true_skipped` tambem aumenta. Chamadas pelo entry point de RoPE fundido contam em `rope_instream`.

## RoPE + SDPA fundido

A extensao expoe `metal_sdpa_extension.rope_scaled_dot_product_attention(query, key, value, rope_cos, rope_sin, attn_mask=None, is_causal=False, scale=None)`, que aplica rotary embeddings interleaved-pair a Q/K na GPU imediatamente antes da atencao — um unico submit de command buffer, sem passadas eager de rotacao, sem materializacoes FP32. Cobre a convencao RoPE compartilhada por FLUX.1, FLUX.2, Krea2 e Z-Image (a formulacao complex-multiply do Z-Image e a mesma rotacao); os modelos diferem apenas no formato da tabela, que o caller adapta.

- Os tensores sao BHSD; views strided (por exemplo `transpose(1, 2)` de uma projecao BSHD, ou views `unbind` de QKV fundido) sao consumidas sem copias.
- `rope_cos`/`rope_sin` sao tabelas pair-duplicated (`cos[2i] == cos[2i+1]`), com formato `[S, D]`, `[1, S, D]` ou por amostra `[B, S, D]`; qualquer dtype float e normalizado para FP32 internamente.
- O treinamento flui pela rota fundida: um autograd custom aplica a rotacao inversa (a mesma rotacao pairwise com sin negado — RoPE e ortonormal) a dQ/dK no backward, devolvendo gradientes em relacao a Q/K pre-RoPE. Verificado exato contra uma referencia diferenciavel em FP32, incluindo tabelas batched por amostra. Causal tambem e suportado com gradientes; chamadas com mask ou GQA que exigem gradientes ainda usam rotacao eager.
- Entradas GQA (menos heads de K/V) sao rotacionadas na quantidade de heads de K/V e expandidas depois.

No shape DiT do Z-Image `(1, 30, 4128, 128)` em BF16, a rota fundida mediu 4.4 ms/layer mais rapida que rotacao eager + SDPA em um benchmark de cadeia de 12 layers. A integracao por modelo no SimpleTuner esta pendente; ate la o entry point esta disponivel para uso direto.

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
- Fallback inesperado: verifique tensores MPS FP32/FP16/BF16 4D BHSD (qualquer quantidade de heads e sequence length; single-head e suportado), `dropout_p=0` e `enable_gqa` desativado. Masks bool/additive de ate 4D sao elegiveis. Chamadas causais sao elegiveis com e sem gradientes. Confirme que o build UMFA registra a dispatch key PyTorch `MPS` e expoe `get_dispatch_stats()`; builds registrados apenas em `PrivateUse1` sao ignorados por tensores `torch.device("mps")`.
- 2048px falha antes da atencao: provavelmente e pressao de memoria da cache VAE, nao memoria de atencao UMFA. Ative `vae_enable_tiling=true` ou gere/reuse latentes com um cache workflow de menor memoria.
