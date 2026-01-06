# Guia do FlexAttention

**FlexAttention requer dispositivos CUDA.**

FlexAttention e o kernel de atencao em nivel de bloco do PyTorch que chegou no PyTorch 2.5.0. Ele reescreve o calculo de SDPA como um loop programavel para que voce expresse estrategias de mascara sem escrever CUDA. O Diffusers expoe isso pelo novo dispatcher `attention_backend`, e o SimpleTuner conecta esse dispatcher a `--attention_mechanism=flex`.

> AVISO: O FlexAttention ainda esta rotulado como “prototype” upstream. Espere recompilar quando trocar drivers, versoes do CUDA ou builds do PyTorch.

## Pre-requisitos

1. **GPU Ampere+** - NVIDIA SM80 (A100), Ada (4090/L40S) ou Hopper (H100/H200) sao suportadas. Placas mais antigas falham na verificacao de capacidade durante o registro do kernel.
2. **Toolchain de compilacao** - os kernels compilam em runtime com `nvcc`. Instale `cuda-nvcc` que corresponda ao wheel (CUDA 12.8 nas releases atuais) e garanta que `nvcc` esteja no `$PATH`.

## Compilando os kernels

A primeira importacao de `torch.nn.attention.flex_attention` compila a extensao CUDA no cache lazy do PyTorch. Voce pode fazer isso antes para expor erros de build mais cedo:

```bash
python - <<'PY'
import torch
from torch.nn.attention import flex_attention

assert torch.__version__ >= "2.5.0", torch.__version__
flex_attention.build_flex_attention_kernels()  # no-op when already compiled
print("FlexAttention kernels installed at", flex_attention.kernel_root)
PY
```

- Se voce ver `AttributeError: flex_attention has no attribute build_flex_attention_kernels`, atualize o PyTorch - o helper chegou no 2.5.0+.
- O cache fica em `~/.cache/torch/kernels`. Remova-o se voce atualizar o CUDA e precisar forcar uma recompilacao.

## Habilitando FlexAttention no SimpleTuner

Depois que os kernels existirem, selecione o backend via `config.json`:

```json
{
  "attention_mechanism": "flex"
}
```

O que esperar:

- Apenas blocos transformer com dispatch habilitado (Flux, Wan 2.2, LTXVideo, QwenImage, etc.) passam pelo `attention_backend` do Diffusers. UNets classicos de SD/SDXL continuam chamando o SDPA do PyTorch diretamente, entao o FlexAttention nao tem efeito ali.
- O FlexAttention atualmente suporta tensores BF16/FP16. Se voce usar pesos FP32 ou FP8, vera `ValueError: Query, key, and value must be either bfloat16 or float16`.
- O backend respeita apenas `is_causal=False`. Fornecer uma mascara a converte para a block mask que o kernel espera, mas mascaras irregulares arbitrarias ainda nao sao suportadas (espelha o comportamento upstream).

## Checklist de solucao de problemas

| Sintoma | Correcao |
| --- | --- |
| `RuntimeError: Flex Attention backend 'flex' is not usable because of missing package` | A build do PyTorch e < 2.5 ou nao inclui CUDA. Instale um wheel CUDA mais novo. |
| `Could not compile flex_attention kernels` | Garanta que `nvcc` corresponda a versao de CUDA esperada pelo seu wheel do torch (12.1+). Defina `export CUDA_HOME=/usr/local/cuda-12.4` se o instalador nao achar os headers. |
| `ValueError: Query, key, and value must be on a CUDA device` | FlexAttention e somente CUDA. Remova o backend em execucoes Apple/ROCm. |
| O treinamento nunca troca para o backend | Garanta que voce esta usando uma familia de modelo que ja usa o `dispatch_attention_fn` do Diffusers (Flux/Wan/LTXVideo). UNets SD padrao continuarao usando SDPA do PyTorch independentemente do backend selecionado. |

Consulte a documentacao upstream para mais detalhes de internals e flags de API: [PyTorch FlexAttention docs](https://pytorch.org/docs/stable/nn.attention.html#flexattention).
