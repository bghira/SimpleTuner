# ConvRot / Hadamard SDNQ

O SimpleTuner expõe uma rotação no estilo ConvRot pelo caminho Hadamard do SDNQ. Isso é útil para jobs PEFT grandes em que o modelo base congelado deve rodar em int8 enquanto adaptadores LoRA ou LyCORIS continuam treináveis em bf16.

Isto não carrega diretamente buffers de checkpoints ConvRot externos. Carregue os pesos originais do modelo e deixe o SimpleTuner quantizar o componente treinado com SDNQ depois do carregamento.

## Configuração rápida

```json
{
  "base_model_precision": "int8-sdnq",
  "gradient_checkpointing": true,
  "sdnq_use_hadamard": true,
  "sdnq_hadamard_group_size": 128,
  "sdnq_group_size": -1,
  "sdnq_use_quantized_matmul": true,
  "sdnq_compile_mode": "compile"
}
```

Para modelos grandes, mantenha `quantize_via` em `cpu`, a menos que o guia do modelo diga o contrário. A quantização em CPU reduz o pico de memória do acelerador durante a inicialização.

## O que as opções fazem

- `base_model_precision: int8-sdnq` seleciona quantização SDNQ int8 pós-carregamento para o componente base treinado.
- `sdnq_use_hadamard: true` ativa o caminho de rotação Hadamard.
- `sdnq_hadamard_group_size: 128` define o tamanho de bloco de rotação usado pelo SDNQ.
- `sdnq_group_size: -1` usa scales estáticos por linha de peso. Isso evita o caminho dinâmico agrupado, mais voltado para full fine-tuning, que pode requantizar pesos durante o treino.
- `sdnq_use_quantized_matmul: true` mantém ativo o caminho SDNQ int8 matmul.
- `sdnq_compile_mode: compile` compila helpers e kernels de quantização onde o SDNQ oferece suporte.
- `gradient_checkpointing: true` permite que o SDNQ use o caminho de treino de menor overhead para workloads PEFT. O SimpleTuner passa isso ao SDNQ como `use_grad_ckpt=True`; com gradient checkpointing habilitado, definir esse flag do SDNQ como false só adiciona trabalho para salvar entradas backward quantizadas que o checkpointing descarta imediatamente.

## Comportamento PEFT

O transformer base é quantizado pelo SDNQ. Os pesos do adaptador continuam treináveis e usam o dtype normal de precisão mista, geralmente bf16.

Alguns modelos carregam adaptadores auxiliares fixos antes do treino. O Z-Image Turbo, por exemplo, tem uma assistant LoRA. O SimpleTuner adia esse adaptador até depois da quantização SDNQ, para que o SDNQ veja os módulos transformer originais em vez dos proxy weights do wrapper PEFT.

## Requisitos e limites

- Use um build do SDNQ com suporte a Hadamard. A verificação em H100 usou o SDNQ upstream `0.2.3`; PyPI `0.2.2` não inclui a mesma correção Hadamard para bf16.
- Este preset é voltado para treino LoRA e LyCORIS em modelos grandes. Full fine-tuning com SDNQ Hadamard precisa de validação separada.
- Os primeiros steps podem ser lentos porque SDNQ e Torch compilam kernels durante a inicialização e o início do treino.
- Validação e inferência usam o modelo base quantizado mais o adaptador ativo, igual ao treino.

## Modelos de exemplo

O SimpleTuner inclui exemplos SDNQ Hadamard para Z-Image Turbo, Krea 2, FLUX.2, Cosmos 3 e LTXVideo 2.3. Esses exemplos usam `sdnq_group_size: -1` porque essa configuração combinou melhor com PEFT do que o padrão dinâmico agrupado de treino.
