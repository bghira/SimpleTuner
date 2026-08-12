# ConvRot / Hadamard SDNQ

O SimpleTuner expõe uma rotação no estilo ConvRot pelo caminho Hadamard do SDNQ. Isso é útil para jobs PEFT grandes em que o modelo base congelado deve rodar em int8 enquanto adaptadores LoRA ou LyCORIS continuam treináveis em bf16.

O SimpleTuner não consome buffers sidecar ConvRot arbitrários como um recurso separado. No caminho comum, carregue os pesos originais do modelo e deixe o SimpleTuner quantizar o componente treinado com SDNQ depois do carregamento. Loaders que aceitam pesos transformer quantizados em arquivo único também podem carregar safetensors transformer INT8 ConvRot compatíveis e executá-los via SDNQ Hadamard.

## Configuração rápida

```json
{
  "base_model_precision": "int8-sdnq",
  "gradient_checkpointing": true,
  "sdnq_use_hadamard": true,
  "sdnq_hadamard_group_size": 256,
  "sdnq_group_size": -1,
  "sdnq_use_quantized_matmul": true,
  "sdnq_compile_mode": "compile"
}
```

Para modelos grandes, mantenha `quantize_via` em `cpu`, a menos que o guia do modelo diga o contrário. A quantização em CPU reduz o pico de memória do acelerador durante a inicialização.

## O que as opções fazem

- `base_model_precision: int8-sdnq` seleciona quantização SDNQ int8 pós-carregamento para o componente base treinado.
- `sdnq_use_hadamard: true` ativa o caminho de rotação Hadamard.
- `sdnq_hadamard_group_size: 256` define o tamanho de bloco de rotação usado pelo SDNQ. Use `256` para ConvRot; blocos menores selecionam um caminho no estilo QuaRot.
- `sdnq_group_size: -1` usa scales estáticos por linha de peso. Isso evita o caminho dinâmico agrupado, mais voltado para full fine-tuning, que pode requantizar pesos durante o treino.
- `sdnq_use_quantized_matmul: true` mantém ativo o caminho SDNQ int8 matmul.
- `sdnq_compile_mode: compile` compila helpers e kernels de quantização onde o SDNQ oferece suporte.
- `gradient_checkpointing: true` permite que o SDNQ use o caminho de treino de menor overhead para workloads PEFT. O SimpleTuner passa isso ao SDNQ como `use_grad_ckpt=True`; com gradient checkpointing habilitado, definir esse flag do SDNQ como false só adiciona trabalho para salvar entradas backward quantizadas que o checkpointing descarta imediatamente.

## Comportamento PEFT

O transformer base é quantizado pelo SDNQ. Os pesos do adaptador continuam treináveis e usam o dtype normal de precisão mista, geralmente bf16.

Alguns modelos carregam adaptadores auxiliares fixos antes do treino. O Z-Image Turbo, por exemplo, tem uma assistant LoRA. O SimpleTuner adia esse adaptador até depois da quantização SDNQ, para que o SDNQ veja os módulos transformer originais em vez dos proxy weights do wrapper PEFT.

## Requisitos e limites

- O SimpleTuner instala e configura a dependência de treinamento SDNQ para os targets de instalação suportados.
- Este preset é voltado para treino LoRA e LyCORIS em modelos grandes. Full fine-tuning com SDNQ Hadamard precisa de validação separada.
- Os primeiros steps podem ser lentos porque SDNQ e Torch compilam kernels durante a inicialização e o início do treino.
- Validação e inferência usam o modelo base quantizado mais o adaptador ativo, igual ao treino.
- ConvRot pode reduzir o dano de quantização, mas não garante que INT8 iguale BF16 ou FP8 em todos os modelos. Valide a curva de loss e as amostras geradas antes de iniciar um run longo.
- Inferência standalone com SDNQ ConvRot fica fora deste guia de treinamento. Para APIs diretas de inferência SDNQ, siga a [documentação upstream do SDNQ](https://github.com/Disty0/sdnq) porque essa API muda com mais frequência do que a configuração de treinamento do SimpleTuner.

## Resultados medidos

Estas são medições do trainer real do SimpleTuner por modelo, não resultados sintéticos apenas de GEMM. `Loop s/step` é o tempo de parede do loop de treinamento por passo. `Passo médio` exclui os cinco primeiros passos de warmup.

| Modelo | GPU | Passos | Caminho dos pesos | Loop s/step | Passo médio | p50 | p95 | VRAM alocada máxima |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Z-Image Turbo LoRA | H100 80GB | 1000 | quantização post-load SDNQ Hadamard | 1.107 | 1.087 | 1.071 | 1.109 | 9.70 GiB |
| Z-Image Turbo LoRA | L40S | 1000 | quantização post-load SDNQ Hadamard | 1.026 | 1.018 | 1.002 | 1.040 | 9.66 GiB |
| Z-Image Turbo LoRA | L40S | 1000 | baseline SDNQ Hadamard | 1.131 | 1.072 | 1.055 | 1.102 | 9.66 GiB |
| Krea 2 Raw LoRA | H100 80GB | 100 | pesos transformer `lilcheaty/Krea2-INT8-ConvRot`, atenção diffusers | 0.787 | 0.399 | 0.397 | 0.411 | 32.15 GiB |
| Krea 2 Raw LoRA | L40S | 100 | pesos transformer `lilcheaty/Krea2-INT8-ConvRot`, atenção cuDNN | 0.945 | 0.794 | 0.793 | 0.799 | 31.89 GiB |
| Mage-Flow LoRA, crop quadrado | H100 80GB | 100 | quantização post-load SDNQ INT8 vanilla | 1.113 | 0.277 | 0.276 | 0.286 | 20.12 GiB |
| Mage-Flow LoRA, crop quadrado | H100 80GB | 100 | quantização post-load SDNQ ConvRot 256 | 0.436 | 0.299 | 0.297 | 0.308 | 20.15 GiB |

Na comparação Z-Image com cache quente na L40S, o caminho atual foi 10.3% mais rápido pelo tempo de loop e 5.2% mais rápido pela média medida de passo do que o baseline SDNQ Hadamard. As linhas Krea 2 verificam o caminho de pesos transformer INT8 ConvRot do Hugging Face em execuções reais de treinamento de 100 passos. As linhas Mage-Flow mostram por que a validação por modelo importa: o crop quadrado removeu a maior parte do churn de compilação por formas, ConvRot reduziu o tempo total do loop frente ao INT8 vanilla, mas o passo medido ja quente foi um pouco mais lento que INT8 vanilla.

## Modelos de exemplo

O SimpleTuner inclui exemplos SDNQ Hadamard para Z-Image Turbo, Krea 2, FLUX.2, Cosmos 3 e LTXVideo 2.3. Esses exemplos usam `sdnq_group_size: -1` porque essa configuração combinou melhor com PEFT do que o padrão dinâmico agrupado de treino.
