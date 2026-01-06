# Documentacao de Treinamento TREAD

> AVISO: **Recurso Experimental**: O suporte a TREAD no SimpleTuner foi implementado recentemente. Embora funcional, configuracoes otimas ainda estao sendo exploradas e alguns comportamentos podem mudar em futuras releases.

## Visao geral

TREAD (Token Routing for Efficient Architecture-agnostic Diffusion Training) e um metodo de aceleracao de treinamento que acelera o treino de modelos de difusao ao rotear tokens de forma inteligente pelos layers transformer. Ao processar seletivamente apenas os tokens mais importantes em certas camadas, o TREAD pode reduzir significativamente os custos computacionais mantendo a qualidade do modelo.

Com base na pesquisa de [Krause et al. (2025)](https://arxiv.org/abs/2501.04765), o TREAD alcanca aceleracoes de treinamento ao:
- Selecionar dinamicamente quais tokens processar em cada camada transformer
- Manter o fluxo de gradiente por todos os tokens via conexoes de skip
- Usar decisoes de roteamento baseadas em importancia

O speedup e diretamente proporcional ao `selection_ratio` - quanto mais proximo de 1.0, mais tokens sao descartados e mais rapido o treinamento fica.

## Como o TREAD funciona

### Conceito central

Durante o treinamento, o TREAD:
1. **Roteia tokens** - Para camadas transformer especificas, seleciona um subconjunto de tokens para processar com base na importancia
2. **Processa o subconjunto** - Apenas os tokens selecionados passam pelas operacoes caras de atencao e MLP
3. **Restaura a sequencia completa** - Apos o processamento, a sequencia completa de tokens e restaurada com gradientes fluindo para todos os tokens

### Selecao de tokens

Tokens sao selecionados com base na norma L1 (pontuacao de importancia), com randomizacao opcional para exploracao:
- Tokens de maior importancia tem mais chance de ser mantidos
- Uma mistura de selecao por importancia e aleatoria evita overfitting em padroes especificos
- Mascara de force-keep pode garantir que certos tokens (como regioes mascaradas) nunca sejam descartados

## Configuracao

### Configuracao basica

Para habilitar treinamento TREAD no SimpleTuner, adicione o seguinte ao seu config:

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": 5
      }
    ]
  }
}
```

### Configuracao de rotas

Cada rota define uma janela onde o roteamento de tokens esta ativo:
- `selection_ratio`: Fracao de tokens a descartar (0.5 = manter 50% dos tokens)
- `start_layer_idx`: Primeira camada onde o roteamento comeca (indexado a partir de 0)
- `end_layer_idx`: Ultima camada onde o roteamento esta ativo

Indices negativos sao suportados: `-1` se refere a ultima camada.

### Exemplo avancado

Multiplas janelas de roteamento com diferentes selection ratios:

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.3,
        "start_layer_idx": 1,
        "end_layer_idx": 3
      },
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 4,
        "end_layer_idx": 8
      },
      {
        "selection_ratio": 0.7,
        "start_layer_idx": -4,
        "end_layer_idx": -1
      }
    ]
  }
}
```

## Compatibilidade

### Modelos suportados
- **FLUX Dev/Kontext, Wan, AuraFlow, PixArt e SD3** - Atualmente as unicas familias de modelos suportadas
- Suporte futuro planejado para outros diffusion transformers

### Funciona bem com
- **Treinamento com loss mascarada** - O TREAD preserva automaticamente regioes mascaradas quando combinado com condicionamento de mask/segmentation
- **Treinamento multi-GPU** - Compativel com setups de treinamento distribuido
- **Treinamento quantizado** - Pode ser usado com quantizacao int8/int4/NF4

### Limitacoes
- Ativo apenas durante o treinamento (nao na inferencia)
- Requer computacao de gradiente (nao funciona em modo eval)
- Implementacao especifica de FLUX e Wan no momento; nao disponivel para Lumina2 e outras arquiteturas ainda

## Consideracoes de desempenho

### Beneficios de velocidade
- O speedup do treino e proporcional ao `selection_ratio` (mais proximo de 1.0 = mais tokens descartados = treinamento mais rapido)
- **Os maiores ganhos ocorrem com entradas de video longas e resolucoes altas** devido a complexidade O(n^2) da atencao
- Tipicamente 20-40% de speedup, mas os resultados variam conforme a configuracao
- Com treinamento de loss mascarada, o speedup e reduzido porque tokens mascarados nao podem ser descartados

### Trade-offs de qualidade
- **Maior descarte de tokens leva a maior loss inicial** ao iniciar treino LoRA/LoKr
- A loss tende a corrigir rapidamente e as imagens se normalizam logo, a menos que um selection ratio alto esteja em uso
  - Isso pode ser a rede ajustando para menos tokens em camadas intermediarias
- Ratios conservadores (0.1-0.25) geralmente mantem a qualidade
- Ratios agressivos (>0.35) impactam a convergencia

### Consideracoes especificas de LoRA
- O desempenho pode depender dos dados - configs de roteamento otimas precisam de mais exploracao
- O pico inicial de loss e mais evidente em LoRA/LoKr do que em fine-tuning completo

### Configuracoes recomendadas

Para equilibrio entre velocidade e qualidade:
```json
{
  "routes": [
    {"selection_ratio": 0.5, "start_layer_idx": 2, "end_layer_idx": -2}
  ]
}
```

Para velocidade maxima (espere um pico grande de loss):
```json
{
  "routes": [
    {"selection_ratio": 0.7, "start_layer_idx": 1, "end_layer_idx": -1}
  ]
}
```

Para treinamento em alta resolucao (1024px+):
```json
{
  "routes": [
    {"selection_ratio": 0.6, "start_layer_idx": 2, "end_layer_idx": -3}
  ]
}
```

## Detalhes tecnicos

### Implementacao do roteador

O roteador TREAD (classe `TREADRouter`) cuida de:
- Calculo da importancia do token via norma L1
- Geracao de permutacoes para roteamento eficiente
- Restauracao de tokens preservando gradientes

### Integracao com atencao

O TREAD modifica os embeddings de posicao rotatoria (RoPE) para corresponder a sequencia roteada:
- Tokens de texto mantem posicoes originais
- Tokens de imagem usam posicoes embaralhadas/recortadas
- Garante consistencia posicional durante o roteamento
- **Nota**: A implementacao de RoPE para FLUX pode nao estar 100% correta, mas parece funcional na pratica

### Compatibilidade com loss mascarada

Ao usar treinamento com loss mascarada:
- Tokens dentro da mascara sao automaticamente force-kept
- Evita que sinais importantes de treinamento sejam descartados
- Ativado via `conditioning_type` em ["mask", "segmentation"]
- **Nota**: Isso reduz o speedup, pois mais tokens precisam ser processados

## Problemas conhecidos e limitacoes

### Status da implementacao
- **Recurso experimental** - O suporte TREAD foi implementado recentemente e pode ter problemas ainda nao descobertos
- **Tratamento de RoPE** - A implementacao de embeddings de posicao rotatoria para roteamento pode nao estar perfeitamente correta
- **Testes limitados** - Configuracoes de roteamento otimas nao foram exploradas extensivamente

### Comportamento do treinamento
- **Pico inicial de loss** - Ao iniciar treino LoRA/LoKr com TREAD, espere loss inicial maior que se corrige rapidamente
- **Desempenho de LoRA** - Algumas configuracoes podem mostrar leves lentidoes com treino LoRA
- **Sensibilidade de configuracao** - O desempenho depende muito das escolhas de configuracao de roteamento

### Bugs conhecidos (corrigidos)
- O treinamento com loss mascarada estava quebrado em versoes anteriores, mas foi corrigido com checagem adequada de model flavour (`kontext` guard)

## Solucao de problemas

### Problemas comuns

**"TREAD training requires you to configure the routes"**
- Garanta que `tread_config` inclui um array `routes`
- Cada rota precisa de `selection_ratio`, `start_layer_idx` e `end_layer_idx`

**Treinamento mais lento do que o esperado**
- Verifique se as rotas cobrem intervalos de camadas relevantes
- Considere selection ratios mais agressivos
- Verifique se o gradient checkpointing nao esta conflitando
- Para treino LoRA, alguma lentidao e esperada - tente outras configs de roteamento

**Loss inicial alta com LoRA/LoKr**
- Isso e esperado - a rede precisa se adaptar a menos tokens
- A loss normalmente corrige em alguns centenas de steps
- Se a loss nao melhorar, reduza `selection_ratio` (mantenha mais tokens)

**Degradacao de qualidade**
- Reduza selection ratios (mantenha mais tokens)
- Evite roteamento nas camadas iniciais (0-2) ou finais
- Garanta dados de treinamento suficientes para a eficiencia aumentada

## Exemplos praticos

### Treinamento em alta resolucao (1024px+)
Para maximo beneficio ao treinar em altas resolucoes:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.6, "start_layer_idx": 2, "end_layer_idx": -3}
    ]
  }
}
```

### Fine-tuning LoRA
Config conservadora para minimizar o pico inicial de loss:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.4, "start_layer_idx": 3, "end_layer_idx": -4}
    ]
  }
}
```

### Treinamento com loss mascarada
Ao treinar com masks, tokens em regioes mascaradas sao preservados:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.7, "start_layer_idx": 2, "end_layer_idx": -2}
    ]
  }
}
```
Nota: O speedup real sera menor do que 0.7 sugere devido a preservacao forcada de tokens.

## Trabalho futuro

Como o suporte TREAD no SimpleTuner foi implementado recentemente, existem varias areas para melhoria futura:

- **Otimizacao de configuracao** - Mais testes necessarios para encontrar configs de roteamento otimas para diferentes casos de uso
- **Desempenho de LoRA** - Investigacao do motivo de algumas configuracoes de LoRA apresentarem lentidoes
- **Implementacao de RoPE** - Refinamento do tratamento de embeddings de posicao rotatoria para melhor corretude
- **Suporte estendido a modelos** - Implementacao para outras arquiteturas de diffusion transformers alem de Flux
- **Configuracao automatizada** - Ferramentas para determinar automaticamente roteamentos ideais com base no modelo e no dataset

Contribuicoes da comunidade e resultados de testes sao bem-vindos para ajudar a melhorar o suporte ao TREAD.

## Referencias

- [TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training](https://arxiv.org/abs/2501.04765)
- [Documentacao do SimpleTuner Flux](quickstart/FLUX.md#tread-training)

## Citacao

```bibtex
@misc{krause2025treadtokenroutingefficient,
      title={TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training},
      author={Felix Krause and Timy Phan and Vincent Tao Hu and Bjorn Ommer},
      year={2025},
      eprint={2501.04765},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.04765},
}
```
