# Scheduled Sampling (Rollout)

## Contexto

O treinamento padrao de difusao depende de "Teacher Forcing". Pegamos uma imagem limpa, adicionamos uma quantidade precisa de ruido e pedimos ao modelo para predizer esse ruido (ou a velocidade/imagem original). A entrada para o modelo e sempre "perfeitamente" ruidosa - ela fica exatamente no schedule teorico de ruido.

No entanto, durante a inferencia (geracao), o modelo se alimenta das suas proprias saidas. Se ele comete um pequeno erro no step $t$, esse erro alimenta o step $t-1$. Esses erros se acumulam, fazendo a geracao sair da manifold de imagens validas. Essa discrepancia entre treinamento (entradas perfeitas) e inferencia (entradas imperfeitas) se chama **Exposure Bias**.

**Scheduled Sampling** (frequentemente chamado de "Rollout" neste contexto) resolve isso treinando o modelo com suas proprias saidas geradas.

## Como funciona

Em vez de simplesmente adicionar ruido a uma imagem limpa, o loop de treino ocasionalmente faz uma mini-sessao de inferencia:

1.  Escolha um **timestep alvo** $t$ (o step que queremos treinar).
2.  Escolha um **timestep de origem** $t+k$ (um step mais ruidoso mais atras no schedule).
3.  Use os pesos *atuais* do modelo para realmente gerar (denoise) de $t+k$ ate $t$.
4.  Use esse latente auto-gerado e levemente imperfeito no step $t$ como entrada para o passe de treino.

Ao fazer isso, o modelo ve entradas que contem exatamente o tipo de artefatos e erros que ele produz atualmente. Ele aprende a dizer: "Ah, eu cometi este erro, aqui esta como corrijo", efetivamente puxando a geracao de volta para o caminho valido.

## Configuracao

Este recurso e experimental e adiciona overhead computacional, mas pode melhorar bastante a aderencia ao prompt e a estabilidade estrutural, especialmente em datasets pequenos (Dreambooth).

Para habilitar, voce deve configurar um `max_step_offset` nao-zero.

### Configuracao basica

Adicione o seguinte ao seu `config.json`:

```json
{
  "scheduled_sampling_max_step_offset": 10,
  "scheduled_sampling_probability": 1.0,
  "scheduled_sampling_sampler": "unipc"
}
```

### Referencia de opcoes

#### `scheduled_sampling_max_step_offset` (Integer)
**Padrao:** `0` (Desativado)
O numero maximo de steps para rollout. Se definido como `10`, o trainer escolhera um comprimento de rollout aleatorio entre 0 e 10 para cada amostra.
> **Recomendacao:** Comece pequeno (ex.: `5` a `10`). Mesmo rollouts curtos ajudam o modelo a aprender a corrigir erros sem desacelerar demais o treino.

#### `scheduled_sampling_probability` (Float)
**Padrao:** `0.0`
A chance (0.0 a 1.0) de qualquer item do batch passar por rollout.
*   `1.0`: Toda amostra passa por rollout (compute mais pesado).
*   `0.5`: 50% das amostras sao treino padrao, 50% sao rollout.

#### `scheduled_sampling_ramp_steps` (Integer)
**Padrao:** `0`
Se definido, a probabilidade sobe linearmente de `scheduled_sampling_prob_start` (padrao 0.0) para `scheduled_sampling_prob_end` (padrao 0.5) ao longo desse numero de steps globais.
> **Dica:** Isso funciona como um "warmup". Permite que o modelo aprenda o denoising basico antes de introduzir a tarefa mais dificil de corrigir seus proprios erros.

#### `scheduled_sampling_sampler` (String)
**Padrao:** `unipc`
O solver usado para os steps de rollout.
*   **Opcoes:** `unipc` (recomendado, rapido e preciso), `euler`, `dpm`, `rk4`.
*   `unipc` e geralmente o melhor trade-off entre velocidade e precisao para esses pequenos bursts de amostragem.

### Flow Matching + ReflexFlow

Para modelos flow-matching (`--prediction_type flow_matching`), o scheduled sampling agora suporta mitigacao de exposure bias no estilo ReflexFlow:

*   `scheduled_sampling_reflexflow`: Habilita melhorias ReflexFlow durante o rollout (auto-habilitado para modelos flow-matching quando scheduled sampling esta ativo; passe `--scheduled_sampling_reflexflow=false` para sair).
*   `scheduled_sampling_reflexflow_alpha`: Escala o peso da loss baseada em exposure bias (compensacao de frequencia).
*   `scheduled_sampling_reflexflow_beta1`: Escala o regularizador direcional anti-drift (padrao 10.0 para espelhar o paper).
*   `scheduled_sampling_reflexflow_beta2`: Escala a loss compensada por frequencia (padrao 1.0).

Esses reutilizam as predicoes/latentes de rollout ja calculados, evitando um passe de gradiente extra, e ajudam a manter rollouts enviesados alinhados a trajetoria limpa enquanto enfatizam componentes de baixa frequencia ausentes no inicio do denoising.

### Impacto de desempenho

AVISO: Habilitar rollout requer rodar o modelo em modo inferencia *dentro* do loop de treinamento.

Se voce definir `max_step_offset=10`, o modelo pode rodar ate 10 passes forward extras por step de treinamento. Isso reduz seu `it/s` (iteracoes por segundo). Ajuste `scheduled_sampling_probability` para equilibrar velocidade de treino vs. ganhos de qualidade.
