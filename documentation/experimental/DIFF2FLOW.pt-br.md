# Diff2Flow (Ponte de Difusao para Flow)

## Contexto

Historicamente, modelos de difusao foram categorizados pelos seus alvos de predicao:
*   **Epsilon ($\epsilon$):** Prediz o ruido adicionado a imagem (SD 1.5, SDXL).
*   **V-Prediction ($v$):** Prediz uma velocidade combinando ruido e dados (SD 2.0, SDXL Refiner).

Modelos mais novos como **Flux**, **Stable Diffusion 3** e **AuraFlow** usam **Flow Matching** (especificamente Rectified Flow). Flow Matching trata o processo de geracao como uma Equacao Diferencial Ordinaria (ODE) que move particulas de uma distribuicao de ruido para uma distribuicao de dados em trajetorias retas.

Essa trajetoria em linha reta e geralmente mais facil para os solvers percorrerem, permitindo menos steps e geracao mais estavel.

## A ponte

**Diff2Flow** e um adapter leve que permite treinar modelos "legados" (Epsilon ou V-pred) com um objetivo de Flow Matching sem mudar a arquitetura subjacente.

Ele funciona convertendo matematicamente a saida nativa do modelo (por exemplo, uma predicao epsilon) em um campo vetorial de flow $u_t(x|1)$ e, em seguida, computa a loss contra o alvo de flow ($x_1 - x_0$, ou `noise - latents`).

> **Status experimental:** Este recurso muda a paisagem de loss que o modelo enxerga. Embora teoricamente consistente, ele altera significativamente a dinamica de treinamento. E voltado principalmente para pesquisa e experimentacao.

## Configuracao

Para usar Diff2Flow, voce precisa habilitar a ponte e opcionalmente trocar a funcao de loss.

### Configuracao basica

Adicione estas chaves ao seu `config.json`:

```json
{
  "diff2flow_enabled": true,
  "diff2flow_loss": true
}
```

### Referencia de opcoes

#### `--diff2flow_enabled` (Boolean)
**Padrao:** `false`
Inicializa a ponte matematica. Isso aloca um pequeno buffer para calculos de timestep, mas nao muda o comportamento de treinamento por si so, a menos que `diff2flow_loss` tambem seja definido.
*   **Obrigatorio para:** `diff2flow_loss`.
*   **Modelos suportados:** Qualquer modelo usando `epsilon` ou `v_prediction` (SD1.5, SD2.x, SDXL, DeepFloyd IF, PixArt Alpha).

#### `--diff2flow_loss` (Boolean)
**Padrao:** `false`
Troca o objetivo de treinamento.
*   **False:** O modelo minimiza o erro entre sua predicao e o alvo padrao (ex.: `MSE(pred_noise, real_noise)`).
*   **True:** O modelo minimiza o erro entre a predicao *convertida para flow* e o alvo de flow (`noise - latents`).

### Sinergias

Diff2Flow combina muito bem com **Scheduled Sampling**.

Quando voce combina:
1.  **Diff2Flow** (Endireitando as trajetorias)
2.  **Scheduled Sampling** (Treino em rollouts auto-gerados)

Voce efetivamente aproxima a receita de treinamento usada por modelos **Reflow** ou **Rectified Flow**, potencialmente trazendo estabilidade moderna e qualidade para arquiteturas mais antigas como SDXL.
