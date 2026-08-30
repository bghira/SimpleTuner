# NextLat

NextLat é um objetivo auxiliar que ensina um transformer a tornar seus hidden states preditivos do próximo hidden state.

O paper de Next-Latent Prediction estuda transformers de linguagem e argumenta que next-token prediction padrão não força o modelo a comprimir o histórico em estados internos compactos e estáveis. NextLat adiciona uma transição autosupervisionada em latent space: a partir do hidden state atual, prever o próximo hidden state. No SimpleTuner, a ideia vira um regularizador experimental para famílias transformer compatíveis.

A inferência não muda. NextLat adiciona uma loss de treinamento e um pequeno predictor; não adiciona um sampler novo.

## ELI5

Treinamento padrão diz: "com o que você viu, preveja o próximo output".

NextLat acrescenta: "também faça suas notas internas preverem suas próximas notas internas".

Em modelos de imagem, vídeo e áudio, essas notas internas são hidden tokens dentro do transformer. Se o modelo aprende transições internas coerentes, ele pode formar um plano melhor entre tokens, frames, patches ou posições de códigos RVQ.

## O Que Muda

Durante treinamento:

1. SimpleTuner captura hidden states de um bloco transformer.
2. O predictor recebe cada hidden token exceto o último.
3. Ele prevê o hidden token seguinte.
4. O hidden token real seguinte é usado como target sem gradiente.
5. A loss auxiliar é somada à loss normal.

O modelo base continua treinando no objetivo principal. NextLat é um objetivo lateral para incentivar dinâmica preditiva nos estados internos.

## Pseudocódigo

```text
para cada batch:
    predicao = modelo(batch)
    main_loss = loss_normal(predicao, target)

    hidden = hidden_states_capturados
    atual = hidden tokens 0..N-2
    proximo = hidden tokens 1..N-1

    pred_proximo = nextlat_predictor(atual)
    nextlat_loss = distancia(pred_proximo, stop_gradient(proximo))

    total_loss = main_loss + nextlat_weight * nextlat_loss
    treinar_com(total_loss)
```

Se a família expõe uma logits head compatível:

```text
pred_logits = logits_head(pred_proximo)
target_logits = logits_head(stop_gradient(proximo))
total_loss += nextlat_kl_weight * agreement_loss(pred_logits, target_logits)
```

A maioria dos usuários deve manter `nextlat_kl_weight=0`.

## Comportamento no SimpleTuner

- Funciona em famílias transformer que expõem hidden states.
- Captura um bloco escolhido por `nextlat_block_index`.
- `-1` significa o último bloco suportado.
- Achata hidden states de imagem, vídeo, áudio ou tokens em uma sequência.
- Prevê um passo à frente na ordem dos hidden tokens.
- O target é detached.
- O predictor é salvo como módulo treinável extra quando o modo de treinamento suporta isso.

Use PEFT LoRA padrão ou full-model training, salvo se o guia do modelo indicar outro modo compatível.

## Configuração Rápida

### WebUI

1. Abra **Training → Loss functions**.
2. Ative **NextLat**.
3. Mantenha **NextLat Block Index** em `-1` na primeira execução.
4. Defina **NextLat Weight** como um valor pequeno e positivo.
5. Deixe **NextLat State Loss** em `smooth_l1`.
6. Deixe **NextLat KL Weight** em `0`, salvo recomendação.

### Config JSON / CLI

```json
{
  "nextlat_enabled": true,
  "nextlat_block_index": -1,
  "nextlat_weight": 0.05,
  "nextlat_state_loss": "smooth_l1",
  "nextlat_kl_weight": 0.0
}
```

## Opções

- `nextlat_enabled`: ativa NextLat.
- `nextlat_block_index`: bloco transformer zero-based; `-1` usa o último suportado.
- `nextlat_weight`: multiplicador da loss auxiliar; deve ser maior que zero.
- `nextlat_state_loss`: `smooth_l1` por padrão ou `mse`.
- `nextlat_kl_weight`: KL opcional quando há logits head compatível.

## Escolhendo Valores

| Situação | Começo sugerido |
| --- | --- |
| Primeira LoRA transformer | `nextlat_block_index=-1`, `nextlat_weight=0.02` a `0.05` |
| Planner AR/RVQ | bloco tardio, `smooth_l1`, peso pequeno |
| Transformer de vídeo | bloco médio-tardio se o final restringir demais |
| Loss auxiliar instável | reduza `nextlat_weight` antes de trocar o bloco |
| Guia recomenda KL | use apenas o valor documentado |

## Logs

- `nextlat_loss`: loss auxiliar ponderada.
- `nextlat_state_loss`: loss bruta de previsão de hidden states.
- `nextlat_kl_loss`: termo KL opcional.

A loss bruta serve para acompanhar tendência; ela não precisa ter a mesma escala da loss principal.

## Compatibilidade

Veja a tabela de recursos no [Quick Start](../QUICKSTART.pt-BR.md).

Requisitos:

- O modelo precisa expor hidden states do transformer.
- O bloco escolhido precisa existir e ser capturável.
- A sequência capturada precisa ter pelo menos dois hidden tokens.
- O modo de treinamento precisa salvar o predictor do NextLat.

NextLat combina naturalmente com LayerSync, Internal Guidance e CREPA, mas aumenta memória porque hidden states precisam ficar disponíveis até a loss auxiliar.

## O Que Esperar

NextLat tende a ajudar quando transições internas coerentes importam: planners de códigos RVQ/audio, video transformers, image transformers com estrutura espacial e modelos multimodais que precisam de um plano interno estável.

Pode ajudar menos em experimentos muito pequenos, quando o peso domina a loss principal, ou quando a família não expõe hidden states úteis.

## Conselhos Práticos

- Comece com uma ablação curta.
- Mantenha `nextlat_weight` baixo.
- Prefira `smooth_l1`.
- Teste `-1` primeiro; depois tente bloco médio-tardio se necessário.
- Deixe KL desligado sem recomendação específica.
- Se VRAM subir muito, reduza batch size ou desligue outros regularizadores de hidden states.

## Referências

- [Paper Next-Latent Prediction](https://arxiv.org/abs/2511.05963)
- [Código de referência NextLat](https://github.com/JaydenTeoh/NextLat)
