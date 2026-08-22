# Explorative Modeling (XM)

Explorative Modeling, abreviado como XM no SimpleTuner, é uma técnica de treinamento que permite ao modelo tentar mais de uma escolha oculta para o mesmo exemplo supervisionado e aprender apenas com a escolha que melhor explica o alvo.

O trabalho original de Explorative Modeling trata exploration como um terceiro eixo de escala para modelos generativos: além de mais dados e mais parâmetros, o modelo pode gastar mais compute de treinamento explorando candidatos. No SimpleTuner, XM é um objetivo experimental para famílias compatíveis de imagem, vídeo, áudio e modelos autorregressivos.

A inferência não muda. XM altera apenas como o batch de treinamento é construído, pontuado e reduzido em loss.

## ELI5

Imagine pedir a alguém para desenhar uma imagem alvo, mas deixar a pessoa fazer quatro tentativas antes da nota. Em vez de tirar a média das quatro, você corrige a tentativa que chegou mais perto do alvo.

A ideia central:

1. Criar vários candidatos para o mesmo sample.
2. Rodar o modelo em todos os candidatos.
3. Comparar cada candidato com o target real.
4. Manter o melhor candidato por sample ou bloco de tokens.
5. Fazer backprop apenas pela loss selecionada.

Isso ajuda quando o target pode ser explicado de várias formas válidas. Um caminho único pode ensinar o modelo a tirar uma média; vários caminhos permitem escolher um modo plausível.

## O Que Muda

XM não adiciona sampler de inferência, formato novo de checkpoint nem outro teacher model. Ele muda a seleção durante treinamento:

- Treinamento padrão usa um candidato e aprende com ele.
- XM usa `K` candidatos e aprende com o candidato de menor loss.
- `K` maior dá mais exploration, mas custa mais compute.

Em modelos de difusão e flow, o candidato geralmente é o ruído usado para construir o latent ruidoso no timestep.

Em modelos autorregressivos de tokens, como planners RVQ/audio, o candidato é um route embedding aprendido que dá ao modelo vários caminhos internos para a mesma sequência supervisionada.

## Comportamento no SimpleTuner

### Modelos de Difusão e Flow

Para famílias de difusão ou flow matching compatíveis, use `xm_training_target=noise`.

SimpleTuner:

1. Amostra o timestep ou sigma normal.
2. Repete o batch `xm_candidate_count` vezes.
3. Gera um ruído diferente para cada candidato.
4. Constrói noised latents para cada candidato.
5. Executa o modelo no batch expandido.
6. Calcula a loss normal para cada candidato.
7. Seleciona o candidato de menor loss por sample original.
8. Faz backprop da loss selecionada.

O modelo continua aprendendo seu prediction type normal: flow velocity, epsilon, v-prediction ou sample prediction, dependendo da família.

### Modelos Autorregressivos e RVQ

Para planners autorregressivos compatíveis, use `xm_training_target=route`.

SimpleTuner:

1. Adiciona uma pequena tabela aprendida de route embeddings.
2. Repete cada sequência supervisionada entre os candidatos de rota.
3. Insere o sinal de rota na entrada do modelo.
4. Calcula token losses para cada rota.
5. Escolhe a melhor rota para o sample inteiro ou para blocos configurados.
6. Faz backprop apenas da loss da rota selecionada.

Isso é útil para planners tipo global LM que predizem códigos RVQ de áudio ou outros streams discretos. A rota oferece múltiplas explicações internas para o mesmo target sem mudar o decode de inferência.

## Pseudocódigo

```text
para cada batch:
    candidatos = []

    para candidate_id em 1..K:
        entrada = criar_candidato(batch, candidate_id)
        predicao = modelo(entrada)
        loss = comparar(predicao, target)
        candidatos.append(loss)

    loss_selecionada = menor_loss_por_sample_ou_bloco(candidatos)
    treinar_com(loss_selecionada)
```

Para difusão:

```text
entrada = adicionar_ruido(clean_latent, ruido_candidato, timestep)
loss = diffusion_or_flow_loss(modelo(entrada), target_de_treinamento)
```

Para seleção de rotas:

```text
entrada = adicionar_route_embedding(sequencia_tokens, rota_candidata)
loss = token_loss(modelo(entrada), tokens_alvo)
```

## Configuração Rápida

### WebUI

1. Abra **Training → Loss functions**.
2. Ative **XM**.
3. Defina **XM Candidates** como `2` ou `4`.
4. Escolha **XM Training Target**:
   - `noise` para difusão ou flow.
   - `route` para planners autorregressivos/RVQ.
5. Mantenha **XM Selection Scope** em `sample`, a menos que o guia do modelo recomende blocos.
6. Deixe **XM Block Size** em `0`, salvo para route block selection.

### Config JSON / CLI

```json
{
  "xm_enabled": true,
  "xm_candidate_count": 4,
  "xm_training_target": "noise",
  "xm_selection_scope": "sample",
  "xm_block_size": 0
}
```

Para AR/RVQ com rotas:

```json
{
  "xm_enabled": true,
  "xm_candidate_count": 4,
  "xm_training_target": "route",
  "xm_selection_scope": "block",
  "xm_block_size": 16
}
```

## Opções

- `xm_enabled`: ativa XM.
- `xm_candidate_count`: candidatos por sample; deve ser pelo menos `2` quando XM está ativo.
- `xm_training_target`: `noise` para difusão/flow, `route` para planners de tokens.
- `xm_selection_scope`: `sample` escolhe um vencedor por sample; `block` escolhe por blocos quando suportado.
- `xm_block_size`: tamanho do bloco de tokens ou frames; `0` significa a sequência supervisionada inteira.

## Escolhendo Valores

| Situação | Começo sugerido |
| --- | --- |
| LoRA de imagem ou vídeo | `xm_candidate_count=2`, `xm_training_target=noise`, `xm_selection_scope=sample` |
| Dataset ambíguo ou batch maior | Teste `xm_candidate_count=4` |
| Planner RVQ/audio | `xm_training_target=route`, `xm_selection_scope=block`, block size do guia |
| Primeira execução numa família | Mantenha block size `0` e compare com baseline sem XM |

O custo costuma crescer aproximadamente com o número de candidatos.

## Logs

XM pode registrar:

- `xm_loss`: loss selecionada.
- `xm_candidate_loss_mean`: média da loss dos candidatos antes da seleção.
- `xm_candidate_0_wins`, `xm_candidate_1_wins`: quantas vezes cada candidato venceu.
- `xm_route_usage`: uso de rotas em modelos AR/RVQ.

Bons sinais: vários candidatos vencem, validação melhora e route usage não colapsa por muito tempo.

Sinais ruins: um candidato vence sempre desde o início, training loss cai mas validação piora, ou o custo de memória/tempo fica alto demais.

## Compatibilidade

Veja a tabela de recursos no [Quick Start](../QUICKSTART.pt-BR.md).

Regras gerais:

- XM de difusão/flow usa candidatos de ruído e seleção por sample.
- XM AR/RVQ usa candidatos de rota e pode suportar seleção por blocos.
- Famílias não suportadas falham explicitamente.

Para XM de ruído em difusão, SimpleTuner atualmente trata TwinFlow, Scheduled Sampling, `input_perturbation`, CREPA self-flow e loss de segmentação estocástica como incompatíveis, salvo indicação específica da família.

## Relação com Outros Recursos

- **MixFlow** muda a trajetória de treinamento; XM muda a seleção de candidatos.
- **Diff2Flow** muda o target de modelos legacy.
- **NextLat** regulariza hidden-state dynamics; XM escolhe rotas ou ruídos.
- **LayerSync e CREPA** alinham representações; XM escolhe o candidato mais explicativo.

## Conselhos Práticos

- Use seeds fixos de validação ao comparar.
- Reduza batch size se `xm_candidate_count` pressionar VRAM.
- Não julgue XM só pela training loss; veja validação e diversidade.
- Em AR/RVQ, evite block size `1` sem recomendação do guia.
- Comece com uma ablação curta.

## Referências

- [Página do projeto Explorative Modeling](https://explorative-modeling.github.io/)
- [Paper Explorative Modeling](https://arxiv.org/abs/2607.27372)
