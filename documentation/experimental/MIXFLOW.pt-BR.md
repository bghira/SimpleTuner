# Treinamento MixFlow

MixFlow é um método de pós-treinamento para modelos flow-matching. O modelo no timestep $t$ recebe uma interpolação real com mais ruído. Isso reduz a diferença entre as interpolações exatas do treino e os latents imperfeitos encontrados durante a amostragem.

## Configuração

```json
{
  "mixflow_enabled": true,
  "mixflow_gamma": 0.8
}
```

`mixflow_gamma` controla o intervalo da interpolação desacelerada. `0.8` é o padrão do artigo. `0.0` preserva a interpolação padrão, mas mantém a amostragem de timesteps do MixFlow.

MixFlow amostra o timestep orientado aos dados de $Beta(2,1)$. SimpleTuner armazena flow sigma na direção oposta, orientada ao ruído. A implementação usa $sigma = 1 - sqrt(U)$ e aplica o flow schedule shift configurado. O modelo recebe o timestep original. O latent de entrada usa:

$$
sigma_{input} = sigma + U' gamma (1 - sigma)
$$

O alvo de velocidade não muda para um caminho flow linear. A inferência não muda.

## Suporte

Todas as famílias SimpleTuner com prediction type `flow_matching` usam o caminho MixFlow compartilhado. Os wrappers tratam convenções data-ward, transformações sigma não lineares e entradas conjuntas de áudio/video.

MixFlow não pode ser combinado com custom/uniform/Beta/fast flow schedules, Self-Flow, TwinFlow, scheduled sampling ou distillation. Schedule shift continua suportado.

Use MixFlow para pós-treinar um modelo flow existente. Comece com o learning rate e optimizer de uma continuação convencional curta e compare amostras de validação com seed fixo ao checkpoint inicial.

## Referências

- [Artigo MixFlow](https://arxiv.org/abs/2512.19311)
- [Implementação de referência](https://github.com/fudan-generative-vision/MixFlow)
