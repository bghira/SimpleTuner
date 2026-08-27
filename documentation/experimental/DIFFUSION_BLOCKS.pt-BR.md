# DiffusionBlocks

DiffusionBlocks converte um diffusion Transformer compativel em grupos de camadas treinados de forma independente. Cada grupo atende uma faixa de ruido; um forward executa apenas o grupo da batch atual.

Esta e uma conversao experimental baseada em [DiffusionBlocks](https://arxiv.org/abs/2506.14202), nao um simples congelamento de camadas. A inferencia deve usar o mesmo roteamento do treino.

## Configuracao

```json
{
  "diffusion_blocks_config": {
    "layers_per_block": 4,
    "overlap": 0.05
  },
  "find_unused_parameters": true
}
```

O DDP ativa `find_unused_parameters` automaticamente. Definir `false` gera erro.

| Chave | Padrao | Significado |
| --- | --- | --- |
| `layers_per_block` | obrigatorio | Maximo de camadas Transformer consecutivas por bloco de ruido. |
| `overlap` | `0.05` | Expansao fracionaria das faixas vizinhas, entre `0.0` e `0.5`. |
| `blocks_to_train` | `"all"` | Indices pertencentes ao job. Os demais grupos sao congelados apos criar o adapter. |
| `block_paths` | automatico | Caminhos `ModuleList` explicitos quando a descoberta automatica nao basta. |
| `timestep_boundaries` | automatico | Limites crescentes de `0.0` a `1.0`, com `num_blocks + 1` valores. |

Os limites automaticos dividem a distribuicao de timestep em faixas de igual probabilidade. O bloco `0` recebe o maior ruido e as primeiras camadas.

## Suporte

O mecanismo compartilhado aceita familias diffusion e flow-matching com listas homogeneas de blocos Transformer: stage unico, joint/single stream, double/single stream, `blocks` e `layers`.

UNet, ControlNet, Musubi block swap, TwinFlow, scheduled sampling com varios timesteps, CREPA com captura de camada fixa e LayerSync sao rejeitados no setup. Rotas TREAD mantem os indices globais das camadas do modelo e sao recortadas para a faixa global do grupo ativo.

O routing muda a arquitetura do denoiser. A perda inicial e a qualidade nao precisam corresponder a um run normal com profundidade completa. Ativar esta opcao nao transforma um LoRA normal existente em um adapter DiffusionBlocks treinado.

Use `block_paths` somente para stages sequenciais do denoiser. Nao selecione adapters de texto, blocos de VAE ou stages UNet com skip connection.
Stacks Transformer encoder-decoder com dependencia de skip, como `in_blocks`/`out_blocks` do i1, nao sao descobertos porque um grupo de saida nao pode executar sem as activations do grupo de entrada correspondente.

## Memoria

Somente o grupo ativo cria activations do Transformer. Um run com todos os blocos termina com optimizer state para todos os grupos treinaveis.

Para jobs independentes, use `blocks_to_train` diferente em cada job. Grupos nao pertencentes sao congelados e nao recebem optimizer state. Combine os checkpoints por propriedade de parametros antes da inferencia.


## Inferencia

A validacao do SimpleTuner usa o controller automaticamente. Um pipeline Diffusers comum nao deduz a conversao a partir do LoRA.

```python
from simpletuner.helpers.training.diffusion_blocks import DiffusionBlocksConfig, DiffusionBlocksController

config = DiffusionBlocksConfig.from_dict({"layers_per_block": 4, "overlap": 0.05})
controller = DiffusionBlocksController(pipe.transformer, config)
```

Mantenha `controller` durante a vida do pipeline e use a configuracao exata de `simpletuner_config.json`.

## Exemplo Anima

Veja `simpletuner/examples/anima.peft-lora+diffusion-blocks/config.json`. As 28 camadas do Anima v1.0 formam 7 blocos com `layers_per_block=4`.

```bash
simpletuner train env=examples/anima.peft-lora+diffusion-blocks max_train_steps=10 validation_steps=10
```

Ao retomar, nao altere paths, numero de camadas, limites, `blocks_to_train`, topologia, world size, batch sampling ou timesteps. Executar todas as camadas na inferencia invalida o objetivo treinado.
