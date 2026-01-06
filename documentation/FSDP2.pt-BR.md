# Treinamento sharded / multi-GPU com FSDP2

O SimpleTuner agora oferece suporte de primeira classe ao PyTorch Fully Sharded Data Parallel v2 (FSDP com DTensor). A WebUI usa a implementacao v2 como padrao para execucoes full-model e expoe as flags mais importantes do accelerate para voce escalar para hardware multi-GPU sem escrever scripts de lancamento customizados.

> AVISO: O FSDP2 mira as releases recentes do PyTorch 2.x com a pilha distribuida DTensor habilitada em builds CUDA. A WebUI so expoe controles de context-parallel em hosts CUDA; outros backends sao considerados experimentais.

## O que e FSDP2?

O FSDP2 e a proxima iteracao do mecanismo de data-parallel sharded do PyTorch. Em vez da logica legada de parametros achatados do FSDP v1, o plugin v2 fica em cima do DTensor. Ele shardia parametros do modelo, gradientes e otimizadores entre ranks enquanto mantem um pequeno conjunto de trabalho por rank. Em comparacao com abordagens classicas no estilo ZeRO, ele mantem o fluxo de lancamento do Hugging Face accelerate, entao checkpoints, otimizadores e caminhos de inferencia permanecem compativeis com o restante do SimpleTuner.

## Visao geral de recursos

- Toggle na WebUI (Hardware -> Accelerate) que gera um FullyShardedDataParallelPlugin com defaults sensatos
- Normalizacao automatica de CLI (`--fsdp_version`, `--fsdp_state_dict_type`, `--fsdp_auto_wrap_policy`) para tornar a grafia das flags tolerante
- Sharding opcional de context-parallel (`--context_parallel_size`, `--context_parallel_comm_strategy`) em cima do FSDP2 para modelos de sequencias longas
- Modal integrado de descoberta de blocos transformer que inspeciona o modelo base e sugere nomes de classes para auto-wrapping
- Metadados de deteccao em cache em `~/.simpletuner/fsdp_block_cache.json`, com acoes de manutencao em um clique nas configuracoes da WebUI
- Alternador de formato de checkpoint (sharded vs full) mais um modo de retomar eficiente em RAM de CPU para limites apertados de memoria do host

## Limitacoes conhecidas

- O FSDP2 so pode ser habilitado quando `model_type` e `full`. Execucoes PEFT/LoRA continuam usando caminhos padrao de um unico dispositivo.
- DeepSpeed e FSDP sao mutuamente exclusivos. Fornecer `--fsdp_enable` e um config DeepSpeed gera um erro explicito nos fluxos de CLI e WebUI.
- O paralelismo de contexto e limitado a sistemas CUDA e exige `--context_parallel_size > 1` com `--fsdp_version=2`.
- Passes de validacao agora funcionam com `--fsdp_reshard_after_forward=true` - modelos com FSDP sao passados diretamente para pipelines, que lidam com all-gather/reshard de forma transparente.
- A deteccao de blocos instancia o modelo base localmente. Espere uma breve pausa e aumento do uso de memoria do host ao escanear checkpoints grandes.
- O FSDP v1 permanece por compatibilidade, mas esta marcado como deprecated em toda a UI e logs de CLI.

## Habilitando FSDP2

### Metodo 1: WebUI (recomendado)

1. Abra a WebUI do SimpleTuner e carregue a configuracao de treinamento que voce pretende executar.
2. Va para **Hardware -> Accelerate**.
3. Ative **Enable FSDP v2**. O seletor de versao vai ficar em `2` como padrao; deixe assim a menos que voce precise intencionalmente do v1.
4. (Opcional) Ajuste:
   - **Reshard After Forward** para trocar VRAM por comunicacao
   - **Checkpoint Format** entre `Sharded` e `Full`
   - **CPU RAM Efficient Loading** se for retomar com limites apertados de memoria do host
   - **Auto Wrap Policy** e **Transformer Classes to Wrap** (veja o fluxo de deteccao abaixo)
   - **Context Parallel Size / Rotation** quando voce precisar de sharding de sequencia
5. Salve a configuracao. A superficie de lancamento do trainer agora vai passar o plugin do accelerate correto.

### Metodo 2: CLI

Use `simpletuner-train` com as mesmas flags expostas na WebUI. Exemplo para um treinamento SDXL full-model em duas GPUs:

```bash
simpletuner-train \
  --model_type=full \
  --model_family=sdxl \
  --output_dir=/data/experiments/sdxl-fsdp2 \
  --fsdp_enable \
  --fsdp_version=2 \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
  --num_processes=2
```

Se voce ja mantem um arquivo de configuracao do accelerate, pode continuar usando; o SimpleTuner mescla o plugin FSDP nos parametros de lancamento em vez de sobrescrever toda a configuracao.

## Paralelismo de contexto

O paralelismo de contexto esta disponivel como uma camada opcional sobre o FSDP2 para hosts CUDA. Defina `--context_parallel_size` (ou o campo correspondente na WebUI) para o numero de GPUs que devem dividir a dimensao da sequencia. A comunicacao acontece via:

- `allgather` (padrao) - prioriza overlap e e o melhor ponto de partida
- `alltoall` - workloads de nicho com janelas de atencao muito grandes podem se beneficiar, ao custo de orquestracao extra

O trainer impoe `fsdp_enable` e `fsdp_version=2` quando o paralelismo de contexto e solicitado. Ajustar o tamanho de volta para `1` desativa a funcionalidade de forma limpa e normaliza a string de rotacao para que configs salvas continuem consistentes.

## Fluxo de deteccao de blocos FSDP

O SimpleTuner inclui um detector que inspeciona o modelo base selecionado e exibe as classes de modulos mais adequadas para auto-wrapping do FSDP:

1. Selecione um **Model Family** (e opcionalmente um **Model Flavour**) no formulario do trainer.
2. Informe o caminho do checkpoint se voce estiver treinando a partir de um diretorio de pesos customizado.
3. Clique em **Detect Blocks** ao lado de **Transformer Classes to Wrap**. O SimpleTuner vai instanciar o modelo, percorrer seus modulos e registrar totais de parametros por classe.
4. Revise a analise no modal:
   - **Select** as classes que devem ser envolvidas (checkboxes na primeira coluna)
   - **Total Params** destaca quais modulos dominam seu orcamento de parametros
   - `_no_split_modules` (se presente) aparece como badges e deve ser adicionado as suas listas de exclusao
5. Pressione **Apply Selection** para preencher `--fsdp_transformer_layer_cls_to_wrap`.
6. Aberturas subsequentes reutilizam o resultado em cache, a menos que voce clique em **Refresh Detection**.

Os resultados da deteccao ficam em `~/.simpletuner/fsdp_block_cache.json`, chaveados por family de modelo, caminho do checkpoint e flavour. Use **Settings -> WebUI Preferences -> Cache Maintenance -> Clear FSDP Detection Cache** ao alternar entre checkpoints divergentes ou apos atualizar pesos do modelo.

## Manipulacao de checkpoints

- **Sharded state dict** (`SHARDED_STATE_DICT`) salva shards locais por rank e escala bem para modelos grandes.
- **Full state dict** (`FULL_STATE_DICT`) agrega parametros no rank 0 para compatibilidade com ferramentas externas; espere maior pressao de memoria.
- **CPU RAM Efficient Loading** atrasa a materializacao all-rank durante a retomada para reduzir picos de memoria do host.
- **Reshard After Forward** mantem os shards de parametros enxutos entre passes forward. A validacao agora funciona corretamente passando modelos com FSDP diretamente para pipelines do diffusers.

Escolha a combinacao que se alinha a sua cadencia de retomada e ferramentas downstream. Checkpoints sharded mais carregamento eficiente em RAM sao o pareamento mais seguro para modelos muito grandes.

## Ferramentas de manutencao

A WebUI expoe auxiliares de manutencao em **WebUI Preferences -> Cache Maintenance**:

- **Clear FSDP Detection Cache** remove todos os escaneamentos de blocos em cache (wrapper sobre `FSDP_SERVICE.clear_cache()`).
- **Clear DeepSpeed Offload Cache** permanece disponivel para usuarios ZeRO; opera de forma independente do FSDP.

Ambas as acoes mostram toasts e atualizam a area de status de manutencao para que voce confirme o resultado sem vasculhar arquivos de log.

## Solucao de problemas

| Sintoma | Causa provavel | Correcao |
|---------|----------------|----------|
| `"FSDP and DeepSpeed cannot be enabled simultaneously."` | Ambos os plugins especificados (ex.: JSON do DeepSpeed junto com `--fsdp_enable`). | Remova o config do DeepSpeed ou desative o FSDP. |
| `"Context parallelism requires FSDP2."` | `context_parallel_size > 1` enquanto o FSDP esta desligado ou ainda no v1. | Habilite FSDP, mantenha `--fsdp_version=2`, ou reduza o tamanho para `1`. |
| Falha na deteccao de blocos com `Unknown model_family` | O formulario nao tem uma family ou flavour suportada. | Escolha um modelo no dropdown; families customizadas devem ser registradas em `model_families`. |
| A deteccao mostra classes desatualizadas | Resultado em cache reutilizado. | Clique em **Refresh Detection** ou limpe o cache em WebUI Preferences. |
| Retomada esgota RAM do host | Agregacao de full state dict durante o carregamento. | Troque para `SHARDED_STATE_DICT` e/ou habilite CPU RAM efficient loading. |

## Referencia de flags da CLI

- `--fsdp_enable` - ativa FullyShardedDataParallelPlugin
- `--fsdp_version` - escolhe entre `1` e `2` (padrao `2`, v1 esta deprecated)
- `--fsdp_reshard_after_forward` - libera shards de parametros apos o forward (padrao `true`)
- `--fsdp_state_dict_type` - `SHARDED_STATE_DICT` (padrao) ou `FULL_STATE_DICT`
- `--fsdp_cpu_ram_efficient_loading` - reduz picos de memoria do host ao retomar
- `--fsdp_auto_wrap_policy` - `TRANSFORMER_BASED_WRAP`, `SIZE_BASED_WRAP`, `NO_WRAP`, ou um caminho callable pontuado
- `--fsdp_transformer_layer_cls_to_wrap` - lista de classes separada por virgulas preenchida pelo detector
- `--context_parallel_size` - shardia atencao entre este numero de ranks (somente CUDA + FSDP2)
- `--context_parallel_comm_strategy` - estrategia de rotacao `allgather` (padrao) ou `alltoall`
- `--num_processes` - total de ranks passado para accelerate quando nenhum arquivo de config e fornecido

Estas correspondem 1:1 aos controles da WebUI em Hardware -> Accelerate, entao uma configuracao exportada da interface pode ser reproduzida na CLI sem ajustes adicionais.
