# Métricas locais de treinamento

O SimpleTuner pode registrar métricas sem um serviço externo. Configure:

```json
{"report_to": "simpletuner"}
```

Use valores separados por virgula, como `report_to=simpletuner,wandb`, para ativar o tracker local junto com um externo. Ele funciona com todas as famílias de modelos e, em DDP, somente o processo principal grava os arquivos.

**Training Runs** também mostra os últimos 30 minutos de progresso dos workers ativos, inclusive quando `report_to: none`. Selecione o job para ver gráficos de perda, taxa de aprendizado e rendimento. O histórico fica no servidor em execução e recomeça após uma reinicialização. Relatórios, mídias de validação e distribuições de timesteps continuam disponíveis na saída salva do ambiente.

## Arquivos de saída

- `training_metrics.jsonl`: escalares por etapa, somente anexados.
- `training_metrics.json`: manifesto atômico, status e nomes das métricas.
- `validation_media.jsonl`: índice de imagens, vídeos e áudios de validação.
- `timestep_distribution.jsonl`: amostras de passos de difusão agrupadas por etapa global quando `log_timestep_distribution` está ativo.
- `training_report.html`: relatório autônomo que abre sem servidor.

Ao retomar, novos registros são anexados. Arquive o HTML com o diretório de saída, pois os caminhos de mídia são relativos.

Quando um tracker não coleta telemetria do sistema nativamente, o SimpleTuner registra métricas numéricas de CPU, memória, disco, rede e GPU para esse tracker. O WandB é ignorado porque seu cliente já relata métricas do host.

## WebUI e API

Abra **Metrics** e **Training Runs** para selecionar escalares por etapa ou minutos, ver a distribuição de timesteps, revisar galerias de validação por etapa e abrir o relatório. Em **System**, use **Machine** para selecionar o orquestrador ou um worker que você possa visualizar. Workers atualizados enviam leituras da GPU e uso de CPU e RAM com seus heartbeats. O histórico das GPUs é separado por máquina. A página sinaliza leituras antigas e informa quando uma amostra ainda está pendente.

```text
GET /api/metrics/training/runs
GET /api/metrics/training/jobs/{job_id}
GET /api/metrics/training/runs/{environment}?max_points=2000&metric=train_loss
GET /api/metrics/training/runs/{environment}/media/{path}
GET /api/metrics/training/runs/{environment}/report
```

Os endpoints de execuções locais resolvem apenas diretórios de ambientes configurados. Para reduzir o arquivo, use `validation_image_format=webp` e `validation_image_quality=90`; PNG continua sendo o padrão.

Para salvar em JPEG XL, defina `validation_image_format=jxl`. Instale o extra `simpletuner[jxl]` e abra a galeria ou o relatório em um navegador que decodifique `image/jxl`.
