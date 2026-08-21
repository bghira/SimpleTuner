# Métricas locais de treinamento

O SimpleTuner pode registrar métricas sem um serviço externo. Configure:

```json
{"report_to": "simpletuner"}
```

`report_to=all` também ativa o tracker local. Ele funciona com todas as famílias de modelos e, em DDP, somente o processo principal grava os arquivos.

## Arquivos de saída

- `training_metrics.jsonl`: escalares por etapa, somente anexados.
- `training_metrics.json`: manifesto atômico, status e nomes das métricas.
- `validation_media.jsonl`: índice de imagens, vídeos e áudios de validação.
- `training_report.html`: relatório autônomo que abre sem servidor.

Ao retomar, novos registros são anexados. Arquive o HTML com o diretório de saída, pois os caminhos de mídia são relativos.

## WebUI e API

Abra **Metrics** e **Training Runs** para selecionar escalares, comparar validações por prompt/etapa e abrir o relatório. **System** mantém a saúde das GPUs e a configuração do Prometheus.

```text
GET /api/metrics/training/runs
GET /api/metrics/training/runs/{environment}?max_points=2000&metric=train_loss
GET /api/metrics/training/runs/{environment}/media/{path}
GET /api/metrics/training/runs/{environment}/report
```

A API resolve apenas diretórios de ambientes configurados. Para reduzir o arquivo, use `validation_image_format=webp` e `validation_image_quality=90`; PNG continua sendo o padrão.
