# Métricas locales de entrenamiento

SimpleTuner puede registrar métricas sin un servicio externo. Configura:

```json
{"report_to": "simpletuner"}
```

Usa valores separados por comas, como `report_to=simpletuner,wandb`, para activar el tracker local junto con uno externo. Funciona con todas las familias de modelos y, con DDP, solo escribe el proceso principal.

## Archivos de salida

El directorio de salida contiene:

- `training_metrics.jsonl`: escalares por paso, en formato append-only.
- `training_metrics.json`: manifiesto atómico, estado y nombres de métricas.
- `validation_media.jsonl`: índice de imágenes, vídeo y audio de validación.
- `timestep_distribution.jsonl`: muestras de timestep agrupadas por paso global.
- `training_report.html`: informe autónomo que abre sin servidor.

Una reanudación añade registros. Archiva el informe HTML junto con el directorio de salida porque usa rutas relativas para los medios.

Cuando un tracker no recopila telemetría del sistema de forma nativa, SimpleTuner registra métricas numéricas de CPU, memoria, disco, red y GPU para ese tracker. WandB se omite porque su cliente ya informa métricas del host.

## WebUI y API

Abre **Metrics** y **Training Runs** para elegir escalares por paso o minutos, ver la distribución de timesteps, revisar galerías de validación por paso y abrir el informe. **System** conserva salud de GPU y Prometheus.

```text
GET /api/metrics/training/runs
GET /api/metrics/training/runs/{environment}?max_points=2000&metric=train_loss
GET /api/metrics/training/runs/{environment}/media/{path}
GET /api/metrics/training/runs/{environment}/report
```

La API solo resuelve directorios de entornos configurados. Para ahorrar espacio usa `validation_image_format=webp` y `validation_image_quality=90`; PNG sigue siendo el valor predeterminado.
