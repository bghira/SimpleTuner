# Métricas locales de entrenamiento

SimpleTuner puede registrar métricas sin un servicio externo. Configura:

```json
{"report_to": "simpletuner"}
```

Usa valores separados por comas, como `report_to=simpletuner,wandb`, para activar el tracker local junto con uno externo. Funciona con todas las familias de modelos y, con DDP, solo escribe el proceso principal.

**Training Runs** también muestra los últimos 30 minutos de progreso de los workers activos, incluso con `report_to: none`. Selecciona el trabajo para ver la pérdida, la tasa de aprendizaje o el rendimiento. El historial reside en el servidor en ejecución y se reinicia al reiniciar el servidor. Los informes, los medios de validación y las distribuciones de timesteps se consultan desde la salida guardada del entorno.

## Archivos de salida

El directorio de salida contiene:

- `training_metrics.jsonl`: escalares por paso, en formato append-only.
- `training_metrics.json`: manifiesto atómico, estado y nombres de métricas.
- `validation_media.jsonl`: índice de imágenes, vídeo y audio de validación.
- `timestep_distribution.jsonl`: muestras de pasos de difusión agrupadas por paso global cuando se activa `log_timestep_distribution`.
- `training_report.html`: informe autónomo que abre sin servidor.

Una reanudación añade registros. Archiva el informe HTML junto con el directorio de salida porque usa rutas relativas para los medios.

Cuando un tracker no recopila telemetría del sistema de forma nativa, SimpleTuner registra métricas numéricas de CPU, memoria, disco, red y GPU para ese tracker. WandB se omite porque su cliente ya informa métricas del host.

## WebUI y API

Abre **Metrics** y **Training Runs** para elegir escalares por paso o minutos, ver la distribución de timesteps, revisar galerías de validación por paso y abrir el informe. **System** conserva salud de GPU y Prometheus.

```text
GET /api/metrics/training/runs
GET /api/metrics/training/jobs/{job_id}
GET /api/metrics/training/runs/{environment}?max_points=2000&metric=train_loss
GET /api/metrics/training/runs/{environment}/media/{path}
GET /api/metrics/training/runs/{environment}/report
```

Los endpoints de ejecuciones locales solo resuelven directorios de entornos configurados. Para ahorrar espacio usa `validation_image_format=webp` y `validation_image_quality=90`; PNG sigue siendo el valor predeterminado.

Configura `validation_image_format=jxl` para guardar en JPEG XL. Instala el extra `simpletuner[jxl]` y abre la galería o el informe en un navegador compatible con `image/jxl`.
