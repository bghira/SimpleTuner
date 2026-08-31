# Local training metrics

SimpleTuner can record training metrics without an external tracking service. Set:

```json
{
  "report_to": "simpletuner"
}
```

Use comma-separated values such as `report_to=simpletuner,wandb` to enable the local tracker alongside an external tracker. The tracker is model-independent and receives the same scalar calls used by the other Accelerate trackers.

## Output contract

The training output directory contains:

| File | Purpose |
| --- | --- |
| `training_metrics.jsonl` | Append-only scalar records with step and UTC timestamp |
| `training_metrics.json` | Atomic run manifest, metric names, status, and selected configuration values |
| `validation_media.jsonl` | Indexed validation image, video, and audio paths |
| `training_report.html` | Self-contained report that opens without a server |

The JSONL files are the raw interface for analysis tools. A resumed run appends records instead of replacing them. Under DDP, only the main process writes these files.

The HTML report embeds a bounded copy of the scalar history and uses relative paths for validation media. Archive it with the output directory.

When a tracker does not collect system telemetry natively, SimpleTuner records numeric CPU, memory, disk, network, and GPU telemetry for that tracker. WandB is skipped for manual system telemetry because its client already reports host metrics.

## WebUI

Open **Metrics**, then **Training Runs**. Runs are discovered from saved WebUI environments. The page provides:

- dynamic scalar selection, up to eight series
- latest values and step history
- validation comparison by prompt and training step
- a link to the offline report

The **System** section retains GPU health and Prometheus configuration.

## Validation image storage

PNG remains the default. For smaller archives, set:

```json
{
  "validation_image_format": "webp",
  "validation_image_quality": 90
}
```

`validation_image_quality` applies to WebP and JPEG. Audio and video keep their existing formats.

## API

All endpoints require normal WebUI authentication:

```text
GET /api/metrics/training/runs
GET /api/metrics/training/runs/{environment}?max_points=2000&metric=train_loss
GET /api/metrics/training/runs/{environment}/media/{path}
GET /api/metrics/training/runs/{environment}/report
```

The API resolves output directories from configured environments. It does not accept arbitrary filesystem paths. Scalar reads can be bounded by `start_step`, `end_step`, `max_points`, and repeated `metric` parameters.
