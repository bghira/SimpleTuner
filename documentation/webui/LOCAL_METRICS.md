# Local training metrics

SimpleTuner can record training metrics without an external tracking service. Set:

```json
{
  "report_to": "simpletuner"
}
```

Use comma-separated values such as `report_to=simpletuner,wandb` to enable the local tracker alongside an external tracker. The tracker is model-independent and receives the same scalar calls used by the other Accelerate trackers.

**Training Runs** also shows the last 30 minutes of progress reported by active workers, including jobs using `report_to: none`. Select the worker job to chart the reported loss, learning rate, or throughput. This history lives in the running server and starts again after a restart. Offline reports, validation media, and timestep distributions remain available through the saved environment output.

## Output contract

The training output directory contains:

| File | Purpose |
| --- | --- |
| `training_metrics.jsonl` | Append-only scalar records with step and UTC timestamp |
| `training_metrics.json` | Atomic run manifest, metric names, status, and selected configuration values |
| `validation_media.jsonl` | Indexed validation image, video, and audio paths |
| `timestep_distribution.jsonl` | Timestep samples grouped by global step when `log_timestep_distribution` is enabled |
| `training_report.html` | Self-contained report that opens without a server |

The JSONL files are the raw interface for analysis tools. A resumed run appends records instead of replacing them. Under DDP, only the main process writes these files.

The HTML report embeds a bounded copy of the scalar history and uses relative paths for validation media. Archive it with the output directory.

When a tracker does not collect system telemetry natively, SimpleTuner records numeric CPU, memory, disk, network, and GPU telemetry for that tracker. WandB is skipped for manual system telemetry because its client already reports host metrics.

## WebUI

Open **Metrics**, then **Training Runs**. Runs are discovered from saved WebUI environments. The page provides:

- dynamic scalar selection, up to eight series
- latest values over step or elapsed minutes
- timestep distribution over global step
- validation galleries grouped by prompt for each training step
- a link to the offline report

In **System**, use **Machine** to select the orchestrator or a worker you can view. Updated workers report GPU readings, CPU usage, and RAM usage with their heartbeats. The GPU chart history is scoped to that machine. The page labels stale readings and shows when a worker has yet to send a sample.

For Prometheus, enable the **Workers** category in **Metrics → System**. `/api/metrics/prometheus` exports the latest visible worker hardware and active training samples. Worker metrics use `worker_id`, with `gpu_index` for devices and `job_id` for training. Sample timestamp gauges show when data last arrived. Stale hardware readings are omitted.

## Validation image storage

PNG remains the default. For smaller archives, set:

```json
{
  "validation_image_format": "webp",
  "validation_image_quality": 90
}
```

`validation_image_quality` applies to WebP, JPEG, and JPEG XL. Audio and video keep their existing formats.

Set `validation_image_format` to `jxl` to save JPEG XL. Install the `simpletuner[jxl]` extra and view the gallery or report in a browser that decodes `image/jxl`.

## API

All endpoints require normal WebUI authentication:

```text
GET /api/metrics/training/runs
GET /api/metrics/training/jobs/{job_id}
GET /api/metrics/training/runs/{environment}?max_points=2000&metric=train_loss
GET /api/metrics/training/runs/{environment}/media/{path}
GET /api/metrics/training/runs/{environment}/report
```

The local run endpoints resolve output directories from configured environments. They accept configured environment names. Scalar reads can be bounded by `start_step`, `end_step`, `max_points`, and repeated `metric` parameters.
