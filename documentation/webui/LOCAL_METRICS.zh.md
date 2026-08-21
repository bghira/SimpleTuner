# 本地训练指标

SimpleTuner 可以在不使用外部服务的情况下记录训练指标：

```json
{"report_to": "simpletuner"}
```

`report_to=all` 也会启用本地 tracker。它适用于所有模型系列；在 DDP 下仅主进程写入文件。

## 输出文件

- `training_metrics.jsonl`：按 step 追加的标量记录。
- `training_metrics.json`：原子写入的运行清单、状态和指标名称。
- `validation_media.jsonl`：验证图像、视频和音频索引。
- `training_report.html`：无需服务器即可打开的独立报告。

恢复训练时会追加记录。HTML 使用相对媒体路径，因此应与输出目录一起归档。

## WebUI 与 API

打开 **Metrics** 和 **Training Runs**，可选择标量、按 prompt/step 比较验证结果并打开离线报告。**System** 保留 GPU 健康和 Prometheus 设置。

```text
GET /api/metrics/training/runs
GET /api/metrics/training/runs/{environment}?max_points=2000&metric=train_loss
GET /api/metrics/training/runs/{environment}/media/{path}
GET /api/metrics/training/runs/{environment}/report
```

API 只解析已配置 environment 的输出目录。若要减小归档大小，可设置 `validation_image_format=webp` 和 `validation_image_quality=90`；默认仍为 PNG。
