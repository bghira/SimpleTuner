# ローカル学習メトリクス

外部サービスなしでメトリクスを保存できます。

```json
{"report_to": "simpletuner"}
```

`report_to=simpletuner,wandb` のようにカンマ区切りで指定すると、ローカルトラッカーと外部 tracker を同時に有効にできます。全モデルファミリーで共通に動作し、DDP ではメインプロセスだけが書き込みます。

## 出力ファイル

- `training_metrics.jsonl`: step ごとの追記専用スカラー記録
- `training_metrics.json`: 状態、メトリクス名、設定を含むアトミックなマニフェスト
- `validation_media.jsonl`: 検証画像・動画・音声のインデックス
- `training_report.html`: サーバーなしで開ける自己完結レポート

再開時は既存履歴へ追記します。HTML は相対メディアパスを使うため、出力ディレクトリと一緒に保存してください。

tracker が system telemetry を標準で収集しない場合、SimpleTuner は CPU、メモリ、ディスク、ネットワーク、GPU の数値メトリクスをその tracker に記録します。WandB はクライアント側で host metrics を収集するため、手動記録の対象外です。

## WebUI と API

**Metrics** の **Training Runs** でスカラーを選択し、prompt と step ごとに検証結果を比較できます。**System** には GPU health と Prometheus 設定があります。

```text
GET /api/metrics/training/runs
GET /api/metrics/training/runs/{environment}?max_points=2000&metric=train_loss
GET /api/metrics/training/runs/{environment}/media/{path}
GET /api/metrics/training/runs/{environment}/report
```

API は設定済み environment の出力だけを参照します。容量を減らすには `validation_image_format=webp` と `validation_image_quality=90` を使います。既定値は PNG です。
