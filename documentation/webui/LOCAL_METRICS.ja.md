# ローカル学習メトリクス

外部サービスなしでメトリクスを保存できます。

```json
{"report_to": "simpletuner"}
```

`report_to=simpletuner,wandb` のようにカンマ区切りで指定すると、ローカルトラッカーと外部 tracker を同時に有効にできます。全モデルファミリーで共通に動作し、DDP ではメインプロセスだけが書き込みます。

**Training Runs** では、`report_to: none` のジョブも含め、稼働中のワーカーが報告した直近 30 分の進捗を確認できます。ジョブを選ぶと、損失、学習率、スループットをグラフに表示できます。履歴はサーバーの実行中に保持され、再起動すると記録を再開します。レポート、検証メディア、timestep 分布は保存済みの環境出力から参照できます。

## 出力ファイル

- `training_metrics.jsonl`: step ごとの追記専用スカラー記録
- `training_metrics.json`: 状態、メトリクス名、設定を含むアトミックなマニフェスト
- `validation_media.jsonl`: 検証画像・動画・音声のインデックス
- `timestep_distribution.jsonl`: `log_timestep_distribution` が有効な場合に、グローバルステップごとに記録されるタイムステップのサンプル
- `training_report.html`: サーバーなしで開ける自己完結レポート

再開時は既存履歴へ追記します。HTML は相対メディアパスを使うため、出力ディレクトリと一緒に保存してください。

tracker が system telemetry を標準で収集しない場合、SimpleTuner は CPU、メモリ、ディスク、ネットワーク、GPU の数値メトリクスをその tracker に記録します。WandB はクライアント側で host metrics を収集するため、手動記録の対象外です。

## WebUI と API

**Metrics** の **Training Runs** で step または分単位のスカラー、timestep 分布、step ごとの検証ギャラリー、オフラインレポートを確認できます。**System** の **Machine** で、オーケストレーターまたは閲覧権限のあるワーカーを選択できます。更新済みのワーカーはハートビートで GPU の測定値、CPU 使用率、RAM 使用率を送信します。GPU 履歴はマシンごとに表示されます。古い測定値と未受信の状態は区別して表示されます。

Prometheus では **Metrics → System** の **Workers** カテゴリを有効にします。`/api/metrics/prometheus` は閲覧できるワーカーの最新のハードウェア測定値と実行中の学習サンプルをエクスポートします。ラベルは `worker_id` と、デバイス用の `gpu_index`、学習用の `job_id` です。時刻のゲージは最後に受信した時点を示します。古いハードウェア測定値は除外されます。

```text
GET /api/metrics/training/runs
GET /api/metrics/training/jobs/{job_id}
GET /api/metrics/training/runs/{environment}?max_points=2000&metric=train_loss
GET /api/metrics/training/runs/{environment}/media/{path}
GET /api/metrics/training/runs/{environment}/report
```

API は設定済み environment の出力だけを参照します。容量を減らすには `validation_image_format=webp` と `validation_image_quality=90` を使います。既定値は PNG です。

JPEG XL で保存するには `validation_image_format=jxl` を設定します。`simpletuner[jxl]` を追加インストールし、`image/jxl` に対応するブラウザーでギャラリーやレポートを表示します。
