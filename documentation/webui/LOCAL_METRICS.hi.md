# स्थानीय प्रशिक्षण मेट्रिक्स

SimpleTuner बाहरी सेवा के बिना प्रशिक्षण मेट्रिक्स लिख सकता है:

```json
{"report_to": "simpletuner"}
```

`report_to=simpletuner,wandb` जैसे comma-separated values से local tracker और external tracker साथ में चालू करें। यह सभी model families पर लागू है; DDP में केवल main process फाइलें लिखता है।

## आउटपुट फाइलें

- `training_metrics.jsonl`: हर step के append-only scalar records।
- `training_metrics.json`: atomic run manifest, status और metric names।
- `validation_media.jsonl`: validation image, video और audio index।
- `training_report.html`: बिना server के खुलने वाली self-contained report।

Resume पर records जोड़े जाते हैं। HTML relative media paths उपयोग करता है, इसलिए इसे output directory के साथ archive करें।

जब कोई tracker system telemetry native रूप से collect नहीं करता, SimpleTuner उस tracker के लिए CPU, memory, disk, network और GPU की numeric metrics लिखता है। WandB को छोड़ा जाता है क्योंकि उसका client host metrics पहले से report करता है।

## WebUI और API

**Metrics** में **Training Runs** खोलें। यहाँ scalar चुन सकते हैं, prompt/step के अनुसार validation तुलना कर सकते हैं और offline report खोल सकते हैं। **System** में GPU health और Prometheus configuration रहती है।

```text
GET /api/metrics/training/runs
GET /api/metrics/training/runs/{environment}?max_points=2000&metric=train_loss
GET /api/metrics/training/runs/{environment}/media/{path}
GET /api/metrics/training/runs/{environment}/report
```

API केवल configured environments के output directories पढ़ती है। छोटी archive के लिए `validation_image_format=webp` और `validation_image_quality=90` उपयोग करें; default PNG है।
