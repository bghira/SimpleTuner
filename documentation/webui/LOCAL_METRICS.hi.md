# स्थानीय प्रशिक्षण मेट्रिक्स

SimpleTuner बाहरी सेवा के बिना प्रशिक्षण मेट्रिक्स लिख सकता है:

```json
{"report_to": "simpletuner"}
```

`report_to=simpletuner,wandb` जैसे comma-separated values से local tracker और external tracker साथ में चालू करें। यह सभी model families पर लागू है; DDP में केवल main process फाइलें लिखता है।

**Training Runs** सक्रिय workers से मिले पिछले 30 मिनट के प्रगति नमूने भी दिखाता है, जिनमें `report_to: none` वाले jobs शामिल हैं। loss, learning rate या throughput के लिए job चुनें। इतिहास चल रहे सर्वर में रहता है और पुनः आरंभ करने के बाद फिर से बनता है। रिपोर्ट, validation media और timestep distribution सहेजे गए environment आउटपुट से उपलब्ध रहते हैं।

## आउटपुट फाइलें

- `training_metrics.jsonl`: हर step के append-only scalar records।
- `training_metrics.json`: atomic run manifest, status और metric names।
- `validation_media.jsonl`: validation image, video और audio index।
- `timestep_distribution.jsonl`: `log_timestep_distribution` चालू होने पर ग्लोबल स्टेप के अनुसार दर्ज किए गए टाइमस्टेप सैंपल।
- `training_report.html`: बिना server के खुलने वाली self-contained report।

Resume पर records जोड़े जाते हैं। HTML relative media paths उपयोग करता है, इसलिए इसे output directory के साथ archive करें।

जब कोई tracker system telemetry native रूप से collect नहीं करता, SimpleTuner उस tracker के लिए CPU, memory, disk, network और GPU की numeric metrics लिखता है। WandB को छोड़ा जाता है क्योंकि उसका client host metrics पहले से report करता है।

## WebUI और API

**Metrics** में **Training Runs** खोलें। यहाँ step या minutes पर scalar, timestep distribution, हर step की validation gallery और offline report देख सकते हैं। **System** में GPU health और Prometheus configuration रहती है।

```text
GET /api/metrics/training/runs
GET /api/metrics/training/jobs/{job_id}
GET /api/metrics/training/runs/{environment}?max_points=2000&metric=train_loss
GET /api/metrics/training/runs/{environment}/media/{path}
GET /api/metrics/training/runs/{environment}/report
```

स्थानीय runs के API endpoints केवल configured environments की output directories पढ़ते हैं। छोटी archive के लिए `validation_image_format=webp` और `validation_image_quality=90` उपयोग करें; default PNG है।

JPEG XL में सेव करने के लिए `validation_image_format=jxl` सेट करें। `simpletuner[jxl]` एक्स्ट्रा इंस्टॉल करें और गैलरी या रिपोर्ट को ऐसे ब्राउज़र में खोलें जिसमें `image/jxl` सपोर्ट हो।
