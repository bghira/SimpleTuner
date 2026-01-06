# Cloud Training ट्यूटोरियल

यह गाइड SimpleTuner training jobs को cloud GPU infrastructure पर चलाने की प्रक्रिया बताती है। इसमें Web UI और REST API दोनों workflows शामिल हैं।

## प्रीरिक्विज़िट्स

- SimpleTuner इंस्टॉल और server चल रहा हो (देखें [local API tutorial](../../api/TUTORIAL.md#start-the-server))
- Local datasets captions के साथ तैयार हों (local training जैसे ही [dataset requirements](../../api/TUTORIAL.md#optional-upload-datasets-over-the-api-local-backends))
- कोई cloud provider account (देखें [Supported Providers](#provider-setup))
- API उपयोग के लिए: `curl` और `jq` वाला shell

## Provider Setup {#provider-setup}

Cloud training के लिए आपके चुने हुए provider की credentials चाहिए। अपने provider के setup guide को फॉलो करें:

| Provider | Setup Guide |
|----------|-------------|
| Replicate | [REPLICATE.md](REPLICATE.md#quick-start) |

Provider setup पूरा करने के बाद यहाँ लौटकर jobs submit करें।

## Quick Start

Provider कॉन्फ़िगर होने के बाद:

1. `http://localhost:8001` खोलें और **Cloud** tab पर जाएँ
2. **Settings** (gear icon) → **Validate** में credentials verify करें
3. Model/Training/Dataloader tabs में training कॉन्फ़िगर करें
4. **Train in Cloud** क्लिक करें
5. Upload summary समीक्षा करें और **Submit** क्लिक करें

## प्रशिक्षित मॉडल प्राप्त करना

Training पूरी होने के बाद, मॉडल को किसी destination की जरूरत होती है। अपनी पहली job से पहले इनमें से एक कॉन्फ़िगर करें।

### Option 1: HuggingFace Hub (Recommended)

सीधे अपने HuggingFace अकाउंट पर push करें:

1. write access वाला [HuggingFace token](https://huggingface.co/settings/tokens) लें
2. environment variable सेट करें:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```
3. **Publishing** tab में "Push to Hub" सक्षम करें और अपना repo name सेट करें

### Option 2: Webhook के जरिए Local Download

मॉडल्स को वापस अपनी मशीन पर upload करवाएँ। इसके लिए server को internet पर expose करना होगा।

1. Tunnel शुरू करें:
   ```bash
   ngrok http 8001   # या: cloudflared tunnel --url http://localhost:8001
   ```
2. Public URL कॉपी करें (जैसे `https://abc123.ngrok.io`)
3. Cloud tab → Settings → Webhook URL में URL पेस्ट करें
4. Models `~/.simpletuner/cloud_outputs/` में आएँगे

### Option 3: External S3

किसी भी S3-compatible endpoint (AWS S3, MinIO, Backblaze B2, Cloudflare R2) पर upload करें:

1. **Publishing** tab में S3 settings कॉन्फ़िगर करें
2. endpoint, bucket, access key, secret key दें

## Web UI Workflow

### Jobs सबमिट करना

1. Model/Training/Dataloader tabs में **अपनी training कॉन्फ़िगर करें**
2. **Cloud tab** पर जाएँ और provider चुनें
3. **Train in Cloud** क्लिक करें ताकि pre-submit dialog खुले
4. **Upload summary** समीक्षा करें—local datasets पैक होकर अपलोड होंगे
5. **Optional run name** सेट करें (tracking के लिए)
6. **Submit** क्लिक करें

### Jobs मॉनिटर करना

Job list में सभी cloud और local jobs दिखते हैं:

- **Status indicator**: Queued → Running → Completed/Failed
- **Live progress**: Training step, loss values (यदि उपलब्ध)
- **Cost tracking**: GPU time के आधार पर अनुमानित लागत

किसी job पर क्लिक करके विवरण देखें:
- Job configuration snapshot
- Real-time logs ( **View Logs** क्लिक करें)
- Actions: Cancel, Delete (completion के बाद)

### Settings Panel

Gear icon पर क्लिक करके देखें:

- **API Key validation** और account status
- Local model delivery के लिए **Webhook URL**
- runaway खर्च से बचने के लिए **Cost limits**
- **Hardware info** (GPU type, cost per hour)

## API Workflow

### Job सबमिट करें

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-training-config",
    "tracker_run_name": "api-test-run"
  }' | jq
```

`PROVIDER` को अपने provider नाम (जैसे `replicate`) से बदलें।

या inline config के साथ submit करें:

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config": {
      "--model_family": "flux",
      "--model_type": "lora",
      "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
      "--output_dir": "/outputs/flux-lora",
      "--max_train_steps": 1000,
      "--lora_rank": 16
    },
    "dataloader_config": [
      {
        "id": "training-images",
        "type": "local",
        "dataset_type": "image",
        "instance_data_dir": "/data/datasets/my-dataset",
        "caption_strategy": "textfile",
        "resolution": 1024
      }
    ]
  }' | jq
```

### Job Status मॉनिटर करें

```bash
# Get job details
curl -s http://localhost:8001/api/cloud/jobs/JOB_ID | jq

# List all jobs
curl -s 'http://localhost:8001/api/cloud/jobs?limit=10' | jq

# Sync status of active jobs from provider
curl -s 'http://localhost:8001/api/cloud/jobs?sync_active=true' | jq
```

### Job Logs प्राप्त करें

```bash
curl -s http://localhost:8001/api/cloud/jobs/JOB_ID/logs | jq '.logs'
```

### Running Job Cancel करें

```bash
curl -s -X POST http://localhost:8001/api/cloud/jobs/JOB_ID/cancel | jq
```

### Completed Job Delete करें

```bash
curl -s -X DELETE http://localhost:8001/api/cloud/jobs/JOB_ID | jq
```

## CI/CD Integration

### Idempotent Job Submission

Idempotency keys के साथ duplicate jobs रोकें:

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-config",
    "idempotency_key": "ci-build-12345"
  }' | jq
```

यदि वही key 24 घंटे के भीतर दोबारा submit की जाती है, तो duplicate बनाने के बजाय मौजूदा job वापस मिलता है।

### GitHub Actions उदाहरण

```yaml
name: Cloud Training

on:
  push:
    branches: [main]
    paths:
      - 'training-configs/**'

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Submit Training Job
        env:
          SIMPLETUNER_URL: ${{ secrets.SIMPLETUNER_URL }}
        run: |
          RESPONSE=$(curl -s -X POST "$SIMPLETUNER_URL/api/cloud/jobs/submit?provider=replicate" \
            -H 'Content-Type: application/json' \
            -d '{
              "config_name_to_load": "production-lora",
              "idempotency_key": "gh-${{ github.sha }}",
              "tracker_run_name": "gh-run-${{ github.run_number }}"
            }')

          JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
          echo "Submitted job: $JOB_ID"
          echo "JOB_ID=$JOB_ID" >> $GITHUB_ENV

      - name: Wait for Completion
        run: |
          while true; do
            STATUS=$(curl -s "$SIMPLETUNER_URL/api/cloud/jobs/$JOB_ID" | jq -r '.job.status')
            echo "Job status: $STATUS"

            case $STATUS in
              completed) exit 0 ;;
              failed|cancelled) exit 1 ;;
              *) sleep 60 ;;
            esac
          done
```

### API Key Authentication

Automated pipelines के लिए session auth की बजाय API keys बनाएं।

**UI के जरिए:** Cloud tab → Settings → API Keys → Create New Key

**API के जरिए:**

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/auth/api-keys' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_SESSION_TOKEN' \
  -d '{
    "name": "ci-pipeline",
    "expires_days": 90,
    "scoped_permissions": ["job.submit", "job.view.own"]
  }'
```

पूर्ण key केवल एक बार लौटता है। इसे सुरक्षित रखें।

**API key का उपयोग:**

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer stk_abc123...' \
  -d '{...}'
```

**Scoped permissions:**

| Permission | Description |
|------------|-------------|
| `job.submit` | नए jobs submit करें |
| `job.view.own` | अपने jobs देखें |
| `job.cancel.own` | अपने jobs cancel करें |
| `job.view.all` | सभी jobs देखें (admin) |

## Troubleshooting

Provider-specific समस्याओं (credentials, queuing, hardware) के लिए provider डॉक्यूमेंटेशन देखें:

- [Replicate Troubleshooting](REPLICATE.md#troubleshooting)

### General Issues

**Data Upload Fails**
- dataset paths मौजूद हैं और readable हैं या नहीं जांचें
- zip packaging के लिए उपलब्ध disk space जांचें
- browser console या API response में errors देखें

**Webhook Not Receiving Events**
- आपका local instance publicly accessible है (tunnel चल रहा है) या नहीं देखें
- webhook URL सही है या नहीं जांचें (https:// सहित)
- SimpleTuner terminal output में webhook handling errors देखें

## API Reference

### Provider-Agnostic Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cloud/jobs` | GET | Optional filters के साथ jobs सूचीबद्ध करें |
| `/api/cloud/jobs/submit` | POST | नया training job submit करें |
| `/api/cloud/jobs/sync` | POST | Provider से jobs sync करें |
| `/api/cloud/jobs/{id}` | GET | Job details प्राप्त करें |
| `/api/cloud/jobs/{id}/logs` | GET | Job logs लाएँ |
| `/api/cloud/jobs/{id}/cancel` | POST | Running job cancel करें |
| `/api/cloud/jobs/{id}` | DELETE | Completed job delete करें |
| `/api/metrics` | GET | Job और cost metrics प्राप्त करें |
| `/api/cloud/metrics/cost-limit` | GET | वर्तमान cost limit status प्राप्त करें |
| `/api/cloud/providers/{provider}` | PUT | Provider settings अपडेट करें |
| `/api/cloud/storage/{bucket}/{key}` | PUT | S3-compatible upload endpoint |

Provider-specific endpoints के लिए देखें:
- [Replicate API Reference](REPLICATE.md#api-reference)

Full schema विवरण के लिए OpenAPI docs `http://localhost:8001/docs` देखें।

## See Also

- [README.md](README.md) – आर्किटेक्चर ओवरव्यू और provider status
- [REPLICATE.md](REPLICATE.md) – Replicate provider setup और विवरण
- [ENTERPRISE.md](../server/ENTERPRISE.md) – SSO, approvals, और governance
- [End-to-end cloud operations tutorial](OPERATIONS_TUTORIAL.md) – Production deployment और monitoring
- [End-to-end local API Tutorial](../../api/TUTORIAL.md) – API के जरिए complete local training
- [Dataloader Configuration](../../DATALOADER.md) – डेटासेट सेटअप संदर्भ
