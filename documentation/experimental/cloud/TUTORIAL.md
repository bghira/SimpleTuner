# Cloud Training Tutorial

This guide walks through running SimpleTuner training jobs on cloud GPU infrastructure. It covers both the Web UI and REST API workflows.

## Prerequisites

- SimpleTuner installed and server running (see the [local API tutorial](../../api/TUTORIAL.md#start-the-server))
- Datasets staged locally with captions (same [dataset requirements](../../api/TUTORIAL.md#optional-upload-datasets-over-the-api-local-backends) as local training)
- A cloud provider account (see [Supported Providers](#provider-setup))
- For API usage: a shell with `curl` and `jq`

## Provider Setup {#provider-setup}

Cloud training requires credentials for your chosen provider. Follow the setup guide for your provider:

| Provider | Setup Guide |
|----------|-------------|
| Replicate | [REPLICATE.md](REPLICATE.md#quick-start) |

After completing provider setup, return here to submit jobs.

## Quick Start

With your provider configured:

1. Open `http://localhost:8001` and go to the **Cloud** tab
2. Verify your credentials in **Settings** (gear icon) → **Validate**
3. Configure your training in the Model/Training/Dataloader tabs
4. Click **Train in Cloud**
5. Review the upload summary and click **Submit**

## Receiving Trained Models

After training completes, your model needs a destination. Configure one of these before your first job.

### Option 1: HuggingFace Hub (Recommended)

Push directly to your HuggingFace account:

1. Get a [HuggingFace token](https://huggingface.co/settings/tokens) with write access
2. Set the environment variable:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```
3. In the **Publishing** tab, enable "Push to Hub" and set your repo name

### Option 2: Local Download via Webhook

Have models upload back to your machine. Requires exposing your server to the internet.

1. Start a tunnel:
   ```bash
   ngrok http 8001   # or: cloudflared tunnel --url http://localhost:8001
   ```
2. Copy the public URL (e.g., `https://abc123.ngrok.io`)
3. In Cloud tab → Settings → Webhook URL, paste the URL
4. Models land in `~/.simpletuner/cloud_outputs/`

### Option 3: External S3

Upload to any S3-compatible endpoint (AWS S3, MinIO, Backblaze B2, Cloudflare R2):

1. In the **Publishing** tab, configure S3 settings
2. Provide endpoint, bucket, access key, secret key

## Web UI Workflow

### Submitting Jobs

1. **Configure your training** in the Model/Training/Dataloader tabs
2. **Navigate to Cloud tab** and select your provider
3. **Click Train in Cloud** to open the pre-submit dialog
4. **Review the upload summary**—local datasets will be packaged and uploaded
5. **Optionally set a run name** for tracking
6. **Click Submit**

### Monitoring Jobs

The job list shows all cloud and local jobs with:

- **Status indicator**: Queued → Running → Completed/Failed
- **Live progress**: Training step, loss values (when available)
- **Cost tracking**: Estimated cost based on GPU time

Click a job to see details:
- Job configuration snapshot
- Real-time logs (click **View Logs**)
- Actions: Cancel, Delete (after completion)

### Settings Panel

Click the gear icon to access:

- **API Key validation** and account status
- **Webhook URL** for local model delivery
- **Cost limits** to prevent runaway spending
- **Hardware info** (GPU type, cost per hour)

## API Workflow

### Submit a Job

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-training-config",
    "tracker_run_name": "api-test-run"
  }' | jq
```

Replace `PROVIDER` with your provider name (e.g., `replicate`).

Or submit with inline config:

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

### Monitor Job Status

```bash
# Get job details
curl -s http://localhost:8001/api/cloud/jobs/JOB_ID | jq

# List all jobs
curl -s 'http://localhost:8001/api/cloud/jobs?limit=10' | jq

# Sync status of active jobs from provider
curl -s 'http://localhost:8001/api/cloud/jobs?sync_active=true' | jq
```

### Fetch Job Logs

```bash
curl -s http://localhost:8001/api/cloud/jobs/JOB_ID/logs | jq '.logs'
```

### Cancel a Running Job

```bash
curl -s -X POST http://localhost:8001/api/cloud/jobs/JOB_ID/cancel | jq
```

### Delete a Completed Job

```bash
curl -s -X DELETE http://localhost:8001/api/cloud/jobs/JOB_ID | jq
```

## CI/CD Integration

### Idempotent Job Submission

Prevent duplicate jobs with idempotency keys:

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-config",
    "idempotency_key": "ci-build-12345"
  }' | jq
```

If the same key is submitted again within 24 hours, you get back the existing job instead of creating a duplicate.

### GitHub Actions Example

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

For automated pipelines, create API keys instead of session authentication.

**Via UI:** Cloud tab → Settings → API Keys → Create New Key

**Via API:**

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

The full key is only returned once. Store it securely.

**Using an API key:**

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer stk_abc123...' \
  -d '{...}'
```

**Scoped permissions:**

| Permission | Description |
|------------|-------------|
| `job.submit` | Submit new jobs |
| `job.view.own` | View own jobs |
| `job.cancel.own` | Cancel own jobs |
| `job.view.all` | View all jobs (admin) |

## Troubleshooting

For provider-specific issues (credentials, queuing, hardware), see your provider's documentation:

- [Replicate Troubleshooting](REPLICATE.md#troubleshooting)

### General Issues

**Data Upload Fails**
- Verify dataset paths exist and are readable
- Check available disk space for zip packaging
- Look for errors in the browser console or API response

**Webhook Not Receiving Events**
- Ensure your local instance is publicly accessible (tunnel running)
- Verify the webhook URL is correct (including https://)
- Check SimpleTuner's terminal output for webhook handling errors

## API Reference

### Provider-Agnostic Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cloud/jobs` | GET | List jobs with optional filters |
| `/api/cloud/jobs/submit` | POST | Submit a new training job |
| `/api/cloud/jobs/sync` | POST | Sync jobs from provider |
| `/api/cloud/jobs/{id}` | GET | Get job details |
| `/api/cloud/jobs/{id}/logs` | GET | Fetch job logs |
| `/api/cloud/jobs/{id}/cancel` | POST | Cancel a running job |
| `/api/cloud/jobs/{id}` | DELETE | Delete a completed job |
| `/api/metrics` | GET | Get job and cost metrics |
| `/api/cloud/metrics/cost-limit` | GET | Get current cost limit status |
| `/api/cloud/providers/{provider}` | PUT | Update provider settings |
| `/api/cloud/storage/{bucket}/{key}` | PUT | S3-compatible upload endpoint |

For provider-specific endpoints, see:
- [Replicate API Reference](REPLICATE.md#api-reference)

For full schema details, see the OpenAPI docs at `http://localhost:8001/docs`.

## See Also

- [README.md](README.md) – Architecture overview and provider status
- [REPLICATE.md](REPLICATE.md) – Replicate provider setup and details
- [ENTERPRISE.md](../server/ENTERPRISE.md) – SSO, approvals, and governance
- [End-to-end cloud operations tutorial](OPERATIONS_TUTORIAL.md) – Production deployment and monitoring
- [End-to-end local API Tutorial](../../api/TUTORIAL.md) – Complete local training via API
- [Dataloader Configuration](../../DATALOADER.md) – Dataset setup reference
